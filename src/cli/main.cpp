// xpu-serve command line: serve | run | bench | convert | info.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "chat/chat_template.h"
#include "chat/output_parser.h"
#include "cli/args.h"
#include "convert/convert.h"
#include "models/config_util.h"
#include "models/model.h"
#include "models/policy.h"
#include "policy/runner.h"
#include "server/server.h"
#include "service/engine_service.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"
#include "util/half.h"
#include "util/log.h"

#ifndef XS_VERSION
#define XS_VERSION "dev"
#endif

namespace xs::cli {
namespace {

const char* kUsage =
    "xpu-serve " XS_VERSION " - LLM serving on Intel GPUs\n"
    "\n"
    "usage:\n"
    "  xpu-serve serve <model-dir> [options]   OpenAI-compatible server\n"
    "  xpu-serve run   <model-dir> [options]   chat in the terminal (--prompt for one shot)\n"
    "  xpu-serve bench <model-dir> [options]   prefill / decode speed\n"
    "  xpu-serve convert <model-dir> --out <dir> [--quant q4|q8] [--method rtn|awq]   quantized pack\n"
    "  xpu-serve info                          backends and devices\n"
    "\n"
    "<model-dir> is a Hugging Face model directory (config.json, *.safetensors, tokenizer.json) or a pack that\n"
    "'xpu-serve convert' wrote from one.\n"
    "Run 'xpu-serve <command> --help' for the options of a command.\n";

std::string default_backend() {
    const auto all = available_backends();
    return all.empty() ? "cpu-ref" : all.front();
}

void add_engine_options(Args& a) {
    a.value("backend", "backend: " + [] {
        std::string s;
        for (const auto& b : available_backends()) s += (s.empty() ? "" : " | ") + b;
        return s;
    }(), default_backend())
        .value("device", "device, e.g. GPU, GPU.1, CPU (default: the backend's choice)")
        .value("precision", "compute precision: f16 | f32", "f16")
        .value("kv-dtype", "KV cache: auto (the precision's float) | i8 (int8 with a scale per token and head)",
               "auto")
        .value("ctx", "context length in tokens (prompt + output)", "8192")
        .value("prefill-chunk", "tokens per prefill step", "1024")
        .value("backend-opt", "backend options as key=value[,key=value]")
        .value("log-level", "debug | info | warn | error", "info")
        .flag("no-vision", "do not load the model's vision encoder (text only)")
        .flag("help", "show this help");
}

ServiceOptions service_options(const Args& a) {
    ServiceOptions o;
    o.backend = a.get("backend");
    o.session.device = a.get("device");
    const std::string& p = a.get("precision");
    if (p == "f16") {
        o.session.precision = DType::F16;
    } else if (p == "f32") {
        o.session.precision = DType::F32;
    } else {
        fail("--precision must be f16 or f32, got '", p, "'");
    }
    const std::string& kv_dtype = a.get("kv-dtype");
    if (kv_dtype == "i8") {
        o.session.kv_dtype = DType::I8;
    } else if (kv_dtype != "auto") {
        fail("--kv-dtype must be auto or i8, got '", kv_dtype, "'");
    }
    o.vision = !a.has("no-vision");
    o.engine.max_context = a.get_int("ctx");
    o.engine.prefill_chunk = a.get_int("prefill-chunk");
    const std::string& opts = a.get("backend-opt");
    for (size_t pos = 0; pos < opts.size();) {
        const size_t comma = std::min(opts.find(',', pos), opts.size());
        const std::string kv = opts.substr(pos, comma - pos);
        const size_t eq = kv.find('=');
        if (eq == std::string::npos || eq == 0) fail("--backend-opt expects key=value, got '", kv, "'");
        o.session.extra[kv.substr(0, eq)] = kv.substr(eq + 1);
        pos = comma + 1;
    }
    return o;
}

std::string model_dir_arg(const Args& a, const char* command) {
    if (a.positionals().size() != 1) fail("usage: xpu-serve ", command, " <model-dir> [options]");
    return a.positionals()[0];
}

bool wants_help(const Args& a, const char* command, const char* summary) {
    if (!a.has("help")) return false;
    std::printf("usage: xpu-serve %s %s\n\noptions:\n%s", command, summary, a.usage().c_str());
    return true;
}

int cmd_convert(const std::vector<std::string>& argv) {
    Args a;
    a.value("out", "output directory for the pack (must not exist or be empty)")
        .value("quant", "format of the matmul weights: q4 | q8 (embedding tables are always q8)", "q4")
        .value("group", "columns per scale: 32 | 64 | 128 (default: 32 for q4, 64 for q8)")
        .value("q8", "comma-separated parts of weight names to keep at q8 (e.g. linear_attn,down_proj)")
        .value("scope", "which matmul weights: bandwidth (those it makes faster: decode, a policy's action expert) | all",
               "bandwidth")
        .value("method", "q4 quantization: rtn | awq (activation-aware, calibrates on a backend)", "rtn")
        .value("calib", "AWQ: a text file to calibrate with (default: a built-in sample)")
        .value("calib-tokens", "AWQ: calibration tokens, at most", "8192")
        .value("backend", "AWQ: calibration backend (default: the first of this build)")
        .value("device", "AWQ: calibration device")
        .value("log-level", "debug | info | warn | error", "info")
        .flag("help", "show this help");
    a.parse(argv);
    if (wants_help(a, "convert", "<model-dir> --out <dir> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    ConvertOptions o;
    o.source = model_dir_arg(a, "convert");
    o.output = a.get("out");
    if (o.output.empty()) fail("usage: xpu-serve convert <model-dir> --out <dir> [options]");
    const std::string& quant = a.get("quant");
    if (quant != "q4" && quant != "q8") fail("--quant must be q4 or q8, got '", quant, "'");
    o.format = quant == "q4" ? DType::Q4 : DType::Q8;
    o.group = a.get("group").empty() ? (o.format == DType::Q4 ? 32 : 64) : a.get_int("group");
    for (std::string list = a.get("q8"); !list.empty();) {
        const size_t comma = list.find(',');
        o.q8_names.push_back(list.substr(0, comma));
        list = comma == std::string::npos ? std::string() : list.substr(comma + 1);
    }
    const std::string& scope = a.get("scope");
    if (scope != "bandwidth" && scope != "all") fail("--scope must be bandwidth or all, got '", scope, "'");
    o.scope = scope == "all" ? QuantScope::All : QuantScope::Bandwidth;
    const std::string& method = a.get("method");
    if (method != "rtn" && method != "awq") fail("--method must be rtn or awq, got '", method, "'");
    o.method = method == "awq" ? QuantMethod::Awq : QuantMethod::Rtn;
    if (!a.get("calib").empty()) {
        std::ifstream in(a.get("calib"));
        if (!in) fail("cannot read --calib file '", a.get("calib"), "'");
        o.calibration_text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    o.calibration_tokens = a.get_int("calib-tokens");
    o.backend = a.get("backend");
    o.device = a.get("device");
    o.converter = std::string("xpu-serve ") + XS_VERSION;
    const ConvertReport r = convert_model(o);
    std::printf("wrote %s: %lld tensors q4, %lld q8, %lld kept; %.2f GB -> %.2f GB in %d shard(s)\n",
                o.output.string().c_str(), static_cast<long long>(r.q4), static_cast<long long>(r.q8),
                static_cast<long long>(r.kept), static_cast<double>(r.source_bytes) / 1e9,
                static_cast<double>(r.output_bytes) / 1e9, r.shards);
    if (r.awq > 0) {
        std::printf("awq: %lld q4 tensors, weighted error %.1f%% below round to nearest on average\n",
                    static_cast<long long>(r.awq), 100.0 * r.awq_gain);
    }
    return 0;
}

// Branch exp/manual-quant: next-token distributions of a model (or pack) against a reference written by
// tools/manual_quant/ref_logits.py (tokens.bin, logprobs.bin, manifest.json). Every position runs as a decode step,
// as serving decodes. Prints mean KL(reference || model), top-1 agreement and perplexity, overall and per source.
int cmd_quant_eval(const std::vector<std::string>& argv) {
    namespace fs = std::filesystem;
    Args a;
    add_engine_options(a);
    a.value("ref", "directory of the reference (ref_logits.py --out)").value("seqs", "sequences to use (0 = all)", "0");
    a.flag("prefill", "score every position from one prefill per sequence (XS_ALL_LOGITS) instead of decode steps");
    a.parse(argv);
    if (wants_help(a, "quant-eval", "<model-dir> --ref <dir> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    const bool prefill = a.has("prefill");
    if (prefill) setenv("XS_ALL_LOGITS", "1", 1);
    const std::string dir = model_dir_arg(a, "quant-eval");
    const fs::path ref = a.get("ref");
    const auto manifest = cfg::read_json(ref / "manifest.json");
    const int64_t seq = manifest.at("seq").get<int64_t>();
    const int64_t vocab = manifest.at("vocab").get<int64_t>();
    int64_t count = manifest.at("count").get<int64_t>();
    if (a.get_int("seqs") > 0) count = std::min<int64_t>(count, a.get_int("seqs"));
    const auto sources = manifest.at("sources").get<std::vector<std::string>>();
    std::vector<int32_t> tokens(static_cast<size_t>(count * seq));
    std::ifstream(ref / "tokens.bin", std::ios::binary)
        .read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(int32_t)));
    std::ifstream lp_file(ref / "logprobs.bin", std::ios::binary);

    const ServiceOptions so = service_options(a);
    auto model = load_model(dir);
    SessionOptions opts = so.session;
    opts.max_context = seq + 8;
    auto backend = make_backend(so.backend);
    auto session = backend->compile(model->graph(), model->weights(), opts);
    struct Acc {
        double kl = 0, nll = 0, ref_nll = 0;
        int64_t top1 = 0, n = 0;
    };
    std::map<std::string, Acc> by_source;
    Acc all;
    std::vector<uint16_t> ref_lp(static_cast<size_t>(vocab));
    std::vector<double> ours(static_cast<size_t>(vocab));
    const auto start = std::chrono::steady_clock::now();
    for (int64_t s = 0; s < count; ++s) {
        session->reset();
        Acc acc;
        TensorMap all_outputs;
        if (prefill) {
            TensorMap inputs;
            model->prepare_inputs(std::span<const int32_t>(tokens.data() + s * seq, static_cast<size_t>(seq)), 0, {},
                                  inputs);
            session->run(inputs, all_outputs);
        }
        for (int64_t p = 0; p < seq; ++p) {
            lp_file.read(reinterpret_cast<char*>(ref_lp.data()), static_cast<std::streamsize>(vocab * 2));
            TensorMap inputs, outputs;
            if (!prefill) {
                const int32_t tok = tokens[static_cast<size_t>(s * seq + p)];
                model->prepare_inputs(std::span<const int32_t>(&tok, 1), p, {}, inputs);
                session->run(inputs, outputs);
            }
            if (p == seq - 1) break;  // the last position has no next token
            const float* l = prefill ? all_outputs.at("logits").data<float>() + p * vocab
                                     : outputs.at("logits").data<float>();
            double mx = -1e30;
            int64_t arg = 0;
            for (int64_t v = 0; v < vocab; ++v) {
                if (l[v] > mx) mx = l[v], arg = v;
            }
            double sum = 0;
            for (int64_t v = 0; v < vocab; ++v) sum += std::exp(static_cast<double>(l[v]) - mx);
            const double lse = mx + std::log(sum);
            double kl = 0;
            int64_t ref_arg = 0;
            float ref_best = -1e30f;
            for (int64_t v = 0; v < vocab; ++v) {
                const float r = f16_to_f32(ref_lp[static_cast<size_t>(v)]);
                if (r > ref_best) ref_best = r, ref_arg = v;
                const double pr = std::exp(static_cast<double>(r));
                if (pr > 0) kl += pr * (static_cast<double>(r) - (static_cast<double>(l[v]) - lse));
            }
            const int32_t next = tokens[static_cast<size_t>(s * seq + p + 1)];
            acc.kl += kl;
            acc.nll += lse - l[next];
            acc.ref_nll -= f16_to_f32(ref_lp[static_cast<size_t>(next)]);
            acc.top1 += arg == ref_arg ? 1 : 0;
            ++acc.n;
        }
        if (seq > 0) lp_file.seekg(static_cast<std::streamoff>(s + 1) * seq * vocab * 2);
        const std::string& src = sources[static_cast<size_t>(s)];
        for (Acc* t : {&by_source[src], &all}) {
            t->kl += acc.kl, t->nll += acc.nll, t->ref_nll += acc.ref_nll, t->top1 += acc.top1, t->n += acc.n;
        }
        std::printf("seq %2lld (%s): kl %.5f top1 %.4f ppl %.3f (ref %.3f)\n", static_cast<long long>(s), src.c_str(),
                    acc.kl / acc.n, static_cast<double>(acc.top1) / acc.n, std::exp(acc.nll / acc.n),
                    std::exp(acc.ref_nll / acc.n));
        std::fflush(stdout);
    }
    const auto line = [](const std::string& name, const Acc& t) {
        std::printf("%-10s kl %.5f  top1 %.4f  ppl %.4f  (ref %.4f, +%.2f%%)  positions %lld\n", name.c_str(),
                    t.kl / t.n, static_cast<double>(t.top1) / t.n, std::exp(t.nll / t.n), std::exp(t.ref_nll / t.n),
                    100.0 * (std::exp((t.nll - t.ref_nll) / t.n) - 1.0), static_cast<long long>(t.n));
    };
    for (const auto& [name, t] : by_source) line(name, t);
    line("all", all);
    std::printf("eval time %.0f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return 0;
}

// Branch exp/manual-quant: runs calibration token sequences (int32 [N, seq]) through a qwen3_5 model built with
// XS_CAPTURE=1 and appends each "cap.*" output (the inputs of the linear maps) as f16 rows to <out>/<name>.f16.
int cmd_quant_capture(const std::vector<std::string>& argv) {
    namespace fs = std::filesystem;
    Args a;
    add_engine_options(a);
    a.value("tokens", "int32 token file [N, seq]").value("seq", "tokens per sequence", "2048").value("out", "output dir");
    a.parse(argv);
    if (wants_help(a, "quant-capture", "<model-dir> --tokens <file> --out <dir> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    setenv("XS_CAPTURE", "1", 1);
    const std::string dir = model_dir_arg(a, "quant-capture");
    const int64_t seq = a.get_int("seq");
    const int64_t chunk = a.get_int("prefill-chunk");
    std::ifstream in(a.get("tokens"), std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<int64_t>(in.tellg());
    std::vector<int32_t> tokens(static_cast<size_t>(bytes / 4));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(tokens.data()), bytes);
    const int64_t count = static_cast<int64_t>(tokens.size()) / seq;
    const fs::path out = a.get("out");
    fs::create_directories(out);
    const ServiceOptions so = service_options(a);
    auto model = load_model(dir);
    SessionOptions opts = so.session;
    opts.max_context = seq + 8;
    auto session = make_backend(so.backend)->compile(model->graph(), model->weights(), opts);
    std::map<std::string, std::ofstream> files;
    const auto start = std::chrono::steady_clock::now();
    for (int64_t s = 0; s < count; ++s) {
        session->reset();
        for (int64_t p = 0; p < seq; p += chunk) {
            const int64_t n = std::min(chunk, seq - p);
            TensorMap inputs, outputs;
            model->prepare_inputs(std::span<const int32_t>(tokens.data() + s * seq + p, static_cast<size_t>(n)), p, {},
                                  inputs);
            session->run(inputs, outputs);
            for (const auto& [name, t] : outputs) {
                if (name.rfind("cap.", 0) != 0) continue;
                auto it = files.find(name);
                if (it == files.end()) it = files.emplace(name, std::ofstream(out / (name.substr(4) + ".f16"), std::ios::binary)).first;
                const std::vector<float> v = t.to_f32();
                std::vector<uint16_t> h(v.size());
                for (size_t i = 0; i < v.size(); ++i) h[i] = f32_to_f16(v[i]);
                it->second.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(h.size() * 2));
            }
        }
        std::printf("seq %lld captured (%.0f s)\n", static_cast<long long>(s),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        std::fflush(stdout);
    }
    return 0;
}

// policy-eval (branch exp/manual-quant): a policy's actions on recorded observations with fixed noise, as f32
// [cases, horizon, action_dim], to compare quantized packs. <cases>/manifest.json: {"cases": [{"cameras": {name:
// image file}, "state": [...], "instruction": "...", "noise": "<f32 [horizon, noise_dim] file>"}, ...]}.
int cmd_policy_eval(const std::vector<std::string>& argv) {
    namespace fs = std::filesystem;
    Args a;
    add_engine_options(a);
    a.value("cases", "directory with manifest.json").value("out", "actions file").value("steps", "flow steps (0 = the checkpoint's)", "0");
    a.value("capture", "also write every weight matmul's input as f16 rows to <dir>/<weight>.f16 (XS_CAPTURE=matmul)");
    a.parse(argv);
    if (wants_help(a, "policy-eval", "<policy-dir> --cases <dir> --out <file> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    const std::string dir = model_dir_arg(a, "policy-eval");
    const fs::path cases = a.get("cases");
    const std::string capture = a.get("capture");
    if (!capture.empty()) {
        setenv("XS_CAPTURE", "matmul", 1);
        fs::create_directories(capture);
    }
    std::map<std::string, std::ofstream> capture_files;
    const auto manifest = cfg::read_json(cases / "manifest.json");
    const ServiceOptions so = service_options(a);
    PolicyRunnerOptions ro;
    ro.backend = so.backend;
    ro.session = so.session;
    auto runner = PolicyRunner::create(dir, ro);
    const PolicySpec& spec = runner->spec();
    const int steps = a.get_int("steps") > 0 ? a.get_int("steps") : spec.denoise_steps;
    const auto read_file = [](const fs::path& p) {
        std::ifstream f(p, std::ios::binary);
        XS_CHECK(f.good(), "cannot read ", p.string());
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
    };
    std::ofstream out(a.get("out"), std::ios::binary);
    int64_t n = 0;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& c : manifest.at("cases")) {
        Observation obs;
        for (const CameraSpec& cam : spec.cameras) {
            if (c.at("cameras").contains(cam.name))
                obs.cameras.emplace_back(cam.name, image::decode(read_file(cases / c.at("cameras").at(cam.name).get<std::string>())));
        }
        obs.state = c.at("state").get<std::vector<float>>();
        obs.instruction = c.at("instruction").get<std::string>();
        const std::vector<uint8_t> raw = read_file(cases / c.at("noise").get<std::string>());
        std::vector<float> noise(static_cast<size_t>(spec.horizon * spec.noise_dim));
        XS_CHECK(raw.size() == noise.size() * sizeof(float), "noise file size: want [", spec.horizon, ", ", spec.noise_dim, "] f32");
        std::memcpy(noise.data(), raw.data(), raw.size());
        const HostTensor nt = HostTensor::from_vector(DType::F32, {spec.horizon, spec.noise_dim}, noise);
        ActOptions o;
        o.steps = steps;
        o.noise = &nt;
        const ActResult r = runner->act(obs, o);
        out.write(reinterpret_cast<const char*>(r.actions.data<float>()), static_cast<std::streamsize>(r.actions.nbytes()));
        for (const auto& [name, t] : r.captures) {  // "cap.<weight>#<n>": rows of every use go to one file
            const std::string weight = name.substr(4, name.rfind('#') - 4);
            auto it = capture_files.find(weight);
            if (it == capture_files.end())
                it = capture_files.emplace(weight, std::ofstream(fs::path(capture) / (weight + ".f16"), std::ios::binary)).first;
            const std::vector<float> v = t.to_f32();
            std::vector<uint16_t> h(v.size());
            for (size_t i = 0; i < v.size(); ++i) h[i] = f32_to_f16(v[i]);
            it->second.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(h.size() * 2));
        }
        ++n;
    }
    std::printf("%lld cases, actions [%lld, %lld] each, %.1f s\n", static_cast<long long>(n),
                static_cast<long long>(spec.horizon), static_cast<long long>(spec.action_dim),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return 0;
}

int cmd_info() {
    std::printf("xpu-serve %s\n", XS_VERSION);
    for (const auto& name : available_backends()) {
        std::printf("backend %s\n", name.c_str());
        try {
            for (const auto& d : make_backend(name)->devices()) {
                std::printf("  %-8s %s%s", d.id.c_str(), d.name.c_str(), d.integrated ? " (integrated)" : "");
                if (d.memory_bytes) std::printf(", %.1f GiB", static_cast<double>(d.memory_bytes) / (1ull << 30));
                std::printf("\n");
            }
        } catch (const std::exception& e) {
            std::printf("  unavailable: %s\n", e.what());
        }
    }
    return 0;
}

int cmd_serve(const std::vector<std::string>& argv) {
    Args a;
    add_engine_options(a);
    a.value("host", "address to bind", "127.0.0.1")
        .value("port", "port", "8080")
        .value("api-key", "require 'Authorization: Bearer <key>'")
        .value("model-id", "model name reported by /v1/models (default: directory name)")
        .value("http-threads", "HTTP worker threads", "8");
    a.parse(argv);
    if (wants_help(a, "serve", "<model-dir> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    ServiceOptions so = service_options(a);
    so.model_id = a.get("model-id");
    auto service = EngineService::create(model_dir_arg(a, "serve"), so);
    server::ServerOptions opts;
    opts.host = a.get("host");
    opts.port = a.get_int("port");
    opts.api_key = a.get("api-key");
    opts.http_threads = a.get_int("http-threads");
    server::Server srv(*service, opts);
    log::info("serving ", service->model().id, " on http://", opts.host, ":", opts.port);
    srv.run();
    return 0;
}

SamplingParams sampling_from(const Args& a, SamplingParams p) {
    if (a.has("temperature")) p.temperature = static_cast<float>(a.get_double("temperature"));
    if (a.has("top-k")) p.top_k = a.get_int("top-k");
    if (a.has("top-p")) p.top_p = static_cast<float>(a.get_double("top-p"));
    if (a.has("seed")) p.seed = static_cast<uint64_t>(a.get_int("seed"));
    return p;
}

// Streams one assistant turn to stdout and returns the raw generated text (for the conversation history). `media`
// holds every image of the conversation so far, in order.
std::string run_turn(EngineService& service, std::string_view prompt_text, const std::vector<MediaInput>& media,
                     const SamplingParams& sampling, int max_tokens) {
    const std::vector<int32_t> prompt = service.expand_media(service.tokenizer().encode(prompt_text), media);
    GenerateRequest req;
    req.prompt = prompt;
    req.media = media;
    req.sampling = sampling;
    req.max_new_tokens = max_tokens;
    req.stop_token_ids = service.model().stop_token_ids;
    tok::StreamDecoder decoder(service.tokenizer(), false);
    auto parser = chat::make_output_parser(service.model().dialect, chat::prompt_tail(prompt_text));
    std::string raw;
    bool in_reasoning = false;
    auto print = [&](const chat::ParsedDelta& d) {
        if (!d.reasoning.empty()) {
            if (!in_reasoning) std::fputs("\x1b[2m", stdout);
            in_reasoning = true;
            std::fputs(d.reasoning.c_str(), stdout);
        }
        if (!d.content.empty() || !d.tool_calls.empty()) {
            if (in_reasoning) std::fputs("\x1b[0m\n", stdout);
            in_reasoning = false;
            std::fputs(d.content.c_str(), stdout);
            for (const auto& c : d.tool_calls) std::printf("\n[tool call] %s(%s)\n", c.name.c_str(), c.arguments.c_str());
        }
        std::fflush(stdout);
    };
    const GenerateResult r = service.generate(req, [&](int32_t token) {
        const std::string text = decoder.push(token);
        raw += text;
        print(parser->feed(text));
        return true;
    });
    const std::string tail = decoder.flush();
    raw += tail;
    print(parser->feed(tail));
    print(parser->finish());
    if (in_reasoning) std::fputs("\x1b[0m", stdout);
    std::fputs("\n", stdout);
    std::fflush(stdout);  // the stats line goes to stderr and must come after the reply
    if (r.finish == FinishReason::Error) fail("generation failed: ", r.error);
    const GenerateStats& s = r.stats;
    std::string images;
    if (s.images_encoded > 0) {
        images = str_cat(s.images_encoded, s.images_encoded == 1 ? " image" : " images", " encoded in ",
                         static_cast<int>(s.encode_ms), " ms | ");
    }
    std::fprintf(stderr, "[%s%d prompt tokens (%d cached), prefill %.1f tok/s | %d tokens, decode %.1f tok/s | %s]\n",
                 images.c_str(), s.prompt_tokens, s.cached_tokens, s.prefill_tok_per_s(), s.generated_tokens,
                 s.decode_tok_per_s(), finish_reason_name(r.finish));
    return raw;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    XS_CHECK(in.good(), "cannot read '", path, "'");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int cmd_run(const std::vector<std::string>& argv) {
    Args a;
    add_engine_options(a);
    a.value("prompt", "one-shot prompt (default: interactive chat on stdin)")
        .value("system", "system prompt")
        .value("max-tokens", "maximum new tokens per turn", "1024")
        .value("temperature", "sampling temperature (default: the model's)")
        .value("top-k", "top-k (default: the model's)")
        .value("top-p", "top-p (default: the model's)")
        .value("seed", "sampling seed")
        .value("image", "image file(s) for the first message, comma-separated (PNG, JPEG, BMP, GIF)")
        .flag("think", "enable the model's thinking mode")
        .flag("raw", "feed --prompt to the model as is, without the chat template");
    a.parse(argv);
    if (wants_help(a, "run", "<model-dir> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    auto service = EngineService::create(model_dir_arg(a, "run"), service_options(a));
    const SamplingParams sampling = sampling_from(a, service->model().default_sampling);
    const int max_tokens = a.get_int("max-tokens");
    const nlohmann::json extra = {{"enable_thinking", a.has("think")}};

    if (a.has("raw")) {
        XS_CHECK(a.has("prompt"), "--raw needs --prompt");
        XS_CHECK(!a.has("image"), "--image needs the chat template; drop --raw");
        run_turn(*service, a.get("prompt"), {}, sampling, max_tokens);
        return 0;
    }
    nlohmann::json messages = nlohmann::json::array();
    if (a.has("system")) messages.push_back({{"role", "system"}, {"content", a.get("system")}});
    std::vector<MediaInput> media;
    std::string pending_images = a.get("image");
    auto turn = [&](const std::string& user_text) {
        nlohmann::json content = user_text;
        if (!pending_images.empty()) {
            content = nlohmann::json::array();
            for (size_t pos = 0; pos <= pending_images.size();) {
                const size_t comma = std::min(pending_images.find(',', pos), pending_images.size());
                const std::string path = pending_images.substr(pos, comma - pos);
                pos = comma + 1;
                if (path.empty()) continue;
                media.push_back(service->prepare_image(read_file(path)));
                content.push_back({{"type", "image"}});
            }
            content.push_back({{"type", "text"}, {"text", user_text}});
            pending_images.clear();
        }
        messages.push_back({{"role", "user"}, {"content", content}});
        const std::string text = service->chat_template().render(messages, nullptr, true, extra);
        const std::string reply = run_turn(*service, text, media, sampling, max_tokens);
        auto parser = chat::make_output_parser(service->model().dialect, chat::prompt_tail(text));
        chat::ParsedDelta d = parser->feed(reply);
        const chat::ParsedDelta end = parser->finish();
        messages.push_back({{"role", "assistant"}, {"content", d.content + end.content}});
    };
    if (a.has("prompt")) {
        turn(a.get("prompt"));
        return 0;
    }
    std::fprintf(stderr, "chatting with %s; empty line or Ctrl-D to quit\n", service->model().id.c_str());
    for (std::string line;;) {
        std::fputs("> ", stdout);
        std::fflush(stdout);
        if (!std::getline(std::cin, line) || line.empty()) break;
        turn(line);
    }
    return 0;
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const double at = p * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<size_t>(at);
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (at - static_cast<double>(lo));
}

// Latency of a robot policy: warm-up (compile, then untimed runs), then timed calls on a synthetic observation.
int bench_policy(const Args& a, const std::string& dir) {
    const ServiceOptions so = service_options(a);
    PolicyRunnerOptions ro;
    ro.backend = so.backend;
    ro.session = so.session;
    auto runner = PolicyRunner::create(dir, ro);
    const PolicySpec& spec = runner->spec();
    const int n_cam = a.get_int("cameras") > 0 ? a.get_int("cameras") : static_cast<int>(spec.cameras.size());
    XS_CHECK(n_cam >= 1 && n_cam <= static_cast<int>(spec.cameras.size()), "--cameras must be 1 to ", spec.cameras.size());
    std::vector<int> cameras;
    for (int c = 0; c < n_cam; ++c) cameras.push_back(c);
    const int steps = a.get_int("steps") > 0 ? a.get_int("steps") : spec.denoise_steps;
    const Observation obs = synthetic_observation(spec, cameras, "pick up the red cup and place it on the plate");
    const double compile_ms = runner->warm_up(obs, steps, a.get_int("warmup"));
    std::vector<double> total, prepare, device;
    size_t prompt_tokens = 0;
    for (int run = 0; run < a.get_int("runs"); ++run) {
        ActOptions o;
        o.steps = steps;
        o.seed = static_cast<uint64_t>(run);
        const ActResult r = runner->act(obs, o);
        prompt_tokens = r.prompt_ids.size();
        total.push_back(r.timing.total_ms);
        prepare.push_back(r.timing.prepare_ms);
        device.push_back(r.timing.device_ms);
        if (a.has("verbose")) {
            std::printf("run %d: %.2f ms (prepare %.2f, device %.2f, finish %.2f)\n", run, r.timing.total_ms,
                        r.timing.prepare_ms, r.timing.device_ms, r.timing.finish_ms);
        }
    }
    std::printf("%s: %d camera(s) %lldx%lld, %d flow steps, prompt %zu tokens, %s %s %s, compile %.0f ms\n",
                spec.id.c_str(), n_cam, static_cast<long long>(spec.cameras[0].height),
                static_cast<long long>(spec.cameras[0].width), steps, prompt_tokens, so.backend.c_str(),
                so.session.device.empty() ? "default" : so.session.device.c_str(),
                std::string(dtype_name(so.session.precision)).c_str(), compile_ms);
    std::printf("latency over %zu runs: p50 %.2f ms, p90 %.2f, min %.2f, max %.2f | device p50 %.2f, host prepare p50 "
                "%.2f\n",
                total.size(), percentile(total, 0.5), percentile(total, 0.9), percentile(total, 0.0),
                percentile(total, 1.0), percentile(device, 0.5), percentile(prepare, 0.5));
    return 0;
}

int cmd_bench(const std::vector<std::string>& argv) {
    Args a;
    add_engine_options(a);
    a.value("prompt-tokens", "prompt length (language models)", "512")
        .value("gen-tokens", "tokens to generate (language models)", "128")
        .value("runs", "timed runs", "3")
        .value("cameras", "policies: cameras present, the first N (default: all)", "0")
        .value("steps", "policies: flow steps (default: the checkpoint's)", "0")
        .value("warmup", "policies: untimed runs after compiling", "3")
        .flag("verbose", "policies: print every run");
    a.parse(argv);
    if (wants_help(a, "bench", "<model-dir> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    const std::string dir = model_dir_arg(a, "bench");
    if (model_kind(dir) == ModelKind::Policy) return bench_policy(a, dir);
    auto service = EngineService::create(dir, service_options(a));
    const int n_prompt = a.get_int("prompt-tokens");
    const int n_gen = a.get_int("gen-tokens");
    std::string text;
    while (static_cast<int>(service->tokenizer().encode(text).size()) < n_prompt + 16) {
        text += "The quick brown fox jumps over the lazy dog while the engine measures how fast it reads and writes. ";
    }
    for (int run = 0; run < a.get_int("runs"); ++run) {
        // A different first token per run, so no run reuses the previous run's KV cache.
        std::vector<int32_t> prompt = service->tokenizer().encode("Run " + std::to_string(run) + ": " + text);
        prompt.resize(static_cast<size_t>(n_prompt));
        GenerateRequest req;
        req.prompt = prompt;
        req.sampling.temperature = 0.0f;
        req.max_new_tokens = n_gen;
        const GenerateResult r = service->generate(req, {});
        if (r.finish == FinishReason::Error) fail("generation failed: ", r.error);
        std::printf("run %d: prompt %d tok, prefill %.1f ms (%.1f tok/s) | generated %d tok, decode %.1f tok/s\n", run,
                    r.stats.prompt_tokens, r.stats.prefill_ms, r.stats.prefill_tok_per_s(), r.stats.generated_tokens,
                    r.stats.decode_tok_per_s());
    }
    return 0;
}

}  // namespace
}  // namespace xs::cli

int main(int argc, char** argv) {
    using namespace xs::cli;
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "--help" || args[0] == "-h" || args[0] == "help") {
        std::fputs(kUsage, stdout);
        return args.empty() ? 1 : 0;
    }
    const std::string command = args[0];
    args.erase(args.begin());
    try {
        if (command == "serve") return cmd_serve(args);
        if (command == "run") return cmd_run(args);
        if (command == "bench") return cmd_bench(args);
        if (command == "convert") return cmd_convert(args);
        if (command == "quant-eval") return cmd_quant_eval(args);
        if (command == "quant-capture") return cmd_quant_capture(args);
        if (command == "policy-eval") return cmd_policy_eval(args);
        if (command == "info") return cmd_info();
        if (command == "--version" || command == "version") {
            std::printf("xpu-serve %s\n", XS_VERSION);
            return 0;
        }
        std::fprintf(stderr, "unknown command '%s'\n\n%s", command.c_str(), kUsage);
        return 1;
    } catch (const std::exception& e) {
        xs::log::error(e.what());
        return 1;
    }
}
