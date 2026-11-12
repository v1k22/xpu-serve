// xpu-serve command line: serve | run | bench | convert | info.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "chat/chat_template.h"
#include "chat/output_parser.h"
#include "cli/args.h"
#include "convert/convert.h"
#include "decision/runner.h"
#include "decision/service.h"
#include "models/letter_readout/letter_readout.h"
#include "models/policy.h"
#include "policy/runner.h"
#include "policy/service.h"
#include "server/decision_handlers.h"
#include "server/server.h"
#include "service/engine_service.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"
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
    "  xpu-serve serve <model-dir> [options]   OpenAI-compatible server (a decision model: the System One API)\n"
    "  xpu-serve run   <model-dir> [options]   chat in the terminal (--prompt for one shot)\n"
    "  xpu-serve bench <model-dir> [options]   prefill / decode speed, or a decision model's latency\n"
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
    a.value(
         "backend",
         "backend: " +
             [] {
                 std::string s;
                 for (const auto& b : available_backends())
                     s += (s.empty() ? "" : " | ") + b;
                 return s;
             }(),
         default_backend())
        .value("device", "device, e.g. GPU, GPU.1, CPU (default: the backend's choice)")
        .value("precision", "compute precision: f16 | f32", "f16")
        .value("kv-dtype", "KV cache: auto (the precision's float) | i8 (int8 with a scale per token and head)", "auto")
        .value("ctx", "context length in tokens (prompt + output)", "8192")
        .value("prefill-chunk", "tokens per prefill step", "1024")
        .value("backend-opt", "backend options as key=value[,key=value]")
        .value("spec",
               "speculative decoding: off | mtp (the model's MTP head, or a Gemma 4 assistant with --drafter) | "
               "dflash | dspine (a block drafter checkpoint with --drafter)",
               "off")
        .value("spec-k", "speculative decoding: drafts per round (default: 3, or 7 of a block drafter's block)")
        .value("drafter",
               "speculative decoding: a drafter checkpoint (a Gemma 4 assistant, a DFlash or DSpine drafter)")
        .value("draft-shortlist",
               "speculative decoding: an MTP head picks among this many tokens, chosen by a one-bit "
               "sketch of the output layer and re-scored exactly (0: it reads the whole layer)",
               "64")
        .value("readout",
               "serve a language model as a decision model, its answers read at the option letters: letters | "
               "openjev (OpenJev's published calibration)")
        .value("readout-temperature", "--readout: the letter scores are divided by it (default: the preset's)")
        .value("readout-noul-temperature",
               "--readout: the slope of a yes / no answer's calibration (default: the preset's)")
        .value("readout-noul-bias", "--readout: the bias of a yes / no answer's calibration (default: the preset's)")
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
    const std::string& spec = a.get("spec");
    if (spec == "mtp" || spec == "dflash" || spec == "dspine") {
        o.spec = spec;
        o.engine.spec_k = a.get("spec-k").empty() ? kDraftersCount : a.get_int("spec-k");
        if (o.engine.spec_k != kDraftersCount && o.engine.spec_k < 1) fail("--spec-k must be at least 1, got ", o.engine.spec_k);
    } else if (spec != "off") {
        fail("--spec must be off, mtp, dflash or dspine, got '", spec, "'");
    }
    o.drafter = a.get("drafter");
    if (!o.drafter.empty() && spec == "off") fail("--drafter drafts for --spec mtp, dflash or dspine");
    if (o.drafter.empty() && (spec == "dflash" || spec == "dspine")) fail("--spec ", spec, " needs --drafter <its checkpoint>");
    o.draft_shortlist = a.get_int("draft-shortlist");
    if (o.draft_shortlist < 0) fail("--draft-shortlist must be 0 or more, got ", o.draft_shortlist);
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

// A model directory answers decisions: it holds a decision model, or --readout reads a language model as one.
bool decides(const Args& a, const std::string& dir) {
    return a.has("readout") || model_kind(dir) == ModelKind::Decision;
}

// The decision runtime for `dir`: its decision model, or with --readout its language model read at the option letters.
std::unique_ptr<DecisionRunner> decision_runner(const Args& a, const std::string& dir, const DecisionRunnerOptions& o) {
    if (!a.has("readout")) return DecisionRunner::create(dir, o);
    if (model_kind(dir) != ModelKind::Language) fail("--readout reads a language model; '", dir, "' is not one");
    std::optional<letter_readout::Settings> settings = letter_readout::preset(a.get("readout"));
    if (!settings) {
        std::string list;
        for (const auto& n : letter_readout::preset_names())
            list += (list.empty() ? "" : ", ") + n;
        fail("--readout must be one of ", list, "; got '", a.get("readout"), "'");
    }
    if (a.has("readout-temperature")) settings->temperature = a.get_double("readout-temperature");
    if (a.has("readout-noul-temperature")) settings->noul_temperature = a.get_double("readout-noul-temperature");
    if (a.has("readout-noul-bias")) settings->noul_bias = a.get_double("readout-noul-bias");
    return DecisionRunner::create(dir, o,
                                  letter_readout::make(load_model(dir, LoadOptions{.vision = false}), *settings));
}

DecisionRunnerOptions decision_options(const ServiceOptions& so) {
    DecisionRunnerOptions o;
    o.backend = so.backend;
    o.session = so.session;
    o.session.max_context = so.engine.max_context;
    o.chunk_tokens = so.engine.prefill_chunk;
    o.model_id = so.model_id;
    if (!so.spec.empty()) fail("--spec is for language models: a decision model generates no text");
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
        .value("target", "<model-dir> is a drafter checkpoint (DFlash, DSpine, an assistant) of this model or pack")
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
    o.target = a.get("target");
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
        .value("http-threads", "HTTP worker threads", "8")
        .value("cameras", "policies: warm up for the first N cameras (default: all; others compile on first use)", "0");
    a.parse(argv);
    if (wants_help(a, "serve", "<model-dir> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    ServiceOptions so = service_options(a);
    so.model_id = a.get("model-id");
    const std::string dir = model_dir_arg(a, "serve");
    server::ServerOptions opts;
    opts.host = a.get("host");
    opts.port = a.get_int("port");
    opts.api_key = a.get("api-key");
    opts.http_threads = a.get_int("http-threads");
    if (model_kind(dir) == ModelKind::Policy) {  // the policy API; /ready once warmed up
        PolicyRunnerOptions ro;
        ro.backend = so.backend;
        ro.session = so.session;
        auto policy = PolicyService::create(PolicyRunner::create(dir, ro), static_cast<int>(a.get_int("cameras")));
        server::Server srv(*policy, opts);
        log::info("serving policy ", policy->spec().id, " on http://", opts.host, ":", opts.port);
        srv.run();
        return 0;
    }
    if (decides(a, dir)) {  // the System One API
        auto decisions = DecisionService::create(decision_runner(a, dir, decision_options(so)));
        server::Server srv(*decisions, opts);
        log::info("serving decision model ", decisions->spec().id, " on http://", opts.host, ":", opts.port);
        srv.run();
        return 0;
    }
    auto service = EngineService::create(dir, so);
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
    char spec[48] = "";
    if (s.spec_rounds > 0) std::snprintf(spec, sizeof spec, " (%.2f per verify)", s.tokens_per_round());
    std::fprintf(stderr, "[%s%d prompt tokens (%d cached), prefill %.1f tok/s | %d tokens, decode %.1f tok/s%s | %s]\n",
                 images.c_str(), s.prompt_tokens, s.cached_tokens, s.prefill_tok_per_s(), s.generated_tokens,
                 s.decode_tok_per_s(), spec, finish_reason_name(r.finish));
    return raw;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    XS_CHECK(in.good(), "cannot read '", path, "'");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// A System One request body from a file ("model" may be left out here).
DecisionRequest read_decision_request(const std::string& path) {
    std::ifstream in(path);
    XS_CHECK(in.good(), "cannot read '", path, "'");
    nlohmann::ordered_json body = nlohmann::ordered_json::parse(in, nullptr, false);
    XS_CHECK(body.is_object(), "'", path, "' is not a JSON object");
    if (!body.contains("model")) body["model"] = "";
    try {
        return server::parse_systemone_request(body);
    } catch (const server::SystemOneInvalid& e) {
        std::string where;
        for (const std::string& part : e.loc())
            where += (where.empty() ? "" : ".") + part;
        fail("'", path, "': ", where, ": ", e.what());
    }
}

// One request through a decision model: the answers on stdout, tokens and timings on stderr.
int run_decision(const Args& a, const std::string& dir) {
    XS_CHECK(a.has("request"), "a decision model answers a request: --request <file with {\"state\", \"questions\"}>");
    const DecisionRequest request = read_decision_request(a.get("request"));
    auto runner = decision_runner(a, dir, decision_options(service_options(a)));
    const DecisionResult r = runner->decide(request);
    std::printf("%s\n", server::systemone_response(request, r, runner->spec().id).dump(2).c_str());
    std::fflush(stdout);
    std::fprintf(stderr, "[%lld tokens in %d sequence(s), %d steps: %.1f ms (prompt %.1f, device %.1f)]\n",
                 static_cast<long long>(r.input_tokens), static_cast<int>(r.sequences.size()), r.steps,
                 r.timing.total_ms, r.timing.prompt_ms, r.timing.device_ms);
    return 0;
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
        .value("request",
               "decision models: a file holding a System One request ({\"state\", \"questions\"}); its "
               "answers are printed as the API returns them")
        .flag("think", "enable the model's thinking mode")
        .flag("raw", "feed --prompt to the model as is, without the chat template");
    a.parse(argv);
    if (wants_help(a, "run", "<model-dir> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    if (decides(a, model_dir_arg(a, "run"))) return run_decision(a, model_dir_arg(a, "run"));
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

// Latency of a decision model: timed requests after the runner's warm-up. The request is the one in --request, or
// the runtime's sample; --state-tokens swaps its state for filler text of that many tokens.
int bench_decision(const Args& a, const std::string& dir) {
    const ServiceOptions so = service_options(a);
    auto runner = decision_runner(a, dir, decision_options(so));
    DecisionRequest request = a.has("request") ? read_decision_request(a.get("request")) : sample_decision_request();
    // A text state gets a line of its own in front, different every run, so that each run reads a state the session
    // has not read; --state-tokens sizes the prompt with that line in place.
    const bool text = request.state.is_string() || a.get_int("state-tokens") > 0;
    std::string state = request.state.is_string() ? request.state.get<std::string>() : "";
    const auto with_run = [&](int run) { request.state = str_cat("Run ", run, ".\n", state); };
    if (const int want = a.get_int("state-tokens"); want > 0) {
        const auto tokens = [&] {
            with_run(0);
            return static_cast<int>(runner->tokens(request).front().size());
        };
        state.clear();
        int line = 0;
        const auto grow = [&](int lines) {
            for (int i = 0; i < lines; ++i) {
                state += str_cat("Entry ", ++line,
                                 ": the order shipped on time, the invoice matched and the customer confirmed receipt.\n");
            }
        };
        for (int have = tokens(); have < want; have = tokens()) grow(1 + (want - have) / 24);  // a line: about 20 tokens
        for (int have = tokens(); have > want && state.size() > 8; have = tokens()) {
            state.resize(state.size() - std::max<size_t>(1, static_cast<size_t>(have - want) * 3));
        }
    }
    // Each run asks twice: about a state the session has not read, then about that state again, which a model that
    // keeps a state's prefix answers from there.
    std::vector<double> fresh, again, device;
    DecisionResult last;
    for (int run = 0; run < a.get_int("runs"); ++run) {
        if (text) with_run(run);
        last = runner->decide(request);
        fresh.push_back(last.timing.total_ms);
        device.push_back(last.timing.device_ms);
        const DecisionResult repeat = runner->decide(request);
        again.push_back(repeat.timing.total_ms);
        if (a.has("verbose")) {
            std::printf("run %d: %.2f ms (prompt %.2f, device %.2f), the same state again %.2f ms\n", run,
                        last.timing.total_ms, last.timing.prompt_ms, last.timing.device_ms, repeat.timing.total_ms);
        }
    }
    size_t options = 0;
    for (const DecisionQuestion& q : request.questions)
        options += q.options.size();
    std::printf("%s: %zu question(s) with %zu options, %lld tokens in %d sequence(s), %d steps, %s %s %s\n",
                runner->spec().id.c_str(), request.questions.size(), options, static_cast<long long>(last.input_tokens),
                static_cast<int>(last.sequences.size()), last.steps, so.backend.c_str(),
                so.session.device.empty() ? "default" : so.session.device.c_str(),
                std::string(dtype_name(so.session.precision)).c_str());
    std::printf(
        "latency over %zu runs: p50 %.2f ms, p90 %.2f, min %.2f, max %.2f (device p50 %.2f) | the same state "
        "again: p50 %.2f ms\n",
        fresh.size(), percentile(fresh, 0.5), percentile(fresh, 0.9), percentile(fresh, 0.0), percentile(fresh, 1.0),
        percentile(device, 0.5), percentile(again, 0.5));
    return 0;
}

// Greedy decode speed over real prompts (what speculative decoding is measured on: its acceptance depends on the text).
int bench_prompts(EngineService& service, const std::string& path, int n_gen, int runs) {
    std::ifstream in(path);
    XS_CHECK(in.good(), "cannot read '", path, "'");
    std::vector<std::vector<int32_t>> prompts;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        const nlohmann::json messages = nlohmann::json::array({{{"role", "user"}, {"content", line}}});
        const std::string text = service.chat_template().render(messages, nlohmann::json::array(), true,
                                                                nlohmann::json{{"enable_thinking", false}});
        prompts.push_back(service.tokenizer().encode(text));
    }
    XS_CHECK(!prompts.empty(), "'", path, "' holds no prompt");
    for (int run = 0; run < runs; ++run) {
        GenerateStats total;
        for (const auto& prompt : prompts) {
            GenerateRequest req;
            req.prompt = prompt;
            req.sampling.temperature = 0.0f;
            req.max_new_tokens = n_gen;
            req.stop_token_ids = service.model().stop_token_ids;
            const GenerateResult r = service.generate(req, {});
            if (r.finish == FinishReason::Error) fail("generation failed: ", r.error);
            const GenerateStats& s = r.stats;
            total.prompt_tokens += s.prompt_tokens - s.cached_tokens;
            total.prefill_ms += s.prefill_ms;
            total.generated_tokens += s.generated_tokens - 1;  // decode speed counts the tokens after the first
            total.decode_ms += s.decode_ms;
            total.spec_rounds += s.spec_rounds;
            total.spec_drafted += s.spec_drafted;
            total.spec_accepted += s.spec_accepted;
            total.spec_draft_ms += s.spec_draft_ms;
            total.spec_verify_ms += s.spec_verify_ms;
        }
        std::printf("run %d: %zu prompts, %d prompt tok, prefill %.1f tok/s | %d tok after the first, decode %.1f tok/s",
                    run, prompts.size(), total.prompt_tokens, total.prefill_tok_per_s(), total.generated_tokens,
                    total.decode_ms > 0 ? total.generated_tokens * 1000.0 / total.decode_ms : 0.0);
        if (total.spec_rounds > 0) {
            std::printf(" | %.2f tok per verify, %d of %d drafts accepted | a draft %.2f ms, a verify %.1f ms, a round %.1f ms",
                        total.tokens_per_round(), total.spec_accepted, total.spec_drafted,
                        total.spec_drafted > 0 ? total.spec_draft_ms / total.spec_drafted : 0.0,
                        total.spec_verify_ms / total.spec_rounds, total.decode_ms / total.spec_rounds);
        }
        std::printf("\n");
    }
    return 0;
}

int cmd_bench(const std::vector<std::string>& argv) {
    Args a;
    add_engine_options(a);
    a.value("prompt-tokens", "prompt length (language models)", "512")
        .value("gen-tokens", "tokens to generate (language models)", "128")
        .value("prompts", "language models: a file of prompts, one per line, each a user message through the chat "
               "template (thinking off) generating up to --gen-tokens until the end of the turn; instead of "
               "--prompt-tokens")
        .value("runs", "timed runs", "3")
        .value("cameras", "policies: cameras present, the first N (default: all)", "0")
        .value("steps", "policies: flow steps (default: the checkpoint's)", "0")
        .value("warmup", "policies: untimed runs after compiling", "3")
        .value("request",
               "decision models: a file holding a System One request (default: a short sample of each "
               "question type)")
        .value("state-tokens",
               "decision models: replace the request's state with filler so the prompt has this many "
               "tokens",
               "0")
        .flag("verbose", "policies and decision models: print every run");
    a.parse(argv);
    if (wants_help(a, "bench", "<model-dir> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    const std::string dir = model_dir_arg(a, "bench");
    if (model_kind(dir) == ModelKind::Policy) return bench_policy(a, dir);
    if (decides(a, dir)) return bench_decision(a, dir);
    auto service = EngineService::create(dir, service_options(a));
    const int n_gen = a.get_int("gen-tokens");
    if (!a.get("prompts").empty()) return bench_prompts(*service, a.get("prompts"), n_gen, a.get_int("runs"));
    const int n_prompt = a.get_int("prompt-tokens");
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
        std::printf("run %d: prompt %d tok, prefill %.1f ms (%.1f tok/s) | generated %d tok, decode %.1f tok/s", run,
                    r.stats.prompt_tokens, r.stats.prefill_ms, r.stats.prefill_tok_per_s(), r.stats.generated_tokens,
                    r.stats.decode_tok_per_s());
        if (r.stats.spec_rounds > 0) std::printf(" (%.2f per verify)", r.stats.tokens_per_round());
        std::printf("\n");
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
