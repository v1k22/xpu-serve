// xpu-serve command line: serve | run | bench | info.

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
    "  xpu-serve serve <model-dir> [options]   OpenAI-compatible server\n"
    "  xpu-serve run   <model-dir> [options]   chat in the terminal (--prompt for one shot)\n"
    "  xpu-serve bench <model-dir> [options]   prefill / decode speed\n"
    "  xpu-serve info                          backends and devices\n"
    "\n"
    "<model-dir> is a Hugging Face model directory (config.json, *.safetensors, tokenizer.json).\n"
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
        .value("ctx", "context length in tokens (prompt + output)", "8192")
        .value("prefill-chunk", "tokens per prefill step", "1024")
        .value("cache-dir", "backend compile cache directory")
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
    o.vision = !a.has("no-vision");
    o.engine.max_context = a.get_int("ctx");
    o.engine.prefill_chunk = a.get_int("prefill-chunk");
    o.session.cache_dir = a.get("cache-dir");
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

int cmd_bench(const std::vector<std::string>& argv) {
    Args a;
    add_engine_options(a);
    a.value("prompt-tokens", "prompt length", "512").value("gen-tokens", "tokens to generate", "128").value("runs", "runs", "3");
    a.parse(argv);
    if (wants_help(a, "bench", "<model-dir> [options]")) return 0;
    log::set_level(log::parse_level(a.get("log-level")));
    auto service = EngineService::create(model_dir_arg(a, "bench"), service_options(a));
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
