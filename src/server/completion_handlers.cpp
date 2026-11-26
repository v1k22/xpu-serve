#include "server/completion_handlers.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "chat/chat_template.h"
#include "server/anthropic_format.h"
#include "server/anthropic_request.h"
#include "server/generation_run.h"
#include "server/openai_format.h"
#include "server/openai_request.h"
#include "server/response_format.h"
#include "server/token_output.h"
#include "tokenizer/tokenizer.h"
#include "util/log.h"

namespace xs::server {

using nlohmann::json;

// Everything a response needs once the prompt is tokenized. Copyable: the stream provider keeps a copy.
struct CompletionHandlers::Job {
    std::string id;  // for the log
    std::function<std::unique_ptr<ResponseFormat>()> format;
    GenerateRequest request;
    GenerationOptions options;
    std::string dialect;  // output parser; raw completions are always "plain"
    std::string prompt_tail;  // chat::prompt_tail of the prompt text: the parser's start state
};

namespace {

std::string random_id(std::string_view prefix) {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    char hex[25];
    std::snprintf(hex, sizeof(hex), "%016llx%08llx", static_cast<unsigned long long>(rng()),
                  static_cast<unsigned long long>(rng() & 0xffffffffULL));
    return std::string(prefix) + hex;
}

std::string fixed1(double value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", value);
    return buf;
}

// Budget for new tokens: what the client asked for, capped by what is left of the context window.
GenerateRequest make_generate_request(std::vector<int32_t> prompt, const GenerationOptions& options,
                                      const ModelCard& card, const char* prompt_param) {
    if (prompt.empty()) throw InvalidRequest(str_cat("'", prompt_param, "' encodes to no tokens"), prompt_param);
    GenerateRequest r;
    const int64_t prompt_tokens = int64_t(prompt.size());
    if (card.context_length > 0) {
        if (prompt_tokens >= card.context_length) {
            throw InvalidRequest(str_cat("the prompt has ", prompt_tokens, " tokens but the model's context is ",
                                         card.context_length, " tokens"),
                                 prompt_param);
        }
        const int64_t room = card.context_length - prompt_tokens;
        r.max_new_tokens = int(options.max_tokens.has_value() ? std::min<int64_t>(*options.max_tokens, room) : room);
    } else if (options.max_tokens.has_value()) {
        r.max_new_tokens = *options.max_tokens;
    }
    r.prompt = std::move(prompt);
    r.sampling = options.sampling;
    r.stop_token_ids = card.stop_token_ids;
    return r;
}

std::string_view outcome_finish_reason(const GenerationOutcome& outcome, bool has_tool_calls) {
    return finish_reason_text(outcome.result.finish, outcome.stop_string_hit, has_tool_calls);
}

void log_summary(std::string_view id, const GenerationOutcome& outcome, bool has_tool_calls) {
    const GenerateStats& s = outcome.result.stats;
    std::string finish;
    if (outcome.client_gone) {
        finish = "cancelled (client gone)";
    } else if (outcome.result.finish == FinishReason::Error) {
        finish = "error: " + outcome.result.error;
    } else {
        finish = outcome_finish_reason(outcome, has_tool_calls);
    }
    const std::string media =
        s.media_encoded > 0 ? str_cat(" media_encoded=", s.media_encoded, " (", fixed1(s.encode_ms), " ms)") : "";
    log::info(id, ": prompt=", s.prompt_tokens, " cached=", s.cached_tokens, media, " generated=", s.generated_tokens,
              " prefill=", fixed1(s.prefill_tok_per_s()), " tok/s decode=", fixed1(s.decode_tok_per_s()),
              " tok/s finish=", finish);
}

}  // namespace

CompletionHandlers::Prompt CompletionHandlers::chat_prompt(const ChatRequest& chat) const {
    const ModelCard& card = service_.model();
    Prompt p;
    for (const MediaPart& part : chat.media) {
        if (!service_.accepts(part.modality)) {
            const char* kind = modality_name(part.modality);
            throw InvalidRequest(str_cat("model '", card.id, "' does not take ", kind, " input"), part.param);
        }
        try {
            p.media.push_back(service_.prepare_media(part.modality, part.bytes));
        } catch (const Error& e) {
            throw InvalidRequest(str_cat(part.param, ": ", e.what()), part.param);
        }
    }
    size_t tail = 0;  // tokens of the generation prompt, at the prompt's end
    try {
        p.text = service_.chat_template().render(chat.messages, chat.tools, /*add_generation_prompt=*/true,
                                                 chat.template_kwargs);
        p.tokens = service_.tokenizer().encode(p.text, /*parse_special=*/true);
        const std::string messages = service_.chat_template().render(chat.messages, chat.tools,
                                                                     /*add_generation_prompt=*/false,
                                                                     chat.template_kwargs);
        if (messages.size() < p.text.size() && p.text.starts_with(messages)) {
            const std::vector<int32_t> generation =
                service_.tokenizer().encode(std::string_view(p.text).substr(messages.size()), /*parse_special=*/true);
            if (generation.size() < p.tokens.size() &&
                std::equal(generation.rbegin(), generation.rend(), p.tokens.rbegin())) {
                tail = generation.size();
            }
        }
    } catch (const std::exception& e) {
        throw InvalidRequest(str_cat("could not render the chat template: ", e.what()), "messages");
    }
    try {
        p.tokens = service_.expand_media(std::move(p.tokens), p.media);
    } catch (const Error& e) {
        throw InvalidRequest(e.what(), "messages");
    }
    if (tail > 0) p.messages_end = static_cast<int64_t>(p.tokens.size() - tail);  // media sit among the messages
    return p;
}

void CompletionHandlers::chat(const httplib::Request& req, httplib::Response& res) const {
    const ModelCard& card = service_.model();
    ChatRequest chat = parse_chat_request(parse_json_body(req.body), card.default_sampling);
    Prompt p = chat_prompt(chat);
    GenerateRequest request = make_generate_request(std::move(p.tokens), chat.options, card, "messages");
    request.media = std::move(p.media);
    request.snapshot_at = p.messages_end;
    const ResponseMeta meta{random_id("chatcmpl-"), card.id, unix_seconds_now()};
    const bool usage = chat.options.include_usage;
    Job job{.id = meta.id,
            .format = [meta, usage] { return openai_chat_format(meta, usage); },
            .request = std::move(request),
            .options = std::move(chat.options),
            .dialect = card.dialect,
            .prompt_tail = std::string(chat::prompt_tail(p.text))};
    respond(job, req, res);
}

void CompletionHandlers::messages(const httplib::Request& req, httplib::Response& res) const {
    const ModelCard& card = service_.model();
    ChatRequest chat = parse_messages_request(parse_json_body(req.body), card.default_sampling, card.dialect);
    Prompt p = chat_prompt(chat);
    const auto input_tokens = static_cast<int64_t>(p.tokens.size());
    GenerateRequest request = make_generate_request(std::move(p.tokens), chat.options, card, "messages");
    request.media = std::move(p.media);
    request.snapshot_at = p.messages_end;
    const std::string id = random_id("msg_");
    Job job{.id = id,
            .format = [id, model = card.id, input_tokens] { return anthropic_format(id, model, input_tokens); },
            .request = std::move(request),
            .options = std::move(chat.options),
            .dialect = card.dialect,
            .prompt_tail = std::string(chat::prompt_tail(p.text))};
    respond(job, req, res);
}

void CompletionHandlers::count_tokens(const httplib::Request& req, httplib::Response& res) const {
    const ModelCard& card = service_.model();
    const ChatRequest chat =
        parse_messages_request(parse_json_body(req.body), card.default_sampling, card.dialect, /*count_only=*/true);
    res.set_content(json{{"input_tokens", chat_prompt(chat).tokens.size()}}.dump(), "application/json");
}

void CompletionHandlers::completion(const httplib::Request& req, httplib::Response& res) const {
    const ModelCard& card = service_.model();
    CompletionRequest completion = parse_completion_request(parse_json_body(req.body), card.default_sampling);
    std::vector<int32_t> prompt;
    try {
        prompt = service_.tokenizer().encode(completion.prompt, /*parse_special=*/true);
    } catch (const std::exception& e) {
        throw InvalidRequest(str_cat("could not tokenize the prompt: ", e.what()), "prompt");
    }
    const ResponseMeta meta{random_id("cmpl-"), card.id, unix_seconds_now()};
    const bool usage = completion.options.include_usage;
    Job job{.id = meta.id,
            .format = [meta, usage] { return openai_text_format(meta, usage); },
            .request = make_generate_request(std::move(prompt), completion.options, card, "prompt"),
            .options = std::move(completion.options),
            .dialect = "plain",
            .prompt_tail = std::string(chat::prompt_tail(completion.prompt))};
    respond(job, req, res);
}

void CompletionHandlers::respond(const Job& job, const httplib::Request& req, httplib::Response& res) const {
    if (!job.options.stream) return write_full(job, req, res);
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream", [this, job, &req](size_t, httplib::DataSink& sink) {
        // Returning false tells httplib the client is gone; it then drops the connection.
        if (!write_stream(job, req, sink)) return false;
        sink.done();
        return true;
    });
}

void CompletionHandlers::write_full(const Job& job, const httplib::Request& req, httplib::Response& res) const {
    TokenOutput output(service_.tokenizer(), chat::make_output_parser(job.dialect, job.prompt_tail), job.options.stop);
    chat::ParsedDelta message;
    const GenerationOutcome outcome = run_generation(
        service_, job.request, output,
        [&](const chat::ParsedDelta& delta) {
            append_delta(message, delta);
            return true;
        },
        [&] { return !req.is_connection_closed(); });
    log_summary(job.id, outcome, !message.tool_calls.empty());
    if (outcome.client_gone) return;
    if (outcome.result.finish == FinishReason::Error) fail(outcome.result.error);
    res.set_content(job.format()->body(message, outcome).dump(), "application/json");
}

bool CompletionHandlers::write_stream(const Job& job, const httplib::Request& req, httplib::DataSink& sink) const {
    const auto send = [&sink](std::string_view event) {
        return event.empty() || sink.write(event.data(), event.size());
    };
    const std::unique_ptr<ResponseFormat> format = job.format();
    if (!send(format->opening())) return false;

    GenerationOutcome outcome;
    try {
        TokenOutput output(service_.tokenizer(), chat::make_output_parser(job.dialect, job.prompt_tail), job.options.stop);
        outcome = run_generation(
            service_, job.request, output, [&](const chat::ParsedDelta& delta) { return send(format->delta(delta)); },
            [&] { return !req.is_connection_closed(); });
    } catch (const std::exception& e) {
        outcome.result.finish = FinishReason::Error;
        outcome.result.error = e.what();
    }
    log_summary(job.id, outcome, format->has_tool_calls());
    if (outcome.client_gone) return false;
    if (outcome.result.finish == FinishReason::Error) return send(format->error(outcome.result.error));
    return send(format->closing(outcome));
}

}  // namespace xs::server
