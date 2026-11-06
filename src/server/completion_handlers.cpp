#include "server/completion_handlers.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "chat/chat_template.h"
#include "server/generation_run.h"
#include "server/openai_format.h"
#include "server/openai_request.h"
#include "server/token_output.h"
#include "tokenizer/tokenizer.h"
#include "util/log.h"

namespace xs::server {

using nlohmann::json;

// Everything a response needs once the prompt is tokenized. Copyable: the stream provider keeps a copy.
struct CompletionHandlers::Job {
    bool chat = true;  // chat.completion objects vs text_completion objects
    ResponseMeta meta;
    GenerateRequest request;
    GenerationOptions options;
    std::string dialect;  // output parser; raw completions are always "plain"
};

namespace {

// The per-endpoint shape of a response: chat deltas with roles and tool calls, or plain text.
class ResponseShape {
public:
    explicit ResponseShape(ResponseMeta meta) : meta_(std::move(meta)) {}
    virtual ~ResponseShape() = default;
    virtual std::optional<json> opening_chunk() const = 0;
    virtual json delta_chunk(const chat::ParsedDelta& delta) = 0;
    virtual json closing_chunk(std::string_view finish_reason) const = 0;
    virtual json usage_chunk(const GenerateStats& stats) const = 0;
    virtual json full_response(const chat::ParsedDelta& output, std::string_view finish_reason,
                               const GenerateStats& stats) const = 0;

protected:
    ResponseMeta meta_;
};

class ChatShape final : public ResponseShape {
public:
    using ResponseShape::ResponseShape;
    std::optional<json> opening_chunk() const override {
        return chat_chunk_json(meta_, {{"role", "assistant"}, {"content", ""}});
    }
    json delta_chunk(const chat::ParsedDelta& delta) override {
        json chunk = chat_chunk_json(meta_, chat_delta_json(delta, tool_calls_sent_));
        tool_calls_sent_ += int(delta.tool_calls.size());
        return chunk;
    }
    json closing_chunk(std::string_view finish_reason) const override {
        return chat_chunk_json(meta_, json::object(), finish_reason);
    }
    json usage_chunk(const GenerateStats& stats) const override { return chat_usage_chunk_json(meta_, stats); }
    json full_response(const chat::ParsedDelta& output, std::string_view finish_reason,
                       const GenerateStats& stats) const override {
        return chat_completion_json(meta_, output, finish_reason, stats);
    }

private:
    int tool_calls_sent_ = 0;
};

class TextShape final : public ResponseShape {
public:
    using ResponseShape::ResponseShape;
    std::optional<json> opening_chunk() const override { return std::nullopt; }
    json delta_chunk(const chat::ParsedDelta& delta) override { return text_chunk_json(meta_, delta.content); }
    json closing_chunk(std::string_view finish_reason) const override {
        return text_chunk_json(meta_, "", finish_reason);
    }
    json usage_chunk(const GenerateStats& stats) const override { return text_usage_chunk_json(meta_, stats); }
    json full_response(const chat::ParsedDelta& output, std::string_view finish_reason,
                       const GenerateStats& stats) const override {
        return text_completion_json(meta_, output.content, finish_reason, stats);
    }
};

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

std::unique_ptr<ResponseShape> make_shape(bool chat, const ResponseMeta& meta) {
    if (chat) return std::make_unique<ChatShape>(meta);
    return std::make_unique<TextShape>(meta);
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
    log::info(id, ": prompt=", s.prompt_tokens, " cached=", s.cached_tokens, " generated=", s.generated_tokens,
              " prefill=", fixed1(s.prefill_tok_per_s()), " tok/s decode=", fixed1(s.decode_tok_per_s()),
              " tok/s finish=", finish);
}

}  // namespace

void CompletionHandlers::chat(const httplib::Request& req, httplib::Response& res) const {
    const ModelCard& card = service_.model();
    ChatRequest chat = parse_chat_request(parse_json_body(req.body), card.default_sampling);
    std::vector<int32_t> prompt;
    try {
        const std::string text = service_.chat_template().render(chat.messages, chat.tools,
                                                                 /*add_generation_prompt=*/true, chat.template_kwargs);
        prompt = service_.tokenizer().encode(text, /*parse_special=*/true);
    } catch (const std::exception& e) {
        throw InvalidRequest(str_cat("could not render the chat template: ", e.what()), "messages");
    }
    Job job{.chat = true,
            .meta = {random_id("chatcmpl-"), card.id, unix_seconds_now()},
            .request = make_generate_request(std::move(prompt), chat.options, card, "messages"),
            .options = std::move(chat.options),
            .dialect = card.dialect};
    respond(job, req, res);
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
    Job job{.chat = false,
            .meta = {random_id("cmpl-"), card.id, unix_seconds_now()},
            .request = make_generate_request(std::move(prompt), completion.options, card, "prompt"),
            .options = std::move(completion.options),
            .dialect = "plain"};
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
    TokenOutput output(service_.tokenizer(), chat::make_output_parser(job.dialect), job.options.stop);
    chat::ParsedDelta message;
    const GenerationOutcome outcome = run_generation(
        service_, job.request, output,
        [&](const chat::ParsedDelta& delta) {
            append_delta(message, delta);
            return true;
        },
        [&] { return !req.is_connection_closed(); });
    const bool has_tool_calls = !message.tool_calls.empty();
    log_summary(job.meta.id, outcome, has_tool_calls);
    if (outcome.client_gone) return;
    if (outcome.result.finish == FinishReason::Error) fail(outcome.result.error);
    const json body = make_shape(job.chat, job.meta)
                          ->full_response(message, outcome_finish_reason(outcome, has_tool_calls),
                                          outcome.result.stats);
    res.set_content(body.dump(), "application/json");
}

bool CompletionHandlers::write_stream(const Job& job, const httplib::Request& req, httplib::DataSink& sink) const {
    const auto send = [&sink](std::string_view event) { return sink.write(event.data(), event.size()); };
    const std::unique_ptr<ResponseShape> shape = make_shape(job.chat, job.meta);
    if (const auto opening = shape->opening_chunk(); opening.has_value() && !send(sse_event(*opening))) return false;

    GenerationOutcome outcome;
    bool has_tool_calls = false;
    try {
        TokenOutput output(service_.tokenizer(), chat::make_output_parser(job.dialect), job.options.stop);
        outcome = run_generation(
            service_, job.request, output,
            [&](const chat::ParsedDelta& delta) {
                has_tool_calls = has_tool_calls || !delta.tool_calls.empty();
                return send(sse_event(shape->delta_chunk(delta)));
            },
            [&] { return !req.is_connection_closed(); });
    } catch (const std::exception& e) {
        outcome.result.finish = FinishReason::Error;
        outcome.result.error = e.what();
    }
    log_summary(job.meta.id, outcome, has_tool_calls);
    if (outcome.client_gone) return false;

    bool ok = true;
    if (outcome.result.finish == FinishReason::Error) {
        ok = send(sse_event(error_body(outcome.result.error, "server_error")));
    } else {
        ok = send(sse_event(shape->closing_chunk(outcome_finish_reason(outcome, has_tool_calls))));
        if (ok && job.options.include_usage) ok = send(sse_event(shape->usage_chunk(outcome.result.stats)));
    }
    return ok && send(kSseDone);
}

}  // namespace xs::server
