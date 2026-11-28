#include "server/openai_format.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace xs::server {
namespace {

using nlohmann::json;

json null_if_empty(std::string_view s) { return s.empty() ? json(nullptr) : json(s); }

json finish_json(std::optional<std::string_view> finish_reason) {
    return finish_reason.has_value() ? json(*finish_reason) : json(nullptr);
}

json envelope(const ResponseMeta& meta, std::string_view object) {
    return {{"id", meta.id}, {"object", object}, {"created", meta.created}, {"model", meta.model}};
}

json tool_call_json(const chat::ToolCall& call) {
    return {{"id", call.id}, {"type", "function"}, {"function", {{"name", call.name}, {"arguments", call.arguments}}}};
}

json token_json(const LogprobToken& t) {
    json bytes = json::array();
    for (const char c : t.bytes) bytes.push_back(static_cast<unsigned char>(c));
    return {{"token", t.text}, {"logprob", logprob_value(t.logprob)}, {"bytes", std::move(bytes)}};
}

// Code points of UTF-8 text: the characters a client's string counts.
int64_t code_points(std::string_view text) {
    return std::count_if(text.begin(), text.end(), [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; });
}

}  // namespace

json chat_logprobs_json(std::span<const LogprobEntry> entries) {
    json content = json::array();
    for (const LogprobEntry& e : entries) {
        json entry = token_json(e.token);
        json top = json::array();
        for (const LogprobToken& t : e.top) top.push_back(token_json(t));
        entry["top_logprobs"] = std::move(top);
        content.push_back(std::move(entry));
    }
    return {{"content", std::move(content)}};
}

json text_logprobs_json(std::span<const LogprobEntry> entries, int64_t& offset) {
    json tokens = json::array(), token_logprobs = json::array(), top_logprobs = json::array(), offsets = json::array();
    for (const LogprobEntry& e : entries) {
        tokens.push_back(e.token.text);
        token_logprobs.push_back(logprob_value(e.token.logprob));
        json top = json::object();
        top[e.token.text] = logprob_value(e.token.logprob);
        for (const LogprobToken& t : e.top) {
            if (!top.contains(t.text) || top[t.text].get<double>() < logprob_value(t.logprob)) {
                top[t.text] = logprob_value(t.logprob);
            }
        }
        top_logprobs.push_back(std::move(top));
        offsets.push_back(offset);
        offset += code_points(e.token.text);
    }
    return {{"tokens", std::move(tokens)},
            {"token_logprobs", std::move(token_logprobs)},
            {"top_logprobs", std::move(top_logprobs)},
            {"text_offset", std::move(offsets)}};
}

int64_t unix_seconds_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string sse_event(const json& payload) { return "data: " + payload.dump() + "\n\n"; }

json error_body(std::string_view message, std::string_view type, std::string_view param, std::string_view code) {
    return {{"error",
             {{"message", message}, {"type", type}, {"param", null_if_empty(param)}, {"code", null_if_empty(code)}}}};
}

std::string_view finish_reason_text(FinishReason finish, bool stop_string_hit, bool has_tool_calls) {
    if (finish == FinishReason::Length && !stop_string_hit) return "length";
    return has_tool_calls ? "tool_calls" : "stop";
}

json usage_json(const GenerateStats& stats) {
    return {{"prompt_tokens", stats.prompt_tokens},
            {"completion_tokens", stats.generated_tokens},
            {"total_tokens", stats.prompt_tokens + stats.generated_tokens},
            {"prompt_tokens_details", {{"cached_tokens", stats.cached_tokens}}}};
}

json timings_json(const GenerateStats& stats) {
    json t = {{"prefill_ms", stats.prefill_ms},
              {"prefill_tokens_per_second", stats.prefill_tok_per_s()},
              {"decode_ms", stats.decode_ms},
              {"decode_tokens_per_second", stats.decode_tok_per_s()}};
    if (stats.spec_rounds > 0) {
        t["spec_rounds"] = stats.spec_rounds;
        t["spec_drafted_tokens"] = stats.spec_drafted;
        t["spec_accepted_tokens"] = stats.spec_accepted;
    }
    if (stats.constrained_rows > 0) {  // response_format: the rows its grammar masked, and what that took
        t["constrained_rows"] = stats.constrained_rows;
        t["constraint_ms"] = stats.constraint_ms;
    }
    return t;
}

json chat_completion_json(const ResponseMeta& meta, const chat::ParsedDelta& message, std::string_view finish_reason,
                          const GenerateStats& stats, json logprobs) {
    json msg = {{"role", "assistant"}, {"content", message.content}};
    if (!message.reasoning.empty()) msg["reasoning_content"] = message.reasoning;
    if (!message.tool_calls.empty()) {
        json calls = json::array();
        for (const chat::ToolCall& call : message.tool_calls) calls.push_back(tool_call_json(call));
        msg["tool_calls"] = std::move(calls);
    }
    json out = envelope(meta, "chat.completion");
    out["choices"] = json::array({{{"index", 0},
                                   {"message", std::move(msg)},
                                   {"logprobs", std::move(logprobs)},
                                   {"finish_reason", finish_reason}}});
    out["usage"] = usage_json(stats);
    out["timings"] = timings_json(stats);
    return out;
}

json chat_delta_json(const chat::ParsedDelta& delta, int first_tool_index) {
    json out = json::object();
    if (!delta.reasoning.empty()) out["reasoning_content"] = delta.reasoning;
    if (!delta.content.empty()) out["content"] = delta.content;
    if (!delta.tool_calls.empty()) {
        json calls = json::array();
        int index = first_tool_index;
        for (const chat::ToolCall& call : delta.tool_calls) {
            json c = tool_call_json(call);
            c["index"] = index++;
            calls.push_back(std::move(c));
        }
        out["tool_calls"] = std::move(calls);
    }
    return out;
}

json chat_chunk_json(const ResponseMeta& meta, json delta, std::optional<std::string_view> finish_reason,
                     json logprobs) {
    json out = envelope(meta, "chat.completion.chunk");
    out["choices"] = json::array({{{"index", 0},
                                   {"delta", std::move(delta)},
                                   {"logprobs", std::move(logprobs)},
                                   {"finish_reason", finish_json(finish_reason)}}});
    return out;
}

json chat_usage_chunk_json(const ResponseMeta& meta, const GenerateStats& stats) {
    json out = envelope(meta, "chat.completion.chunk");
    out["choices"] = json::array();
    out["usage"] = usage_json(stats);
    out["timings"] = timings_json(stats);
    return out;
}

json text_completion_json(const ResponseMeta& meta, std::string_view text, std::string_view finish_reason,
                          const GenerateStats& stats, json logprobs) {
    json out = text_chunk_json(meta, text, finish_reason, std::move(logprobs));
    out["usage"] = usage_json(stats);
    out["timings"] = timings_json(stats);
    return out;
}

json text_chunk_json(const ResponseMeta& meta, std::string_view text, std::optional<std::string_view> finish_reason,
                     json logprobs) {
    json out = envelope(meta, "text_completion");
    out["choices"] = json::array({{{"index", 0},
                                   {"text", text},
                                   {"logprobs", std::move(logprobs)},
                                   {"finish_reason", finish_json(finish_reason)}}});
    return out;
}

json text_usage_chunk_json(const ResponseMeta& meta, const GenerateStats& stats) {
    json out = envelope(meta, "text_completion");
    out["choices"] = json::array();
    out["usage"] = usage_json(stats);
    out["timings"] = timings_json(stats);
    return out;
}

json model_list_json(std::string_view id, int64_t context_length, int64_t created) {
    json model = {{"id", id}, {"object", "model"}, {"created", created}, {"owned_by", "xpu-serve"}};
    if (context_length > 0) model["context_length"] = context_length;
    return {{"object", "list"}, {"data", json::array({std::move(model)})}};
}

json model_list_json(const ModelCard& card, int64_t created) {
    return model_list_json(card.id, card.context_length, created);
}

namespace {

std::string_view finish_of(const GenerationOutcome& o, bool has_tool_calls) {
    return finish_reason_text(o.result.finish, o.stop_string_hit, has_tool_calls);
}

class ChatFormat final : public ResponseFormat {
public:
    ChatFormat(ResponseMeta meta, ChatFormatOptions options)
        : meta_(std::move(meta)), include_usage_(options.include_usage), logprobs_(options.logprobs),
          named_call_(options.named_call) {}
    std::string opening() override {
        return sse_event(chat_chunk_json(meta_, {{"role", "assistant"}, {"content", ""}}));
    }
    std::string delta(const OutputPiece& piece) override {
        const chat::ParsedDelta& d = piece.delta;
        json logprobs = logprobs_ && !piece.logprobs.empty() ? chat_logprobs_json(piece.logprobs) : json(nullptr);
        std::string event =
            sse_event(chat_chunk_json(meta_, chat_delta_json(d, tool_calls_), std::nullopt, std::move(logprobs)));
        tool_calls_ += static_cast<int>(d.tool_calls.size());
        return event;
    }
    std::string closing(const GenerationOutcome& o) override {
        std::string out =
            sse_event(chat_chunk_json(meta_, json::object(), finish_of(o, has_tool_calls() && !named_call_)));
        if (include_usage_) out += sse_event(chat_usage_chunk_json(meta_, o.result.stats));
        return out + std::string(kSseDone);
    }
    std::string error(std::string_view message) override {
        return sse_event(error_body(message, "server_error")) + std::string(kSseDone);
    }
    json body(const OutputPiece& output, const GenerationOutcome& o) override {
        return chat_completion_json(meta_, output.delta,
                                    finish_of(o, !output.delta.tool_calls.empty() && !named_call_), o.result.stats,
                                    logprobs_ ? chat_logprobs_json(output.logprobs) : json(nullptr));
    }

private:
    ResponseMeta meta_;
    bool include_usage_;
    bool logprobs_;
    bool named_call_;
};

class TextFormat final : public ResponseFormat {
public:
    TextFormat(ResponseMeta meta, ChatFormatOptions options)
        : meta_(std::move(meta)), include_usage_(options.include_usage), logprobs_(options.logprobs) {}
    std::string opening() override { return {}; }
    std::string delta(const OutputPiece& piece) override {
        json logprobs = logprobs_ ? text_logprobs_json(piece.logprobs, offset_) : json(nullptr);
        return sse_event(text_chunk_json(meta_, piece.delta.content, std::nullopt, std::move(logprobs)));
    }
    std::string closing(const GenerationOutcome& o) override {
        std::string out = sse_event(text_chunk_json(meta_, "", finish_of(o, false)));
        if (include_usage_) out += sse_event(text_usage_chunk_json(meta_, o.result.stats));
        return out + std::string(kSseDone);
    }
    std::string error(std::string_view message) override {
        return sse_event(error_body(message, "server_error")) + std::string(kSseDone);
    }
    json body(const OutputPiece& output, const GenerationOutcome& o) override {
        int64_t offset = 0;
        return text_completion_json(meta_, output.delta.content, finish_of(o, false), o.result.stats,
                                    logprobs_ ? text_logprobs_json(output.logprobs, offset) : json(nullptr));
    }

private:
    ResponseMeta meta_;
    bool include_usage_;
    bool logprobs_;
    int64_t offset_ = 0;  // code points of the text streamed so far (text_offset)
};

}  // namespace

std::unique_ptr<ResponseFormat> openai_chat_format(ResponseMeta meta, ChatFormatOptions options) {
    return std::make_unique<ChatFormat>(std::move(meta), options);
}

std::unique_ptr<ResponseFormat> openai_text_format(ResponseMeta meta, ChatFormatOptions options) {
    return std::make_unique<TextFormat>(std::move(meta), options);
}

}  // namespace xs::server
