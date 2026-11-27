#include "server/openai_format.h"

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

}  // namespace

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
                          const GenerateStats& stats) {
    json msg = {{"role", "assistant"}, {"content", message.content}};
    if (!message.reasoning.empty()) msg["reasoning_content"] = message.reasoning;
    if (!message.tool_calls.empty()) {
        json calls = json::array();
        for (const chat::ToolCall& call : message.tool_calls) calls.push_back(tool_call_json(call));
        msg["tool_calls"] = std::move(calls);
    }
    json out = envelope(meta, "chat.completion");
    out["choices"] = json::array({{{"index", 0}, {"message", std::move(msg)}, {"finish_reason", finish_reason}}});
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

json chat_chunk_json(const ResponseMeta& meta, json delta, std::optional<std::string_view> finish_reason) {
    json out = envelope(meta, "chat.completion.chunk");
    out["choices"] =
        json::array({{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", finish_json(finish_reason)}}});
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
                          const GenerateStats& stats) {
    json out = text_chunk_json(meta, text, finish_reason);
    out["usage"] = usage_json(stats);
    out["timings"] = timings_json(stats);
    return out;
}

json text_chunk_json(const ResponseMeta& meta, std::string_view text, std::optional<std::string_view> finish_reason) {
    json out = envelope(meta, "text_completion");
    out["choices"] = json::array(
        {{{"index", 0}, {"text", text}, {"logprobs", nullptr}, {"finish_reason", finish_json(finish_reason)}}});
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
    ChatFormat(ResponseMeta meta, bool include_usage) : meta_(std::move(meta)), include_usage_(include_usage) {}
    std::string opening() override {
        return sse_event(chat_chunk_json(meta_, {{"role", "assistant"}, {"content", ""}}));
    }
    std::string delta(const chat::ParsedDelta& d) override {
        std::string event = sse_event(chat_chunk_json(meta_, chat_delta_json(d, tool_calls_)));
        tool_calls_ += static_cast<int>(d.tool_calls.size());
        return event;
    }
    std::string closing(const GenerationOutcome& o) override {
        std::string out = sse_event(chat_chunk_json(meta_, json::object(), finish_of(o, has_tool_calls())));
        if (include_usage_) out += sse_event(chat_usage_chunk_json(meta_, o.result.stats));
        return out + std::string(kSseDone);
    }
    std::string error(std::string_view message) override {
        return sse_event(error_body(message, "server_error")) + std::string(kSseDone);
    }
    json body(const chat::ParsedDelta& output, const GenerationOutcome& o) override {
        return chat_completion_json(meta_, output, finish_of(o, !output.tool_calls.empty()), o.result.stats);
    }

private:
    ResponseMeta meta_;
    bool include_usage_;
};

class TextFormat final : public ResponseFormat {
public:
    TextFormat(ResponseMeta meta, bool include_usage) : meta_(std::move(meta)), include_usage_(include_usage) {}
    std::string opening() override { return {}; }
    std::string delta(const chat::ParsedDelta& d) override { return sse_event(text_chunk_json(meta_, d.content)); }
    std::string closing(const GenerationOutcome& o) override {
        std::string out = sse_event(text_chunk_json(meta_, "", finish_of(o, false)));
        if (include_usage_) out += sse_event(text_usage_chunk_json(meta_, o.result.stats));
        return out + std::string(kSseDone);
    }
    std::string error(std::string_view message) override {
        return sse_event(error_body(message, "server_error")) + std::string(kSseDone);
    }
    json body(const chat::ParsedDelta& output, const GenerationOutcome& o) override {
        return text_completion_json(meta_, output.content, finish_of(o, false), o.result.stats);
    }

private:
    ResponseMeta meta_;
    bool include_usage_;
};

}  // namespace

std::unique_ptr<ResponseFormat> openai_chat_format(ResponseMeta meta, bool include_usage) {
    return std::make_unique<ChatFormat>(std::move(meta), include_usage);
}

std::unique_ptr<ResponseFormat> openai_text_format(ResponseMeta meta, bool include_usage) {
    return std::make_unique<TextFormat>(std::move(meta), include_usage);
}

}  // namespace xs::server
