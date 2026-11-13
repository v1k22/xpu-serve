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

}  // namespace xs::server
