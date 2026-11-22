#include "server/anthropic_format.h"

#include <string_view>

#include "util/error.h"

namespace xs::server {
namespace {

using json = nlohmann::json;

std::string event(std::string_view name, const json& data) {
    return str_cat("event: ", name, "\ndata: ", data.dump(), "\n\n");
}

// stop_reason, and the stop sequence that ended the message (null for the other reasons).
std::pair<std::string_view, json> stop_of(const GenerationOutcome& o, bool tool_calls) {
    if (o.stop_string_hit) return {"stop_sequence", o.stop_string};
    if (tool_calls) return {"tool_use", nullptr};
    if (o.result.finish == FinishReason::Length) return {"max_tokens", nullptr};
    return {"end_turn", nullptr};
}

json tool_use_block(const chat::ToolCall& call) {
    json input = json::parse(call.arguments, nullptr, /*allow_exceptions=*/false);
    if (!input.is_object()) input = json::object();
    return json{{"type", "tool_use"}, {"id", call.id}, {"name", call.name}, {"input", std::move(input)}};
}

json usage(const GenerateStats& s) {
    return json{{"input_tokens", s.prompt_tokens - s.cached_tokens},
                {"cache_creation_input_tokens", 0},
                {"cache_read_input_tokens", s.cached_tokens},
                {"output_tokens", s.generated_tokens}};
}

class AnthropicFormat final : public ResponseFormat {
public:
    AnthropicFormat(std::string id, std::string model, int64_t input_tokens)
        : id_(std::move(id)), model_(std::move(model)), input_tokens_(input_tokens) {}

    std::string opening() override {
        const json message{{"id", id_},
                           {"type", "message"},
                           {"role", "assistant"},
                           {"model", model_},
                           {"content", json::array()},
                           {"stop_reason", nullptr},
                           {"stop_sequence", nullptr},
                           {"usage", {{"input_tokens", input_tokens_}, {"output_tokens", 0}}}};
        return event("message_start", {{"type", "message_start"}, {"message", message}}) +
               event("ping", {{"type", "ping"}});
    }

    std::string delta(const chat::ParsedDelta& d) override {
        std::string out;
        if (!d.reasoning.empty()) {
            out += open(Block::Thinking, {{"type", "thinking"}, {"thinking", ""}, {"signature", ""}});
            out += block_delta({{"type", "thinking_delta"}, {"thinking", d.reasoning}});
        }
        if (!d.content.empty()) {
            out += open(Block::Text, {{"type", "text"}, {"text", ""}});
            out += block_delta({{"type", "text_delta"}, {"text", d.content}});
        }
        for (const chat::ToolCall& call : d.tool_calls) {
            out += close();
            json start = tool_use_block(call);
            start["input"] = json::object();
            out += open(Block::ToolUse, start);
            out += block_delta({{"type", "input_json_delta"}, {"partial_json", call.arguments}});
            out += close();
            ++tool_calls_;
        }
        return out;
    }

    std::string closing(const GenerationOutcome& o) override {
        const auto [reason, sequence] = stop_of(o, has_tool_calls());
        return close() +
               event("message_delta", {{"type", "message_delta"},
                                       {"delta", {{"stop_reason", reason}, {"stop_sequence", sequence}}},
                                       {"usage", usage(o.result.stats)}}) +
               event("message_stop", {{"type", "message_stop"}});
    }

    std::string error(std::string_view message) override {
        return close() + event("error", anthropic_error_body("api_error", message));
    }

    json body(const chat::ParsedDelta& output, const GenerationOutcome& o) override {
        json content = json::array();
        if (!output.reasoning.empty()) {
            content.push_back({{"type", "thinking"}, {"thinking", output.reasoning}, {"signature", ""}});
        }
        if (!output.content.empty()) content.push_back({{"type", "text"}, {"text", output.content}});
        for (const chat::ToolCall& call : output.tool_calls) content.push_back(tool_use_block(call));
        const auto [reason, sequence] = stop_of(o, !output.tool_calls.empty());
        return json{{"id", id_},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"model", model_},
                    {"content", std::move(content)},
                    {"stop_reason", reason},
                    {"stop_sequence", sequence},
                    {"usage", usage(o.result.stats)}};
    }

private:
    enum class Block { None, Thinking, Text, ToolUse };

    // Starts a block of `kind` unless one is open: the open one of another kind closes first.
    std::string open(Block kind, const json& start) {
        if (open_ == kind && kind != Block::ToolUse) return {};
        std::string out = close();
        open_ = kind;
        return out + event("content_block_start",
                           {{"type", "content_block_start"}, {"index", index_}, {"content_block", start}});
    }
    std::string block_delta(const json& delta) {
        return event("content_block_delta", {{"type", "content_block_delta"}, {"index", index_}, {"delta", delta}});
    }
    std::string close() {
        if (open_ == Block::None) return {};
        open_ = Block::None;
        return event("content_block_stop", {{"type", "content_block_stop"}, {"index", index_++}});
    }

    std::string id_, model_;
    int64_t input_tokens_ = 0;
    Block open_ = Block::None;
    int index_ = 0;
};

}  // namespace

std::unique_ptr<ResponseFormat> anthropic_format(std::string id, std::string model, int64_t input_tokens) {
    return std::make_unique<AnthropicFormat>(std::move(id), std::move(model), input_tokens);
}

json anthropic_error_body(std::string_view type, std::string_view message) {
    return json{{"type", "error"}, {"error", {{"type", type}, {"message", message}}}};
}

}  // namespace xs::server
