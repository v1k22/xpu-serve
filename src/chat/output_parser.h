#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace xs::chat {

struct ToolCall {
    std::string id;         // "call_<...>", unique within a response
    std::string name;
    std::string arguments;  // JSON object as text
};

// What one feed() produced. Text never contains the dialect's control markers.
struct ParsedDelta {
    std::string content;
    std::string reasoning;
    std::vector<ToolCall> tool_calls;  // complete calls closed by this delta
    bool empty() const { return content.empty() && reasoning.empty() && tool_calls.empty(); }
};

// Streaming parser that splits a model's raw output text into content, reasoning and tool calls. It must handle
// markers split across feed() calls.
class OutputParser {
public:
    virtual ~OutputParser() = default;
    virtual ParsedDelta feed(std::string_view text) = 0;
    // End of generation: returns anything held back (e.g. a partial marker that never completed is plain content).
    virtual ParsedDelta finish() = 0;
    // Inside reasoning (where a prompt left it, before any feed), the marker that ends it; empty otherwise.
    virtual std::string_view reasoning_end() const { return {}; }
};

// Dialects: "gemma4", "qwen3" (Qwen 3's JSON tool calls), "qwen3_5" (Qwen 3.5's XML ones), "inkling2" (messages of
// a kind each), "plain" (everything is content). Throws xs::Error naming the known dialects otherwise.
// prompt_tail: the end of the rendered prompt (see prompt_tail()); the parser starts in the state it leaves open, e.g.
// in reasoning when the prompt ends inside an opened thought channel (Gemma 4) or an open <think> (Qwen 3.5).
std::unique_ptr<OutputParser> make_output_parser(std::string_view dialect, std::string_view prompt_tail = {});

// How a prompt asks a model of `dialect` for its answer at once, without thinking: the chat template's kwargs, and
// the text the model writes after the generation prompt before its answer's text, which such a prompt ends with
// (Inkling 2's template stops before the model's message header: <|message_model|><|content_text|>). Throws as
// make_output_parser for an unknown dialect.
struct DirectAnswer {
    nlohmann::json kwargs;
    std::string opening;
};
DirectAnswer direct_answer(std::string_view dialect);

// Where an answer held to a format (response_format) starts in the output of a model of `dialect` after a prompt
// ending in `prompt_tail`: when the prompt leaves the model reasoning (Qwen 3.5 with thinking on), once it has written
// `reasoning_end`, the marker that closes the thought; then after `opening`, what the dialect writes before an
// answer's text (direct_answer's: Inkling 2's message header). Throws as make_output_parser for an unknown dialect.
struct AnswerStart {
    std::string reasoning_end;  // empty: the prompt leaves the model writing its answer
    std::string opening;
};
AnswerStart answer_start(std::string_view dialect, std::string_view prompt_tail);

// The end of a rendered prompt that make_output_parser needs; keeping only this avoids copying whole prompts.
inline std::string_view prompt_tail(std::string_view rendered_prompt) {
    constexpr size_t kTailBytes = 256;
    return rendered_prompt.substr(rendered_prompt.size() > kTailBytes ? rendered_prompt.size() - kTailBytes : 0);
}

}  // namespace xs::chat
