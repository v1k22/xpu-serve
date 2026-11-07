#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

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
};

// Dialects: "gemma4", "qwen3_5", "plain" (everything is content). Throws xs::Error naming the known dialects otherwise.
// prompt_tail: the end of the rendered prompt (see prompt_tail()); the parser starts in the state it leaves open, e.g.
// in reasoning when the prompt ends inside an opened thought channel (Gemma 4) or an open <think> (Qwen 3.5).
std::unique_ptr<OutputParser> make_output_parser(std::string_view dialect, std::string_view prompt_tail = {});

// The end of a rendered prompt that make_output_parser needs; keeping only this avoids copying whole prompts.
inline std::string_view prompt_tail(std::string_view rendered_prompt) {
    constexpr size_t kTailBytes = 256;
    return rendered_prompt.substr(rendered_prompt.size() > kTailBytes ? rendered_prompt.size() - kTailBytes : 0);
}

}  // namespace xs::chat
