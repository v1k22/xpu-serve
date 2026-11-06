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

// Dialects: "gemma4", "plain" (everything is content). Throws xs::Error naming the known dialects otherwise.
std::unique_ptr<OutputParser> make_output_parser(std::string_view dialect);

}  // namespace xs::chat
