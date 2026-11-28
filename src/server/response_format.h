#pragma once

// How an endpoint writes one generation (CompletionHandlers runs it): as one JSON body at the end, or as server-sent
// events while it runs. OpenAI's chat and text completions (openai_format.h) and Anthropic's messages
// (anthropic_format.h) are each one.

#include <memory>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "chat/output_parser.h"
#include "server/generation_run.h"

namespace xs::server {

class ResponseFormat {
public:
    virtual ~ResponseFormat() = default;
    // Events before the first piece, after each piece, and after the last one (its finish, its usage, the stream's end
    // marker); "" where there are none.
    virtual std::string opening() = 0;
    virtual std::string delta(const OutputPiece& piece) = 0;
    virtual std::string closing(const GenerationOutcome& outcome) = 0;
    // The event a failure is once the stream has started.
    virtual std::string error(std::string_view message) = 0;
    // The whole response, `output` being every piece appended.
    virtual nlohmann::json body(const OutputPiece& output, const GenerationOutcome& outcome) = 0;
    // Whether the response holds tool calls so far (a finish reason says so).
    bool has_tool_calls() const { return tool_calls_ > 0; }

protected:
    int tool_calls_ = 0;  // sent in deltas
};

}  // namespace xs::server
