#pragma once

// How an endpoint writes one generation (CompletionHandlers runs it): as one JSON body at the end, or as server-sent
// events while it runs. OpenAI's chat and text completions (openai_format.h) and Anthropic's messages
// (anthropic_format.h) are each one.

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat/output_parser.h"
#include "server/generation_run.h"

namespace xs::server {

// One choice of a response: every piece of it appended, and how its generation ended.
struct ChoiceOutput {
    OutputPiece output;
    GenerationOutcome outcome;
};

class ResponseFormat {
public:
    virtual ~ResponseFormat() = default;
    // Events before any piece, for a response of `choices` choices (OpenAI's n); "" where there are none.
    virtual std::string opening(int choices) = 0;
    // Events after a piece of choice `choice`.
    virtual std::string delta(int choice, const OutputPiece& piece) = 0;
    // Events after choice `choice`'s last piece (its finish).
    virtual std::string finished(int choice, const GenerationOutcome& outcome) = 0;
    // Events after every choice finished (the usage, the stream's end marker), the choices' outcomes in order.
    virtual std::string closing(std::span<const GenerationOutcome> outcomes) = 0;
    // The event a failure is once the stream has started.
    virtual std::string error(std::string_view message) = 0;
    // The whole response.
    virtual nlohmann::json body(std::span<const ChoiceOutput> choices) = 0;
    // Whether choice `choice` holds tool calls so far (a finish reason says so).
    bool has_tool_calls(int choice) const {
        return choice < static_cast<int>(tool_calls_.size()) && tool_calls_[static_cast<size_t>(choice)] > 0;
    }

protected:
    // The tool calls choice `choice` sent in deltas so far.
    int& tool_calls(int choice) {
        if (choice >= static_cast<int>(tool_calls_.size())) tool_calls_.resize(static_cast<size_t>(choice) + 1, 0);
        return tool_calls_[static_cast<size_t>(choice)];
    }

private:
    std::vector<int> tool_calls_;
};

}  // namespace xs::server
