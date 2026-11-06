#pragma once

// Parsing and validation of OpenAI request bodies. Unknown fields are ignored; a field set to null counts as absent.

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/types.h"
#include "util/error.h"

namespace xs::server {

// A request the client got wrong: answered with 400, naming the offending field in `param` (empty when none).
class InvalidRequest : public Error {
public:
    InvalidRequest(const std::string& message, std::string param) : Error(message), param_(std::move(param)) {}
    const std::string& param() const { return param_; }

private:
    std::string param_;
};

// Fields shared by /v1/chat/completions and /v1/completions.
struct GenerationOptions {
    SamplingParams sampling;
    std::optional<int> max_tokens;  // nullopt = up to the end of the context
    std::vector<std::string> stop;
    bool stream = false;
    bool include_usage = false;  // stream_options.include_usage
};

struct ChatRequest {
    nlohmann::json messages;         // validated: non-empty array of objects with a string "role"
    nlohmann::json tools;            // array, or null when absent, empty or tool_choice is "none"
    nlohmann::json template_kwargs;  // chat_template_kwargs, always an object
    GenerationOptions options;
};

struct CompletionRequest {
    std::string prompt;
    GenerationOptions options;
};

// Parses a request body that must be a JSON object.
nlohmann::json parse_json_body(std::string_view body);

// Absent sampling fields take their value from `defaults` (the model's generation_config).
ChatRequest parse_chat_request(const nlohmann::json& body, const SamplingParams& defaults);
CompletionRequest parse_completion_request(const nlohmann::json& body, const SamplingParams& defaults);

}  // namespace xs::server
