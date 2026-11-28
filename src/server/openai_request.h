#pragma once

// Parsing and validation of OpenAI request bodies. Unknown fields are ignored; a field set to null counts as absent.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/types.h"
#include "util/error.h"

namespace xs::grammar {
class Grammar;
}

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
    // response_format json_object or json_schema: the grammar the answer is held to; null for free text.
    std::shared_ptr<const grammar::Grammar> format;
    // Each generated token's log-probability with this many alternatives beside it (chat: logprobs and top_logprobs;
    // text completions: logprobs); -1: none.
    int logprobs = -1;
    // Choices to generate for the prompt (n), each sampled on its own; with a seed, choice i draws with seed + i.
    int choices = 1;
};

// The most choices a request may ask for (n).
inline constexpr int kMaxChoices = 128;

// The most alternatives a token's log-probability is given with (vLLM's default --max-logprobs).
inline constexpr int kMaxLogprobs = 20;

// A media part of a chat message: an image from its data: URL, an audio clip from input_audio's base64 (the bytes
// of the file, not yet decoded).
struct MediaPart {
    std::string param;  // where it came from, e.g. "messages[1].content[0]", for error messages
    Modality modality = Modality::Image;
    std::vector<uint8_t> bytes;
};

// Request JSON keeps the client's key order: chat templates can print it (tool schemas, tool-call arguments).
using RequestJson = nlohmann::ordered_json;

// tool_choice as a request resolves it, vLLM's way: with tools, "auto" unless it says otherwise; without them, none.
struct ToolChoice {
    enum class Mode { none, automatic, required, named };
    Mode mode = Mode::none;
    std::string name;  // named: the function the reply calls
};

struct ChatRequest {
    RequestJson messages;         // validated: non-empty array of objects with a string "role"; media parts are
                                  // reduced to {"type": "image"} or {"type": "audio"}, their bytes are in `media`
    RequestJson tools;            // array, or null when absent, empty or tool_choice is "none"; each tool an object
                                  // whose "function" has a string "name"
    ToolChoice tool_choice;
    bool parallel_tool_calls = true;  // false: one call at most
    RequestJson template_kwargs;  // chat_template_kwargs, always an object
    std::vector<MediaPart> media;    // in message order, then part order
    GenerationOptions options;
};

struct CompletionRequest {
    std::string prompt;
    GenerationOptions options;
};

// The sampling fields of a body (temperature, top_p, top_k, min_p, the penalties, seed); absent ones from `defaults`.
SamplingParams parse_sampling(const RequestJson& body, const SamplingParams& defaults);

// Parses a request body that must be a JSON object.
RequestJson parse_json_body(std::string_view body);

// Absent sampling fields take their value from `defaults` (the model's generation_config).
ChatRequest parse_chat_request(const RequestJson& body, const SamplingParams& defaults);
CompletionRequest parse_completion_request(const RequestJson& body, const SamplingParams& defaults);

}  // namespace xs::server
