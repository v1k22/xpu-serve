#pragma once

// OpenAI response bodies and stream chunks. Pure functions of their inputs, so every shape is unit-testable.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "chat/output_parser.h"
#include "engine/service.h"
#include "engine/types.h"
#include "server/logprobs.h"
#include "server/response_format.h"

namespace xs::server {

// What every object of one response shares.
struct ResponseMeta {
    std::string id;     // "chatcmpl-..." or "cmpl-..."
    std::string model;
    int64_t created = 0;  // unix seconds
};

int64_t unix_seconds_now();

inline constexpr std::string_view kSseDone = "data: [DONE]\n\n";

// One server-sent event carrying `payload`.
std::string sse_event(const nlohmann::json& payload);

// {"error": {...}}; an empty param or code is written as null.
nlohmann::json error_body(std::string_view message, std::string_view type, std::string_view param = {},
                          std::string_view code = {});

// OpenAI finish_reason for a finished generation. A stop-string cancel reads as "stop"; tool calls as "tool_calls".
std::string_view finish_reason_text(FinishReason finish, bool stop_string_hit, bool has_tool_calls);

nlohmann::json usage_json(const GenerateStats& stats);
// Non-standard: prefill / decode speed, so clients can see performance without parsing logs.
nlohmann::json timings_json(const GenerateStats& stats);

// `logprobs`: the choice's log-probabilities (chat_logprobs_json, text_logprobs_json), null when not asked for.
nlohmann::json chat_completion_json(const ResponseMeta& meta, const chat::ParsedDelta& message,
                                    std::string_view finish_reason, const GenerateStats& stats,
                                    nlohmann::json logprobs = nullptr);
// The delta object of a chat chunk; tool calls are numbered from `first_tool_index`.
nlohmann::json chat_delta_json(const chat::ParsedDelta& delta, int first_tool_index);
nlohmann::json chat_chunk_json(const ResponseMeta& meta, nlohmann::json delta,
                               std::optional<std::string_view> finish_reason = std::nullopt,
                               nlohmann::json logprobs = nullptr);
// Final chunk of a stream with stream_options.include_usage: no choices, only usage.
nlohmann::json chat_usage_chunk_json(const ResponseMeta& meta, const GenerateStats& stats);

nlohmann::json text_completion_json(const ResponseMeta& meta, std::string_view text, std::string_view finish_reason,
                                    const GenerateStats& stats, nlohmann::json logprobs = nullptr);
nlohmann::json text_chunk_json(const ResponseMeta& meta, std::string_view text,
                               std::optional<std::string_view> finish_reason = std::nullopt,
                               nlohmann::json logprobs = nullptr);

// A chat choice's log-probabilities: {"content": [{"token", "logprob", "bytes", "top_logprobs": [{"token", "logprob",
// "bytes"}]}]}, each token's most likely alternatives in order.
nlohmann::json chat_logprobs_json(std::span<const LogprobEntry> entries);
// A text completion choice's: {"tokens", "token_logprobs", "top_logprobs": [{text: logprob}], "text_offset"}, OpenAI's
// legacy shape. Each position's top_logprobs holds the token and its alternatives (one entry a text, the most likely
// one's), text_offset counts code points from `offset`, the text before these tokens, which it moves past them.
nlohmann::json text_logprobs_json(std::span<const LogprobEntry> entries, int64_t& offset);
nlohmann::json text_usage_chunk_json(const ResponseMeta& meta, const GenerateStats& stats);

// GET /v1/models for the one model served; context_length 0 is left out.
nlohmann::json model_list_json(std::string_view id, int64_t context_length, int64_t created);
nlohmann::json model_list_json(const ModelCard& card, int64_t created);

struct ChatFormatOptions {
    bool include_usage = false;  // a usage chunk before [DONE]
    bool logprobs = false;       // the request asked for log-probabilities, which the pieces carry
    bool named_call = false;     // tool_choice named a function: its call ends with finish_reason "stop" (OpenAI's)
};

// OpenAI's chat completions and text completions as a ResponseFormat: chunks, a usage chunk when asked for, then
// [DONE].
std::unique_ptr<ResponseFormat> openai_chat_format(ResponseMeta meta, ChatFormatOptions options);
std::unique_ptr<ResponseFormat> openai_text_format(ResponseMeta meta, ChatFormatOptions options);

}  // namespace xs::server
