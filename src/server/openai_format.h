#pragma once

// OpenAI response bodies and stream chunks. Pure functions of their inputs, so every shape is unit-testable.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "chat/output_parser.h"
#include "engine/service.h"
#include "engine/types.h"

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

nlohmann::json chat_completion_json(const ResponseMeta& meta, const chat::ParsedDelta& message,
                                    std::string_view finish_reason, const GenerateStats& stats);
// The delta object of a chat chunk; tool calls are numbered from `first_tool_index`.
nlohmann::json chat_delta_json(const chat::ParsedDelta& delta, int first_tool_index);
nlohmann::json chat_chunk_json(const ResponseMeta& meta, nlohmann::json delta,
                               std::optional<std::string_view> finish_reason = std::nullopt);
// Final chunk of a stream with stream_options.include_usage: no choices, only usage.
nlohmann::json chat_usage_chunk_json(const ResponseMeta& meta, const GenerateStats& stats);

nlohmann::json text_completion_json(const ResponseMeta& meta, std::string_view text, std::string_view finish_reason,
                                    const GenerateStats& stats);
nlohmann::json text_chunk_json(const ResponseMeta& meta, std::string_view text,
                               std::optional<std::string_view> finish_reason = std::nullopt);
nlohmann::json text_usage_chunk_json(const ResponseMeta& meta, const GenerateStats& stats);

// GET /v1/models for the one model served; context_length 0 is left out.
nlohmann::json model_list_json(std::string_view id, int64_t context_length, int64_t created);
nlohmann::json model_list_json(const ModelCard& card, int64_t created);

}  // namespace xs::server
