#pragma once

// Anthropic's Messages API responses as a ResponseFormat: a message of content blocks (thinking, text, tool_use) with
// its stop reason and usage, or its stream of named events (message_start, content_block_start / _delta / _stop for
// each block, message_delta, message_stop).

#include <cstdint>
#include <memory>
#include <string>

#include "server/response_format.h"

namespace xs::server {

// `input_tokens`: the prompt's, which message_start reports before the generation runs.
std::unique_ptr<ResponseFormat> anthropic_format(std::string id, std::string model, int64_t input_tokens);

// The body of an error answered on these routes: {"type": "error", "error": {"type": ..., "message": ...}}.
nlohmann::json anthropic_error_body(std::string_view type, std::string_view message);

}  // namespace xs::server
