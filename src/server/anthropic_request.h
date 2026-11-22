#pragma once

// Anthropic's Messages API requests (POST /v1/messages, /v1/messages/count_tokens), read as the chat request a chat
// template renders: the system prompt as a system message, content blocks as OpenAI's content parts, an assistant's
// thinking as its reasoning_content and its tool_use blocks as tool_calls, a tool_result as a tool message, tools as
// functions. Unknown fields are ignored; a block or tool this server cannot take is refused by name.

#include <string_view>

#include "server/openai_request.h"

namespace xs::server {

// `dialect` is the model's output dialect: "thinking": {"type": "disabled"} turns its thinking off the way it reads
// (chat::direct_answer). `count_only`: a count_tokens body, which has no max_tokens.
ChatRequest parse_messages_request(const RequestJson& body, const SamplingParams& defaults, std::string_view dialect,
                                   bool count_only = false);

}  // namespace xs::server
