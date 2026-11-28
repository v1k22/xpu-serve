#pragma once

// A reply held to tool calls (OpenAI's tool_choice "required" or a named function; "auto" beside a response_format,
// where the reply is calls or the formatted answer), as a grammar over the model's own call syntax, its dialect's: the
// dialect's parser then reads the calls as it reads free ones, and the model writes them as it was trained to.
//
// A call's arguments follow the tool's parameters schema when the tool is strict (`function.strict: true`), as vLLM
// holds them; any arguments otherwise. Dialects: qwen3_5 (Qwen 3.5's XML parameters; LFM3 writes them too), qwen3
// (a JSON object a call), gemma4 (Gemma 4's own argument syntax), inkling2 (a message a call, its body JSON), llama3
// (the reply's JSON objects {"name": ..., "parameters": ...}, "; " between two).

#include <memory>
#include <optional>
#include <string_view>

#include "grammar/grammar.h"
#include "grammar/vocabulary.h"
#include "server/openai_request.h"

namespace xs::server {

struct ToolCalls {
    const RequestJson* tools = nullptr;  // the request's, validated (parse_chat_request)
    ToolChoice choice;                   // required or named; automatic with `answer`
    bool parallel = true;                // several calls may follow each other
    std::shared_ptr<const grammar::Grammar> answer;  // the reply's other way, beside the calls (response_format)
    // Before `answer`, what the dialect writes ahead of an answer's text (chat::answer_start's opening).
    std::string answer_opening;
};

// Whether a dialect writes tool calls this module has a syntax for.
bool writes_tool_calls(std::string_view dialect);

// Throws InvalidRequest naming the keyword of a strict tool's schema it does not take
// (tools[1].function.parameters.properties.age.minimum), or the dialect when it writes no tool calls.
std::shared_ptr<const grammar::Grammar> tool_call_grammar(std::string_view dialect, const ToolCalls& calls,
                                                          const grammar::Vocabulary& vocabulary);

// tool_choice "auto" with strict tools (`function.strict: true`): the reply is free, but a call, once the model writes
// its opener (the marker token `opener`: <tool_call>, <|tool_call>, Inkling 2's <|content_invoke_tool_json|>), is held
// to `grammar` up to its closer, a strict tool's arguments to its schema (vLLM's triggered tags).
struct CallsAfterOpener {
    int32_t opener;
    std::shared_ptr<const grammar::Grammar> grammar;
};
// Nothing when no tool is strict, the dialect writes no calls, or its opener is no marker token of the vocabulary.
// Throws as tool_call_grammar.
std::optional<CallsAfterOpener> strict_calls(std::string_view dialect, const RequestJson& tools,
                                             const grammar::Vocabulary& vocabulary);

}  // namespace xs::server
