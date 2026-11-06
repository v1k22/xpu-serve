#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace xs::chat::gemma4 {

// The escape token Gemma 4 wraps string values in.
inline constexpr std::string_view kStringDelim = "<|\"|>";

// Converts one Gemma 4 tool-call argument object, as the chat template's format_argument writes it
// ({key:<|"|>text<|"|>,n:1.5,flag:true,none:null,obj:{...},list:[...]}), to JSON object text. Also accepts JSON
// spellings ("quoted" keys and strings). A bare value that is not a JSON literal becomes a string. Returns nullopt
// unless `text` is exactly one well-formed object, optionally surrounded by whitespace.
std::optional<std::string> args_to_json(std::string_view text);

}  // namespace xs::chat::gemma4
