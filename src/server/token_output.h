#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "chat/output_parser.h"
#include "server/stop_strings.h"
#include "tokenizer/tokenizer.h"

namespace xs::server {

// Appends `from` to `into` (text concatenated, tool calls appended in order).
void append_delta(chat::ParsedDelta& into, chat::ParsedDelta from);

// Turns generated token ids into response pieces: detokenize (special tokens kept so the parser sees dialect
// markers), split into content / reasoning / tool calls, then cut content at the first stop string.
class TokenOutput {
public:
    TokenOutput(const tok::Tokenizer& tokenizer, std::unique_ptr<chat::OutputParser> parser,
                std::vector<std::string> stop_strings);

    chat::ParsedDelta push(int32_t token);
    // End of generation: releases everything still held back by the decoder, the parser and the stop matcher.
    chat::ParsedDelta finish();
    // True once a stop string matched; the caller should then cancel generation.
    bool stop_string_hit() const { return stops_.stopped(); }

private:
    chat::ParsedDelta apply_stop_strings(chat::ParsedDelta delta);

    tok::StreamDecoder decoder_;
    std::unique_ptr<chat::OutputParser> parser_;
    StopStringMatcher stops_;
};

}  // namespace xs::server
