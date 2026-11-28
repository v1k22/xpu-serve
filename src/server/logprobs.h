#pragma once

// Generated tokens' log-probabilities as the OpenAI APIs write them (GenerateRequest::logprobs): each token as the
// text it adds to the output, read in order. A byte-level or byte-fallback token may hold part of a character: as vLLM
// writes them, a token that leaves a character unfinished is "" and the one that finishes it carries the whole
// character; `bytes` is what the token itself adds.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "engine/types.h"

namespace xs::tok {
class Tokenizer;
}

namespace xs::server {

struct LogprobToken {
    std::string text;
    std::string bytes;
    float logprob = 0.0f;  // -infinity for a token the output constraint masked
};

// One generated token's: the token, and the most likely tokens at its position.
struct LogprobEntry {
    LogprobToken token;
    std::vector<LogprobToken> top;
};

// The entries of one generation's tokens, given in order.
class LogprobTexts {
public:
    explicit LogprobTexts(const tok::Tokenizer& tokenizer) : tokenizer_(tokenizer) {}
    // `token`, generated after the ones given before, and its alternatives at that position.
    LogprobEntry entry(int32_t token, const TokenLogprobs& logprobs);

private:
    // What a token with these bytes reads as at this point of the output, and the bytes it leaves unfinished.
    std::string text(const std::string& bytes, std::string* unfinished) const;
    LogprobToken token(int32_t id, float logprob, std::string* unfinished = nullptr) const;

    const tok::Tokenizer& tokenizer_;
    std::string unfinished_;  // the output's bytes after its last whole character
};

// What a log-probability is in JSON: -infinity as -9999 (vLLM's floor).
double logprob_value(float logprob);

}  // namespace xs::server
