#include "server/logprobs.h"

#include <algorithm>

#include "tokenizer/tokenizer.h"
#include "tokenizer/utf8.h"

namespace xs::server {
namespace {

constexpr size_t kLongestCharacter = 4;
// Bytes held for a character still unfinished, with the whole characters of the tokens that hold them; past this, the
// whole ones are let go (a byte sequence that never completes).
constexpr size_t kMostHeld = 16;

// The length of the unfinished character `s` ends with (a lead byte and fewer continuation bytes than it announces),
// 0 when it ends with a whole one or an invalid byte.
size_t unfinished_tail(std::string_view s) {
    for (size_t back = 1; back <= std::min(s.size(), kLongestCharacter - 1); ++back) {
        const auto b = static_cast<unsigned char>(s[s.size() - back]);
        if ((b & 0xC0) == 0x80) continue;  // a continuation byte: its lead is further back
        const size_t length = (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 0;
        return length > back ? back : 0;
    }
    return 0;
}

}  // namespace

double logprob_value(float logprob) {
    return std::max(static_cast<double>(logprob), -9999.0);
}

std::string LogprobTexts::text(const std::string& bytes, std::string* unfinished) const {
    const std::string joined = unfinished_ + bytes;
    const size_t tail = unfinished_tail(joined);
    if (tail > 0) {
        if (unfinished != nullptr) *unfinished = joined.size() > kMostHeld ? joined.substr(joined.size() - tail) : joined;
        return "";
    }
    if (unfinished != nullptr) unfinished->clear();
    return tok::utf8::lossy(joined);
}

LogprobToken LogprobTexts::token(int32_t id, float logprob, std::string* unfinished) const {
    LogprobToken t;
    t.bytes = tokenizer_.token_bytes(id);
    t.text = text(t.bytes, unfinished);
    t.logprob = logprob;
    return t;
}

LogprobEntry LogprobTexts::entry(int32_t token, const TokenLogprobs& logprobs) {
    LogprobEntry e;
    for (const auto& [id, logprob] : logprobs.top) e.top.push_back(this->token(id, logprob));
    std::string unfinished;
    e.token = this->token(token, logprobs.logprob, &unfinished);
    unfinished_ = std::move(unfinished);
    return e;
}

}  // namespace xs::server
