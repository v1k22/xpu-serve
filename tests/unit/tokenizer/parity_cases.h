#pragma once

// What the whole-tokenizer parity tests share: the encode and decode cases of a fixture written by
// tests/parity/tokenizer/gen_*_cases.py, checked against a loaded tokenizer.

#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tokenizer/tokenizer.h"
#include "tokenizer/utf8.h"

namespace xs::test {

// Streams ids one at a time; every delta must be valid UTF-8 (no character split across deltas).
inline std::string stream_decode(const tok::Tokenizer& tok, const std::vector<int32_t>& ids, bool skip_special) {
    tok::StreamDecoder stream(tok, skip_special);
    std::string text;
    for (const int32_t id : ids) {
        const std::string delta = stream.push(id);
        CHECK(tok::utf8::is_valid(delta));
        text += delta;
    }
    return text + stream.flush();
}

// The fixture's vocabulary facts, then every case: encode, decode and stream decode, with and without specials.
inline void check_tokenizer_cases(const tok::Tokenizer& tok, const nlohmann::json& f) {
    using Ids = std::vector<int32_t>;
    CHECK(tok.vocab_size() == f["vocab_size"].get<int32_t>());
    if (f["bos_id"].is_null()) {
        CHECK_FALSE(tok.bos_id().has_value());
    } else {
        CHECK(tok.bos_id() == f["bos_id"].get<int32_t>());
    }
    CHECK(tok.eos_id() == f["eos_id"].get<int32_t>());
    for (const int32_t id : f["special_ids"].get<Ids>()) CHECK(tok.is_special(id));

    for (const auto& c : f["encode_cases"]) {
        CAPTURE(c["name"].get<std::string>());
        const std::string text = c["text"].get<std::string>();
        const Ids expected = c["ids"].get<Ids>();
        CHECK(tok.encode(text, c["parse_special"].get<bool>()) == expected);
        CHECK(tok.decode(expected) == c["decoded"].get<std::string>());
        CHECK(tok.decode(expected, true) == c["decoded_skip_special"].get<std::string>());
        CHECK(stream_decode(tok, expected, false) == c["decoded"].get<std::string>());
        CHECK(stream_decode(tok, expected, true) == c["decoded_skip_special"].get<std::string>());
    }
    for (const auto& c : f["decode_cases"]) {
        CAPTURE(c["name"].get<std::string>());
        const Ids ids = c["ids"].get<Ids>();
        CHECK(tok.decode(ids) == c["decoded"].get<std::string>());
        CHECK(tok.decode(ids, true) == c["decoded_skip_special"].get<std::string>());
        CHECK(stream_decode(tok, ids, false) == c["decoded"].get<std::string>());
    }
}

}  // namespace xs::test
