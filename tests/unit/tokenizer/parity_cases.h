#pragma once

// What the whole-tokenizer parity tests share: the encode and decode cases of a fixture written by
// tests/parity/tokenizer/gen_*_cases.py, checked against a loaded tokenizer.

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
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

// The fixture's vocabulary facts: size, bos and eos (null: none), special ids.
inline void check_vocabulary_facts(const tok::Tokenizer& tok, const nlohmann::json& f) {
    CHECK(tok.vocab_size() == f["vocab_size"].get<int32_t>());
    for (const auto& [key, id] : {std::pair{"bos_id", tok.bos_id()}, std::pair{"eos_id", tok.eos_id()}}) {
        CAPTURE(key);
        if (f[key].is_null()) {
            CHECK_FALSE(id.has_value());
        } else {
            CHECK(id == f[key].get<int32_t>());
        }
    }
    for (const int32_t id : f["special_ids"].get<std::vector<int32_t>>()) CHECK(tok.is_special(id));
}

// The vocabulary facts, then every case: encode, decode and stream decode, with and without specials.
inline void check_tokenizer_cases(const tok::Tokenizer& tok, const nlohmann::json& f) {
    using Ids = std::vector<int32_t>;
    check_vocabulary_facts(tok, f);
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

// The same for fixtures that also hold each text's ids with the post-processor's special tokens
// (`ids_with_special`): their decodes are of those ids.
inline void check_special_token_cases(const tok::Tokenizer& tok, const nlohmann::json& f) {
    using Ids = std::vector<int32_t>;
    check_vocabulary_facts(tok, f);
    for (const auto& c : f["encode_cases"]) {
        CAPTURE(c["name"].get<std::string>());
        const Ids ids = tok.encode(c["text"].get<std::string>(), c["parse_special"].get<bool>());
        CHECK(ids == c["ids"].get<Ids>());
        const Ids expected = c["ids_with_special"].get<Ids>();
        CHECK(tok.with_special_tokens(ids) == expected);
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

// FNV-1a 64 of `text` as 16 hex digits: how fixtures summarize a component's output over every code point.
inline std::string fnv1a64(std::string_view text) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (const char b : text) h = (h ^ static_cast<uint8_t>(b)) * 0x100000001B3ull;
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));
    return hex;
}

// The cases stored under `key` in `all`, following {"same_as": <key>}.
inline const nlohmann::json& cases_of(const nlohmann::json& all, const std::string& key) {
    const nlohmann::json& f = all.at(key);
    return f.contains("same_as") ? all.at(f["same_as"].get<std::string>()) : f;
}

}  // namespace xs::test
