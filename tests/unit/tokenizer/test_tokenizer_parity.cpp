// Gemma 4 tokenizer parity against Hugging Face (fixture: tests/parity/tokenizer/gen_cases.py), plus the encode-speed
// bar. Both need XS_TEST_MODEL_DIR pointing at google/gemma-4-E2B-it and skip without it.

#include <doctest/doctest.h>

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tokenizer/tokenizer.h"
#include "unit/test_support.h"

namespace {
using Ids = std::vector<int32_t>;

nlohmann::json gemma4_cases() {
    std::ifstream in(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/gemma4_cases.json");
    REQUIRE(in.good());
    return nlohmann::json::parse(in);
}

std::string stream_decode(const xs::tok::Tokenizer& tok, const Ids& ids, bool skip_special) {
    xs::tok::StreamDecoder stream(tok, skip_special);
    std::string text;
    for (const int32_t id : ids) text += stream.push(id);
    return text + stream.flush();
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
}  // namespace

TEST_SUITE("tokenizer_parity") {
    TEST_CASE("tokenizer parity: Gemma 4 encode, decode and stream decode match Hugging Face") {
        const auto model_dir = xs::test::model_dir();
        if (!model_dir) return;
        const auto start = std::chrono::steady_clock::now();
        const auto tok = xs::tok::Tokenizer::load(*model_dir);
        MESSAGE("tokenizer.json load: " << seconds_since(start) << " s");
        const nlohmann::json fixture = gemma4_cases();

        CHECK(tok->vocab_size() == fixture["vocab_size"].get<int32_t>());
        CHECK(tok->bos_id() == fixture["bos_id"].get<int32_t>());
        CHECK(tok->eos_id() == fixture["eos_id"].get<int32_t>());
        for (const int32_t id : fixture["special_ids"].get<Ids>()) CHECK(tok->is_special(id));

        for (const auto& c : fixture["encode_cases"]) {
            CAPTURE(c["name"].get<std::string>());
            const std::string text = c["text"].get<std::string>();
            const Ids expected = c["ids"].get<Ids>();
            const Ids ids = tok->encode(text, c["parse_special"].get<bool>());
            CHECK(ids == expected);
            CHECK(tok->decode(expected) == c["decoded"].get<std::string>());
            CHECK(tok->decode(expected, true) == c["decoded_skip_special"].get<std::string>());
            CHECK(tok->decode(ids) == text);  // every Gemma case round-trips
            CHECK(stream_decode(*tok, expected, false) == c["decoded"].get<std::string>());
            CHECK(stream_decode(*tok, expected, true) == c["decoded_skip_special"].get<std::string>());
        }
        for (const auto& c : fixture["decode_cases"]) {
            CAPTURE(c["name"].get<std::string>());
            const Ids ids = c["ids"].get<Ids>();
            CHECK(tok->decode(ids) == c["decoded"].get<std::string>());
            CHECK(tok->decode(ids, true) == c["decoded_skip_special"].get<std::string>());
        }
    }

    TEST_CASE("tokenizer parity: 200 KB of text encodes in well under a second") {
        const auto model_dir = xs::test::model_dir();
        if (!model_dir) return;
        const auto tok = xs::tok::Tokenizer::load(*model_dir);
        const nlohmann::json fixture = gemma4_cases();
        std::string text;
        for (const auto& c : fixture["encode_cases"]) {
            if (c["name"].get<std::string>() == "long_mixed") text = c["text"].get<std::string>();
        }
        REQUIRE(!text.empty());
        while (text.size() < 200 * 1024) text += text;
        text.resize(200 * 1024);
        while (!text.empty() && (static_cast<uint8_t>(text.back()) & 0xC0) == 0x80) text.pop_back();  // whole chars

        const auto start = std::chrono::steady_clock::now();
        const Ids ids = tok->encode(text);
        const double elapsed = seconds_since(start);
        MESSAGE("encoded " << text.size() << " bytes into " << ids.size() << " tokens in " << elapsed << " s");
        CHECK(elapsed < 1.0);  // ~10x headroom over the measured time; an O(n^2) merge takes far longer
        CHECK(tok->decode(ids) == text);
    }
}
