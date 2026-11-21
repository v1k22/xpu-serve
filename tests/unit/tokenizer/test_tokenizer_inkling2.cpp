// Inkling 2's tokenizer (byte-level BPE, o200k_base's Split regex: words by letter case, slashes after punctuation;
// ignore_merges) against Hugging Face, fixture from tests/parity/tokenizer/gen_inkling2_cases.py. The regex cases
// are hermetic; the tokenizer's need the tokenizer.json of thinkingmachines/Inkling-2-Mini in XS_INKLING2_DIR and
// skip without it.

#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "parity_cases.h"
#include "test_support.h"
#include "tokenizer/pre_tokenizer.h"
#include "tokenizer/regex_split.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

namespace {

const nlohmann::json& fixture() {
    static const nlohmann::json f = [] {
        std::ifstream in(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/inkling2_cases.json");
        REQUIRE(in.good());
        return nlohmann::json::parse(in);
    }();
    return f;
}

}  // namespace

TEST_SUITE("tokenizer") {
    TEST_CASE("tokenizer: regex Split matches Hugging Face on the o200k pattern") {
        const nlohmann::json& group = fixture()["split_cases"]["o200k"];
        REQUIRE(group["pattern"].get<std::string>() == xs::tok::split_regex::kO200k);
        REQUIRE(group["cases"].size() >= 600);
        const nlohmann::json config = {{"type", "Split"},
                                       {"pattern", {{"Regex", group["pattern"]}}},
                                       {"behavior", "Isolated"},
                                       {"invert", false}};
        const auto split = xs::tok::make_pre_tokenizer(config);
        for (const auto& c : group["cases"]) {
            const std::string text = c[0].get<std::string>();
            CAPTURE(text);
            CHECK(split->split({text}) == c[1].get<std::vector<std::string>>());
        }
    }

    TEST_CASE("tokenizer: letter cases of the Unicode tables") {
        using xs::tok::unicode::letter_case;
        using xs::tok::unicode::LetterCase;
        CHECK(letter_case('A') == LetterCase::kUpper);
        CHECK(letter_case('z') == LetterCase::kLower);
        CHECK(letter_case('1') == LetterCase::kNone);
        CHECK(letter_case(0xC9) == LetterCase::kUpper);    // É
        CHECK(letter_case(0xDF) == LetterCase::kLower);    // ß
        CHECK(letter_case(0x1C5) == LetterCase::kUpper);   // ǅ: titlecase counts with uppercase
        CHECK(letter_case(0x2B0) == LetterCase::kNone);    // ʰ: a modifier letter
        CHECK(letter_case(0x4E2D) == LetterCase::kNone);   // 中: a letter without case
        CHECK(letter_case(0x301) == LetterCase::kNone);    // a combining mark
        CHECK(letter_case(0x1D400) == LetterCase::kUpper);  // 𝐀, beyond the BMP
    }
}

TEST_SUITE("tokenizer_parity") {
    TEST_CASE("tokenizer parity: Inkling 2 encode, decode and stream decode match Hugging Face") {
        const auto dir = xs::test::inkling2_dir();
        if (!dir) return;
        REQUIRE(fixture()["encode_cases"].size() >= 45);
        const auto tok = xs::tok::Tokenizer::load(*dir);
        xs::test::check_tokenizer_cases(*tok, fixture());
    }
}
