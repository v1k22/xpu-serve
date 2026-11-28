// Llama 3's tokenizer (byte-level BPE, the Llama 3 Split regex, ignore_merges, a post-processor Sequence that puts
// <|begin_of_text|> first) against Hugging Face, fixture from tests/parity/tokenizer/gen_llama3_cases.py. Needs the
// tokenizer.json of a Llama 3.x checkpoint in XS_LLAMA3_DIR and skips without it.

#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "parity_cases.h"
#include "test_support.h"
#include "tokenizer/tokenizer.h"

TEST_SUITE("tokenizer_parity") {
    TEST_CASE("tokenizer parity: Llama 3 encode, the begin-of-text token, decode and stream decode match HF") {
        const auto dir = xs::test::llama3_dir();
        if (!dir) return;
        std::ifstream in(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/llama3_cases.json");
        REQUIRE(in.good());
        const nlohmann::json fixture = nlohmann::json::parse(in);
        REQUIRE(fixture["encode_cases"].size() >= 45);
        const auto tok = xs::tok::Tokenizer::load(*dir);
        xs::test::check_special_token_cases(*tok, fixture);
    }
}
