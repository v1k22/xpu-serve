// LFM3's tokenizer (byte-level BPE, the Llama 3 Split regex with digits in groups of three, ignore_merges) against
// Hugging Face, fixture from tests/parity/tokenizer/gen_lfm3_cases.py. Needs the tokenizer.json of LiquidAI/LFM3-1.7B
// in XS_LFM3_DIR and skips without it.

#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "parity_cases.h"
#include "test_support.h"
#include "tokenizer/tokenizer.h"

TEST_SUITE("tokenizer_parity") {
    TEST_CASE("tokenizer parity: LFM3 encode, decode and stream decode match Hugging Face") {
        const auto dir = xs::test::lfm3_dir();
        if (!dir) return;
        std::ifstream in(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/lfm3_cases.json");
        REQUIRE(in.good());
        const nlohmann::json fixture = nlohmann::json::parse(in);
        REQUIRE(fixture["encode_cases"].size() >= 45);
        const auto tok = xs::tok::Tokenizer::load(*dir);
        xs::test::check_tokenizer_cases(*tok, fixture);
    }
}
