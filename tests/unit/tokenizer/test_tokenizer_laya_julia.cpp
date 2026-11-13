// The tokenizers of the Laya and Julia-1 decision models against Hugging Face: ModernBERT's (NFC, ByteLevel with the
// GPT-2 regex, byte-level BPE, normalized added tokens) and mmBERT's (Gemma's vocabulary: Replace, Metaspace, BPE with
// byte fallback). Hermetic: the GPT-2 regex, ByteLevel and Metaspace on the fixture's texts and the regex over every
// code point (digests), and Metaspace's refusals. The whole-tokenizer parity tests read the checkpoints in XS_LAYA_DIR
// and XS_JULIA_DIR and skip without them. Fixture: tests/parity/tokenizer/gen_laya_julia_cases.py.

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "parity_cases.h"
#include "tokenizer/byte_level.h"
#include "tokenizer/pre_tokenizer.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/utf8.h"
#include "util/error.h"

namespace {
using Pieces = std::vector<std::string>;

const nlohmann::json& fixture() {
    static const nlohmann::json j = [] {
        std::ifstream in(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/laya_julia_cases.json");
        REQUIRE(in.good());
        return nlohmann::json::parse(in);
    }();
    return j;
}

std::optional<std::filesystem::path> env_dir(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') return std::nullopt;
    return std::filesystem::path(env);
}

// Each tokenizer of a checkpoint, as the fixture keys it: "<model>/<folder>".
void check_checkpoint(const char* env, const std::string& model, const std::vector<std::string>& folders) {
    const auto root = env_dir(env);
    if (!root) return;
    for (const std::string& folder : folders) {
        const std::string key = model + "/" + folder;
        CAPTURE(key);
        const nlohmann::json& f = xs::test::cases_of(fixture()["tokenizers_by_folder"], key);
        REQUIRE(f["encode_cases"].size() >= 450);
        xs::test::check_special_token_cases(*xs::tok::Tokenizer::load(*root / folder), f);
    }
}

std::string pre_tokenizer_error(const nlohmann::json& config) {
    try {
        xs::tok::make_pre_tokenizer(config);
    } catch (const xs::Error& e) {
        return e.what();
    }
    return "built without error";
}
}  // namespace

TEST_SUITE("tokenizer") {
    TEST_CASE("pre-tokenizers: the GPT-2 regex, ByteLevel and Metaspace match Hugging Face") {
        const nlohmann::json& components = fixture()["components"];
        for (const auto& group : components["byte_level"]) {
            const auto pre_tokenizer = xs::tok::make_pre_tokenizer(group["config"]);
            REQUIRE(pre_tokenizer->byte_level() == group["byte_level"].get<bool>());
            for (const auto& c : group["cases"]) {
                CAPTURE(group["config"].dump());
                CAPTURE(c[0].get<std::string>());
                Pieces pieces = pre_tokenizer->split({c[0].get<std::string>()});
                if (pre_tokenizer->byte_level()) {  // Hugging Face's pieces are in the byte-level alphabet already
                    for (std::string& piece : pieces) {
                        std::string mapped;
                        for (const char b : piece) mapped += xs::tok::byte_level::byte_piece(static_cast<uint8_t>(b));
                        piece = std::move(mapped);
                    }
                }
                CHECK(pieces == c[1].get<Pieces>());
            }
        }
        for (const auto& group : components["metaspace"]) {
            const auto pre_tokenizer = xs::tok::make_pre_tokenizer(group["config"]);
            for (const auto& c : group["cases"]) {
                CAPTURE(group["config"].dump());
                CAPTURE(c[0].get<std::string>());
                CHECK(pre_tokenizer->split({c[0].get<std::string>()}) == c[1].get<Pieces>());
            }
        }
    }

    TEST_CASE("pre-tokenizers: the GPT-2 regex matches Hugging Face on every code point") {
        const nlohmann::json& components = fixture()["components"];
        const auto block = components["block"].get<uint32_t>();
        const auto& digests = components["gpt2_digests"];
        REQUIRE(digests.size() * block == 0x110000);
        const auto split = xs::tok::make_pre_tokenizer(components["byte_level"][0]["config"]);
        for (size_t b = 0; b < digests.size(); ++b) {
            CAPTURE(b * block);
            std::string text;
            for (char32_t c = static_cast<char32_t>(b * block); c < (b + 1) * block; ++c) {
                if (c >= 0xD800 && c < 0xE000) continue;
                for (const char32_t part : {c, U'a', c, U'1', U' ', c}) xs::tok::utf8::append(text, part);
            }
            std::string summary;
            for (const std::string& piece : split->split({text})) summary += std::to_string(piece.size()) + ":" + piece;
            CHECK(xs::test::fnv1a64(summary) == digests[b].get<std::string>());
        }
    }

    TEST_CASE("pre-tokenizers: what Metaspace and ByteLevel do not implement is refused by name") {
        const std::string kMetaspace = "tokenizer.json pre_tokenizer 'Metaspace'";
        const nlohmann::json metaspace = {{"type", "Metaspace"}, {"replacement", "▁"}, {"split", true}};
        nlohmann::json j = metaspace;
        j["prepend_scheme"] = "first";
        CHECK(pre_tokenizer_error(j) == kMetaspace + " prepend_scheme 'first' is not supported");
        j = metaspace;
        j["add_prefix_space"] = false;  // Hugging Face's older form; with the default scheme, always, it refuses it
        CHECK(pre_tokenizer_error(j) ==
              kMetaspace + ": add_prefix_space false does not match prepend_scheme 'always'");
        j["prepend_scheme"] = "never";
        CHECK(pre_tokenizer_error(j) == "built without error");
        j = metaspace;
        j["replacement"] = "__";
        CHECK(pre_tokenizer_error(j) == kMetaspace + ": 'replacement' must be exactly one character");
        CHECK(pre_tokenizer_error({{"type", "Split"}, {"pattern", {{"Regex", "\\s+"}}}, {"behavior", "Isolated"}}) ==
              "tokenizer.json pre_tokenizer 'Split' Regex '\\s+' is not supported");
    }
}

TEST_SUITE("tokenizer_parity") {
    TEST_CASE("tokenizer parity: Laya's three tokenizers match Hugging Face") {
        check_checkpoint("XS_LAYA_DIR", "laya", {"tokenizer", "typed-decisions/tokenizer", "multilingual/tokenizer"});
    }

    TEST_CASE("tokenizer parity: Julia-1's tokenizer matches Hugging Face") {
        check_checkpoint("XS_JULIA_DIR", "julia", {"tokenizer"});
    }
}
