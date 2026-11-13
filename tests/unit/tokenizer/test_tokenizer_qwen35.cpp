// Byte-level BPE (Qwen3.5) against Hugging Face, fixture from tests/parity/tokenizer/gen_qwen35_cases.py. The regex
// splitter, NFC and byte-level tests are hermetic; the whole-tokenizer tests need Qwen/Qwen3.5-4B's tokenizer.json
// in XS_QWEN35_DIR (a Qwen/Qwen3.5-4B checkpoint directory) and skip without it.

#include <doctest/doctest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "parity_cases.h"
#include "scratch_model_dir.h"
#include "tokenizer/byte_level.h"
#include "tokenizer/pre_tokenizer.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"
#include "tokenizer/utf8.h"
#include "util/error.h"

namespace {
using Ids = std::vector<int32_t>;
using xs::test::ScratchModelDir;
using xs::test::stream_decode;

const nlohmann::json& fixture() {
    static const nlohmann::json j = [] {
        std::ifstream in(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/qwen35_cases.json");
        REQUIRE(in.good());
        return nlohmann::json::parse(in);
    }();
    return j;
}

std::optional<std::filesystem::path> qwen35_dir() {
    const char* env = std::getenv("XS_QWEN35_DIR");
    if (env == nullptr || *env == '\0') return std::nullopt;
    std::filesystem::path dir = env;
    if (!std::filesystem::exists(dir / "tokenizer.json")) return std::nullopt;
    return dir;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::string chars(std::string_view bytes) {
    std::string out;
    for (const char b : bytes) out += xs::tok::byte_level::byte_piece(static_cast<uint8_t>(b));
    return out;
}

// A tiny byte-level tokenizer in the Qwen3.5 layout: ids 0-255 are the byte characters; merges make 256 = "a" + the
// first byte of "你", 257 = the last byte of "你" + the first of "好" (ranked first), 259 = "你" and 261 = "好".
nlohmann::json tiny_byte_level_json() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int b = 0; b < 256; ++b) vocab[chars(std::string(1, static_cast<char>(b)))] = b;
    const std::string ni = "\xE4\xBD\xA0";  // 你
    const std::string hao = "\xE5\xA5\xBD";  // 好
    const std::vector<std::pair<std::string, std::string>> merges = {
        {"a", ni.substr(0, 1)},           {ni.substr(2, 1), hao.substr(0, 1)}, {ni.substr(0, 1), ni.substr(1, 1)},
        {ni.substr(0, 2), ni.substr(2, 1)}, {hao.substr(0, 1), hao.substr(1, 1)}, {hao.substr(0, 2), hao.substr(2, 1)},
    };
    nlohmann::json merge_list = nlohmann::json::array();
    int next_id = 256;
    for (const auto& [left, right] : merges) {
        merge_list.push_back(chars(left) + " " + chars(right));
        vocab[chars(left + right)] = next_id++;
    }
    const std::string pattern = fixture()["split_cases"]["qwen35"]["pattern"].get<std::string>();
    return {
        {"normalizer", {{"type", "NFC"}}},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           {{{"type", "Split"}, {"pattern", {{"Regex", pattern}}}, {"behavior", "Isolated"}, {"invert", false}},
            {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", false}, {"use_regex", false}}}}}},
        {"decoder", {{"type", "ByteLevel"}}},
        {"post_processor", {{"type", "ByteLevel"}}},
        {"added_tokens", {{{"id", 300}, {"content", "<|im_end|>"}, {"special", true}, {"normalized", false}}}},
        {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", merge_list}, {"byte_fallback", false}}},
    };
}
}  // namespace

TEST_SUITE("tokenizer") {
    TEST_CASE("tokenizer: regex Split matches Hugging Face on the Qwen3.5, Qwen2 and Llama 3 patterns") {
        for (const auto& [name, group] : fixture()["split_cases"].items()) {
            const nlohmann::json config = {{"type", "Split"},
                                           {"pattern", {{"Regex", group["pattern"]}}},
                                           {"behavior", "Isolated"},
                                           {"invert", false}};
            const auto split = xs::tok::make_pre_tokenizer(config);
            for (const auto& c : group["cases"]) {
                const std::string text = c[0].get<std::string>();
                CAPTURE(name);
                CAPTURE(text);
                CHECK(split->split({text}) == c[1].get<std::vector<std::string>>());
            }
        }
    }

    TEST_CASE("tokenizer: NFC matches Hugging Face") {
        for (const auto& c : fixture()["nfc_cases"]) {
            const std::string text = c[0].get<std::string>();
            CAPTURE(text);
            CHECK(xs::tok::unicode::nfc(text) == c[1].get<std::string>());
        }
        CHECK(xs::tok::unicode::nfc("plain ascii stays") == "plain ascii stays");
        CHECK(xs::tok::unicode::nfc("a\xFF" "e\xCC\x81\xC3") == "a\xFF\xC3\xA9\xC3");  // invalid bytes pass through
    }

    TEST_CASE("tokenizer: Unicode tables") {
        using xs::tok::unicode::Category;
        using xs::tok::unicode::category;
        CHECK(category(U'a') == Category::kLetter);
        CHECK(category(U'中') == Category::kLetter);
        CHECK(category(0x0301) == Category::kMark);
        CHECK(category(0x093F) == Category::kMark);  // Devanagari vowel sign I (Mc)
        CHECK(category(U'7') == Category::kNumber);
        CHECK(category(0x0663) == Category::kNumber);  // Arabic-Indic three
        CHECK(category(0x00BD) == Category::kNumber);  // one half (No)
        CHECK(category(0x2167) == Category::kNumber);  // roman numeral eight (Nl)
        CHECK(category(U' ') == Category::kOther);
        CHECK(category(0x1F600) == Category::kOther);
        CHECK(category(0x10D4A) == Category::kLetter);  // Garay: newer than Python's unicodedata, known to the regex
        CHECK(xs::tok::unicode::combining_class(0x0301) == 230);
        CHECK(xs::tok::unicode::combining_class(0x0323) == 220);
        CHECK(xs::tok::unicode::combining_class(0x093C) == 7);
        CHECK(xs::tok::unicode::combining_class(U'a') == 0);
        CHECK(xs::tok::unicode::combining_class(0x1DF6) == 0);  // newer than the tokenizers normalizer's Unicode
    }

    TEST_CASE("tokenizer: byte-level alphabet and lossy UTF-8") {
        using namespace xs::tok;
        CHECK(byte_level::byte_to_char('A') == U'A');
        CHECK(byte_level::byte_to_char(' ') == 0x120);   // Ġ
        CHECK(byte_level::byte_to_char('\n') == 0x10A);  // Ċ
        CHECK(byte_level::byte_to_char(0x7F) == 0x121);
        CHECK(byte_level::byte_to_char(0xAD) == 0x143);
        CHECK(byte_level::byte_to_char(0xE9) == 0xE9);
        for (int b = 0; b < 256; ++b) CHECK(byte_level::char_to_byte(byte_level::byte_to_char(b)) == b);
        CHECK_FALSE(byte_level::char_to_byte(0x144).has_value());
        CHECK_FALSE(byte_level::char_to_byte(0xAD).has_value());
        const std::string r = "\xEF\xBF\xBD";
        CHECK(utf8::lossy("ok \xC3\xA9") == "ok \xC3\xA9");
        CHECK(utf8::lossy("\xE4\xBD") == r);
        CHECK(utf8::lossy("\xE4\xBDx") == r + "x");
        CHECK(utf8::lossy("\xF0\x9F\x98") == r);
        CHECK(utf8::lossy("\xC0\xAF") == r + r);
        CHECK(utf8::lossy("\xED\xA0\x80") == r + r + r);
        CHECK(utf8::lossy("a\x80" "b") == "a" + r + "b");
    }

    TEST_CASE("tokenizer: byte-level BPE encodes bytes and merges across character boundaries") {
        const ScratchModelDir dir(tiny_byte_level_json());
        const auto tok = xs::tok::Tokenizer::load(dir.path());
        CHECK(tok->encode("你") == Ids{259});
        CHECK(tok->encode("好") == Ids{261});
        CHECK(tok->encode("a你好") == Ids{256, 0xBD, 257, 0xA5, 0xBD});
        CHECK(tok->encode(" a") == Ids{' ', 'a'});
        CHECK(tok->encode("x<|im_end|>") == Ids{'x', 300});
        CHECK(tok->decode(Ids{256, 0xBD, 257, 0xA5, 0xBD}) == "a你好");
        CHECK(tok->decode(Ids{256}) == "a\xEF\xBF\xBD");
    }

    TEST_CASE("tokenizer: byte-level stream decoding never splits a character") {
        const ScratchModelDir dir(tiny_byte_level_json());
        const auto tok = xs::tok::Tokenizer::load(dir.path());
        // Tokens that end inside a character: the complete text before it comes out at once, the rest waits.
        xs::tok::StreamDecoder stream(*tok);
        CHECK(stream.push(256) == "a");
        CHECK(stream.push(0xBD) == "");
        CHECK(stream.push(257) == "你");
        CHECK(stream.push(0xA5) == "");
        CHECK(stream.push(0xBD) == "好");
        CHECK(stream.flush() == "");

        // One token per byte, through 2-, 3- and 4-byte characters and a special token.
        const std::string text = "é你😀<|im_end|>ok";
        Ids ids;
        for (const char b : std::string("é你😀")) ids.push_back(static_cast<uint8_t>(b));
        ids.push_back(300);
        ids.push_back('o');
        ids.push_back('k');
        CHECK(stream_decode(*tok, ids, false) == text);
        CHECK(stream_decode(*tok, ids, true) == "é你😀ok");

        // Bytes that never complete a character come out as U+FFFD at flush, like a full decode.
        const Ids broken = {'x', 0xE4, 0xBD};
        CHECK(stream_decode(*tok, broken, false) == tok->decode(broken));
    }

    TEST_CASE("tokenizer: byte-level components refuse what they do not implement") {
        auto error = [](const nlohmann::json& j) -> std::string {
            const ScratchModelDir dir(j);
            try {
                xs::tok::Tokenizer::load(dir.path());
            } catch (const xs::Error& e) {
                return e.what();
            }
            return "loaded without error";
        };
        nlohmann::json j = tiny_byte_level_json();
        std::swap(j["pre_tokenizer"]["pretokenizers"][0], j["pre_tokenizer"]["pretokenizers"][1]);
        CHECK(error(j) == "tokenizer.json pre_tokenizer 'ByteLevel' must be the last step");

        j = tiny_byte_level_json();
        j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "\\p{L}+";
        CHECK(error(j) == "tokenizer.json pre_tokenizer 'Split' Regex '\\p{L}+' is not supported");
    }

    TEST_CASE("tokenizer: a byte the byte-level vocab lacks is dropped, as Hugging Face drops it without unk") {
        nlohmann::json j = tiny_byte_level_json();
        j["model"]["vocab"].erase(chars("\x01"));
        j["model"]["vocab"]["spare"] = 1;
        const ScratchModelDir dir(j);
        CHECK(xs::tok::Tokenizer::load(dir.path())->encode("a\x01" "b") == Ids{'a', 'b'});
    }
}

TEST_SUITE("tokenizer_parity") {
    TEST_CASE("tokenizer parity: Qwen3.5 encode, decode and stream decode match Hugging Face") {
        const auto dir = qwen35_dir();
        if (!dir) return;
        const auto start = std::chrono::steady_clock::now();
        const auto tok = xs::tok::Tokenizer::load(*dir);
        MESSAGE("Qwen3.5 tokenizer.json load: " << seconds_since(start) << " s");
        REQUIRE(fixture()["bos_id"].is_null());
        xs::test::check_tokenizer_cases(*tok, fixture());
    }

    TEST_CASE("tokenizer parity: a 5k-token Qwen3.5 prompt encodes in milliseconds") {
        const auto dir = qwen35_dir();
        if (!dir) return;
        const auto tok = xs::tok::Tokenizer::load(*dir);
        std::string text;
        Ids expected;
        while (expected.size() < 5000) {
            for (const auto& c : fixture()["encode_cases"]) {
                if (!c["parse_special"].get<bool>()) continue;
                text += c["text"].get<std::string>();
                const Ids ids = c["ids"].get<Ids>();
                expected.insert(expected.end(), ids.begin(), ids.end());
            }
        }
        const auto start = std::chrono::steady_clock::now();
        const Ids ids = tok->encode(text);
        const double elapsed = seconds_since(start);
        MESSAGE("encoded " << text.size() << " bytes into " << ids.size() << " tokens in " << elapsed * 1e3 << " ms");
        CHECK(elapsed < 0.1);
        CHECK(tok->decode(ids) == tok->decode(expected));
    }
}
