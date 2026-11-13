// WordPiece (BERT-family) tokenizers against Hugging Face. Hermetic: the components on a hand-written vocabulary, and
// the BERT normalizer and pre-tokenizer over every code point (digests in the fixture). The whole-tokenizer parity
// tests read the checkpoints listed in XS_WORDPIECE_DIRS and skip without it. Fixture:
// tests/parity/tokenizer/gen_wordpiece_cases.py.

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "parity_cases.h"
#include "scratch_model_dir.h"
#include "tokenizer/decoder.h"
#include "tokenizer/normalizer.h"
#include "tokenizer/pre_tokenizer.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/utf8.h"
#include "util/error.h"

namespace {
using Ids = std::vector<int32_t>;
using xs::test::ScratchModelDir;
using xs::test::stream_decode;

const nlohmann::json& fixture() {
    static const nlohmann::json j = [] {
        std::ifstream in(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/wordpiece_cases.json");
        REQUIRE(in.good());
        return nlohmann::json::parse(in);
    }();
    return j;
}

// XS_WORDPIECE_DIRS: checkpoint directories separated by ':', each named as the fixture names its model.
std::vector<std::filesystem::path> wordpiece_dirs() {
    const char* env = std::getenv("XS_WORDPIECE_DIRS");
    std::vector<std::filesystem::path> dirs;
    if (env == nullptr) return dirs;
    for (std::string_view rest = env; !rest.empty();) {
        const size_t colon = rest.find(':');
        std::string dir(rest.substr(0, colon));
        while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
        if (!dir.empty()) dirs.emplace_back(dir);
        rest = colon == std::string_view::npos ? std::string_view() : rest.substr(colon + 1);
    }
    return dirs;
}

// The vocabulary of the hand-written tokenizer below, ids in order.
const char* const kPieces[] = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "un", "##aff", "##able", "aff", "able",
                               "hello", "world", ",", ".", "!", "'", "isn", "t", "##s", "中", "e", "##e", "é", "##é"};

nlohmann::json added_token(int32_t id) {
    return {{"id", id},          {"content", kPieces[id]}, {"single_word", false}, {"lstrip", false},
            {"rstrip", false},   {"normalized", false},    {"special", true}};
}

// A BERT-style tokenizer.json, laid out as all-MiniLM-L6-v2's (without truncation and padding).
nlohmann::json tiny_wordpiece_json() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int32_t id = 0; id < static_cast<int32_t>(std::size(kPieces)); ++id) vocab[kPieces[id]] = id;
    nlohmann::json added = nlohmann::json::array();
    for (int32_t id = 0; id < 5; ++id) added.push_back(added_token(id));
    return {
        {"version", "1.0"},
        {"truncation", nullptr},
        {"padding", nullptr},
        {"added_tokens", added},
        {"normalizer",
         {{"type", "BertNormalizer"},
          {"clean_text", true},
          {"handle_chinese_chars", true},
          {"strip_accents", nullptr},
          {"lowercase", true}}},
        {"pre_tokenizer", {{"type", "BertPreTokenizer"}}},
        {"post_processor", nlohmann::json::parse(R"({"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 1}}],
            "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                               "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}}})")},
        {"decoder", {{"type", "WordPiece"}, {"prefix", "##"}, {"cleanup", true}}},
        {"model",
         {{"type", "WordPiece"},
          {"unk_token", "[UNK]"},
          {"continuing_subword_prefix", "##"},
          {"max_input_chars_per_word", 100},
          {"vocab", vocab}}},
    };
}

std::unique_ptr<xs::tok::Tokenizer> load(const nlohmann::json& tokenizer_json) {
    const ScratchModelDir dir(tokenizer_json);
    return xs::tok::Tokenizer::load(dir.path());
}

std::string load_error(const nlohmann::json& tokenizer_json) {
    try {
        load(tokenizer_json);
    } catch (const xs::Error& e) {
        return e.what();
    }
    return "loaded without error";
}

std::string normalize(const nlohmann::json& config, std::string_view text) {
    return xs::tok::make_normalizer(config)->normalize(text);
}

nlohmann::json bert_normalizer(bool clean_text, bool chinese, nlohmann::json strip_accents, bool lowercase) {
    return {{"type", "BertNormalizer"},
            {"clean_text", clean_text},
            {"handle_chinese_chars", chinese},
            {"strip_accents", strip_accents},
            {"lowercase", lowercase}};
}

std::vector<std::string> pre_tokenize(std::string_view text) {
    const auto pre_tokenizer = xs::tok::make_pre_tokenizer({{"type", "BertPreTokenizer"}});
    std::vector<std::string> pieces;
    for (const std::string_view piece : pre_tokenizer->split({text})) pieces.emplace_back(piece);
    return pieces;
}

std::string decode_tokens(const nlohmann::json& config, std::vector<std::string> tokens) {
    xs::tok::make_decoder(config)->decode(tokens);
    std::string text;
    for (const auto& token : tokens) text += token;
    return text;
}

std::string fnv1a64(std::string_view text) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (const char b : text) h = (h ^ static_cast<uint8_t>(b)) * 0x100000001B3ull;
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h));
    return hex;
}

// The code points of one block (surrogates left out), each followed by `after` except the last when `between`.
std::string block_text(char32_t first, char32_t count, std::string_view after, bool between) {
    std::string text;
    for (char32_t c = first; c < first + count; ++c) {
        if (c >= 0xD800 && c < 0xE000) continue;
        if (between && !text.empty()) text += after;
        xs::tok::utf8::append(text, c);
        if (!between) text += after;
    }
    return text;
}
}  // namespace

TEST_SUITE("tokenizer") {
    TEST_CASE("wordpiece: greedy longest match first, with continuation pieces") {
        const auto tok = load(tiny_wordpiece_json());
        CHECK(tok->encode("Unaffable!") == Ids{5, 6, 7, 14});
        CHECK(tok->encode("affable") == Ids{8, 7});
        CHECK(tok->encode("unable") == Ids{5, 7});
        CHECK(tok->encode("HELLO, World.") == Ids{10, 12, 11, 13});
        CHECK(tok->encode("Héllo") == Ids{10});  // accents stripped, as lowercase is on
        CHECK(tok->encode("isn't") == Ids{16, 15, 17});
        CHECK(tok->encode("中文") == Ids{19, 1});
        CHECK(tok->encode("").empty());
        CHECK(tok->encode(" \t\n　 ").empty());
        CHECK(tok->vocab_size() == static_cast<int32_t>(std::size(kPieces)));
        CHECK(tok->token_text(6) == "##aff");
        CHECK(tok->is_special(1));
        CHECK_FALSE(tok->is_special(5));
    }

    TEST_CASE("wordpiece: a word with no whole split becomes one [UNK], with nothing of its start") {
        const auto tok = load(tiny_wordpiece_json());
        CHECK(tok->encode("unaffablex") == Ids{1});
        CHECK(tok->encode("hello unaffablex world") == Ids{10, 1, 11});
        CHECK(tok->encode("[CLS] hello [SEP]") == Ids{2, 10, 3});
        CHECK(tok->encode("[CLS] hello [SEP]", false) == Ids{1, 1, 1, 10, 1, 1, 1});
    }

    TEST_CASE("wordpiece: max_input_chars_per_word counts characters, not bytes") {
        nlohmann::json j = tiny_wordpiece_json();
        j["model"]["max_input_chars_per_word"] = 10;
        auto tok = load(j);
        CHECK(tok->encode("unaffables") == Ids{5, 6, 7, 18});
        CHECK(tok->encode("unaffabless") == Ids{1});
        j["normalizer"] = nullptr;
        tok = load(j);
        CHECK(tok->encode("éééééééééé") == Ids{22, 23, 23, 23, 23, 23, 23, 23, 23, 23});
        CHECK(tok->encode("ééééééééééé") == Ids{1});
    }

    TEST_CASE("wordpiece: the continuing-subword prefix is any string, empty included") {
        nlohmann::json j = tiny_wordpiece_json();
        j["model"]["vocab"] = {{"[UNK]", 0}, {"un", 1}, {"@@aff", 2}, {"@@able", 3}, {"aff", 4}};
        j["model"]["continuing_subword_prefix"] = "@@";
        j["added_tokens"] = nlohmann::json::array();
        j["post_processor"] = nullptr;
        j["decoder"]["prefix"] = "@@";
        auto tok = load(j);
        CHECK(tok->encode("unaffable") == Ids{1, 2, 3});
        CHECK(tok->decode(Ids{1, 2, 3, 4}) == "unaffable aff");

        j["model"]["vocab"] = {{"[UNK]", 0}, {"un", 1}, {"aff", 2}, {"able", 3}, {"unaff", 4}};
        j["model"]["continuing_subword_prefix"] = "";
        j["decoder"]["prefix"] = "";
        tok = load(j);
        CHECK(tok->encode("unaffable") == Ids{4, 3});
        CHECK(tok->encode("affun") == Ids{2, 1});
        CHECK(tok->decode(Ids{1, 2, 3}) == "unaffable");
    }

    TEST_CASE("wordpiece: BertNormalizer cleans, pads CJK, strips accents and lowercases, in that order") {
        using namespace std::string_view_literals;
        const auto defaults = bert_normalizer(true, true, nullptr, true);
        CHECK(normalize(defaults, "Héllo\tWörld\x00\xEF\xBF\xBD​!"sv) == "hello world!");
        CHECK(normalize(defaults, "a中b文c") == "a 中 b 文 c");
        CHECK(normalize(defaults, "İstanbul") == "istanbul");    // İ: NFD, the dot dropped, then lowercase
        CHECK(normalize(defaults, "ΣΑΣ") == "σασ");                  // no final sigma: one character at a time
        CHECK(normalize(defaults, "a\xFF" "b") == "ab");              // invalid UTF-8 reads as U+FFFD
        const auto lowercase = bert_normalizer(false, false, false, true);
        CHECK(normalize(lowercase, "İ") == "i̇");
        CHECK(normalize(lowercase, "Héllo") == "héllo");
        CHECK(normalize(lowercase, "a\xFF" "b") == "a�" "b");
        CHECK(normalize(bert_normalizer(false, false, nullptr, false), "Héllo") == "Héllo");  // strip follows lowercase
        CHECK(normalize(bert_normalizer(true, false, false, false),
                        "a\tb\nc\r\nd e　f\x0bg\u0085h\x7fi­j﻿k\U000e0001lm") ==
              "a b c  d e fghijklm");
        const auto strip = bert_normalizer(false, false, true, false);
        CHECK(normalize(strip, "Ångström ạ́̇ ṩ ̈́") == "Angstrom a s ");
        CHECK(normalize(strip, "ṩ̣") == "s");
        CHECK(normalize(strip, "क़") == "क");
    }

    TEST_CASE("wordpiece: BertPreTokenizer drops whitespace and isolates punctuation") {
        CHECK(pre_tokenize("Hey friend!     How are you?!?") ==
              std::vector<std::string>{"Hey", "friend", "!", "How", "are", "you", "?", "!", "?"});
        CHECK(pre_tokenize("a b　c$d+e~f¿g—h") ==
              std::vector<std::string>{"a", "b", "c", "$", "d", "+", "e", "~", "f", "¿", "g", "—", "h"});
        CHECK(pre_tokenize("  ").empty());
    }

    TEST_CASE("wordpiece: the WordPiece decoder joins words, and cleans up token by token") {
        const nlohmann::json plain = {{"type", "WordPiece"}, {"prefix", "##"}, {"cleanup", false}};
        CHECK(decode_tokens(plain, {"##uelo", "Ara", "##új", "##o", "No", "##guera"}) == "##uelo Araújo Noguera");
        const nlohmann::json cleanup = {{"type", "WordPiece"}, {"prefix", "##"}, {"cleanup", true}};
        const std::vector<std::string> tokens = {"hello", ",", "world", ".", "isn", "'",
                                                 "t",     "do", "not", "'", "s",   "x do not"};
        CHECK(decode_tokens(cleanup, tokens) == "hello, world. isn ' t do not ' s x don't");
        const auto tok = load(tiny_wordpiece_json());
        CHECK(tok->decode(Ids{2, 5, 6, 7, 14, 3}) == "[CLS] unaffable! [SEP]");
        CHECK(tok->decode(Ids{2, 5, 6, 7, 14, 3}, true) == "unaffable!");
        CHECK(stream_decode(*tok, Ids{2, 6, 5, 6, 7, 14, 3}, false) == tok->decode(Ids{2, 6, 5, 6, 7, 14, 3}));
    }

    TEST_CASE("wordpiece: with_special_tokens puts [CLS] and [SEP] around the sequence") {
        const auto tok = load(tiny_wordpiece_json());
        CHECK(tok->with_special_tokens(tok->encode("hello world")) == Ids{2, 10, 11, 3});
        CHECK(tok->with_special_tokens(tok->encode("")) == Ids{2, 3});
    }

    TEST_CASE("wordpiece: truncation and padding in tokenizer.json are not applied") {
        nlohmann::json j = tiny_wordpiece_json();
        j["truncation"] = {{"max_length", 2}, {"strategy", "LongestFirst"}, {"stride", 0}, {"direction", "Right"}};
        j["padding"] = {{"strategy", {{"Fixed", 8}}}, {"direction", "Right"}, {"pad_to_multiple_of", nullptr},
                        {"pad_id", 0}, {"pad_type_id", 0}, {"pad_token", "[PAD]"}};
        const auto tok = load(j);
        CHECK(tok->encode("hello world hello") == Ids{10, 11, 10});
        CHECK(tok->with_special_tokens(tok->encode("hello world hello")) == Ids{2, 10, 11, 10, 3});
    }

    TEST_CASE("wordpiece: what is not implemented is refused by name") {
        nlohmann::json j = tiny_wordpiece_json();
        j["pre_tokenizer"] = {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"use_regex", false}};
        CHECK(load_error(j) == "tokenizer.json model 'WordPiece' after a ByteLevel pre-tokenizer is not supported");

        j = tiny_wordpiece_json();
        j["model"]["unk_token"] = "[NOPE]";
        CHECK(load_error(j) == "tokenizer.json WordPiece unk_token '[NOPE]' is not in the vocab");

        j = tiny_wordpiece_json();
        j["model"].erase("max_input_chars_per_word");
        CHECK(load_error(j) == "tokenizer.json model: field 'max_input_chars_per_word' must be a non-negative integer");

        j = tiny_wordpiece_json();
        j["normalizer"].erase("lowercase");
        CHECK(load_error(j) == "tokenizer.json normalizer: field 'lowercase' must be a boolean");

        j = tiny_wordpiece_json();
        j["decoder"].erase("cleanup");
        CHECK(load_error(j) == "tokenizer.json decoder: field 'cleanup' must be a boolean");

        j = tiny_wordpiece_json();
        j["post_processor"] = {{"type", "BertProcessing"}, {"sep", {"[SEP]", 3}}, {"cls", {"[CLS]", 2}}};
        CHECK(load_error(j) == "tokenizer.json post_processor 'BertProcessing' is not supported");
    }

    TEST_CASE("wordpiece: BERT normalizer and pre-tokenizer match Hugging Face on every code point") {
        const nlohmann::json& components = fixture()["components"];
        const auto block = static_cast<char32_t>(components["block"].get<uint32_t>());
        const auto pre_tokenizer = xs::tok::make_pre_tokenizer({{"type", "BertPreTokenizer"}});
        for (const auto& [name, normalizer] : components["normalizers"].items()) {
            const auto bert = xs::tok::make_normalizer(normalizer["config"]);
            const auto& digests = normalizer["digests"];
            REQUIRE(digests.size() * block == 0x110000);
            for (size_t b = 0; b < digests.size(); ++b) {
                CAPTURE(name);
                CAPTURE(b * block);
                const std::string text = block_text(static_cast<char32_t>(b * block), block, " ", true);
                CHECK(fnv1a64(bert->normalize(text)) == digests[b].get<std::string>());
            }
        }
        const auto& digests = components["pre_tokenizer"];
        REQUIRE(digests.size() * block == 0x110000);
        for (size_t b = 0; b < digests.size(); ++b) {
            CAPTURE(b * block);
            const std::string text = block_text(static_cast<char32_t>(b * block), block, "a", false);
            std::string joined;
            for (const std::string_view piece : pre_tokenizer->split({text})) {
                if (!joined.empty()) joined += '\n';
                joined += piece;
            }
            CHECK(fnv1a64(joined) == digests[b].get<std::string>());
        }
    }
}

TEST_SUITE("tokenizer_parity") {
    TEST_CASE("tokenizer parity: WordPiece encode, special tokens, decode and stream decode match Hugging Face") {
        const auto dirs = wordpiece_dirs();
        if (dirs.empty()) return;
        const nlohmann::json& models = fixture()["models"];
        for (const auto& dir : dirs) {
            const std::string name = dir.filename().string();
            CAPTURE(name);
            REQUIRE_MESSAGE(models.contains(name), "XS_WORDPIECE_DIRS: no WordPiece cases for a directory named '"
                                                       << name << "'");
            const nlohmann::json* f = &models[name];
            if (f->contains("same_as")) f = &models[(*f)["same_as"].get<std::string>()];
            const auto tok = xs::tok::Tokenizer::load(dir);
            CHECK(tok->vocab_size() == (*f)["vocab_size"].get<int32_t>());
            CHECK_FALSE(tok->bos_id().has_value());
            CHECK_FALSE(tok->eos_id().has_value());
            for (const int32_t id : (*f)["special_ids"].get<Ids>()) CHECK(tok->is_special(id));
            REQUIRE((*f)["encode_cases"].size() >= 480);
            for (const auto& c : (*f)["encode_cases"]) {
                CAPTURE(c["name"].get<std::string>());
                const Ids ids = tok->encode(c["text"].get<std::string>(), c["parse_special"].get<bool>());
                CHECK(ids == c["ids"].get<Ids>());
                const Ids expected = c["ids_with_special"].get<Ids>();
                CHECK(tok->with_special_tokens(ids) == expected);
                CHECK(tok->decode(expected) == c["decoded"].get<std::string>());
                CHECK(tok->decode(expected, true) == c["decoded_skip_special"].get<std::string>());
                CHECK(stream_decode(*tok, expected, false) == c["decoded"].get<std::string>());
                CHECK(stream_decode(*tok, expected, true) == c["decoded_skip_special"].get<std::string>());
            }
            for (const auto& c : (*f)["decode_cases"]) {
                CAPTURE(c["name"].get<std::string>());
                const Ids ids = c["ids"].get<Ids>();
                CHECK(tok->decode(ids) == c["decoded"].get<std::string>());
                CHECK(tok->decode(ids, true) == c["decoded_skip_special"].get<std::string>());
                CHECK(stream_decode(*tok, ids, false) == c["decoded"].get<std::string>());
            }
        }
    }
}
