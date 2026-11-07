// Tokenizer unit tests on a hand-written tokenizer.json (tests/fixtures/tokenizer/tiny). Expected ids and texts were
// produced by Hugging Face tokenizers on the same file.

#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tokenizer/tokenizer.h"
#include "util/error.h"

namespace {
using Ids = std::vector<int32_t>;

std::filesystem::path tiny_dir() { return std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/tiny"; }

nlohmann::json tiny_json() {
    std::ifstream in(tiny_dir() / "tokenizer.json");
    return nlohmann::json::parse(in);
}

// A scratch model directory holding one tokenizer.json, removed on destruction.
class ScratchModelDir {
public:
    explicit ScratchModelDir(const nlohmann::json& tokenizer_json)
        : path_(std::filesystem::temp_directory_path() /
                ("xs_tok_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)))) {
        std::filesystem::create_directories(path_);
        std::ofstream(path_ / "tokenizer.json") << tokenizer_json.dump();
    }
    ~ScratchModelDir() { std::filesystem::remove_all(path_); }
    ScratchModelDir(const ScratchModelDir&) = delete;
    ScratchModelDir& operator=(const ScratchModelDir&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::string load_error(const nlohmann::json& tokenizer_json) {
    const ScratchModelDir dir(tokenizer_json);
    try {
        xs::tok::Tokenizer::load(dir.path());
    } catch (const xs::Error& e) {
        return e.what();
    }
    return "loaded without error";
}

std::string stream_decode(const xs::tok::Tokenizer& tok, const Ids& ids, bool skip_special) {
    xs::tok::StreamDecoder stream(tok, skip_special);
    std::string text;
    for (const int32_t id : ids) text += stream.push(id);
    return text + stream.flush();
}
}  // namespace

TEST_SUITE("tokenizer") {
    TEST_CASE("tokenizer: BPE merges apply lowest rank first") {
        const auto tok = xs::tok::Tokenizer::load(tiny_dir());
        CHECK(tok->encode("hello world") == Ids{21, 26});
        CHECK(tok->encode("hellohello") == Ids{21, 21});
        CHECK(tok->encode("helo") == Ids{18, 27});
        CHECK(tok->encode("lol") == Ids{27, 7});
        CHECK(tok->encode("  ") == Ids{30});
        CHECK(tok->encode("").empty());
    }

    TEST_CASE("tokenizer: byte fallback, and fused unk when bytes are missing") {
        const auto tok = xs::tok::Tokenizer::load(tiny_dir());
        CHECK(tok->encode("é€") == Ids{13, 14, 15, 16, 17});
        CHECK(tok->encode("xyz") == Ids{0});
        CHECK(tok->encode("é€xy") == Ids{13, 14, 15, 16, 17, 0});
        // Hugging Face emits a pending unk only at the next in-vocab character or the end, after any byte tokens.
        CHECK(tok->encode("éxé") == Ids{13, 14, 13, 14, 0});
    }

    TEST_CASE("tokenizer: added tokens honor parse_special, lstrip and rstrip") {
        const auto tok = xs::tok::Tokenizer::load(tiny_dir());
        CHECK(tok->encode("hello<s>world</s>") == Ids{21, 1, 9, 23, 7, 11, 2});
        CHECK(tok->encode("hello<s>world</s>", false) == Ids{21, 0, 9, 23, 7, 11, 0});
        CHECK(tok->encode(" hello  <mark>  world") == Ids{29, 3, 9, 23, 7, 11});
        CHECK(tok->encode(" hello  <mark>  world", false) == Ids{29, 3, 9, 23, 7, 11});
    }

    TEST_CASE("tokenizer: normalized added tokens match the normalized text") {
        nlohmann::json j = tiny_json();
        j["added_tokens"].push_back({{"id", 31}, {"content", "▁hel"}, {"normalized", true}, {"special", false}});
        const ScratchModelDir dir(j);
        const auto tok = xs::tok::Tokenizer::load(dir.path());
        CHECK(tok->encode(" hello") == Ids{31, 27});
        CHECK(tok->encode("a hello") == Ids{0, 31, 27});
        CHECK(tok->vocab_size() == 32);
    }

    TEST_CASE("tokenizer: invalid UTF-8 input reaches byte fallback or unk instead of being dropped") {
        const auto tok = xs::tok::Tokenizer::load(tiny_dir());
        CHECK(tok->encode("h\xC3h") == Ids{5, 13, 5});
        CHECK(tok->encode("h\xFFh") == Ids{5, 0, 5});
    }

    TEST_CASE("tokenizer: decoder chain, byte runs and skip_special") {
        const auto tok = xs::tok::Tokenizer::load(tiny_dir());
        CHECK(tok->decode(Ids{29, 3, 9, 23, 7, 11}) == "hello<mark>world");  // Strip drops the leading space
        CHECK(tok->decode(Ids{21, 1, 9, 23, 7, 11, 2}) == "hello<s>world</s>");
        CHECK(tok->decode(Ids{21, 1, 9, 23, 7, 11, 2}, true) == "helloworld");
        CHECK(tok->decode(Ids{15, 16, 17}) == "€");
        // An invalid byte run decodes to one U+FFFD per byte.
        CHECK(tok->decode(Ids{13, 14, 15}) == "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD");
        CHECK(tok->decode(Ids{13, 1, 14}) == "\xEF\xBF\xBD<s>\xEF\xBF\xBD");
        CHECK(tok->decode(Ids{13, 1, 14}, true) == "é");
        CHECK(tok->decode(Ids{4, 5}) == "h");
    }

    TEST_CASE("tokenizer: vocab queries and bos/eos from tokenizer_config.json") {
        const auto tok = xs::tok::Tokenizer::load(tiny_dir());
        CHECK(tok->vocab_size() == 31);
        CHECK(tok->bos_id() == 1);
        CHECK(tok->eos_id() == 2);
        CHECK(tok->is_special(1));
        CHECK_FALSE(tok->is_special(3));
        CHECK(tok->token_text(29) == "▁hello");
        CHECK(tok->token_id("<mark>") == 3);
        CHECK(tok->token_id("▁world") == 26);
        CHECK_FALSE(tok->token_id("nope").has_value());
        CHECK_THROWS_AS(tok->token_text(31), xs::Error);
    }

    TEST_CASE("tokenizer: stream decoding matches full decoding") {
        const auto tok = xs::tok::Tokenizer::load(tiny_dir());
        for (const std::string text : {"hello world", "é€é€xyz", " hello  <mark>  world", "hello<s>é</s>"}) {
            const Ids ids = tok->encode(text);
            CHECK(stream_decode(*tok, ids, false) == tok->decode(ids));
            CHECK(stream_decode(*tok, ids, true) == tok->decode(ids, true));
        }
    }

    TEST_CASE("tokenizer: unsupported components are refused by name") {
        nlohmann::json j = tiny_json();
        j["normalizer"] = {{"type", "NFKC"}};
        CHECK(load_error(j) == "tokenizer.json normalizer 'NFKC' is not supported");

        j = tiny_json();
        j["pre_tokenizer"] = {{"type", "Metaspace"}};
        CHECK(load_error(j) == "tokenizer.json pre_tokenizer 'Metaspace' is not supported");

        j = tiny_json();
        j["decoder"]["decoders"][1] = {{"type", "WordPiece"}};
        CHECK(load_error(j) == "tokenizer.json decoder 'WordPiece' is not supported");

        j = tiny_json();
        j["model"]["type"] = "Unigram";
        CHECK(load_error(j) == "tokenizer.json model 'Unigram' is not supported");

        j = tiny_json();
        j["pre_tokenizer"]["pattern"] = {{"Regex", "\\s+"}};
        CHECK(load_error(j) == "tokenizer.json pre_tokenizer 'Split' Regex '\\s+' is not supported");

        j = tiny_json();
        j["model"]["dropout"] = 0.1;
        CHECK(load_error(j) == "tokenizer.json BPE option 'dropout' is not supported");

        j = tiny_json();
        j["post_processor"] = {{"type", "RobertaProcessing"}};
        CHECK(load_error(j) == "tokenizer.json post_processor 'RobertaProcessing' is not supported");
    }
}
