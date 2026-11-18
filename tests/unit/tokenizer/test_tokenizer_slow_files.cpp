// Tokenizers read from transformers' slow files (vocab.json, merges.txt, tokenizer_config.json) without a
// tokenizer.json: a hand-written Qwen2Tokenizer the tests write, and Qwen/Qwen3-ASR-1.7B's against Hugging Face
// (fixture from tests/parity/tokenizer/gen_qwen3_asr_cases.py), which needs the checkpoint's files in XS_QWEN3_ASR_DIR
// and skips without them.

#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "parity_cases.h"
#include "scratch_model_dir.h"
#include "test_support.h"
#include "tokenizer/byte_level.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"

namespace {
using Ids = std::vector<int32_t>;
using Files = std::map<std::string, std::string>;
using nlohmann::json;
using xs::test::ScratchModelDir;

std::string chars(std::string_view bytes) {
    std::string out;
    for (const char b : bytes) out += xs::tok::byte_level::byte_piece(static_cast<uint8_t>(b));
    return out;
}

json added(const std::string& content, bool special, bool lstrip = false) {
    return {{"content", content}, {"lstrip", lstrip},     {"normalized", false},
            {"rstrip", false},    {"single_word", false}, {"special", special}};
}

// A Qwen2Tokenizer in miniature: ids 0-255 are the byte characters; merges make 256 "he", 257 "ll", 258 "hell",
// 259 "hello", 260 " w", and 261 / 262 "x" + U+0301 (a mark Qwen2's regex splits from the letter before it). Added
// tokens from 263; <|im_end|> is written not special but is the eos token, "hello" is named the audio token.
Files tiny_files(const std::string& line_end = "\n") {
    json vocab = json::object();
    for (int b = 0; b < 256; ++b) vocab[chars(std::string(1, static_cast<char>(b)))] = b;
    const std::vector<std::pair<std::string, std::string>> merges = {
        {"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o"}, {" ", "w"}, {"x", "\xCC"}, {"x\xCC", "\x81"}};
    std::string merges_txt = "#version: 0.2" + line_end;
    int next_id = 256;
    for (const auto& [left, right] : merges) {
        merges_txt += chars(left) + " " + chars(right) + line_end;
        vocab[chars(left + right)] = next_id++;
    }
    const json config = {
        {"tokenizer_class", "Qwen2Tokenizer"},
        {"added_tokens_decoder",
         {{"263", added("<|endoftext|>", true)},
          {"264", added("<|im_start|>", true)},
          {"265", added("<|im_end|>", false)},
          {"266", added("<think>", false)},
          {"267", added("<tag>", false, true)}}},
        {"bos_token", nullptr},
        {"eos_token", "<|im_end|>"},
        {"pad_token", "<|endoftext|>"},
        {"unk_token", nullptr},
        {"audio_token", "hello"},
        {"add_prefix_space", false},
        {"errors", "replace"},
    };
    return {{"vocab.json", vocab.dump()}, {"merges.txt", merges_txt}, {"tokenizer_config.json", config.dump()}};
}

// The tokenizer_config.json of tiny_files() with `key` set to `value` (null removes it).
Files with_config(Files files, const std::string& key, const json& value) {
    json config = json::parse(files["tokenizer_config.json"]);
    if (value.is_null()) {
        config.erase(key);
    } else {
        config[key] = value;
    }
    files["tokenizer_config.json"] = config.dump();
    return files;
}

std::string load_error(const Files& files) {
    const ScratchModelDir dir(files);
    try {
        xs::tok::Tokenizer::load(dir.path());
    } catch (const xs::Error& e) {
        return e.what();
    }
    return "loaded without error";
}
}  // namespace

TEST_SUITE("tokenizer") {
    TEST_CASE("tokenizer: a Qwen2Tokenizer's slow files load as the tokenizer transformers builds from them") {
        const ScratchModelDir dir(tiny_files());
        const auto tok = xs::tok::Tokenizer::load(dir.path());
        CHECK(tok->vocab_size() == 268);
        CHECK(tok->encode("hell") == Ids{258});
        CHECK(tok->encode(" world") == Ids{260, 'o', 'r', 'l', 'd'});
        CHECK(tok->encode("x\xCC\x81") == Ids{'x', 0xCC, 0x81});  // Qwen2's regex: the mark is a piece of its own
        CHECK(tok->decode(Ids{'x', 0xCC, 0x81, 258, 'o'}) == "x\xCC\x81hello");

        CHECK(tok->eos_id() == 265);
        CHECK_FALSE(tok->bos_id().has_value());
        CHECK(tok->encode("a<|im_end|>") == Ids{'a', 265});
        const Ids spelled = tok->encode("a<|im_end|>", false);
        CHECK(std::find(spelled.begin(), spelled.end(), 265) == spelled.end());
        CHECK(tok->decode(spelled) == "a<|im_end|>");
        CHECK(tok->encode("<think>x", false) == Ids{266, 'x'});  // added but not special: matched either way
        CHECK(tok->encode("a  <tag>b") == Ids{'a', 267, 'b'});  // lstrip

        // Special: the decoder's, the named eos token it lists as not special, the audio token that is a vocab piece.
        for (const int32_t id : {263, 264, 265, 259}) CHECK(tok->is_special(id));
        for (const int32_t id : {258, 266, 267}) CHECK_FALSE(tok->is_special(id));
        CHECK(tok->decode(Ids{'a', 265, 266, 259}) == "a<|im_end|><think>hello");
        CHECK(tok->decode(Ids{'a', 265, 266, 259}, true) == "a<think>");
    }

    TEST_CASE("tokenizer: slow files: CRLF merges, Qwen2TokenizerFast, and the class's eos when none is named") {
        const auto ids_of = [](const Files& files) {
            const ScratchModelDir dir(files);
            return xs::tok::Tokenizer::load(dir.path())->encode("hello world<|im_end|>");
        };
        const Ids expected = {259, 260, 'o', 'r', 'l', 'd', 265};
        CHECK(ids_of(tiny_files()) == expected);
        CHECK(ids_of(tiny_files("\r\n")) == expected);
        CHECK(ids_of(with_config(tiny_files(), "tokenizer_class", "Qwen2TokenizerFast")) == expected);

        const ScratchModelDir dir(with_config(tiny_files(), "eos_token", nullptr));
        const auto tok = xs::tok::Tokenizer::load(dir.path());
        CHECK(tok->eos_id() == 263);  // Qwen2Tokenizer's own eos_token, <|endoftext|>
        CHECK_FALSE(tok->is_special(265));
    }

    TEST_CASE("tokenizer: slow files refuse what transformers would read otherwise") {
        CHECK(load_error(with_config(tiny_files(), "tokenizer_class", "LlamaTokenizer"))
                  .ends_with(" has no tokenizer.json, and its tokenizer_class LlamaTokenizer is not one xpu-serve "
                             "builds from vocab.json and merges.txt (Qwen2Tokenizer is)"));
        CHECK(load_error(with_config(tiny_files(), "tokenizer_class", nullptr))
                  .find("its tokenizer_class (none in a tokenizer_config.json) is not one") != std::string::npos);

        Files files = tiny_files();
        files.erase("merges.txt");
        CHECK(load_error(files).ends_with(" has no tokenizer.json, and no merges.txt for its Qwen2Tokenizer"));

        files = tiny_files();
        files["merges.txt"] = "#version: 0.2\nh e\n\nl l\n";
        CHECK(load_error(files).ends_with("merges.txt' line 3 is not two pieces separated by one space"));
        files["merges.txt"] = "h e x\n";
        CHECK(load_error(files).ends_with("merges.txt' line 1 is not two pieces separated by one space"));
        files["merges.txt"] = "h q\n";
        CHECK(load_error(files).find("merge ['h', 'q'] uses a piece that is not in the vocab") != std::string::npos);

        json decoder = json::parse(tiny_files()["tokenizer_config.json"])["added_tokens_decoder"];
        decoder.erase("264");  // <|im_start|> gone: transformers gives <|im_end|> id 264
        CHECK(load_error(with_config(tiny_files(), "added_tokens_decoder", decoder))
                  .ends_with("added_tokens_decoder gives '<|im_end|>' id 265, where transformers adds it as id 264"));
        CHECK(load_error(with_config(tiny_files(), "image_token", "<|image_pad|>"))
                  .ends_with("names '<|image_pad|>' a special token, which neither its added_tokens_decoder nor "
                             "vocab.json has"));
        CHECK(load_error(with_config(tiny_files(), "add_bos_token", true))
                  .ends_with("sets add_bos_token, which a Qwen2Tokenizer built from vocab.json and merges.txt does not "
                             "support"));
    }
}

TEST_SUITE("tokenizer_parity") {
    TEST_CASE("tokenizer parity: Qwen3-ASR's slow files encode, decode and stream decode as Hugging Face") {
        const auto dir = xs::test::qwen3_asr_dir();
        if (!dir) return;
        REQUIRE_MESSAGE(!std::filesystem::exists(*dir / "tokenizer.json"), "XS_QWEN3_ASR_DIR holds a tokenizer.json");
        std::ifstream in(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/qwen3_asr_cases.json");
        REQUIRE(in.good());
        const json fixture = json::parse(in);
        REQUIRE(fixture["encode_cases"].size() >= 50);

        const auto start = std::chrono::steady_clock::now();
        const auto tok = xs::tok::Tokenizer::load(*dir);
        MESSAGE("Qwen3-ASR slow files load: "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s");
        xs::test::check_tokenizer_cases(*tok, fixture);
        REQUIRE(fixture["added_tokens"].size() == 62);
        for (const json& token : fixture["added_tokens"]) {
            const int32_t id = token[0].get<int32_t>();
            const std::string content = token[1].get<std::string>();
            CAPTURE(content);
            CHECK(tok->token_text(id) == content);
            CHECK(tok->token_id(content) == id);
            CHECK(tok->is_special(id) == token[2].get<bool>());
        }
    }
}
