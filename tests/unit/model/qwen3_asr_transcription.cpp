// Qwen3-ASR's transcription format (models/qwen3_asr): forcing a language by code or name, reading the language and
// transcript a reply carries (silence, no header, a header still being written while streaming), and the reference
// package's clean-up of repeated text, against what that package's own functions return.
#include <doctest/doctest.h>

#include "models/qwen3_asr/qwen3_asr.h"
#include "util/error.h"

namespace {

std::unique_ptr<xs::Transcription> format() { return xs::qwen3_asr::make_transcription({}, 13.0); }

}  // namespace

TEST_SUITE("qwen3_asr transcription") {
    TEST_CASE("a language is forced by ISO 639-1 code or by name, any case; others are refused by name") {
        const auto f = format();
        CHECK(f->language_prefix("") == "");
        CHECK(f->language_prefix("en") == "language English<asr_text>");
        CHECK(f->language_prefix("EN") == "language English<asr_text>");
        CHECK(f->language_prefix("english") == "language English<asr_text>");
        CHECK(f->language_prefix("yue") == "language Cantonese<asr_text>");
        CHECK(f->language_prefix("zh") == "language Chinese<asr_text>");
        CHECK_THROWS_WITH_AS(f->language_prefix("klingon"), doctest::Contains("klingon"), xs::Error);
        CHECK_THROWS_AS(f->language_prefix("sw"), xs::Error);  // Swahili: not one of the 30
        // a checkpoint's own list narrows the names
        const auto narrow = xs::qwen3_asr::make_transcription({"English"}, 13.0);
        CHECK_THROWS_AS(narrow->language_prefix("de"), xs::Error);
    }

    TEST_CASE("a reply's language and transcript; silence; no header; a forced language") {
        const auto f = format();
        auto t = f->parse("", "language English<asr_text> Hello there. ", true);
        CHECK(t.language == "English");
        CHECK(t.text == "Hello there.");
        t = f->parse("", "language None<asr_text>", true);
        CHECK(t.language.empty());
        CHECK(t.text.empty());
        t = f->parse("", "language chinese\n<asr_text>你好", true);
        CHECK(t.language == "Chinese");
        CHECK(t.text == "你好");
        t = f->parse("", "Just text, no header.", true);
        CHECK(t.language.empty());
        CHECK(t.text == "Just text, no header.");
        t = f->parse("language German<asr_text>", "Guten Tag.", true);
        CHECK(t.language == "German");
        CHECK(t.text == "Guten Tag.");
    }

    TEST_CASE("streaming: a header being written is held back; the text after it is passed on as it grows") {
        const auto f = format();
        CHECK(f->parse("", "lang", false).text.empty());
        CHECK(f->parse("", "language Eng", false).text.empty());
        CHECK(f->parse("", "language English<asr_text>Hel", false).text == "Hel");
        CHECK(f->parse("", "Hello, no header here", false).text == "Hello, no header here");
        CHECK(f->parse("language English<asr_text>", "Hel", false).text == "Hel");
    }

    TEST_CASE("repeated text is cleaned as the reference package's detect_and_fix_repetitions cleans it") {
        const auto f = format();
        const auto clean = [&](const std::string& reply) {
            return f->parse("language English<asr_text>", reply, true).text;
        };
        // inputs and outputs of qwen_asr.inference.utils.detect_and_fix_repetitions
        CHECK(clean(std::string(25, 'a') + "b") == "ab");
        CHECK(clean(std::string(20, 'a') + "b") == std::string(20, 'a') + "b");
        std::string ab, hello, nihao, na, cat;
        for (int i = 0; i < 25; ++i) ab += "ab";
        for (int i = 0; i < 30; ++i) hello += "hello world ";
        for (int i = 0; i < 30; ++i) nihao += "你好";
        for (int i = 0; i < 21; ++i) na += "na";
        for (int i = 0; i < 19; ++i) cat += "The cat. ";
        CHECK(clean(ab + "c") == "abc");
        CHECK(clean("x" + hello) == "xhello world hello world");  // trimmed first: the last repeat lost its space
        CHECK(clean(nihao + "。") == "你好。");
        CHECK(clean("xyz" + na + " end") == "xyzna end");
        CHECK(clean(std::string(21, 'x') + std::string(21, 'y')) == "xy");
        CHECK(clean(cat + "The end.") == cat + "The end.");  // 19 repeats are left
        CHECK(clean("short text") == "short text");
        CHECK(f->max_reply_tokens() == 4096);
    }
}
