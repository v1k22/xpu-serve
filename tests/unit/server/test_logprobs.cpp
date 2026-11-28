// Generated tokens' log-probabilities as text (server/logprobs.h): a token that leaves a character unfinished reads
// as "", the one that finishes it carries the whole character, alternatives read after the same tokens; and the
// two JSON shapes, chat's and the text completions' (alternatives that read alike merged, offsets in characters).
#include <doctest/doctest.h>

#include <limits>

#include "server/logprobs.h"
#include "server/openai_format.h"
#include "unit/server/fake_service.h"

using nlohmann::json;
using xs::TokenLogprobs;
using xs::server::LogprobEntry;
using xs::server::LogprobTexts;

TEST_SUITE("server.logprobs") {
    TEST_CASE("token texts: a character split across tokens is carried by the token that finishes it") {
        const xs::test::ByteTokenizer tokenizer;  // a token a byte
        LogprobTexts texts(tokenizer);
        const auto entry = [&](int32_t byte, std::vector<std::pair<int32_t, float>> top = {}) {
            return texts.entry(byte, TokenLogprobs{-0.5f, std::move(top)});
        };
        CHECK(entry('a').token.text == "a");
        // The euro sign, E2 82 AC: the first two bytes read as "", the third as the whole character.
        const LogprobEntry first = entry(0xE2);
        CHECK(first.token.text == "");
        CHECK(first.token.bytes == "\xE2");
        CHECK(entry(0x82).token.text == "");
        const LogprobEntry third = entry(0xAC, {{0xAC, -0.5f}, {0xAD, -2.0f}, {'x', -3.0f}});
        CHECK(third.token.text == "\xE2\x82\xAC");
        CHECK(third.token.bytes == "\xAC");
        // Alternatives read after the same tokens: AD finishes another character, 'x' cannot follow (U+FFFD).
        CHECK(third.top[0].text == "\xE2\x82\xAC");
        CHECK(third.top[1].text == "\xE2\x82\xAD");
        CHECK(third.top[2].text == "\xEF\xBF\xBD" "x");  // one U+FFFD for the unfinished two bytes
        CHECK(third.top[1].logprob == -2.0f);
        CHECK(entry('b').token.text == "b");  // whole again
        // A lone continuation byte is not a character's start: U+FFFD at once.
        CHECK(entry(0x82).token.text == "\xEF\xBF\xBD");
    }

    TEST_CASE("logprob JSON: chat's entries, the text completions' tokens, alternatives and offsets") {
        const float masked = -std::numeric_limits<float>::infinity();
        std::vector<LogprobEntry> entries(2);
        entries[0].token = {"h\xC3\xA9", "h\xC3\xA9", -0.25f};
        entries[0].top = {{"h\xC3\xA9", "h\xC3\xA9", -0.25f}, {"x", "x", masked}};
        entries[1].token = {"y", "y", -1.0f};
        entries[1].top = {{"z", "z", -0.5f}, {"y", "y", -1.0f}, {"z", "", -0.75f}};
        const json chat = xs::server::chat_logprobs_json(entries);
        REQUIRE(chat["content"].size() == 2);
        CHECK(chat["content"][0]["token"] == "h\xC3\xA9");
        CHECK(chat["content"][0]["bytes"] == json::array({'h', 0xC3, 0xA9}));
        CHECK(chat["content"][0]["top_logprobs"][1]["logprob"] == -9999.0);  // -infinity as vLLM writes it
        CHECK(chat["content"][1]["top_logprobs"].size() == 3);
        int64_t offset = 4;
        const json text = xs::server::text_logprobs_json(entries, offset);
        CHECK(text["tokens"] == json::array({"h\xC3\xA9", "y"}));
        CHECK(text["text_offset"] == json::array({4, 6}));  // "hé" is two characters
        CHECK(offset == 7);
        CHECK(text["top_logprobs"][0].size() == 2);
        CHECK(text["top_logprobs"][1].size() == 2);  // z twice: the likelier kept
        CHECK(text["top_logprobs"][1]["z"] == -0.5);
        CHECK(text["top_logprobs"][1]["y"] == -1.0);
    }
}
