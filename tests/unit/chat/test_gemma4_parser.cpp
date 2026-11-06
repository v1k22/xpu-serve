#include <doctest/doctest.h>

#include <random>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat/gemma4_args.h"
#include "chat/output_parser.h"
#include "util/error.h"

using nlohmann::json;
using xs::chat::make_output_parser;
using xs::chat::OutputParser;
using xs::chat::ParsedDelta;

namespace {

const std::string Q = "<|\"|>";  // Gemma's string delimiter token

// Everything a parser produced over a whole response; ids are compared separately since they are random.
struct Collected {
    std::string content;
    std::string reasoning;
    std::vector<std::pair<std::string, json>> calls;  // name, parsed arguments
    std::vector<std::string> ids;

    void add(const ParsedDelta& d) {
        content += d.content;
        reasoning += d.reasoning;
        for (const auto& c : d.tool_calls) {
            calls.emplace_back(c.name, json::parse(c.arguments));
            ids.push_back(c.id);
        }
    }
    bool operator==(const Collected& o) const {
        return content == o.content && reasoning == o.reasoning && calls == o.calls;
    }
};

Collected parse_chunks(const std::vector<std::string>& chunks) {
    auto parser = make_output_parser("gemma4");
    Collected out;
    for (const auto& chunk : chunks)
        out.add(parser->feed(chunk));
    out.add(parser->finish());
    return out;
}

Collected parse(const std::string& text) {
    return parse_chunks({text});
}

json args(std::string_view gemma) {
    const auto converted = xs::chat::gemma4::args_to_json(gemma);
    REQUIRE_MESSAGE(converted.has_value(), "did not convert: ", std::string(gemma));
    return json::parse(*converted);
}

const std::string kRealistic =
    "<|channel>thought\nThe user wants the weather 🌦 in Paris.\nI should call the tool.\n<channel|>"
    "Let me check — un instant.\n"
    "<|tool_call>call:get_weather{city:" +
    Q + "Paris, \"FR\" {x}" + Q +
    ",days:3,precise:true,ratio:-0.5e-3,"
    "note:null,opts:{nested:{deep:[1," +
    Q + "two" + Q +
    ",{x:false}]}},tags:[]}<tool_call|>"
    "<|tool_call>call:list_files{}<tool_call|>";

Collected realistic_expected() {
    Collected c;
    c.reasoning = "The user wants the weather 🌦 in Paris.\nI should call the tool.";
    c.content = "Let me check — un instant.\n";
    c.calls = {{"get_weather", json::parse(R"({"city":"Paris, \"FR\" {x}","days":3,"precise":true,"ratio":-0.5e-3,
                  "note":null,"opts":{"nested":{"deep":[1,"two",{"x":false}]}},"tags":[]})")},
               {"list_files", json::object()}};
    return c;
}

}  // namespace

TEST_SUITE("chat gemma4 args") {
    TEST_CASE("every value type converts to JSON") {
        CHECK(args("{s:" + Q + "text" + Q + ",i:-12,f:1.5e+20,t:true,f2:false,n:null}") ==
              json::parse(R"({"s":"text","i":-12,"f":1.5e20,"t":true,"f2":false,"n":null})"));
        CHECK(args("{o:{a:{b:[]}},a:[[1,2],{k:" + Q + "v" + Q + "}],e:{}}") ==
              json::parse(R"({"o":{"a":{"b":[]}},"a":[[1,2],{"k":"v"}],"e":{}})"));
        CHECK(args("{}") == json::object());
    }

    TEST_CASE("string contents are kept verbatim and escaped") {
        CHECK(args("{s:" + Q + "a \"q\" \\ b\n\tc, {d}: [e]" + Q + "}")["s"] == "a \"q\" \\ b\n\tc, {d}: [e]");
        CHECK(args("{" + Q + "Ключ" + Q + ":" + Q + "日本 🎉" + Q + "}") == json::parse(R"({"Ключ":"日本 🎉"})"));
    }

    TEST_CASE("JSON spellings and whitespace are accepted") {
        CHECK(args(R"( { "a" : "x\"y" , "b" : [ 1 , 2 ] } )") == json::parse(R"({"a":"x\"y","b":[1,2]})"));
    }

    TEST_CASE("bare non-literals become strings") {
        CHECK(args("{cmd:ls -la ,v:1.2.3,x:inf}") == json::parse(R"({"cmd":"ls -la","v":"1.2.3","x":"inf"})"));
    }

    TEST_CASE("invalid UTF-8 is replaced, never invalid JSON") {
        CHECK(args("{s:" + Q + "a\xff" + "b" + Q + "}")["s"] ==
              "a\xEF\xBF\xBD"
              "b");
    }

    TEST_CASE("malformed objects are rejected") {
        for (const std::string& bad :
             std::vector<std::string>{"", "[]", "{", "{a:1", "{a:1}}", "{a:1} x", "{a}", "{:1}", "{a:}", "{a:1,}",
                                      "{a:[1,}", "{a:" + Q + "unterminated}", "{a:\"unterminated}"}) {
            CAPTURE(bad);
            CHECK_FALSE(xs::chat::gemma4::args_to_json(bad).has_value());
        }
        CHECK_FALSE(xs::chat::gemma4::args_to_json(std::string(100, '{') + std::string(100, '}')).has_value());
    }
}

TEST_SUITE("chat gemma4 parser") {
    TEST_CASE("plain text is content") {
        const Collected c = parse("Hello, world! 1 < 2 and a\nnewline.");
        CHECK(c.content == "Hello, world! 1 < 2 and a\nnewline.");
        CHECK(c.reasoning.empty());
        CHECK(c.calls.empty());
    }

    TEST_CASE("thinking then answer") {
        const Collected c = parse("<|channel>thought\nStep 1.\nStep 2.\n<channel|>The answer is 4.");
        CHECK(c.reasoning == "Step 1.\nStep 2.");
        CHECK(c.content == "The answer is 4.");
    }

    TEST_CASE("empty thinking channel") {
        const Collected c = parse("<|channel>thought\n<channel|>Hi");
        CHECK(c.reasoning.empty());
        CHECK(c.content == "Hi");
    }

    TEST_CASE("realistic output with reasoning, content and tool calls") {
        const Collected c = parse(kRealistic);
        CHECK(c == realistic_expected());
        REQUIRE(c.ids.size() == 2);
        CHECK(c.ids[0].starts_with("call_"));
        CHECK(c.ids[0] != c.ids[1]);
    }

    TEST_CASE("unicode passes through every channel") {
        const Collected c = parse("<|channel>thought\n考える 🤔\n<channel|>Ответ: ✓<|tool_call>call:say{text:" + Q +
                                  "naïve — 日本" + Q + "}<tool_call|>");
        CHECK(c.reasoning == "考える 🤔");
        CHECK(c.content == "Ответ: ✓");
        REQUIRE(c.calls.size() == 1);
        CHECK(c.calls[0].second["text"] == "naïve — 日本");
    }

    TEST_CASE("every single split point equals feeding at once") {
        const Collected whole = parse(kRealistic);
        for (size_t i = 0; i <= kRealistic.size(); ++i) {
            CAPTURE(i);
            CHECK(parse_chunks({kRealistic.substr(0, i), kRealistic.substr(i)}) == whole);
        }
    }

    TEST_CASE("byte-by-byte and random chunking equal feeding at once") {
        const Collected whole = parse(kRealistic);
        std::vector<std::string> bytes;
        for (const char ch : kRealistic)
            bytes.emplace_back(1, ch);
        CHECK(parse_chunks(bytes) == whole);

        std::mt19937 rng(1234);
        for (int trial = 0; trial < 300; ++trial) {
            std::vector<std::string> chunks;
            for (size_t pos = 0; pos < kRealistic.size();) {
                const size_t len = std::uniform_int_distribution<size_t>(1, 16)(rng);
                chunks.push_back(kRealistic.substr(pos, len));
                pos += len;
            }
            CAPTURE(trial);
            CHECK(parse_chunks(chunks) == whole);
        }
    }

    TEST_CASE("holds back only a possible marker prefix") {
        auto parser = make_output_parser("gemma4");
        CHECK(parser->feed("Hello <|chan").content == "Hello ");
        CHECK(parser->feed("nel>thought\nhm").reasoning == "hm");
        CHECK(parser->feed("m\n").reasoning == "m");  // the "\n" may start "\n<channel|>"
        const ParsedDelta d = parser->feed("ok\n<channel|>Done <");
        CHECK(d.reasoning == "\nok");
        CHECK(d.content == "Done ");
        CHECK(parser->feed("x").content == "<x");
    }

    TEST_CASE("content is streamed without waiting for the end") {
        auto parser = make_output_parser("gemma4");
        CHECK(parser->feed("Line one\n").content == "Line one\n");
        CHECK(parser->feed("<|tool_call>call:f{a:1}").content.empty());
        CHECK(parser->feed("<tool_call|>").tool_calls.size() == 1);
    }

    TEST_CASE("an unterminated partial marker at the end is plain text") {
        CHECK(parse("Hi <|tool_ca").content == "Hi <|tool_ca");
        CHECK(parse("<|channel>").reasoning.empty());
        const Collected c = parse("<|channel>thought\nhmm\n<chan");
        CHECK(c.reasoning == "hmm\n<chan");  // still inside the reasoning channel
        CHECK(c.content.empty());
    }

    TEST_CASE("unterminated reasoning stays reasoning") {
        const Collected c = parse("<|channel>thought\nlong thoughts");
        CHECK(c.reasoning == "long thoughts");
        CHECK(c.content.empty());
    }

    TEST_CASE("an unclosed tool call is a call when complete, else content") {
        const Collected complete = parse("<|tool_call>call:f{a:" + Q + "x" + Q + "}");
        REQUIRE(complete.calls.size() == 1);
        CHECK(complete.calls[0].second == json::parse(R"({"a":"x"})"));
        CHECK(complete.content.empty());

        const Collected cut = parse("Sure.<|tool_call>call:f{a:" + Q + "half");
        CHECK(cut.calls.empty());
        CHECK(cut.content == "Sure.call:f{a:" + Q + "half");
    }

    TEST_CASE("malformed calls become content, never lost") {
        CHECK(parse("<|tool_call>not a call<tool_call|>after").content == "not a callafter");
        CHECK(parse("<|tool_call>call:{a:1}<tool_call|>").content == "call:{a:1}");
        CHECK(parse("<|tool_call>call:f g{}<tool_call|>").content == "call:f g{}");
        CHECK(parse("<|tool_call>call:f{a:1} junk<tool_call|>").content == "call:f{a:1} junk");
        CHECK(parse("<|tool_call>call:f{a:1}<tool_call|>").calls.size() == 1);
    }

    TEST_CASE("stray closing markers are dropped from content") {
        CHECK(parse("a<channel|>b<tool_call|>c").content == "abc");
    }

    TEST_CASE("a tool call inside reasoning ends the reasoning") {
        const Collected c = parse("<|channel>thought\nI'll call it<|tool_call>call:f{}<tool_call|>");
        CHECK(c.reasoning == "I'll call it");
        CHECK(c.calls.size() == 1);
    }

    TEST_CASE("ids are unique across many calls") {
        std::string text;
        for (int i = 0; i < 25; ++i)
            text += "<|tool_call>call:f{i:" + std::to_string(i) + "}<tool_call|>";
        const Collected c = parse(text);
        REQUIRE(c.calls.size() == 25);
        CHECK(c.calls[24].second["i"] == 24);
        CHECK(std::set<std::string>(c.ids.begin(), c.ids.end()).size() == 25);
    }

    TEST_CASE("finish with nothing pending is empty") {
        auto parser = make_output_parser("gemma4");
        CHECK(parser->finish().empty());
    }
}

TEST_SUITE("chat output parser") {
    TEST_CASE("plain dialect passes everything through as content") {
        auto parser = make_output_parser("plain");
        CHECK(parser->feed("<|channel>x").content == "<|channel>x");
        CHECK(parser->finish().empty());
    }

    TEST_CASE("unknown dialect is refused by name") {
        CHECK_THROWS_WITH_AS(make_output_parser("nope"), doctest::Contains("known: gemma4, plain"), xs::Error);
    }
}
