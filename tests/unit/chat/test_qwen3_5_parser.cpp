#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat/output_parser.h"

using nlohmann::json;
using nlohmann::ordered_json;
using xs::chat::make_output_parser;
using xs::chat::ParsedDelta;

namespace {

// Everything a parser produced over a whole response; ids are compared separately since they are random.
struct Collected {
    std::string content;
    std::string reasoning;
    std::vector<std::pair<std::string, ordered_json>> calls;  // name, parsed arguments (key order matters)
    std::vector<std::string> ids;

    void add(const ParsedDelta& d) {
        content += d.content;
        reasoning += d.reasoning;
        for (const auto& c : d.tool_calls) {
            calls.emplace_back(c.name, ordered_json::parse(c.arguments));
            ids.push_back(c.id);
        }
    }
    bool operator==(const Collected& o) const {
        return content == o.content && reasoning == o.reasoning && calls == o.calls;
    }
};

// The prompt Hugging Face renders for a named case of tests/fixtures/chat/<set>_cases.json.
std::string rendered_prompt(const std::string& case_name, const std::string& set = "qwen3_5") {
    std::ifstream in(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/chat" / (set + "_cases.json"));
    for (const json& c : json::parse(in)) {
        if (c["name"] == case_name) return c["expected"].get<std::string>();
    }
    FAIL("no fixture case ", case_name);
    return {};
}

const std::string kThinkingOn = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n";
const std::string kThinkingOff = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

Collected parse_chunks(const std::vector<std::string>& chunks, std::string_view prompt = {}) {
    auto parser = make_output_parser("qwen3_5", xs::chat::prompt_tail(prompt));
    Collected out;
    for (const auto& chunk : chunks)
        out.add(parser->feed(chunk));
    out.add(parser->finish());
    return out;
}

Collected parse(const std::string& text, std::string_view prompt = {}) {
    return parse_chunks({text}, prompt);
}

Collected parse_bytes(const std::string& text, std::string_view prompt = {}) {
    std::vector<std::string> bytes;
    for (const char ch : text)
        bytes.emplace_back(1, ch);
    return parse_chunks(bytes, prompt);
}

// Feeding whole, byte by byte, at every single split point and in random chunks all give the same result.
Collected parse_every_way(const std::string& text, std::string_view prompt = {}) {
    const Collected whole = parse(text, prompt);
    CHECK(parse_bytes(text, prompt) == whole);
    for (size_t i = 0; i <= text.size(); ++i) {
        CAPTURE(i);
        CHECK(parse_chunks({text.substr(0, i), text.substr(i)}, prompt) == whole);
    }
    std::mt19937 rng(1234);
    for (int trial = 0; trial < 100; ++trial) {
        std::vector<std::string> chunks;
        for (size_t pos = 0; pos < text.size();) {
            const size_t len = std::uniform_int_distribution<size_t>(1, 16)(rng);
            chunks.push_back(text.substr(pos, len));
            pos += len;
        }
        CAPTURE(trial);
        CHECK(parse_chunks(chunks, prompt) == whole);
    }
    return whole;
}

const std::string kRealistic =
    "The user wants the weather 🌦 in Paris.\nI should call the tool.\n\n</think>\n\n"
    "Let me check — un instant.\n\n"
    "<tool_call>\n<function=get_weather>\n"
    "<parameter=city>\nParis, \"FR\" <b>{x}</b>\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n"
    "<parameter=precise>\ntrue\n</parameter>\n"
    "<parameter=ratio>\n-0.5e-3\n</parameter>\n"
    "<parameter=note>\nnull\n</parameter>\n"
    "<parameter=opts>\n{\"nested\": {\"deep\": [1, \"two\", {\"x\": false}]}}\n</parameter>\n"
    "<parameter=body>\nline one\n\n  line three\n</parameter>\n"
    "</function>\n</tool_call>\n"
    "<tool_call>\n<function=list_files>\n</function>\n</tool_call>";

Collected realistic_expected() {
    Collected c;
    c.reasoning = "The user wants the weather 🌦 in Paris.\nI should call the tool.";
    c.content = "Let me check — un instant.";
    c.calls = {{"get_weather", ordered_json::parse(R"({"city":"Paris, \"FR\" <b>{x}</b>","days":3,"precise":true,
                  "ratio":-0.5e-3,"note":null,"opts":{"nested":{"deep":[1,"two",{"x":false}]}},
                  "body":"line one\n\n  line three"})")},
               {"list_files", ordered_json::object()}};
    return c;
}

ordered_json value_of(const std::string& value_text) {
    const Collected c = parse("<tool_call>\n<function=f>\n<parameter=v>" + value_text +
                              "</parameter>\n</function>\n"
                              "</tool_call>");
    REQUIRE(c.calls.size() == 1);
    return c.calls[0].second["v"];
}

}  // namespace

TEST_SUITE("chat qwen3_5 parser") {
    TEST_CASE("plain text is content") {
        const Collected c = parse_every_way("Hello, world! 1 < 2, a <b>tag</b>, <thinking> and a\nnewline.");
        CHECK(c.content == "Hello, world! 1 < 2, a <b>tag</b>, <thinking> and a\nnewline.");
        CHECK(c.reasoning.empty());
        CHECK(c.calls.empty());
    }

    TEST_CASE("realistic output with reasoning, preamble and two tool calls") {
        const Collected c = parse_every_way(kRealistic, kThinkingOn);
        CHECK(c == realistic_expected());
        REQUIRE(c.ids.size() == 2);
        CHECK(c.ids[0].starts_with("call_"));
        CHECK(c.ids[0] != c.ids[1]);
    }

    TEST_CASE("thinking on: the template's newlines around the reasoning are not part of it") {
        const Collected c = parse_every_way("\nStep 1.\n\nStep 2.  \n</think>\n\nThe answer is 4.\n", kThinkingOn);
        CHECK(c.reasoning == "Step 1.\n\nStep 2.");
        CHECK(c.content == "The answer is 4.\n");
    }

    TEST_CASE("thinking off: output starts in content, leading newlines dropped") {
        const Collected c = parse_every_way("\n\nHello!  How are you?", kThinkingOff);
        CHECK(c.content == "Hello!  How are you?");
        CHECK(c.reasoning.empty());
    }

    TEST_CASE("empty reasoning") {
        const Collected c = parse_every_way("\n</think>\n\nHi", kThinkingOn);
        CHECK(c.reasoning.empty());
        CHECK(c.content == "Hi");
    }

    TEST_CASE("without a prompt the model's own <think> opens reasoning") {
        const Collected c = parse_every_way("<think>\nhmm\n</think>\n\nSunny.");
        CHECK(c.reasoning == "hmm");
        CHECK(c.content == "Sunny.");
    }

    TEST_CASE("unicode passes through every channel") {
        const Collected c = parse_every_way(
            "考える 🤔\n</think>\n\nОтвет: ✓\n\n<tool_call>\n<function=say>\n"
            "<parameter=текст>\nnaïve — 日本\n</parameter>\n</function>\n</tool_call>",
            kThinkingOn);
        CHECK(c.reasoning == "考える 🤔");
        CHECK(c.content == "Ответ: ✓");
        REQUIRE(c.calls.size() == 1);
        CHECK(c.calls[0].second == ordered_json::parse(R"({"текст": "naïve — 日本"})"));
    }

    TEST_CASE("whitespace between, before and after tool calls is dropped") {
        const std::string call = "<tool_call>\n<function=f>\n</function>\n</tool_call>";
        const Collected c = parse_every_way("  \n" + call + "\n\n" + call + "\n \n", kThinkingOff);
        CHECK(c.content.empty());
        CHECK(c.calls.size() == 2);
    }

    TEST_CASE("content after a tool call is kept, without its leading whitespace") {
        const Collected c =
            parse_every_way("<tool_call>\n<function=f>\n</function>\n</tool_call>\n\nOops, more text.", kThinkingOff);
        CHECK(c.calls.size() == 1);
        CHECK(c.content == "Oops, more text.");
    }

    TEST_CASE("parameter values: one newline stripped each side, JSON spellings become JSON values") {
        CHECK(value_of("\n42\n") == 42);
        CHECK(value_of("\n-1.5e3\n") == -1.5e3);
        CHECK(value_of("\ntrue\n") == true);
        CHECK(value_of("\nfalse\n") == false);
        CHECK(value_of("\nnull\n").is_null());
        CHECK(value_of("\n[1, {\"a\": null}]\n") == ordered_json::parse(R"([1, {"a": null}])"));
        CHECK(value_of("\n{\"b\": 1, \"a\": [true]}\n").dump() == R"({"b":1,"a":[true]})");
        CHECK(value_of("\n\"quoted\"\n") == "quoted");
        CHECK(value_of("\nParis\n") == "Paris");
        CHECK(value_of("\n\nkeeps inner newlines\n\n") == "\nkeeps inner newlines\n");
        CHECK(value_of("no newlines") == "no newlines");
        CHECK(value_of("\n\n") == "");
        CHECK(value_of("") == "");
        CHECK(value_of("\n1.2.3\n") == "1.2.3");
        CHECK(value_of("\n{not json}\n") == "{not json}");
        CHECK(value_of("\n  7  \n") == 7);
    }

    TEST_CASE("Python spellings the template renders (True, False, None) stay strings without a schema") {
        CHECK(value_of("\nTrue\n") == "True");
        CHECK(value_of("\nNone\n") == "None");
    }

    TEST_CASE("invalid UTF-8 in a value is replaced, never invalid JSON") {
        CHECK(value_of("\na\xff"
                       "b\n") ==
              "a\xEF\xBF\xBD"
              "b");
    }

    TEST_CASE("arguments keep the order the model wrote them in") {
        const Collected c = parse(
            "<tool_call>\n<function=f>\n<parameter=zeta>\n1\n</parameter>\n<parameter=alpha>\n"
            "2\n</parameter>\n</function>\n</tool_call>");
        REQUIRE(c.calls.size() == 1);
        CHECK(c.calls[0].second.dump() == R"({"zeta":1,"alpha":2})");
    }

    TEST_CASE("lenient like vLLM: missing </parameter> and missing </function>") {
        const Collected c = parse_every_way(
            "<tool_call>\n<function=f>\n<parameter=a>\nx\n<parameter=b>\n2\n</function>\n</tool_call>"
            "<tool_call>\n<function=g>\n<parameter=c>\ny\n</parameter>\n</tool_call>");
        REQUIRE(c.calls.size() == 2);
        CHECK(c.calls[0].second == ordered_json::parse(R"({"a": "x", "b": 2})"));
        CHECK(c.calls[1].second == ordered_json::parse(R"({"c": "y"})"));
    }

    TEST_CASE("a <function=...> block without the <tool_call> wrapper is a call") {
        const Collected c =
            parse_every_way("Sure.\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\nDone.", kThinkingOff);
        REQUIRE(c.calls.size() == 1);
        CHECK(c.calls[0].first == "f");
        CHECK(c.content == "Sure.Done.");
    }

    TEST_CASE("holds back only a possible marker prefix, and releases it when it is not one") {
        auto parser = make_output_parser("qwen3_5", kThinkingOn);
        CHECK(parser->feed("hm").reasoning == "hm");
        CHECK(parser->feed("m </thi").reasoning == "m");  // the space waits for what follows
        ParsedDelta d = parser->feed("nk>\n\nDone <");
        CHECK(d.reasoning.empty());
        CHECK(d.content == "Done");
        d = parser->feed("tool_box>");
        CHECK(d.content == " <tool_box>");
        CHECK(parser->feed(" <tool_c").content.empty());
        CHECK(parser->feed("all").content.empty());
        CHECK(parser->feed(">\n<function=f>\n</function>\n</tool_call>").tool_calls.size() == 1);
        CHECK(parser->finish().empty());
    }

    TEST_CASE("content streams without waiting for the end") {
        auto parser = make_output_parser("qwen3_5", kThinkingOff);
        CHECK(parser->feed("Line one\nLine").content == "Line one\nLine");
        CHECK(parser->feed(" two\n\n<tool_call>\n<function=f>").content == " two");
        CHECK(parser->feed("\n</function>\n</tool_call>").tool_calls.size() == 1);
    }

    TEST_CASE("an unterminated partial marker at the end is plain text") {
        CHECK(parse("Hi <tool_ca", kThinkingOff).content == "Hi <tool_ca");
        CHECK(parse("Hi <thi").content == "Hi <thi");
        const Collected c = parse("hmm\n</thi", kThinkingOn);
        CHECK(c.reasoning == "hmm\n</thi");  // still inside the reasoning
        CHECK(c.content.empty());
    }

    TEST_CASE("unterminated reasoning stays reasoning") {
        const Collected c = parse_every_way("long thoughts\nstill going", kThinkingOn);
        CHECK(c.reasoning == "long thoughts\nstill going");
        CHECK(c.content.empty());
    }

    TEST_CASE("a call cut short is content; one missing only </tool_call> is a call") {
        const std::string cut = "Sure.\n\n<tool_call>\n<function=f>\n<parameter=a>\nhal";
        const Collected c = parse_every_way(cut, kThinkingOff);
        CHECK(c.calls.empty());
        CHECK(c.content == "Sure.<tool_call>\n<function=f>\n<parameter=a>\nhal");

        const Collected complete =
            parse_every_way("<tool_call>\n<function=f>\n<parameter=a>\nx\n</parameter>\n</function>\n", kThinkingOff);
        REQUIRE(complete.calls.size() == 1);
        CHECK(complete.calls[0].second == ordered_json::parse(R"({"a": "x"})"));
        CHECK(complete.content.empty());

        CHECK(parse("<function=f>\n<parameter=a>\nx", kThinkingOff).content == "<function=f>\n<parameter=a>\nx");
    }

    TEST_CASE("malformed calls become content with their markers, never lost") {
        CHECK(parse("<tool_call>not a call</tool_call>after").content == "<tool_call>not a call</tool_call>after");
        CHECK(parse("<tool_call>\n<function=>\n</function>\n</tool_call>").content ==
              "<tool_call>\n<function=>\n</function>\n</tool_call>");
        CHECK(parse("<tool_call>\n<function=a b>\n</function>\n</tool_call>").calls.empty());
        CHECK(parse("<tool_call>\n<function=f>\njunk\n</function>\n</tool_call>").calls.empty());
        CHECK(parse("<tool_call>\n<function=f>\n</function>\njunk</tool_call>").calls.empty());
        CHECK(
            parse("<tool_call>\n<function=f>\n<parameter=>\nx\n</parameter>\n</function>\n</tool_call>").calls.empty());
    }

    TEST_CASE("stray closing markers are dropped from content") {
        CHECK(parse("a</think>b</tool_call>c", kThinkingOff).content == "abc");
    }

    TEST_CASE("a tool call inside reasoning ends the reasoning") {
        const Collected c =
            parse_every_way("I'll call it.\n<tool_call>\n<function=f>\n</function>\n</tool_call>", kThinkingOn);
        CHECK(c.reasoning == "I'll call it.");
        CHECK(c.calls.size() == 1);
        CHECK(c.content.empty());
    }

    TEST_CASE("ids are unique across many calls") {
        std::string text;
        for (int i = 0; i < 25; ++i)
            text += "<tool_call>\n<function=f>\n<parameter=i>\n" + std::to_string(i) +
                    "\n</parameter>\n</function>\n</tool_call>\n";
        const Collected c = parse(text);
        REQUIRE(c.calls.size() == 25);
        CHECK(c.calls[24].second["i"] == 24);
        CHECK(std::set<std::string>(c.ids.begin(), c.ids.end()).size() == 25);
    }

    TEST_CASE("finish with nothing pending is empty") {
        CHECK(make_output_parser("qwen3_5")->finish().empty());
        CHECK(make_output_parser("qwen3_5", kThinkingOn)->finish().empty());
    }
}

TEST_SUITE("chat qwen3_5 parser prompt tail") {
    TEST_CASE("Hugging Face prompts with thinking on end inside <think>") {
        for (const std::string name : {"user_only", "thinking_on", "multi_step_tools_keep_reasoning_after_last_user",
                                       "tool_call_object_arguments", "image_parts"}) {
            CAPTURE(name);
            const std::string prompt = rendered_prompt(name);
            REQUIRE(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
            const Collected c = parse_every_way("Hmm.\n</think>\n\nSunny.", prompt);
            CHECK(c.reasoning == "Hmm.");
            CHECK(c.content == "Sunny.");
        }
    }

    TEST_CASE("Hugging Face prompts with thinking off start in content") {
        for (const std::string name :
             {"thinking_off", "tools_thinking_off", "tool_round_trip_answered_then_new_user"}) {
            CAPTURE(name);
            const std::string prompt = rendered_prompt(name);
            REQUIRE(prompt.ends_with("<think>\n\n</think>\n\n"));
            const Collected c = parse_every_way("Plain answer.", prompt);
            CHECK(c.content == "Plain answer.");
            CHECK(c.reasoning.empty());
        }
    }

    TEST_CASE("a prefilled reply after its reasoning, or without one, continues in content") {
        CHECK(parse(" more", rendered_prompt("reasoning_content_after_last_user")).content == " more");
        CHECK(parse(" more", rendered_prompt("assistant_prefill")).content == " more");
    }

    TEST_CASE("think markers in earlier turns do not count") {
        const std::string prompt = "<|im_start|>user\nWhat does <think> mean?<|im_end|>\n<|im_start|>assistant\n";
        CHECK(parse("It opens reasoning.", prompt).content == "It opens reasoning.");
        const std::string open_in_user = "<|im_start|>user\nexplain <think><|im_end|>\n";
        CHECK(parse("x", open_in_user).content == "x");
    }

    TEST_CASE("prompt_tail keeps enough of a long prompt") {
        const std::string prompt = std::string(10000, 'x') + kThinkingOn;
        CHECK(parse("r", prompt).reasoning == "r");
    }

    TEST_CASE("a parsed reply rendered back by the template is what the model wrote") {
        // tool_call_with_preamble_and_reasoning: the assistant turn after the generation prompt, as HF rendered it.
        const std::string prompt = rendered_prompt("tool_call_with_preamble_and_reasoning");
        const std::string opener = "<|im_start|>assistant\n<think>\n";
        const size_t begin = prompt.find(opener) + opener.size();
        const std::string reply = prompt.substr(begin, prompt.find("<|im_end|>", begin) - begin);
        const Collected c = parse_every_way(reply, kThinkingOn);
        CHECK(c.reasoning == "Need the tool.");
        CHECK(c.content == "Let me check.");
        REQUIRE(c.calls.size() == 1);
        CHECK(c.calls[0].first == "get_weather");
        CHECK(c.calls[0].second.dump() == R"({"city":"Paris","unit":"celsius"})");
    }
}

// LFM3 writes the same markers, with nothing around the reasoning: "<think>R</think>C".
TEST_SUITE("chat qwen3_5 parser on LFM3 prompts") {
    TEST_CASE("Hugging Face prompts with thinking on end inside <think>") {
        for (const std::string name : {"user_only", "tools_simple", "multi_step_tools_keep_reasoning_after_last_user",
                                       "tool_response_names_deferred_tools", "preserve_thinking"}) {
            CAPTURE(name);
            const std::string prompt = rendered_prompt(name, "lfm3");
            REQUIRE(prompt.ends_with("<|im_start|>assistant\n<think>"));
            const Collected c = parse_every_way("Hmm, 12 * 12.</think>It is 144.", prompt);
            CHECK(c.reasoning == "Hmm, 12 * 12.");
            CHECK(c.content == "It is 144.");
        }
    }

    TEST_CASE("Hugging Face prompts with thinking off start in content") {
        for (const std::string name :
             {"thinking_off", "tools_thinking_off", "tool_round_trip_answered_then_new_user"}) {
            CAPTURE(name);
            const std::string prompt = rendered_prompt(name, "lfm3");
            REQUIRE(prompt.ends_with("<|im_start|>assistant\n<think></think>"));
            const Collected c = parse_every_way("Plain answer.", prompt);
            CHECK(c.content == "Plain answer.");
            CHECK(c.reasoning.empty());
        }
    }

    TEST_CASE("a parsed reply rendered back by the template is what the model wrote") {
        // The assistant turn with reasoning, a preamble and a call, and the turn of three calls, as HF rendered them.
        const auto reply_of = [](const std::string& prompt, const std::string& opener) {
            const size_t begin = prompt.find(opener) + opener.size();
            return prompt.substr(begin, prompt.find("<|im_end|>", begin) - begin);
        };
        const std::string thinking_on = rendered_prompt("user_only", "lfm3");
        Collected c = parse_every_way(reply_of(rendered_prompt("tool_call_with_preamble_and_reasoning", "lfm3"),
                                               "<|im_start|>assistant\n<think>"),
                                      thinking_on);
        CHECK(c.reasoning == "Need the tool.");
        CHECK(c.content == "Let me check.");
        REQUIRE(c.calls.size() == 1);
        CHECK(c.calls[0].first == "get_weather");
        CHECK(c.calls[0].second.dump() == R"({"city":"Paris","unit":"celsius"})");

        c = parse_every_way(reply_of(rendered_prompt("multiple_tool_calls_and_responses", "lfm3"),
                                     "<|im_start|>assistant\n<think></think>"),
                            rendered_prompt("thinking_off", "lfm3"));
        CHECK(c.reasoning.empty());
        CHECK(c.content == "I will use three tools.");
        REQUIRE(c.calls.size() == 3);
        CHECK(c.calls[1].first == "search_docs");
        CHECK(c.calls[1].second.dump() == R"({"query":"rain \"gear\"","Limit":5,"exact":false,)"
                                          R"("filters":{"tags":["a","b"],"since":null,"score":0.75},"URL":"x.org"})");
        CHECK(c.calls[2].second.dump() == R"({"texte":"il pleut","langue":"日本語"})");
    }
}
