// The Inkling 2 output dialect (chat/inkling2_parser.cpp): messages of a kind each, split into reasoning, content
// and tool calls, however the text is cut into deltas.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat/output_parser.h"

using nlohmann::json;
using xs::chat::make_output_parser;
using xs::chat::ParsedDelta;

namespace {

struct Parsed {
    std::string content, reasoning;
    std::vector<std::pair<std::string, json>> calls;  // name, arguments
    std::vector<std::string> ids;
};

// The whole of `text` through a parser, `step` bytes a feed (0: at once).
Parsed parse(const std::string& text, size_t step = 0, std::string_view prompt_tail = {}) {
    const auto parser = make_output_parser("inkling2", prompt_tail);
    Parsed out;
    const auto add = [&](const ParsedDelta& d) {
        out.content += d.content;
        out.reasoning += d.reasoning;
        for (const auto& c : d.tool_calls) {
            out.calls.emplace_back(c.name, json::parse(c.arguments));
            out.ids.push_back(c.id);
        }
    };
    if (step == 0) {
        add(parser->feed(text));
    } else {
        for (size_t i = 0; i < text.size(); i += step) add(parser->feed(std::string_view(text).substr(i, step)));
    }
    add(parser->finish());
    return out;
}

const std::string kTurn =
    "<|message_model|><|content_thinking|>The user wants 2 + 2.\nThat is 4.<|end_message|>"
    "<|message_model|><|content_text|>2 + 2 = 4, and 1 < 2 <| 3.<|end_message|><|content_model_end_sampling|>";

}  // namespace

TEST_CASE("inkling2 dialect: thinking is reasoning, text is content, no marker in either") {
    const Parsed p = parse(kTurn);
    CHECK(p.reasoning == "The user wants 2 + 2.\nThat is 4.");
    CHECK(p.content == "2 + 2 = 4, and 1 < 2 <| 3.");
    CHECK(p.calls.empty());
}

TEST_CASE("inkling2 dialect: the same however the text is cut into deltas") {
    const Parsed whole = parse(kTurn);
    for (const size_t step : {size_t{1}, size_t{2}, size_t{3}, size_t{7}, size_t{16}}) {
        CAPTURE(step);
        const Parsed p = parse(kTurn, step);
        CHECK(p.reasoning == whole.reasoning);
        CHECK(p.content == whole.content);
    }
    // A delta never holds back more than a marker's start: text comes out as it arrives.
    const auto parser = make_output_parser("inkling2");
    parser->feed("<|message_model|><|content_text|>");
    CHECK(parser->feed("Hello").content == "Hello");
    CHECK(parser->feed(" there <").content == " there ");
    CHECK(parser->feed(" 3").content == "< 3");
}

TEST_CASE("inkling2 dialect: tool calls, their names and arguments, each with an id of its own") {
    const std::string turn =
        "<|message_model|><|content_thinking|>Two lookups.<|end_message|>"
        "<|message_model|>get_weather<|content_invoke_tool_json|>"
        R"({"name":"get_weather","args":{"city":"Paris","days":2}})"
        "<|end_message|><|message_model|>get_time<|content_invoke_tool_json|>"
        R"({"name":"get_time","args":{"zone":"Europe/Paris"}})"
        "<|end_message|><|content_model_end_sampling|>";
    for (const size_t step : {size_t{0}, size_t{1}, size_t{5}}) {
        CAPTURE(step);
        const Parsed p = parse(turn, step);
        CHECK(p.reasoning == "Two lookups.");
        CHECK(p.content.empty());
        REQUIRE(p.calls.size() == 2);
        CHECK(p.calls[0].first == "get_weather");
        CHECK(p.calls[0].second == json{{"city", "Paris"}, {"days", 2}});
        CHECK(p.calls[1].first == "get_time");
        CHECK(p.calls[1].second == json{{"zone", "Europe/Paris"}});
        CHECK(p.ids[0] != p.ids[1]);
        CHECK(p.ids[0].starts_with("call_"));
    }
    // A body without a name takes its message's; one without arguments has none.
    const Parsed named = parse("<|message_model|>ping<|content_invoke_tool_json|>{}<|end_message|>");
    REQUIRE(named.calls.size() == 1);
    CHECK(named.calls[0].first == "ping");
    CHECK(named.calls[0].second == json::object());
    // A call the model never closed still counts when its body is whole.
    const Parsed open = parse(R"(<|message_model|>ping<|content_invoke_tool_json|>{"name":"ping","args":{"n":1}})");
    REQUIRE(open.calls.size() == 1);
    CHECK(open.calls[0].second == json{{"n", 1}});
}

TEST_CASE("inkling2 dialect: what is not the template's writing stays text") {
    // A body that is not JSON is content, as it was written.
    const Parsed broken = parse("<|message_model|>f<|content_invoke_tool_json|>{\"name\": oops<|end_message|>");
    CHECK(broken.calls.empty());
    CHECK(broken.content == "{\"name\": oops");
    // Text before any message, a kind the parser does not know, a second text message on a line of its own.
    CHECK(parse("bare text").content == "bare text");
    CHECK(parse("<|message_model|><|content_xml|><a/><|end_message|>").content == "<a/>");
    const std::string one = "<|message_model|><|content_text|>One.<|end_message|>";
    CHECK(parse(one + "<|message_model|><|content_text|>Two.<|end_message|>").content == "One.\nTwo.");
    // A marker's start that never completes is text at the end.
    CHECK(parse("<|message_model|><|content_text|>cut <|end_mess").content == "cut <|end_mess");
}

TEST_CASE("inkling2 dialect: a prompt that ends inside a message starts the parser there") {
    CHECK(parse("still thinking<|end_message|><|message_model|><|content_text|>Done.<|end_message|>", 0,
                "<|message_user|><|content_text|>Hi<|end_message|><|message_model|><|content_thinking|>")
              .reasoning == "still thinking");
    const Parsed header = parse("<|content_text|>Hello.<|end_message|>", 0, "...<|end_message|><|message_model|>");
    CHECK(header.content == "Hello.");
    CHECK(header.reasoning.empty());
    // A closed message leaves nothing open.
    CHECK(parse("<|message_model|><|content_text|>Hi.<|end_message|>", 0,
                "<|message_model|><|content_thinking|>old<|end_message|>")
              .content == "Hi.");
}

TEST_CASE("inkling2: an answer at once opens the model's text message after the prompt, with no thinking effort") {
    const xs::chat::DirectAnswer a = xs::chat::direct_answer("inkling2");
    CHECK(a.opening == "<|message_model|><|content_text|>");
    CHECK(a.kwargs.at("reasoning_effort") == "none");
    // What the parser reads after that opening is the answer's text.
    CHECK(parse(a.opening + "B<|end_message|>", 0, "").content == "B");
    // Other dialects' templates end their generation prompts where the answer starts.
    for (const char* d : {"plain", "qwen3", "qwen3_5", "gemma4"}) {
        CAPTURE(d);
        const xs::chat::DirectAnswer other = xs::chat::direct_answer(d);
        CHECK(other.opening.empty());
        CHECK(other.kwargs == json{{"enable_thinking", false}});
    }
    CHECK_THROWS(xs::chat::direct_answer("nope"));
}
