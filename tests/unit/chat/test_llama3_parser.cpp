// Llama 3's output dialect (chat/llama3_parser.cpp), as vLLM's llama3_json parser reads it: with tools offered, a reply
// whose first character is "{", or the text after <|python_tag|>, is JSON calls separated by ";" ("parameters" or
// "arguments"); anything that is not such a sequence is content, whole; without tools everything is content.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat/output_parser.h"

namespace {

using nlohmann::json;
using xs::chat::ParsedDelta;

struct Reply {
    std::string content;
    std::vector<std::pair<std::string, json>> calls;
    bool operator==(const Reply& o) const { return content == o.content && calls == o.calls; }
};

Reply parse_chunks(const std::vector<std::string>& chunks, bool tools) {
    auto parser = xs::chat::make_output_parser("llama3", "<|start_header_id|>assistant<|end_header_id|>\n\n", tools);
    Reply r;
    const auto add = [&](const ParsedDelta& d) {
        CHECK(d.reasoning.empty());
        r.content += d.content;
        for (const auto& c : d.tool_calls) r.calls.emplace_back(c.name, json::parse(c.arguments));
    };
    for (const auto& c : chunks) add(parser->feed(c));
    add(parser->finish());
    return r;
}

// Whole, byte by byte and split at every point: the same reply.
Reply parse(const std::string& text, bool tools = true) {
    const Reply whole = parse_chunks({text}, tools);
    std::vector<std::string> bytes;
    for (const char c : text) bytes.emplace_back(1, c);
    CHECK(parse_chunks(bytes, tools) == whole);
    for (size_t at = 1; at < text.size(); ++at) {
        CAPTURE(at);
        CHECK(parse_chunks({text.substr(0, at), text.substr(at)}, tools) == whole);
    }
    return whole;
}

}  // namespace

TEST_SUITE("chat llama3 parser") {
    TEST_CASE("a reply that opens with a JSON object is calls; several separated by ';'") {
        const Reply one = parse(R"({"name": "get_weather", "parameters": {"city": "Paris", "unit": "celsius"}})");
        CHECK(one.content.empty());
        REQUIRE(one.calls.size() == 1);
        CHECK(one.calls[0].first == "get_weather");
        CHECK(one.calls[0].second == json{{"city", "Paris"}, {"unit", "celsius"}});
        const Reply two = parse(R"({"name": "a", "arguments": {"x": "1;2"}}; {"name": "b", "parameters": {}};)");
        REQUIRE(two.calls.size() == 2);
        CHECK(two.calls[0].second == json{{"x", "1;2"}});  // a ";" inside a string separates nothing
        CHECK(two.calls[1].first == "b");
        const Reply text_args = parse(R"({"name": "a", "parameters": "{\"x\": 1}"})");  // arguments as JSON text
        REQUIRE(text_args.calls.size() == 1);
        CHECK(text_args.calls[0].second == json{{"x", 1}});
    }

    TEST_CASE("<|python_tag|> opens calls after content; ids unique within a reply") {
        const Reply r = parse(R"(Checking.<|python_tag|>{"name": "search", "parameters": {"q": "rain"}})");
        CHECK(r.content == "Checking.");
        REQUIRE(r.calls.size() == 1);
        CHECK(r.calls[0].first == "search");
        auto parser = xs::chat::make_output_parser("llama3", "", true);
        parser->feed(R"({"name": "a", "parameters": {}}; {"name": "b", "parameters": {}})");
        const ParsedDelta d = parser->finish();
        REQUIRE(d.tool_calls.size() == 2);
        CHECK(d.tool_calls[0].id.starts_with("call_"));
        CHECK(d.tool_calls[0].id != d.tool_calls[1].id);
    }

    TEST_CASE("what is not a sequence of calls is content, whole; without tools everything is") {
        const std::string answer = R"({"answer": 42})";
        CHECK(parse(answer) == Reply{answer, {}});  // no name
        CHECK(parse(R"({"name": "a"})") == Reply{R"({"name": "a"})", {}});  // no arguments
        CHECK(parse(R"({"name": "a", "parameters": [1]})").calls.empty());
        CHECK(parse(R"({"name": "a", "parameters": {}} and more)").calls.empty());
        CHECK(parse(R"({"name": "a", "parameters": {)").content == R"({"name": "a", "parameters": {)");  // cut short
        CHECK(parse("<|python_tag|>brave_search.call(query=\"x\")") ==
              Reply{"<|python_tag|>brave_search.call(query=\"x\")", {}});
        // vLLM reads the first text as it comes: a call after a space is text.
        const std::string spaced = R"( {"name": "a", "parameters": {}})";
        CHECK(parse(spaced) == Reply{spaced, {}});
        CHECK(parse("The answer is 4. <|python_ta") == Reply{"The answer is 4. <|python_ta", {}});
        const std::string call = R"({"name": "get_weather", "parameters": {"city": "Paris"}})";
        CHECK(parse(call, /*tools=*/false) == Reply{call, {}});
        CHECK(parse("plain text", false) == Reply{"plain text", {}});
    }
}
