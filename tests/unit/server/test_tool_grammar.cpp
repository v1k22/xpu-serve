// Tool calls a reply is held to (server/tool_grammar.h), in each dialect's own syntax over a vocabulary whose markers
// are tokens of their own: what the chat templates render is taken, a strict tool's arguments follow its schema, a
// named function alone or calls one after another as tool_choice and parallel_tool_calls say, the formatted answer
// beside the calls; the dialect's parser reads the calls back; refusals by name.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "chat/output_parser.h"
#include "grammar/json_schema.h"
#include "grammar/matcher.h"
#include "server/grammar_constraint.h"
#include "server/tool_grammar.h"

using nlohmann::json;
using xs::server::InvalidRequest;
using xs::server::RequestJson;
using xs::server::ToolCalls;
using xs::server::ToolChoice;
using namespace xs::grammar;

namespace {

const std::vector<std::string> kMarkers = {"<tool_call>",       "</tool_call>",
                                           "<|tool_call>",      "<tool_call|>",
                                           "<|\"|>",            "<|message_model|>",
                                           "<|content_invoke_tool_json|>", "<|end_message|>",
                                           "<|content_text|>"};

// The 256 bytes, then the markers as tokens of their own.
std::shared_ptr<const Vocabulary> vocabulary() {
    static const auto v = [] {
        std::vector<Vocabulary::Token> tokens;
        for (int b = 0; b < 256; ++b) tokens.push_back({std::string(1, static_cast<char>(b)), false});
        for (const std::string& m : kMarkers) tokens.push_back({m, true});
        return std::make_shared<const Vocabulary>(std::move(tokens));
    }();
    return v;
}

// `text` as the model writes it: each marker its token, the rest a token a byte.
std::vector<int32_t> tokens_of(std::string_view text) {
    std::vector<int32_t> out;
    for (size_t at = 0; at < text.size();) {
        bool found = false;
        for (size_t m = 0; m < kMarkers.size() && !found; ++m) {
            if (text.substr(at).starts_with(kMarkers[m])) {
                out.push_back(256 + static_cast<int32_t>(m));
                at += kMarkers[m].size();
                found = true;
            }
        }
        if (!found) out.push_back(static_cast<unsigned char>(text[at++]));
    }
    return out;
}

bool takes(const std::shared_ptr<const Grammar>& g, std::string_view text) {
    Matcher m(std::make_shared<Automaton>(g, vocabulary()));
    for (const int32_t t : tokens_of(text)) {
        if (!m.accept_token(t)) return false;
    }
    return m.complete();
}

const RequestJson kTools = RequestJson::parse(R"([
  {"type": "function", "function": {"name": "get_weather", "strict": true, "parameters": {"type": "object",
     "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "unit": {"enum": ["c", "f"]}},
     "required": ["city"], "additionalProperties": false}}},
  {"type": "function", "function": {"name": "search", "parameters": {"type": "object",
     "properties": {"q": {"type": "string"}}}}}])");

std::shared_ptr<const Grammar> grammar(std::string_view dialect, ToolChoice::Mode mode, std::string name = {},
                                       bool parallel = true, std::shared_ptr<const Grammar> answer = nullptr,
                                       std::string opening = {}) {
    const ToolCalls calls{.tools = &kTools,
                          .choice = {mode, std::move(name)},
                          .parallel = parallel,
                          .answer = std::move(answer),
                          .answer_opening = std::move(opening)};
    return xs::server::tool_call_grammar(dialect, calls, *vocabulary());
}

std::string refused_param(const std::function<void()>& build) {
    try {
        build();
    } catch (const InvalidRequest& e) {
        return e.param();
    }
    return "(accepted)";
}

const std::string kWeather = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                             "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>";
const std::string kSearch = "<tool_call>\n<function=search>\n<parameter=q>\nrain in\nParis\n</parameter>\n"
                            "<parameter=extra>\n{\"x\": 1}\n</parameter>\n</function>\n</tool_call>";

}  // namespace

TEST_SUITE("server.tool_grammar") {
    TEST_CASE("Qwen 3.5's XML calls: required, a strict tool's schema, any arguments of the others, one or several") {
        const auto g = grammar("qwen3_5", ToolChoice::Mode::required);
        CHECK(takes(g, kWeather));
        CHECK(takes(g, "\n\n" + kWeather));  // after a thought's close
        CHECK(takes(g, kSearch));
        CHECK(takes(g, kWeather + "\n" + kSearch));  // several, parallel
        CHECK(takes(g, "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n<parameter=unit>\n"
                       "\"f\"\n</parameter>\n</function>\n</tool_call>"));
        CHECK_FALSE(takes(g, ""));                       // at least one call
        CHECK_FALSE(takes(g, "Sure. " + kWeather));      // nothing but calls
        CHECK_FALSE(takes(g, "<tool_call>\n<function=get_weather>\n<parameter=days>\n2\n</parameter>\n</function>\n"
                             "</tool_call>"));           // the strict tool's required city missing
        CHECK_FALSE(takes(g, "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter="
                             "days>\ntwo\n</parameter>\n</function>\n</tool_call>"));  // days is an integer
        CHECK_FALSE(takes(g, "<tool_call>\n<function=ping>\n</function>\n</tool_call>"));  // no such tool
        // The markers are tokens: their bytes spelled out are not a call.
        Matcher m(std::make_shared<Automaton>(g, vocabulary()));
        CHECK_FALSE(m.accept_bytes("<tool_call>"));
        // One call when parallel calls are off; the named function alone, once.
        const auto one = grammar("qwen3_5", ToolChoice::Mode::required, {}, false);
        CHECK(takes(one, kWeather));
        CHECK_FALSE(takes(one, kWeather + "\n" + kSearch));
        const auto named = grammar("qwen3_5", ToolChoice::Mode::named, "search");
        CHECK(takes(named, kSearch));
        CHECK_FALSE(takes(named, kWeather));
        CHECK_FALSE(takes(named, kSearch + "\n" + kSearch));
    }

    TEST_CASE("the calls a grammar takes are the calls the dialect's parser reads") {
        auto parser = xs::chat::make_output_parser("qwen3_5", "<|im_start|>assistant\n<think>\n\n</think>\n\n");
        xs::chat::ParsedDelta d = parser->feed(kWeather + "\n" + kSearch);
        xs::chat::ParsedDelta rest = parser->finish();
        d.tool_calls.insert(d.tool_calls.end(), rest.tool_calls.begin(), rest.tool_calls.end());
        REQUIRE(d.tool_calls.size() == 2);
        CHECK(d.tool_calls[0].name == "get_weather");
        CHECK(json::parse(d.tool_calls[0].arguments) == json{{"city", "Paris"}, {"days", 2}});
        CHECK(d.tool_calls[1].name == "search");
        CHECK(json::parse(d.tool_calls[1].arguments) == json{{"q", "rain in\nParis"}, {"extra", {{"x", 1}}}});
        CHECK(d.content.empty());
    }

    TEST_CASE("Qwen 3's JSON calls, Gemma 4's own arguments, Inkling 2's messages") {
        const auto hermes = grammar("qwen3", ToolChoice::Mode::required);
        CHECK(takes(hermes, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"));
        CHECK(takes(hermes, "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"anything\": [1, 2]}}\n</tool_call>"));
        CHECK_FALSE(takes(hermes, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"town\": \"Paris\"}}\n"
                                  "</tool_call>"));
        const auto gemma = grammar("gemma4", ToolChoice::Mode::required);
        CHECK(takes(gemma, "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:2,unit:<|\"|>c<|\"|>}<tool_call|>"));
        CHECK(takes(gemma, "<|tool_call>call:search{q:<|\"|>x<|\"|>,n:[1,true,{a:null}]}<tool_call|>"
                           "<|tool_call>call:get_weather{city:<|\"|>Rome<|\"|>}<tool_call|>"));
        CHECK_FALSE(takes(gemma, "<|tool_call>call:get_weather{days:2}<tool_call|>"));
        CHECK_FALSE(takes(gemma, "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,unit:<|\"|>k<|\"|>}<tool_call|>"));
        const auto inkling = grammar("inkling2", ToolChoice::Mode::named, "get_weather");
        CHECK(takes(inkling, "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\","
                             "\"args\":{\"city\":\"Paris\",\"days\":3}}<|end_message|>"));
        CHECK_FALSE(takes(inkling, "<|message_model|>search<|content_invoke_tool_json|>{\"name\":\"search\","
                                   "\"args\":{}}<|end_message|>"));
    }

    // Llama 3's calls are the reply's JSON, as its template writes one: no opener, "; " between two.
    TEST_CASE("Llama 3's JSON calls: the reply's first character a call's, several with \"; \", read back") {
        const auto llama = grammar("llama3", ToolChoice::Mode::required);
        const std::string weather = R"({"name": "get_weather", "parameters": {"city": "Paris", "days": 2}})";
        const std::string search = R"({"name": "search", "parameters": {"q": "rain"}})";
        CHECK(takes(llama, weather));
        CHECK(takes(llama, weather + "; " + search));
        CHECK_FALSE(takes(llama, " " + weather));  // the parser reads a call only from the first character
        CHECK_FALSE(takes(llama, R"({"name": "get_weather", "parameters": {"days": 2}})"));  // strict: city required
        CHECK_FALSE(takes(llama, R"({"name": "get_weather", "arguments": {"city": "Paris"}})"));
        CHECK_FALSE(takes(grammar("llama3", ToolChoice::Mode::required, {}, false), weather + "; " + search));
        const auto named = grammar("llama3", ToolChoice::Mode::named, "search");
        CHECK(takes(named, search));
        CHECK_FALSE(takes(named, weather));
        auto parser = xs::chat::make_output_parser("llama3", "", /*tools=*/true);
        xs::chat::ParsedDelta d = parser->feed(weather + "; " + search);
        const xs::chat::ParsedDelta rest = parser->finish();
        d.tool_calls.insert(d.tool_calls.end(), rest.tool_calls.begin(), rest.tool_calls.end());
        REQUIRE(d.tool_calls.size() == 2);
        CHECK(d.tool_calls[0].name == "get_weather");
        CHECK(json::parse(d.tool_calls[0].arguments) == json{{"city", "Paris"}, {"days", 2}});
        CHECK(d.tool_calls[1].name == "search");
        CHECK(d.content.empty());
        // Beside a format: the calls, or the answer.
        const auto either = grammar("llama3", ToolChoice::Mode::automatic, {}, true, json_object_grammar());
        CHECK(takes(either, weather));
        CHECK(takes(either, "{\"answer\": 42}"));
        // No opener to hold a strict call from under auto.
        CHECK_FALSE(xs::server::strict_calls("llama3", kTools, *vocabulary()).has_value());
    }

    TEST_CASE("auto beside a response_format: the calls, or the answer held to the format after its opening") {
        const auto answer = json_object_grammar();
        const auto qwen = grammar("qwen3_5", ToolChoice::Mode::automatic, {}, true, answer);
        CHECK(takes(qwen, kWeather));
        CHECK(takes(qwen, "{\"answer\": 42}"));
        CHECK_FALSE(takes(qwen, "The answer is 42."));
        const auto inkling =
            grammar("inkling2", ToolChoice::Mode::automatic, {}, true, answer, "<|message_model|><|content_text|>");
        CHECK(takes(inkling, "<|message_model|><|content_text|>{\"answer\": 42}"));
        CHECK(takes(inkling, "<|message_model|>search<|content_invoke_tool_json|>{\"name\":\"search\",\"args\":{}}"
                             "<|end_message|>"));
        CHECK_FALSE(takes(inkling, "{\"answer\": 42}"));
    }

    TEST_CASE("auto with strict tools: free text, a call held from its opener to its closer, then free again") {
        const auto calls = xs::server::strict_calls("qwen3_5", kTools, *vocabulary());
        REQUIRE(calls.has_value());
        CHECK(calls->opener == 256);  // <tool_call>
        const xs::server::OpenerConstraint constraint(calls->opener, calls->grammar, vocabulary());
        const auto takes_reply = [&](std::string_view text) {
            const auto c = constraint.start();
            const std::vector<int32_t> tokens = tokens_of(text);
            if (c->allowed_prefix(tokens) != tokens.size()) return false;
            for (const int32_t t : tokens) c->accept(t);
            return c->complete();
        };
        CHECK(takes_reply("Let me check. " + kWeather + " Done."));
        CHECK(takes_reply("No tools needed: it is 4."));
        CHECK(takes_reply(kSearch + "\n" + kWeather));
        CHECK_FALSE(takes_reply("<tool_call>\n<function=get_weather>\n<parameter=days>\n2\n</parameter>\n</function>\n"
                                "</tool_call>"));  // the strict tool's city missing
        CHECK_FALSE(takes_reply(kWeather.substr(0, 40)));  // a call left open cannot end the reply
        // Masks: none while free, the call's grammar inside one.
        const auto c = constraint.start();
        std::vector<float> logits(static_cast<size_t>(vocabulary()->size()), 0.0f);
        CHECK(c->mask(logits));
        CHECK(std::count(logits.begin(), logits.end(), 0.0f) == static_cast<long>(logits.size()));
        c->accept(256);
        CHECK_FALSE(c->complete());
        c->mask(logits);
        CHECK(logits['\n'] == 0.0f);
        CHECK(std::isinf(logits['x']));
        // No strict tool, or a dialect without calls: nothing to hold.
        RequestJson loose = kTools;
        loose[0]["function"]["strict"] = false;
        CHECK_FALSE(xs::server::strict_calls("qwen3_5", loose, *vocabulary()).has_value());
        CHECK_FALSE(xs::server::strict_calls("plain", kTools, *vocabulary()).has_value());
        for (const char* dialect : {"qwen3", "gemma4", "inkling2"}) {
            CAPTURE(dialect);
            CHECK(xs::server::strict_calls(dialect, kTools, *vocabulary()).has_value());
        }
    }

    TEST_CASE("refused by name: a strict schema's keyword the compiler does not take, a dialect without calls") {
        RequestJson tools = kTools;
        tools[0]["function"]["parameters"]["properties"]["days"]["minimum"] = 1;
        const ToolCalls calls{.tools = &tools, .choice = {ToolChoice::Mode::required, {}}};
        for (const char* dialect : {"qwen3", "qwen3_5", "inkling2"}) {
            CAPTURE(dialect);
            CHECK(refused_param([&] { xs::server::tool_call_grammar(dialect, calls, *vocabulary()); }) ==
                  "tools[0].function.parameters.properties.days.minimum");
        }
        // Not strict: its schema is not compiled.
        tools[0]["function"]["strict"] = false;
        CHECK(refused_param([&] { xs::server::tool_call_grammar("qwen3", calls, *vocabulary()); }) == "(accepted)");
        CHECK(refused_param([&] { xs::server::tool_call_grammar("plain", calls, *vocabulary()); }) == "tool_choice");
        CHECK_FALSE(xs::server::writes_tool_calls("plain"));
    }
}
