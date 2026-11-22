// Anthropic's Messages API: requests read as the chat request a template renders (server/anthropic_request.h), and
// responses as a message of blocks or its stream of events (server/anthropic_format.h).
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "server/anthropic_format.h"
#include "server/anthropic_request.h"
#include "util/base64.h"

using xs::server::InvalidRequest;
using xs::server::RequestJson;
using json = nlohmann::json;

namespace {

xs::server::ChatRequest parse(const RequestJson& body, std::string_view dialect = "plain") {
    return xs::server::parse_messages_request(body, xs::SamplingParams{}, dialect);
}

RequestJson base(RequestJson extra = RequestJson::object()) {
    RequestJson body = {{"model", "m"}, {"max_tokens", 64},
                        {"messages", RequestJson::array({{{"role", "user"}, {"content", "hi"}}})}};
    body.update(extra);
    return body;
}

// The named events of a stream: (name, data).
std::vector<std::pair<std::string, json>> events_of(const std::string& text) {
    std::vector<std::pair<std::string, json>> out;
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t end = text.find("\n\n", pos);
        REQUIRE(end != std::string::npos);
        const std::string e = text.substr(pos, end - pos);
        REQUIRE(e.rfind("event: ", 0) == 0);
        const size_t nl = e.find('\n');
        REQUIRE(e.compare(nl + 1, 6, "data: ") == 0);
        out.emplace_back(e.substr(7, nl - 7), json::parse(e.substr(nl + 7)));
        pos = end + 2;
    }
    return out;
}

}  // namespace

TEST_CASE("anthropic request: system, blocks, tool use and results, thinking and tools as a template reads them") {
    const std::string png = xs::base64_encode(std::vector<uint8_t>{1, 2, 3});
    const RequestJson body = base(
        {{"system", RequestJson::array({{{"type", "text"}, {"text", "Be brief."}}})},
         {"messages",
          RequestJson::array(
              {{{"role", "user"},
                {"content", RequestJson::array({{{"type", "text"}, {"text", "Look: "}},
                                                {{"type", "image"},
                                                 {"source", {{"type", "base64"}, {"media_type", "image/png"},
                                                             {"data", png}}}}})}},
               {{"role", "assistant"},
                {"content", RequestJson::array({{{"type", "thinking"}, {"thinking", "Hmm."}, {"signature", "x"}},
                                                {{"type", "text"}, {"text", "Let me check."}},
                                                {{"type", "tool_use"}, {"id", "t1"}, {"name", "look"},
                                                 {"input", {{"q", 1}}}}})}},
               {{"role", "user"},
                {"content", RequestJson::array({{{"type", "tool_result"}, {"tool_use_id", "t1"}, {"content", "red"}},
                                                {{"type", "text"}, {"text", "And now?"}}})}}})},
         {"tools", RequestJson::array({{{"name", "look"}, {"description", "d"},
                                        {"input_schema", {{"type", "object"}}}}})},
         {"stop_sequences", {"END"}},
         {"temperature", 0.2},
         {"top_k", 5},
         {"stream", true}});
    const xs::server::ChatRequest r = parse(body);
    REQUIRE(r.messages.size() == 5);
    CHECK(r.messages[0] == RequestJson{{"role", "system"}, {"content", "Be brief."}});
    CHECK(r.messages[1]["role"] == "user");
    CHECK(r.messages[1]["content"][1] == RequestJson{{"type", "image"}});
    REQUIRE(r.media.size() == 1);
    CHECK(r.media[0].bytes == std::vector<uint8_t>{1, 2, 3});
    CHECK(r.messages[2]["reasoning_content"] == "Hmm.");
    CHECK(r.messages[2]["content"] == "Let me check.");
    CHECK(r.messages[2]["tool_calls"][0]["function"] == RequestJson{{"name", "look"}, {"arguments", {{"q", 1}}}});
    CHECK(r.messages[3] == RequestJson{{"role", "tool"}, {"tool_call_id", "t1"}, {"content", "red"}});
    CHECK(r.messages[4] == RequestJson{{"role", "user"}, {"content", "And now?"}});
    CHECK(r.tools[0]["function"]["parameters"] == RequestJson{{"type", "object"}});
    CHECK(r.options.stop == std::vector<std::string>{"END"});
    CHECK(r.options.max_tokens == 64);
    CHECK(r.options.sampling.temperature == doctest::Approx(0.2));
    CHECK(r.options.sampling.top_k == 5);
    CHECK(r.options.stream);
}

TEST_CASE("anthropic request: thinking on and off, tool_choice none, and what is refused by name") {
    CHECK(parse(base({{"thinking", {{"type", "enabled"}, {"budget_tokens", 1024}}}})).template_kwargs ==
          RequestJson{{"enable_thinking", true}});
    CHECK(parse(base({{"thinking", {{"type", "disabled"}}}}), "inkling2").template_kwargs["reasoning_effort"] ==
          "none");
    const RequestJson tools = RequestJson::array({{{"name", "f"}, {"input_schema", RequestJson::object()}}});
    CHECK(parse(base({{"tools", tools}, {"tool_choice", {{"type", "none"}}}})).tools.is_null());
    const auto refused = [](const RequestJson& body, const char* param) {
        CAPTURE(body.dump());
        try {
            parse(body);
            FAIL("accepted");
        } catch (const InvalidRequest& e) {
            CHECK(e.param() == param);
        }
    };
    RequestJson no_max = base();
    no_max.erase("max_tokens");
    refused(no_max, "max_tokens");
    CHECK_NOTHROW(xs::server::parse_messages_request(no_max, {}, "plain", /*count_only=*/true));
    refused(base({{"tools", RequestJson::array({{{"type", "web_search_20250305"}, {"name", "w"}}})}}), "tools[0]");
    refused(base({{"tool_choice", {{"type", "any"}}}}), "tool_choice");
    refused(base({{"messages", RequestJson::array({{{"role", "user"},
                                                    {"content", RequestJson::array({{{"type", "document"}}})}}})}}),
            "messages[0].content[0]");
    const RequestJson url_image = {{"type", "image"}, {"source", {{"type", "url"}, {"url", "x"}}}};
    const RequestJson url_message = {{"role", "user"}, {"content", RequestJson::array({url_image})}};
    refused(base({{"messages", RequestJson::array({url_message})}}), "messages[0].content[0]");
    refused(base({{"messages", RequestJson::array({{{"role", "system"}, {"content", "x"}}})}}), "messages[0].role");
}

TEST_CASE("anthropic format: a stream of blocks, each opened, filled and closed in turn, then the stop reason") {
    const auto f = xs::server::anthropic_format("msg_1", "m", 12);
    xs::chat::ParsedDelta think, text, more, call;
    think.reasoning = "Hm";
    text.content = "Hi";
    more.content = " there";
    call.tool_calls.push_back({"call_1", "look", R"({"q":1})"});
    std::string stream = f->opening() + f->delta(think) + f->delta(text) + f->delta(more) + f->delta(call);
    xs::server::GenerationOutcome o;
    o.result.finish = xs::FinishReason::Stop;
    o.result.stats.prompt_tokens = 12;
    o.result.stats.cached_tokens = 4;
    o.result.stats.generated_tokens = 5;
    stream += f->closing(o);
    const auto events = events_of(stream);
    std::vector<std::string> names;
    for (const auto& [name, data] : events) names.push_back(name);
    CHECK(names == std::vector<std::string>{"message_start", "ping", "content_block_start", "content_block_delta",
                                            "content_block_stop", "content_block_start", "content_block_delta",
                                            "content_block_delta", "content_block_stop", "content_block_start",
                                            "content_block_delta", "content_block_stop", "message_delta",
                                            "message_stop"});
    CHECK(events[0].second["message"]["usage"]["input_tokens"] == 12);
    CHECK(events[2].second["content_block"]["type"] == "thinking");
    CHECK(events[3].second["delta"] == json{{"type", "thinking_delta"}, {"thinking", "Hm"}});
    CHECK(events[5].second["index"] == 1);
    CHECK(events[7].second["delta"]["text"] == " there");
    CHECK(events[9].second["content_block"] == json{{"type", "tool_use"}, {"id", "call_1"}, {"name", "look"},
                                                    {"input", json::object()}});
    CHECK(events[10].second["delta"] == json{{"type", "input_json_delta"}, {"partial_json", R"({"q":1})"}});
    CHECK(events[12].second["delta"]["stop_reason"] == "tool_use");
    CHECK(events[12].second["usage"]["cache_read_input_tokens"] == 4);
    CHECK(events[12].second["usage"]["input_tokens"] == 8);

    xs::chat::ParsedDelta all = think;
    all.content = "Hi there";
    all.tool_calls = call.tool_calls;
    const json body = xs::server::anthropic_format("msg_1", "m", 12)->body(all, o);
    CHECK(body["content"].size() == 3);
    CHECK(body["content"][2]["input"] == json{{"q", 1}});
    CHECK(body["stop_reason"] == "tool_use");
    // A stop sequence names itself; a budget spent is max_tokens.
    o.stop_string_hit = true;
    o.stop_string = "END";
    const json stopped = xs::server::anthropic_format("msg_2", "m", 1)->body(text, o);
    CHECK(stopped["stop_reason"] == "stop_sequence");
    CHECK(stopped["stop_sequence"] == "END");
    o.stop_string_hit = false;
    o.result.finish = xs::FinishReason::Length;
    CHECK(xs::server::anthropic_format("msg_3", "m", 1)->body(text, o)["stop_reason"] == "max_tokens");
}
