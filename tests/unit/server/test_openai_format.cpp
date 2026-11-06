#include <doctest/doctest.h>

#include "server/openai_format.h"

using nlohmann::json;
using namespace xs::server;

namespace {

const ResponseMeta kMeta{"chatcmpl-1", "fake-model", 1700000000};

xs::GenerateStats stats() {
    xs::GenerateStats s;
    s.prompt_tokens = 10;
    s.cached_tokens = 4;
    s.generated_tokens = 5;
    s.prefill_ms = 6.0;
    s.decode_ms = 8.0;
    return s;
}

}  // namespace

TEST_SUITE("server.format") {
    TEST_CASE("error body has the OpenAI shape with null for empty param and code") {
        const json e = error_body("bad", "invalid_request_error", "n")["error"];
        CHECK(e["message"] == "bad");
        CHECK(e["type"] == "invalid_request_error");
        CHECK(e["param"] == "n");
        CHECK(e["code"].is_null());
    }

    TEST_CASE("finish reasons") {
        CHECK(finish_reason_text(xs::FinishReason::Stop, false, false) == "stop");
        CHECK(finish_reason_text(xs::FinishReason::Length, false, false) == "length");
        CHECK(finish_reason_text(xs::FinishReason::Cancelled, true, false) == "stop");
        CHECK(finish_reason_text(xs::FinishReason::Stop, false, true) == "tool_calls");
    }

    TEST_CASE("usage and timings") {
        const json u = usage_json(stats());
        CHECK(u["prompt_tokens"] == 10);
        CHECK(u["completion_tokens"] == 5);
        CHECK(u["total_tokens"] == 15);
        CHECK(u["prompt_tokens_details"]["cached_tokens"] == 4);
        const json t = timings_json(stats());
        CHECK(t["prefill_tokens_per_second"].get<double>() == doctest::Approx(1000.0));
        CHECK(t["decode_tokens_per_second"].get<double>() == doctest::Approx(500.0));
    }

    TEST_CASE("chat completion message carries reasoning and tool calls only when present") {
        xs::chat::ParsedDelta msg;
        msg.content = "hi";
        json c = chat_completion_json(kMeta, msg, "stop", stats());
        CHECK(c["object"] == "chat.completion");
        CHECK(c["id"] == "chatcmpl-1");
        CHECK(c["choices"][0]["message"]["content"] == "hi");
        CHECK_FALSE(c["choices"][0]["message"].contains("reasoning_content"));
        CHECK_FALSE(c["choices"][0]["message"].contains("tool_calls"));
        CHECK(c.contains("timings"));

        msg.reasoning = "thinking";
        msg.tool_calls.push_back({"call_1", "get_weather", R"({"city":"Paris"})"});
        c = chat_completion_json(kMeta, msg, "tool_calls", stats());
        const json& m = c["choices"][0]["message"];
        CHECK(m["reasoning_content"] == "thinking");
        CHECK(m["tool_calls"][0]["id"] == "call_1");
        CHECK(m["tool_calls"][0]["type"] == "function");
        CHECK(m["tool_calls"][0]["function"]["name"] == "get_weather");
        CHECK(m["tool_calls"][0]["function"]["arguments"] == R"({"city":"Paris"})");
        CHECK(c["choices"][0]["finish_reason"] == "tool_calls");
    }

    TEST_CASE("chat delta numbers tool calls from the given index") {
        xs::chat::ParsedDelta d;
        d.tool_calls.push_back({"call_a", "f", "{}"});
        d.tool_calls.push_back({"call_b", "g", R"({"x":1})"});
        const json delta = chat_delta_json(d, 2);
        CHECK_FALSE(delta.contains("content"));
        REQUIRE(delta["tool_calls"].size() == 2);
        CHECK(delta["tool_calls"][0]["index"] == 2);
        CHECK(delta["tool_calls"][1]["index"] == 3);
        CHECK(delta["tool_calls"][1]["id"] == "call_b");
        CHECK(delta["tool_calls"][1]["type"] == "function");
        CHECK(delta["tool_calls"][1]["function"]["arguments"] == R"({"x":1})");
    }

    TEST_CASE("chat chunks") {
        xs::chat::ParsedDelta d;
        d.content = "a";
        d.reasoning = "r";
        const json chunk = chat_chunk_json(kMeta, chat_delta_json(d, 0));
        CHECK(chunk["object"] == "chat.completion.chunk");
        CHECK(chunk["choices"][0]["delta"]["content"] == "a");
        CHECK(chunk["choices"][0]["delta"]["reasoning_content"] == "r");
        CHECK(chunk["choices"][0]["finish_reason"].is_null());
        CHECK(chat_chunk_json(kMeta, json::object(), "length")["choices"][0]["finish_reason"] == "length");
        const json usage = chat_usage_chunk_json(kMeta, stats());
        CHECK(usage["choices"].empty());
        CHECK(usage["usage"]["total_tokens"] == 15);
    }

    TEST_CASE("text completion objects and SSE framing") {
        const json c = text_completion_json(kMeta, "abc", "stop", stats());
        CHECK(c["object"] == "text_completion");
        CHECK(c["choices"][0]["text"] == "abc");
        CHECK(c["usage"]["completion_tokens"] == 5);
        CHECK(sse_event(json{{"a", 1}}) == "data: {\"a\":1}\n\n");
    }

    TEST_CASE("model list") {
        xs::ModelCard card;
        card.id = "m";
        card.context_length = 128;
        const json list = model_list_json(card, 1);
        CHECK(list["object"] == "list");
        CHECK(list["data"][0]["id"] == "m");
        CHECK(list["data"][0]["object"] == "model");
        CHECK(list["data"][0]["context_length"] == 128);
    }
}
