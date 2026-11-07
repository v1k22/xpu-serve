#include <doctest/doctest.h>

#include <string>

#include "server/openai_request.h"

using nlohmann::json;
using xs::server::InvalidRequest;
using xs::server::parse_chat_request;
using xs::server::parse_completion_request;
using xs::server::parse_json_body;

namespace {

const json kMessages = json::array({{{"role", "user"}, {"content", "hi"}}});

json chat_body(json extra) {
    json body = {{"model", "m"}, {"messages", kMessages}};
    body.update(extra);
    return body;
}

// The `param` of the InvalidRequest that parsing `body` throws.
std::string rejected_param(const json& body) {
    try {
        parse_chat_request(body, {});
    } catch (const InvalidRequest& e) {
        return e.param();
    }
    return "<accepted>";
}

}  // namespace

TEST_SUITE("server.request") {
    TEST_CASE("body must be a JSON object") {
        CHECK_THROWS_AS(parse_json_body("{not json"), InvalidRequest);
        CHECK_THROWS_AS(parse_json_body("[1, 2]"), InvalidRequest);
        CHECK(parse_json_body(R"({"a": 1})")["a"] == 1);
    }

    TEST_CASE("absent sampling fields take the model defaults, present ones override") {
        xs::SamplingParams defaults;
        defaults.temperature = 0.6f;
        defaults.top_k = 64;
        const auto r = parse_chat_request(chat_body({{"top_p", 0.9}, {"seed", 7}, {"temperature", nullptr}}), defaults);
        CHECK(r.options.sampling.temperature == doctest::Approx(0.6));
        CHECK(r.options.sampling.top_k == 64);
        CHECK(r.options.sampling.top_p == doctest::Approx(0.9));
        CHECK(r.options.sampling.seed == 7u);
        CHECK_FALSE(r.options.max_tokens.has_value());
        CHECK_FALSE(r.options.stream);
    }

    TEST_CASE("all generation options are read") {
        const auto r = parse_chat_request(chat_body({{"max_tokens", 10},
                                                     {"max_completion_tokens", 20},
                                                     {"top_k", -1},
                                                     {"min_p", 0.05},
                                                     {"presence_penalty", 0.5},
                                                     {"frequency_penalty", -0.5},
                                                     {"repetition_penalty", 1.1},
                                                     {"stop", "###"},
                                                     {"n", 1},
                                                     {"stream", true},
                                                     {"stream_options", {{"include_usage", true}}},
                                                     {"response_format", {{"type", "text"}}},
                                                     {"unknown_field", "ignored"}}),
                                          {});
        CHECK(r.options.max_tokens == 20);
        CHECK(r.options.sampling.top_k == 0);
        CHECK(r.options.sampling.min_p == doctest::Approx(0.05));
        CHECK(r.options.sampling.presence_penalty == doctest::Approx(0.5));
        CHECK(r.options.sampling.frequency_penalty == doctest::Approx(-0.5));
        CHECK(r.options.sampling.repetition_penalty == doctest::Approx(1.1));
        CHECK(r.options.stop == std::vector<std::string>{"###"});
        CHECK(r.options.stream);
        CHECK(r.options.include_usage);
    }

    TEST_CASE("tools, tool_choice and chat_template_kwargs") {
        const json tools = json::array({{{"type", "function"}, {"function", {{"name", "f"}}}}});
        CHECK(parse_chat_request(chat_body({{"tools", tools}}), {}).tools == tools);
        CHECK(parse_chat_request(chat_body({{"tools", tools}, {"tool_choice", "auto"}}), {}).tools == tools);
        CHECK(parse_chat_request(chat_body({{"tools", tools}, {"tool_choice", "none"}}), {}).tools.is_null());
        CHECK(parse_chat_request(chat_body({{"tools", json::array()}}), {}).tools.is_null());
        const auto r = parse_chat_request(chat_body({{"chat_template_kwargs", {{"enable_thinking", true}}}}), {});
        CHECK(r.template_kwargs["enable_thinking"] == true);
        CHECK(parse_chat_request(chat_body(json::object()), {}).template_kwargs == json::object());
    }

    TEST_CASE("invalid fields are rejected by name") {
        CHECK(rejected_param({{"model", "m"}}) == "messages");
        CHECK(rejected_param(chat_body({{"messages", json::array()}})) == "messages");
        CHECK(rejected_param(chat_body({{"messages", json::array({1})}})) == "messages[0]");
        CHECK(rejected_param(chat_body({{"messages", json::array({{{"content", "x"}}})}})) == "messages[0].role");
        CHECK(rejected_param(chat_body({{"n", 2}})) == "n");
        CHECK(rejected_param(chat_body({{"temperature", "hot"}})) == "temperature");
        CHECK(rejected_param(chat_body({{"temperature", -1}})) == "temperature");
        CHECK(rejected_param(chat_body({{"top_p", 1.5}})) == "top_p");
        CHECK(rejected_param(chat_body({{"top_k", 2.5}})) == "top_k");
        CHECK(rejected_param(chat_body({{"seed", -3}})) == "seed");
        CHECK(rejected_param(chat_body({{"max_tokens", 0}})) == "max_tokens");
        CHECK(rejected_param(chat_body(parse_json_body(R"({"max_tokens": 0})"))) == "max_tokens");
        CHECK(rejected_param(chat_body(parse_json_body(R"({"n": 3})"))) == "n");
        CHECK(rejected_param(chat_body({{"max_completion_tokens", "10"}})) == "max_completion_tokens");
        CHECK(rejected_param(chat_body({{"stop", 5}})) == "stop");
        CHECK(rejected_param(chat_body({{"stop", json::array({"a", 1})}})) == "stop");
        CHECK(rejected_param(chat_body({{"stream", "yes"}})) == "stream");
        CHECK(rejected_param(chat_body({{"stream_options", {{"include_usage", 1}}}})) ==
              "stream_options.include_usage");
        CHECK(rejected_param(chat_body({{"tools", "f"}})) == "tools");
        CHECK(rejected_param(chat_body({{"tool_choice", "required"}})) == "tool_choice");
        CHECK(rejected_param(chat_body({{"tool_choice", "sometimes"}})) == "tool_choice");
        CHECK(rejected_param(chat_body({{"chat_template_kwargs", "x"}})) == "chat_template_kwargs");
    }

    TEST_CASE("structured response formats are refused as not supported yet") {
        for (const char* type : {"json_object", "json_schema"}) {
            try {
                parse_chat_request(chat_body({{"response_format", {{"type", type}}}}), {});
                FAIL("accepted response_format ", type);
            } catch (const InvalidRequest& e) {
                CHECK(e.param() == "response_format");
                CHECK(std::string(e.what()).find("not supported yet") != std::string::npos);
            }
        }
    }

    TEST_CASE("completion prompt is a string or a one-string array") {
        CHECK(parse_completion_request({{"prompt", "abc"}}, {}).prompt == "abc");
        CHECK(parse_completion_request({{"prompt", json::array({"xyz"})}}, {}).prompt == "xyz");
        CHECK_THROWS_AS(parse_completion_request({{"prompt", json::array({"a", "b"})}}, {}), InvalidRequest);
        CHECK_THROWS_AS(parse_completion_request(json::object(), {}), InvalidRequest);
        CHECK_THROWS_AS(parse_completion_request({{"prompt", "a"}, {"n", 3}}, {}), InvalidRequest);
    }

    TEST_CASE("image_url parts: data URLs are decoded, the part is reduced for the template, the rest refused") {
        const json image = {{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,iVBORw0K"}}}};
        const json text = {{"type", "text"}, {"text", "what is this?"}};
        const json messages = json::array(
            {{{"role", "system"}, {"content", "be brief"}},
             {{"role", "user"}, {"content", json::array({image, text, {{"type", "image_url"}, {"image_url", "data:image/jpeg;base64,/9j/"}}})}}});
        const auto r = parse_chat_request(chat_body({{"messages", messages}}), {});
        REQUIRE(r.images.size() == 2);
        CHECK(r.images[0].param == "messages[1].content[0]");
        CHECK(r.images[0].bytes == std::vector<uint8_t>{0x89, 'P', 'N', 'G', '\r', '\n'});
        CHECK(r.images[1].param == "messages[1].content[2]");
        CHECK(r.messages[1]["content"][0] == json{{"type", "image"}});
        CHECK(r.messages[1]["content"][1] == text);

        const auto param_of = [&text](const json& part) {
            return rejected_param(chat_body(
                {{"messages", json::array({{{"role", "user"}, {"content", json::array({text, part})}}})}}));
        };
        const std::string where = "messages[0].content[1]";
        CHECK(param_of({{"type", "image_url"}, {"image_url", {{"url", "https://example.com/cat.png"}}}}) == where);
        CHECK(param_of({{"type", "image_url"}, {"image_url", {{"url", "data:image/png,rawbytes"}}}}) == where);
        CHECK(param_of({{"type", "image_url"}, {"image_url", {{"url", "data:text/plain;base64,aGk="}}}}) == where);
        CHECK(param_of({{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,!!"}}}}) == where);
        CHECK(param_of({{"type", "image_url"}, {"image_url", 42}}) == where);
        CHECK(param_of({{"type", "image"}, {"image", "cat.png"}}) == where);
        CHECK(param_of({{"type", "input_audio"}, {"input_audio", {{"data", "AAAA"}, {"format", "wav"}}}}) == where);
    }

    TEST_CASE("request JSON keeps the client's key order, which templates can print") {
        const auto body = parse_json_body(
            R"({"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function":
               {"name": "f", "parameters": {"type": "object", "properties": {"zeta": {}, "alpha": {}, "mid": {}}}}}]})");
        const auto r = parse_chat_request(body, {});
        std::string order;
        for (const auto& [key, value] : r.tools[0]["function"]["parameters"]["properties"].items()) order += key + " ";
        CHECK(order == "zeta alpha mid ");
    }
}
