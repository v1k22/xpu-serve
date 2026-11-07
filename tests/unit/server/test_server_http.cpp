#include <doctest/doctest.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <httplib/httplib.h>

#include "server/server.h"
#include "unit/server/fake_service.h"

using nlohmann::json;
using namespace std::chrono_literals;

namespace {

// A server on an ephemeral port, serving `service` until destroyed.
class RunningServer {
public:
    explicit RunningServer(xs::InferenceService& service, std::string api_key = {})
        : server_(service, options(std::move(api_key))), thread_([this] { server_.run(); }) {
        REQUIRE(server_.wait_until_listening(5000));
    }
    ~RunningServer() {
        server_.stop();
        thread_.join();
    }
    httplib::Client client() const {
        httplib::Client c("127.0.0.1", server_.bound_port());
        c.set_read_timeout(10, 0);
        return c;
    }

private:
    static xs::server::ServerOptions options(std::string api_key) {
        xs::server::ServerOptions o;
        o.port = 0;
        o.api_key = std::move(api_key);
        o.http_threads = 4;
        return o;
    }
    xs::server::Server server_;
    std::thread thread_;
};

json chat_body(json extra = json::object()) {
    json body = {{"model", "fake-model"}, {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
    body.update(extra);
    return body;
}

json post_json(httplib::Client& client, const std::string& path, const json& body, int expected_status = 200) {
    const auto res = client.Post(path, body.dump(), "application/json");
    REQUIRE(res);
    CHECK(res->status == expected_status);
    return json::parse(res->body);
}

// Parses an SSE body into its JSON events; `done` tells whether it ended with [DONE].
std::vector<json> parse_sse(const std::string& body, bool& done) {
    std::vector<json> events;
    done = false;
    size_t pos = 0;
    while (pos < body.size()) {
        const size_t end = body.find("\n\n", pos);
        REQUIRE(end != std::string::npos);
        const std::string event = body.substr(pos, end - pos);
        REQUIRE(event.rfind("data: ", 0) == 0);
        const std::string data = event.substr(6);
        CHECK_FALSE(done);  // nothing after [DONE]
        if (data == "[DONE]") {
            done = true;
        } else {
            events.push_back(json::parse(data));
        }
        pos = end + 2;
    }
    return events;
}

std::vector<json> post_sse(httplib::Client& client, const std::string& path, const json& body) {
    const auto res = client.Post(path, body.dump(), "application/json");
    REQUIRE(res);
    CHECK(res->status == 200);
    CHECK(res->get_header_value("Content-Type") == "text/event-stream");
    bool done = false;
    auto events = parse_sse(res->body, done);
    CHECK(done);
    return events;
}

std::string streamed_content(const std::vector<json>& events, const char* field = "content") {
    std::string out;
    for (const json& e : events) {
        if (e["choices"].empty()) continue;
        out += e["choices"][0]["delta"].value(field, "");
    }
    return out;
}

bool wait_for(const std::atomic<bool>& flag, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!flag && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(5ms);
    return flag;
}

std::string decoded_prompt(const xs::test::FakeService& service) {
    const auto prompt = service.last_request().prompt;
    return {prompt.begin(), prompt.end()};
}

}  // namespace

TEST_SUITE("server.http") {
    TEST_CASE("health and models") {
        xs::test::FakeService service;
        RunningServer server(service);
        auto client = server.client();
        auto res = client.Get("/health");
        REQUIRE(res);
        CHECK(json::parse(res->body) == json{{"status", "ok"}});
        res = client.Get("/v1/models");
        REQUIRE(res);
        const json models = json::parse(res->body);
        CHECK(models["data"][0]["id"] == "fake-model");
    }

    TEST_CASE("non-stream chat completion") {
        xs::test::FakeService service;
        RunningServer server(service);
        auto client = server.client();
        const json r = post_json(client, "/v1/chat/completions", chat_body({{"seed", 3}}));
        CHECK(r["object"] == "chat.completion");
        CHECK(r["model"] == "fake-model");
        CHECK(r["choices"][0]["message"]["role"] == "assistant");
        CHECK(r["choices"][0]["message"]["content"] == "Hello!");
        CHECK(r["choices"][0]["finish_reason"] == "stop");
        CHECK(r["usage"]["prompt_tokens"] == int(std::string("user: hi\nassistant:").size()));
        CHECK(r["usage"]["completion_tokens"] == 6);
        CHECK(r.contains("timings"));
        CHECK(decoded_prompt(service) == "user: hi\nassistant:");
        const auto req = service.last_request();
        CHECK(req.sampling.temperature == doctest::Approx(0.7));  // the card's default
        CHECK(req.sampling.seed == 3u);
        CHECK(req.max_new_tokens == 4096 - int(req.prompt.size()));
    }

    TEST_CASE("tools and chat_template_kwargs reach the template; tool_choice none drops tools") {
        xs::test::FakeService service;
        RunningServer server(service);
        auto client = server.client();
        const json tools = json::array({{{"type", "function"}, {"function", {{"name", "f"}}}}});
        post_json(client, "/v1/chat/completions",
                  chat_body({{"tools", tools}, {"chat_template_kwargs", {{"enable_thinking", true}}}}));
        CHECK(decoded_prompt(service) == "user: hi\n[tools:1][think]assistant:");
        post_json(client, "/v1/chat/completions", chat_body({{"tools", tools}, {"tool_choice", "none"}}));
        CHECK(decoded_prompt(service) == "user: hi\nassistant:");
    }

    TEST_CASE("streamed chat completion with usage") {
        xs::test::FakeService service;
        RunningServer server(service);
        auto client = server.client();
        const auto events = post_sse(client, "/v1/chat/completions",
                                     chat_body({{"stream", true}, {"stream_options", {{"include_usage", true}}}}));
        REQUIRE(events.size() >= 3);
        CHECK(events.front()["object"] == "chat.completion.chunk");
        CHECK(events.front()["choices"][0]["delta"]["role"] == "assistant");
        CHECK(streamed_content(events) == "Hello!");
        const json& finish = events[events.size() - 2];
        CHECK(finish["choices"][0]["finish_reason"] == "stop");
        const json& usage = events.back();
        CHECK(usage["choices"].empty());
        CHECK(usage["usage"]["completion_tokens"] == 6);
        for (size_t i = 0; i + 1 < events.size(); ++i) CHECK_FALSE(events[i].contains("usage"));
    }

    TEST_CASE("streamed chat without include_usage ends with the finish chunk") {
        xs::test::FakeService service;
        RunningServer server(service);
        auto client = server.client();
        const auto events = post_sse(client, "/v1/chat/completions", chat_body({{"stream", true}}));
        CHECK(events.back()["choices"][0]["finish_reason"] == "stop");
    }

    TEST_CASE("max_tokens ends with finish_reason length") {
        xs::test::FakeService service;
        RunningServer server(service);
        auto client = server.client();
        const json r = post_json(client, "/v1/chat/completions", chat_body({{"max_tokens", 3}}));
        CHECK(r["choices"][0]["message"]["content"] == "Hel");
        CHECK(r["choices"][0]["finish_reason"] == "length");
        CHECK(service.last_request().max_new_tokens == 3);
    }

    TEST_CASE("stop string split across tokens cuts the reply and cancels generation") {
        xs::test::FakeService service;
        service.reply = "one two STOP three";
        RunningServer server(service);
        auto client = server.client();
        const json r = post_json(client, "/v1/chat/completions", chat_body({{"stop", json::array({"STOP"})}}));
        CHECK(r["choices"][0]["message"]["content"] == "one two ");
        CHECK(r["choices"][0]["finish_reason"] == "stop");
        CHECK(r["usage"]["completion_tokens"] == 12);  // the engine stopped at the match
        CHECK(service.cancelled);
    }

    TEST_CASE("stop string matching the very first token streams no content") {
        xs::test::FakeService service;
        service.reply = "Xyz";
        RunningServer server(service);
        auto client = server.client();
        const auto events = post_sse(client, "/v1/chat/completions", chat_body({{"stream", true}, {"stop", "X"}}));
        CHECK(streamed_content(events).empty());
        CHECK(events.back()["choices"][0]["finish_reason"] == "stop");
        CHECK(service.cancelled);
    }

    TEST_CASE("text completions, plain and streamed") {
        xs::test::FakeService service;
        service.reply = "world";
        RunningServer server(service);
        auto client = server.client();
        const json r = post_json(client, "/v1/completions", {{"prompt", "hello "}, {"max_tokens", 4}});
        CHECK(r["object"] == "text_completion");
        CHECK(r["choices"][0]["text"] == "worl");
        CHECK(r["choices"][0]["finish_reason"] == "length");
        CHECK(decoded_prompt(service) == "hello ");  // raw prompt, no template

        const auto events = post_sse(client, "/v1/completions",
                                     {{"prompt", "hello "},
                                      {"stream", true},
                                      {"stop", "rl"},
                                      {"stream_options", {{"include_usage", true}}}});
        std::string text;
        for (const json& e : events) {
            if (!e["choices"].empty()) text += e["choices"][0]["text"].get<std::string>();
        }
        CHECK(text == "wo");
        CHECK(events[events.size() - 2]["choices"][0]["finish_reason"] == "stop");
        CHECK(events.back()["usage"]["prompt_tokens"] == 6);
    }

    TEST_CASE("invalid requests get 400 naming the field") {
        xs::test::FakeService service;
        RunningServer server(service);
        auto client = server.client();
        auto res = client.Post("/v1/chat/completions", "{bad json", "application/json");
        REQUIRE(res);
        CHECK(res->status == 400);
        CHECK(json::parse(res->body)["error"]["type"] == "invalid_request_error");

        CHECK(post_json(client, "/v1/chat/completions", chat_body({{"n", 2}}), 400)["error"]["param"] == "n");
        CHECK(post_json(client, "/v1/chat/completions", chat_body({{"max_tokens", "ten"}}), 400)["error"]["param"] ==
              "max_tokens");
        CHECK(post_json(client, "/v1/chat/completions", chat_body({{"messages", "hi"}}), 400)["error"]["param"] ==
              "messages");
        const json boom = chat_body({{"messages", json::array({{{"role", "user"}, {"content", "BOOM"}}})}});
        const json e = post_json(client, "/v1/chat/completions", boom, 400)["error"];
        CHECK(e["param"] == "messages");
        CHECK(e["message"].get<std::string>().find("template exploded") != std::string::npos);
        CHECK(post_json(client, "/v1/completions", json{{"prompt", 5}}, 400)["error"]["param"] == "prompt");
        CHECK(service.calls() == 0);
    }

    TEST_CASE("image parts reach the engine as media with their placeholders expanded") {
        // A 2x1 grayscale PNG.
        const std::string png =
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAAAAADRSSBWAAAAC0lEQVR4nGNg/wUAAQsBAsdd+yMAAAAASUVORK5CYII=";
        const auto user_with = [](const json& image_part) {
            return json::array({{{"role", "user"},
                                 {"content", json::array({{{"type", "text"}, {"text", "see"}}, image_part})}}});
        };
        const json image = {{"type", "image_url"}, {"image_url", {{"url", png}}}};
        xs::test::FakeService service;
        RunningServer running(service);
        auto client = running.client();

        // A model without vision refuses images, naming the part.
        json err = post_json(client, "/v1/chat/completions", chat_body({{"messages", user_with(image)}}), 400);
        CHECK(err["error"]["param"] == "messages[0].content[1]");

        service.images = true;
        post_json(client, "/v1/chat/completions", chat_body({{"messages", user_with(image)}}));
        const xs::GenerateRequest req = service.last_request();
        REQUIRE(req.media.size() == 1);
        CHECK(std::count(req.prompt.begin(), req.prompt.end(), xs::test::kImageByte) == 2);

        // Undecodable image bytes and a stray placeholder in the text are 400s too.
        const json broken = {{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,iVBORw0KGgo="}}}};
        err = post_json(client, "/v1/chat/completions", chat_body({{"messages", user_with(broken)}}), 400);
        CHECK(err["error"]["param"] == "messages[0].content[1]");
        CHECK(err["error"]["message"].get<std::string>().find("could not be decoded") != std::string::npos);
        const json stray = json::array({{{"role", "user"}, {"content", std::string("look ") + xs::test::kImageByte}}});
        err = post_json(client, "/v1/chat/completions", chat_body({{"messages", stray}}), 400);
        CHECK(err["error"]["param"] == "messages");
    }

    TEST_CASE("prompt longer than the context is refused") {
        xs::test::FakeService service;
        service.card().context_length = 8;
        RunningServer server(service);
        auto client = server.client();
        const json e = post_json(client, "/v1/completions", {{"prompt", "a long prompt"}}, 400)["error"];
        CHECK(e["param"] == "prompt");
        post_json(client, "/v1/completions", {{"prompt", "abc"}, {"max_tokens", 100}});
        CHECK(service.last_request().max_new_tokens == 5);
    }

    TEST_CASE("engine errors are 500 non-stream and an error event when streaming") {
        xs::test::FakeService service;
        service.error = "device lost";
        RunningServer server(service);
        auto client = server.client();
        const json e = post_json(client, "/v1/chat/completions", chat_body(), 500)["error"];
        CHECK(e["message"] == "device lost");
        CHECK(e["type"] == "server_error");
        const auto events = post_sse(client, "/v1/chat/completions", chat_body({{"stream", true}}));
        CHECK(events.back()["error"]["message"] == "device lost");
    }

    TEST_CASE("unknown routes are 404 in the OpenAI error shape") {
        xs::test::FakeService service;
        RunningServer server(service);
        auto client = server.client();
        const auto res = client.Get("/v1/nope");
        REQUIRE(res);
        CHECK(res->status == 404);
        CHECK(json::parse(res->body)["error"]["message"].get<std::string>().find("/v1/nope") != std::string::npos);
    }

    TEST_CASE("api key is required when configured") {
        xs::test::FakeService service;
        RunningServer server(service, "secret");
        auto client = server.client();
        auto res = client.Get("/v1/models");
        REQUIRE(res);
        CHECK(res->status == 401);
        CHECK(json::parse(res->body)["error"]["code"] == "invalid_api_key");
        res = client.Post("/v1/chat/completions", {{"Authorization", "Bearer wrong"}}, chat_body().dump(),
                          "application/json");
        REQUIRE(res);
        CHECK(res->status == 401);
        res = client.Post("/v1/chat/completions", {{"Authorization", "Bearer secret"}}, chat_body().dump(),
                          "application/json");
        REQUIRE(res);
        CHECK(res->status == 200);
        res = client.Get("/health");
        REQUIRE(res);
        CHECK(res->status == 200);
    }

    TEST_CASE("concurrent requests queue and both complete") {
        xs::test::FakeService service;
        service.token_delay = 5ms;
        RunningServer server(service);
        std::vector<std::string> contents(2);
        std::vector<std::thread> clients;
        for (size_t i = 0; i < contents.size(); ++i) {
            clients.emplace_back([&, i] {
                auto client = server.client();
                const auto res = client.Post("/v1/chat/completions", chat_body().dump(), "application/json");
                if (res && res->status == 200) contents[i] = json::parse(res->body)["choices"][0]["message"]["content"];
            });
        }
        for (auto& t : clients) t.join();
        CHECK(contents[0] == "Hello!");
        CHECK(contents[1] == "Hello!");
        CHECK(service.calls() == 2);
    }

    TEST_CASE("client disconnect during a stream cancels generation") {
        xs::test::FakeService service;
        service.reply = std::string(500, 'x');
        service.token_delay = 2ms;
        RunningServer server(service);
        auto client = server.client();
        int chunks = 0;
        client.Post("/v1/chat/completions", httplib::Headers{}, chat_body({{"stream", true}}).dump(),
                    "application/json", [&](const char*, size_t) { return ++chunks < 3; });
        CHECK(wait_for(service.cancelled, 3s));
        CHECK(service.last_request().prompt.size() > 0);
    }

    TEST_CASE("client disconnect during a non-stream request cancels generation") {
        xs::test::FakeService service;
        service.reply = std::string(500, 'x');
        service.token_delay = 2ms;
        RunningServer server(service);
        auto client = server.client();
        client.set_read_timeout(0, 100'000);  // give up after 100 ms and close the connection
        const auto res = client.Post("/v1/chat/completions", chat_body().dump(), "application/json");
        CHECK_FALSE(res);
        client.stop();
        CHECK(wait_for(service.cancelled, 3s));
    }

    TEST_CASE("stop before run returns immediately") {
        xs::test::FakeService service;
        xs::server::ServerOptions o;
        o.port = 0;
        xs::server::Server server(service, o);
        server.stop();
        server.run();
        CHECK_FALSE(server.wait_until_listening(10));
    }
}
