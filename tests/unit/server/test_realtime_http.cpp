// GET /v1/realtime end to end: a WebSocket to a running server with a fake speech recognition service
// (fake_service.h): session.created, a turn committed and transcribed; the "realtime" subprotocol; the API key as a
// header or as OpenAI's browser subprotocol; no route for a model that does not transcribe.
#include <doctest/doctest.h>

#include <string>
#include <thread>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "server/server.h"
#include "unit/server/fake_service.h"
#include "util/base64.h"

using nlohmann::json;

namespace {

class RunningServer {
public:
    RunningServer(xs::InferenceService& service, std::string api_key = {})
        : server_(service, options(std::move(api_key))), thread_([this] { server_.run(); }) {
        REQUIRE(server_.wait_until_listening(5000));
    }
    ~RunningServer() {
        server_.stop();
        thread_.join();
    }
    std::string url(const std::string& query = "?intent=transcription") const {
        return "ws://127.0.0.1:" + std::to_string(server_.bound_port()) + "/v1/realtime" + query;
    }

private:
    static xs::server::ServerOptions options(std::string api_key) {
        xs::server::ServerOptions o;
        o.port = 0;
        o.http_threads = 4;
        o.api_key = std::move(api_key);
        return o;
    }
    xs::server::Server server_;
    std::thread thread_;
};

// The next event of `type`, skipping others; null after a failed read.
json next(httplib::ws::WebSocketClient& ws, const std::string& type) {
    std::string message;
    while (ws.read(message) != httplib::ws::Fail) {
        const json event = json::parse(message);
        if (event["type"] == type) return event;
    }
    return nullptr;
}

std::string append(int ms) {
    std::vector<uint8_t> bytes(static_cast<size_t>(ms * 24 * 2));
    for (size_t i = 0; i < bytes.size(); i += 2) bytes[i + 1] = (i / 2) % 48 < 24 ? 0x10 : 0xf0;  // a square wave
    return json{{"type", "input_audio_buffer.append"}, {"audio", xs::base64_encode(bytes)}}.dump();
}

}  // namespace

TEST_SUITE("realtime http") {
    TEST_CASE("a session over a WebSocket: created, a committed turn transcribed, the realtime subprotocol") {
        xs::test::FakeService service;
        service.transcriber = true;
        service.reply = "Over the wire.";
        RunningServer server(service);
        httplib::ws::WebSocketClient ws(server.url(), {{"Sec-WebSocket-Protocol", "realtime"}});
        ws.set_read_timeout(5);
        REQUIRE(ws.connect());
        CHECK(ws.subprotocol() == "realtime");
        CHECK(next(ws, "session.created")["session"]["type"] == "transcription");
        REQUIRE(ws.send(R"({"type": "session.update", "session": {"type": "transcription",
                            "audio": {"input": {"turn_detection": null}}}})"));
        CHECK(next(ws, "session.updated")["session"]["audio"]["input"]["turn_detection"].is_null());
        REQUIRE(ws.send(append(500)));
        REQUIRE(ws.send(R"({"type": "input_audio_buffer.commit"})"));
        const json done = next(ws, "conversation.item.input_audio_transcription.completed");
        CHECK(done["transcript"] == "Over the wire.");
        CHECK(service.last_transcription().clip.samples.size() == 12000);
        REQUIRE(ws.send("not json"));
        CHECK(next(ws, "error")["error"]["code"] == "invalid_json");
        ws.close();
    }

    TEST_CASE("the API key: refused without it; as a bearer header; as OpenAI's browser subprotocol") {
        xs::test::FakeService service;
        service.transcriber = true;
        RunningServer server(service, "sekrit");
        httplib::ws::WebSocketClient none(server.url());
        CHECK_FALSE(none.connect());
        httplib::ws::WebSocketClient header(server.url(), {{"Authorization", "Bearer sekrit"}});
        header.set_read_timeout(5);
        REQUIRE(header.connect());
        CHECK_FALSE(next(header, "session.created").is_null());
        httplib::ws::WebSocketClient browser(server.url(),
                                             {{"Sec-WebSocket-Protocol", "realtime, openai-insecure-api-key.sekrit"}});
        browser.set_read_timeout(5);
        REQUIRE(browser.connect());
        CHECK(browser.subprotocol() == "realtime");
        CHECK_FALSE(next(browser, "session.created").is_null());
        httplib::ws::WebSocketClient wrong(server.url(), {{"Sec-WebSocket-Protocol", "openai-insecure-api-key.nope"}});
        CHECK_FALSE(wrong.connect());
    }

    TEST_CASE("another intent is refused; a model that does not transcribe has no realtime route") {
        xs::test::FakeService service;
        service.transcriber = true;
        RunningServer server(service);
        httplib::ws::WebSocketClient other(server.url("?intent=conversation"));
        other.set_read_timeout(5);
        REQUIRE(other.connect());
        CHECK(next(other, "error")["error"]["param"] == "intent");

        xs::test::FakeService chat;
        RunningServer plain(chat);
        httplib::ws::WebSocketClient ws(plain.url());
        CHECK_FALSE(ws.connect());
    }
}
