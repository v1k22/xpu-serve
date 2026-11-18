// POST /v1/audio/transcriptions end to end over HTTP with a fake speech recognition service (fake_service.h): the
// form's fields reach the service, json, text and streamed responses, and every refusal names its field.
#include <doctest/doctest.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <httplib/httplib.h>

#include "server/server.h"
#include "unit/server/fake_service.h"

using nlohmann::json;

namespace {

class RunningServer {
public:
    explicit RunningServer(xs::InferenceService& service)
        : server_(service, options()), thread_([this] { server_.run(); }) {
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
    static xs::server::ServerOptions options() {
        xs::server::ServerOptions o;
        o.port = 0;
        o.http_threads = 2;
        return o;
    }
    xs::server::Server server_;
    std::thread thread_;
};

const std::string kWav = "RIFF....WAVEfmt ";  // the fake service does not decode it

httplib::UploadFormDataItems form(std::vector<std::pair<std::string, std::string>> fields, bool with_file = true) {
    httplib::UploadFormDataItems items;
    if (with_file) items.push_back({"file", kWav, "clip.wav", "audio/wav"});
    for (auto& [name, value] : fields) items.push_back({name, value, "", ""});
    return items;
}

// The 400's body names `param`.
void check_refused(httplib::Client& c, const httplib::UploadFormDataItems& items, const std::string& param) {
    const auto res = c.Post("/v1/audio/transcriptions", items);
    REQUIRE(res);
    CHECK(res->status == 400);
    CHECK(json::parse(res->body)["error"]["param"] == param);
}

}  // namespace

TEST_SUITE("transcriptions http") {
    TEST_CASE("json and text responses; the form's fields reach the service") {
        xs::test::FakeService service;
        service.transcriber = true;
        service.reply = "Hello from the fake transcriber.";
        RunningServer server(service);
        auto c = server.client();
        const auto res = c.Post("/v1/audio/transcriptions",
                                form({{"model", "fake-model"}, {"language", "en"}, {"prompt", "names: Bhavik"},
                                      {"temperature", "0"}}));
        REQUIRE(res);
        REQUIRE(res->status == 200);
        const json body = json::parse(res->body);
        CHECK(body["text"] == service.reply);
        CHECK(body["usage"]["type"] == "tokens");
        CHECK(body["usage"]["input_tokens"] == 30);
        CHECK(body["usage"]["output_tokens"] == 7);
        CHECK(body["usage"]["total_tokens"] == 37);
        const xs::TranscriptionRequest r = service.last_transcription();
        CHECK(r.language == "en");
        CHECK(r.prompt == "names: Bhavik");
        CHECK(r.sampling.temperature == 0.0f);
        CHECK(std::string(r.audio.begin(), r.audio.end()) == kWav);

        const auto text = c.Post("/v1/audio/transcriptions", form({{"response_format", "text"}}));
        REQUIRE(text);
        CHECK(text->status == 200);
        CHECK(text->body == service.reply);
        CHECK(text->get_header_value("Content-Type").rfind("text/plain", 0) == 0);
        CHECK(service.last_transcription().sampling.temperature == 0.7f);  // the model's default
    }

    TEST_CASE("stream: transcript.text.delta events, then transcript.text.done with the text and usage") {
        xs::test::FakeService service;
        service.transcriber = true;
        service.reply = "Streaming works.";
        RunningServer server(service);
        auto c = server.client();
        const auto res = c.Post("/v1/audio/transcriptions", form({{"stream", "true"}}));
        REQUIRE(res);
        REQUIRE(res->status == 200);
        std::string deltas;
        json done;
        size_t pos = 0;
        while (pos < res->body.size()) {
            const size_t end = res->body.find("\n\n", pos);
            REQUIRE(end != std::string::npos);
            REQUIRE(res->body.compare(pos, 6, "data: ") == 0);
            const json event = json::parse(res->body.substr(pos + 6, end - pos - 6));
            if (event["type"] == "transcript.text.delta") {
                CHECK(done.is_null());
                deltas += event["delta"].get<std::string>();
            } else {
                CHECK(event["type"] == "transcript.text.done");
                done = event;
            }
            pos = end + 2;
        }
        CHECK(deltas == service.reply);
        CHECK(done["text"] == service.reply);
        CHECK(done["usage"]["output_tokens"] == 7);
    }

    TEST_CASE("refusals name the field; translations and a model that does not transcribe are refused") {
        xs::test::FakeService service;
        service.transcriber = true;
        RunningServer server(service);
        auto c = server.client();
        check_refused(c, form({{"model", "fake-model"}}, /*with_file=*/false), "file");
        check_refused(c, form({{"response_format", "srt"}}), "response_format");
        check_refused(c, form({{"response_format", "verbose_json"}}), "response_format");
        check_refused(c, form({{"response_format", "xml"}}), "response_format");
        check_refused(c, form({{"timestamp_granularities[]", "word"}}), "timestamp_granularities[]");
        check_refused(c, form({{"chunking_strategy", "auto"}}), "chunking_strategy");
        check_refused(c, form({{"temperature", "2"}}), "temperature");
        check_refused(c, form({{"temperature", "warm"}}), "temperature");
        check_refused(c, form({{"stream", "yes"}}), "stream");
        const auto translate = c.Post("/v1/audio/translations", form({}));
        REQUIRE(translate);
        CHECK(translate->status == 400);
        const auto json_body = c.Post("/v1/audio/transcriptions", R"({"file": "x"})", "application/json");
        REQUIRE(json_body);
        CHECK(json_body->status == 400);

        service.error = "audio: Ogg files are not supported";
        check_refused(c, form({}), "file");

        xs::test::FakeService chat_only;
        RunningServer plain(chat_only);
        auto p = plain.client();
        const auto res = p.Post("/v1/audio/transcriptions", form({}));
        REQUIRE(res);
        CHECK(res->status == 404);
    }
}
