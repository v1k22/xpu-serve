// A realtime transcription session (server/realtime_session.h) with a fake speech recognition service: the events of
// OpenAI's GA transcription session in order, turns committed by the client and by the server's VAD, the audio and
// settings each turn hands the service, and every refusal as an error event naming its field.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>

#include "server/realtime_session.h"
#include "unit/server/fake_service.h"
#include "util/base64.h"

using nlohmann::json;
using xs::server::RealtimeSession;

namespace {

// The events a session sends, and a wait for one of a type.
class Events {
public:
    RealtimeSession::Send sink() {
        return [this](const nlohmann::ordered_json& e) {
            std::lock_guard lock(mutex_);
            events_.push_back(json::parse(e.dump()));
            ready_.notify_all();
        };
    }
    // The first event of `type` from `from` on; fails after 5 s.
    json wait(const std::string& type, size_t from = 0) {
        std::unique_lock lock(mutex_);
        json found;
        const bool ok = ready_.wait_for(lock, std::chrono::seconds(5), [&] {
            for (size_t i = from; i < events_.size(); ++i) {
                if (events_[i]["type"] == type) {
                    found = events_[i];
                    return true;
                }
            }
            return false;
        });
        REQUIRE_MESSAGE(ok, "no ", type, " event");
        return found;
    }
    std::vector<json> all() {
        std::lock_guard lock(mutex_);
        return events_;
    }
    std::vector<std::string> types() {
        std::vector<std::string> t;
        for (const json& e : all()) t.push_back(e["type"]);
        return t;
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::vector<json> events_;
};

// Noise at `db` dBFS, `ms` long, as 16-bit PCM at 24 kHz.
std::vector<int16_t> pcm(int ms, double db) {
    static std::mt19937 rng(5);
    std::normal_distribution<double> noise(0.0, 32768.0 * std::pow(10.0, db / 20.0));
    std::vector<int16_t> s(static_cast<size_t>(ms * 24));
    for (int16_t& v : s) v = static_cast<int16_t>(std::clamp(noise(rng), -32768.0, 32767.0));
    return s;
}

std::string append_event(const std::vector<int16_t>& samples) {
    std::vector<uint8_t> bytes;
    for (const int16_t v : samples) {
        bytes.push_back(static_cast<uint8_t>(v & 0xff));
        bytes.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    }
    return json{{"type", "input_audio_buffer.append"}, {"audio", xs::base64_encode(bytes)}}.dump();
}

// Appends in pieces of 100 ms, as a client streaming a microphone does.
void stream(RealtimeSession& s, const std::vector<int16_t>& samples) {
    for (size_t at = 0; at < samples.size(); at += 2400) {
        const auto end = std::min(samples.size(), at + 2400);
        s.handle(append_event({samples.begin() + static_cast<std::ptrdiff_t>(at),
                               samples.begin() + static_cast<std::ptrdiff_t>(end)}));
    }
}

std::string manual_update(const json& transcription = json::object()) {
    json input = {{"turn_detection", nullptr}};
    if (!transcription.empty()) input["transcription"] = transcription;
    return json{{"type", "session.update"}, {"session", {{"type", "transcription"}, {"audio", {{"input", input}}}}}}
        .dump();
}

// The param of the error event `message` gets.
std::string refused(RealtimeSession& s, Events& events, const std::string& message) {
    const size_t before = events.all().size();
    s.handle(message);
    const json error = events.wait("error", before)["error"];
    return error["param"].is_string() ? error["param"].get<std::string>() : "";
}

}  // namespace

TEST_SUITE("realtime session") {
    TEST_CASE("session.created first, with the defaults: PCM at 24 kHz, server VAD, the model") {
        xs::test::FakeService service;
        service.transcriber = true;
        Events events;
        RealtimeSession session(service, events.sink());
        const json created = events.wait("session.created");
        CHECK(created["event_id"].is_string());
        const json& s = created["session"];
        CHECK(s["type"] == "transcription");
        CHECK(s["audio"]["input"]["format"] == json{{"type", "audio/pcm"}, {"rate", 24000}});
        CHECK(s["audio"]["input"]["transcription"]["model"] == "fake-model");
        CHECK(s["audio"]["input"]["transcription"]["language"].is_null());
        const json& vad = s["audio"]["input"]["turn_detection"];
        CHECK(vad["type"] == "server_vad");
        CHECK(vad["threshold"] == 0.5);
        CHECK(vad["prefix_padding_ms"] == 300);
        CHECK(vad["silence_duration_ms"] == 500);
    }

    TEST_CASE("a turn the client commits: committed, item added, deltas, completed, item done; the service's input") {
        xs::test::FakeService service;
        service.transcriber = true;
        service.reply = "Hello there, realtime.";
        Events events;
        RealtimeSession session(service, events.sink());
        session.handle(manual_update({{"language", "en"}, {"prompt", "names: Bhavik"}}));
        const json updated = events.wait("session.updated");
        CHECK(updated["session"]["audio"]["input"]["turn_detection"].is_null());
        CHECK(updated["session"]["audio"]["input"]["transcription"]["language"] == "en");
        stream(session, pcm(1000, -20.0));
        session.handle(R"({"type": "input_audio_buffer.commit", "event_id": "c1"})");
        const json committed = events.wait("input_audio_buffer.committed");
        CHECK(committed["previous_item_id"].is_null());
        const std::string item = committed["item_id"];
        const json added = events.wait("conversation.item.added");
        CHECK(added["item"]["id"] == item);
        CHECK(added["item"]["role"] == "user");
        CHECK(added["item"]["content"][0]["type"] == "input_audio");
        const json completed = events.wait("conversation.item.input_audio_transcription.completed");
        CHECK(completed["item_id"] == item);
        CHECK(completed["transcript"] == service.reply);
        CHECK(completed["usage"] == json{{"type", "tokens"}, {"input_tokens", 30}, {"output_tokens", 7},
                                         {"total_tokens", 37}});
        CHECK(events.wait("conversation.item.done")["item"]["content"][0]["transcript"] == service.reply);
        std::string deltas;
        for (const json& e : events.all()) {
            if (e["type"] == "conversation.item.input_audio_transcription.delta") {
                CHECK(e["item_id"] == item);
                deltas += e["delta"].get<std::string>();
            }
        }
        CHECK(deltas == service.reply);
        const xs::TranscriptionRequest r = service.last_transcription();
        CHECK(r.clip.sample_rate == 24000);
        CHECK(r.clip.samples.size() == 24000);
        CHECK(r.language == "en");
        CHECK(r.prompt == "names: Bhavik");
        CHECK(r.sampling.temperature == 0.7f);  // the model's default

        // the next turn follows the first
        stream(session, pcm(300, -20.0));
        const size_t before = events.all().size();
        session.handle(R"({"type": "input_audio_buffer.commit"})");
        CHECK(events.wait("input_audio_buffer.committed", before)["previous_item_id"] == item);
        events.wait("conversation.item.done", before);
        CHECK(service.last_transcription().clip.samples.size() == 7200);
    }

    TEST_CASE("server VAD: speech_started, speech_stopped, then the turn's commit; the turn's audio is transcribed") {
        xs::test::FakeService service;
        service.transcriber = true;
        service.reply = "A turn.";
        Events events;
        RealtimeSession session(service, events.sink());
        stream(session, pcm(1000, -60.0));
        stream(session, pcm(1500, -20.0));
        stream(session, pcm(1000, -60.0));
        const json started = events.wait("input_audio_buffer.speech_started");
        const json stopped = events.wait("input_audio_buffer.speech_stopped");
        CHECK(started["audio_start_ms"] == 700);  // the prefix before speech at 1000 ms
        CHECK(stopped["audio_end_ms"] == 3000);   // speech to 2500 ms, then the silence
        CHECK(stopped["item_id"] == started["item_id"]);
        CHECK(events.wait("input_audio_buffer.committed")["item_id"] == started["item_id"]);
        CHECK(events.wait("conversation.item.input_audio_transcription.completed")["transcript"] == "A turn.");
        CHECK(service.last_transcription().clip.samples.size() == 2300 * 24);
        const auto types = events.types();
        const auto index = [&](const std::string& t) {
            return std::find(types.begin(), types.end(), t) - types.begin();
        };
        CHECK(index("input_audio_buffer.speech_started") < index("input_audio_buffer.speech_stopped"));
        CHECK(index("input_audio_buffer.speech_stopped") < index("input_audio_buffer.committed"));
        CHECK(index("conversation.item.added") < index("conversation.item.input_audio_transcription.delta"));

        // A commit during a turn ends it, under the turn's item.
        const size_t before = events.all().size();
        stream(session, pcm(800, -20.0));
        const json second = events.wait("input_audio_buffer.speech_started", before);
        session.handle(R"({"type": "input_audio_buffer.commit"})");
        CHECK(events.wait("input_audio_buffer.committed", before)["item_id"] == second["item_id"]);
        stream(session, pcm(1000, -60.0));
        for (const json& e : events.all()) CHECK(e["type"] != "error");
    }

    TEST_CASE("clear empties the buffer; a commit of under 100 ms is refused") {
        xs::test::FakeService service;
        service.transcriber = true;
        Events events;
        RealtimeSession session(service, events.sink());
        session.handle(manual_update());
        stream(session, pcm(500, -20.0));
        session.handle(R"({"type": "input_audio_buffer.clear"})");
        events.wait("input_audio_buffer.cleared");
        session.handle(R"({"type": "input_audio_buffer.commit", "event_id": "evt_9"})");
        const json error = events.wait("error")["error"];
        CHECK(error["code"] == "input_audio_buffer_commit_empty");
        CHECK(error["event_id"] == "evt_9");
        stream(session, pcm(50, -20.0));
        CHECK(refused(session, events, R"({"type": "input_audio_buffer.commit"})").empty());
    }

    TEST_CASE("a failed transcription is a failed event for its item") {
        xs::test::FakeService service;
        service.transcriber = true;
        service.error = "qwen3_asr: language 'xx' is not one the model knows";
        Events events;
        RealtimeSession session(service, events.sink());
        session.handle(manual_update());
        stream(session, pcm(400, -20.0));
        session.handle(R"({"type": "input_audio_buffer.commit"})");
        const json failed = events.wait("conversation.item.input_audio_transcription.failed");
        CHECK(failed["item_id"] == events.wait("input_audio_buffer.committed")["item_id"]);
        CHECK(failed["error"]["message"] == service.error);
    }

    TEST_CASE("refusals are error events naming the field; the session goes on") {
        xs::test::FakeService service;
        service.transcriber = true;
        Events events;
        RealtimeSession session(service, events.sink());
        const auto update = [](const json& input) {
            const json config = {{"type", "transcription"}, {"audio", {{"input", input}}}};
            return json{{"type", "session.update"}, {"session", config}}.dump();
        };
        const std::string in = "session.audio.input";
        CHECK(refused(session, events, "not json") == "");
        CHECK(refused(session, events, R"({"type": "response.create"})") == "type");
        CHECK(refused(session, events, R"({"no_type": 1})") == "type");
        CHECK(refused(session, events, R"({"type": "input_audio_buffer.append", "audio": "!!"})") == "audio");
        CHECK(refused(session, events, R"({"type": "input_audio_buffer.append", "audio": "AA=="})") == "audio");
        CHECK(refused(session, events, R"({"type": "session.update", "session": {"type": "realtime"}})") ==
              "session.type");
        CHECK(refused(session, events, R"({"type": "session.update", "session": {"voice": "x"}})") == "session.voice");
        CHECK(refused(session, events, update({{"turn_detection", {{"type", "semantic_vad"}}}})) ==
              in + ".turn_detection.type");
        CHECK(refused(session, events, update({{"turn_detection", {{"threshold", 2}}}})) ==
              in + ".turn_detection.threshold");
        CHECK(refused(session, events, update({{"turn_detection", {{"idle_timeout_ms", 5000}}}})) ==
              in + ".turn_detection.idle_timeout_ms");
        CHECK(refused(session, events, update({{"format", {{"type", "audio/pcmu"}}}})) == in + ".format.type");
        CHECK(refused(session, events, update({{"format", {{"type", "audio/pcm"}, {"rate", 16000}}}})) ==
              in + ".format.rate");
        CHECK(refused(session, events, update({{"noise_reduction", {{"type", "near_field"}}}})) ==
              in + ".noise_reduction");
        CHECK(refused(session, events, update({{"transcription", {{"keywords", {"xpu"}}}}})) ==
              in + ".transcription.keywords");
        CHECK(refused(session, events, update({{"transcription", nullptr}})) == in + ".transcription");
        // a refused update changes nothing
        session.handle(update({{"turn_detection", {{"type", "server_vad"}, {"silence_duration_ms", 200}}}}));
        CHECK(events.wait("session.updated")["session"]["audio"]["input"]["turn_detection"]["silence_duration_ms"] ==
              200);
    }

    TEST_CASE("the session ends cleanly with a transcription in progress") {
        xs::test::FakeService service;
        service.transcriber = true;
        service.reply = std::string(3000, 'x');
        Events events;
        {
            RealtimeSession session(service, events.sink());
            session.handle(manual_update());
            stream(session, pcm(300, -20.0));
            session.handle(R"({"type": "input_audio_buffer.commit"})");
        }
        const size_t after = events.all().size();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(events.all().size() == after);  // nothing sent once the session is gone
    }
}
