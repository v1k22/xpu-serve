#pragma once

// A realtime transcription session: OpenAI's Realtime API with a session of type "transcription" (GET /v1/realtime, a
// WebSocket), transport aside. Client events go to handle(); server events, in OpenAI's GA shapes, to `send`:
//   session.update            -> session.updated: audio.input.format audio/pcm at 24 kHz (16-bit mono), .transcription
//                                {model, language, prompt}, .turn_detection server_vad {threshold, prefix_padding_ms,
//                                silence_duration_ms} or null (the client commits)
//   input_audio_buffer.append -> with server_vad, input_audio_buffer.speech_started and .speech_stopped, and at a stop
//                                the commit below
//   input_audio_buffer.commit -> input_audio_buffer.committed, conversation.item.added; then
//                                conversation.item.input_audio_transcription.delta ..., .completed (or .failed),
//                                conversation.item.done
//   input_audio_buffer.clear  -> input_audio_buffer.cleared
// session.created comes first. A refused event or field gets an `error` event naming it (semantic_vad, noise
// reduction, logprobs, other formats, any other client event) and the session goes on. Committed turns are
// transcribed in order on a thread of the session's own, so audio keeps arriving while one is.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "audio/vad.h"
#include "engine/service.h"
#include "server/channel.h"

namespace xs::server {

class RealtimeSession {
public:
    using Send = std::function<void(const nlohmann::ordered_json& event)>;

    // Sends session.created. `send` is never called from two threads at once.
    RealtimeSession(InferenceService& service, Send send);
    // The client is gone: stops the transcription in progress, drops the queued ones, waits for the worker.
    ~RealtimeSession();
    RealtimeSession(const RealtimeSession&) = delete;
    RealtimeSession& operator=(const RealtimeSession&) = delete;

    // One client event, a JSON text message.
    void handle(std::string_view message);

    static constexpr int kSampleRate = 24000;

private:
    struct Config {
        std::string model;
        std::string language;  // empty: the model finds it
        std::string prompt;
        bool server_vad = true;
        audio::VadSettings vad;
    };
    struct Job {
        std::string item_id;
        audio::Audio clip;
        Config config;
    };

    void emit(nlohmann::ordered_json event);
    void refuse(const std::string& message, const std::string& param, const std::string& code,
                const nlohmann::json& event_id);
    nlohmann::ordered_json session_json() const;
    Config updated(const nlohmann::json& session) const;  // throws InvalidRequest naming the field
    void append(std::span<const int16_t> pcm);
    void commit(int64_t begin, int64_t end, const std::string& item_id);
    std::string next_item_id();
    void transcribe_loop();
    void transcribe(const Job& job);

    InferenceService& service_;
    Send send_;
    std::mutex send_mutex_;
    int64_t event_count_ = 0;
    const std::string id_;
    Config config_;
    audio::EnergyVad vad_;
    int64_t vad_origin_ = 0;       // the session's sample where vad_'s stream starts
    std::vector<float> buffer_;    // samples from base_ on, of all the session's audio
    int64_t base_ = 0;
    std::string turn_item_;        // the item of the turn the VAD is in, while it is
    int64_t turn_start_ = 0;
    std::string last_item_;
    int64_t items_ = 0;
    std::atomic<bool> closing_ = false;
    Channel<Job> jobs_;
    std::jthread worker_;
};

}  // namespace xs::server
