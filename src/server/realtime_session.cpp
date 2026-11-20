#include "server/realtime_session.h"

#include <algorithm>
#include <deque>
#include <random>
#include <span>

#include "server/openai_request.h"
#include "util/base64.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::server {
namespace {

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

constexpr int64_t kMinCommitMs = 100;  // OpenAI's least audio for a commit
constexpr int64_t kKeptMs = 200;       // kept past the prefix before speech: the VAD dates a turn this far back at most

int64_t ms(int64_t samples) { return samples * 1000 / RealtimeSession::kSampleRate; }

std::string random_id(std::string_view prefix) {
    static constexpr char kDigits[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::random_device device;
    std::mt19937 rng(device());
    std::string id(prefix);
    for (int i = 0; i < 16; ++i) id += kDigits[rng() % 36];
    return id;
}

// An object's keys must be among `known`.
void known_keys(const json& j, std::initializer_list<std::string_view> known, const std::string& where) {
    for (const auto& [key, value] : j.items()) {
        if (std::find(known.begin(), known.end(), key) == known.end()) {
            throw InvalidRequest(str_cat("'", where, ".", key, "' is not supported"), where + "." + key);
        }
    }
}

const json& object_at(const json& j, const char* key, const std::string& where) {
    const json& v = j.at(key);
    if (!v.is_object()) throw InvalidRequest(str_cat("'", where, "' must be an object"), where);
    return v;
}

std::string string_at(const json& j, const char* key, const std::string& where) {
    const json& v = j.at(key);
    if (!v.is_string()) throw InvalidRequest(str_cat("'", where, "' must be a string"), where);
    return v.get<std::string>();
}

int ms_at(const json& j, const char* key, const std::string& where) {
    const json& v = j.at(key);
    if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() > 60000) {
        throw InvalidRequest(str_cat("'", where, "' must be milliseconds from 0 to 60000"), where);
    }
    return v.get<int>();
}

ordered_json user_item(const std::string& id, const json& transcript) {
    return {{"id", id},
            {"object", "realtime.item"},
            {"type", "message"},
            {"status", "completed"},
            {"role", "user"},
            {"content", ordered_json::array({{{"type", "input_audio"}, {"transcript", transcript}}})}};
}

}  // namespace

RealtimeSession::RealtimeSession(InferenceService& service, Send send)
    : service_(service),
      send_(std::move(send)),
      id_(random_id("sess_")),
      vad_(kSampleRate, {}),
      worker_([this] { transcribe_loop(); }) {
    config_.model = service_.model().id;
    emit({{"type", "session.created"}, {"session", session_json()}});
}

RealtimeSession::~RealtimeSession() {
    closing_ = true;
    jobs_.close();
    if (worker_.joinable()) worker_.join();
}

void RealtimeSession::emit(ordered_json event) {
    std::lock_guard lock(send_mutex_);
    ordered_json out = {{"event_id", str_cat("event_", ++event_count_)}};
    for (auto& [key, value] : event.items()) out[key] = std::move(value);
    send_(out);
}

void RealtimeSession::refuse(const std::string& message, const std::string& param, const std::string& code,
                             const json& event_id) {
    ordered_json error = {{"type", "invalid_request_error"}, {"code", code}, {"message", message}};
    error["param"] = param.empty() ? json(nullptr) : json(param);
    error["event_id"] = event_id.is_string() ? event_id : json(nullptr);
    emit({{"type", "error"}, {"error", error}});
}

ordered_json RealtimeSession::session_json() const {
    ordered_json turn = nullptr;
    if (config_.server_vad) {
        turn = {{"type", "server_vad"},
                {"threshold", config_.vad.threshold},
                {"prefix_padding_ms", config_.vad.prefix_ms},
                {"silence_duration_ms", config_.vad.silence_ms}};
    }
    const ordered_json transcription = {
        {"model", config_.model},
        {"language", config_.language.empty() ? json(nullptr) : json(config_.language)},
        {"prompt", config_.prompt}};
    const ordered_json input = {{"format", {{"type", "audio/pcm"}, {"rate", kSampleRate}}},
                                {"transcription", transcription},
                                {"turn_detection", turn},
                                {"noise_reduction", nullptr}};
    return {{"type", "transcription"},
            {"id", id_},
            {"object", "realtime.transcription_session"},
            {"audio", {{"input", input}}},
            {"include", nullptr}};
}

RealtimeSession::Config RealtimeSession::updated(const json& session) const {
    Config c = config_;
    known_keys(session, {"type", "audio", "include", "id", "object"}, "session");
    if (session.contains("type") && session.at("type") != "transcription") {
        throw InvalidRequest("this server's realtime sessions transcribe audio: 'session.type' must be transcription",
                             "session.type");
    }
    if (session.contains("include") && !session.at("include").is_null() && !session.at("include").empty()) {
        throw InvalidRequest("'session.include' (logprobs) is not supported", "session.include");
    }
    if (!session.contains("audio")) return c;
    const json& audio = object_at(session, "audio", "session.audio");
    known_keys(audio, {"input", "output"}, "session.audio");
    if (audio.contains("output") && !audio.at("output").is_null()) {
        throw InvalidRequest("a transcription session has no audio output", "session.audio.output");
    }
    if (!audio.contains("input")) return c;
    const std::string in = "session.audio.input";
    const json& input = object_at(audio, "input", in);
    known_keys(input, {"format", "transcription", "turn_detection", "noise_reduction"}, in);
    if (input.contains("format")) {
        const json& f = object_at(input, "format", in + ".format");
        known_keys(f, {"type", "rate"}, in + ".format");
        const std::string type = f.contains("type") ? string_at(f, "type", in + ".format.type") : "audio/pcm";
        if (type != "audio/pcm") {
            throw InvalidRequest(str_cat("audio format '", type, "' is not supported yet: send audio/pcm (16-bit ",
                                         "mono at ", kSampleRate, " Hz)"),
                                 in + ".format.type");
        }
        if (f.contains("rate") && f.at("rate") != kSampleRate) {
            throw InvalidRequest(str_cat("'", in, ".format.rate' must be ", kSampleRate), in + ".format.rate");
        }
    }
    if (input.contains("noise_reduction") && !input.at("noise_reduction").is_null()) {
        throw InvalidRequest("noise reduction is not supported", in + ".noise_reduction");
    }
    if (input.contains("transcription")) {
        const std::string t = in + ".transcription";
        if (input.at("transcription").is_null()) {
            throw InvalidRequest("a transcription session always transcribes: 'transcription' cannot be null", t);
        }
        const json& tr = object_at(input, "transcription", t);
        known_keys(tr, {"model", "language", "prompt", "keywords", "languages", "delay"}, t);
        for (const char* key : {"keywords", "languages", "delay"}) {
            if (tr.contains(key) && !tr.at(key).is_null()) {
                throw InvalidRequest(str_cat("'", t, ".", key, "' is not supported"), t + "." + key);
            }
        }
        if (tr.contains("model")) c.model = string_at(tr, "model", t + ".model");
        const auto optional_string = [&](const char* key) {
            return tr.at(key).is_null() ? std::string() : string_at(tr, key, t + "." + key);
        };
        if (tr.contains("language")) c.language = optional_string("language");
        if (tr.contains("prompt")) c.prompt = optional_string("prompt");
    }
    if (input.contains("turn_detection")) {
        const std::string t = in + ".turn_detection";
        const json& td = input.at("turn_detection");
        c.server_vad = !td.is_null();
        if (c.server_vad) {
            if (!td.is_object()) throw InvalidRequest(str_cat("'", t, "' must be an object or null"), t);
            known_keys(td, {"type", "threshold", "prefix_padding_ms", "silence_duration_ms", "create_response",
                            "interrupt_response", "idle_timeout_ms"},
                       t);
            const std::string type = td.contains("type") ? string_at(td, "type", t + ".type") : "server_vad";
            if (type != "server_vad") {
                throw InvalidRequest(str_cat("turn detection '", type, "' is not supported: server_vad or null"),
                                     t + ".type");
            }
            if (td.contains("idle_timeout_ms") && !td.at("idle_timeout_ms").is_null()) {
                throw InvalidRequest("'idle_timeout_ms' is not supported", t + ".idle_timeout_ms");
            }
            c.vad = {};
            if (td.contains("threshold")) {
                const json& v = td.at("threshold");
                if (!v.is_number() || v.get<double>() < 0.0 || v.get<double>() > 1.0) {
                    throw InvalidRequest(str_cat("'", t, ".threshold' must be from 0 to 1"), t + ".threshold");
                }
                c.vad.threshold = v.get<double>();
            }
            if (td.contains("prefix_padding_ms")) {
                c.vad.prefix_ms = ms_at(td, "prefix_padding_ms", t + ".prefix_padding_ms");
            }
            if (td.contains("silence_duration_ms")) {
                c.vad.silence_ms = ms_at(td, "silence_duration_ms", t + ".silence_duration_ms");
            }
        }
    }
    return c;
}

void RealtimeSession::handle(std::string_view message) {
    json event;
    try {
        event = json::parse(message);
    } catch (const std::exception&) {
        refuse("the message is not JSON", "", "invalid_json", nullptr);
        return;
    }
    const json event_id = event.is_object() && event.contains("event_id") ? event.at("event_id") : json(nullptr);
    if (!event.is_object() || !event.contains("type") || !event.at("type").is_string()) {
        refuse("an event is a JSON object with a string 'type'", "type", "invalid_value", event_id);
        return;
    }
    const std::string type = event.at("type").get<std::string>();
    try {
        if (type == "input_audio_buffer.append") {
            if (!event.contains("audio") || !event.at("audio").is_string()) {
                throw InvalidRequest("'audio' must be a base64 string of 16-bit PCM", "audio");
            }
            const auto bytes = base64_decode(event.at("audio").get<std::string>());
            if (!bytes || bytes->size() % 2 != 0) {
                throw InvalidRequest("'audio' must be base64 of 16-bit little-endian PCM samples", "audio");
            }
            std::vector<int16_t> pcm(bytes->size() / 2);
            for (size_t i = 0; i < pcm.size(); ++i) {
                pcm[i] = static_cast<int16_t>((*bytes)[2 * i] | ((*bytes)[2 * i + 1] << 8));
            }
            append(pcm);
        } else if (type == "input_audio_buffer.commit") {
            const int64_t end = base_ + static_cast<int64_t>(buffer_.size());
            if (ms(end - base_) < kMinCommitMs) {
                refuse(str_cat("buffer too small: expected at least ", kMinCommitMs, " ms of audio, but the buffer ",
                               "has ", ms(end - base_), " ms"),
                       "", "input_audio_buffer_commit_empty", event_id);
                return;
            }
            const std::string item = turn_item_.empty() ? next_item_id() : turn_item_;
            vad_.end_turn();
            turn_item_.clear();
            commit(base_, end, item);
        } else if (type == "input_audio_buffer.clear") {
            base_ += static_cast<int64_t>(buffer_.size());
            buffer_.clear();
            vad_.end_turn();
            turn_item_.clear();
            emit({{"type", "input_audio_buffer.cleared"}});
        } else if (type == "session.update") {
            if (!event.contains("session") || !event.at("session").is_object()) {
                throw InvalidRequest("'session' must be an object", "session");
            }
            const Config c = updated(event.at("session"));
            if (c.server_vad != config_.server_vad || c.vad.threshold != config_.vad.threshold ||
                c.vad.prefix_ms != config_.vad.prefix_ms || c.vad.silence_ms != config_.vad.silence_ms) {
                vad_ = audio::EnergyVad(kSampleRate, c.vad);  // a new detector, from the audio that comes next
                vad_origin_ = base_ + static_cast<int64_t>(buffer_.size());
                turn_item_.clear();
            }
            config_ = c;
            emit({{"type", "session.updated"}, {"session", session_json()}});
        } else {
            refuse(str_cat("event '", type, "' is not supported in a transcription session (session.update and ",
                           "input_audio_buffer.append, commit and clear are)"),
                   "type", "invalid_value", event_id);
        }
    } catch (const InvalidRequest& e) {
        refuse(e.what(), e.param(), "invalid_value", event_id);
    }
}

void RealtimeSession::append(std::span<const int16_t> pcm) {
    const size_t at = buffer_.size();
    buffer_.resize(at + pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) buffer_[at + i] = static_cast<float>(pcm[i]) / 32768.0f;
    if (!config_.server_vad) return;
    for (const audio::VadEvent& e : vad_.push(std::span(buffer_).subspan(at))) {
        const int64_t sample = vad_origin_ + e.sample;
        if (e.kind == audio::VadEvent::Kind::SpeechStarted) {
            turn_item_ = next_item_id();
            turn_start_ = std::max(sample, base_);
            emit({{"type", "input_audio_buffer.speech_started"}, {"audio_start_ms", ms(turn_start_)},
                  {"item_id", turn_item_}});
        } else if (!turn_item_.empty()) {
            emit({{"type", "input_audio_buffer.speech_stopped"}, {"audio_end_ms", ms(sample)},
                  {"item_id", turn_item_}});
            const std::string item = std::exchange(turn_item_, {});
            commit(turn_start_, sample, item);
        }
    }
    // Out of a turn, audio older than a turn could start at is never committed: let it go.
    if (turn_item_.empty()) {
        const int64_t end = base_ + static_cast<int64_t>(buffer_.size());
        const int64_t keep = (config_.vad.prefix_ms + kKeptMs) * kSampleRate / 1000;
        if (end - base_ > 2 * keep) {
            buffer_.erase(buffer_.begin(), buffer_.end() - keep);
            base_ = end - keep;
        }
    }
}

// Audio [begin, end) of the session becomes item `item_id`; the buffer keeps what follows it.
void RealtimeSession::commit(int64_t begin, int64_t end, const std::string& item_id) {
    Job job;
    job.item_id = item_id;
    job.clip.sample_rate = kSampleRate;
    job.clip.samples.assign(buffer_.begin() + (begin - base_), buffer_.begin() + (end - base_));
    job.config = config_;
    buffer_.erase(buffer_.begin(), buffer_.begin() + (end - base_));
    base_ = end;
    const json previous = last_item_.empty() ? json(nullptr) : json(last_item_);
    emit({{"type", "input_audio_buffer.committed"}, {"previous_item_id", previous}, {"item_id", item_id}});
    emit({{"type", "conversation.item.added"},
          {"previous_item_id", previous},
          {"item", user_item(item_id, nullptr)}});
    last_item_ = item_id;
    jobs_.push(std::move(job));
}

std::string RealtimeSession::next_item_id() { return str_cat("item_", id_.substr(5, 8), "_", ++items_); }

void RealtimeSession::transcribe_loop() {
    std::deque<Job> batch;
    while (jobs_.wait_take(batch, std::chrono::milliseconds(200))) {
        for (const Job& job : batch) {
            if (closing_) return;
            transcribe(job);
        }
    }
}

void RealtimeSession::transcribe(const Job& job) {
    TranscriptionRequest request;
    request.clip = job.clip;
    request.language = job.config.language;
    request.prompt = job.config.prompt;
    request.sampling = service_.model().default_sampling;
    TranscriptionResult r;
    try {
        r = service_.transcribe(request, [&](std::string_view delta) {
            if (closing_) return false;
            emit({{"type", "conversation.item.input_audio_transcription.delta"}, {"item_id", job.item_id},
                  {"content_index", 0}, {"delta", std::string(delta)}});
            return true;
        });
    } catch (const std::exception& e) {
        r.finish = FinishReason::Error;
        r.error = e.what();
    }
    if (closing_ || r.finish == FinishReason::Cancelled) return;
    const GenerateStats& s = r.stats;
    log::info("realtime: ", job.item_id, ", ", ms(static_cast<int64_t>(job.clip.samples.size())), " ms of audio, ",
              s.prompt_tokens, " prompt tokens (encode ", static_cast<int>(s.encode_ms), " ms, prefill ",
              static_cast<int>(s.prefill_ms), " ms), ", s.generated_tokens, " tokens (decode ",
              static_cast<int>(s.decode_ms), " ms)", r.finish == FinishReason::Error ? ", error: " + r.error : "");
    if (r.finish == FinishReason::Error) {
        const ordered_json error = {
            {"type", "invalid_request_error"}, {"code", nullptr}, {"message", r.error}, {"param", nullptr}};
        emit({{"type", "conversation.item.input_audio_transcription.failed"},
              {"item_id", job.item_id},
              {"content_index", 0},
              {"error", error}});
        return;
    }
    const ordered_json usage = {{"type", "tokens"},
                                {"input_tokens", s.prompt_tokens},
                                {"output_tokens", s.generated_tokens},
                                {"total_tokens", s.prompt_tokens + s.generated_tokens}};
    emit({{"type", "conversation.item.input_audio_transcription.completed"}, {"item_id", job.item_id},
          {"content_index", 0}, {"transcript", r.text}, {"usage", usage}});
    emit({{"type", "conversation.item.done"}, {"item", user_item(job.item_id, r.text)}});
}

}  // namespace xs::server
