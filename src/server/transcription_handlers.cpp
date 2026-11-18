#include "server/transcription_handlers.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <set>
#include <string>
#include <thread>

#include "server/channel.h"
#include "server/openai_format.h"
#include "server/openai_request.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::server {
namespace {

using ordered_json = nlohmann::ordered_json;

constexpr auto kAlivePollInterval = std::chrono::milliseconds(100);

bool parse_bool(const std::string& value, const char* field) {
    if (value == "true") return true;
    if (value == "false") return false;
    throw InvalidRequest(str_cat("'", field, "' must be true or false"), field);
}

TranscriptionResult transcribe_or_error(InferenceService& service, const TranscriptionRequest& request,
                                        const TextCallback& on_text) {
    try {
        return service.transcribe(request, on_text);
    } catch (const std::exception& e) {
        TranscriptionResult failed;
        failed.finish = FinishReason::Error;
        failed.error = e.what();
        return failed;
    }
}

void log_summary(const TranscriptionResult& r, bool client_gone) {
    const GenerateStats& s = r.stats;
    log::info("transcription: ", static_cast<int>(r.seconds * 10) / 10.0, " s of audio in ", r.pieces, " piece(s), ",
              s.prompt_tokens, " prompt tokens (encode ", static_cast<int>(s.encode_ms), " ms, prefill ",
              static_cast<int>(s.prefill_ms), " ms), ", s.generated_tokens, " tokens (decode ",
              static_cast<int>(s.decode_ms), " ms), language ", r.language.empty() ? "?" : r.language, ", finish ",
              client_gone ? "cancelled (client gone)" : r.finish == FinishReason::Error ? "error: " + r.error
                                                                                        : finish_reason_name(r.finish));
}

}  // namespace

TranscriptionForm parse_transcription_form(const httplib::MultipartFormData& form, const SamplingParams& defaults) {
    static const std::set<std::string> known = {"file", "model", "language", "prompt", "temperature",
                                                "response_format", "stream"};
    for (const auto& [name, field] : form.fields) {
        if (name == "timestamp_granularities[]" || name == "timestamp_granularities") {
            throw InvalidRequest("timestamps need a forced aligner, which this server does not run yet", name);
        }
        if (known.count(name) == 0) {
            throw InvalidRequest(str_cat("'", name, "' is not supported (file, model, language, prompt, temperature, ",
                                         "response_format and stream are)"),
                                 name);
        }
    }
    if (!form.has_file("file")) {
        throw InvalidRequest("'file' is required: the audio, sent as a file part (curl -F file=@clip.wav)", "file");
    }
    TranscriptionForm f;
    const std::string audio = form.get_file("file").content;
    f.request.audio.assign(audio.begin(), audio.end());
    if (f.request.audio.empty()) throw InvalidRequest("'file' is empty", "file");
    f.request.language = form.get_field("language");
    f.request.prompt = form.get_field("prompt");
    f.request.sampling = defaults;
    if (form.has_field("temperature")) {
        const std::string t = form.get_field("temperature");
        char* end = nullptr;
        const double value = std::strtod(t.c_str(), &end);
        if (t.empty() || *end != '\0' || !(value >= 0.0 && value <= 1.0)) {
            throw InvalidRequest("'temperature' must be a number from 0 to 1", "temperature");
        }
        f.request.sampling.temperature = static_cast<float>(value);
    }
    const std::string format = form.has_field("response_format") ? form.get_field("response_format") : "json";
    if (format == "verbose_json" || format == "srt" || format == "vtt" || format == "diarized_json") {
        throw InvalidRequest(str_cat("response_format '", format, "' needs timestamps (a forced aligner), which this ",
                                     "server does not run yet; json and text are supported"),
                             "response_format");
    }
    if (format != "json" && format != "text") {
        throw InvalidRequest("'response_format' must be json or text", "response_format");
    }
    f.text = format == "text";
    if (form.has_field("stream")) f.stream = parse_bool(form.get_field("stream"), "stream");
    return f;
}

ordered_json transcription_response(const TranscriptionResult& r) {
    const GenerateStats& s = r.stats;
    const ordered_json usage = {{"type", "tokens"},
                                {"input_tokens", s.prompt_tokens},
                                {"output_tokens", s.generated_tokens},
                                {"total_tokens", s.prompt_tokens + s.generated_tokens}};
    return {{"text", r.text}, {"usage", usage}};
}

void TranscriptionHandlers::transcribe(const httplib::Request& req, httplib::Response& res) const {
    if (!service_.transcribes()) {
        throw InvalidRequest(str_cat("model '", service_.model().id, "' does not transcribe audio"), "model");
    }
    if (!req.is_multipart_form_data()) {
        throw InvalidRequest("send the audio as multipart/form-data (curl -F file=@clip.wav -F model=...)", "file");
    }
    const TranscriptionForm form = parse_transcription_form(req.form, service_.model().default_sampling);
    if (!form.stream) {
        const TranscriptionResult r = transcribe_or_error(service_, form.request, {});
        log_summary(r, false);
        if (r.finish == FinishReason::Error) {
            // A clip the decoder or the model refused is the client's to fix.
            throw InvalidRequest(r.error, "file");
        }
        if (form.text) {
            res.set_content(r.text, "text/plain; charset=utf-8");
        } else {
            res.set_content(transcription_response(r).dump(), "application/json");
        }
        return;
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream", [this, form, &req](size_t, httplib::DataSink& sink) {
        const auto send = [&sink](const ordered_json& event) {
            const std::string text = sse_event(nlohmann::json(event));
            return sink.write(text.data(), text.size());
        };
        std::atomic<bool> client_gone = false;
        Channel<std::string> channel;
        TranscriptionResult result;
        std::jthread worker([&] {
            result = transcribe_or_error(service_, form.request, [&](std::string_view delta) {
                if (client_gone.load()) return false;
                channel.push(std::string(delta));
                return true;
            });
            channel.close();
        });
        std::deque<std::string> batch;
        while (channel.wait_take(batch, kAlivePollInterval)) {
            for (const std::string& delta : batch) {
                if (!client_gone && !send({{"type", "transcript.text.delta"}, {"delta", delta}})) client_gone = true;
            }
            if (!client_gone && req.is_connection_closed()) client_gone = true;
        }
        worker.join();
        log_summary(result, client_gone.load());
        if (client_gone) return false;
        bool ok = true;
        if (result.finish == FinishReason::Error) {
            ok = send(ordered_json(error_body(result.error, "invalid_request_error", "file")));
        } else {
            const ordered_json body = transcription_response(result);
            ordered_json done = {{"type", "transcript.text.done"}};
            for (const auto& [key, value] : body.items()) done[key] = value;
            ok = send(done);
        }
        if (ok) sink.done();
        return ok;
    });
}

void TranscriptionHandlers::translate(const httplib::Request&, httplib::Response&) const {
    throw InvalidRequest(str_cat("model '", service_.model().id, "' transcribes audio in the language spoken; it does ",
                                 "not translate (POST /v1/audio/transcriptions)"),
                         "model");
}

}  // namespace xs::server
