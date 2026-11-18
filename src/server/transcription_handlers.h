#pragma once

// POST /v1/audio/transcriptions for a speech recognition model: OpenAI's transcription API.
//   multipart/form-data: file (WAV, MP3, FLAC), model?, language? (ISO 639-1 or a name), prompt?, temperature? (0 to
//   1), response_format? ("json" | "text"), stream? ("true" | "false")
//   -> json: {"text", "usage": {"type": "tokens", "input_tokens", "output_tokens", "total_tokens"}}; text: the
//      transcript as text/plain;
//      stream: server-sent events {"type": "transcript.text.delta", "delta"} ..., then {"type":
//      "transcript.text.done", "text", "usage"}.
// Timestamps (verbose_json, srt, vtt, timestamp_granularities[]) need a forced aligner and are refused by name, as are
// chunking_strategy, include[] and known speakers; `model` is not checked, as for chat. POST /v1/audio/translations is
// refused: the model transcribes only. Errors are 400s with OpenAI's error body naming the field.

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "engine/service.h"

namespace xs::server {

struct TranscriptionForm {
    TranscriptionRequest request;
    bool text = false;    // response_format "text"
    bool stream = false;
};

// Throws InvalidRequest naming the field.
TranscriptionForm parse_transcription_form(const httplib::MultipartFormData& form, const SamplingParams& defaults);
// The "json" body; the done event carries the same fields.
nlohmann::ordered_json transcription_response(const TranscriptionResult& result);

class TranscriptionHandlers {
public:
    explicit TranscriptionHandlers(InferenceService& service) : service_(service) {}
    void transcribe(const httplib::Request& req, httplib::Response& res) const;
    void translate(const httplib::Request& req, httplib::Response& res) const;

private:
    InferenceService& service_;
};

}  // namespace xs::server
