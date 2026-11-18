#pragma once

// What the HTTP server needs from the inference side. The server is written against this interface only, so it can
// be tested with a fake.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/types.h"
#include "util/error.h"

namespace xs {
namespace tok {
class Tokenizer;
}
namespace chat {
class ChatTemplate;
class OutputParser;
}  // namespace chat

struct ModelCard {
    std::string id;           // what /v1/models lists and requests may name
    std::string arch;         // e.g. "gemma4"
    int64_t context_length = 0;
    std::string dialect;      // output dialect for chat::make_output_parser, e.g. "gemma4"
    SamplingParams default_sampling;  // from generation_config.json
    std::vector<int32_t> stop_token_ids;
};

class InferenceService {
public:
    virtual ~InferenceService() = default;
    virtual const ModelCard& model() const = 0;
    virtual const tok::Tokenizer& tokenizer() const = 0;
    virtual const chat::ChatTemplate& chat_template() const = 0;
    // Runs one generation and blocks until it ends. Requests are served one at a time in arrival order;
    // `on_token` runs on the engine thread. Thread-safe.
    virtual GenerateResult generate(const GenerateRequest& request, const TokenCallback& on_token) = 0;

    // Media. Only a model with a loaded encoder of a modality takes input of it.
    virtual bool accepts(Modality /*m*/) const { return false; }
    // Decodes one encoded item (an image: PNG, JPEG, BMP, GIF; an audio clip: WAV, MP3, FLAC) and preprocesses it
    // for the model's encoder of `m`. Throws xs::Error with a message meant for the client. Thread-safe.
    virtual MediaInput prepare_media(Modality m, std::span<const uint8_t> /*encoded*/) const {
        fail("model '", model().id, "' does not take ", modality_name(m), " input");
    }
    // Speech recognition: only a model that transcribes takes it. Blocks until the transcript is complete; `on_text`
    // (may be empty) receives it as it decodes, on the engine thread. Thread-safe.
    virtual bool transcribes() const { return false; }
    virtual TranscriptionResult transcribe(const TranscriptionRequest& /*request*/, const TextCallback& /*on_text*/) {
        fail("model '", model().id, "' does not transcribe audio");
    }
    // `prompt` with each media placeholder the chat template rendered replaced by its item's token run. Throws
    // xs::Error when placeholders and items do not pair up.
    virtual std::vector<int32_t> expand_media(std::vector<int32_t> prompt, const std::vector<MediaInput>& media) const {
        XS_CHECK(media.empty(), "model '", model().id, "' does not take media input");
        return prompt;
    }
};

}  // namespace xs
