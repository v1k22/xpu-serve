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

    // Images. Only a model with a loaded vision encoder takes them.
    virtual bool accepts_images() const { return false; }
    // Decodes one encoded image (PNG, JPEG, BMP, GIF) and preprocesses it for the vision encoder. Throws xs::Error
    // with a message meant for the client. Thread-safe.
    virtual MediaInput prepare_image(std::span<const uint8_t> /*encoded*/) const {
        fail("model '", model().id, "' does not take images");
    }
    // `prompt` with each image placeholder the chat template rendered replaced by its image's token run. Throws
    // xs::Error when placeholders and images do not pair up.
    virtual std::vector<int32_t> expand_media(std::vector<int32_t> prompt, const std::vector<MediaInput>& media) const {
        XS_CHECK(media.empty(), "model '", model().id, "' does not take images");
        return prompt;
    }
};

}  // namespace xs
