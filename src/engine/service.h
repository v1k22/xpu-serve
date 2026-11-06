#pragma once

// What the HTTP server needs from the inference side. The server is written against this interface only, so it can
// be tested with a fake.

#include <memory>
#include <string>

#include "engine/types.h"

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
};

}  // namespace xs
