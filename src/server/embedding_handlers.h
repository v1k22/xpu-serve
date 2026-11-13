#pragma once

// POST /v1/embeddings for an embedding model: OpenAI's embeddings API, as vLLM and llama.cpp serve it.
//   {"input": "text" | ["text", ...] | [id, ...] | [[id, ...], ...], "encoding_format"?: "float" | "base64",
//    "dimensions"?, "model"?, "user"?}
//   -> {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": [value, ...] | "<base64>"}],
//       "model", "usage": {"prompt_tokens", "total_tokens"}}
// A text is read as the model's tokenizer encodes it, inside the special tokens it puts around a sequence; token ids
// are read as given. An input the model cannot read whole (empty, longer than its limit, an id outside the vocabulary)
// is refused, never cut. base64 is the float32 values, little-endian. `dimensions` is accepted only as the model's
// own: none of these models makes shorter embeddings. `model` is not checked, as for chat, and `user` is ignored.
// Errors are 400s with OpenAI's error body naming the field ("input", "input[2]", "encoding_format", "dimensions").
// Timings travel in a Server-Timing header.

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "embedding/service.h"
#include "server/openai_request.h"

namespace xs::server {

struct EmbeddingsRequest {
    std::vector<std::vector<int32_t>> sequences;  // what the model reads, one per input, in order
    bool base64 = false;
};

// Throws InvalidRequest naming the field. `tokenize` gives the ids of a text input.
EmbeddingsRequest parse_embeddings_request(const RequestJson& body, const EmbeddingSpec& spec,
                                           const std::function<std::vector<int32_t>(std::string_view)>& tokenize);
nlohmann::ordered_json embeddings_response(const EmbeddingResult& result, bool base64, const std::string& model);

class EmbeddingHandlers {
public:
    explicit EmbeddingHandlers(EmbeddingService& service) : service_(service) {}
    void embed(const httplib::Request& req, httplib::Response& res) const;
    void models(const httplib::Request& req, httplib::Response& res, int64_t created) const;

private:
    EmbeddingService& service_;
};

}  // namespace xs::server
