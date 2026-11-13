#pragma once

// The embedding runtime: one embedding model on one device. The encoder is compiled once into a session without state;
// a sequence is one run of it, and its embedding is pooled from the final hidden states on the host
// (embedding/pooling.h). The runner takes token ids: text is tokenized by its callers (EmbeddingService, the CLI).

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "models/embedding.h"

namespace xs {

struct EmbeddingRunnerOptions {
    std::string backend = "sycl";
    SessionOptions session;  // max_context lowers the longest sequence below the model's
    std::string model_id;    // empty = the model directory name
};

struct EmbeddingTiming {
    double device_ms = 0;  // the runs, with input upload and output download
    double total_ms = 0;   // those and the pooling
};

struct EmbeddingResult {
    std::vector<std::vector<float>> embeddings;  // one per sequence, in order
    int64_t tokens = 0;                          // of every sequence
    EmbeddingTiming timing;
};

// Why the model cannot read `ids` ("no token", "600 tokens, more than ...", "token id ..., outside the vocabulary
// ...": what an input "has"), or nullopt. Inputs are refused, never cut.
std::optional<std::string> sequence_problem(const EmbeddingSpec& spec, std::span<const int32_t> ids);

class EmbeddingRunner {
public:
    static std::unique_ptr<EmbeddingRunner> create(const std::filesystem::path& dir,
                                                   const EmbeddingRunnerOptions& options);
    ~EmbeddingRunner();
    EmbeddingRunner(const EmbeddingRunner&) = delete;
    EmbeddingRunner& operator=(const EmbeddingRunner&) = delete;

    // The model's spec, its max_tokens lowered to SessionOptions::max_context when that is smaller.
    const EmbeddingSpec& spec() const { return spec_; }
    // The encoder's final hidden states [T, dimension] of one sequence. Throws xs::Error for a sequence
    // sequence_problem refuses.
    HostTensor hidden_states(std::span<const int32_t> ids);
    // One embedding per sequence, each sequence one run of the encoder.
    EmbeddingResult embed(const std::vector<std::vector<int32_t>>& sequences);

private:
    EmbeddingRunner() = default;

    EmbeddingSpec spec_;
    std::unique_ptr<EmbeddingModel> model_;
    std::unique_ptr<Session> session_;
};

}  // namespace xs
