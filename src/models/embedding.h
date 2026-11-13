#pragma once

// An embedding model: a token sequence in, one vector out (POST /v1/embeddings). The encoder reads the whole sequence
// in one stateless run, every token seeing every other; the sentence embedding is its final hidden states pooled and,
// for some checkpoints, L2-normalized, as the checkpoint's sentence-transformers modules (modules.json) say. The model
// code builds the encoder; tokenizing, the device and the pooling belong to the runtime (src/embedding/).

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "graph/graph.h"
#include "io/safetensors.h"
#include "tensor/tensor.h"

namespace xs {

enum class Pooling : uint8_t { Mean, Cls };  // the average of every row; the first row (the [CLS] token's)
const char* pooling_name(Pooling pooling);    // "mean", "cls"

struct EmbeddingSpec {
    std::string id;          // display name, the checkpoint directory name by default
    std::string arch;        // e.g. "bert"
    int64_t dimension = 0;   // values per embedding: the encoder's hidden size
    int64_t max_tokens = 0;  // the longest sequence the model reads, special tokens included
    int64_t vocab_size = 0;  // token ids are in [0, vocab_size)
    Pooling pooling = Pooling::Mean;
    bool normalize = false;  // the pooled vector is divided by its L2 norm (a Normalize module follows the Pooling)
};

class EmbeddingModel {
public:
    virtual ~EmbeddingModel() = default;
    virtual const EmbeddingSpec& spec() const = 0;
    virtual const WeightStore& weights() const = 0;
    // The encoder: one program without state over the T tokens of a sequence, its output "hidden" [T, dimension] (the
    // final hidden states, sentence-transformers' token embeddings).
    virtual const graph::Graph& graph() const = 0;
    // The program's inputs for `ids`: 1 to spec().max_tokens ids, each in [0, spec().vocab_size).
    virtual TensorMap prepare_inputs(std::span<const int32_t> ids) const = 0;
};

// Loads the embedding model in `dir`: a checkpoint that model_kind() calls ModelKind::Embedding. Throws xs::Error
// naming what is not supported.
std::unique_ptr<EmbeddingModel> load_embedding_model(const std::filesystem::path& dir);
std::vector<std::string> supported_embedding_models();  // the model_type values served as embedding models

}  // namespace xs
