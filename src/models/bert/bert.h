#pragma once

// BERT (Hugging Face model_type "bert": transformers' BertModel), the encoder of sentence-transformers checkpoints such
// as all-MiniLM-L6-v2 and bge-small-en-v1.5, served as an embedding model.
//
// The word, absolute position and token type 0 rows of each token are summed under a LayerNorm; then post-norm layers
// (models/common/encoder.h): attention with biased query, key, value and output projections, an MLP of two biased
// projections around the exact GELU. Every norm is a LayerNorm with a bias.

#include <filesystem>
#include <memory>

#include <nlohmann/json.hpp>

#include "models/common/sentence_transformers.h"
#include "models/embedding.h"

namespace xs::bert {

struct Config {
    int64_t hidden_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t head_dim = 0;
    int64_t max_positions = 0;  // rows of the position table
    double layer_norm_eps = 1e-12;
};

Config parse_config(const nlohmann::json& config_json);

// `pipeline`: the checkpoint's sentence-transformers modules. The longest input is the Transformer module's
// max_seq_length, never more than the position table has rows.
std::unique_ptr<EmbeddingModel> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                                     const st::Pipeline& pipeline);

}  // namespace xs::bert
