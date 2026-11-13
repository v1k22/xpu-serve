#pragma once

// NomicBERT (Hugging Face model_type "nomic_bert"; reference: modeling_hf_nomic_bert.py of nomic-ai/nomic-bert-2048,
// NomicBertModel), the encoder of nomic-embed-text-v1 / v1.5, served as an embedding model.
//
// A BERT without a position table: the word and token type 0 rows under a LayerNorm (emb_ln), then post-norm layers
// (models/common/encoder.h) whose attention reads one fused projection (Wqkv: the query, key and value rows one after
// another) and rotates queries and keys over the whole head (NeoX RoPE, positions from 0), and whose MLP is SwiGLU
// (fc2(fc11(x) * silu(fc12(x)))). No projection has a bias; every norm is a LayerNorm with one. The rotation goes on
// unchanged past the positions the model was trained on (max_trained_positions): the reference scales it only with a
// rotary_scaling_factor, which is refused.

#include <filesystem>
#include <memory>

#include <nlohmann/json.hpp>

#include "models/common/sentence_transformers.h"
#include "models/embedding.h"

namespace xs::nomic_bert {

struct Config {
    int64_t hidden_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t head_dim = 0;
    int64_t max_positions = 0;  // max_position_embeddings: the longest input when sentence-transformers names none
    double layer_norm_eps = 1e-5;
    double rope_theta = 10000.0;
};

Config parse_config(const nlohmann::json& config_json);

// `pipeline`: the checkpoint's sentence-transformers modules; the longest input is the Transformer module's
// max_seq_length.
std::unique_ptr<EmbeddingModel> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                                     const st::Pipeline& pipeline);

}  // namespace xs::nomic_bert
