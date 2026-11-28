#pragma once

// Llama 3 (Hugging Face model_type "llama"; transformers models/llama/): GQA attention with RoPE over the whole head
// (plain, or Llama 3's scaling of the long wavelengths), a SwiGLU MLP, RMSNorm (x * w), the head tied to the embedding
// table or not. No biases. Llama 3, 3.1, 3.2 and 3.3 checkpoints; its turn is written in the "llama3" dialect (tool
// calls as JSON objects).

#include <filesystem>
#include <memory>
#include <optional>

#include <nlohmann/json.hpp>

#include "models/common/blocks.h"
#include "models/model.h"

namespace xs::llama {

struct Config {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    double rms_norm_eps = 1e-5;
    double rope_theta = 0.0;
    std::optional<blocks::Llama3RopeScaling> llama3_rope;  // rope_type "llama3"; none: plain RoPE
    int64_t max_position = 0;
    bool tie_word_embeddings = false;
};

// config.json of a llama checkpoint. Biases, other activations, other RoPE types (linear, dynamic, yarn, ...) and a
// partial rotary factor are refused by name.
Config parse_config(const nlohmann::json& config);

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options);

}  // namespace xs::llama
