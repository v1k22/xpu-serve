#pragma once

// Gemma 4 text decoder (Hugging Face model_type "gemma4" / "gemma4_text"). Reference: transformers
// models/gemma4/modeling_gemma4.py (Gemma4TextModel, Gemma4TextDecoderLayer, Gemma4TextAttention).
//
// v0.1 serves the dense E-series layout (E2B, E4B): sliding + full attention, per-layer embeddings (PLE) and a
// KV-shared tail. Mixture-of-experts layers and K=V attention (26B-A4B, 31B) are refused by name until they land.

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/model.h"

namespace xs::gemma4 {

enum class AttnKind { Sliding, Full };

struct RopeConfig {
    double theta = 10000.0;
    double partial_rotary_factor = 1.0;  // "proportional" RoPE rotates this share of each head
    bool proportional = false;
};

struct Config {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;         // sliding layers
    int64_t global_head_dim = 0;  // full-attention layers
    int64_t sliding_window = 0;
    int64_t num_kv_shared_layers = 0;
    int64_t ple_dim = 0;          // hidden_size_per_layer_input; 0 = no per-layer embeddings
    int64_t ple_vocab_size = 0;
    bool double_wide_mlp = false;
    double rms_norm_eps = 1e-6;
    double final_logit_softcap = 0.0;
    int64_t max_position = 0;
    std::vector<AttnKind> layer_kinds;
    RopeConfig rope_sliding;
    RopeConfig rope_full;

    int64_t first_kv_shared_layer() const { return num_layers - num_kv_shared_layers; }
    bool is_kv_shared(int64_t layer) const { return num_kv_shared_layers > 0 && layer >= first_kv_shared_layer(); }
    int64_t layer_head_dim(int64_t layer) const;
    int64_t layer_intermediate_size(int64_t layer) const;
    // The earlier layer whose KV a shared layer reads: the last non-shared layer of the same kind.
    int64_t kv_source_layer(int64_t layer) const;
};

// Parses config.json (the text_config object of a multimodal config, or the root of a text-only one).
Config parse_config(const nlohmann::json& config_json);

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json);

}  // namespace xs::gemma4
