#pragma once

// LFM3 (Liquid AI; Hugging Face model_type "lfm3", and the text decoder of "lfm3_vl"). Reference: the checkpoint's
// modeling_lfm3.py (Lfm3Model, Lfm3DecoderLayer, Lfm3ShortConv, Lfm3Attention).
//
// A hybrid decoder: most layers are gated short convolutions (one projection gives B, C and x; the causal depthwise
// conv of B * x, times C, then a projection: a window of conv_L_cache tokens, a fixed-size state per sequence), the
// others attention with q/k norms and a sigmoid gate per head, over the whole context ("full_attention", the first
// half of each head rotary) or a sliding window. Every layer's operator and MLP outputs are normed again before they
// join the residual stream (sandwich norms); the head is tied to the embedding. Served: the text decoder
// (lfm3_decoder.h builds its layers), as a language model and under the d1 decision model (models/d1/).

#include <filesystem>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/model.h"

namespace xs::lfm3 {

enum class LayerKind { Conv, Full, Sliding };

// RoPE of one attention kind: the first `pairs` of a head's head_dim / 2 channel pairs rotate, at the frequencies of
// the whole head (theta^(-2 i / head_dim)); the rest do not.
struct Rope {
    double theta = 0.0;
    int64_t pairs = 0;
};

struct Config {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    double norm_eps = 1e-5;
    int64_t max_position = 0;
    std::vector<LayerKind> layer_kinds;
    // Attention.
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t sliding_window = 0;  // of the sliding layers (0: the model has none)
    Rope full_rope, sliding_rope;
    // Gated short convolution.
    int64_t conv_kernel = 0;
};

// `text`: the text decoder's part of config.json (text_config()).
Config parse_config(const nlohmann::json& text);

// The text decoder's part of config.json: all of an lfm3 checkpoint's, the text_config of an lfm3_vl one (its vision
// tower and projector are not read). Throws naming any other model_type.
const nlohmann::json& text_config(const nlohmann::json& config_json);

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options);

}  // namespace xs::lfm3
