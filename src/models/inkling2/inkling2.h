#pragma once

// Inkling 2 Mini (Thinking Machines; Hugging Face model_type "inkling2_loop"). Reference: transformers'
// modeling_inkling2_loop.py (Inkling2LoopTextModel and its layers).
//
// A decoder without rotary positions whose middle layers run several times. Per layer: attention whose scores add a
// bias by how far back a key sits (a row per head mixed from the query's own projection) and whose softmax has one
// more logit, a sink; queries and keys clipped to a norm, values normed, keys and values through a short causal
// convolution (in the full-attention layers with taps per token); a mixture of experts whose hidden state is a
// softcapped SwiGLU; a short convolution after each sublayer. Most layers attend within a sliding window, every sixth
// over the whole context. A layer's input is not the residual stream alone: learned queries mix it with the states
// at the start of each block of six layers (block residuals). The layers `loop.start` to `loop.end` run `loop.passes`
// times with the same weights, each pass from the state before the first, with KV and convolution state of its own
// and, from the second on, messages mixed from the pass before it through learned gates.
//
// Images (transformers' Inkling2VisionModel): a tower of pre-norm layers with 2-D rotary attention and layer scale
// over patches of `patch` pixels a side and one class token, then a two-layer projection of each square of `merge`
// by `merge` patches to one row of the decoder's width, which stands where the prompt holds the image token.

#include <filesystem>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/model.h"

namespace xs::inkling2 {

struct Loop {
    int64_t start = 0;      // the first looped layer
    int64_t end = 0;        // one past the last
    int64_t passes = 1;
    int64_t gate_rank = 0;  // width of a message gate's inner projection
};

struct Config {
    int64_t vocab_size = 0;      // rows of the embedding table, and of the output layer as stored
    int64_t output_size = 0;     // rows of the output layer that are tokens (unpadded_vocab_size); the rest are padding
    int64_t hidden_size = 0;
    int64_t num_layers = 0;
    double norm_eps = 1e-6;
    int64_t max_position = 0;
    // Attention.
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    std::vector<bool> sliding;   // per layer: attends within sliding_window (else over the whole context)
    int64_t sliding_window = 0;
    int64_t rel_width = 0;       // d_rel: a head's relative state
    int64_t rel_extent = 0;      // keys this far back or more get no bias (full layers; sliding ones: their window)
    double qk_clip = 0.0;        // largest L2 norm of a query or key head
    double rel_clip = 0.0;       // and of a relative state
    // Full layers scale queries and relative states by 1 + alpha * ln(max((position + 1) / floor, 1)).
    double log_scale_alpha = 0.0;
    double log_scale_floor = 1.0;
    int64_t conv_kernel = 0;     // taps of every short convolution
    int64_t block_layers = 0;    // layers a block residual spans
    // Mixture of experts.
    int64_t experts = 0;
    int64_t experts_per_token = 0;
    int64_t expert_width = 0;
    // The fused gate and up bank's rows: gate 0, up 0, gate 1, ... as released, or the gate rows then the up rows
    // (a GGUF import: gate_up_rows "halves").
    bool gate_up_alternates = true;
    double route_scale = 1.0;
    double gate_softcap = 0.0;
    double up_softcap = 0.0;
    double logit_softcap = 0.0;
    Loop loop;

    // Layer applications a token goes through: the looped layers count once per pass.
    int64_t applications() const { return num_layers + (loop.passes - 1) * (loop.end - loop.start); }
};

// `config_json`: an inkling2_loop checkpoint's config.json. Throws naming what is not supported.
Config parse_config(const nlohmann::json& config_json);

struct VisionConfig {
    int64_t hidden = 0;
    int64_t layers = 0;
    int64_t heads = 0;
    int64_t intermediate = 0;
    int64_t channels = 3;
    int64_t patch = 0;       // native_patch_size: pixels a side
    int64_t merge = 0;       // adapter_merge_size: patches a side of the square one image token holds
    int64_t grid = 0;        // posemb_grid_size: side of the position table, interpolated to an image's patches
    int64_t projection = 0;  // projection_hidden_dim
    int64_t out_hidden = 0;  // text_hidden_size
    double norm_eps = 1e-5;
    double rope_theta = 10000.0;
    int32_t image_token = 0;  // what the chat template renders once an image and the prompt holds once a row
};

// The vision_config of `config_json`, with the image token (image_token_id). Throws naming what is not supported.
VisionConfig parse_vision_config(const nlohmann::json& config_json);

// The tower over `weights` (the released names, under model.visual.).
std::unique_ptr<VisionEncoder> make_vision(const VisionConfig& cfg, const WeightStore& weights);

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options);

}  // namespace xs::inkling2
