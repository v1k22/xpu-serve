#pragma once

// Qwen3.5 (Hugging Face model_type "qwen3_5" / "qwen3_5_text"). Reference: transformers models/qwen3_5/
// modeling_qwen3_5.py (Qwen3_5TextModel, Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet, Qwen3_5Attention). Also SenseNova
// Loop ("sensenova_loop", the checkpoint's modeling_sensenova_loop.py): the same decoder with a block of layers run
// several times (loop_settings) and manifold-constrained hyper-connections inside it (mhc).
//
// A hybrid decoder: most layers are Gated DeltaNet linear attention (a causal depthwise conv over q, k, v, then the
// gated delta rule with a fixed-size state per sequence), every `full_attention_interval`-th layer is gated full
// attention (sigmoid output gate, q/k norms, partial rotary). Norms are zero-centered (x * (1 + w)). Served: the text
// decoder, dense ("qwen3_5") or mixture of experts in every layer ("qwen3_5_moe": softmax router, top-k renormalized,
// expert banks whose gate and up projections come fused, as Hugging Face stores them, or apart, as a GGUF import keeps
// them, a sigmoid-gated shared expert), and its MTP head (mtp.*) as the drafter for speculative decoding. The vision
// tower is not loaded.

#include <filesystem>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/model.h"

namespace xs::qwen3_5 {

enum class LayerKind { Linear, Full };

// How a checkpoint orders linear attention's value heads, r = value heads / key heads of them per key head, in every
// tensor indexed by value head (in_proj_qkv's value rows, in_proj_z, in_proj_a, in_proj_b, A_log, dt_bias, the conv's
// value channels, out_proj's columns). Grouped (Hugging Face): head k * r + s is the s-th of key head k's. Tiled (GGUF
// files, which llama.cpp's converter writes so): head s * key heads + k is. config.json linear_value_head_order names
// it ("grouped" when absent).
enum class ValueHeadOrder : uint8_t { Grouped, Tiled };

// Manifold-constrained hyper-connections (SenseNova Loop's MHCRouter): layers [start, end) carry `streams` residual
// streams of hidden_size each. Each of a layer's two branches (attention, MLP) reads a weighted sum of the streams and
// writes its output back into every stream, through a router of its own (mhc_attn, mhc_ffn): from the RMS-normed
// streams, read weights sigmoid(.), write weights 2 sigmoid(.) and a stream mix made doubly stochastic by `iterations`
// of Sinkhorn. The first layer copies the hidden state into every stream, the last averages them.
struct Mhc {
    int64_t start = 0;
    int64_t end = 0;
    int64_t streams = 0;  // 0: no hyper-connections
    int iterations = 0;
    float pre_scale = 1.0f;  // multiplies what a branch reads
};

struct Config {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;  // dense MLP (0 for MoE)
    // Mixture of experts (num_experts > 0).
    int64_t num_experts = 0;
    int64_t experts_per_token = 0;
    int64_t moe_intermediate_size = 0;
    int64_t shared_expert_intermediate_size = 0;  // 0: no shared expert
    bool norm_topk_prob = true;
    int64_t num_layers = 0;
    double rms_norm_eps = 1e-6;
    int64_t max_position = 0;
    std::vector<LayerKind> layer_kinds;
    // Full attention.
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    bool attn_output_gate = true;
    double rope_theta = 10000.0;
    int64_t rotary_dim = 0;  // head_dim * partial_rotary_factor
    // Gated DeltaNet.
    int64_t linear_key_heads = 0;
    int64_t linear_value_heads = 0;
    int64_t linear_key_dim = 0;
    int64_t linear_value_dim = 0;
    int64_t conv_kernel = 0;
    ValueHeadOrder value_head_order = ValueHeadOrder::Grouped;
    int64_t mtp_layers = 0;  // decoder layers of the MTP head (mtp.layers.<i>); 0: no head
    // The layers in the order they run (loop_settings: [0, start) once, [start, end) eval_count times, the rest
    // once); every run of a layer has state of its own and the layer's weights. Empty: each layer once, in order.
    std::vector<int64_t> plan;
    Mhc mhc;
    bool looped() const { return !plan.empty() && static_cast<int64_t>(plan.size()) != num_layers; }
};

// Parses config.json (the text_config object of a multimodal config, or the root of a text-only one).
Config parse_config(const nlohmann::json& config_json);

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options);

}  // namespace xs::qwen3_5
