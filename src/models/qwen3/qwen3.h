#pragma once

// The Qwen3 text decoder (Hugging Face model_type "qwen3" and "qwen3_moe"; transformers models/qwen3/ and
// qwen3_moe/): GQA attention with RMS-normed q and k per head and RoPE over the whole head, a SwiGLU MLP or a mixture
// of SwiGLU experts (a softmax router's top k, renormalized; no shared expert), RMSNorm (x * w). The decoder of the
// models that read it (Qwen3-ASR, Qwen3-Omni's thinker), and a language model of its own (load).

#include <cstdint>
#include <filesystem>
#include <memory>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/types.h"
#include "graph/graph.h"
#include "io/safetensors.h"
#include "models/model.h"
#include "tensor/tensor.h"

namespace xs::qwen3 {

struct Config {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    double rms_norm_eps = 1e-6;
    double rope_theta = 1e6;
    int64_t max_position = 0;
    bool tie_word_embeddings = false;
    // A mixture of experts when num_experts > 0, in the layers sparse() names.
    int64_t num_experts = 0;
    int64_t experts_per_token = 0;
    int64_t moe_intermediate_size = 0;
    bool norm_topk_prob = true;
    int64_t decoder_sparse_step = 1;
    std::vector<int64_t> mlp_only_layers;
    // Layer i's MLP is the mixture of experts (Qwen3MoeDecoderLayer's rule).
    bool sparse(int64_t layer) const;
    // Multimodal RoPE (rope_scaling's mrope_section, interleaved): the position axis (0 time, 1 height, 2 width) each
    // of the head's frequencies reads, when the model feeds positions on three axes. Empty: none in the config.
    std::vector<int32_t> mrope_axes;
};

// A Qwen3 text config (`where` names it in errors). Multimodal RoPE (rope_scaling with mrope_section) is plain RoPE
// for text and for media whose tokens take one position each on all three axes, as audio's do; sliding windows,
// attention biases, shared experts and other RoPE types are refused by name.
Config parse_config(const nlohmann::json& text_config, const std::string& where);

class Decoder {
public:
    // `prefix` holds embed_tokens, layers.<i> and norm (e.g. "model." or a multimodal checkpoint's
    // "thinker.model."); `lm_head` names the output layer (empty: the embedding table, tied).
    // `three_axes`: positions on three axes (time, height, width) through the config's multimodal RoPE, as a model
    // with images feeds them; else one a token.
    Decoder(Config cfg, const WeightStore& weights, std::string prefix, std::string lm_head, bool three_axes = false);

    const Config& config() const { return cfg_; }

    struct Inputs {
        graph::Value ids;  // "token_ids" [kT] i32
        graph::Value pos;  // "positions" [kT] i32, or [kT, 3] with three axes
    };
    Inputs inputs(graph::Graph& g) const;
    // Fills those inputs for `tokens` at positions start_pos, start_pos + 1, ... (one a token).
    void token_inputs(std::span<const int32_t> tokens, int64_t start_pos, TensorMap& inputs) const;
    // The same at each token's position on three axes.
    void token_inputs(std::span<const int32_t> tokens, std::span<const RotaryPosition> positions,
                      TensorMap& inputs) const;

    graph::Value embed(graph::Graph& g, graph::Value ids) const;
    // The embedded step through every layer, each causal through a KV slot of its own ("layers.<i>"); before the final
    // norm. `after_layer` (optional) takes each layer's output and gives what the next layer reads.
    using LayerHook = std::function<graph::Value(int64_t layer, graph::Value h)>;
    graph::Value layers(graph::Graph& g, graph::Value h, graph::Value pos, const LayerHook& after_layer = {}) const;
    graph::Value final_norm(graph::Graph& g, graph::Value h) const;
    graph::Value lm_head(graph::Graph& g) const;

    // Where each sparse layer's experts are ("<prefix>layers.<i>.mlp.experts."): for join_expert_banks, when a
    // checkpoint stores them expert by expert.
    std::vector<std::string> expert_prefixes() const;

private:
    graph::Value mlp(graph::Graph& g, const std::string& layer, int64_t index, graph::Value y) const;
    graph::Value weight(graph::Graph& g, const std::string& name) const { return g.weight(weights_, name); }
    graph::Value norm(graph::Graph& g, graph::Value x, const std::string& name) const;

    Config cfg_;
    const WeightStore& weights_;
    std::string prefix_;
    std::string lm_head_;
    bool three_axes_;
};

// Qwen3 as a language model (model_type qwen3, qwen3_moe; experts stored one by one joined into banks).
std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options);

}  // namespace xs::qwen3
