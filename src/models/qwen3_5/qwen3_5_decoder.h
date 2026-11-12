#pragma once

// The Qwen3.5 text decoder as graph builders over one checkpoint's weights: what the language model makes its
// programs from, and what models built on this decoder (decision models) make theirs from.

#include <span>
#include <string>
#include <vector>

#include "models/common/blocks.h"
#include "models/qwen3_5/qwen3_5.h"

namespace xs::qwen3_5 {

class Decoder {
public:
    // How a full-attention layer sees the sequence: causally through a KV slot of its own (a language model, fed in
    // steps), or every token of the step with no cache (a model that reads its whole input in one step).
    enum class Attention : uint8_t { Causal, WholeStep };

    // The decoder's weights sit under "model.language_model." (multimodal checkpoints), "model." (text-only ones) or
    // "language_model." (a bare Qwen3_5Model).
    Decoder(const Config& cfg, const WeightStore& weights);

    const Config& config() const { return cfg_; }
    const std::string& prefix() const { return prefix_; }

    struct Inputs {
        graph::Value ids;  // "token_ids" [kT] i32
        graph::Value pos;  // "positions" [kT] i32
    };
    Inputs inputs(graph::Graph& g) const;
    // Fills those inputs for `tokens` at positions start_pos, start_pos + 1, ...
    void token_inputs(std::span<const int32_t> tokens, int64_t start_pos, TensorMap& inputs) const;

    // The embedded tokens through every decoder layer: the state after each layer (before the final norm).
    std::vector<graph::Value> layers(graph::Graph& g, const Inputs& in, Attention attention = Attention::Causal) const;
    // One decoder layer from the weights under `L` (".../layers.<i>."); `label` names its state slots.
    graph::Value layer(graph::Graph& g, const std::string& L, const std::string& label, LayerKind kind, graph::Value h,
                       graph::Value pos, graph::Value freq, Attention attention = Attention::Causal) const;
    // The norm after the last layer.
    graph::Value final_norm(graph::Graph& g, graph::Value h) const { return norm(g, h, prefix_ + "norm.weight"); }

    // Qwen3_5RMSNorm is zero-centered: x * rsqrt(mean(x^2) + eps) * (1 + w).
    graph::Value norm(graph::Graph& g, graph::Value x, const std::string& name) const;
    graph::Value weight(graph::Graph& g, const std::string& name) const { return g.weight(weights_, name); }
    graph::Value embedding(graph::Graph& g) const { return weight(g, prefix_ + "embed_tokens.weight"); }
    // The output layer: lm_head.weight, or the embedding table of a checkpoint that ties them.
    std::string lm_head_name() const;
    graph::Value lm_head(graph::Graph& g) const { return weight(g, lm_head_name()); }
    graph::Value inv_freq(graph::Graph& g) const;

private:
    float eps() const { return static_cast<float>(cfg_.rms_norm_eps); }
    blocks::Linear proj(graph::Graph& g, const std::string& name) const;
    graph::Value moe(graph::Graph& g, const std::string& L, graph::Value y) const;
    graph::Value linear_attention(graph::Graph& g, const std::string& layer, const std::string& label,
                                  graph::Value x) const;
    graph::Value full_attention(graph::Graph& g, const std::string& layer, const std::string& label, graph::Value x,
                                graph::Value pos, graph::Value freq, Attention attention) const;

    const Config& cfg_;
    const WeightStore& weights_;
    std::string prefix_;
};

}  // namespace xs::qwen3_5
