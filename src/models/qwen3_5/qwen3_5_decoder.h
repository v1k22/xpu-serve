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
    // "language_model." (a bare Qwen3_5Model). `three_axes`: positions on three axes (time, height, width) through the
    // config's multimodal RoPE (Config::mrope_axes), as a model that reads images feeds them.
    Decoder(const Config& cfg, const WeightStore& weights, bool three_axes = false);

    const Config& config() const { return cfg_; }
    const std::string& prefix() const { return prefix_; }

    struct Inputs {
        graph::Value ids;  // "token_ids" [kT] i32
        graph::Value pos;  // "positions" [kT] i32, or [kT, 3] with three axes
    };
    Inputs inputs(graph::Graph& g) const;
    // Fills those inputs for `tokens` at positions start_pos, start_pos + 1, ... (on every axis).
    void token_inputs(std::span<const int32_t> tokens, int64_t start_pos, TensorMap& inputs) const;
    // The same at each token's position on three axes (a decoder built with three_axes).
    void token_inputs(std::span<const int32_t> tokens, std::span<const RotaryPosition> positions,
                      TensorMap& inputs) const;

    // The full-attention layers' rotary frequencies, and with three axes the axis each one reads.
    struct Rope {
        graph::Value freq;
        graph::Value axes;  // invalid without three axes
    };
    Rope rope(graph::Graph& g) const;

    // The embedded tokens through the decoder's layers in the order they run (Config::plan): the state after each
    // run (before the final norm; inside the hyper-connections, their streams [T, streams, hidden]). A layer's first
    // run keeps its state slots under "layers.<i>", a later one under "layers.<i>.pass<n>".
    std::vector<graph::Value> layers(graph::Graph& g, const Inputs& in, Attention attention = Attention::Causal) const;
    // The same from rows `h` [kT, hidden] in place of the embedded tokens (an image's rows among them).
    std::vector<graph::Value> layers(graph::Graph& g, const Inputs& in, graph::Value h,
                                     Attention attention = Attention::Causal) const;
    // The tokens' rows of the embedding table.
    graph::Value embed(graph::Graph& g, const Inputs& in) const { return g.gather(embedding(g), in.ids); }
    // One decoder layer from the weights under `L` (".../layers.<i>."); `label` names its state slots.
    graph::Value layer(graph::Graph& g, const std::string& L, const std::string& label, LayerKind kind, graph::Value h,
                       graph::Value pos, const Rope& rope, Attention attention = Attention::Causal) const;
    // The norm after the last layer.
    graph::Value final_norm(graph::Graph& g, graph::Value h) const { return norm(g, h, prefix_ + "norm.weight"); }

    // Qwen3_5RMSNorm is zero-centered: x * rsqrt(mean(x^2) + eps) * (1 + w).
    graph::Value norm(graph::Graph& g, graph::Value x, const std::string& name) const;
    graph::Value weight(graph::Graph& g, const std::string& name) const { return g.weight(weights_, name); }
    graph::Value embedding(graph::Graph& g) const { return weight(g, prefix_ + "embed_tokens.weight"); }
    // The output layer: lm_head.weight, or the embedding table of a checkpoint that ties them.
    std::string lm_head_name() const;
    graph::Value lm_head(graph::Graph& g) const { return weight(g, lm_head_name()); }

private:
    float eps() const { return static_cast<float>(cfg_.rms_norm_eps); }
    blocks::Linear proj(graph::Graph& g, const std::string& name) const;
    // A layer's two branches, on their normed inputs: attention of its kind, and the MLP (dense or experts).
    graph::Value mixer(graph::Graph& g, const std::string& L, const std::string& label, LayerKind kind, graph::Value x,
                       graph::Value pos, const Rope& rope, Attention attention) const;
    graph::Value mlp(graph::Graph& g, const std::string& L, graph::Value y) const;
    // Layer `physical` of the hyper-connections (Config::mhc): h is the hidden state [T, hidden] at their first
    // layer, else their streams [T, streams, hidden]; the result is the streams, or their mean after the last layer.
    graph::Value mhc_layer(graph::Graph& g, const std::string& L, const std::string& label, int64_t physical,
                           graph::Value h, graph::Value pos, const Rope& rope, Attention attention) const;
    template <typename Branch>
    graph::Value mhc_branch(graph::Graph& g, const std::string& router, graph::Value r, Branch branch) const;
    graph::Value moe(graph::Graph& g, const std::string& L, graph::Value y) const;
    graph::Value linear_attention(graph::Graph& g, const std::string& layer, const std::string& label,
                                  graph::Value x) const;
    graph::Value full_attention(graph::Graph& g, const std::string& layer, const std::string& label, graph::Value x,
                                graph::Value pos, const Rope& rope, Attention attention) const;

    const Config& cfg_;
    const WeightStore& weights_;
    std::string prefix_;
    bool three_axes_;
};

}  // namespace xs::qwen3_5
