#pragma once

// The LFM3 text decoder as graph builders over one checkpoint's weights: what the language model makes its program
// from, and what models built on this decoder (d1, a decision model) make theirs from.

#include <span>
#include <string>

#include "graph/graph.h"
#include "io/safetensors.h"
#include "models/common/blocks.h"
#include "models/lfm3/lfm3.h"

namespace xs::lfm3 {

class Decoder {
public:
    // How a full-attention layer sees the sequence: causally through a KV slot of its own (a language model, fed in
    // steps), or the tokens of the step in groups with no cache (a model that reads its whole input in one step:
    // group 0 sees every token and every token sees it, any other group sees group 0 and itself; graph.h). Sliding
    // layers stay causal within their window either way.
    enum class Attention : uint8_t { Causal, Grouped };

    // The decoder's weights sit under "model." (lfm3) or "model.language_model." (lfm3_vl).
    Decoder(const Config& cfg, const WeightStore& weights);

    const Config& config() const { return cfg_; }
    const std::string& prefix() const { return prefix_; }

    struct Inputs {
        graph::Value ids;     // "token_ids" [kT] i32
        graph::Value pos;     // "positions" [kT] i32
        graph::Value groups;  // "groups" [kT] i32, each token's group: Attention::Grouped only
    };
    Inputs inputs(graph::Graph& g, Attention attention = Attention::Causal) const;
    // Fills the token ids and positions for `tokens` at positions start_pos, start_pos + 1, ...
    void token_inputs(std::span<const int32_t> tokens, int64_t start_pos, TensorMap& inputs) const;

    // The embedded tokens through every decoder layer: the state after the last one (before the final norm). The
    // full-attention layers read in groups when `in` has them (Attention::Grouped), else causally.
    graph::Value layers(graph::Graph& g, const Inputs& in) const;
    // The norm after the last layer.
    graph::Value final_norm(graph::Graph& g, graph::Value h) const {
        return norm(g, h, prefix_ + "embedding_norm.weight");
    }
    // The embedding table, which is also the output layer (the head is tied to it).
    std::string embedding_name() const { return prefix_ + "embed_tokens.weight"; }
    graph::Value embedding(graph::Graph& g) const { return weight(g, embedding_name()); }

private:
    graph::Value weight(graph::Graph& g, const std::string& name) const { return g.weight(weights_, name); }
    graph::Value norm(graph::Graph& g, graph::Value x, const std::string& name) const;
    blocks::Linear proj(graph::Graph& g, const std::string& name) const;
    graph::Value inv_freq(graph::Graph& g, const Rope& rope) const;
    // `groups`: the tokens' groups for full-attention layers in groups; invalid: causal.
    graph::Value layer(graph::Graph& g, const std::string& L, const std::string& label, LayerKind kind, graph::Value h,
                       graph::Value pos, graph::Value freq, graph::Value groups) const;
    graph::Value short_conv(graph::Graph& g, const std::string& layer, const std::string& label, graph::Value x) const;
    graph::Value self_attention(graph::Graph& g, const std::string& layer, const std::string& label, LayerKind kind,
                                graph::Value x, graph::Value pos, graph::Value freq, graph::Value groups) const;

    const Config& cfg_;
    const WeightStore& weights_;
    std::string prefix_;
};

}  // namespace xs::lfm3
