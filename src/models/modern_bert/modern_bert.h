#pragma once

// ModernBERT (Hugging Face model_type "modernbert"; reference: transformers' modeling_modernbert.py, ModernBertModel),
// the encoder of the Laya and Julia-1 decision models (models/laya).
//
// Pre-norm layers: x += Wo(attention(norm(x))); x += Wo(gelu(a) * b), [a | b] = Wi(norm(x)); every LayerNorm without
// a bias, the first layer's attention without its norm. Attention reads one fused projection (Wqkv: the query, key and
// value columns one after another), rotates queries and keys over the whole head (NeoX RoPE, positions from 0) and
// sees every token in a global layer, the tokens within local_attention / 2 on either side in a local one; the two
// kinds rotate with their own base. Token embeddings go through a norm; final_norm closes the encoder. No projection
// has a bias. Every variant beyond that (biases, another activation, RoPE scaling) is refused by name.

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph/graph.h"
#include "io/safetensors.h"

namespace xs::modern_bert {

struct Config {
    int64_t hidden_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;  // Wi writes twice this: the GELU's input, then its gate
    int64_t vocab_size = 0;
    int64_t max_positions = 0;
    double norm_eps = 1e-5;
    int64_t local_attention = 128;  // a local layer's token i sees j when |i - j| <= local_attention / 2
    double global_rope_theta = 160000.0;
    double local_rope_theta = 10000.0;
    std::vector<bool> local;  // per layer: a local (sliding_attention) layer
};

// `where` names the file in errors ("encoder/config.json"). Absent keys take ModernBertConfig's defaults.
Config parse_config(const nlohmann::json& config_json, const std::string& where);

// The encoder over `ids` [T] at `positions` [T] (0, 1, ...): its final hidden states [T, hidden_size]. The weights are
// ModernBertModel's, each name after `prefix` ("encoder." in a decision checkpoint).
graph::Value encode(graph::Graph& g, const WeightStore& weights, const Config& c, graph::Value ids,
                    graph::Value positions, const std::string& prefix);

}  // namespace xs::modern_bert
