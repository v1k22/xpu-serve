#include "models/lfm3/lfm3_decoder.h"

#include <cmath>

#include "util/error.h"

namespace xs::lfm3 {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

std::string probe_prefix(const WeightStore& weights) {
    for (const char* prefix : {"model.language_model.", "model."}) {
        if (weights.contains(std::string(prefix) + "embed_tokens.weight")) return prefix;
    }
    fail("lfm3: no embed_tokens.weight under model.language_model. or model.; is this an LFM3 checkpoint?");
}

}  // namespace

Decoder::Decoder(const Config& cfg, const WeightStore& weights)
    : cfg_(cfg), weights_(weights), prefix_(probe_prefix(weights)) {}

Decoder::Inputs Decoder::inputs(Graph& g) const {
    return {g.input("token_ids", DType::I32, {kT}), g.input("positions", DType::I32, {kT})};
}

void Decoder::token_inputs(std::span<const int32_t> tokens, int64_t start_pos, TensorMap& inputs) const {
    const auto n = static_cast<int64_t>(tokens.size());
    XS_CHECK(n > 0, "lfm3: a step needs at least one token");
    HostTensor ids(DType::I32, {n});
    HostTensor pos(DType::I32, {n});
    for (int64_t i = 0; i < n; ++i) {
        const int32_t t = tokens[static_cast<size_t>(i)];
        if (t < 0 || t >= cfg_.vocab_size) {
            fail("lfm3: token id ", t, " outside the vocabulary (", cfg_.vocab_size, ")");
        }
        ids.data<int32_t>()[i] = t;
        pos.data<int32_t>()[i] = static_cast<int32_t>(start_pos + i);
    }
    inputs["token_ids"] = std::move(ids);
    inputs["positions"] = std::move(pos);
}

Value Decoder::layers(Graph& g, const Inputs& in, Attention attention) const {
    Value h = g.gather(embedding(g), in.ids);
    const Value full_freq = inv_freq(g, cfg_.full_rope);
    const Value sliding_freq = inv_freq(g, cfg_.sliding_rope);
    for (int64_t i = 0; i < cfg_.num_layers; ++i) {
        const std::string label = "layers." + std::to_string(i);
        const LayerKind kind = cfg_.layer_kinds[static_cast<size_t>(i)];
        h = layer(g, prefix_ + label + ".", label, kind, h, in.pos,
                  kind == LayerKind::Sliding ? sliding_freq : full_freq, attention);
    }
    return h;
}

Value Decoder::norm(Graph& g, Value x, const std::string& name) const {
    return g.rms_norm(x, weight(g, name), static_cast<float>(cfg_.norm_eps));
}

// A projection whose weight node is made when the block uses it.
blocks::Linear Decoder::proj(Graph& g, const std::string& name) const {
    return [this, &g, name](Value x) { return g.matmul(x, weight(g, name)); };
}

// The frequencies of one attention kind; invalid when the model has no layer of it.
Value Decoder::inv_freq(Graph& g, const Rope& rope) const {
    if (rope.pairs == 0) return {};
    return g.constant(blocks::rope_inv_freq(rope.theta, cfg_.head_dim, rope.pairs, cfg_.head_dim / 2));
}

// Lfm3DecoderLayer: the operator and the MLP each read a normed stream and are normed again before joining it.
Value Decoder::layer(Graph& g, const std::string& L, const std::string& label, LayerKind kind, Value h, Value pos,
                     Value freq, Attention attention) const {
    Graph::Scope s(g, label);
    const Value x = norm(g, h, L + "operator_norm.weight");
    const Value mixed = kind == LayerKind::Conv ? short_conv(g, L, label, x)
                                                : self_attention(g, L, label, kind, x, pos, freq, attention);
    h = g.add(h, norm(g, mixed, L + "post_operator_norm.weight"));
    const Value y = norm(g, h, L + "ffn_norm.weight");
    const std::string M = L + "feed_forward.";  // w2(silu(w1 x) * w3 x)
    const Value mlp = blocks::gated_mlp(g, y, proj(g, M + "w1.weight"), proj(g, M + "w3.weight"),
                                        proj(g, M + "w2.weight"), blocks::Act::Silu);
    return g.add(h, norm(g, mlp, L + "post_ffn_norm.weight"));
}

// Lfm3ShortConv: in_proj gives [B | C | x]; out_proj(C * conv(B * x)), a causal depthwise conv over conv_kernel
// tokens.
Value Decoder::short_conv(Graph& g, const std::string& layer, const std::string& label, Value x) const {
    Graph::Scope s(g, "conv");
    const std::string L = layer + "conv.";
    const int64_t width = cfg_.hidden_size, kernel = cfg_.conv_kernel;
    const Value in_w = weight(g, L + "in_proj.weight");
    const Value conv_w = weight(g, L + "conv.weight");
    XS_CHECK(g.type(in_w).shape == (Dims{3 * width, width}) && g.type(conv_w).shape == (Dims{width, 1, kernel}),
             "lfm3: ", L, "in_proj.weight and conv.weight have shapes ", dims_to_string(g.type(in_w).shape), " and ",
             dims_to_string(g.type(conv_w).shape), ", expected [", 3 * width, ", ", width, "] and [", width, ", 1, ",
             kernel, "]");
    const Value bcx = g.matmul(x, in_w);
    const Value b = g.slice(bcx, 1, 0, width);
    const Value c = g.slice(bcx, 1, width, 2 * width);
    const Value gated = g.mul(b, g.slice(bcx, 1, 2 * width, 3 * width));
    const int slot = g.add_conv_slot(width, kernel, label);
    const Value conv = g.causal_conv(slot, gated, g.reshape(conv_w, {width, kernel}));
    return g.matmul(g.mul(c, conv), weight(g, L + "out_proj.weight"));
}

// Lfm3Attention: q/k norms per head, RoPE, and each head's output scaled by sigmoid(gate_proj(x)).
Value Decoder::self_attention(Graph& g, const std::string& layer, const std::string& label, LayerKind kind, Value x,
                              Value pos, Value freq, Attention attention) const {
    Graph::Scope s(g, "self_attn");
    const std::string L = layer + "self_attn.";
    blocks::AttentionSpec spec;
    spec.heads = cfg_.num_heads;
    spec.kv_heads = cfg_.num_kv_heads;
    spec.head_dim = cfg_.head_dim;
    spec.attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg_.head_dim)));
    spec.attrs.sliding_window = kind == LayerKind::Sliding ? cfg_.sliding_window : 0;
    spec.kv_window = spec.attrs.sliding_window;
    spec.kv_label = label;
    if (kind == LayerKind::Full && attention == Attention::WholeStep) {
        spec.kv_cache = false;
        spec.attrs.causal = false;
    }
    blocks::AttentionParts parts;
    parts.q = proj(g, L + "q_proj.weight");
    parts.k = proj(g, L + "k_proj.weight");
    parts.v = proj(g, L + "v_proj.weight");
    parts.o = proj(g, L + "o_proj.weight");
    parts.q_norm = [this, &g, L](Value q) { return norm(g, q, L + "q_norm.weight"); };
    parts.k_norm = [this, &g, L](Value k) { return norm(g, k, L + "k_norm.weight"); };
    parts.head_gate = proj(g, L + "gate_proj.weight");
    return blocks::attention(g, spec, parts, x, pos, freq).out;
}

}  // namespace xs::lfm3
