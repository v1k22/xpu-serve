#include "models/qwen3_5/qwen3_5_decoder.h"

#include <cmath>
#include <cstring>

#include "util/error.h"
#include "util/half.h"

namespace xs::qwen3_5 {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

std::string probe_prefix(const WeightStore& weights) {
    for (const char* prefix : {"model.language_model.", "model.", "language_model."}) {
        if (weights.contains(std::string(prefix) + "embed_tokens.weight")) return prefix;
    }
    fail("qwen3_5: no embed_tokens.weight under model.language_model., model. or language_model.; is this a Qwen3.5 ",
         "checkpoint?");
}

HostTensor to_f32(const std::string& name, const TensorView& t) {
    HostTensor out(DType::F32, t.shape);
    const auto n = static_cast<size_t>(t.numel());
    switch (t.dtype) {
        case DType::F32:
            std::memcpy(out.raw(), t.data, n * sizeof(float));
            break;
        case DType::BF16:
            convert_bf16_to_f32(static_cast<const uint16_t*>(t.data), out.data<float>(), n);
            break;
        case DType::F16:
            convert_f16_to_f32(static_cast<const uint16_t*>(t.data), out.data<float>(), n);
            break;
        default:
            fail("qwen3_5: '", name, "' has dtype ", dtype_name(t.dtype));
    }
    return out;
}

}  // namespace

Decoder::Decoder(const Config& cfg, const WeightStore& weights)
    : cfg_(cfg), weights_(weights), prefix_(probe_prefix(weights)) {}

Decoder::Inputs Decoder::inputs(Graph& g) const {
    return {g.input("token_ids", DType::I32, {kT}), g.input("positions", DType::I32, {kT})};
}

void Decoder::token_inputs(std::span<const int32_t> tokens, int64_t start_pos, TensorMap& inputs) const {
    const auto n = static_cast<int64_t>(tokens.size());
    XS_CHECK(n > 0, "qwen3_5: a step needs at least one token");
    HostTensor ids(DType::I32, {n});
    HostTensor pos(DType::I32, {n});
    for (int64_t i = 0; i < n; ++i) {
        const int32_t t = tokens[static_cast<size_t>(i)];
        if (t < 0 || t >= cfg_.vocab_size)
            fail("qwen3_5: token id ", t, " outside the vocabulary (", cfg_.vocab_size, ")");
        ids.data<int32_t>()[i] = t;
        pos.data<int32_t>()[i] = static_cast<int32_t>(start_pos + i);
    }
    inputs["token_ids"] = std::move(ids);
    inputs["positions"] = std::move(pos);
}

std::vector<Value> Decoder::layers(Graph& g, const Inputs& in, Attention attention) const {
    Value h = g.gather(embedding(g), in.ids);
    const Value freq = inv_freq(g);
    std::vector<Value> after;
    for (int64_t i = 0; i < cfg_.num_layers; ++i) {
        const std::string name = "layers." + std::to_string(i);
        h = layer(g, prefix_ + name + ".", name, cfg_.layer_kinds[static_cast<size_t>(i)], h, in.pos, freq, attention);
        after.push_back(h);
    }
    return after;
}

Value Decoder::layer(Graph& g, const std::string& L, const std::string& label, LayerKind kind, Value h, Value pos,
                     Value freq, Attention attention) const {
    Graph::Scope s(g, label);
    const Value x = norm(g, h, L + "input_layernorm.weight");
    const Value mixed = kind == LayerKind::Linear ? linear_attention(g, L, label, x)
                                                  : full_attention(g, L, label, x, pos, freq, attention);
    h = g.add(h, mixed);
    const Value y = norm(g, h, L + "post_attention_layernorm.weight");
    if (cfg_.num_experts > 0) return g.add(h, moe(g, L + "mlp.", y));
    return g.add(h, blocks::gated_mlp(g, y, proj(g, L + "mlp.gate_proj.weight"), proj(g, L + "mlp.up_proj.weight"),
                                      proj(g, L + "mlp.down_proj.weight"), blocks::Act::Silu));
}

Value Decoder::norm(Graph& g, Value x, const std::string& name) const {
    return blocks::rms_norm_one_plus(g, x, weights_, name, eps());
}

std::string Decoder::lm_head_name() const {
    return weights_.contains("lm_head.weight") ? "lm_head.weight" : prefix_ + "embed_tokens.weight";
}

Value Decoder::inv_freq(Graph& g) const {
    return g.constant(
        blocks::rope_inv_freq(cfg_.rope_theta, cfg_.rotary_dim, cfg_.rotary_dim / 2, cfg_.rotary_dim / 2));
}

// A projection whose weight node is made when the block uses it.
blocks::Linear Decoder::proj(Graph& g, const std::string& name) const {
    return [this, &g, name](Value x) { return g.matmul(x, weight(g, name)); };
}

// Qwen3_5MoeSparseMoeBlock.
Value Decoder::moe(Graph& g, const std::string& L, Value y) const {
    blocks::MoeSpec spec;
    spec.experts = cfg_.num_experts;
    spec.top_k = cfg_.experts_per_token;
    spec.router = blocks::Router::Softmax;
    spec.normalize = cfg_.norm_topk_prob;
    blocks::MoeParts parts;
    parts.router = weight(g, L + "gate.weight");
    if (!weights_.contains(L + "experts.gate_up_proj")) {
        fail("qwen3_5: ", L, "experts.gate_up_proj is missing: the expert banks come fused, or expert by expert ",
             "as fp8 tensors (experts.<e>.gate_proj.weight)");
    }
    parts.gate_up = weight(g, L + "experts.gate_up_proj");
    parts.down = weight(g, L + "experts.down_proj");
    const int64_t e = cfg_.num_experts, h = cfg_.hidden_size, inter = cfg_.moe_intermediate_size;
    XS_CHECK(g.type(parts.gate_up).shape == (Dims{e, 2 * inter, h}) && g.type(parts.down).shape == (Dims{e, h, inter}),
             "qwen3_5: ", L, " expert banks have shapes ", dims_to_string(g.type(parts.gate_up).shape), " and ",
             dims_to_string(g.type(parts.down).shape), ", expected [", e, ", ", 2 * inter, ", ", h, "] and [", e, ", ",
             h, ", ", inter, "]");
    if (cfg_.shared_expert_intermediate_size > 0) {
        parts.shared_gate = proj(g, L + "shared_expert.gate_proj.weight");
        parts.shared_up = proj(g, L + "shared_expert.up_proj.weight");
        parts.shared_down = proj(g, L + "shared_expert.down_proj.weight");
        parts.shared_expert_gate = weight(g, L + "shared_expert_gate.weight");
    }
    return blocks::moe(g, y, spec, parts);
}

// Qwen3_5GatedDeltaNet.
Value Decoder::linear_attention(Graph& g, const std::string& layer, const std::string& label, Value x) const {
    Graph::Scope s(g, "linear_attn");
    const std::string L = layer + "linear_attn.";
    const int64_t hk = cfg_.linear_key_heads, hv = cfg_.linear_value_heads;
    const int64_t dk = cfg_.linear_key_dim, dv = cfg_.linear_value_dim;
    const int64_t key_width = hk * dk, value_width = hv * dv;
    const int64_t conv_width = 2 * key_width + value_width;

    // q, k and v share one projection and one causal conv (with SiLU), then split.
    const Value conv_w = weight(g, L + "conv1d.weight");
    XS_CHECK(g.type(conv_w).shape == (Dims{conv_width, 1, cfg_.conv_kernel}), "qwen3_5: ", L, "conv1d.weight has ",
             "shape ", dims_to_string(g.type(conv_w).shape));
    const int conv_slot = g.add_conv_slot(conv_width, cfg_.conv_kernel, label);
    const Value qkv = g.silu(g.causal_conv(conv_slot, g.matmul(x, weight(g, L + "in_proj_qkv.weight")),
                                           g.reshape(conv_w, {conv_width, cfg_.conv_kernel})));
    const Value q = g.reshape(g.slice(qkv, 1, 0, key_width), {kT, hk, dk});
    const Value k = g.reshape(g.slice(qkv, 1, key_width, 2 * key_width), {kT, hk, dk});
    const Value v = g.reshape(g.slice(qkv, 1, 2 * key_width, conv_width), {kT, hv, dv});

    // beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), the per-token log decay.
    const Value beta = g.sigmoid(g.matmul(x, weight(g, L + "in_proj_b.weight")));
    HostTensor neg_a = to_f32(L + "A_log", weights_.get(L + "A_log"));
    for (int64_t j = 0; j < neg_a.numel(); ++j)
        neg_a.data<float>()[j] = -std::exp(neg_a.data<float>()[j]);
    const Value decay =
        g.mul(g.softplus(g.add(g.matmul(x, weight(g, L + "in_proj_a.weight")), weight(g, L + "dt_bias"))),
              g.constant(std::move(neg_a)));

    const int slot = g.add_recurrent_slot(hv, dk, dv, label);
    Value o = g.gated_delta(slot, q, k, v, decay, beta, 1e-6f);

    // Qwen3_5RMSNormGated: rms_norm(o) * w * silu(z), per value head.
    const Value z = g.reshape(g.matmul(x, weight(g, L + "in_proj_z.weight")), {kT, hv, dv});
    o = blocks::gated_rms_norm(g, o, weight(g, L + "norm.weight"), z, eps());
    return g.matmul(g.reshape(o, {kT, value_width}), weight(g, L + "out_proj.weight"));
}

// Qwen3_5Attention: q_proj yields [query | gate] per head; the output is gated by sigmoid(gate).
Value Decoder::full_attention(Graph& g, const std::string& layer, const std::string& label, Value x, Value pos,
                              Value freq, Attention attention) const {
    Graph::Scope s(g, "self_attn");
    const std::string L = layer + "self_attn.";
    blocks::AttentionSpec spec;
    spec.heads = cfg_.num_heads;
    spec.kv_heads = cfg_.num_kv_heads;
    spec.head_dim = cfg_.head_dim;
    spec.attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg_.head_dim)));
    spec.rotary_dim = cfg_.rotary_dim;
    spec.output_gate = cfg_.attn_output_gate;
    spec.kv_label = label;
    if (attention == Attention::WholeStep) {
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
    return blocks::attention(g, spec, parts, x, pos, freq).out;
}

}  // namespace xs::qwen3_5
