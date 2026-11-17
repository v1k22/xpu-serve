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

// Heads of x [kT, a * b, d] (or [kT, a * b] when d is 0) with head i * b + j moved to j * a + i: value heads from the
// tiled order to the grouped one with a = r (value heads per key head), b = key heads; back with a and b swapped.
Value swap_head_order(Graph& g, Value x, int64_t a, int64_t b, int64_t d) {
    if (d == 0) return g.reshape(g.transpose(g.reshape(x, {kT, a, b}), {0, 2, 1}), {kT, a * b});
    return g.reshape(g.transpose(g.reshape(x, {kT, a, b, d}), {0, 2, 1, 3}), {kT, a * b, d});
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
    std::vector<int64_t> plan = cfg_.plan;
    if (plan.empty()) {
        for (int64_t i = 0; i < cfg_.num_layers; ++i) plan.push_back(i);
    }
    std::vector<int64_t> runs(static_cast<size_t>(cfg_.num_layers), 0);
    std::vector<Value> after;
    for (const int64_t i : plan) {
        const std::string name = "layers." + std::to_string(i);
        const int64_t pass = runs[static_cast<size_t>(i)]++;
        const std::string label = pass == 0 ? name : name + ".pass" + std::to_string(pass);
        const std::string L = prefix_ + name + ".";
        if (cfg_.mhc.streams > 0 && i >= cfg_.mhc.start && i < cfg_.mhc.end) {
            h = mhc_layer(g, L, label, i, h, in.pos, freq, attention);
        } else {
            h = layer(g, L, label, cfg_.layer_kinds[static_cast<size_t>(i)], h, in.pos, freq, attention);
        }
        after.push_back(h);
    }
    return after;
}

Value Decoder::layer(Graph& g, const std::string& L, const std::string& label, LayerKind kind, Value h, Value pos,
                     Value freq, Attention attention) const {
    Graph::Scope s(g, label);
    h = g.add(h, mixer(g, L, label, kind, norm(g, h, L + "input_layernorm.weight"), pos, freq, attention));
    return g.add(h, mlp(g, L, norm(g, h, L + "post_attention_layernorm.weight")));
}

Value Decoder::mixer(Graph& g, const std::string& L, const std::string& label, LayerKind kind, Value x, Value pos,
                     Value freq, Attention attention) const {
    return kind == LayerKind::Linear ? linear_attention(g, L, label, x)
                                     : full_attention(g, L, label, x, pos, freq, attention);
}

Value Decoder::mlp(Graph& g, const std::string& L, Value y) const {
    if (cfg_.num_experts > 0) return moe(g, L + "mlp.", y);
    return blocks::gated_mlp(g, y, proj(g, L + "mlp.gate_proj.weight"), proj(g, L + "mlp.up_proj.weight"),
                             proj(g, L + "mlp.down_proj.weight"), blocks::Act::Silu);
}

// MHCDecoderLayer: enter (copy into the streams), each branch through its router, leave (mean of the streams).
Value Decoder::mhc_layer(Graph& g, const std::string& L, const std::string& label, int64_t physical, Value h, Value pos,
                         Value freq, Attention attention) const {
    Graph::Scope s(g, label);
    const int64_t streams = cfg_.mhc.streams, hidden = cfg_.hidden_size;
    Value r = h;
    if (physical == cfg_.mhc.start) {
        r = g.concat(std::vector<Value>(static_cast<size_t>(streams), g.reshape(h, {kT, 1, hidden})), 1);
    }
    const LayerKind kind = cfg_.layer_kinds[static_cast<size_t>(physical)];
    r = mhc_branch(g, L + "mhc_attn.", r, [&](Value x) {
        return mixer(g, L, label, kind, norm(g, x, L + "input_layernorm.weight"), pos, freq, attention);
    });
    r = mhc_branch(g, L + "mhc_ffn.", r,
                   [&](Value x) { return mlp(g, L, norm(g, x, L + "post_attention_layernorm.weight")); });
    if (physical == cfg_.mhc.end - 1) return g.scale(g.reduce_sum(r, 1), 1.0f / static_cast<float>(streams));
    return r;
}

// MHCRouter around one branch of streams r [T, S, hidden]: from W rms(flatten(r)) + b the read weights (pre), the
// write weights (post) and the stream mix (graph.h, MhcRoute); the new streams are mix @ r + post (outer)
// branch(pre @ r).
template <typename Branch>
Value Decoder::mhc_branch(Graph& g, const std::string& router, Value r, Branch branch) const {
    const int64_t s = cfg_.mhc.streams, hidden = cfg_.hidden_size, width = 2 * s + s * s;
    const Value w = weight(g, router + "weight");
    const Value b = weight(g, router + "bias");
    XS_CHECK(g.type(w).shape == (Dims{width, s * hidden}) && g.type(b).shape == Dims{width}, "qwen3_5: ", router,
             "weight and bias have shapes ", dims_to_string(g.type(w).shape), " and ", dims_to_string(g.type(b).shape),
             ", expected [", width, ", ", s * hidden, "] and [", width, "]");
    const graph::MhcMaps maps =
        g.mhc_route(g.matmul(g.rms_norm(g.reshape(r, {kT, s * hidden}), 1e-6f), w), b, s, cfg_.mhc.iterations);
    Value x = g.reshape(g.stream_mix(g.reshape(maps.pre, {kT, 1, s}), r), {kT, hidden});
    if (cfg_.mhc.pre_scale != 1.0f) x = g.scale(x, cfg_.mhc.pre_scale);
    return g.stream_mix(maps.mix, r, maps.post, branch(x));
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

// Qwen3_5MoeSparseMoeBlock. The gate and up banks come fused (experts.gate_up_proj: an expert's gate rows, then its up
// rows), or apart (experts.gate_proj and experts.up_proj: a GGUF import's, each in the format its file stores).
Value Decoder::moe(Graph& g, const std::string& L, Value y) const {
    blocks::MoeSpec spec;
    spec.experts = cfg_.num_experts;
    spec.top_k = cfg_.experts_per_token;
    spec.router = blocks::Router::Softmax;
    spec.normalize = cfg_.norm_topk_prob;
    blocks::MoeParts parts;
    parts.router = weight(g, L + "gate.weight");
    const int64_t e = cfg_.num_experts, h = cfg_.hidden_size, inter = cfg_.moe_intermediate_size;
    const std::string E = L + "experts.";
    const auto bank = [&](const char* name, const Dims& shape) {
        const Value w = weight(g, E + name);
        XS_CHECK(g.type(w).shape == shape, "qwen3_5: ", E, name, " has shape ", dims_to_string(g.type(w).shape),
                 ", expected ", dims_to_string(shape));
        return w;
    };
    const bool fused = weights_.contains(E + "gate_up_proj");
    const bool apart = weights_.contains(E + "gate_proj") || weights_.contains(E + "up_proj");
    if (fused && apart) {
        fail("qwen3_5: ", L, "experts holds the fused bank (gate_up_proj) and the banks apart (gate_proj, up_proj): ",
             "a checkpoint has one layout or the other");
    }
    if (fused) {
        parts.gate_up = bank("gate_up_proj", {e, 2 * inter, h});
    } else if (apart) {
        parts.gate = bank("gate_proj", {e, inter, h});
        parts.up = bank("up_proj", {e, inter, h});
    } else {
        fail("qwen3_5: ", L, "experts has no gate and up banks: they come fused (experts.gate_up_proj), apart ",
             "(experts.gate_proj and experts.up_proj), or expert by expert (experts.<e>.gate_proj.weight)");
    }
    parts.down = bank("down_proj", {e, h, inter});
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
    Value v = g.reshape(g.slice(qkv, 1, 2 * key_width, conv_width), {kT, hv, dv});

    // beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), the per-token log decay.
    Value beta = g.sigmoid(g.matmul(x, weight(g, L + "in_proj_b.weight")));
    HostTensor neg_a = to_f32(L + "A_log", weights_.get(L + "A_log"));
    for (int64_t j = 0; j < neg_a.numel(); ++j)
        neg_a.data<float>()[j] = -std::exp(neg_a.data<float>()[j]);
    Value decay = g.mul(g.softplus(g.add(g.matmul(x, weight(g, L + "in_proj_a.weight")), weight(g, L + "dt_bias"))),
                        g.constant(std::move(neg_a)));

    // The recurrence pairs value head h with key head h / r: a tiled checkpoint's value heads go to that order, and
    // its outputs back to the order of z, the norm and out_proj. With one key head, or one value head per key head,
    // the two orders are the same.
    const int64_t r = hv / hk;
    const bool tiled = cfg_.value_head_order == ValueHeadOrder::Tiled && hk > 1 && r > 1;
    if (tiled) {
        v = swap_head_order(g, v, r, hk, dv);
        beta = swap_head_order(g, beta, r, hk, 0);
        decay = swap_head_order(g, decay, r, hk, 0);
    }
    const int slot = g.add_recurrent_slot(hv, dk, dv, label);
    Value o = g.gated_delta(slot, q, k, v, decay, beta, 1e-6f);
    if (tiled) o = swap_head_order(g, o, hk, r, dv);

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
