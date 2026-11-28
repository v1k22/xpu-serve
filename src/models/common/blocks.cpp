#include "models/common/blocks.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <tuple>

#include "util/error.h"
#include "util/half.h"

namespace xs::blocks {

using graph::kT;

Value activation(Graph& g, Act act, Value x) {
    switch (act) {
        case Act::Silu: return g.silu(x);
        case Act::GeluTanh: return g.gelu_tanh(x);
        case Act::Gelu: return g.gelu(x);
    }
    fail("blocks: unknown activation");
}

Value gated_mlp(Graph& g, Value x, const Linear& gate, const Linear& up, const Linear& down, Act act) {
    Graph::Scope s(g, "mlp");
    const Value a = activation(g, act, gate(x));
    return down(g.mul(a, up(x)));
}

Shortlist shortlist_head(Graph& g, Value x, Value head, const quant::SignSketch& sketch, int64_t count) {
    Graph::Scope scope(g, "shortlist");
    const Value scores = g.sign_matmul(x, g.constant(sketch.signs), g.constant(sketch.scales));  // [1, V]
    const Value candidates = g.reshape(g.topk(scores, count).second, {count});
    return {g.matmul(x, g.gather(head, candidates)), candidates};
}

Shortlist shortlist_head_rows(Graph& g, Value x, Value head, const quant::SignSketch& sketch, int64_t count) {
    Graph::Scope scope(g, "shortlist");
    const Value scores = g.sign_matmul(x, g.constant(sketch.signs), g.constant(sketch.scales));  // [T, V]
    const Value candidates = g.topk(scores, count).second;                                       // [T, count]
    const int64_t hidden = g.type(x).shape.back();
    const Value rows = g.gather(head, candidates);                                              // [T, count, hidden]
    const Value logits = g.reduce_sum(g.mul(g.reshape(x, {graph::kT, 1, hidden}), rows), 2);   // [T, count]
    return {logits, candidates};
}

Value argmax_rows(Graph& g, Value logits) {
    return g.reshape(g.topk(logits, 1).second, {g.type(logits).shape[0]});
}

Value moe(Graph& g, Value x, const MoeSpec& s, const MoeParts& p) {
    Graph::Scope scope(g, "moe");
    XS_CHECK(s.top_k >= 1 && s.top_k <= s.experts, "moe: top_k ", s.top_k, " of ", s.experts, " experts");
    const Value logits = g.matmul(p.router_x.valid() ? p.router_x : x, p.router);  // [T, E]
    const Value probs = s.router == Router::Softmax ? g.softmax(logits) : g.sigmoid(logits);
    Value weights, idx;  // [T, k]
    if (p.select_bias.valid()) {
        std::tie(weights, idx) = g.topk(g.add(probs, p.select_bias), s.top_k);
        const Value bias = g.gather(g.reshape(p.select_bias, {s.experts, 1}), idx);  // [T, k, 1]
        weights = g.add(weights, g.scale(g.reshape(bias, {kT, s.top_k}), -1.0f));
    } else {
        std::tie(weights, idx) = g.topk(probs, s.top_k);
    }
    if (s.normalize) weights = g.div(weights, g.reshape(g.reduce_sum(weights, 1), {kT, 1}));
    if (s.routed_scale != 1.0f) weights = g.scale(weights, s.routed_scale);
    if (p.expert_scale.valid()) {
        const Value per = g.gather(g.reshape(p.expert_scale, {s.experts, 1}), idx);  // [T, k, 1]
        weights = g.mul(weights, g.reshape(per, {kT, s.top_k}));
    }
    Value gate, up;  // [T, k, I]
    if (p.gate_up.valid()) {
        const int64_t inter = g.type(p.gate_up).shape[1] / 2;
        Value both = g.expert_matmul(x, p.gate_up, idx);  // [T, k, 2 I]
        if (p.gate_up_scale.valid()) {
            both = g.mul(both, g.gather(g.reshape(p.gate_up_scale, {s.experts, 1}), idx));  // by [T, k, 1]
        }
        if (p.interleaved) {
            const Value pairs = g.reshape(both, {kT, s.top_k, inter, 2});
            gate = g.reshape(g.slice(pairs, 3, 0, 1), {kT, s.top_k, inter});
            up = g.reshape(g.slice(pairs, 3, 1, 2), {kT, s.top_k, inter});
        } else {
            gate = g.slice(both, 2, 0, inter);
            up = g.slice(both, 2, inter, 2 * inter);
        }
    } else {
        gate = g.expert_matmul(x, p.gate, idx);
        up = g.expert_matmul(x, p.up, idx);
    }
    const Value h = p.hidden ? p.hidden(gate, up) : g.mul(activation(g, s.act, gate), up);
    const Value y = g.expert_matmul(h, p.down, idx);  // [T, k, hidden]
    Value out = g.reduce_sum(g.mul(y, g.reshape(weights, {kT, s.top_k, 1})), 1);
    if (p.shared_gate) {
        Value shared = gated_mlp(g, x, p.shared_gate, p.shared_up, p.shared_down, s.act);
        if (p.shared_expert_gate.valid()) shared = g.mul(shared, g.sigmoid(g.matmul(x, p.shared_expert_gate)));
        out = g.add(out, shared);
    }
    return out;
}

Value splice_media(Graph& g, Value embedded, int64_t width) {
    const Value media = g.input("media", DType::F32, {graph::kT, width});
    const Value keep = g.input("media_keep", DType::F32, {graph::kT, 1});
    return g.add(g.mul(embedded, keep), media);
}

void media_inputs(std::span<const float* const> rows, int64_t n, int64_t width, TensorMap& inputs) {
    XS_CHECK(rows.empty() || static_cast<int64_t>(rows.size()) == n, rows.size(), " media rows for ", n, " tokens");
    HostTensor media(DType::F32, {n, width});
    HostTensor keep(DType::F32, {n, 1});
    for (int64_t i = 0; i < n; ++i) {
        const float* row = rows.empty() ? nullptr : rows[static_cast<size_t>(i)];
        keep.data<float>()[i] = row == nullptr ? 1.0f : 0.0f;
        if (row != nullptr) std::copy(row, row + width, media.data<float>() + i * width);
    }
    inputs["media"] = std::move(media);
    inputs["media_keep"] = std::move(keep);
}

Value rms_norm_one_plus(Graph& g, Value x, const WeightStore& weights, const std::string& name, float eps) {
    const TensorView& w = weights.get(name);
    XS_CHECK(is_float(w.dtype) && w.shape.size() == 1, "norm weight '", name, "' must be a float vector, got ",
             dtype_name(w.dtype), dims_to_string(w.shape));
    HostTensor scale(DType::F32, w.shape);
    const size_t n = static_cast<size_t>(w.numel());
    float* s = scale.data<float>();
    if (w.dtype == DType::BF16) {
        convert_bf16_to_f32(static_cast<const uint16_t*>(w.data), s, n);
    } else if (w.dtype == DType::F16) {
        convert_f16_to_f32(static_cast<const uint16_t*>(w.data), s, n);
    } else {
        std::copy(static_cast<const float*>(w.data), static_cast<const float*>(w.data) + n, s);
    }
    for (size_t i = 0; i < n; ++i)
        s[i] += 1.0f;
    return g.rms_norm(x, g.constant(std::move(scale)), eps);
}

Value gated_rms_norm(Graph& g, Value x, Value w, Value gate, float eps) {
    return g.mul(g.rms_norm(x, w, eps), g.silu(gate));
}

HostTensor rope_inv_freq(double theta, int64_t dim, int64_t pairs, int64_t length) {
    XS_CHECK(dim > 0 && pairs >= 0 && pairs <= length, "rope_inv_freq: ", pairs, " rotated pairs of ", length,
             " for dim ", dim);
    HostTensor t(DType::F32, {length});
    float* f = t.data<float>();
    for (int64_t i = 0; i < length; ++i) {
        f[i] = i < pairs ? static_cast<float>(1.0 / std::pow(theta, static_cast<double>(2 * i) / static_cast<double>(dim)))
                         : 0.0f;
    }
    return t;
}

HostTensor llama3_rope_inv_freq(double theta, int64_t dim, const Llama3RopeScaling& s) {
    XS_CHECK(s.factor > 0 && s.high_freq_factor > s.low_freq_factor && s.low_freq_factor > 0 &&
                 s.original_max_position > 0,
             "llama3 RoPE scaling: factor ", s.factor, ", low_freq_factor ", s.low_freq_factor, ", high_freq_factor ",
             s.high_freq_factor, ", original_max_position_embeddings ", s.original_max_position);
    HostTensor t = rope_inv_freq(theta, dim, dim / 2, dim / 2);
    const double low_freq_wavelen = s.original_max_position / s.low_freq_factor;
    const double high_freq_wavelen = s.original_max_position / s.high_freq_factor;
    float* f = t.data<float>();
    for (int64_t i = 0; i < dim / 2; ++i) {
        const double inv_freq = f[i];
        const double wavelen = 2.0 * std::numbers::pi / inv_freq;
        if (wavelen > low_freq_wavelen) {
            f[i] = static_cast<float>(inv_freq / s.factor);
        } else if (wavelen >= high_freq_wavelen) {
            const double smooth =
                (s.original_max_position / wavelen - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor);
            f[i] = static_cast<float>((1.0 - smooth) * inv_freq / s.factor + smooth * inv_freq);
        }
    }
    return t;
}

std::vector<int32_t> interleaved_mrope_axes(std::span<const int32_t> section, int64_t pairs, const std::string& where) {
    XS_CHECK(section.size() == 3 && section[0] + section[1] + section[2] == pairs, where,
             ": mrope_section must be three counts that add up to the ", pairs, " rotated frequencies of a head");
    std::vector<int32_t> axes(static_cast<size_t>(pairs), 0);
    for (int64_t i = 0; i < pairs; ++i) {
        if (i % 3 == 1 && i < 3 * section[1]) axes[static_cast<size_t>(i)] = 1;
        if (i % 3 == 2 && i < 3 * section[2]) axes[static_cast<size_t>(i)] = 2;
    }
    return axes;
}

AttentionOut attention(Graph& g, const AttentionSpec& s, const AttentionParts& p, Value x, Value pos, Value inv_freq,
                       const std::pair<Value, Value>* shared_kv) {
    const int64_t hd = s.head_dim;
    const auto heads = [&](Value v, int64_t n) { return g.reshape(v, {kT, n, hd}); };
    const auto position = [&](Value v) {
        return inv_freq.valid() ? g.rope(v, pos, inv_freq, s.rotary_dim, s.rope_axes) : v;
    };

    Value q = p.q(x);
    Value gate;
    if (s.output_gate) {
        const Value both = g.reshape(q, {kT, s.heads, 2 * hd});
        q = g.slice(both, 2, 0, hd);
        gate = g.reshape(g.slice(both, 2, hd, 2 * hd), {kT, s.heads * hd});
    } else {
        q = heads(q, s.heads);
    }
    if (p.q_norm) q = p.q_norm(q);
    q = position(q);

    AttentionOut r;
    if (shared_kv != nullptr) {
        XS_CHECK(shared_kv->first.valid(), "attention: the shared KV is empty");
        r.kv = *shared_kv;
    } else {
        const Value k_proj = heads(p.k(x), s.kv_heads);
        Value k = k_proj;
        if (p.k_norm) k = p.k_norm(k);
        k = position(k);
        Value v = p.v ? heads(p.v(x), s.kv_heads) : k_proj;
        if (p.v_norm) v = p.v_norm(v);
        if (s.kv_cache) {
            r.kv = g.kv_cache(g.add_kv_slot(s.kv_heads, hd, s.kv_window, s.kv_label), k, v);
        } else {
            r.kv = {k, v};
        }
    }
    Value o = s.groups.valid() ? g.attention_in_groups(q, r.kv.first, r.kv.second, s.groups, s.attrs)
                               : g.attention(q, r.kv.first, r.kv.second, s.attrs);
    if (p.head_gate) o = g.mul(o, g.reshape(g.sigmoid(p.head_gate(x)), {kT, s.heads, 1}));
    o = g.reshape(o, {kT, s.heads * hd});
    if (gate.valid()) o = g.mul(o, g.sigmoid(gate));
    r.out = p.o(o);
    return r;
}

}  // namespace xs::blocks
