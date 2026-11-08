#include "models/common/blocks.h"

#include <cmath>

#include "util/error.h"
#include "util/half.h"

namespace xs::blocks {

using graph::kT;

Value activation(Graph& g, Act act, Value x) {
    return act == Act::Silu ? g.silu(x) : g.gelu_tanh(x);
}

Value gated_mlp(Graph& g, Value x, const Linear& gate, const Linear& up, const Linear& down, Act act) {
    Graph::Scope s(g, "mlp");
    const Value a = activation(g, act, gate(x));
    return down(g.mul(a, up(x)));
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

AttentionOut attention(Graph& g, const AttentionSpec& s, const AttentionParts& p, Value x, Value pos, Value inv_freq,
                       const std::pair<Value, Value>* shared_kv) {
    const int64_t hd = s.head_dim;
    const auto heads = [&](Value v, int64_t n) { return g.reshape(v, {kT, n, hd}); };
    const auto position = [&](Value v) { return inv_freq.valid() ? g.rope(v, pos, inv_freq, s.rotary_dim) : v; };

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
        Value k = heads(p.k(x), s.kv_heads);
        if (p.k_norm) k = p.k_norm(k);
        k = position(k);
        Value v = heads(p.v(x), s.kv_heads);
        if (p.v_norm) v = p.v_norm(v);
        if (s.kv_cache) {
            r.kv = g.kv_cache(g.add_kv_slot(s.kv_heads, hd, s.kv_window, s.kv_label), k, v);
        } else {
            r.kv = {k, v};
        }
    }
    Value o = g.reshape(g.attention(q, r.kv.first, r.kv.second, s.attrs), {kT, s.heads * hd});
    if (gate.valid()) o = g.mul(o, g.sigmoid(gate));
    r.out = p.o(o);
    return r;
}

}  // namespace xs::blocks
