#pragma once

// A small Gemma-shaped decoder (embed, per layer: rms_norm, q/k/v, rope, KV cache, attention, o_proj, gelu MLP,
// residuals; tied logits) with random bf16 weights, for session-level and cross-backend tests and timing.

#include <cmath>
#include <string>
#include <vector>

#include "backend_test_util.h"
#include "graph/graph.h"

namespace xs::test::ov {

struct TinyModelSpec {
    int64_t vocab = 64;
    int64_t hidden = 32;
    int64_t heads = 4;
    int64_t kv_heads = 2;
    int64_t head_dim = 16;
    int64_t ffn = 64;
    int64_t layers = 2;
    int64_t sliding_window = 4;  // used by every layer except each `global_every`-th
    int64_t global_every = 2;
    float attn_softcap = 0.0f;
    bool logits = true;  // false: output the final hidden state instead of logits
};

inline void add_tiny_weights(const TinyModelSpec& s, MemoryWeights& w, TensorRng& rng) {
    const auto scaled = [&](Dims shape, int64_t fan_in) {
        return rng.uniform(DType::BF16, std::move(shape), 1.0f / std::sqrt(static_cast<float>(fan_in)));
    };
    w.add("embed", rng.uniform(DType::BF16, {s.vocab, s.hidden}));
    w.add("final_norm", rng.uniform(DType::BF16, {s.hidden}, 2.0f));
    for (int64_t l = 0; l < s.layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        w.add(p + "attn_norm", rng.uniform(DType::BF16, {s.hidden}, 2.0f));
        w.add(p + "mlp_norm", rng.uniform(DType::BF16, {s.hidden}, 2.0f));
        w.add(p + "q", scaled({s.heads * s.head_dim, s.hidden}, s.hidden));
        w.add(p + "k", scaled({s.kv_heads * s.head_dim, s.hidden}, s.hidden));
        w.add(p + "v", scaled({s.kv_heads * s.head_dim, s.hidden}, s.hidden));
        w.add(p + "o", scaled({s.hidden, s.heads * s.head_dim}, s.heads * s.head_dim));
        w.add(p + "up", scaled({s.ffn, s.hidden}, s.hidden));
        w.add(p + "down", scaled({s.hidden, s.ffn}, s.ffn));
    }
}

// Inputs: "ids" [T] i32, "pos" [T] i32. Outputs: "logits" [T, vocab] (or "hidden" [T, hidden]) and "last" (the
// last row).
inline graph::Graph tiny_model(const TinyModelSpec& s, const WeightStore& w) {
    using graph::kT;
    using graph::Value;
    graph::Graph g;
    const Value ids = g.input("ids", DType::I32, {kT});
    const Value pos = g.input("pos", DType::I32, {kT});
    std::vector<float> inv_freq(static_cast<size_t>(s.head_dim / 2));
    for (size_t i = 0; i < inv_freq.size(); ++i) {
        inv_freq[i] = std::pow(10000.0f, -2.0f * static_cast<float>(i) / static_cast<float>(s.head_dim));
    }
    const Value freq = g.constant(HostTensor::from_vector(DType::F32, {s.head_dim / 2}, inv_freq));
    Value x = g.gather(g.weight(w, "embed"), ids);
    for (int64_t l = 0; l < s.layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        graph::Graph::Scope scope(g, "layers." + std::to_string(l));
        const bool global = s.global_every > 0 && (l + 1) % s.global_every == 0;
        const int64_t window = global ? 0 : s.sliding_window;
        const int slot = g.add_kv_slot(s.kv_heads, s.head_dim, window, p + "kv");
        const Value h = g.rms_norm(x, g.weight(w, p + "attn_norm"), 1e-6f);
        const auto heads = [&](const std::string& name, int64_t n) {
            return g.reshape(g.matmul(h, g.weight(w, p + name)), {kT, n, s.head_dim});
        };
        const Value q = g.rope(heads("q", s.heads), pos, freq);
        const Value k = g.rope(heads("k", s.kv_heads), pos, freq);
        const auto [k_all, v_all] = g.kv_cache(slot, k, heads("v", s.kv_heads));
        const float scale = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
        const Value a = g.attention(q, k_all, v_all, graph::AttentionAttrs{scale, window, s.attn_softcap});
        x = g.add(x, g.matmul(g.reshape(a, {kT, s.heads * s.head_dim}), g.weight(w, p + "o")));
        const Value m = g.rms_norm(x, g.weight(w, p + "mlp_norm"), 1e-6f);
        x = g.add(x, g.matmul(g.gelu_tanh(g.matmul(m, g.weight(w, p + "up"))), g.weight(w, p + "down")));
    }
    const Value final_hidden = g.rms_norm(x, g.weight(w, "final_norm"), 1e-6f);
    const Value out = s.logits ? g.matmul(final_hidden, g.weight(w, "embed")) : final_hidden;
    g.output(s.logits ? "logits" : "hidden", out);
    g.output("last", g.slice(out, 0, -1));
    return g;
}

// ids and positions for tokens [begin, end) of a fixed pseudo-random prompt.
inline TensorMap tiny_inputs(const TinyModelSpec& s, int32_t begin, int32_t end) {
    std::vector<int32_t> ids;
    for (int32_t i = begin; i < end; ++i)
        ids.push_back(static_cast<int32_t>((i * 37 + 11) % s.vocab));
    return {{"ids", HostTensor::from_vector(DType::I32, {end - begin}, ids)}, {"pos", arange_i32(begin, end)}};
}

}  // namespace xs::test::ov
