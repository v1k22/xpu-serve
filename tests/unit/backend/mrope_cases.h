#pragma once

// Multimodal RoPE cases (Rope with pair_axes), shared by the cpu-ref and device op tests: Qwen3-Omni's interleaved
// sections over (t, h, w), Qwen3-VL's 2-D vision rotary over (h, w), and plain RoPE as equal positions on every axis.

#include <cmath>
#include <string>
#include <vector>

#include "backend_test_util.h"
#include "graph/graph.h"
#include "reference_ops.h"

namespace xs::test::mrope {

struct Case {
    std::string name;
    int64_t heads = 0, dim = 0, axes_count = 0;
    std::vector<int> axes;       // [dim / 2]
    std::vector<float> inv_freq;  // [dim / 2]
    bool equal_positions = false;  // every axis the same position: plain RoPE
};

inline std::vector<float> freqs(int64_t n, double theta, int64_t period) {
    std::vector<float> f(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const double exponent = static_cast<double>(i % period) / static_cast<double>(period);
        f[static_cast<size_t>(i)] = static_cast<float>(std::pow(theta, -exponent));
    }
    return f;
}

inline std::vector<Case> cases() {
    std::vector<Case> out;
    // Qwen3-Omni's text (head 128, sections 24, 20, 20) at a small head: 32 pairs, sections 12, 10, 10.
    out.push_back({"interleaved t/h/w", 4, 64, 3, ref::interleaved_mrope_axes(32, 10, 10), freqs(32, 1e6, 32)});
    // The vision tower's: the first half of the frequencies read h, the second w, the same frequencies for each.
    std::vector<int> vision(16, 0);
    for (size_t i = 8; i < 16; ++i) vision[i] = 1;
    out.push_back({"vision h/w", 3, 32, 2, vision, freqs(16, 1e4, 8)});
    out.push_back({"equal positions are plain RoPE", 2, 64, 3, ref::interleaved_mrope_axes(32, 10, 10),
                   freqs(32, 1e6, 32), true});
    return out;
}

// positions [T, axes] for a case: random per axis, or the same on every axis.
inline HostTensor positions(TensorRng& rng, const Case& c, int64_t tokens) {
    HostTensor p = rng.ints(DType::I32, {tokens, c.axes_count}, 0, 300);
    if (c.equal_positions) {
        for (int64_t t = 0; t < tokens; ++t) {
            int32_t* row = p.data<int32_t>() + t * c.axes_count;
            for (int64_t a = 1; a < c.axes_count; ++a) row[a] = row[0];
        }
    }
    return p;
}

inline std::vector<std::vector<int64_t>> rows(const HostTensor& p) {
    std::vector<std::vector<int64_t>> out;
    for (int64_t t = 0; t < p.dim(0); ++t) {
        out.emplace_back();
        for (int64_t a = 0; a < p.dim(1); ++a) out.back().push_back(p.data<int32_t>()[t * p.dim(1) + a]);
    }
    return out;
}

inline graph::Graph graph_of(const Case& c) {
    graph::Graph g;
    const auto half = static_cast<int64_t>(c.axes.size());
    std::vector<int32_t> axes(c.axes.begin(), c.axes.end());
    const graph::Value freq = g.constant(make_tensor(DType::F32, {half}, c.inv_freq));
    const graph::Value table = g.constant(make_tensor(DType::I32, {half}, axes));
    g.output("out", g.rope(g.input("x", DType::F32, {graph::kT, c.heads, c.dim}),
                           g.input("pos", DType::I32, {graph::kT, c.axes_count}), freq, 0, table));
    return g;
}

}  // namespace xs::test::mrope
