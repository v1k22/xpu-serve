#pragma once

// Independent fp64 reference implementations of the graph ops (graph/graph.h is the spec), written for clarity and
// in the formulation of the Hugging Face models where one exists. Never call a backend from here.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

#include "backend_test_util.h"

namespace xs::test::ref {

inline Tensor64 matmul(const Tensor64& x, const Tensor64& w) {
    const int64_t n = w.shape[0];
    const int64_t k = w.shape[1];
    Dims shape = x.shape;
    shape.back() = n;
    Tensor64 out(shape);
    for (int64_t m = 0; m < x.numel() / k; ++m) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0;
            for (int64_t i = 0; i < k; ++i)
                s += x[m * k + i] * w[j * k + i];
            out[m * n + j] = s;
        }
    }
    return out;
}

// numpy broadcasting by explicit multi-index: each operand index is the output index with size-1 dims pinned to 0.
template <typename Fn>
Tensor64 broadcast(const Tensor64& a, const Tensor64& b, Fn fn) {
    const size_t rank = std::max(a.shape.size(), b.shape.size());
    const auto padded = [rank](const Dims& s) {
        Dims p(rank - s.size(), 1);
        p.insert(p.end(), s.begin(), s.end());
        return p;
    };
    const Dims pa = padded(a.shape);
    const Dims pb = padded(b.shape);
    Dims shape(rank);
    for (size_t d = 0; d < rank; ++d)
        shape[d] = std::max(pa[d], pb[d]);
    Tensor64 out(shape);
    for (int64_t flat = 0; flat < out.numel(); ++flat) {
        int64_t rem = flat;
        int64_t ia = 0, ib = 0, sa = 1, sb = 1;
        for (size_t d = rank; d-- > 0;) {
            const int64_t idx = rem % shape[d];
            rem /= shape[d];
            ia += (pa[d] == 1 ? 0 : idx) * sa;
            ib += (pb[d] == 1 ? 0 : idx) * sb;
            sa *= pa[d];
            sb *= pb[d];
        }
        out[flat] = fn(a[ia], b[ib]);
    }
    return out;
}

template <typename Fn>
Tensor64 map(const Tensor64& x, Fn fn) {
    Tensor64 out(x.shape);
    for (int64_t i = 0; i < x.numel(); ++i)
        out[i] = fn(x[i]);
    return out;
}

inline double gelu_tanh(double x) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / std::numbers::pi) * (x + 0.044715 * x * x * x)));
}

inline Tensor64 rms_norm(const Tensor64& x, const Tensor64* weight, double eps) {
    const int64_t d = x.shape.back();
    Tensor64 out(x.shape);
    for (int64_t r = 0; r < x.numel() / d; ++r) {
        double ms = 0;
        for (int64_t i = 0; i < d; ++i)
            ms += x[r * d + i] * x[r * d + i] / static_cast<double>(d);
        for (int64_t i = 0; i < d; ++i) {
            out[r * d + i] = x[r * d + i] / std::sqrt(ms + eps) * (weight ? (*weight)[i] : 1.0);
        }
    }
    return out;
}

inline Tensor64 gather(const Tensor64& table, const std::vector<int64_t>& ids) {
    const int64_t width = table.shape[1];
    Tensor64 out({static_cast<int64_t>(ids.size()), width});
    for (size_t r = 0; r < ids.size(); ++r) {
        for (int64_t c = 0; c < width; ++c)
            out[static_cast<int64_t>(r) * width + c] = table[ids[r] * width + c];
    }
    return out;
}

// x[..., b:e, ...] on `axis` with concrete b, e.
inline Tensor64 slice(const Tensor64& x, size_t axis, int64_t b, int64_t e) {
    Dims shape = x.shape;
    shape[axis] = e - b;
    Tensor64 out(shape);
    for (int64_t flat = 0; flat < out.numel(); ++flat) {
        int64_t rem = flat, src = 0, stride = 1;
        for (size_t d = shape.size(); d-- > 0;) {
            int64_t idx = rem % shape[d];
            rem /= shape[d];
            if (d == axis) idx += b;
            src += idx * stride;
            stride *= x.shape[d];
        }
        out[flat] = x[src];
    }
    return out;
}

enum class RopeLayout { RotateHalf, Interleaved /* planted-bug control: GPT-J pairs (2i, 2i+1) */ };

// HF formulation: out = x * cat(cos, cos) + rotate_half(x) * cat(sin, sin), rotate_half(x) = cat(-x2, x1).
inline Tensor64 rope(const Tensor64& x, const std::vector<int64_t>& positions, const std::vector<double>& inv_freq,
                     RopeLayout layout = RopeLayout::RotateHalf) {
    const int64_t heads = x.shape[1];
    const int64_t dim = x.shape[2];
    const int64_t half = dim / 2;
    Tensor64 out(x.shape);
    for (size_t t = 0; t < positions.size(); ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            const int64_t base = (static_cast<int64_t>(t) * heads + h) * dim;
            for (int64_t d = 0; d < dim; ++d) {
                if (layout == RopeLayout::RotateHalf) {
                    const double angle = static_cast<double>(positions[t]) * inv_freq[static_cast<size_t>(d % half)];
                    const double rotated = d < half ? -x[base + d + half] : x[base + d - half];
                    out[base + d] = x[base + d] * std::cos(angle) + rotated * std::sin(angle);
                } else {
                    const double angle = static_cast<double>(positions[t]) * inv_freq[static_cast<size_t>(d / 2)];
                    const double rotated = d % 2 == 0 ? -x[base + d + 1] : x[base + d - 1];
                    out[base + d] = x[base + d] * std::cos(angle) + rotated * std::sin(angle);
                }
            }
        }
    }
    return out;
}

struct AttentionSpec {
    double scale = 1.0;
    int64_t sliding_window = 0;
    double softcap = 0.0;
    int64_t window_slack = 0;  // planted-bug control: > 0 lets queries see that many keys beyond the window
};

// Builds the full [T, L] mask and score matrix per head, then a max-subtracted softmax.
inline Tensor64 attention(const Tensor64& q, const Tensor64& k, const Tensor64& v, const AttentionSpec& spec) {
    const int64_t t_len = q.shape[0], hq = q.shape[1], dim = q.shape[2];
    const int64_t l_len = k.shape[0], hkv = k.shape[1];
    Tensor64 out(q.shape);
    for (int64_t h = 0; h < hq; ++h) {
        const int64_t kvh = h / (hq / hkv);
        for (int64_t i = 0; i < t_len; ++i) {
            const int64_t p = l_len - t_len + i;
            std::vector<double> scores(static_cast<size_t>(l_len), -std::numeric_limits<double>::infinity());
            for (int64_t j = 0; j < l_len; ++j) {
                const bool visible =
                    j <= p && (spec.sliding_window == 0 || p - j < spec.sliding_window + spec.window_slack);
                if (!visible) continue;
                double s = 0;
                for (int64_t d = 0; d < dim; ++d)
                    s += q[(i * hq + h) * dim + d] * k[(j * hkv + kvh) * dim + d];
                s *= spec.scale;
                if (spec.softcap > 0) s = spec.softcap * std::tanh(s / spec.softcap);
                scores[static_cast<size_t>(j)] = s;
            }
            const double mx = *std::max_element(scores.begin(), scores.end());
            double sum = 0;
            for (double& s : scores)
                sum += (s = std::exp(s - mx));
            for (int64_t d = 0; d < dim; ++d) {
                double acc = 0;
                for (int64_t j = 0; j < l_len; ++j)
                    acc += scores[static_cast<size_t>(j)] / sum * v[(j * hkv + kvh) * dim + d];
                out[(i * hq + h) * dim + d] = acc;
            }
        }
    }
    return out;
}

}  // namespace xs::test::ref
