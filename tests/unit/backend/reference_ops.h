#pragma once

// Independent fp64 reference implementations of the graph ops (graph/graph.h is the spec), written for clarity and
// in the formulation of the Hugging Face models where one exists. Never call a backend from here.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <utility>
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

inline double gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::numbers::sqrt2));
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

// out.shape[i] = x.shape[perm[i]].
inline Tensor64 transpose(const Tensor64& x, const std::vector<int>& perm) {
    const size_t rank = x.shape.size();
    Dims shape(rank);
    for (size_t i = 0; i < rank; ++i)
        shape[i] = x.shape[static_cast<size_t>(perm[i])];
    Tensor64 out(shape);
    std::vector<int64_t> in_index(rank);
    for (int64_t flat = 0; flat < out.numel(); ++flat) {
        int64_t rem = flat;
        for (size_t i = rank; i-- > 0;) {
            in_index[static_cast<size_t>(perm[i])] = rem % shape[i];
            rem /= shape[i];
        }
        int64_t src = 0;
        for (size_t d = 0; d < rank; ++d)
            src = src * x.shape[d] + in_index[d];
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

// Axial RoPE as transformers' apply_multidimensional_rope: split the head dim into S equal parts, apply the
// rotate-half rope to part s with positions[t][s], concatenate.
inline Tensor64 axial_rope(const Tensor64& x, const std::vector<std::vector<int64_t>>& positions,
                           const std::vector<double>& inv_freq) {
    const int64_t tokens = x.shape[0], heads = x.shape[1], dim = x.shape[2];
    const int64_t sections = static_cast<int64_t>(positions[0].size());
    const int64_t width = dim / sections;
    Tensor64 out(x.shape);
    for (int64_t s = 0; s < sections; ++s) {
        const Tensor64 part = slice(x, 2, s * width, (s + 1) * width);
        std::vector<int64_t> pos;
        for (const auto& p : positions)
            pos.push_back(p[static_cast<size_t>(s)]);
        const Tensor64 rotated = rope(part, pos, inv_freq);
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t h = 0; h < heads; ++h) {
                for (int64_t c = 0; c < width; ++c)
                    out[(t * heads + h) * dim + s * width + c] = rotated[(t * heads + h) * width + c];
            }
        }
    }
    return out;
}

// Partial rotary (Qwen): rotate-half rope over the first `rotary_dim` channels, the rest unchanged.
inline Tensor64 partial_rope(const Tensor64& x, const std::vector<int64_t>& positions,
                             const std::vector<double>& inv_freq, int64_t rotary_dim) {
    const int64_t tokens = x.shape[0], heads = x.shape[1], dim = x.shape[2];
    const Tensor64 rotated = rope(slice(x, 2, 0, rotary_dim), positions, inv_freq);
    Tensor64 out = x;
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t c = 0; c < rotary_dim; ++c)
                out[(t * heads + h) * dim + c] = rotated[(t * heads + h) * rotary_dim + c];
        }
    }
    return out;
}

// Depthwise causal conv over a whole sequence x [N, C] with w [C, K] (zeros before the first input), as
// torch.nn.functional.conv1d(padding=K-1)[..., :N].
inline Tensor64 causal_conv(const Tensor64& x, const Tensor64& w) {
    const int64_t n = x.shape[0], c = x.shape[1], k = w.shape[1];
    Tensor64 out(x.shape);
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t ch = 0; ch < c; ++ch) {
            double acc = 0;
            for (int64_t j = 0; j < k; ++j) {
                const int64_t src = t - (k - 1) + j;
                if (src >= 0) acc += w[ch * k + j] * x[src * c + ch];
            }
            out[t * c + ch] = acc;
        }
    }
    return out;
}

// transformers' torch_recurrent_gated_delta_rule over a whole sequence: q, k [N, Hk, Dk], v [N, Hv, Dv],
// g, beta [N, Hv]; key heads repeat_interleave'd to the value heads. `group_of` maps a value head to its key head
// (planted-bug controls pass a wrong one).
inline Tensor64 gated_delta(const Tensor64& q, const Tensor64& k, const Tensor64& v, const Tensor64& g,
                            const Tensor64& beta, double l2norm_eps, int64_t (*group_of)(int64_t h, int64_t hk, int64_t hv) = nullptr) {
    const int64_t n = q.shape[0], hk = q.shape[1], dk = q.shape[2], hv = v.shape[1], dv = v.shape[2];
    Tensor64 out(v.shape);
    for (int64_t h = 0; h < hv; ++h) {
        const int64_t kh = group_of ? group_of(h, hk, hv) : h / (hv / hk);
        std::vector<double> S(static_cast<size_t>(dk * dv), 0.0);
        for (int64_t t = 0; t < n; ++t) {
            std::vector<double> qt(static_cast<size_t>(dk)), kt(static_cast<size_t>(dk));
            double sq = 0, sk = 0;
            for (int64_t i = 0; i < dk; ++i) {
                qt[static_cast<size_t>(i)] = q[(t * hk + kh) * dk + i];
                kt[static_cast<size_t>(i)] = k[(t * hk + kh) * dk + i];
                sq += qt[static_cast<size_t>(i)] * qt[static_cast<size_t>(i)];
                sk += kt[static_cast<size_t>(i)] * kt[static_cast<size_t>(i)];
            }
            for (int64_t i = 0; i < dk; ++i) {
                if (l2norm_eps > 0) {
                    qt[static_cast<size_t>(i)] /= std::sqrt(sq + l2norm_eps);
                    kt[static_cast<size_t>(i)] /= std::sqrt(sk + l2norm_eps);
                }
                qt[static_cast<size_t>(i)] /= std::sqrt(static_cast<double>(dk));
            }
            const double decay = std::exp(g[t * hv + h]);
            for (double& s : S) s *= decay;
            for (int64_t j = 0; j < dv; ++j) {
                double mem = 0;
                for (int64_t i = 0; i < dk; ++i) mem += S[static_cast<size_t>(i * dv + j)] * kt[static_cast<size_t>(i)];
                const double delta = (v[(t * hv + h) * dv + j] - mem) * beta[t * hv + h];
                for (int64_t i = 0; i < dk; ++i) S[static_cast<size_t>(i * dv + j)] += kt[static_cast<size_t>(i)] * delta;
            }
            for (int64_t j = 0; j < dv; ++j) {
                double o = 0;
                for (int64_t i = 0; i < dk; ++i) o += S[static_cast<size_t>(i * dv + j)] * qt[static_cast<size_t>(i)];
                out[(t * hv + h) * dv + j] = o;
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
    bool causal = true;        // false: keys after a query are visible too (every key, without a window)
    std::vector<double> key_bias;  // empty, or one per key: added to every query's score of that key
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
                const bool visible = (!spec.causal || j <= p) &&
                                     (spec.sliding_window == 0 || p - j < spec.sliding_window + spec.window_slack);
                if (!visible) continue;
                double s = 0;
                for (int64_t d = 0; d < dim; ++d)
                    s += q[(i * hq + h) * dim + d] * k[(j * hkv + kvh) * dim + d];
                s *= spec.scale;
                if (spec.softcap > 0) s = spec.softcap * std::tanh(s / spec.softcap);
                if (!spec.key_bias.empty()) s += spec.key_bias[static_cast<size_t>(j)];
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

inline Tensor64 layer_norm(const Tensor64& x, const Tensor64& w, const Tensor64& b, double eps) {
    const int64_t width = x.shape.back();
    Tensor64 out(x.shape);
    for (int64_t r = 0; r < x.numel() / width; ++r) {
        double mean = 0, var = 0;
        for (int64_t i = 0; i < width; ++i)
            mean += x[r * width + i];
        mean /= static_cast<double>(width);
        for (int64_t i = 0; i < width; ++i)
            var += (x[r * width + i] - mean) * (x[r * width + i] - mean);
        var /= static_cast<double>(width);
        for (int64_t i = 0; i < width; ++i)
            out[r * width + i] = (x[r * width + i] - mean) / std::sqrt(var + eps) * w[i] + b[i];
    }
    return out;
}

inline Tensor64 softmax(const Tensor64& x) {
    const int64_t width = x.shape.back();
    Tensor64 out(x.shape);
    for (int64_t r = 0; r < x.numel() / width; ++r) {
        double peak = -std::numeric_limits<double>::infinity();
        for (int64_t i = 0; i < width; ++i)
            peak = std::max(peak, x[r * width + i]);
        double sum = 0;
        for (int64_t i = 0; i < width; ++i)
            sum += std::exp(x[r * width + i] - peak);
        for (int64_t i = 0; i < width; ++i)
            out[r * width + i] = std::exp(x[r * width + i] - peak) / sum;
    }
    return out;
}

// The k largest per row of the last axis, largest first, ties to the lower index: (values, indices).
inline std::pair<Tensor64, std::vector<int64_t>> topk(const Tensor64& x, int64_t k) {
    const int64_t width = x.shape.back();
    Dims shape = x.shape;
    shape.back() = k;
    Tensor64 values(shape);
    std::vector<int64_t> indices;
    for (int64_t r = 0; r < x.numel() / width; ++r) {
        std::vector<int64_t> order(static_cast<size_t>(width));
        for (int64_t i = 0; i < width; ++i)
            order[static_cast<size_t>(i)] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int64_t a, int64_t b) { return x[r * width + a] > x[r * width + b]; });
        for (int64_t j = 0; j < k; ++j) {
            values[r * k + j] = x[r * width + order[static_cast<size_t>(j)]];
            indices.push_back(order[static_cast<size_t>(j)]);
        }
    }
    return {values, indices};
}

inline Tensor64 reduce_sum(const Tensor64& x, size_t axis) {
    int64_t outer = 1;
    for (size_t i = 0; i < axis; ++i)
        outer *= x.shape[i];
    const int64_t dim = x.shape[axis];
    const int64_t inner = x.numel() / (outer * dim);
    Dims shape = x.shape;
    shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(axis));
    Tensor64 out(shape);
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t d = 0; d < dim; ++d)
            for (int64_t i = 0; i < inner; ++i)
                out[o * inner + i] += x[(o * dim + d) * inner + i];
    return out;
}

// x [T, K] or [T, S, K], w [E, N, K], idx T * S entries -> [T, S, N].
inline Tensor64 expert_matmul(const Tensor64& x, const Tensor64& w, const std::vector<int64_t>& idx, int64_t slots) {
    const int64_t tokens = x.shape[0];
    const int64_t n = w.shape[1];
    const int64_t k = w.shape[2];
    const bool per_slot = x.shape.size() == 3;
    Tensor64 out({tokens, slots, n});
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t s = 0; s < slots; ++s) {
            const int64_t e = idx[static_cast<size_t>(t * slots + s)];
            const int64_t row = per_slot ? t * slots + s : t;
            for (int64_t c = 0; c < n; ++c) {
                double acc = 0;
                for (int64_t i = 0; i < k; ++i)
                    acc += x[row * k + i] * w[(e * n + c) * k + i];
                out[(t * slots + s) * n + c] = acc;
            }
        }
    }
    return out;
}

// x [M, K] @ the one-bit weight of signs [N, K / 32] (bit c % 32 of word c / 32: set = positive) and scales [N].
inline Tensor64 sign_matmul(const Tensor64& x, const std::vector<int64_t>& signs, const Tensor64& scales) {
    const int64_t n = scales.numel();
    const int64_t k = x.shape.back();
    const int64_t rows = x.numel() / k;
    Tensor64 out({rows, n});
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < n; ++i) {
            double acc = 0;
            for (int64_t c = 0; c < k; ++c) {
                const auto word = static_cast<uint32_t>(signs[static_cast<size_t>(i * (k / 32) + c / 32)]);
                acc += (word >> (c % 32)) & 1u ? x[r * k + c] : -x[r * k + c];
            }
            out[r * n + i] = acc * scales[i];
        }
    }
    return out;
}

}  // namespace xs::test::ref
