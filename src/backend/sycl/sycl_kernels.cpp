#include "backend/sycl/sycl_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace xs::sycl_be {

// Event-less launches (sycl_ext_oneapi_enqueue_functions): the in-order queue orders them, and skipping the event
// makes each launch cheaper on the host.
namespace syclex = sycl::ext::oneapi::experimental;

namespace {

constexpr int kSubGroup = 16;

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

template <UnaryOp Op>
inline float apply_unary(float x, float a, float b) {
    if constexpr (Op == UnaryOp::Scale) {
        return x * a;
    } else if constexpr (Op == UnaryOp::Tanh) {
        return sycl::tanh(x);
    } else if constexpr (Op == UnaryOp::GeluTanh) {
        const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + sycl::tanh(inner));
    } else if constexpr (Op == UnaryOp::Softcap) {
        return a * sycl::tanh(x / a);
    } else if constexpr (Op == UnaryOp::Sigmoid) {
        return 1.0f / (1.0f + sycl::exp(-x));
    } else if constexpr (Op == UnaryOp::Silu) {
        return x / (1.0f + sycl::exp(-x));
    } else if constexpr (Op == UnaryOp::Softplus) {
        return x > 20.0f ? x : sycl::log1p(sycl::exp(x));  // torch's threshold: identity above 20
    } else {
        return sycl::fmin(sycl::fmax(x, a), b);
    }
}

template <UnaryOp Op>
void unary_t(sycl::queue& q, const float* x, float* y, int64_t n, float a, float b) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)),
                   [=](sycl::id<1> i) { y[i] = apply_unary<Op>(x[i], a, b); });
}

template <UnaryOp Act>
void act_mul_t(sycl::queue& q, const float* x, const float* b, float* y, int64_t n) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)),
                         [=](sycl::id<1> i) { y[i] = apply_unary<Act>(x[i], 0.0f, 0.0f) * b[i]; });
}

template <BinaryOp Op>
inline float apply_binary(float a, float b) {
    if constexpr (Op == BinaryOp::Add) {
        return a + b;
    } else {
        return a * b;
    }
}

template <BinaryOp Op>
void binary_t(sycl::queue& q, const float* a, const float* b, float* y, int64_t n, const Broadcast& s) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> idx) {
        int64_t i = static_cast<int64_t>(idx[0]);
        int64_t oa = 0;
        int64_t ob = 0;
        for (int d = s.rank - 1; d >= 0; --d) {
            const int64_t c = i % s.out[d];
            i /= s.out[d];
            oa += c * s.stride_a[d];
            ob += c * s.stride_b[d];
        }
        y[idx] = apply_binary<Op>(a[oa], b[ob]);
    });
}

template <BinaryOp Op>
void binary_same_t(sycl::queue& q, const float* a, const float* b, float* y, int64_t n) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) { y[i] = apply_binary<Op>(a[i], b[i]); });
}

template <BinaryOp Op>
void binary_period_t(sycl::queue& q, const float* a, const float* b, float* y, int64_t n, int64_t period) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
        y[i] = apply_binary<Op>(a[i], b[static_cast<int64_t>(i[0]) % period]);
    });
}

template <typename E>
void slice_t(sycl::queue& q, const E* x, E* y, int64_t outer, int64_t dim, int64_t begin, int64_t len, int64_t inner) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(outer * len * inner)), [=](sycl::id<1> idx) {
        const int64_t i = static_cast<int64_t>(idx[0]);
        const int64_t j = i % inner;
        const int64_t r = (i / inner) % len;
        const int64_t o = i / (inner * len);
        y[i] = x[(o * dim + begin + r) * inner + j];
    });
}

template <typename E>
void transpose_t(sycl::queue& q, const E* x, E* y, int64_t n, const Permute& p) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> idx) {
        int64_t i = static_cast<int64_t>(idx[0]);
        int64_t off = 0;
        for (int d = p.rank - 1; d >= 0; --d) {
            off += (i % p.out[d]) * p.stride[d];
            i /= p.out[d];
        }
        y[idx] = x[off];
    });
}

// Sum over the `P` adjacent lanes of a sub-group that share one column.
template <int P>
inline float reduce_lanes(sycl::sub_group sg, float v) {
    for (int m = 1; m < P; m <<= 1)
        v += sycl::permute_group_by_xor(sg, v, m);
    return v;
}

// Gated delta rule. One work-group per v head; P lanes per state column, each holding kDeltaRows rows of it in
// registers, so a sub-group covers 16 / P columns and the work-group all Dv columns (Dv * P threads). Tokens go in
// blocks of one per sub-group: the block's q and k (l2-normed, q scaled), v, exp(g) and beta are staged in local
// memory in parallel, then every column steps through the block in order.
constexpr int kDeltaRows = 32;

template <int P>
void gated_delta_t(sycl::queue& q, const float* qp, const float* kp, const float* vp, const float* g,
                   const float* beta, const float* state_in, float* state_out, float* out, int64_t tokens,
                   int64_t k_heads, int64_t v_heads, int64_t dk, int64_t dv, float eps) {
    constexpr int kCols = kSubGroup / P;
    const int64_t group = v_heads / k_heads;
    const int64_t wg_size = dv * P;
    const int64_t block = wg_size / kSubGroup;  // tokens staged at a time: one per sub-group
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(dk));
    syclex::submit(q, [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(static_cast<size_t>(block * dk)), h);
        sycl::local_accessor<float, 1> ks(sycl::range<1>(static_cast<size_t>(block * dk)), h);
        sycl::local_accessor<float, 1> vs(sycl::range<1>(static_cast<size_t>(block * dv)), h);
        sycl::local_accessor<float, 1> gb(sycl::range<1>(static_cast<size_t>(2 * block)), h);
        syclex::nd_launch(h, sycl::nd_range<1>(static_cast<size_t>(v_heads * wg_size), static_cast<size_t>(wg_size)),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t sgid = sg.get_group_linear_id();
            const int lane = static_cast<int>(sg.get_local_linear_id());
            const int64_t hv = static_cast<int64_t>(it.get_group(0));
            const int64_t kh = hv / group;
            const int64_t col = sgid * kCols + lane / P;
            const int part = lane % P;
            const int64_t row0 = static_cast<int64_t>(part) * kDeltaRows;

            float s[kDeltaRows];
            const float* s_in = state_in + hv * dk * dv;
            for (int r = 0; r < kDeltaRows; ++r)
                s[r] = s_in[(row0 + r) * dv + col];

            for (int64_t t0 = 0; t0 < tokens; t0 += block) {
                const int64_t count = std::min<int64_t>(block, tokens - t0);
                if (sgid < count) {  // sub-group sgid stages token t0 + sgid
                    const int64_t t = t0 + sgid;
                    const float* qt = qp + (t * k_heads + kh) * dk;
                    const float* kt = kp + (t * k_heads + kh) * dk;
                    float sq = 0.0f;
                    float sk = 0.0f;
                    for (int64_t d = lane; d < dk; d += kSubGroup) {
                        sq += qt[d] * qt[d];
                        sk += kt[d] * kt[d];
                    }
                    float q_norm = q_scale;
                    float k_norm = 1.0f;
                    if (eps > 0.0f) {
                        q_norm = q_scale / sycl::sqrt(sycl::reduce_over_group(sg, sq, sycl::plus<float>()) + eps);
                        k_norm = 1.0f / sycl::sqrt(sycl::reduce_over_group(sg, sk, sycl::plus<float>()) + eps);
                    }
                    for (int64_t d = lane; d < dk; d += kSubGroup) {
                        qs[sgid * dk + d] = qt[d] * q_norm;
                        ks[sgid * dk + d] = kt[d] * k_norm;
                    }
                    const float* vt = vp + (t * v_heads + hv) * dv;
                    for (int64_t d = lane; d < dv; d += kSubGroup)
                        vs[sgid * dv + d] = vt[d];
                    if (lane == 0) {
                        gb[2 * sgid] = sycl::exp(g[t * v_heads + hv]);
                        gb[2 * sgid + 1] = beta[t * v_heads + hv];
                    }
                }
                sycl::group_barrier(it.get_group());
                for (int64_t j = 0; j < count; ++j) {
                    const int64_t kq = j * dk + row0;
                    const float decay = gb[2 * j];
                    float kv = 0.0f;
                    for (int r = 0; r < kDeltaRows; ++r) {
                        s[r] *= decay;
                        kv += s[r] * ks[kq + r];
                    }
                    kv = reduce_lanes<P>(sg, kv);
                    const float delta = gb[2 * j + 1] * (vs[j * dv + col] - kv);
                    float o = 0.0f;
                    for (int r = 0; r < kDeltaRows; ++r) {
                        s[r] += ks[kq + r] * delta;
                        o += s[r] * qs[kq + r];
                    }
                    o = reduce_lanes<P>(sg, o);
                    if (part == 0) out[((t0 + j) * v_heads + hv) * dv + col] = o;
                }
                sycl::group_barrier(it.get_group());
            }
            float* s_out = state_out + hv * dk * dv;
            for (int r = 0; r < kDeltaRows; ++r)
                s_out[(row0 + r) * dv + col] = s[r];
        });
    });
}

constexpr int kAttnSubGroups = 16;

template <typename KV, int DPL>
void attention_rows_t(sycl::queue& q, const float* qp, const KV* k, const KV* v, float* out, const AttentionShape& s,
                      int64_t split, float* part) {
    constexpr int64_t D = static_cast<int64_t>(DPL) * kSubGroup;
    const int64_t hq_n = s.q_heads;
    const int64_t hkv_n = s.kv_heads;
    const int64_t group = hq_n / hkv_n;
    const int64_t n = s.kv_len;
    const int64_t tokens = s.tokens;
    const float scale = s.scale;
    const float softcap = s.softcap;
    const int64_t window = s.window;
    const bool causal = s.causal;
    constexpr int kWg = kAttnSubGroups * kSubGroup;
    syclex::submit(q, [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(kAttnSubGroups * (D + 2)), h);
        syclex::nd_launch(h, sycl::nd_range<1>(static_cast<size_t>(tokens * hq_n * split * kWg), kWg),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
            const int64_t wg = static_cast<int64_t>(it.get_group(0));
            const int64_t row = wg / split;
            const int64_t sp = wg % split;
            const int64_t t = row / hq_n;
            const int64_t hq = row % hq_n;
            const int64_t kh = hq / group;
            const sycl::sub_group sg = it.get_sub_group();
            const int sgid = static_cast<int>(sg.get_group_linear_id());
            const int lane = static_cast<int>(sg.get_local_linear_id());
            const int64_t pos = n - tokens + t;
            const int64_t first = causal && window > 0 ? std::max<int64_t>(0, pos - window + 1) : 0;
            const int64_t visible_last = causal ? pos : n - 1;
            // This work-group's share of the visible keys [first, visible_last].
            const int64_t chunk = (visible_last - first + split) / split;
            const int64_t lo = first + sp * chunk;
            const int64_t last = std::min<int64_t>(visible_last, lo + chunk - 1);

            float qv[DPL];
            const float* qr = qp + (t * hq_n + hq) * D;
            for (int e = 0; e < DPL; ++e)
                qv[e] = qr[lane + kSubGroup * e];
            float o[DPL];
            for (int e = 0; e < DPL; ++e)
                o[e] = 0.0f;
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.0f;
            for (int64_t j = lo + sgid; j <= last; j += kAttnSubGroups) {
                const KV* kr = k + (j * hkv_n + kh) * D;
                float dot = 0.0f;
                for (int e = 0; e < DPL; ++e)
                    dot += qv[e] * static_cast<float>(kr[lane + kSubGroup * e]);
                float sc = sycl::reduce_over_group(sg, dot, sycl::plus<float>()) * scale;
                if (softcap > 0.0f) sc = softcap * sycl::tanh(sc / softcap);
                const float m_new = sycl::fmax(m, sc);
                const float corr = sycl::exp(m - m_new);
                const float p = sycl::exp(sc - m_new);
                l = l * corr + p;
                const KV* vr = v + (j * hkv_n + kh) * D;
                for (int e = 0; e < DPL; ++e)
                    o[e] = o[e] * corr + p * static_cast<float>(vr[lane + kSubGroup * e]);
                m = m_new;
            }
            const int64_t base = static_cast<int64_t>(sgid) * (D + 2);
            for (int e = 0; e < DPL; ++e)
                slm[base + lane + kSubGroup * e] = o[e];
            if (lane == 0) {
                slm[base + D] = m;
                slm[base + D + 1] = l;
            }
            sycl::group_barrier(it.get_group());
            float m_all = -std::numeric_limits<float>::infinity();
            for (int sgi = 0; sgi < kAttnSubGroups; ++sgi)
                m_all = sycl::fmax(m_all, slm[sgi * (D + 2) + D]);
            for (int64_t d = static_cast<int64_t>(it.get_local_linear_id()); d < D; d += kWg) {
                float acc = 0.0f;
                float denom = 0.0f;
                for (int sgi = 0; sgi < kAttnSubGroups; ++sgi) {
                    const float ms = slm[sgi * (D + 2) + D];
                    if (ms == -std::numeric_limits<float>::infinity()) continue;
                    const float w = sycl::exp(ms - m_all);
                    denom += slm[sgi * (D + 2) + D + 1] * w;
                    acc += slm[sgi * (D + 2) + d] * w;
                }
                if (part == nullptr) {
                    out[(t * hq_n + hq) * D + d] = acc / denom;
                } else {  // unnormalized, relative to m_all; attention_combine merges the splits
                    float* pr = part + wg * (D + 2);
                    pr[d] = acc;
                    if (d == 0) {
                        pr[D] = m_all;
                        pr[D + 1] = denom;
                    }
                }
            }
        });
    });
}

// Each sub-group computes kGemvRows output rows for all m <= M x rows; lanes stride K in chunks of 8.
constexpr int kGemvSgPerWg = 8;
constexpr int kGemvRows = 2;

template <typename W, int M, bool Vec>
void gemv_t(sycl::queue& q, const float* x, const W* w, float* y, int64_t n, int64_t k) {
    constexpr int64_t kRowsPerWg = kGemvSgPerWg * kGemvRows;
    const int64_t wgs = ceil_div(n, kRowsPerWg);
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(wgs * kGemvSgPerWg * kSubGroup), kGemvSgPerWg * kSubGroup),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int64_t row0 = (static_cast<int64_t>(it.get_group(0)) * kGemvSgPerWg + sg.get_group_linear_id()) * kGemvRows;
        int64_t rows[kGemvRows];
        for (int r = 0; r < kGemvRows; ++r)
            rows[r] = std::min<int64_t>(row0 + r, n - 1);
        float acc[kGemvRows][M];
        for (int r = 0; r < kGemvRows; ++r)
            for (int mm = 0; mm < M; ++mm)
                acc[r][mm] = 0.0f;
        if constexpr (Vec) {
            for (int64_t k0 = static_cast<int64_t>(lane) * 8; k0 < k; k0 += kSubGroup * 8) {
                sycl::vec<float, 8> wf[kGemvRows];
                for (int r = 0; r < kGemvRows; ++r) {
                    const auto wv = *reinterpret_cast<const sycl::vec<W, 8>*>(w + rows[r] * k + k0);
                    wf[r] = wv.template convert<float>();
                }
                for (int mm = 0; mm < M; ++mm) {
                    const sycl::vec<float, 8> xv = *reinterpret_cast<const sycl::vec<float, 8>*>(x + mm * k + k0);
                    for (int r = 0; r < kGemvRows; ++r) {
                        const sycl::vec<float, 8> p = wf[r] * xv;
                        acc[r][mm] += ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
                    }
                }
            }
        } else {
            for (int64_t kk = lane; kk < k; kk += kSubGroup) {
                for (int r = 0; r < kGemvRows; ++r) {
                    const float wv = static_cast<float>(w[rows[r] * k + kk]);
                    for (int mm = 0; mm < M; ++mm)
                        acc[r][mm] += wv * x[mm * k + kk];
                }
            }
        }
        for (int r = 0; r < kGemvRows; ++r) {
            for (int mm = 0; mm < M; ++mm) {
                const float sum = sycl::reduce_over_group(sg, acc[r][mm], sycl::plus<float>());
                if (lane == 0 && row0 + r < n) y[mm * n + row0 + r] = sum;
            }
        }
    });
}

template <typename W, int M>
void gemv_m(sycl::queue& q, const float* x, const W* w, float* y, int64_t n, int64_t k) {
    const bool vec = k % 8 == 0 && reinterpret_cast<uintptr_t>(x) % 32 == 0 &&
                     reinterpret_cast<uintptr_t>(w) % (8 * sizeof(W)) == 0;
    if (vec) {
        gemv_t<W, M, true>(q, x, w, y, n, k);
    } else {
        gemv_t<W, M, false>(q, x, w, y, n, k);
    }
}

}  // namespace

void unary(sycl::queue& q, UnaryOp op, const float* x, float* y, int64_t n, float a, float b) {
    switch (op) {
        case UnaryOp::Scale: return unary_t<UnaryOp::Scale>(q, x, y, n, a, b);
        case UnaryOp::Tanh: return unary_t<UnaryOp::Tanh>(q, x, y, n, a, b);
        case UnaryOp::GeluTanh: return unary_t<UnaryOp::GeluTanh>(q, x, y, n, a, b);
        case UnaryOp::Softcap: return unary_t<UnaryOp::Softcap>(q, x, y, n, a, b);
        case UnaryOp::Sigmoid: return unary_t<UnaryOp::Sigmoid>(q, x, y, n, a, b);
        case UnaryOp::Silu: return unary_t<UnaryOp::Silu>(q, x, y, n, a, b);
        case UnaryOp::Softplus: return unary_t<UnaryOp::Softplus>(q, x, y, n, a, b);
        case UnaryOp::Clamp: return unary_t<UnaryOp::Clamp>(q, x, y, n, a, b);
    }
}

void act_mul(sycl::queue& q, UnaryOp act, const float* x, const float* b, float* y, int64_t n) {
    switch (act) {
        case UnaryOp::Silu: return act_mul_t<UnaryOp::Silu>(q, x, b, y, n);
        case UnaryOp::GeluTanh: return act_mul_t<UnaryOp::GeluTanh>(q, x, b, y, n);
        default: return act_mul_t<UnaryOp::Sigmoid>(q, x, b, y, n);
    }
}

void binary(sycl::queue& q, BinaryOp op, const float* a, const float* b, float* y, int64_t n, const Broadcast& shape) {
    if (op == BinaryOp::Add) {
        binary_t<BinaryOp::Add>(q, a, b, y, n, shape);
    } else {
        binary_t<BinaryOp::Mul>(q, a, b, y, n, shape);
    }
}

void binary_same(sycl::queue& q, BinaryOp op, const float* a, const float* b, float* y, int64_t n) {
    if (op == BinaryOp::Add) {
        binary_same_t<BinaryOp::Add>(q, a, b, y, n);
    } else {
        binary_same_t<BinaryOp::Mul>(q, a, b, y, n);
    }
}

void binary_period(sycl::queue& q, BinaryOp op, const float* a, const float* b, float* y, int64_t n, int64_t period) {
    if (op == BinaryOp::Add) {
        binary_period_t<BinaryOp::Add>(q, a, b, y, n, period);
    } else {
        binary_period_t<BinaryOp::Mul>(q, a, b, y, n, period);
    }
}

void rms_norm(sycl::queue& q, const float* x, const float* w, float* y, int64_t rows, int64_t width, float eps) {
    const size_t wg = width >= 1024 ? 512 : (width >= 256 ? 256 : 128);
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(rows) * wg, wg), [=](sycl::nd_item<1> it) {
        const int64_t r = static_cast<int64_t>(it.get_group(0));
        const float* xr = x + r * width;
        float* yr = y + r * width;
        float sum = 0.0f;
        for (int64_t i = static_cast<int64_t>(it.get_local_id(0)); i < width; i += static_cast<int64_t>(wg))
            sum += xr[i] * xr[i];
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float inv = 1.0f / sycl::sqrt(sum / static_cast<float>(width) + eps);
        for (int64_t i = static_cast<int64_t>(it.get_local_id(0)); i < width; i += static_cast<int64_t>(wg))
            yr[i] = w != nullptr ? xr[i] * inv * w[i] : xr[i] * inv;
    });
}

template <typename T, typename I>
void gather(sycl::queue& q, const T* table, const I* ids, float* out, int64_t n, int64_t width) {
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(n), static_cast<size_t>(width)), [=](sycl::item<2> it) {
        const int64_t r = static_cast<int64_t>(it[0]);
        const int64_t c = static_cast<int64_t>(it[1]);
        out[r * width + c] = static_cast<float>(table[static_cast<int64_t>(ids[r]) * width + c]);
    });
}

template <typename I>
void rope(sycl::queue& q, const float* x, const I* positions, const float* inv_freq, float* y, int64_t tokens,
          int64_t heads, int64_t dim, int64_t rotary, int64_t sections) {
    const int64_t width = rotary / sections;
    const int64_t half_w = width / 2;
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(tokens * heads * dim)), [=](sycl::id<1> idx) {
        const int64_t i = static_cast<int64_t>(idx[0]);
        const int64_t d = i % dim;
        if (d >= rotary) {
            y[i] = x[i];
            return;
        }
        const int64_t t = i / (heads * dim);
        const int64_t sec = d / width;
        const int64_t c = d % width;
        const int64_t c0 = c % half_w;
        // f32 angle as Hugging Face computes it (inv_freq @ positions in float32)
        const float angle = static_cast<float>(positions[t * sections + sec]) * inv_freq[c0];
        const float cs = sycl::cos(angle);
        const float sn = sycl::sin(angle);
        const int64_t base = i - c;
        y[i] = c < half_w ? x[i] * cs - x[base + c + half_w] * sn : x[i] * cs + x[base + c - half_w] * sn;
    });
}

void slice(sycl::queue& q, const void* x, void* y, size_t elem, int64_t outer, int64_t dim, int64_t begin, int64_t len,
           int64_t inner) {
    switch (elem) {
        case 1: return slice_t(q, static_cast<const uint8_t*>(x), static_cast<uint8_t*>(y), outer, dim, begin, len, inner);
        case 2: return slice_t(q, static_cast<const uint16_t*>(x), static_cast<uint16_t*>(y), outer, dim, begin, len, inner);
        case 4: return slice_t(q, static_cast<const uint32_t*>(x), static_cast<uint32_t*>(y), outer, dim, begin, len, inner);
        default: return slice_t(q, static_cast<const uint64_t*>(x), static_cast<uint64_t*>(y), outer, dim, begin, len, inner);
    }
}

void transpose(sycl::queue& q, const void* x, void* y, size_t elem, int64_t n, const Permute& p) {
    switch (elem) {
        case 1: return transpose_t(q, static_cast<const uint8_t*>(x), static_cast<uint8_t*>(y), n, p);
        case 2: return transpose_t(q, static_cast<const uint16_t*>(x), static_cast<uint16_t*>(y), n, p);
        case 4: return transpose_t(q, static_cast<const uint32_t*>(x), static_cast<uint32_t*>(y), n, p);
        default: return transpose_t(q, static_cast<const uint64_t*>(x), static_cast<uint64_t*>(y), n, p);
    }
}

void to_half(sycl::queue& q, const float* x, half* y, int64_t n) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) { y[i] = static_cast<half>(x[i]); });
}

void causal_conv(sycl::queue& q, const float* x, const float* w, const float* state_in, float* state_out, float* y,
                 int64_t tokens, int64_t channels, int64_t kernel, bool silu) {
    const int64_t past = kernel - 1;
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(tokens + past), static_cast<size_t>(channels)),
                   [=](sycl::item<2> it) {
        const int64_t t = static_cast<int64_t>(it[0]);
        const int64_t c = static_cast<int64_t>(it[1]);
        // x' = [state_in; x]
        auto ext = [&](int64_t r) { return r < past ? state_in[r * channels + c] : x[(r - past) * channels + c]; };
        if (t < tokens) {
            float acc = 0.0f;
            for (int64_t j = 0; j < kernel; ++j)
                acc += w[c * kernel + j] * ext(t + j);
            y[t * channels + c] = silu ? apply_unary<UnaryOp::Silu>(acc, 0.0f, 0.0f) : acc;
        } else {
            const int64_t r = t - tokens;
            state_out[r * channels + c] = ext(tokens + r);
        }
    });
}

void gated_delta(sycl::queue& q, const float* qp, const float* kp, const float* vp, const float* g, const float* beta,
                 const float* state_in, float* state_out, float* out, int64_t tokens, int64_t k_heads,
                 int64_t v_heads, int64_t dk, int64_t dv, float qk_l2norm_eps) {
    switch (dk / kDeltaRows) {
        case 1: return gated_delta_t<1>(q, qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dk, dv, qk_l2norm_eps);
        case 2: return gated_delta_t<2>(q, qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dk, dv, qk_l2norm_eps);
        case 4: return gated_delta_t<4>(q, qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dk, dv, qk_l2norm_eps);
        case 8: return gated_delta_t<8>(q, qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dk, dv, qk_l2norm_eps);
        default: return gated_delta_t<16>(q, qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dk, dv, qk_l2norm_eps);
    }
}

template <typename KV>
void attention_rows(sycl::queue& q, const float* qp, const KV* k, const KV* v, float* out, const AttentionShape& s,
                    int64_t split, float* part) {
    switch (s.head_dim / kSubGroup) {
        case 4: return attention_rows_t<KV, 4>(q, qp, k, v, out, s, split, part);
        case 8: return attention_rows_t<KV, 8>(q, qp, k, v, out, s, split, part);
        case 16: return attention_rows_t<KV, 16>(q, qp, k, v, out, s, split, part);
        case 32: return attention_rows_t<KV, 32>(q, qp, k, v, out, s, split, part);
        default: return attention_rows_t<KV, 12>(q, qp, k, v, out, s, split, part);  // head_dim 192
    }
}

void attention_combine(sycl::queue& q, const float* part, float* out, int64_t rows, int64_t split, int64_t head_dim) {
    const int64_t d_n = head_dim;
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(rows), static_cast<size_t>(d_n)), [=](sycl::item<2> it) {
        const int64_t r = static_cast<int64_t>(it[0]);
        const int64_t d = static_cast<int64_t>(it[1]);
        const float* pr = part + r * split * (d_n + 2);
        float m_all = -std::numeric_limits<float>::infinity();
        for (int64_t sp = 0; sp < split; ++sp)
            m_all = sycl::fmax(m_all, pr[sp * (d_n + 2) + d_n]);
        float acc = 0.0f;
        float denom = 0.0f;
        for (int64_t sp = 0; sp < split; ++sp) {
            const float ms = pr[sp * (d_n + 2) + d_n];
            if (ms == -std::numeric_limits<float>::infinity()) continue;
            const float w = sycl::exp(ms - m_all);
            acc += pr[sp * (d_n + 2) + d] * w;
            denom += pr[sp * (d_n + 2) + d_n + 1] * w;
        }
        out[r * d_n + d] = acc / denom;
    });
}

template <typename CT>
void attention_pack_q(sycl::queue& q, const float* qp, CT* packed, const AttentionShape& s, int64_t t0, int64_t tc) {
    const int64_t d_n = s.head_dim;
    const int64_t group = s.q_heads / s.kv_heads;
    const int64_t hq_n = s.q_heads;
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(s.kv_heads * tc * group * d_n)), [=](sycl::id<1> idx) {
        const int64_t i = static_cast<int64_t>(idx[0]);
        const int64_t d = i % d_n;
        const int64_t g = (i / d_n) % group;
        const int64_t tl = (i / (d_n * group)) % tc;
        const int64_t kh = i / (d_n * group * tc);
        packed[i] = static_cast<CT>(qp[((t0 + tl) * hq_n + kh * group + g) * d_n + d]);
    });
}

template <typename CT>
void attention_softmax(sycl::queue& q, const float* scores, CT* probs, const AttentionShape& s, int64_t t0, int64_t tc,
                       int64_t k_lo, int64_t keys) {
    const int64_t group = s.q_heads / s.kv_heads;
    const int64_t rows = s.kv_heads * tc * group;
    const int64_t past = s.kv_len - s.tokens;
    const float softcap = s.softcap;
    const int64_t window = s.window;
    const bool causal = s.causal;
    constexpr size_t kWg = 256;
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(rows) * kWg, kWg), [=](sycl::nd_item<1> it) {
        const int64_t r = static_cast<int64_t>(it.get_group(0));
        const int64_t tl = (r / group) % tc;
        const int64_t pos = past + t0 + tl;
        // visible keys, relative to k_lo: [lo, hi]
        int64_t lo = 0;
        int64_t hi = keys - 1;
        if (causal) {
            hi = std::min<int64_t>(hi, pos - k_lo);
            if (window > 0) lo = std::max<int64_t>(lo, pos - window + 1 - k_lo);
        }
        const float* sr = scores + r * keys;
        CT* pr = probs + r * keys;
        const int64_t lid = static_cast<int64_t>(it.get_local_id(0));
        auto score = [&](int64_t j) {
            const float v = sr[j];
            return softcap > 0.0f ? softcap * sycl::tanh(v / softcap) : v;
        };
        float mx = -std::numeric_limits<float>::infinity();
        for (int64_t j = lo + lid; j <= hi; j += static_cast<int64_t>(kWg))
            mx = sycl::fmax(mx, score(j));
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        float sum = 0.0f;
        for (int64_t j = lo + lid; j <= hi; j += static_cast<int64_t>(kWg))
            sum += sycl::exp(score(j) - mx);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float inv = 1.0f / sum;
        for (int64_t j = lid; j < keys; j += static_cast<int64_t>(kWg))
            pr[j] = static_cast<CT>(j >= lo && j <= hi ? sycl::exp(score(j) - mx) * inv : 0.0f);
    });
}

void attention_unpack(sycl::queue& q, const float* packed, float* out, const AttentionShape& s, int64_t t0, int64_t tc) {
    const int64_t d_n = s.head_dim;
    const int64_t group = s.q_heads / s.kv_heads;
    const int64_t hq_n = s.q_heads;
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(s.kv_heads * tc * group * d_n)), [=](sycl::id<1> idx) {
        const int64_t i = static_cast<int64_t>(idx[0]);
        const int64_t d = i % d_n;
        const int64_t g = (i / d_n) % group;
        const int64_t tl = (i / (d_n * group)) % tc;
        const int64_t kh = i / (d_n * group * tc);
        out[((t0 + tl) * hq_n + kh * group + g) * d_n + d] = packed[i];
    });
}

template <typename W>
void gemv(sycl::queue& q, const float* x, const W* w, float* y, int64_t m, int64_t n, int64_t k) {
    switch (m) {
        case 1: return gemv_m<W, 1>(q, x, w, y, n, k);
        case 2: return gemv_m<W, 2>(q, x, w, y, n, k);
        case 3: return gemv_m<W, 3>(q, x, w, y, n, k);
        case 4: return gemv_m<W, 4>(q, x, w, y, n, k);
        case 5: return gemv_m<W, 5>(q, x, w, y, n, k);
        case 6: return gemv_m<W, 6>(q, x, w, y, n, k);
        case 7: return gemv_m<W, 7>(q, x, w, y, n, k);
        default: return gemv_m<W, 8>(q, x, w, y, n, k);
    }
}

template void gather<half, int32_t>(sycl::queue&, const half*, const int32_t*, float*, int64_t, int64_t);
template void gather<half, int64_t>(sycl::queue&, const half*, const int64_t*, float*, int64_t, int64_t);
template void gather<float, int32_t>(sycl::queue&, const float*, const int32_t*, float*, int64_t, int64_t);
template void gather<float, int64_t>(sycl::queue&, const float*, const int64_t*, float*, int64_t, int64_t);
template void rope<int32_t>(sycl::queue&, const float*, const int32_t*, const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
template void rope<int64_t>(sycl::queue&, const float*, const int64_t*, const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);
template void attention_rows<half>(sycl::queue&, const float*, const half*, const half*, float*, const AttentionShape&,
                                   int64_t, float*);
template void attention_rows<float>(sycl::queue&, const float*, const float*, const float*, float*,
                                    const AttentionShape&, int64_t, float*);
template void attention_pack_q<half>(sycl::queue&, const float*, half*, const AttentionShape&, int64_t, int64_t);
template void attention_pack_q<float>(sycl::queue&, const float*, float*, const AttentionShape&, int64_t, int64_t);
template void attention_softmax<half>(sycl::queue&, const float*, half*, const AttentionShape&, int64_t, int64_t, int64_t, int64_t);
template void attention_softmax<float>(sycl::queue&, const float*, float*, const AttentionShape&, int64_t, int64_t, int64_t, int64_t);
template void gemv<half>(sycl::queue&, const float*, const half*, float*, int64_t, int64_t, int64_t);
template void gemv<float>(sycl::queue&, const float*, const float*, float*, int64_t, int64_t, int64_t);

}  // namespace xs::sycl_be
