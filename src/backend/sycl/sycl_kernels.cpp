#include "backend/sycl/sycl_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include "util/error.h"

namespace xs::sycl_be {

// Event-less launches (sycl_ext_oneapi_enqueue_functions): the in-order queue orders them, and skipping the event
// makes each launch cheaper on the host.
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

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
        // 1 + tanh(u) = 2 - 2 / (exp(2u) + 1): the hardware exponential instead of sycl::tanh's long sequence (the
        // product with the gate is a large pass), absolute error near 1e-7.
        const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return x * (1.0f - 1.0f / (sycl::native::exp(2.0f * inner) + 1.0f));
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

template <UnaryOp Op, typename Y>
void unary_t(sycl::queue& q, const float* x, Y* y, int64_t n, float a, float b) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)),
                   [=](sycl::id<1> i) { y[i] = static_cast<Y>(apply_unary<Op>(x[i], a, b)); });
}

template <UnaryOp Act, typename X, typename Y>
void act_mul_t(sycl::queue& q, const X* x, const X* b, Y* y, int64_t n) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
        y[i] = static_cast<Y>(apply_unary<Act>(static_cast<float>(x[i]), 0.0f, 0.0f) * static_cast<float>(b[i]));
    });
}

template <UnaryOp Op, typename Y>
void unary_bias_t(sycl::queue& q, const float* x, const float* bias, Y* y, int64_t n, int64_t period) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
        y[i] = static_cast<Y>(apply_unary<Op>(x[i] + bias[static_cast<int64_t>(i[0]) % period], 0.0f, 0.0f));
    });
}

template <BinaryOp Op>
inline float apply_binary(float a, float b) {
    if constexpr (Op == BinaryOp::Add) {
        return a + b;
    } else if constexpr (Op == BinaryOp::Mul) {
        return a * b;
    } else {
        return a / b;
    }
}

template <BinaryOp Op, typename Y>
void binary_t(sycl::queue& q, const float* a, const float* b, Y* y, int64_t n, const Broadcast& s) {
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
        y[idx] = static_cast<Y>(apply_binary<Op>(a[oa], b[ob]));
    });
}

template <BinaryOp Op, typename Y>
void binary_same_t(sycl::queue& q, const float* a, const float* b, Y* y, int64_t n) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)),
                         [=](sycl::id<1> i) { y[i] = static_cast<Y>(apply_binary<Op>(a[i], b[i])); });
}

template <BinaryOp Op, typename Y>
void binary_period_t(sycl::queue& q, const float* a, const float* b, Y* y, int64_t n, int64_t period) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
        y[i] = static_cast<Y>(apply_binary<Op>(a[i], b[static_cast<int64_t>(i[0]) % period]));
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

// The same for a few tokens (decode): one work-group per (v head, 16 columns of its state), each of its eight
// sub-groups holding RS = dk / 8 rows of those columns in registers, a lane per column. The state then streams in
// whole rows and the device fills at one token; the two sums over the rows (k . S, q . S) go through local memory,
// and every sub-group finds the q and k norms itself.
constexpr int kDeltaColsSgs = 8;
constexpr int64_t kDeltaColsMaxTokens = 8;

template <int RS>
void gated_delta_cols_t(sycl::queue& q, const float* qp, const float* kp, const float* vp, const float* g,
                        const float* beta, const float* state_in, float* state_out, float* out, int64_t tokens,
                        int64_t k_heads, int64_t v_heads, int64_t dk, int64_t dv, float eps) {
    const int64_t group = v_heads / k_heads;
    const int64_t col_blocks = dv / kSubGroup;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(dk));
    syclex::submit(q, [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> kv_parts(sycl::range<1>(kDeltaColsSgs * kSubGroup), h);
        sycl::local_accessor<float, 1> o_parts(sycl::range<1>(kDeltaColsSgs * kSubGroup), h);
        syclex::nd_launch(h, sycl::nd_range<1>(static_cast<size_t>(v_heads * col_blocks * kDeltaColsSgs * kSubGroup),
                                               kDeltaColsSgs * kSubGroup),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int sgid = static_cast<int>(sg.get_group_linear_id());
            const int lane = static_cast<int>(sg.get_local_linear_id());
            const int64_t hv = static_cast<int64_t>(it.get_group(0)) / col_blocks;
            const int64_t col = (static_cast<int64_t>(it.get_group(0)) % col_blocks) * kSubGroup + lane;
            const int64_t kh = hv / group;
            const int64_t row0 = static_cast<int64_t>(sgid) * RS;
            float s[RS];
            const float* s_in = state_in + hv * dk * dv;
            for (int r = 0; r < RS; ++r)
                s[r] = s_in[(row0 + r) * dv + col];
            for (int64_t t = 0; t < tokens; ++t) {
                const float* qt = qp + (t * k_heads + kh) * dk;
                const float* kt = kp + (t * k_heads + kh) * dk;
                float q_norm = q_scale;
                float k_norm = 1.0f;
                if (eps > 0.0f) {
                    float sq = 0.0f;
                    float sk = 0.0f;
                    for (int64_t d = lane; d < dk; d += kSubGroup) {
                        sq += qt[d] * qt[d];
                        sk += kt[d] * kt[d];
                    }
                    q_norm = q_scale / sycl::sqrt(sycl::reduce_over_group(sg, sq, sycl::plus<float>()) + eps);
                    k_norm = 1.0f / sycl::sqrt(sycl::reduce_over_group(sg, sk, sycl::plus<float>()) + eps);
                }
                const float decay = sycl::exp(g[t * v_heads + hv]);
                float kr[RS];
                float kv = 0.0f;
                for (int r = 0; r < RS; ++r) {
                    kr[r] = kt[row0 + r] * k_norm;
                    s[r] *= decay;
                    kv += s[r] * kr[r];
                }
                kv_parts[sgid * kSubGroup + lane] = kv;
                sycl::group_barrier(it.get_group());
                kv = 0.0f;
                for (int j = 0; j < kDeltaColsSgs; ++j)
                    kv += kv_parts[j * kSubGroup + lane];
                const float delta = beta[t * v_heads + hv] * (vp[(t * v_heads + hv) * dv + col] - kv);
                float o = 0.0f;
                for (int r = 0; r < RS; ++r) {
                    s[r] += kr[r] * delta;
                    o += s[r] * (qt[row0 + r] * q_norm);
                }
                o_parts[sgid * kSubGroup + lane] = o;
                sycl::group_barrier(it.get_group());  // also: every kv_parts read is done before the next token writes
                if (sgid == 0) {
                    o = 0.0f;
                    for (int j = 0; j < kDeltaColsSgs; ++j)
                        o += o_parts[j * kSubGroup + lane];
                    out[(t * v_heads + hv) * dv + col] = o;
                }
            }
            float* s_out = state_out + hv * dk * dv;
            for (int r = 0; r < RS; ++r)
                s_out[(row0 + r) * dv + col] = s[r];
        });
    });
}

// The same for many tokens (prefill): one sub-group per (v head, 16 state columns), a lane per column holding all DK
// rows of it in registers (256 per thread), so the sums over rows (k . S, q . S) stay within the lane and the tokens
// step without a barrier. Each lane loads a DK / 16 slice of the token's q and k (one coalesced load, and the norms'
// partial sums) and the rows reach every lane by sub-group broadcasts. (One work-group per v head, the rows split
// between lanes and staged per token, kept 48 work-groups on the GPU's 160 cores: 99 ms of a 512-token Qwen3.8
// prefill; uniform loads of q and k per row cost more still.)
template <int DK>
struct GatedDeltaLanes {
    static constexpr int kSlice = DK / kSubGroup;  // q and k values per lane
    const float* qp;
    const float* kp;
    const float* vp;
    const float* g;
    const float* beta;
    const float* state_in;
    float* state_out;
    float* out;
    int64_t tokens, k_heads, v_heads, dv;
    float eps, q_scale;

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<kSubGroup>, intelex::grf_size<256>};
    }

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int64_t blocks = dv / kSubGroup;
        const int64_t hv = static_cast<int64_t>(it.get_group(0)) / blocks;
        const int64_t col = (static_cast<int64_t>(it.get_group(0)) % blocks) * kSubGroup + lane;
        const int64_t kh = hv / (v_heads / k_heads);
        constexpr int kAcc = 4;  // partial sums: the chains over rows run in parallel
        // every loop over rows unrolled: s stays in registers (a loop indexes it in scratch memory)
        float s[DK];
        const float* s_in = state_in + hv * DK * dv;
        #pragma unroll
        for (int r = 0; r < DK; ++r) s[r] = s_in[r * dv + col];
        for (int64_t t = 0; t < tokens; ++t) {
            const sycl::vec<float, kSlice> kl =
                *reinterpret_cast<const sycl::vec<float, kSlice>*>(kp + (t * k_heads + kh) * DK + lane * kSlice);
            const sycl::vec<float, kSlice> ql =
                *reinterpret_cast<const sycl::vec<float, kSlice>*>(qp + (t * k_heads + kh) * DK + lane * kSlice);
            float q_norm = q_scale;
            float k_norm = 1.0f;
            if (eps > 0.0f) {
                float sq = 0.0f;
                float sk = 0.0f;
                #pragma unroll
                for (int i = 0; i < kSlice; ++i) {
                    sq += ql[i] * ql[i];
                    sk += kl[i] * kl[i];
                }
                q_norm = q_scale / sycl::sqrt(sycl::reduce_over_group(sg, sq, sycl::plus<float>()) + eps);
                k_norm = 1.0f / sycl::sqrt(sycl::reduce_over_group(sg, sk, sycl::plus<float>()) + eps);
            }
            const float decay = sycl::exp(g[t * v_heads + hv]);
            // row r of q and k lives in lane r / kSlice (broadcast where used: a register each, not 2 x DK more)
            float kv[kAcc] = {};
            #pragma unroll
            for (int r = 0; r < DK; ++r) {
                s[r] *= decay;
                kv[r % kAcc] += s[r] * sycl::group_broadcast(sg, kl[r % kSlice], r / kSlice);
            }
            const float kv_sum = (kv[0] + kv[1]) + (kv[2] + kv[3]);
            const float delta = beta[t * v_heads + hv] * (vp[(t * v_heads + hv) * dv + col] - kv_sum * k_norm) * k_norm;
            float o[kAcc] = {};
            #pragma unroll
            for (int r = 0; r < DK; ++r) {
                s[r] += sycl::group_broadcast(sg, kl[r % kSlice], r / kSlice) * delta;
                o[r % kAcc] += s[r] * sycl::group_broadcast(sg, ql[r % kSlice], r / kSlice);
            }
            out[(t * v_heads + hv) * dv + col] = ((o[0] + o[1]) + (o[2] + o[3])) * q_norm;
        }
        float* s_out = state_out + hv * DK * dv;
        #pragma unroll
        for (int r = 0; r < DK; ++r) s_out[r * dv + col] = s[r];
    }
};

template <int DK>
void gated_delta_lanes_t(sycl::queue& q, const float* qp, const float* kp, const float* vp, const float* g,
                         const float* beta, const float* state_in, float* state_out, float* out, int64_t tokens,
                         int64_t k_heads, int64_t v_heads, int64_t dv, float eps) {
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(DK));
    const auto sgs = static_cast<size_t>(v_heads * (dv / kSubGroup));
    syclex::nd_launch(q, sycl::nd_range<1>(sgs * kSubGroup, kSubGroup),
                      GatedDeltaLanes<DK>{qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dv,
                                          eps, q_scale});
}

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

// KV int8_t reads an int8 cache with per-(key, head) scales k_scale / v_scale; other KV types ignore them.
template <typename KV, int DPL>
void attention_rows_t(sycl::queue& q, const float* qp, const KV* k, const KV* v, const float* k_scale,
                      const float* v_scale, float* out, const AttentionShape& s, int64_t split, float* part) {
    constexpr bool kI8 = std::is_same_v<KV, int8_t>;
    constexpr bool kPacked = kI8 && DPL % 4 == 0;
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
    const float* key_bias = s.key_bias;
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
            const int64_t first = window > 0 ? std::max<int64_t>(0, pos - window + 1) : 0;
            const int64_t visible_last = causal ? pos : n - 1;
            // This work-group's share of the visible keys [first, visible_last].
            const int64_t chunk = (visible_last - first + split) / split;
            const int64_t lo = first + sp * chunk;
            const int64_t last = std::min<int64_t>(visible_last, lo + chunk - 1);

            // The lane's e-th dim. Packed int8: 4 consecutive bytes per lane, one 32-bit load (64 bytes a sub-group).
            const auto dim = [lane](int e) -> int64_t {
                if constexpr (kPacked) {
                    return (e / 4) * (4 * kSubGroup) + lane * 4 + e % 4;
                } else {
                    return lane + kSubGroup * e;
                }
            };
            // dot / accumulate over the lane's dims of one key / value row.
            const auto load = [&](const KV* row, float* x) {
                if constexpr (kPacked) {
                    for (int g = 0; g < DPL / 4; ++g) {
                        const sycl::vec<int8_t, 4> c =
                            *reinterpret_cast<const sycl::vec<int8_t, 4>*>(row + g * 4 * kSubGroup + lane * 4);
                        for (int i = 0; i < 4; ++i)
                            x[g * 4 + i] = static_cast<float>(c[i]);
                    }
                } else {
                    for (int e = 0; e < DPL; ++e)
                        x[e] = static_cast<float>(row[lane + kSubGroup * e]);
                }
            };
            float qv[DPL];
            const float* qr = qp + (t * hq_n + hq) * D;
            for (int e = 0; e < DPL; ++e)
                qv[e] = qr[dim(e)];
            float o[DPL];
            for (int e = 0; e < DPL; ++e)
                o[e] = 0.0f;
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.0f;
            for (int64_t j = lo + sgid; j <= last; j += kAttnSubGroups) {
                float x[DPL];
                load(k + (j * hkv_n + kh) * D, x);
                float dot = 0.0f;
                for (int e = 0; e < DPL; ++e)
                    dot += qv[e] * x[e];
                float sc = sycl::reduce_over_group(sg, dot, sycl::plus<float>()) * scale;
                if constexpr (kI8) sc *= k_scale[j * hkv_n + kh];
                if (softcap > 0.0f) sc = softcap * sycl::tanh(sc / softcap);
                if (key_bias != nullptr) sc += key_bias[j];
                if (sc == -std::numeric_limits<float>::infinity()) continue;  // a masked key
                const float m_new = sycl::fmax(m, sc);
                const float corr = sycl::exp(m - m_new);
                const float p = sycl::exp(sc - m_new);
                l = l * corr + p;
                float pv = p;
                if constexpr (kI8) pv *= v_scale[j * hkv_n + kh];
                load(v + (j * hkv_n + kh) * D, x);
                for (int e = 0; e < DPL; ++e)
                    o[e] = o[e] * corr + pv * x[e];
                m = m_new;
            }
            const int64_t base = static_cast<int64_t>(sgid) * (D + 2);
            for (int e = 0; e < DPL; ++e)
                slm[base + dim(e)] = o[e];
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

template <typename KV>
void attention_rows_dispatch(sycl::queue& q, const float* qp, const KV* k, const KV* v, const float* ks,
                             const float* vs, float* out, const AttentionShape& s, int64_t split, float* part) {
    switch (s.head_dim / kSubGroup) {
        case 1: return attention_rows_t<KV, 1>(q, qp, k, v, ks, vs, out, s, split, part);
        case 2: return attention_rows_t<KV, 2>(q, qp, k, v, ks, vs, out, s, split, part);
        case 4: return attention_rows_t<KV, 4>(q, qp, k, v, ks, vs, out, s, split, part);
        case 8: return attention_rows_t<KV, 8>(q, qp, k, v, ks, vs, out, s, split, part);
        case 16: return attention_rows_t<KV, 16>(q, qp, k, v, ks, vs, out, s, split, part);
        case 32: return attention_rows_t<KV, 32>(q, qp, k, v, ks, vs, out, s, split, part);
        default: return attention_rows_t<KV, 12>(q, qp, k, v, ks, vs, out, s, split, part);  // head_dim 192
    }
}

// Each sub-group computes kGemvRows output rows for all m <= M x rows; lanes stride K in chunks of 8. With KS > 1
// the work-group's KS sub-groups of a row pair split K and add up in local memory: a few rows (a gate's) then
// still fill the device.
constexpr int kGemvSgPerWg = 8;
constexpr int kGemvRows = 2;

template <typename W, int M, bool Vec, int KS>
void gemv_t(sycl::queue& q, const float* x, const W* w, float* y, int64_t n, int64_t k) {
    constexpr int kSets = kGemvSgPerWg / KS;
    constexpr int64_t kRowsPerWg = kSets * kGemvRows;
    const int64_t wgs = ceil_div(n, kRowsPerWg);
    syclex::submit(q, [&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>(KS > 1 ? kGemvSgPerWg * kGemvRows * M : 1), h);
        syclex::nd_launch(h, sycl::nd_range<1>(static_cast<size_t>(wgs * kGemvSgPerWg * kSubGroup),
                                               kGemvSgPerWg * kSubGroup),
                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = static_cast<int>(sg.get_local_linear_id());
            const int sgid = static_cast<int>(sg.get_group_linear_id());
            const int set = sgid / KS;
            const int slice = sgid % KS;
            const int64_t row0 = (static_cast<int64_t>(it.get_group(0)) * kSets + set) * kGemvRows;
            int64_t rows[kGemvRows];
            for (int r = 0; r < kGemvRows; ++r)
                rows[r] = std::min<int64_t>(row0 + r, n - 1);
            float acc[kGemvRows][M];
            for (int r = 0; r < kGemvRows; ++r)
                for (int mm = 0; mm < M; ++mm)
                    acc[r][mm] = 0.0f;
            const int64_t first = static_cast<int64_t>(slice) * kSubGroup + lane;  // this lane's first chunk
            if constexpr (Vec) {
                for (int64_t k0 = first * 8; k0 < k; k0 += KS * kSubGroup * 8) {
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
                for (int64_t kk = first; kk < k; kk += KS * kSubGroup) {
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
                    if constexpr (KS == 1) {
                        if (lane == 0 && row0 + r < n) y[mm * n + row0 + r] = sum;
                    } else if (lane == 0) {
                        partial[(sgid * kGemvRows + r) * M + mm] = sum;
                    }
                }
            }
            if constexpr (KS > 1) {
                sycl::group_barrier(it.get_group());
                if (slice == 0 && lane < kGemvRows * M) {
                    const int r = lane / M;
                    const int mm = lane % M;
                    float sum = 0.0f;
                    for (int j = 0; j < KS; ++j)
                        sum += partial[((set * KS + j) * kGemvRows + r) * M + mm];
                    if (row0 + r < n) y[mm * n + row0 + r] = sum;
                }
            }
        });
    });
}

// Quantized GEMV. A sub-group takes R rows; a lane takes a chunk of CB bytes of codes of each (2 * CB q4 or CB q8
// values, inside one group as group >= 32), so the chunk's x loads once for all R rows and its codes decode once for
// all M inputs. Over a chunk dot(w, x) = scale * dot(q, x) + bias * sum(x). A q4 code q is read from its bits with
// no int-to-float conversion: in mantissa bits 19..22 of 1.0f it reads u = 1 + q / 16, so dot(q, x) =
// 16 * (dot(u, x) - sum(x)).
constexpr int kQSgPerWg = 8;

// Rows per sub-group and code bytes per lane by format and inputs, measured on the B60: two rows and 16 bytes
// stream the weight at M = 1 (four with AWQ, which scales each x chunk once for all of them); as M grows, smaller
// chunks keep the decoded codes and the accumulators in registers (q4: 8 bytes up to M = 3, four rows at M = 3).
template <bool Q4, int M, bool AWQ = false>
struct QuantGemvShape {
    static constexpr int kRows = (Q4 && M == 3) || (AWQ && M == 1) ? 4 : 2;
    static constexpr int kBytes = Q4 ? (M == 1 ? 16 : M <= 3 ? 8 : 4) : (M <= 2 ? 16 : M <= 4 ? 8 : 4);
};

inline float q4_unit(uint32_t word, int j) {  // 1 + (nibble j of word) / 16
    const uint32_t m = 4 * j <= 19 ? word << (19 - 4 * j) : word >> (4 * j - 19);
    return sycl::bit_cast<float>((m & 0x00780000u) | 0x3F800000u);
}

// acc[r][m] += dequant(row rows[r] of w) . xs[m] over this lane's chunks.
template <bool Q4, bool AWQ, int M, int R, int CB>
inline void quant_rows_dot(const QuantWeight& w, const int64_t (&rows)[R], const float* const (&xs)[M], int lane,
                           int64_t k, float (&acc)[R][M]) {
    constexpr int kWords = CB / 4;
    constexpr int kChunk = Q4 ? 2 * CB : CB;
    const int64_t groups = k / w.group;
    const int64_t row_bytes = Q4 ? k / 2 : k;
    // the group of a column by a shift (a division is emulated, and ran once per chunk); groups are powers of two
    // in every pack, a division otherwise
    const auto group = static_cast<int32_t>(w.group);
    const bool pow2 = (group & (group - 1)) == 0;
    const int shift = sycl::ctz(group);
    for (int64_t k0 = static_cast<int64_t>(lane) * kChunk; k0 < k; k0 += kSubGroup * kChunk) {
        const int64_t g = pow2 ? static_cast<int32_t>(k0) >> shift : static_cast<int32_t>(k0) / group;
        float wv[R][kChunk];
        float s[R];
        float b[R];
        for (int r = 0; r < R; ++r) {
            const auto c = *reinterpret_cast<const sycl::vec<uint32_t, kWords>*>(w.codes + rows[r] * row_bytes +
                                                                                 (Q4 ? k0 / 2 : k0));
            s[r] = static_cast<float>(w.scales[rows[r] * groups + g]);
            b[r] = Q4 ? static_cast<float>(w.biases[rows[r] * groups + g]) : 0.0f;
            for (int i = 0; i < kWords; ++i) {
                const uint32_t u = c[i];
                if constexpr (Q4) {
                    for (int j = 0; j < 8; ++j)
                        wv[r][8 * i + j] = q4_unit(u, j);
                } else {
                    for (int j = 0; j < 4; ++j)
                        wv[r][4 * i + j] = static_cast<float>(static_cast<int8_t>(u >> (8 * j)));
                }
            }
        }
        for (int mm = 0; mm < M; ++mm) {
            float xv[kChunk];
            const auto* xp = reinterpret_cast<const sycl::vec<float, 4>*>(xs[mm] + k0);
            for (int i = 0; i < kChunk / 4; ++i) {
                sycl::vec<float, 4> v = xp[i];
                if constexpr (AWQ) {  // (x * input_scales) . dequant(w) == x . (dequant(w) * input_scales)
                    v *= reinterpret_cast<const sycl::vec<float, 4>*>(w.input_scales + k0)[i];
                }
                for (int j = 0; j < 4; ++j)
                    xv[4 * i + j] = v[j];
            }
            float xsum = 0.0f;
            if constexpr (Q4) {
                for (int i = 0; i < kChunk; ++i)
                    xsum += xv[i];
            }
            for (int r = 0; r < R; ++r) {
                float dot = 0.0f;
                for (int i = 0; i < kChunk; ++i)
                    dot += wv[r][i] * xv[i];
                if constexpr (Q4) {
                    acc[r][mm] += 16.0f * s[r] * dot + (b[r] - 16.0f * s[r]) * xsum;
                } else {
                    acc[r][mm] += s[r] * dot;
                }
            }
        }
    }
}

// The parts of a GEMV launch and where each one's work-groups start (a work-group's rows lie in one part).
struct QuantParts {
    QuantWeight w[kQuantGemvParts];
    float* y[kQuantGemvParts];
    int64_t n[kQuantGemvParts];
    int64_t wg_begin[kQuantGemvParts + 1];
    int count = 0;
};

template <bool Q4, bool AWQ, int M>
void gemv_quant_t(sycl::queue& q, const float* x, QuantParts parts, int64_t k) {
    using Shape = QuantGemvShape<Q4, M, AWQ>;
    constexpr int R = Shape::kRows;
    for (int j = 0; j < kQuantGemvParts; ++j)
        parts.wg_begin[j + 1] = parts.wg_begin[j] + (j < parts.count ? ceil_div(parts.n[j], kQSgPerWg * R) : 0);
    const int64_t wgs = parts.wg_begin[parts.count];
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(wgs * kQSgPerWg * kSubGroup), kQSgPerWg * kSubGroup),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const auto wg = static_cast<int64_t>(it.get_group(0));
        int part = 0;  // uniform across the work-group; selected without indexing the argument arrays
        for (int j = 1; j < kQuantGemvParts; ++j)
            part += j < parts.count && wg >= parts.wg_begin[j] ? 1 : 0;
        QuantWeight w = parts.w[0];
        float* y = parts.y[0];
        int64_t n = parts.n[0];
        int64_t begin = 0;
        for (int j = 1; j < kQuantGemvParts; ++j) {
            if (part == j) {
                w = parts.w[j];
                y = parts.y[j];
                n = parts.n[j];
                begin = parts.wg_begin[j];
            }
        }
        const int64_t row0 = ((wg - begin) * kQSgPerWg + sg.get_group_linear_id()) * R;
        int64_t rows[R];
        for (int r = 0; r < R; ++r)
            rows[r] = std::min<int64_t>(row0 + r, n - 1);
        const float* xs[M];
        for (int mm = 0; mm < M; ++mm)
            xs[mm] = x + mm * k;
        float acc[R][M] = {};
        quant_rows_dot<Q4, AWQ, M, R, Shape::kBytes>(w, rows, xs, lane, k, acc);
        for (int r = 0; r < R; ++r) {
            for (int mm = 0; mm < M; ++mm) {
                const float sum = sycl::reduce_over_group(sg, acc[r][mm], sycl::plus<float>());
                if (lane == 0 && row0 + r < n) y[mm * n + row0 + r] = sum;
            }
        }
    });
}

template <bool Q4, bool AWQ>
void gemv_quant_q(sycl::queue& q, const float* x, const QuantParts& parts, int64_t m, int64_t k) {
    switch (m) {
        case 1: return gemv_quant_t<Q4, AWQ, 1>(q, x, parts, k);
        case 2: return gemv_quant_t<Q4, AWQ, 2>(q, x, parts, k);
        case 3: return gemv_quant_t<Q4, AWQ, 3>(q, x, parts, k);
        case 4: return gemv_quant_t<Q4, AWQ, 4>(q, x, parts, k);
        case 5: return gemv_quant_t<Q4, AWQ, 5>(q, x, parts, k);
        case 6: return gemv_quant_t<Q4, AWQ, 6>(q, x, parts, k);
        case 7: return gemv_quant_t<Q4, AWQ, 7>(q, x, parts, k);
        default: return gemv_quant_t<Q4, AWQ, 8>(q, x, parts, k);
    }
}

template <typename I>
inline int64_t expert_of(const I* idx, int64_t pair, int64_t experts) {
    const int64_t e = static_cast<int64_t>(idx[pair]);
    return e < 0 || e >= experts ? 0 : e;
}

// One work-group per (pair, block of kQSgPerWg * R rows).
template <bool Q4, bool AWQ, typename I>
void expert_quant_pairs_t(sycl::queue& q, const float* x, bool per_slot, QuantWeight w, const I* idx, float* out,
                          int64_t pairs, int64_t slots, int64_t experts, int64_t n, int64_t k) {
    using Shape = QuantGemvShape<Q4, 1, AWQ>;
    constexpr int kQRows = Shape::kRows;
    constexpr int64_t kRowsPerWg = kQSgPerWg * kQRows;
    const int64_t blocks = ceil_div(n, kRowsPerWg);
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(pairs * blocks * kQSgPerWg * kSubGroup),
                                           kQSgPerWg * kSubGroup),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int64_t p = static_cast<int64_t>(it.get_group(0)) / blocks;
        const int64_t block = static_cast<int64_t>(it.get_group(0)) % blocks;
        const int64_t row0 = (block * kQSgPerWg + sg.get_group_linear_id()) * kQRows;
        const int64_t base = expert_of(idx, p, experts) * n;
        int64_t rows[kQRows];
        for (int r = 0; r < kQRows; ++r)
            rows[r] = base + std::min<int64_t>(row0 + r, n - 1);
        const float* const xs[1] = {x + (per_slot ? p : p / slots) * k};
        float acc[kQRows][1] = {};
        quant_rows_dot<Q4, AWQ, 1, kQRows, Shape::kBytes>(w, rows, xs, lane, k, acc);
        for (int r = 0; r < kQRows; ++r) {
            const float sum = sycl::reduce_over_group(sg, acc[r][0], sycl::plus<float>());
            if (lane == 0 && row0 + r < n) out[p * n + row0 + r] = sum;
        }
    });
}

// grouping (int32): counts [E], offsets [E + 1], cursor [E], active [E], n_active [1], order [pairs].
struct ExpertGrouping {
    int32_t* counts;
    int32_t* offsets;
    int32_t* cursor;
    int32_t* active;
    int32_t* n_active;
    int32_t* order;
    ExpertGrouping(int32_t* g, int64_t experts)
        : counts(g),
          offsets(g + experts),
          cursor(g + 2 * experts + 1),
          active(g + 3 * experts + 1),
          n_active(g + 4 * experts + 1),
          order(g + 4 * experts + 2) {}
};

// Counting sort of the pairs by expert. The order inside an expert is not deterministic, which does not matter: each
// pair's output row is computed the same way wherever it lands.
template <typename I>
void group_by_expert(sycl::queue& q, const I* idx, int64_t pairs, int64_t experts, ExpertGrouping g) {
    q.memset(g.counts, 0, static_cast<size_t>(experts) * sizeof(int32_t));
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(pairs)), [=](sycl::id<1> i) {
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(g.counts[expert_of(idx, static_cast<int64_t>(i[0]), experts)])
            .fetch_add(1);
    });
    syclex::single_task(q, [=]() {
        int32_t sum = 0;
        int32_t active = 0;
        for (int64_t e = 0; e < experts; ++e) {
            g.offsets[e] = sum;
            g.cursor[e] = sum;
            if (g.counts[e] > 0) g.active[active++] = static_cast<int32_t>(e);
            sum += g.counts[e];
        }
        g.offsets[experts] = sum;
        *g.n_active = active;
    });
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(pairs)), [=](sycl::id<1> i) {
        const auto p = static_cast<int64_t>(i[0]);
        const int32_t pos = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>(g.cursor[expert_of(idx, p, experts)])
                                .fetch_add(1);
        g.order[pos] = static_cast<int32_t>(p);
    });
}

// One work-group per (active expert, row block); it walks the expert's pairs kGemvMaxRows at a time.
template <bool Q4, bool AWQ>
void expert_quant_grouped_t(sycl::queue& q, const float* x, bool per_slot, QuantWeight w, float* out, int64_t slots,
                            int64_t n, int64_t k, ExpertGrouping g, int64_t max_active) {
    constexpr int M = static_cast<int>(kGemvMaxRows);
    using Shape = QuantGemvShape<Q4, M, AWQ>;
    constexpr int kQRows = Shape::kRows;
    constexpr int64_t kRowsPerWg = kQSgPerWg * kQRows;
    const int64_t blocks = ceil_div(n, kRowsPerWg);
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(max_active * blocks * kQSgPerWg * kSubGroup),
                                           kQSgPerWg * kSubGroup),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
        const int64_t a = static_cast<int64_t>(it.get_group(0)) / blocks;
        if (a >= *g.n_active) return;  // uniform across the work-group
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int64_t block = static_cast<int64_t>(it.get_group(0)) % blocks;
        const int64_t row0 = (block * kQSgPerWg + sg.get_group_linear_id()) * kQRows;
        const int64_t e = g.active[a];
        const int64_t begin = g.offsets[e];
        const int64_t end = g.offsets[e + 1];
        int64_t rows[kQRows];
        for (int r = 0; r < kQRows; ++r)
            rows[r] = e * n + std::min<int64_t>(row0 + r, n - 1);
        for (int64_t c = begin; c < end; c += M) {
            int64_t ps[M];
            const float* xs[M];
            for (int mm = 0; mm < M; ++mm) {
                ps[mm] = g.order[std::min<int64_t>(c + mm, end - 1)];
                xs[mm] = x + (per_slot ? ps[mm] : ps[mm] / slots) * k;
            }
            float acc[kQRows][M] = {};
            quant_rows_dot<Q4, AWQ, M, kQRows, Shape::kBytes>(w, rows, xs, lane, k, acc);
            for (int r = 0; r < kQRows; ++r) {
                for (int mm = 0; mm < M; ++mm) {
                    const float sum = sycl::reduce_over_group(sg, acc[r][mm], sycl::plus<float>());
                    if (lane == 0 && c + mm < end && row0 + r < n) out[ps[mm] * n + row0 + r] = sum;
                }
            }
        }
    });
}

// Dequantized value of column c of row r.
inline float dequant_at(const QuantWeight& w, int64_t r, int64_t c, int64_t k) {
    const int64_t sg = r * (k / w.group) + c / w.group;
    const float s = static_cast<float>(w.scales[sg]);
    const float in = w.input_scales != nullptr ? w.input_scales[c] : 1.0f;
    if (w.q4) {
        const uint8_t byte = w.codes[(r * k + c) / 2];
        const int code = (c & 1) == 0 ? (byte & 0x0F) : (byte >> 4);
        return (s * static_cast<float>(code) + static_cast<float>(w.biases[sg])) * in;
    }
    return s * static_cast<float>(static_cast<int8_t>(w.codes[r * k + c])) * in;
}

template <typename W, int M>
void gemv_m(sycl::queue& q, const float* x, const W* w, float* y, int64_t n, int64_t k) {
    const bool vec = k % 8 == 0 && reinterpret_cast<uintptr_t>(x) % 32 == 0 &&
                     reinterpret_cast<uintptr_t>(w) % (8 * sizeof(W)) == 0;
    // Under a few hundred rows the sub-groups would not fill the device: split K over a work-group's eight.
    const bool split = n < 512 && k >= 8 * kSubGroup * 8;
    if (vec) {
        split ? gemv_t<W, M, true, 8>(q, x, w, y, n, k) : gemv_t<W, M, true, 1>(q, x, w, y, n, k);
    } else {
        split ? gemv_t<W, M, false, 8>(q, x, w, y, n, k) : gemv_t<W, M, false, 1>(q, x, w, y, n, k);
    }
}

}  // namespace

template <typename Y>
void unary(sycl::queue& q, UnaryOp op, const float* x, Y* y, int64_t n, float a, float b) {
    switch (op) {
        case UnaryOp::Scale: return unary_t<UnaryOp::Scale, Y>(q, x, y, n, a, b);
        case UnaryOp::Tanh: return unary_t<UnaryOp::Tanh, Y>(q, x, y, n, a, b);
        case UnaryOp::GeluTanh: return unary_t<UnaryOp::GeluTanh, Y>(q, x, y, n, a, b);
        case UnaryOp::Softcap: return unary_t<UnaryOp::Softcap, Y>(q, x, y, n, a, b);
        case UnaryOp::Sigmoid: return unary_t<UnaryOp::Sigmoid, Y>(q, x, y, n, a, b);
        case UnaryOp::Silu: return unary_t<UnaryOp::Silu, Y>(q, x, y, n, a, b);
        case UnaryOp::Softplus: return unary_t<UnaryOp::Softplus, Y>(q, x, y, n, a, b);
        case UnaryOp::Clamp: return unary_t<UnaryOp::Clamp, Y>(q, x, y, n, a, b);
    }
}

template <typename X, typename Y>
void act_mul(sycl::queue& q, UnaryOp act, const X* x, const X* b, Y* y, int64_t n) {
    switch (act) {
        case UnaryOp::Silu: return act_mul_t<UnaryOp::Silu, X, Y>(q, x, b, y, n);
        case UnaryOp::GeluTanh: return act_mul_t<UnaryOp::GeluTanh, X, Y>(q, x, b, y, n);
        default: return act_mul_t<UnaryOp::Sigmoid, X, Y>(q, x, b, y, n);
    }
}

template <typename Y>
void unary_bias(sycl::queue& q, UnaryOp act, const float* x, const float* bias, Y* y, int64_t n, int64_t period) {
    switch (act) {
        case UnaryOp::Silu: return unary_bias_t<UnaryOp::Silu, Y>(q, x, bias, y, n, period);
        case UnaryOp::GeluTanh: return unary_bias_t<UnaryOp::GeluTanh, Y>(q, x, bias, y, n, period);
        default: return unary_bias_t<UnaryOp::Sigmoid, Y>(q, x, bias, y, n, period);
    }
}

template <typename Y>
void binary(sycl::queue& q, BinaryOp op, const float* a, const float* b, Y* y, int64_t n, const Broadcast& shape) {
    switch (op) {
        case BinaryOp::Add: return binary_t<BinaryOp::Add, Y>(q, a, b, y, n, shape);
        case BinaryOp::Mul: return binary_t<BinaryOp::Mul, Y>(q, a, b, y, n, shape);
        case BinaryOp::Div: return binary_t<BinaryOp::Div, Y>(q, a, b, y, n, shape);
    }
}

template <typename Y>
void binary_same(sycl::queue& q, BinaryOp op, const float* a, const float* b, Y* y, int64_t n) {
    switch (op) {
        case BinaryOp::Add: return binary_same_t<BinaryOp::Add, Y>(q, a, b, y, n);
        case BinaryOp::Mul: return binary_same_t<BinaryOp::Mul, Y>(q, a, b, y, n);
        case BinaryOp::Div: return binary_same_t<BinaryOp::Div, Y>(q, a, b, y, n);
    }
}

template <typename Y>
void binary_period(sycl::queue& q, BinaryOp op, const float* a, const float* b, Y* y, int64_t n, int64_t period) {
    switch (op) {
        case BinaryOp::Add: return binary_period_t<BinaryOp::Add, Y>(q, a, b, y, n, period);
        case BinaryOp::Mul: return binary_period_t<BinaryOp::Mul, Y>(q, a, b, y, n, period);
        case BinaryOp::Div: return binary_period_t<BinaryOp::Div, Y>(q, a, b, y, n, period);
    }
}

void softmax(sycl::queue& q, const float* x, float* y, int64_t rows, int64_t width) {
    const size_t wg = width >= 256 ? 256 : 64;
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(rows) * wg, wg), [=](sycl::nd_item<1> it) {
        const int64_t r = static_cast<int64_t>(it.get_group(0));
        const float* xr = x + r * width;
        float* yr = y + r * width;
        const int64_t lid = static_cast<int64_t>(it.get_local_id(0));
        float peak = -std::numeric_limits<float>::infinity();
        for (int64_t i = lid; i < width; i += static_cast<int64_t>(wg))
            peak = sycl::fmax(peak, xr[i]);
        peak = sycl::reduce_over_group(it.get_group(), peak, sycl::maximum<float>());
        float sum = 0.0f;
        for (int64_t i = lid; i < width; i += static_cast<int64_t>(wg))
            sum += sycl::exp(xr[i] - peak);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float inv = 1.0f / sum;
        for (int64_t i = lid; i < width; i += static_cast<int64_t>(wg))
            yr[i] = sycl::exp(xr[i] - peak) * inv;
    });
}

// Pick j of a row: the largest (value, then lowest index) strictly after pick j - 1 in that order. Few wide rows (a
// drafter's centroids) take a work-group each, a pick per pair of group reductions; many narrow rows (MoE routers,
// one per token) a work-item each. Very wide rows (a shortlist of a vocabulary) first keep the best of each block of
// kTopkBlock elements, and one sub-group then picks among the blocks, replacing a picked block's best by its next: a
// pick reads the block bests and one block instead of the row, and reduces within the sub-group (no barrier).
constexpr int64_t kTopkGroupWidth = 512;
constexpr int kTopkGroup = 256;
constexpr int64_t kTopkWideWidth = 16384;
constexpr int64_t kTopkBlock = 256;
constexpr int kTopkLoad = 16;  // block bests per load

// Block bests per lane, whole loads: lane l owns blocks [l * per, (l + 1) * per), -inf past the last block.
int64_t topk_blocks_per_lane(int64_t width) {
    return ceil_div(ceil_div(ceil_div(width, kTopkBlock), kSubGroup), kTopkLoad) * kTopkLoad;
}

bool topk_wide(int64_t rows, int64_t width) {
    return width >= kTopkWideWidth && rows <= 64 && width < std::numeric_limits<int32_t>::max();
}

void topk_wide_rows(sycl::queue& q, const float* x, float* values, int32_t* indices, int64_t rows, int64_t width,
                    int64_t k, void* scratch) {
    const int64_t per = topk_blocks_per_lane(width);
    const int64_t slots = per * kSubGroup;  // of block bests per row
    auto* best_v = static_cast<float*>(scratch);
    auto* best_i = reinterpret_cast<int32_t*>(best_v + rows * slots);
    constexpr int32_t kNone = std::numeric_limits<int32_t>::max();
    constexpr float kLowest = -std::numeric_limits<float>::infinity();
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(rows), static_cast<size_t>(slots)), [=](sycl::item<2> it) {
        const auto r = static_cast<int64_t>(it[0]);
        const auto b = static_cast<int64_t>(it[1]);
        float v = kLowest;
        int32_t at = kNone;
        for (int64_t e = b * kTopkBlock; e < std::min(width, (b + 1) * kTopkBlock); ++e) {
            if (at == kNone || x[r * width + e] > v) {
                v = x[r * width + e];
                at = static_cast<int32_t>(e);
            }
        }
        best_v[r * slots + b] = v;
        best_i[r * slots + b] = at;
    });
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(rows * kSubGroup), kSubGroup),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
        using Load = sycl::vec<float, kTopkLoad>;
        const sycl::sub_group sg = it.get_sub_group();
        const auto r = static_cast<int64_t>(it.get_group(0));
        const auto lane = static_cast<int64_t>(it.get_local_id(0));
        const float* xr = x + r * width;
        float* bv = best_v + r * slots;
        int32_t* bi = best_i + r * slots;
        for (int64_t j = 0; j < k; ++j) {
            // the best block: the largest best, the lowest block among equals (its elements have the lower indices)
            float top = kLowest;
            int64_t top_b = slots;
            for (int64_t c = 0; c < per; c += kTopkLoad) {
                const Load v = *reinterpret_cast<const Load*>(bv + lane * per + c);
                #pragma unroll
                for (int e = 0; e < kTopkLoad; ++e) {
                    if (top_b == slots || v[e] > top) {
                        top = v[e];
                        top_b = lane * per + c + e;
                    }
                }
            }
            const float v = sycl::reduce_over_group(sg, top, sycl::maximum<float>());
            const int64_t b = sycl::reduce_over_group(sg, top == v ? top_b : slots, sycl::minimum<int64_t>());
            const int32_t i = bi[b];
            if (lane == 0) {
                values[r * k + j] = v;
                indices[r * k + j] = i;
            }
            // that block's best among what is left: after (v, i) in the picking order
            float next = kLowest;
            int32_t next_i = kNone;
            for (int64_t e = b * kTopkBlock + lane; e < std::min(width, (b + 1) * kTopkBlock); e += kSubGroup) {
                const float ve = xr[e];
                const bool left = ve < v || (ve == v && e > i);
                if (left && (next_i == kNone || ve > next)) {
                    next = ve;
                    next_i = static_cast<int32_t>(e);
                }
            }
            const int32_t any = sycl::reduce_over_group(sg, next_i, sycl::minimum<int32_t>());
            const float nv = sycl::reduce_over_group(sg, next_i == kNone ? kLowest : next, sycl::maximum<float>());
            const int32_t ni =
                sycl::reduce_over_group(sg, next_i != kNone && next == nv ? next_i : kNone, sycl::minimum<int32_t>());
            if (lane == 0) {
                bv[b] = any == kNone ? kLowest : nv;
                bi[b] = any == kNone ? kNone : ni;
            }
        }
    });
}

size_t topk_scratch(int64_t rows, int64_t width) {
    if (!topk_wide(rows, width)) return 0;
    return static_cast<size_t>(rows * topk_blocks_per_lane(width) * kSubGroup) * (sizeof(float) + sizeof(int32_t));
}

void topk(sycl::queue& q, const float* x, float* values, int32_t* indices, int64_t rows, int64_t width, int64_t k,
          void* scratch) {
    if (topk_wide(rows, width)) {
        XS_CHECK(scratch != nullptr && reinterpret_cast<uintptr_t>(scratch) % 64 == 0, "sycl topk: no scratch");
        topk_wide_rows(q, x, values, indices, rows, width, k, scratch);
        return;
    }
    if (width >= kTopkGroupWidth && rows <= 64) {
        syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(rows * kTopkGroup), kTopkGroup),
                          [=](sycl::nd_item<1> it) {
            const auto group = it.get_group();
            const auto r = static_cast<int64_t>(it.get_group(0));
            const auto lane = static_cast<int64_t>(it.get_local_id(0));
            const float* xr = x + r * width;
            float prev_v = std::numeric_limits<float>::infinity();
            int64_t prev_i = -1;
            for (int64_t j = 0; j < k; ++j) {
                float best_v = -std::numeric_limits<float>::infinity();
                int64_t best_i = width;
                for (int64_t i = lane; i < width; i += kTopkGroup) {
                    const float v = xr[i];
                    const bool after_prev = v < prev_v || (v == prev_v && i > prev_i);
                    if (after_prev && (best_i == width || v > best_v)) {
                        best_v = v;
                        best_i = i;
                    }
                }
                const float v = sycl::reduce_over_group(group, best_v, sycl::maximum<float>());
                const int64_t i = sycl::reduce_over_group(group, best_i < width && best_v == v ? best_i : width,
                                                          sycl::minimum<int64_t>());
                if (lane == 0) {
                    values[r * k + j] = v;
                    indices[r * k + j] = static_cast<int32_t>(i);
                }
                prev_v = v;
                prev_i = i;
            }
        });
        return;
    }
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> idx) {
        const int64_t r = static_cast<int64_t>(idx[0]);
        const float* xr = x + r * width;
        float prev_v = std::numeric_limits<float>::infinity();
        int64_t prev_i = -1;
        for (int64_t j = 0; j < k; ++j) {
            float best_v = -std::numeric_limits<float>::infinity();
            int64_t best_i = -1;
            for (int64_t i = 0; i < width; ++i) {
                const float v = xr[i];
                const bool after_prev = v < prev_v || (v == prev_v && i > prev_i);
                if (after_prev && (best_i < 0 || v > best_v)) {
                    best_v = v;
                    best_i = i;
                }
            }
            values[r * k + j] = best_v;
            indices[r * k + j] = static_cast<int32_t>(best_i);
            prev_v = best_v;
            prev_i = best_i;
        }
    });
}

// SignMatMul. x is rounded to kSignBits-bit integers of its largest magnitude, and each bit of those integers over 32
// columns is one word (a bit plane): a row's sum over the columns its signs keep positive is then a popcount per plane
// and word, three operations per plane for 32 columns, where selecting and adding floats takes two per column and
// binds the kernel by its arithmetic instead of the memory. With S the sum of the integers and P their sum over the
// positive columns, x . (+-scale) = scale * step * (2 P - S). A lane owns a row and loads its signs 16 words (a cache
// line) at a time.
constexpr int kSignBits = 12;
constexpr int kSignPrepGroup = 256;
constexpr int kSignWords = 16;  // sign words per load

struct SignRow {  // at the start of the scratch, before the planes [kSignBits][words rounded up to kSignWords]
    float step;
    int32_t sum;
};

int64_t sign_plane_words(int64_t k) {
    return ceil_div(k / 32, kSignWords) * kSignWords;
}

// The bit planes of one row of x, its step and the sum of its integers.
void sign_planes(sycl::queue& q, const float* x, SignRow* header, uint32_t* planes, int64_t k) {
    const int64_t words = k / 32;
    const int64_t stride = sign_plane_words(k);
    syclex::nd_launch(q, sycl::nd_range<1>(kSignPrepGroup, kSignPrepGroup), [=](sycl::nd_item<1> it) {
        const auto group = it.get_group();
        const auto lane = static_cast<int64_t>(it.get_local_id(0));
        float peak = 0.0f;
        for (int64_t c = lane; c < k; c += kSignPrepGroup) peak = sycl::fmax(peak, sycl::fabs(x[c]));
        peak = sycl::reduce_over_group(group, peak, sycl::maximum<float>());
        const auto top = static_cast<float>((1 << (kSignBits - 1)) - 1);
        const float to_int = peak > 0.0f ? top / peak : 0.0f;
        int32_t sum = 0;
        for (int64_t w = lane; w < stride; w += kSignPrepGroup) {
            uint32_t plane[kSignBits] = {};
            if (w < words) {
                for (int c = 0; c < 32; ++c) {
                    const auto v = static_cast<int32_t>(sycl::rint(x[w * 32 + c] * to_int));
                    sum += v;
                    for (int p = 0; p < kSignBits; ++p) plane[p] |= ((static_cast<uint32_t>(v) >> p) & 1u) << c;
                }
            }
            for (int p = 0; p < kSignBits; ++p) planes[p * stride + w] = plane[p];
        }
        sum = sycl::reduce_over_group(group, sum, sycl::plus<int32_t>());
        if (lane == 0) {
            header->step = peak / top;
            header->sum = sum;
        }
    });
}

// WIDE: every row of signs starts on a cache line and is whole loads long.
template <bool WIDE>
void sign_scores(sycl::queue& q, const uint32_t* signs, const float* scales, const SignRow* header,
                 const uint32_t* planes, float* y, int64_t n, int64_t k) {
    const int64_t words = k / 32;
    const int64_t stride = sign_plane_words(k);
    const auto lanes = static_cast<size_t>(ceil_div(n, kSubGroup) * kSubGroup);
    syclex::nd_launch(q, sycl::nd_range<1>(lanes, kSubGroup),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
        const auto row = static_cast<int64_t>(it.get_global_id(0));
        if (row >= n) return;
        const uint32_t* s = signs + row * words;
        int32_t count[kSignBits] = {};  // per plane, the set bits of x among the positive columns
        int64_t w = 0;
        if constexpr (WIDE) {
            using Words = sycl::vec<uint32_t, kSignWords>;
            for (; w < words; w += kSignWords) {
                const Words b = *reinterpret_cast<const Words*>(s + w);
                #pragma unroll
                for (int p = 0; p < kSignBits; ++p) {
                    const Words hits = sycl::popcount(b & *reinterpret_cast<const Words*>(planes + p * stride + w));
                    #pragma unroll
                    for (int j = 0; j < kSignWords; ++j) count[p] += static_cast<int32_t>(hits[j]);
                }
            }
        }
        for (; w < words; ++w) {
            for (int p = 0; p < kSignBits; ++p)
                count[p] += static_cast<int32_t>(sycl::popcount(s[w] & planes[p * stride + w]));
        }
        int32_t positive = -(count[kSignBits - 1] << (kSignBits - 1));  // two's complement: the top plane is the sign
        for (int p = 0; p + 1 < kSignBits; ++p) positive += count[p] << p;
        y[row] = scales[row] * header->step * static_cast<float>(2 * positive - header->sum);
    });
}

size_t sign_matmul_scratch(int64_t k) {
    return 64 + static_cast<size_t>(kSignBits * sign_plane_words(k)) * sizeof(uint32_t);
}

void sign_matmul(sycl::queue& q, const float* x, const uint32_t* signs, const float* scales, float* y, int64_t m,
                 int64_t n, int64_t k, void* scratch) {
    XS_CHECK(k % 32 == 0 && k < (int64_t{1} << 19), "sign_matmul: K ", k, " is not a multiple of 32 under 2^19");
    auto* header = static_cast<SignRow*>(scratch);
    auto* planes = reinterpret_cast<uint32_t*>(static_cast<char*>(scratch) + 64);
    const bool wide = reinterpret_cast<uintptr_t>(signs) % 64 == 0 && (k / 32) % kSignWords == 0 &&
                      reinterpret_cast<uintptr_t>(planes) % 64 == 0;
    for (int64_t r = 0; r < m; ++r) {
        sign_planes(q, x + r * k, header, planes, k);
        wide ? sign_scores<true>(q, signs, scales, header, planes, y + r * n, n, k)
             : sign_scores<false>(q, signs, scales, header, planes, y + r * n, n, k);
    }
}

void prev_row(sycl::queue& q, const float* x, float* y, int64_t rows, int64_t width) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(rows * width)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        y[i] = i < width ? 0.0f : x[i - width];
    });
}

void reduce_sum(sycl::queue& q, const float* x, float* y, int64_t outer, int64_t dim, int64_t inner) {
    // A long axis (a row's width): a work-group per output, each lane summing a stride of the axis.
    constexpr int64_t kLongAxis = 256;
    constexpr int64_t kWg = 4 * kSubGroup;
    if (dim >= kLongAxis) {
        syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(outer * inner * kWg), static_cast<size_t>(kWg)),
                          [=](sycl::nd_item<1> it) {
            const int64_t out = static_cast<int64_t>(it.get_group(0));
            const int64_t o = out / inner;
            const int64_t i = out % inner;
            const int64_t lane = static_cast<int64_t>(it.get_local_id(0));
            float sum = 0.0f;
            for (int64_t d = lane; d < dim; d += kWg) sum += x[(o * dim + d) * inner + i];
            const float total = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            if (lane == 0) y[out] = total;
        });
        return;
    }
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(outer), static_cast<size_t>(inner)), [=](sycl::item<2> it) {
        const int64_t o = static_cast<int64_t>(it[0]);
        const int64_t i = static_cast<int64_t>(it[1]);
        float sum = 0.0f;
        for (int64_t d = 0; d < dim; ++d)
            sum += x[(o * dim + d) * inner + i];
        y[o * inner + i] = sum;
    });
}

// One sub-group per kGemvRows output rows of one (token, slot): the GEMV of gemv_t on the slot's expert.
template <typename W, typename I, bool Vec>
void expert_matmul_t(sycl::queue& q, const float* x, bool per_slot, const W* w, const I* idx, float* out,
                     int64_t tokens, int64_t slots, int64_t experts, int64_t n, int64_t k) {
    constexpr int64_t kRowsPerWg = kGemvSgPerWg * kGemvRows;
    const int64_t blocks = ceil_div(n, kRowsPerWg);
    const int64_t items = tokens * slots;
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(items * blocks * kGemvSgPerWg * kSubGroup),
                                           kGemvSgPerWg * kSubGroup),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int64_t item = static_cast<int64_t>(it.get_group(0)) / blocks;
        const int64_t block = static_cast<int64_t>(it.get_group(0)) % blocks;
        const int64_t row0 = (block * kGemvSgPerWg + sg.get_group_linear_id()) * kGemvRows;
        int64_t e = static_cast<int64_t>(idx[item]);
        if (e < 0 || e >= experts) e = 0;
        const float* xr = x + (per_slot ? item : item / slots) * k;
        const W* we = w + e * n * k;
        float acc[kGemvRows];
        int64_t rows[kGemvRows];
        for (int r = 0; r < kGemvRows; ++r) {
            acc[r] = 0.0f;
            rows[r] = std::min<int64_t>(row0 + r, n - 1);
        }
        if constexpr (Vec) {
            for (int64_t k0 = static_cast<int64_t>(lane) * 8; k0 < k; k0 += kSubGroup * 8) {
                const sycl::vec<float, 8> xv = *reinterpret_cast<const sycl::vec<float, 8>*>(xr + k0);
                for (int r = 0; r < kGemvRows; ++r) {
                    const sycl::vec<float, 8> p =
                        reinterpret_cast<const sycl::vec<W, 8>*>(we + rows[r] * k + k0)->template convert<float>() * xv;
                    acc[r] += ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
                }
            }
        } else {
            for (int64_t kk = lane; kk < k; kk += kSubGroup)
                for (int r = 0; r < kGemvRows; ++r)
                    acc[r] += static_cast<float>(we[rows[r] * k + kk]) * xr[kk];
        }
        for (int r = 0; r < kGemvRows; ++r) {
            const float sum = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
            if (lane == 0 && row0 + r < n) out[item * n + row0 + r] = sum;
        }
    });
}

template <typename W, typename I>
void expert_matmul(sycl::queue& q, const float* x, bool per_slot, const W* w, const I* idx, float* out, int64_t tokens,
                   int64_t slots, int64_t experts, int64_t n, int64_t k) {
    const bool vec = k % 8 == 0 && reinterpret_cast<uintptr_t>(x) % 32 == 0 &&
                     reinterpret_cast<uintptr_t>(w) % (8 * sizeof(W)) == 0;
    if (vec) {
        expert_matmul_t<W, I, true>(q, x, per_slot, w, idx, out, tokens, slots, experts, n, k);
    } else {
        expert_matmul_t<W, I, false>(q, x, per_slot, w, idx, out, tokens, slots, experts, n, k);
    }
}

template void expert_matmul<half, int32_t>(sycl::queue&, const float*, bool, const half*, const int32_t*, float*,
                                           int64_t, int64_t, int64_t, int64_t, int64_t);
template void expert_matmul<half, int64_t>(sycl::queue&, const float*, bool, const half*, const int64_t*, float*,
                                           int64_t, int64_t, int64_t, int64_t, int64_t);
template void expert_matmul<float, int32_t>(sycl::queue&, const float*, bool, const float*, const int32_t*, float*,
                                            int64_t, int64_t, int64_t, int64_t, int64_t);
template void expert_matmul<float, int64_t>(sycl::queue&, const float*, bool, const float*, const int64_t*, float*,
                                            int64_t, int64_t, int64_t, int64_t, int64_t);

template <typename Y>
void rms_norm(sycl::queue& q, const float* x, const float* w, const float* bias, Y* y, int64_t rows, int64_t width,
              float eps) {
    const size_t wg = width >= 1024 ? 512 : (width >= 256 ? 256 : 128);
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(rows) * wg, wg), [=](sycl::nd_item<1> it) {
        const int64_t r = static_cast<int64_t>(it.get_group(0));
        const float* xr = x + r * width;
        Y* yr = y + r * width;
        float sum = 0.0f;
        for (int64_t i = static_cast<int64_t>(it.get_local_id(0)); i < width; i += static_cast<int64_t>(wg))
            sum += xr[i] * xr[i];
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float inv = 1.0f / sycl::sqrt(sum / static_cast<float>(width) + eps);
        for (int64_t i = static_cast<int64_t>(it.get_local_id(0)); i < width; i += static_cast<int64_t>(wg))
            yr[i] = static_cast<Y>((w != nullptr ? xr[i] * inv * w[i] : xr[i] * inv) + (bias != nullptr ? bias[i] : 0.0f));
    });
}

template <typename Y>
void layer_norm(sycl::queue& q, const float* x, const float* shift, const float* w, const float* b, Y* y, int64_t rows,
                int64_t width, float eps) {
    const size_t wg = width >= 1024 ? 512 : (width >= 256 ? 256 : 128);
    // no shift: a row that is always there (w), times zero
    const float* sp = shift != nullptr ? shift : w;
    const float on = shift != nullptr ? 1.0f : 0.0f;
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(rows) * wg, wg), [=](sycl::nd_item<1> it) {
        const int64_t r = static_cast<int64_t>(it.get_group(0));
        const float* xr = x + r * width;
        Y* yr = y + r * width;
        const auto lid = static_cast<int64_t>(it.get_local_id(0));
        float sum = 0.0f;
        for (int64_t i = lid; i < width; i += static_cast<int64_t>(wg))
            sum += xr[i] + on * sp[i];
        const float mean = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>()) / static_cast<float>(width);
        float sq = 0.0f;
        for (int64_t i = lid; i < width; i += static_cast<int64_t>(wg)) {
            const float d = xr[i] + on * sp[i] - mean;
            sq += d * d;
        }
        const float var = sycl::reduce_over_group(it.get_group(), sq, sycl::plus<float>()) / static_cast<float>(width);
        const float inv = 1.0f / sycl::sqrt(var + eps);
        for (int64_t i = lid; i < width; i += static_cast<int64_t>(wg))
            yr[i] = static_cast<Y>((xr[i] + on * sp[i] - mean) * inv * w[i] + b[i]);
    });
}

template <typename Y>
void add_mul(sycl::queue& q, const float* a, const float* b, const float* c, Y* y, int64_t n, int64_t period) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
        y[i] = static_cast<Y>(a[i] + b[i] * c[static_cast<int64_t>(i[0]) % period]);
    });
}

template <typename Y>
void add3(sycl::queue& q, const float* a, const float* b, const float* c, Y* y, int64_t n, int64_t period) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
        y[i] = static_cast<Y>(a[i] + (b[i] + c[static_cast<int64_t>(i[0]) % period]));
    });
}

#define XS_INSTANTIATE_OUT(Y)                                                                                         \
    template void unary<Y>(sycl::queue&, UnaryOp, const float*, Y*, int64_t, float, float);                           \
    template void act_mul<float, Y>(sycl::queue&, UnaryOp, const float*, const float*, Y*, int64_t);                  \
    template void act_mul<half, Y>(sycl::queue&, UnaryOp, const half*, const half*, Y*, int64_t);                     \
    template void unary_bias<Y>(sycl::queue&, UnaryOp, const float*, const float*, Y*, int64_t, int64_t);             \
    template void add3<Y>(sycl::queue&, const float*, const float*, const float*, Y*, int64_t, int64_t);              \
    template void binary<Y>(sycl::queue&, BinaryOp, const float*, const float*, Y*, int64_t, const Broadcast&);      \
    template void binary_same<Y>(sycl::queue&, BinaryOp, const float*, const float*, Y*, int64_t);                   \
    template void binary_period<Y>(sycl::queue&, BinaryOp, const float*, const float*, Y*, int64_t, int64_t);        \
    template void rms_norm<Y>(sycl::queue&, const float*, const float*, const float*, Y*, int64_t, int64_t, float);   \
    template void layer_norm<Y>(sycl::queue&, const float*, const float*, const float*, const float*, Y*, int64_t, \
                                int64_t, float);                                                                    \
    template void add_mul<Y>(sycl::queue&, const float*, const float*, const float*, Y*, int64_t, int64_t);
XS_INSTANTIATE_OUT(float)
XS_INSTANTIATE_OUT(half)
#undef XS_INSTANTIATE_OUT

namespace {

// One work-item per unit U of the output; the part is found by its byte range.
template <typename U>
void concat_t(sycl::queue& q, const ConcatParts& parts, void* out, int64_t outer) {
    struct Units {
        const U* src[kConcatParts];
        int64_t start[kConcatParts + 1];  // a part's first unit in an output row
    } u{};
    for (int k = 0; k < parts.count; ++k) {
        u.src[k] = static_cast<const U*>(parts.src[k]);
        u.start[k + 1] = u.start[k] + static_cast<int64_t>(parts.bytes[k] / sizeof(U));
    }
    const int count = parts.count;
    const int64_t row = u.start[count];
    U* dst = static_cast<U*>(out);
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(outer * row)), [=](sycl::id<1> idx) {
        const int64_t i = static_cast<int64_t>(idx[0]);
        const int64_t o = i / row;
        const int64_t c = i % row;
        int k = 0;
        while (k + 1 < count && c >= u.start[k + 1]) ++k;
        const int64_t width = u.start[k + 1] - u.start[k];
        dst[i] = u.src[k][o * width + c - u.start[k]];
    });
}

}  // namespace

void concat(sycl::queue& q, const ConcatParts& parts, void* out, int64_t outer) {
    size_t all = reinterpret_cast<uintptr_t>(out);
    for (int k = 0; k < parts.count; ++k) all |= parts.bytes[k] | reinterpret_cast<uintptr_t>(parts.src[k]);
    if (all % 16 == 0) return concat_t<sycl::vec<uint32_t, 4>>(q, parts, out, outer);
    if (all % 4 == 0) return concat_t<uint32_t>(q, parts, out, outer);
    if (all % 2 == 0) return concat_t<uint16_t>(q, parts, out, outer);
    concat_t<uint8_t>(q, parts, out, outer);
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
void gather_bits(sycl::queue& q, const uint32_t* table, const I* ids, uint32_t* out, int64_t n, int64_t width) {
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(n), static_cast<size_t>(width)), [=](sycl::item<2> it) {
        const int64_t r = static_cast<int64_t>(it[0]);
        const int64_t c = static_cast<int64_t>(it[1]);
        out[r * width + c] = table[static_cast<int64_t>(ids[r]) * width + c];
    });
}
template void gather_bits(sycl::queue&, const uint32_t*, const int32_t*, uint32_t*, int64_t, int64_t);
template void gather_bits(sycl::queue&, const uint32_t*, const int64_t*, uint32_t*, int64_t, int64_t);

// A work-item per token and rotated pair (or element past the rotated part): the angle's cosine and sine are
// computed once and applied to every head. (An item per element computed them twice a pair for each head, with a
// handful of 64-bit divisions: at a prefill's or a policy's sizes that cost more than reading and writing x.)
template <typename I, typename Y>
void rope(sycl::queue& q, const float* x, const I* positions, const float* inv_freq, Y* y, int64_t tokens,
          int64_t heads, int64_t dim, int64_t rotary, int64_t sections) {
    const auto width = static_cast<int>(rotary / sections);
    const int half_w = width / 2;
    const auto pairs = static_cast<int>(rotary / 2);
    const auto rotated = static_cast<int>(rotary);
    const auto h_n = static_cast<int>(heads);
    const auto d_n = static_cast<int>(dim);
    const sycl::range<2> items(static_cast<size_t>(tokens), static_cast<size_t>(pairs + (dim - rotary)));
    syclex::parallel_for(q, items, [=](sycl::id<2> idx) {
        const auto t = static_cast<int64_t>(idx[0]);
        const auto j = static_cast<int>(idx[1]);
        const float* xt = x + t * (h_n * d_n);
        Y* yt = y + t * (h_n * d_n);
        if (j >= pairs) {
            const int d = rotated + (j - pairs);
            for (int h = 0; h < h_n; ++h) yt[h * d_n + d] = static_cast<Y>(xt[h * d_n + d]);
            return;
        }
        const int sec = j / half_w;
        const int c0 = j % half_w;
        // f32 angle as Hugging Face computes it (inv_freq @ positions in float32)
        const float angle = static_cast<float>(positions[t * sections + sec]) * inv_freq[c0];
        const float cs = sycl::cos(angle);
        const float sn = sycl::sin(angle);
        const int lo = sec * width + c0;
        const int hi = lo + half_w;
        for (int h = 0; h < h_n; ++h) {
            const float a = xt[h * d_n + lo];
            const float b = xt[h * d_n + hi];
            yt[h * d_n + lo] = static_cast<Y>(a * cs - b * sn);
            yt[h * d_n + hi] = static_cast<Y>(b * cs + a * sn);
        }
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

void copy_parts(sycl::queue& q, const CopyPart* parts, int count) {
    XS_CHECK(count >= 1 && count <= kCopyParts, "sycl copy_parts: ", count, " parts");
    CopyPart p[kCopyParts];
    int64_t end[kCopyParts];  // where each part ends in the launch's range
    int64_t total = 0;
    for (int j = 0; j < kCopyParts; ++j) {
        if (j < count) p[j] = parts[j];
        total += p[j].count;
        end[j] = total;
    }
    const CopyPart p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
    const int64_t e0 = end[0], e1 = end[1], e2 = end[2];
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        if (i < e0) {
            p0.dst[i] = p0.src[i];
        } else if (i < e1) {
            p1.dst[i - e0] = p1.src[i - e0];
        } else if (i < e2) {
            p2.dst[i - e1] = p2.src[i - e1];
        } else {
            p3.dst[i - e2] = p3.src[i - e2];
        }
    });
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
    // a lane per state column with every row in registers where they fit (any number of tokens); else a few tokens
    // with the rows split between sub-groups, or a work-group per head
    if (dv % kSubGroup == 0 && (dk == 64 || dk == 128)) {
        return dk == 64 ? gated_delta_lanes_t<64>(q, qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dv, qk_l2norm_eps)
                        : gated_delta_lanes_t<128>(q, qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dv, qk_l2norm_eps);
    }
    if (tokens <= kDeltaColsMaxTokens && dv % kSubGroup == 0 && dk % kDeltaColsSgs == 0) {
        switch (dk / kDeltaColsSgs) {
            case 8: return gated_delta_cols_t<8>(q, qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dk, dv, qk_l2norm_eps);
            case 16: return gated_delta_cols_t<16>(q, qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dk, dv, qk_l2norm_eps);
            case 32: return gated_delta_cols_t<32>(q, qp, kp, vp, g, beta, state_in, state_out, out, tokens, k_heads, v_heads, dk, dv, qk_l2norm_eps);
            default: break;
        }
    }
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
    attention_rows_dispatch(q, qp, k, v, nullptr, nullptr, out, s, split, part);
}

void attention_rows_i8(sycl::queue& q, const float* qp, const KvI8& kv, float* out, const AttentionShape& s,
                       int64_t split, float* part) {
    attention_rows_dispatch(q, qp, kv.k, kv.v, kv.k_scale, kv.v_scale, out, s, split, part);
}

void quantize_kv_i8(sycl::queue& q, const float* x, int8_t* codes, float* scales, int64_t rows, int64_t width) {
    constexpr int kRowsPerWg = 8;
    const int64_t wgs = ceil_div(rows, kRowsPerWg);
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(wgs * kRowsPerWg * kSubGroup), kRowsPerWg * kSubGroup),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int64_t r = static_cast<int64_t>(it.get_group(0)) * kRowsPerWg + sg.get_group_linear_id();
        if (r >= rows) return;
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const float* xr = x + r * width;
        float amax = 0.0f;
        for (int64_t i = lane; i < width; i += kSubGroup)
            amax = sycl::fmax(amax, sycl::fabs(xr[i]));
        amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        int8_t* cr = codes + r * width;
        for (int64_t i = lane; i < width; i += kSubGroup)
            cr[i] = static_cast<int8_t>(sycl::clamp(sycl::rint(xr[i] * inv), -127.0f, 127.0f));
        if (lane == 0) scales[r] = amax / 127.0f;
    });
}

template <typename CT>
void dequantize_kv_i8(sycl::queue& q, const int8_t* codes, const float* scales, CT* out, int64_t rows, int64_t width) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(rows * width)), [=](sycl::id<1> idx) {
        const int64_t i = static_cast<int64_t>(idx[0]);
        out[i] = static_cast<CT>(static_cast<float>(codes[i]) * scales[i / width]);
    });
}

size_t expert_grouping_bytes(int64_t pairs, int64_t experts) {
    return static_cast<size_t>(4 * experts + 2 + pairs) * sizeof(int32_t);
}

template <typename I>
void expert_matmul_quant(sycl::queue& q, const float* x, bool per_slot, const QuantWeight& w, const I* idx, float* out,
                         int64_t tokens, int64_t slots, int64_t experts, int64_t n, int64_t k, int32_t* grouping) {
    const int64_t pairs = tokens * slots;
    if (pairs <= kExpertPairsDirect || grouping == nullptr) {
        quant_dispatch(w, [&]<bool Q4, bool AWQ>() {
            expert_quant_pairs_t<Q4, AWQ>(q, x, per_slot, w, idx, out, pairs, slots, experts, n, k);
        });
        return;
    }
    const ExpertGrouping g(grouping, experts);
    group_by_expert(q, idx, pairs, experts, g);
    const int64_t max_active = std::min(pairs, experts);
    quant_dispatch(w, [&]<bool Q4, bool AWQ>() {
        expert_quant_grouped_t<Q4, AWQ>(q, x, per_slot, w, out, slots, n, k, g, max_active);
    });
}

template <typename I>
void expert_group(sycl::queue& q, const I* idx, int64_t pairs, int64_t experts, int32_t* grouping) {
    group_by_expert(q, idx, pairs, experts, ExpertGrouping(grouping, experts));
}

template <typename CT>
void expert_gather_rows(sycl::queue& q, const float* x, bool per_slot, int64_t slots, const int32_t* order,
                        int64_t pairs, int64_t k, CT* rows) {
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(pairs), static_cast<size_t>(k)), [=](sycl::item<2> it) {
        const auto i = static_cast<int64_t>(it[0]);
        const auto c = static_cast<int64_t>(it[1]);
        const int64_t p = order[i];
        rows[i * k + c] = static_cast<CT>(x[(per_slot ? p : p / slots) * k + c]);
    });
}

void expert_scatter_rows(sycl::queue& q, const float* rows, const int32_t* order, int64_t pairs, int64_t n, float* out) {
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(pairs), static_cast<size_t>(n)), [=](sycl::item<2> it) {
        const auto i = static_cast<int64_t>(it[0]);
        const auto c = static_cast<int64_t>(it[1]);
        out[static_cast<int64_t>(order[i]) * n + c] = rows[i * n + c];
    });
}

template void expert_group<int32_t>(sycl::queue&, const int32_t*, int64_t, int64_t, int32_t*);
template void expert_group<int64_t>(sycl::queue&, const int64_t*, int64_t, int64_t, int32_t*);
template void expert_gather_rows<half>(sycl::queue&, const float*, bool, int64_t, const int32_t*, int64_t, int64_t, half*);
template void expert_gather_rows<float>(sycl::queue&, const float*, bool, int64_t, const int32_t*, int64_t, int64_t,
                                        float*);

template void expert_matmul_quant<int32_t>(sycl::queue&, const float*, bool, const QuantWeight&, const int32_t*, float*,
                                           int64_t, int64_t, int64_t, int64_t, int64_t, int32_t*);
template void expert_matmul_quant<int64_t>(sycl::queue&, const float*, bool, const QuantWeight&, const int64_t*, float*,
                                           int64_t, int64_t, int64_t, int64_t, int64_t, int32_t*);

template void dequantize_kv_i8<half>(sycl::queue&, const int8_t*, const float*, half*, int64_t, int64_t);
template void dequantize_kv_i8<float>(sycl::queue&, const int8_t*, const float*, float*, int64_t, int64_t);

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

template <typename CT, typename QT>
void attention_pack_q(sycl::queue& q, const QT* qp, CT* packed, const AttentionShape& s, int64_t t0, int64_t tc) {
    const int64_t d_n = s.head_dim;
    const int64_t group = s.q_heads / s.kv_heads;
    const int64_t hq_n = s.q_heads;
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(s.kv_heads * tc * group * d_n)), [=](sycl::id<1> idx) {
        const int64_t i = static_cast<int64_t>(idx[0]);
        const int64_t d = i % d_n;
        const int64_t g = (i / d_n) % group;
        const int64_t tl = (i / (d_n * group)) % tc;
        const int64_t kh = i / (d_n * group * tc);
        packed[i] = static_cast<CT>(static_cast<float>(qp[((t0 + tl) * hq_n + kh * group + g) * d_n + d]));
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
    const float* key_bias = s.key_bias;
    constexpr size_t kWg = 256;
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(rows) * kWg, kWg), [=](sycl::nd_item<1> it) {
        const int64_t r = static_cast<int64_t>(it.get_group(0));
        const int64_t tl = (r / group) % tc;
        const int64_t pos = past + t0 + tl;
        // visible keys, relative to k_lo: [lo, hi]
        int64_t lo = 0;
        int64_t hi = keys - 1;
        if (causal) hi = std::min<int64_t>(hi, pos - k_lo);
        if (window > 0) lo = std::max<int64_t>(lo, pos - window + 1 - k_lo);
        const float* sr = scores + r * keys;
        CT* pr = probs + r * keys;
        const int64_t lid = static_cast<int64_t>(it.get_local_id(0));
        auto score = [&](int64_t j) {
            const float v = sr[j];
            const float c = softcap > 0.0f ? softcap * sycl::tanh(v / softcap) : v;
            return key_bias != nullptr ? c + key_bias[k_lo + j] : c;
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

bool quant_gemv_parts_match(const QuantWeight& a, const QuantWeight& b) {
    return a.q4 == b.q4 && (a.input_scales == nullptr) == (b.input_scales == nullptr);
}

void gemv_quant_parts(sycl::queue& q, const float* x, const QuantGemvPart* parts, int count, int64_t m, int64_t k) {
    XS_CHECK(count >= 1 && count <= kQuantGemvParts, "gemv_quant_parts: ", count, " parts");
    QuantParts qp;
    qp.count = count;
    qp.wg_begin[0] = 0;
    for (int j = 0; j < kQuantGemvParts; ++j) {
        const QuantGemvPart& part = parts[std::min(j, count - 1)];  // unused slots repeat the last part
        XS_CHECK(quant_gemv_parts_match(part.w, parts[0].w), "gemv_quant_parts: parts differ in format");
        qp.w[j] = part.w;
        qp.y[j] = part.y;
        qp.n[j] = part.n;
    }
    quant_dispatch(parts[0].w, [&]<bool Q4, bool AWQ>() { gemv_quant_q<Q4, AWQ>(q, x, qp, m, k); });
}

void gemv_quant(sycl::queue& q, const float* x, const QuantWeight& w, float* y, int64_t m, int64_t n, int64_t k) {
    const QuantGemvPart part{w, y, n};
    gemv_quant_parts(q, x, &part, 1, m, k);
}

// One work item per 8 columns (within one group): neighbouring work items write neighbouring 16 (f16) or 32 bytes, so
// every store of a sub-group covers whole cache lines. (A work item per 32 columns, 64 bytes apart, wrote at half the
// bandwidth on the Arc Pro B60.)
template <typename T, bool Q4>
void dequantize_t(sycl::queue& q, const QuantWeight& w, T* out, int64_t n, int64_t k) {
    const QuantWeight wc = w;
    const int64_t groups = k / w.group;
    const auto group = static_cast<int32_t>(w.group);
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(n), static_cast<size_t>(k / 8)), [=](sycl::item<2> it) {
        const auto r = static_cast<int64_t>(it[0]);
        const auto c0 = static_cast<int32_t>(it[1]) * 8;
        const int64_t sg = r * groups + c0 / group;
        const float s = static_cast<float>(wc.scales[sg]);
        float v[8];
        if constexpr (Q4) {
            const uint32_t c = *reinterpret_cast<const uint32_t*>(wc.codes + (r * k + c0) / 2);
            const float b = static_cast<float>(wc.biases[sg]);
            for (int j = 0; j < 8; ++j) v[j] = s * static_cast<float>((c >> (4 * j)) & 0x0F) + b;
        } else {
            const sycl::vec<int8_t, 8> c = *reinterpret_cast<const sycl::vec<int8_t, 8>*>(wc.codes + r * k + c0);
            for (int j = 0; j < 8; ++j) v[j] = s * static_cast<float>(c[j]);
        }
        sycl::vec<T, 8> o;
        for (int j = 0; j < 8; ++j)
            o[j] = static_cast<T>(wc.input_scales != nullptr ? v[j] * wc.input_scales[c0 + j] : v[j]);
        *reinterpret_cast<sycl::vec<T, 8>*>(out + r * k + c0) = o;
    });
}

template <typename T>
void dequantize(sycl::queue& q, const QuantWeight& w, T* out, int64_t n, int64_t k) {
    if (w.q4) {
        dequantize_t<T, true>(q, w, out, n, k);
    } else {
        dequantize_t<T, false>(q, w, out, n, k);
    }
}

template void dequantize<half>(sycl::queue&, const QuantWeight&, half*, int64_t, int64_t);
template void dequantize<float>(sycl::queue&, const QuantWeight&, float*, int64_t, int64_t);

template <typename I>
void gather_quant(sycl::queue& q, const QuantWeight& table, const I* ids, float* out, int64_t count, int64_t width) {
    const QuantWeight t = table;
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(count), static_cast<size_t>(width)), [=](sycl::item<2> it) {
        const int64_t i = static_cast<int64_t>(it[0]);
        const int64_t c = static_cast<int64_t>(it[1]);
        out[i * width + c] = dequant_at(t, static_cast<int64_t>(ids[i]), c, width);
    });
}

template void gather_quant<int32_t>(sycl::queue&, const QuantWeight&, const int32_t*, float*, int64_t, int64_t);
template void gather_quant<int64_t>(sycl::queue&, const QuantWeight&, const int64_t*, float*, int64_t, int64_t);
template void gather<half, int32_t>(sycl::queue&, const half*, const int32_t*, float*, int64_t, int64_t);
template void gather<half, int64_t>(sycl::queue&, const half*, const int64_t*, float*, int64_t, int64_t);
template void gather<float, int32_t>(sycl::queue&, const float*, const int32_t*, float*, int64_t, int64_t);
template void gather<float, int64_t>(sycl::queue&, const float*, const int64_t*, float*, int64_t, int64_t);
#define XS_INSTANTIATE_ROPE(I, Y)                                                                                     \
    template void rope<I, Y>(sycl::queue&, const float*, const I*, const float*, Y*, int64_t, int64_t, int64_t, int64_t, \
                             int64_t);
XS_INSTANTIATE_ROPE(int32_t, float)
XS_INSTANTIATE_ROPE(int64_t, float)
XS_INSTANTIATE_ROPE(int32_t, half)
XS_INSTANTIATE_ROPE(int64_t, half)
#undef XS_INSTANTIATE_ROPE
template void attention_rows<half>(sycl::queue&, const float*, const half*, const half*, float*, const AttentionShape&,
                                   int64_t, float*);
template void attention_rows<float>(sycl::queue&, const float*, const float*, const float*, float*,
                                    const AttentionShape&, int64_t, float*);
template void attention_pack_q<half, float>(sycl::queue&, const float*, half*, const AttentionShape&, int64_t, int64_t);
template void attention_pack_q<float, float>(sycl::queue&, const float*, float*, const AttentionShape&, int64_t, int64_t);
template void attention_pack_q<half, half>(sycl::queue&, const half*, half*, const AttentionShape&, int64_t, int64_t);
template void attention_pack_q<float, half>(sycl::queue&, const half*, float*, const AttentionShape&, int64_t, int64_t);
template void attention_softmax<half>(sycl::queue&, const float*, half*, const AttentionShape&, int64_t, int64_t, int64_t, int64_t);
template void attention_softmax<float>(sycl::queue&, const float*, float*, const AttentionShape&, int64_t, int64_t, int64_t, int64_t);
template void gemv<half>(sycl::queue&, const float*, const half*, float*, int64_t, int64_t, int64_t);
template void gemv<float>(sycl::queue&, const float*, const float*, float*, int64_t, int64_t, int64_t);

}  // namespace xs::sycl_be
