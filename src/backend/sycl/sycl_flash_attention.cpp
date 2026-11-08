// Fused attention on the matrix engines (XMX through sycl_ext_oneapi_matrix): one pass over the keys with an online
// softmax; scores and probabilities stay in registers.
//
// A sub-group owns a tile of 8 consecutive tokens of one query head and a share of their keys (`split` > 1 divides the
// keys between work-groups; attention_combine merges the shares). Per block of BC keys: S = Q K^T with K tiles read
// straight from global memory by bounds-checked 2D block loads (rows past the cache and columns past the tensor read
// as zero), the softmax update on the accumulators (lane = column, element i = row i), P converted to A tiles, then
// O = O * alpha + P V. Q: up to 128 columns it is staged once through local memory, zero-padded to DP columns, and
// held in registers, so any head_dim that is a multiple of 8 works (columns of a K or V tile past head_dim belong to
// the next head: Q's zero padding cancels them in S, and O's columns past head_dim are never written). At 256 the
// accumulators fill the registers, so Q tiles stream from global memory each block (f16 q, head_dim 256).

#include "backend/sycl/sycl_kernels.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace xs::sycl_be {
namespace {

namespace syclex = sycl::ext::oneapi::experimental;
namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace mxi = sycl::ext::intel::experimental::matrix;
namespace intelex = sycl::ext::intel::experimental;

constexpr int kSg = 16;  // the matrix engines' sub-group size (Xe2 and later)
constexpr int kTm = 8;   // rows of an A or accumulator tile
constexpr int kTn = 16;  // columns of a B or accumulator tile
constexpr int kTk = 16;  // depth of a tile (f16)

using Acc = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, kTm, kTn>;
using TileA = mx::joint_matrix<sycl::sub_group, half, mx::use::a, kTm, kTk, mx::layout::row_major>;
using TileKt = mx::joint_matrix<sycl::sub_group, half, mx::use::b, kTk, kTn, mx::layout::col_major>;
using TileV = mx::joint_matrix<sycl::sub_group, half, mx::use::b, kTk, kTn, mx::layout::row_major>;

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

template <typename T>
auto local_ptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

template <typename T>
auto global_ptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

// K, V and streamed Q tiles are read by every sub-group of a head: keep them in L1 (2D block loads skip it otherwise).
using L1Cached = decltype(syclex::properties{intelex::read_hint<
    intelex::cache_control<intelex::cache_mode::cached, syclex::cache_level::L1>,
    intelex::cache_control<intelex::cache_mode::cached, syclex::cache_level::L3>>});
inline syclex::annotated_ptr<half, L1Cached> l1_cached(const half* p) {
    return syclex::annotated_ptr<half, L1Cached>(const_cast<half*>(p));
}

// f(x, row) over an accumulator's elements; a lane holds one column, its element i is row i.
template <typename F>
inline void for_rows(sycl::sub_group sg, Acc& c, F&& f) {
    int i = 0;
    mxi::joint_matrix_apply(sg, c, [&](float& x, size_t, size_t) {
        f(x, i);
        ++i;
    });
}

struct FlashArgs {
    int64_t tiles_per_head = 0;  // ceil(T / 8)
    int64_t tiles = 0;           // per KV head: G * tiles_per_head
    int64_t row_blocks = 0;      // work-groups per KV head and split
    int64_t split = 1;
    int64_t group = 1;
    int64_t q_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t tokens = 0;
    int64_t kv_len = 0;
    int64_t window = 0;
    float scale = 1.0f;
    float softcap = 0.0f;
    bool causal = false;
    const float* key_bias = nullptr;
};

template <int DP, int BC, int SGS, typename QT>
struct FlashKernel {
    static constexpr bool kQRegs = DP <= 128;  // Q held in registers (else streamed per block)
    static constexpr int kWg = SGS * kSg;
    static constexpr int kQs = DP + 8;         // Q row stride in local memory (f16), off the bank period
    static constexpr int kDt = DP / kTn;       // tiles across the head
    static constexpr int kKt = BC / kTn;       // tiles across a key block
    // per sub-group: Q [8, kQs] f16 while loading, then the output [8, DP] f32
    static constexpr size_t kSgFloats = std::max<size_t>(kQRegs ? kTm * kQs / 2 : 0, size_t{kTm} * DP);
    static constexpr size_t kSlmFloats = SGS * kSgFloats;
    static_assert(DP % kTn == 0 && BC % kTn == 0);
    static_assert(kQRegs || std::is_same_v<QT, half>);

    const QT* qp;
    const half* kp;
    const half* vp;
    float* out;
    float* part;
    FlashArgs a;
    sycl::local_accessor<float, 1> slm;

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<kSg>, intelex::grf_size<256>};
    }

    void operator()(sycl::nd_item<1> it) const {
        constexpr float kNegInf = -std::numeric_limits<float>::infinity();
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int sgi = static_cast<int>(sg.get_group_linear_id());
        const auto wg = static_cast<int64_t>(it.get_group(0));
        const int64_t sp = wg % a.split;
        const int64_t rb = (wg / a.split) % a.row_blocks;
        const int64_t hk = wg / (a.split * a.row_blocks);
        const int64_t tile = rb * SGS + sgi;
        if (tile >= a.tiles) return;
        const int64_t h = hk * a.group + tile / a.tiles_per_head;  // query head
        const int64_t t0 = (tile % a.tiles_per_head) * kTm;        // first token
        const int64_t d_n = a.head_dim;
        const int64_t q_stride = a.q_heads * d_n;  // q and out as [T, Hq * D] matrices
        const int64_t past = a.kv_len - a.tokens;
        float* sg_slm = slm.template get_multi_ptr<sycl::access::decorated::no>().get() + sgi * kSgFloats;

        TileA qa[kQRegs ? kDt : 1];
        if constexpr (kQRegs) {
            half* q_s = reinterpret_cast<half*>(sg_slm);
            for (int i = lane; i < kTm * (DP / 8); i += kSg) {
                const int rr = i / (DP / 8);
                const int c = (i % (DP / 8)) * 8;
                sycl::vec<half, 8> val(half(0.0f));
                if (t0 + rr < a.tokens && c < d_n) {
                    const QT* src = qp + (t0 + rr) * q_stride + h * d_n + c;
                    for (int e = 0; e < 8; ++e) val[e] = static_cast<half>(src[e]);
                }
                *reinterpret_cast<sycl::vec<half, 8>*>(q_s + rr * kQs + c) = val;
            }
            sycl::group_barrier(sg);
            for (int i = 0; i < kDt; ++i) mx::joint_matrix_load(sg, qa[i], local_ptr(q_s + i * kTk), kQs);
            sycl::group_barrier(sg);
        }

        // Visible keys of row i (token t0 + i): [row_lo(i), row_hi(i)); the tile's span; its split's share in blocks.
        const int64_t last = std::min<int64_t>(a.tokens - 1, t0 + kTm - 1);
        const int64_t hi = a.causal ? past + last + 1 : a.kv_len;
        const int64_t lo = a.causal && a.window > 0 ? std::max<int64_t>(0, past + t0 - a.window + 1) : 0;
        const int pos0 = static_cast<int>(past + t0);
        const int rows_valid = static_cast<int>(last - t0 + 1);
        const int64_t blocks = ceil_div(hi - lo, BC);
        const int64_t per = ceil_div(blocks, a.split);
        const int64_t b_begin = sp * per;
        const int64_t b_end = std::min(blocks, b_begin + per);

        // Scores in log2 units (exp2 is the hardware's exponential): s2 = log2(e) * softcap(scale * q.k) + bias.
        // Row maxima and rescale factors are uniform across lanes: lane i holds row i's.
        constexpr float kLog2e = 1.4426950408889634f;
        float m2 = kNegInf;
        float ls[kTm];  // this lane's share of each row's denominator
        for (int i = 0; i < kTm; ++i) ls[i] = 0.0f;
        Acc o[kDt];
        for (int i = 0; i < kDt; ++i) mx::joint_matrix_fill(sg, o[i], 0.0f);

        const size_t width = static_cast<size_t>(a.kv_heads * d_n);  // K and V as [kv_len, Hkv * D] matrices
        const size_t height = static_cast<size_t>(a.kv_len);
        const size_t col0 = static_cast<size_t>(hk * d_n);
        const float qk_scale = a.softcap > 0.0f ? a.scale / a.softcap : a.scale * kLog2e;
        const float cap2 = a.softcap * kLog2e;

        for (int64_t b = b_begin; b < b_end; ++b) {
            const int64_t kb = lo + b * BC;
            Acc sc[kKt];
            for (int nt = 0; nt < kKt; ++nt) mx::joint_matrix_fill(sg, sc[nt], 0.0f);
            const auto s_step = [&](int kk, const TileA& q_tile) {
                for (int nt = 0; nt < kKt; ++nt) {
                    TileKt kt;
                    mxi::joint_matrix_load_checked(sg, kt, l1_cached(kp), width, height, width,
                                                   static_cast<size_t>(kb + nt * kTn), col0 + kk * kTk);
                    mx::joint_matrix_mad(sg, sc[nt], q_tile, kt, sc[nt]);
                }
            };
            if constexpr (kQRegs) {
                for (int kk = 0; kk < kDt; ++kk) s_step(kk, qa[kk]);
            } else {
                // Not unrolled: the loads of an unrolled loop are all issued ahead and spill the registers.
#pragma unroll 1
                for (int kk = 0; kk < kDt; ++kk) {
                    mxi::joint_matrix_load_checked(sg, qa[0], l1_cached(reinterpret_cast<const half*>(qp)),
                                                   static_cast<size_t>(q_stride), static_cast<size_t>(a.tokens),
                                                   static_cast<size_t>(q_stride), static_cast<size_t>(t0),
                                                   static_cast<size_t>(h * d_n + kk * kTk));
                    s_step(kk, qa[0]);
                }
            }

            float mb[kTm];
            for (int i = 0; i < kTm; ++i) mb[i] = kNegInf;
            for (int nt = 0; nt < kKt; ++nt) {
                const int j = static_cast<int>(kb) + nt * kTn + lane;
                const bool key_ok = j < a.kv_len;
                const float bias = a.key_bias != nullptr && key_ok ? a.key_bias[j] * kLog2e : 0.0f;
                for_rows(sg, sc[nt], [&](float& x, int i) {
                    float y = x * qk_scale;
                    if (a.softcap > 0.0f) y = cap2 * (1.0f - 2.0f / (sycl::native::exp2(2.0f * kLog2e * y) + 1.0f));
                    bool ok = key_ok && i < rows_valid;
                    if (a.causal) ok = ok && j <= pos0 + i && (a.window <= 0 || j > pos0 + i - a.window);
                    y = ok ? y + bias : kNegInf;
                    x = y;
                    mb[i] = sycl::fmax(mb[i], y);
                });
            }
            float m_blk = kNegInf;  // lane i: row i's block maximum
            for (int i = 0; i < kTm; ++i) {
                const float r = sycl::reduce_over_group(sg, mb[i], sycl::maximum<float>());
                m_blk = lane == i ? r : m_blk;
            }
            const float m_new = sycl::fmax(m2, m_blk);
            const float alpha = m_new == kNegInf ? 1.0f : sycl::native::exp2(m2 - m_new);
            m2 = m_new;
            const bool rescale = sycl::any_of_group(sg, lane < kTm && alpha != 1.0f);
            if (rescale) {
                for (int i = 0; i < kTm; ++i) ls[i] *= sycl::select_from_group(sg, alpha, i);
                for (int dt = 0; dt < kDt; ++dt) {
                    for_rows(sg, o[dt], [&](float& x, int i) { x *= sycl::select_from_group(sg, alpha, i); });
                }
            }
            TileA pa[kKt];
            for (int nt = 0; nt < kKt; ++nt) {
                for_rows(sg, sc[nt], [&](float& x, int i) {
                    const float mi = sycl::select_from_group(sg, m2, i);
                    const float p = mi == kNegInf ? 0.0f : sycl::native::exp2(x - mi);
                    x = p;
                    ls[i] += p;
                });
                mx::joint_matrix_copy(sg, sc[nt], pa[nt]);
            }
            for (int kt = 0; kt < kKt; ++kt) {
                for (int dt = 0; dt < kDt; ++dt) {
                    TileV vt;
                    mxi::joint_matrix_load_checked(sg, vt, l1_cached(vp), width, height, width,
                                                   static_cast<size_t>(kb + kt * kTk), col0 + dt * kTn);
                    mx::joint_matrix_mad(sg, o[dt], pa[kt], vt, o[dt]);
                }
            }
        }

        float l = 0.0f;  // lane i: row i's denominator
        for (int i = 0; i < kTm; ++i) {
            const float r = sycl::reduce_over_group(sg, ls[i], sycl::plus<float>());
            l = lane == i ? r : l;
        }
        if (a.split == 1) {
            for (int dt = 0; dt < kDt; ++dt) {
                for_rows(sg, o[dt], [&](float& x, int i) {
                    const float li = sycl::select_from_group(sg, l, i);
                    x = li > 0.0f ? x / li : 0.0f;
                });
            }
        }
        // Stage O through this sub-group's local memory, then write the valid rows.
        for (int dt = 0; dt < kDt; ++dt) {
            mx::joint_matrix_store(sg, o[dt], local_ptr(sg_slm + dt * kTn), DP, mx::layout::row_major);
        }
        sycl::group_barrier(sg);
        for (int i = 0; i < rows_valid; ++i) {
            const int64_t row_out = (t0 + i) * a.q_heads + h;
            const float* src = sg_slm + i * DP;
            if (a.split == 1) {
                for (int64_t d = lane; d < d_n; d += kSg) out[row_out * d_n + d] = src[d];
            } else {
                float* dst = part + (row_out * a.split + sp) * (d_n + 2);
                for (int64_t d = lane; d < d_n; d += kSg) dst[d] = src[d];
                const float mi = sycl::select_from_group(sg, m2, i) / kLog2e;
                const float li = sycl::select_from_group(sg, l, i);
                if (lane == 0) {
                    dst[d_n] = mi;
                    dst[d_n + 1] = li;
                }
            }
        }
    }
};

template <int DP, int BC, int SGS, typename QT>
void flash_t(sycl::queue& q, const QT* qp, const half* k, const half* v, float* out, float* part,
             const AttentionShape& s, int64_t split) {
    using K = FlashKernel<DP, BC, SGS, QT>;
    FlashArgs a;
    a.group = s.q_heads / s.kv_heads;
    a.tiles_per_head = ceil_div(s.tokens, kTm);
    a.tiles = a.group * a.tiles_per_head;
    a.row_blocks = ceil_div(a.tiles, SGS);
    a.split = split;
    a.q_heads = s.q_heads;
    a.kv_heads = s.kv_heads;
    a.head_dim = s.head_dim;
    a.tokens = s.tokens;
    a.kv_len = s.kv_len;
    a.window = s.window;
    a.scale = s.scale;
    a.softcap = s.softcap;
    a.causal = s.causal;
    a.key_bias = s.key_bias;
    const auto wgs = static_cast<size_t>(s.kv_heads * a.row_blocks * split);
    syclex::submit(q, [&](sycl::handler& h) {
        K kernel{qp, k, v, out, part, a, sycl::local_accessor<float, 1>(sycl::range<1>(K::kSlmFloats), h)};
        syclex::nd_launch(h, sycl::nd_range<1>(wgs * K::kWg, K::kWg), kernel);
    });
}

int flash_attention_dp(int64_t head_dim) {
    for (const int dp : {64, 80, 128}) {
        if (head_dim <= dp) return dp;
    }
    return 0;  // 256: 8 rows per sub-group fill the registers and every tile feeds one DPAS; oneMKL's GEMMs match it
}

int flash_attention_bc(int dp) {
    return dp <= 80 ? 64 : 32;
}

}  // namespace

bool flash_attention_device(const sycl::device& d) {
    using mx::matrix_type;
    try {
        for (const auto& c : d.get_info<syclex::info::device::matrix_combinations>()) {
            if (c.atype == matrix_type::fp16 && c.btype == matrix_type::fp16 && c.ctype == matrix_type::fp32 &&
                c.dtype == matrix_type::fp32 && c.nsize == kTn && c.ksize == kTk &&
                (c.msize == kTm || c.max_msize >= kTm)) {
                return true;
            }
        }
    } catch (const sycl::exception&) {
    }
    return false;
}

bool flash_attention_fits(const AttentionShape& s, const void* q, bool q_f16, const void* k, const void* v) {
    const int dp = flash_attention_dp(s.head_dim);
    if (dp == 0 || s.head_dim % 8 != 0 || (dp == 256 && !q_f16)) return false;
    // 2D block loads: 64-byte aligned bases, rows of at least 64 bytes, extents below 2^24
    const auto aligned = [](const void* p) { return reinterpret_cast<uintptr_t>(p) % 64 == 0; };
    if (!aligned(k) || !aligned(v) || (dp == 256 && !aligned(q))) return false;
    const int64_t row_bytes = s.kv_heads * s.head_dim * 2;
    return row_bytes >= 64 && row_bytes < (int64_t{1} << 24) && s.kv_len < (int64_t{1} << 24) &&
           s.q_heads * s.head_dim * 2 < (int64_t{1} << 24) && s.tokens < (int64_t{1} << 24);
}

int64_t flash_attention_split(const AttentionShape& s, int64_t compute_units) {
    const int64_t tiles = s.q_heads * ceil_div(s.tokens, kTm);
    const int64_t keys = s.causal && s.window > 0 ? std::min(s.window, s.kv_len) : s.kv_len;
    const int64_t blocks = ceil_div(keys, flash_attention_bc(flash_attention_dp(s.head_dim)));
    // about 2.5 sub-groups per execution unit in flight (4 threads each in the large register file)
    const int64_t want = compute_units * 5 / 2 / std::max<int64_t>(tiles, 1);
    return std::clamp<int64_t>(std::min(want, blocks / 2), 1, 64);
}

template <typename QT>
void flash_attention(sycl::queue& q, const QT* qp, const half* k, const half* v, float* out,
                     const AttentionShape& s, int64_t split, float* part) {
    switch (flash_attention_dp(s.head_dim)) {
        case 64: return flash_t<64, 64, 4, QT>(q, qp, k, v, out, part, s, split);
        case 80: return flash_t<80, 64, 4, QT>(q, qp, k, v, out, part, s, split);
        case 128: return flash_t<128, 32, 4, QT>(q, qp, k, v, out, part, s, split);
        default: return;
    }
}

template void flash_attention<float>(sycl::queue&, const float*, const half*, const half*, float*,
                                     const AttentionShape&, int64_t, float*);
template void flash_attention<half>(sycl::queue&, const half*, const half*, const half*, float*,
                                    const AttentionShape&, int64_t, float*);

}  // namespace xs::sycl_be
