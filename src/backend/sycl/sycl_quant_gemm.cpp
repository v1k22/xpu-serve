// y [M, N] = x [M, K] @ dequant(w)^T for a few rows (kGemvMaxRows < M <= kQuantGemmMaxRows) on the matrix engines,
// reading the q8 / q4 weight once and dequantizing it on the way: with few rows the matmul is bound by streaming the
// weight, so its bytes are what counts (the GEMM path instead dequantizes the whole weight to f16 first).
//
// Written against the hardware's DPAS and 2D block load directly (IGC builtins, as Intel's own kernels do): the B
// operand of a DPAS holds, per lane, one column of 16 consecutive K values packed in pairs, which is a weight row, so
// each lane dequantizes its own output feature's codes straight into its B register. (joint_matrix cannot build an
// operand from registers: element access on A and B tiles loses the device.) A sub-group owns 32 output features: a
// transposed 2D block load gives each lane 32 (q8) or 64 (q4) codes of its feature, and per 16 columns of K every
// 8-row tile of x (one 2D block load, rows past M read as zero) multiplies both B registers.
// When N gives too few sub-groups to keep the memory busy, K is split between work-groups and a second kernel adds
// the partial sums.

#include "backend/sycl/sycl_kernels.h"
#include "util/error.h"

#include <algorithm>
#include <cstdint>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace xs::sycl_be {
namespace {

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

typedef float float8_v __attribute__((ext_vector_type(8)));
typedef short short8_v __attribute__((ext_vector_type(8)));
typedef int int8_v __attribute__((ext_vector_type(8)));
typedef int int2_v __attribute__((ext_vector_type(2)));
typedef unsigned short ushort8_v __attribute__((ext_vector_type(8)));
typedef unsigned int uint8_v __attribute__((ext_vector_type(8)));

}  // namespace
}  // namespace xs::sycl_be

#ifdef __SYCL_DEVICE_ONLY__
// acc + A (8 x 16 f16: lane l holds column l of each row) * B (16 x 16 f16: lane l holds column l, K in pairs).
SYCL_EXTERNAL extern "C" xs::sycl_be::float8_v __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(
    xs::sycl_be::float8_v acc, xs::sycl_be::short8_v a, xs::sycl_be::int8_v b);
// An 8 x 16 tile of 16-bit elements at (coord.x, coord.y) of a surface (width and pitch in bytes, minus one), in the
// DPAS A layout; elements past the surface read as zero.
SYCL_EXTERNAL extern "C" xs::sycl_be::ushort8_v __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
    long base, int width_minus_one, int height_minus_one, int pitch_minus_one, xs::sycl_be::int2_v coord);
// 16 rows x 8 dwords at (coord.x dwords, coord.y rows) of a surface, transposed: lane l holds row l's 8 dwords.
SYCL_EXTERNAL extern "C" xs::sycl_be::uint8_v __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(
    long base, int width_minus_one, int height_minus_one, int pitch_minus_one, xs::sycl_be::int2_v coord);
#else
inline xs::sycl_be::uint8_v __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(long, int, int, int,
                                                                                   xs::sycl_be::int2_v) {
    return {};
}
inline xs::sycl_be::float8_v __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(xs::sycl_be::float8_v acc,
                                                                          xs::sycl_be::short8_v, xs::sycl_be::int8_v) {
    return acc;
}
inline xs::sycl_be::ushort8_v __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(long, int, int, int,
                                                                                xs::sycl_be::int2_v) {
    return {};
}
#endif

namespace xs::sycl_be {
namespace {

constexpr int kSg = 16;
constexpr int kTm = 8;    // rows of x per tile
constexpr int kTk = 16;   // K per DPAS
constexpr int kNt = 2;    // 16-feature columns per sub-group (each lane: kNt output features)
constexpr int kSgs = 4;   // sub-groups per work-group

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

inline int pack_half2(half lo, half hi) {
    return static_cast<int>(sycl::bit_cast<uint32_t>(sycl::vec<half, 2>(lo, hi)));
}

template <bool Q4, int MT, typename Y>
struct QuantGemmKernel {
    static constexpr int kFeatures = kNt * kSg;  // per sub-group

    const half* x;
    QuantWeight w;
    Y* y;         // split == 1
    float* part;  // split > 1: [split, M, N]
    int64_t m, n, k, k_chunk, split;
    int group_shift;  // log2(w.group): no 64-bit division in the loop (it is emulated, and slow)

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<kSg>, intelex::grf_size<256>};
    }

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const auto wg = static_cast<int64_t>(it.get_group(0));
        const int64_t sp = wg % split;
        const int64_t n0 = ((wg / split) * kSgs + static_cast<int64_t>(sg.get_group_linear_id())) * kFeatures;
        if (n0 >= n) return;
        const int64_t k_lo = sp * k_chunk;
        const int64_t k_hi = std::min(k, k_lo + k_chunk);
        const int64_t groups = k / w.group;
        const int64_t row_bytes = Q4 ? k / 2 : k;
        const int x_width = static_cast<int>(k * 2) - 1;  // x as a [M, K] f16 surface
        const int x_height = static_cast<int>(m) - 1;
        const auto x_base = reinterpret_cast<long>(x);

        float8_v acc[MT][kNt];
        for (int mt = 0; mt < MT; ++mt) {
            for (int nt = 0; nt < kNt; ++nt) acc[mt][nt] = 0.0f;
        }
        // Per step, one transposed 2D block load per 16 features brings each lane 8 dwords of its feature's codes:
        // kStep columns of K (32 at q8, 64 at q4), kStep / 16 DPAS B operands.
        constexpr int kStep = Q4 ? 64 : 32;
        const int c_width = static_cast<int>(row_bytes) - 1;  // the codes as an [N, row_bytes] surface
        const int c_height = static_cast<int>(n) - 1;
        const auto c_base = reinterpret_cast<long>(w.codes);
        for (int64_t kk = k_lo; kk < k_hi; kk += kStep) {
            int8_v b[kNt][kStep / kTk];
            for (int nt = 0; nt < kNt; ++nt) {
                const int64_t row = n0 + nt * kSg + lane;
                const half* row_scales = w.scales + row * groups;
                const half* row_biases = Q4 ? w.biases + row * groups : nullptr;
                const int2_v at = {static_cast<int>((Q4 ? kk / 2 : kk) / 4), static_cast<int>(n0 + nt * kSg)};
                const uint8_v c = __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(c_base, c_width, c_height,
                                                                                         c_width, at);
                for (int ks = 0; ks < kStep / kTk; ++ks) {
                    const int g = (static_cast<int>(kk) + ks * kTk) >> group_shift;
                    const half s = row_scales[g];
                    if constexpr (Q4) {
                        const half bias = row_biases[g];
                        // 16 columns = 8 bytes = dwords 2 ks, 2 ks + 1; even column in the low nibble
                        for (int d = 0; d < 2; ++d) {
                            const uint32_t word = c[2 * ks + d];
                            for (int i = 0; i < 4; ++i) {
                                const uint32_t byte = (word >> (8 * i)) & 0xFF;
                                b[nt][ks][4 * d + i] = pack_half2(sycl::fma(static_cast<half>(byte & 0x0F), s, bias),
                                                                  sycl::fma(static_cast<half>(byte >> 4), s, bias));
                            }
                        }
                    } else {
                        // 16 columns = 16 bytes = dwords 4 ks .. 4 ks + 3
                        const sycl::vec<uint32_t, 4> words(c[4 * ks], c[4 * ks + 1], c[4 * ks + 2], c[4 * ks + 3]);
                        const sycl::vec<half, 16> v = sycl::bit_cast<sycl::vec<int8_t, 16>>(words).template convert<half>() * s;
                        for (int i = 0; i < 8; ++i) b[nt][ks][i] = pack_half2(v[2 * i], v[2 * i + 1]);
                    }
                    if (w.input_scales != nullptr) {
                        for (int i = 0; i < 8; ++i) {
                            const int64_t col = kk + ks * kTk + 2 * i;
                            const auto p = sycl::bit_cast<sycl::vec<half, 2>>(static_cast<uint32_t>(b[nt][ks][i]));
                            b[nt][ks][i] = pack_half2(static_cast<half>(static_cast<float>(p[0]) * w.input_scales[col]),
                                                      static_cast<half>(static_cast<float>(p[1]) * w.input_scales[col + 1]));
                        }
                    }
                }
            }
            for (int ks = 0; ks < kStep / kTk; ++ks) {
                for (int mt = 0; mt < MT; ++mt) {
                    const int2_v coord = {static_cast<int>(kk + ks * kTk), mt * kTm};
                    const ushort8_v a =
                        __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(x_base, x_width, x_height, x_width, coord);
                    for (int nt = 0; nt < kNt; ++nt) {
                        acc[mt][nt] = __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(
                            acc[mt][nt], __builtin_bit_cast(short8_v, a), b[nt][ks]);
                    }
                }
            }
        }
        // acc[mt][nt] element r, lane l: y row mt * 8 + r, feature n0 + nt * 16 + l.
        for (int mt = 0; mt < MT; ++mt) {
            for (int r = 0; r < kTm; ++r) {
                const int64_t row = mt * kTm + r;
                if (row >= m) break;
                for (int nt = 0; nt < kNt; ++nt) {
                    const int64_t col = n0 + nt * kSg + lane;
                    if (split > 1) {
                        part[(sp * m + row) * n + col] = acc[mt][nt][r];
                    } else {
                        y[row * n + col] = static_cast<Y>(acc[mt][nt][r]);
                    }
                }
            }
        }
    }
};

template <bool Q4, int MT, typename Y>
void quant_gemm_t(sycl::queue& q, const half* x, const QuantWeight& w, Y* y, float* part, int64_t m, int64_t n,
                  int64_t k, int64_t split) {
    using K = QuantGemmKernel<Q4, MT, Y>;
    const int64_t k_chunk = ceil_div(ceil_div(k, split), 64) * 64;  // whole steps (32 or 64 columns)
    const int64_t blocks = ceil_div(n, K::kFeatures);
    const auto wgs = static_cast<size_t>(ceil_div(blocks, kSgs) * split);
    int shift = 0;
    while ((int64_t{1} << shift) < w.group) ++shift;
    syclex::nd_launch(q, sycl::nd_range<1>(wgs * kSgs * kSg, kSgs * kSg),
                      K{x, w, y, part, m, n, k, k_chunk, split, shift});
}

template <typename Y>
void reduce_split(sycl::queue& q, const float* part, Y* y, int64_t mn, int64_t split) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(mn)), [=](sycl::id<1> idx) {
        const auto i = static_cast<int64_t>(idx[0]);
        float s = 0.0f;
        for (int64_t p = 0; p < split; ++p) s += part[p * mn + i];
        y[i] = static_cast<Y>(s);
    });
}

template <bool Q4, typename Y>
void quant_gemm_q(sycl::queue& q, const half* x, const QuantWeight& w, Y* y, float* part, int64_t m, int64_t n,
                  int64_t k, int64_t split) {
    const int64_t mt = ceil_div(m, kTm);
    if (mt <= 2) {
        quant_gemm_t<Q4, 2, Y>(q, x, w, y, part, m, n, k, split);
    } else if (mt <= 4) {
        quant_gemm_t<Q4, 4, Y>(q, x, w, y, part, m, n, k, split);
    } else if (mt <= 7) {
        quant_gemm_t<Q4, 7, Y>(q, x, w, y, part, m, n, k, split);
    } else {
        quant_gemm_t<Q4, 8, Y>(q, x, w, y, part, m, n, k, split);
    }
}

}  // namespace

bool quant_gemm_fits(int64_t m, int64_t n, int64_t k, const QuantWeight& w, const void* x) {
    const auto aligned = [](const void* p, size_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; };
    // 2D block loads of x: a 64-byte aligned base, rows of at least 64 bytes and a multiple of 16
    // and of the codes ([N, K] bytes at q8, [N, K / 2] at q4): rows of at least 64 bytes
    const bool pow2_group = w.group >= kTk && (w.group & (w.group - 1)) == 0 && (w.bits == 8 || w.bits == 4);
    return m > 0 && m <= kQuantGemmMaxRows && n % (kNt * kSg) == 0 && k % 128 == 0 && pow2_group &&
           aligned(x, 64) && aligned(w.codes, 64) && k * 2 < (int64_t{1} << 24) && n < (int64_t{1} << 24);
}

int64_t quant_gemm_split(int64_t n, int64_t k, int64_t compute_units) {
    // about 4 sub-groups per execution unit streaming the weight; at least 256 of K per split
    const int64_t sub_groups = n / (kNt * kSg);
    const int64_t want = ceil_div(compute_units * 4, std::max<int64_t>(sub_groups, 1));
    return std::clamp<int64_t>(std::min(want, k / 256), 1, 16);
}

template <typename Y>
void quant_gemm(sycl::queue& q, const half* x, const QuantWeight& w, Y* y, float* part, int64_t m, int64_t n,
                int64_t k, int64_t split) {
    if (w.bits == 4) {
        quant_gemm_q<true, Y>(q, x, w, y, part, m, n, k, split);
    } else {
        quant_gemm_q<false, Y>(q, x, w, y, part, m, n, k, split);
    }
    if (split > 1) reduce_split(q, part, y, m * n, split);
}

template void quant_gemm<float>(sycl::queue&, const half*, const QuantWeight&, float*, float*, int64_t, int64_t,
                                int64_t, int64_t);
template void quant_gemm<half>(sycl::queue&, const half*, const QuantWeight&, half*, float*, int64_t, int64_t,
                               int64_t, int64_t);

}  // namespace xs::sycl_be
