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
// A work-group owns 32 output features and splits K between its sub-groups, which add their accumulators in local
// memory at the end: enough sub-groups to keep the memory busy without writing partial sums out (at 50 rows a split
// between work-groups spent as long on them as on the weight).

#include "backend/sycl/sycl_kernels.h"
#include "util/error.h"

#include <algorithm>
#include <cstdint>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

namespace xs::sycl_be {
namespace {

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

typedef float float8_v __attribute__((ext_vector_type(8)));
typedef short short8_v __attribute__((ext_vector_type(8)));
typedef int int8_v __attribute__((ext_vector_type(8)));
typedef int int2_v __attribute__((ext_vector_type(2)));
typedef unsigned short ushort8_v __attribute__((ext_vector_type(8)));
typedef unsigned short ushort32_v __attribute__((ext_vector_type(32)));
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
// The same for 32 rows: four 8 x 16 tiles, rows 8 j .. 8 j + 7 in elements 8 j .. 8 j + 7.
SYCL_EXTERNAL extern "C" xs::sycl_be::ushort32_v __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
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
inline xs::sycl_be::ushort32_v __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(long, int, int, int,
                                                                                  xs::sycl_be::int2_v) {
    return {};
}
#endif

namespace xs::sycl_be {
namespace {

constexpr int kSg = 16;
constexpr int kTm = 8;    // rows of x per tile
constexpr int kTk = 16;   // K per DPAS

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

inline int pack_half2(half lo, half hi) {
    return static_cast<int>(sycl::bit_cast<uint32_t>(sycl::vec<half, 2>(lo, hi)));
}

typedef _Float16 h2_v __attribute__((ext_vector_type(2)));

// q8 codes to f16 two at a time: a code c < 1024 OR'ed under the exponent of 1024 (0x6400) is exactly the f16
// 1024 + c, so one packed subtract recovers it and one packed multiply scales it: a few 32-bit operations per pair
// instead of a conversion, a multiply and a repack per value. `pair` holds the codes in bits 0-9 and 16-25.
inline int dequant_pair(uint32_t pair, _Float16 offset, h2_v scale) {
    const h2_v c = __builtin_bit_cast(h2_v, pair | 0x64006400u) - h2_v{offset, offset};
    return __builtin_bit_cast(int, c * scale);
}


// The same with a bias: c * scale + bias (q4's affine codes).
inline uint32_t dequant_pair_affine(uint32_t pair, _Float16 offset, h2_v scale, h2_v bias) {
    const h2_v c = __builtin_bit_cast(h2_v, pair | 0x64006400u) - h2_v{offset, offset};
    return __builtin_bit_cast(uint32_t, c * scale + bias);
}

template <bool Q4, int MT, bool AWQ, typename Y, bool ADD>
struct QuantGemmKernel {
    // 16-feature columns per work-group (each lane: kNt output features); 8 row tiles take one, or the accumulators
    // spill
    static constexpr int kNt = MT >= 8 ? 1 : 2;
    static constexpr int kFeatures = kNt * kSg;
    // Per step, one transposed 2D block load per 16 features brings each lane 8 dwords of its feature's codes:
    // kStep columns of K (32 at q8, 64 at q4), kSlices DPAS B operands.
    static constexpr int kStep = Q4 ? 64 : 32;
    static constexpr int kSlices = kStep / kTk;

    // What one step reads from memory: the codes, and each slice's scale (and bias). A scale is a scattered load (16
    // lanes, 16 rows), so a slice in the same group as the previous one reuses it. (Requesting the next step's ahead
    // measured no faster and spilled registers at 7 and 8 row tiles.)
    struct Step {
        uint8_v codes[kNt];
        half scale[kNt][kSlices];
        half bias[kNt][kSlices];
    };

    const half* x;
    QuantWeight w;
    Y* y;
    sycl::local_accessor<float, 1> slm;  // every slice's accumulators: [slice][mt][nt][8 rows][16 lanes]
    int64_t m, n, k, k_slice;
    int group_shift;  // log2(w.group): no 64-bit division in the loop (it is emulated, and slow)

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<kSg>, intelex::grf_size<256>};
    }

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int slice = static_cast<int>(sg.get_group_linear_id());  // this sub-group's share of K
        const int slices = static_cast<int>(sg.get_group_linear_range());
        const int64_t n0 = static_cast<int64_t>(it.get_group(0)) * kFeatures;
        const int64_t k_lo = std::min(k, slice * k_slice);
        const int64_t k_hi = std::min(k, k_lo + k_slice);
        const int64_t groups = k / w.group;
        const int64_t row_bytes = Q4 ? k / 2 : k;
        const int x_width = static_cast<int>(k * 2) - 1;  // x as a [M, K] f16 surface
        const int x_height = static_cast<int>(m) - 1;
        const auto x_base = reinterpret_cast<long>(x);
        const int c_width = static_cast<int>(row_bytes) - 1;  // the codes as an [N, row_bytes] surface
        const int c_height = static_cast<int>(n) - 1;
        const auto c_base = reinterpret_cast<long>(w.codes);
        const half* scales[kNt];
        const half* biases[kNt];
        for (int nt = 0; nt < kNt; ++nt) {
            const int64_t row = n0 + nt * kSg + lane;
            scales[nt] = w.scales + row * groups;
            biases[nt] = Q4 ? w.biases + row * groups : nullptr;
        }
        const auto fetch = [&](int64_t kk, Step& st) {
            for (int nt = 0; nt < kNt; ++nt) {
                const int2_v at = {static_cast<int>((Q4 ? kk / 2 : kk) / 4), static_cast<int>(n0 + nt * kSg)};
                st.codes[nt] = __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(c_base, c_width, c_height,
                                                                                      c_width, at);
            }
            for (int ks = 0; ks < kSlices; ++ks) {
                const int g = (static_cast<int>(kk) + ks * kTk) >> group_shift;
                const bool same = ks > 0 && g == ((static_cast<int>(kk) + (ks - 1) * kTk) >> group_shift);
                for (int nt = 0; nt < kNt; ++nt) {
                    st.scale[nt][ks] = same ? st.scale[nt][ks - 1] : scales[nt][g];
                    if constexpr (Q4) st.bias[nt][ks] = same ? st.bias[nt][ks - 1] : biases[nt][g];
                }
            }
        };

        float8_v acc[MT][kNt];
        for (int mt = 0; mt < MT; ++mt) {
            for (int nt = 0; nt < kNt; ++nt) acc[mt][nt] = 0.0f;
        }
        for (int64_t kk = k_lo; kk < k_hi; kk += kStep) {
            Step cur;
            fetch(kk, cur);
            int8_v b[kNt][kSlices];
            for (int nt = 0; nt < kNt; ++nt) {
                const uint8_v c = cur.codes[nt];
                for (int ks = 0; ks < kSlices; ++ks) {
                    if constexpr (Q4) {
                        // 16 columns = 8 bytes = dwords 2 ks, 2 ks + 1; even column in the low nibble. (The packed
                        // trick below measured 2-3x slower here: the compiler's code for it spilled.)
                        const half sh = cur.scale[nt][ks];
                        const half bias = cur.bias[nt][ks];
                        for (int d = 0; d < 2; ++d) {
                            const uint32_t word = c[2 * ks + d];
                            for (int i = 0; i < 4; ++i) {
                                const uint32_t byte = (word >> (8 * i)) & 0xFF;
                                b[nt][ks][4 * d + i] = pack_half2(sycl::fma(static_cast<half>(byte & 0x0F), sh, bias),
                                                                  sycl::fma(static_cast<half>(byte >> 4), sh, bias));
                            }
                        }
                    } else {
                        // 16 columns = 16 bytes = dwords 4 ks .. 4 ks + 3; codes biased to unsigned (c + 128)
                        const _Float16 sf = sycl::bit_cast<_Float16>(cur.scale[nt][ks]);
                        const h2_v s2{sf, sf};
                        for (int d = 0; d < 4; ++d) {
                            const uint32_t w8 = c[4 * ks + d] ^ 0x80808080u;
                            b[nt][ks][2 * d] = dequant_pair((w8 & 0xFFu) | ((w8 & 0xFF00u) << 8), 1152, s2);
                            b[nt][ks][2 * d + 1] = dequant_pair(((w8 >> 16) & 0xFFu) | ((w8 >> 8) & 0xFF0000u), 1152, s2);
                        }
                    }
                    if constexpr (AWQ) {
                        for (int i = 0; i < 8; ++i) {
                            const int64_t col = kk + ks * kTk + 2 * i;
                            const auto p = sycl::bit_cast<sycl::vec<half, 2>>(static_cast<uint32_t>(b[nt][ks][i]));
                            b[nt][ks][i] = pack_half2(static_cast<half>(static_cast<float>(p[0]) * w.input_scales[col]),
                                                      static_cast<half>(static_cast<float>(p[1]) * w.input_scales[col + 1]));
                        }
                    }
                }
            }
            for (int ks = 0; ks < kSlices; ++ks) {
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
        // The slices add up in local memory, each one finishing the row tiles mt = slice, slice + slices, ...
        constexpr int kTileFloats = kTm * kSg;
        for (int mt = 0; mt < MT; ++mt) {
            for (int nt = 0; nt < kNt; ++nt) {
                float* t = &slm[((slice * MT + mt) * kNt + nt) * kTileFloats];
                for (int r = 0; r < kTm; ++r) t[r * kSg + lane] = acc[mt][nt][r];
            }
        }
        sycl::group_barrier(it.get_group());
        for (int mt = slice; mt < MT; mt += slices) {
            for (int nt = 0; nt < kNt; ++nt) {
                float8_v sum = 0.0f;
                for (int s = 0; s < slices; ++s) {
                    const float* t = &slm[((s * MT + mt) * kNt + nt) * kTileFloats];
                    for (int r = 0; r < kTm; ++r) sum[r] += t[r * kSg + lane];
                }
                // element r, lane l: y row mt * 8 + r, feature n0 + nt * 16 + l
                for (int r = 0; r < kTm; ++r) {
                    const int64_t row = mt * kTm + r;
                    if (row >= m) break;
                    Y& out = y[row * n + n0 + nt * kSg + lane];
                    out = ADD ? out + static_cast<Y>(sum[r]) : static_cast<Y>(sum[r]);
                }
            }
        }
    }
};

template <bool Q4, int MT, bool AWQ, typename Y, bool ADD>
void quant_gemm_t(sycl::queue& q, const half* x, const QuantWeight& w, Y* y, int64_t m, int64_t n, int64_t k,
                  int64_t slices) {
    using K = QuantGemmKernel<Q4, MT, AWQ, Y, ADD>;
    const int64_t k_slice = ceil_div(ceil_div(k, slices), 64) * 64;  // whole steps (32 or 64 columns)
    const auto wgs = static_cast<size_t>(n / K::kFeatures);
    int shift = 0;
    while ((int64_t{1} << shift) < w.group) ++shift;
    const auto slm_floats = static_cast<size_t>(slices * MT * K::kNt * kTm * kSg);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(slm_floats), h);
        const auto local = static_cast<size_t>(slices) * kSg;
        syclex::nd_launch(h, sycl::nd_range<1>(wgs * local, local), K{x, w, y, slm, m, n, k, k_slice, shift});
    });
}

template <bool Q4, bool AWQ, typename Y, bool ADD>
void quant_gemm_q(sycl::queue& q, const half* x, const QuantWeight& w, Y* y, int64_t m, int64_t n, int64_t k,
                  int64_t slices) {
    const int64_t mt = ceil_div(m, kTm);
    if (mt <= 2) {
        quant_gemm_t<Q4, 2, AWQ, Y, ADD>(q, x, w, y, m, n, k, slices);
    } else if (mt <= 4) {
        quant_gemm_t<Q4, 4, AWQ, Y, ADD>(q, x, w, y, m, n, k, slices);
    } else if (mt <= 7) {
        quant_gemm_t<Q4, 7, AWQ, Y, ADD>(q, x, w, y, m, n, k, slices);
    } else {
        quant_gemm_t<Q4, 8, AWQ, Y, ADD>(q, x, w, y, m, n, k, slices);
    }
}

// The many-row kernel (quant_gemm_tiled). A work-group owns 64 output features and 512 rows; per step of
// 64 columns its 16 weight tiles (4 feature tiles x 4 slices of 16 columns, each a DPAS B operand) are dequantized once,
// shared out between its sub-groups, into one of two local buffers, and every sub-group multiplies its 64 rows by its
// 32 features from there: dequantizing per sub-group instead spent as long on the arithmetic as on the matrix engines.
// While a step multiplies, the next step's codes are in flight and its tiles are written to the other buffer; one
// barrier per step.
constexpr int kTiledMt = 8;         // row tiles of 8 per sub-group (64 rows)
constexpr int kTiledWm = 8;         // sub-groups across rows: a work-group owns 512 rows (fewer rows per work-group
                                    // left each sub-group more tiles to dequantize, and ran slower at any row count)
constexpr int kTiledNt = 2;         // feature tiles of 16 per sub-group (32 features)
constexpr int kTiledWn = 2;         // sub-groups across features: a work-group owns 64 features
constexpr int kTiledSlices = 4;     // 16-column slices per step
constexpr int kTiledStep = kTiledSlices * kTk;
constexpr int kTiledFeatures = kTiledWn * kTiledNt * kSg;
constexpr int kTiledTiles = kTiledWn * kTiledNt * kTiledSlices;  // B tiles per step
constexpr int kTileWords = 8 * kSg;  // a B tile: 8 dwords per lane, stored [dword][lane]

template <bool Q4, bool AWQ, typename Y, bool ADD>
struct QuantGemmTiledKernel {
    static constexpr int kSgs = kTiledWm * kTiledWn;
    static constexpr int kRows = kTiledWm * kTiledMt * kTm;
    static constexpr int kPerSg = kTiledTiles / kSgs;  // tiles each sub-group dequantizes per step
    static constexpr int kWords = Q4 ? 2 : 4;          // code dwords per lane per tile
    static_assert(kTiledTiles % kSgs == 0, "the tiles of a step split evenly between the sub-groups");

    const half* x;
    QuantWeight w;
    Y* y;
    sycl::local_accessor<uint32_t, 1> slm;  // two buffers of kTiledTiles tiles
    int64_t m, n, k, m_blocks;
    int group_shift;

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<kSg>, intelex::grf_size<256>};
    }

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int sgid = static_cast<int>(sg.get_group_linear_id());
        const int sm = sgid / kTiledWn;
        const int sn = sgid % kTiledWn;
        const int64_t wg = it.get_group(0);
        const int64_t m0 = (wg % m_blocks) * kRows + sm * kTiledMt * kTm;
        const int64_t wn0 = (wg / m_blocks) * kTiledFeatures;
        const int64_t groups = k / w.group;
        const int64_t row_bytes = Q4 ? k / 2 : k;
        const int x_width = static_cast<int>(k * 2) - 1;  // x as a [M, K] f16 surface
        const int x_height = static_cast<int>(m) - 1;
        const auto x_base = reinterpret_cast<long>(x);
        uint32_t* const tiles = &slm[0];
        // Every loop of fixed count is unrolled: the codes, the accumulators and the tile arrays stay in registers (a
        // loop over a sub-group's tiles left them in scratch memory: 5 to 8 times slower).

        // tile t of a step: feature tile t / kTiledSlices, slice t % kTiledSlices; this sub-group's are sgid + j kSgs
        const auto load_codes = [&](int64_t kk, uint32_t (&c)[kPerSg][kWords]) {
            #pragma unroll
            for (int j = 0; j < kPerSg; ++j) {
                const int t = sgid + j * kSgs;
                const int64_t row = wn0 + (t / kTiledSlices) * kSg + lane;
                const int64_t col = kk + (t % kTiledSlices) * kTk;
                const auto* src = reinterpret_cast<const uint32_t*>(w.codes + row * row_bytes + (Q4 ? col / 2 : col));
                #pragma unroll
                for (int d = 0; d < kWords; ++d) c[j][d] = src[d];
            }
        };
        const auto dequant = [&](int64_t kk, const uint32_t (&c)[kPerSg][kWords], int buf) {
            #pragma unroll
            for (int j = 0; j < kPerSg; ++j) {
                const int t = sgid + j * kSgs;
                const int64_t row = wn0 + (t / kTiledSlices) * kSg + lane;
                const int64_t col = kk + (t % kTiledSlices) * kTk;
                const int64_t g = row * groups + (static_cast<int>(col) >> group_shift);
                const _Float16 sf = sycl::bit_cast<_Float16>(w.scales[g]);
                const h2_v s2{sf, sf};
                uint32_t pairs[8];
                if constexpr (Q4) {  // 16 columns = 2 dwords; even column in the low nibble
                    const _Float16 bf = sycl::bit_cast<_Float16>(w.biases[g]);
                    const h2_v b2{bf, bf};
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const uint32_t v = c[j][d] >> (8 * i);
                            pairs[4 * d + i] = dequant_pair_affine((v & 0x0Fu) | ((v & 0xF0u) << 12), 1024, s2, b2);
                        }
                    }
                } else {  // 16 columns = 4 dwords; codes biased to unsigned (c + 128)
                    #pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        const uint32_t w8 = c[j][d] ^ 0x80808080u;
                        pairs[2 * d] = static_cast<uint32_t>(dequant_pair((w8 & 0xFFu) | ((w8 & 0xFF00u) << 8), 1152, s2));
                        pairs[2 * d + 1] =
                            static_cast<uint32_t>(dequant_pair(((w8 >> 16) & 0xFFu) | ((w8 >> 8) & 0xFF0000u), 1152, s2));
                    }
                }
                if constexpr (AWQ) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const auto p = sycl::bit_cast<sycl::vec<half, 2>>(pairs[i]);
                        pairs[i] = static_cast<uint32_t>(
                            pack_half2(static_cast<half>(static_cast<float>(p[0]) * w.input_scales[col + 2 * i]),
                                       static_cast<half>(static_cast<float>(p[1]) * w.input_scales[col + 2 * i + 1])));
                    }
                }
                uint32_t* dst = tiles + (buf * kTiledTiles + t) * kTileWords + lane;
                #pragma unroll
                for (int i = 0; i < 8; ++i) dst[i * kSg] = pairs[i];
            }
        };

        float8_v acc[kTiledMt][kTiledNt];
        #pragma unroll
        for (int mt = 0; mt < kTiledMt; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < kTiledNt; ++nt) acc[mt][nt] = 0.0f;
        }
        uint32_t codes[kPerSg][kWords];
        load_codes(0, codes);
        dequant(0, codes, 0);
        sycl::group_barrier(it.get_group());
        const int64_t steps = k / kTiledStep;
        for (int64_t s = 0; s < steps; ++s) {
            const int64_t kk = s * kTiledStep;
            const int cur = static_cast<int>(s & 1);
            if (s + 1 < steps) load_codes(kk + kTiledStep, codes);
            #pragma unroll
            for (int ks = 0; ks < kTiledSlices; ++ks) {
                int8_v b[kTiledNt];
                #pragma unroll
                for (int nt = 0; nt < kTiledNt; ++nt) {
                    uint32_t* src = tiles + (cur * kTiledTiles + (sn * kTiledNt + nt) * kTiledSlices + ks) * kTileWords;
                    const auto lp = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                             sycl::access::decorated::yes>(src);
                    sycl::vec<uint32_t, 8> v;
                    syclex::group_load(sg, lp, v,
                                       syclex::properties{syclex::data_placement_striped, syclex::contiguous_memory,
                                                          syclex::full_group});
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) b[nt][i] = static_cast<int>(v[i]);
                }
                #pragma unroll
                for (int m4 = 0; m4 < kTiledMt; m4 += 4) {
                    const int2_v at = {static_cast<int>(kk + ks * kTk), static_cast<int>(m0 + m4 * kTm)};
                    const ushort32_v a4 =
                        __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(x_base, x_width, x_height, x_width, at);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        ushort8_v a;
                        #pragma unroll
                        for (int r = 0; r < kTm; ++r) a[r] = a4[kTm * j + r];
                        #pragma unroll
                        for (int nt = 0; nt < kTiledNt; ++nt) {
                            acc[m4 + j][nt] = __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(
                                acc[m4 + j][nt], __builtin_bit_cast(short8_v, a), b[nt]);
                        }
                    }
                }
            }
            if (s + 1 < steps) dequant(kk + kTiledStep, codes, cur ^ 1);
            sycl::group_barrier(it.get_group());
        }
        // element r, lane l of acc[mt][nt]: y row m0 + mt * 8 + r, feature wn0 + (sn * 2 + nt) * 16 + l
        #pragma unroll
        for (int mt = 0; mt < kTiledMt; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < kTiledNt; ++nt) {
                #pragma unroll
                for (int r = 0; r < kTm; ++r) {
                    const int64_t row = m0 + mt * kTm + r;
                    if (row >= m) break;
                    Y& out = y[row * n + wn0 + (sn * kTiledNt + nt) * kSg + lane];
                    out = ADD ? out + static_cast<Y>(acc[mt][nt][r]) : static_cast<Y>(acc[mt][nt][r]);
                }
            }
        }
    }
};

template <bool Q4, bool AWQ, typename Y, bool ADD>
void quant_gemm_tiled_t(sycl::queue& q, const half* x, const QuantWeight& w, Y* y, int64_t m, int64_t n, int64_t k) {
    using K = QuantGemmTiledKernel<Q4, AWQ, Y, ADD>;
    const int64_t m_blocks = ceil_div(m, K::kRows);
    const auto wgs = static_cast<size_t>(m_blocks * (n / kTiledFeatures));
    int shift = 0;
    while ((int64_t{1} << shift) < w.group) ++shift;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> slm(sycl::range<1>(2 * kTiledTiles * kTileWords), h);
        constexpr size_t local = K::kSgs * kSg;
        syclex::nd_launch(h, sycl::nd_range<1>(wgs * local, local), K{x, w, y, slm, m, n, k, m_blocks, shift});
    });
}


}  // namespace

bool block_load_aligned(const void* x) {
    return reinterpret_cast<uintptr_t>(x) % 64 == 0;
}

bool quant_gemm_fits(int64_t m, int64_t n, int64_t k, const QuantWeight& w) {
    // 2D block loads of x: rows of at least 64 bytes and a multiple of 16; of the codes ([N, K] bytes at q8,
    // [N, K / 2] at q4): a 64-byte aligned base, rows of at least 64 bytes
    const bool pow2_group = w.group >= kTk && (w.group & (w.group - 1)) == 0;
    return m > 0 && m <= kQuantGemmMaxRows && n % (2 * kSg) == 0 && k % 128 == 0 && pow2_group &&
           block_load_aligned(w.codes) && k * 2 < (int64_t{1} << 24) && n < (int64_t{1} << 24);
}

int64_t quant_gemm_slices(int64_t n, int64_t k, int64_t compute_units) {
    // about 4 sub-groups per execution unit streaming the weight, each slice at least 128 columns of K
    const int64_t blocks = n / (2 * kSg);
    int64_t slices = 1;
    while (slices < kQuantGemmMaxSlices && blocks * slices < compute_units * 4 && k / (slices * 2) >= 128) slices *= 2;
    return slices;
}

template <typename Y>
void quant_gemm(sycl::queue& q, const half* x, const QuantWeight& w, Y* y, int64_t m, int64_t n, int64_t k,
                int64_t slices, bool add) {
    quant_dispatch(w, [&]<bool Q4, bool AWQ>() {
        add ? quant_gemm_q<Q4, AWQ, Y, true>(q, x, w, y, m, n, k, slices)
            : quant_gemm_q<Q4, AWQ, Y, false>(q, x, w, y, m, n, k, slices);
    });
}

bool quant_gemm_tiled_fits(int64_t m, int64_t n, int64_t k, const QuantWeight& w) {
    // every 16-column slice in one group; the codes read in dwords
    const bool pow2_group = w.group >= kTk && (w.group & (w.group - 1)) == 0;
    return m > 0 && n % kTiledFeatures == 0 && k % kTiledStep == 0 && pow2_group &&
           reinterpret_cast<uintptr_t>(w.codes) % 16 == 0 && k * 2 < (int64_t{1} << 24) && m < (int64_t{1} << 24);
}

template <typename Y>
void quant_gemm_tiled(sycl::queue& q, const half* x, const QuantWeight& w, Y* y, int64_t m, int64_t n, int64_t k,
                      bool add) {
    quant_dispatch(w, [&]<bool Q4, bool AWQ>() {
        add ? quant_gemm_tiled_t<Q4, AWQ, Y, true>(q, x, w, y, m, n, k)
            : quant_gemm_tiled_t<Q4, AWQ, Y, false>(q, x, w, y, m, n, k);
    });
}

template void quant_gemm_tiled<float>(sycl::queue&, const half*, const QuantWeight&, float*, int64_t, int64_t, int64_t,
                                      bool);
template void quant_gemm_tiled<half>(sycl::queue&, const half*, const QuantWeight&, half*, int64_t, int64_t, int64_t,
                                     bool);

template void quant_gemm<float>(sycl::queue&, const half*, const QuantWeight&, float*, int64_t, int64_t, int64_t,
                                int64_t, bool);
template void quant_gemm<half>(sycl::queue&, const half*, const QuantWeight&, half*, int64_t, int64_t, int64_t,
                               int64_t, bool);

}  // namespace xs::sycl_be
