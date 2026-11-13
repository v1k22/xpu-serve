#pragma once

// What the kernels written against the matrix engines directly share (sycl_quant_gemm.cpp, sycl_block_kernels.cpp):
// the IGC builtins for DPAS and 2D block loads, their operand types, and the launch that prepares x for the GEMVs on
// them.

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

namespace xs::sycl_be {

typedef float float8_v __attribute__((ext_vector_type(8)));
typedef short short8_v __attribute__((ext_vector_type(8)));
typedef int int8_v __attribute__((ext_vector_type(8)));
typedef int int2_v __attribute__((ext_vector_type(2)));
typedef unsigned short ushort8_v __attribute__((ext_vector_type(8)));
typedef unsigned short ushort32_v __attribute__((ext_vector_type(32)));
typedef _Float16 h2_v __attribute__((ext_vector_type(2)));

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
#else
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

// The GEMVs on the matrix engines read x as the A operand: [M, K] in f16 (rows past M read as zero), and per group of
// columns each row's sum of those f16 values, xs [K / group][8] (zero in the rows past M). xmx_prep writes both from x
// in f32 (times AWQ input scales, or null) into `out`, xmx_prep_bytes long: x at the start, the sums at xmx_x_bytes.
size_t xmx_x_bytes(int64_t m, int64_t k);
size_t xmx_prep_bytes(int64_t m, int64_t k, int group);
void xmx_prep(sycl::queue& q, const float* x, const float* in_scales, void* out, int64_t m, int64_t k, int group);

}  // namespace xs::sycl_be
