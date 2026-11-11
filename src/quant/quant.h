#pragma once

// Round-to-nearest quantization of weight matrices into the pack formats (DESIGN.md section 6.3), and dequantization.

#include <cstdint>
#include <string>
#include <vector>

#include "tensor/tensor.h"

namespace xs::quant {

// A quantized tensor of shape [..., cols] is a matrix of rows = numel / cols (an expert bank [E, N, K] is E * N rows
// of K). Q8: int8 codes [rows, cols]; Q4: uint8 [rows, cols / 2] with the even column in the low nibble. f16 scales
// and (Q4) biases [rows, cols / group]. Stored components keep the leading dims (component_shape).
inline int64_t matrix_rows(const Dims& shape) {
    int64_t rows = 1;
    for (size_t d = 0; d + 1 < shape.size(); ++d)
        rows *= shape[d];
    return rows;
}
// `shape` with its last dim replaced: the shape of a stored component (codes, scales, biases).
inline Dims component_shape(Dims shape, int64_t last) {
    shape.back() = last;
    return shape;
}

struct Quantized {
    DType format = DType::Q4;
    Dims shape;        // logical shape, at least 2-D
    int64_t rows = 0;  // matrix_rows(shape)
    int64_t cols = 0;  // shape.back()
    int64_t group = 0;
    std::vector<uint8_t> codes;
    std::vector<uint16_t> scales;
    std::vector<uint16_t> biases;  // Q4 only
    std::vector<float> input_scales;  // AWQ: per column, multiplies the dequantized weight; empty otherwise

    // The logical view (dtype Q8 / Q4, shape [rows, cols]) over this storage.
    TensorView view() const;
};

// Why a [rows, cols] matrix cannot be stored as `format` with this group size, or empty when it can.
std::string unsupported_reason(DType format, int64_t rows, int64_t cols, int64_t group);

// Round to nearest per group of `group` columns of `w` (at least 2-D; f32 / f16 / bf16, or an FP8 checkpoint's f8).
// Q8 is symmetric (scale = max|w| / 127, codes in [-127, 127]); Q4 is asymmetric (bias = min, scale =
// (max - min) / 15, codes in [0, 15]). Codes are chosen against the f16-rounded scale and bias, so dequantization
// reproduces exactly the values the rounding picked.
Quantized quantize(const TensorView& w, DType format, int64_t group);

// Activation-aware q4 (AWQ-style). `act_ms` is the mean square of each input channel over calibration data. For each
// alpha in 0, 0.1, ..., 1 the columns are scaled by s = rms^alpha (geometric mean 1) before quantizing, every group
// also tries clipping its range, and the choice minimizes sum_c act_ms[c] * (w_eff - w)^2, the matmul output error
// under independent channels (w_eff = dequant(q) / s, stored as input_scales = 1 / s). alpha 0 without clipping is
// round to nearest, so the result is never worse under that objective. `alpha` and the two errors are reported.
struct AwqReport {
    double alpha = 0;
    double error_rtn = 0;
    double error = 0;
};
Quantized quantize_awq(const TensorView& w, int64_t group, const std::vector<double>& act_ms, AwqReport* report);

// Float values of rows [first, first + count) of a Q8 / Q4 view, row-major into `out` (input_scales applied).
void dequantize_rows(const TensorView& v, int64_t first, int64_t count, float* out);

// The same of a float weight however it is stored: f32 / f16 / bf16, a pack's q8 / q4, or an FP8 checkpoint's f8
// (quant/fp8.h). What a backend or a converter reads a weight's values through.
void read_rows(const TensorView& v, int64_t first, int64_t count, float* out);

// The one-bit sketch of a weight [N, K] (f32 / f16 / bf16 / q8 / q4, K a multiple of 32) that graph::SignMatMul
// scores every row with: signs [N, K / 32] i32, bit c % 32 of word c / 32 set where w[n, c] >= 0, and scales [N] f32,
// each row's mean magnitude (the scale that brings +-scale closest to the row).
struct SignSketch {
    HostTensor signs;
    HostTensor scales;
};
SignSketch sign_sketch(const TensorView& w);

// The int8 KV cache rounding of one row (one token's head): code = rint(x * 127 / max|x|) and scale = max|x| / 127,
// so the value is code * scale. Writes `codes` and returns the scale (0 for an all-zero row). Backends match it.
float quantize_kv_row_i8(const float* x, int64_t width, int8_t* codes);
// Replaces each row of x [rows, width] by its int8 KV value (code * scale).
void round_trip_kv_i8(float* x, int64_t rows, int64_t width);

}  // namespace xs::quant
