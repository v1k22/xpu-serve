#pragma once

// Round-to-nearest quantization of weight matrices into the pack formats (DESIGN.md section 6.3), and dequantization.

#include <cstdint>
#include <string>
#include <vector>

#include "tensor/tensor.h"

namespace xs::quant {

// A quantized [rows, cols] matrix. Q8: int8 codes [rows, cols]; Q4: uint8 [rows, cols / 2] with the even column in
// the low nibble. f16 scales and (Q4) biases [rows, cols / group].
struct Quantized {
    DType format = DType::Q4;
    int64_t rows = 0;
    int64_t cols = 0;
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

// Round to nearest per group of `group` columns of `w` (2-D, f32 / f16 / bf16). Q8 is symmetric (scale = max|w| / 127,
// codes in [-127, 127]); Q4 is asymmetric (bias = min, scale = (max - min) / 15, codes in [0, 15]). Codes are chosen
// against the f16-rounded scale and bias, so dequantization reproduces exactly the values the rounding picked.
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

// The int8 KV cache rounding of one row (one token's head): code = rint(x * 127 / max|x|) and scale = max|x| / 127,
// so the value is code * scale. Writes `codes` and returns the scale (0 for an all-zero row). Backends match it.
float quantize_kv_row_i8(const float* x, int64_t width, int8_t* codes);
// Replaces each row of x [rows, width] by its int8 KV value (code * scale).
void round_trip_kv_i8(float* x, int64_t rows, int64_t width);

}  // namespace xs::quant
