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

    // The logical view (dtype Q8 / Q4, shape [rows, cols]) over this storage.
    TensorView view() const;
};

// Why a [rows, cols] matrix cannot be stored as `format` with this group size, or empty when it can.
std::string unsupported_reason(DType format, int64_t rows, int64_t cols, int64_t group);

// Round to nearest per group of `group` columns of `w` (2-D, f32 / f16 / bf16). Q8 is symmetric (scale = max|w| / 127,
// codes in [-127, 127]); Q4 is asymmetric (bias = min, scale = (max - min) / 15, codes in [0, 15]). Codes are chosen
// against the f16-rounded scale and bias, so dequantization reproduces exactly the values the rounding picked.
Quantized quantize(const TensorView& w, DType format, int64_t group);

// Float values of rows [first, first + count) of a Q8 / Q4 view, row-major into `out`.
void dequantize_rows(const TensorView& v, int64_t first, int64_t count, float* out);

}  // namespace xs::quant
