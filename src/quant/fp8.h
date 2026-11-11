#pragma once

// FP8 checkpoints (DeepSeek-V3's layout, which the -FP8 releases of Qwen use): a projection's weight is stored as
// E4M3 codes, a byte each, beside `<name>_scale_inv`, one scale per block of kFp8Block rows and columns:
// w = e4m3(code) * scale. The weight store shows such a pair as one F8 view (io/safetensors.h); a backend reads it
// into its own float format and `convert` quantizes it like any float weight (quant::read_rows).

#include <cstdint>

#include "tensor/tensor.h"

namespace xs::quant {

// The value of an E4M3 code: a sign bit, 4 exponent bits biased by 7 and 3 mantissa bits; subnormals below 2^-6, no
// infinities, 0x7f and 0xff are NaN (torch.float8_e4m3fn).
float e4m3_to_f32(uint8_t code);

// Rows [first, first + count) of an F8 weight's matrix, row-major into `out`.
void fp8_rows(const TensorView& v, int64_t first, int64_t count, float* out);

}  // namespace xs::quant
