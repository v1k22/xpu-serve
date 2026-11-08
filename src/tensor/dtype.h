#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace xs {

// Q8, Q4, Q3 and Q2 are quantized weights from an xpu-serve pack (DESIGN.md section 6.3): logically a float matrix,
// stored as codes plus per-group f16 scales (and, for Q4 / Q3 / Q2, f16 biases) that a TensorView's `quant` points at.
enum class DType : uint8_t { F32, F16, BF16, I64, I32, I8, U8, Bool, Q8, Q4, Q3, Q2 };

// Bytes per element; refused for Q4 (two codes per byte): use storage_bytes.
size_t dtype_size(DType t);
// Bytes that `numel` elements of `t` occupy.
size_t storage_bytes(DType t, int64_t numel);
std::string_view dtype_name(DType t);  // "f32", "f16", "bf16", ...
// Parses a safetensors dtype string ("F32", "F16", "BF16", "I64", "I32", "I8", "U8", "BOOL"); throws on others.
DType dtype_from_safetensors(std::string_view s);
bool is_float(DType t);
bool is_quantized(DType t);
// Bits per code of a quantized dtype (8, 4, 3 or 2); the affine ones (all but Q8) carry a bias per group.
int quant_bits(DType t);

}  // namespace xs
