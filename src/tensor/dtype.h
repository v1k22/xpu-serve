#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace xs {

// Q8 and Q4 are quantized weights from an xpu-serve pack (DESIGN.md section 6.3): logically a float matrix, stored as
// codes plus per-group f16 scales (and, for Q4, f16 biases) that a TensorView's `quant` points at.
// F8 is a weight of an FP8 checkpoint: logically a float matrix, stored as E4M3 codes (a byte each) times one scale
// per block of rows and columns; a TensorView's `fp8` holds its rows, and quant::read_rows reads them.
enum class DType : uint8_t { F32, F16, BF16, I64, I32, I8, U8, Bool, Q8, Q4, F8 };

// Bytes per element; refused for Q4 (two codes per byte): use storage_bytes.
size_t dtype_size(DType t);
// Bytes that `numel` elements of `t` occupy.
size_t storage_bytes(DType t, int64_t numel);
std::string_view dtype_name(DType t);  // "f32", "f16", "bf16", ...
// Parses a safetensors dtype string ("F32", "F16", "BF16", "I64", "I32", "I8", "U8", "BOOL", "F8_E4M3"); throws on
// others.
DType dtype_from_safetensors(std::string_view s);
bool is_float(DType t);      // f32, f16, bf16: float values, element by element
bool is_quantized(DType t);  // q8, q4: a pack's formats

}  // namespace xs
