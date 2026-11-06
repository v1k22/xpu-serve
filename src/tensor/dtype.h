#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace xs {

enum class DType : uint8_t { F32, F16, BF16, I64, I32, I8, U8, Bool };

size_t dtype_size(DType t);
std::string_view dtype_name(DType t);  // "f32", "f16", "bf16", ...
// Parses a safetensors dtype string ("F32", "F16", "BF16", "I64", "I32", "I8", "U8", "BOOL"); throws on others.
DType dtype_from_safetensors(std::string_view s);
bool is_float(DType t);

}  // namespace xs
