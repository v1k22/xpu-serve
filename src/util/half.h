#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace xs {

// IEEE binary16 and bfloat16 conversions. Round-to-nearest-even; NaN stays NaN; values beyond the f16 range become
// +-inf (callers that care must check, see count_f16_overflow).

inline float bf16_to_f32(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((bits >> 16) | 0x0040u);  // quiet NaN
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);

// Bulk conversions (multi-threaded for large inputs).
void convert_bf16_to_f16(const uint16_t* src, uint16_t* dst, size_t n);
void convert_bf16_to_f32(const uint16_t* src, float* dst, size_t n);
void convert_f16_to_f32(const uint16_t* src, float* dst, size_t n);
void convert_f32_to_f16(const float* src, uint16_t* dst, size_t n);

// Number of finite bf16 values in src whose magnitude does not fit in f16 (they would become inf).
size_t count_f16_overflow(const uint16_t* bf16_src, size_t n);

}  // namespace xs
