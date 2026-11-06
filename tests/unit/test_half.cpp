#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <limits>

#include "util/half.h"

TEST_CASE("f16 round trip is exact for every finite f16 value") {
    for (uint32_t h = 0; h < 0x10000; ++h) {
        const auto v = static_cast<uint16_t>(h);
        const float f = xs::f16_to_f32(v);
        if (std::isnan(f)) {
            CHECK(std::isnan(xs::f16_to_f32(xs::f32_to_f16(f))));
            continue;
        }
        CHECK(xs::f32_to_f16(f) == v);
    }
}

TEST_CASE("f32 to f16 rounds to nearest even and saturates to inf") {
    CHECK(xs::f16_to_f32(xs::f32_to_f16(65504.0f)) == 65504.0f);
    CHECK(std::isinf(xs::f16_to_f32(xs::f32_to_f16(65520.0f))));
    CHECK(xs::f16_to_f32(xs::f32_to_f16(65519.0f)) == 65504.0f);
    CHECK(xs::f32_to_f16(1.0f + 0.5f / 1024.0f) == xs::f32_to_f16(1.0f));                 // tie -> even
    CHECK(xs::f32_to_f16(1.0f + 1.5f / 1024.0f) == static_cast<uint16_t>(0x3c00 + 2));    // tie -> even (up)
    CHECK(xs::f16_to_f32(xs::f32_to_f16(std::ldexp(1.0f, -24))) == std::ldexp(1.0f, -24));  // smallest subnormal
    CHECK(xs::f32_to_f16(std::ldexp(1.0f, -25)) == 0);                                     // tie -> zero
    CHECK(xs::f16_to_f32(xs::f32_to_f16(std::ldexp(1.5f, -25))) == std::ldexp(1.0f, -24));
}

TEST_CASE("bf16 conversions") {
    CHECK(xs::bf16_to_f32(xs::f32_to_bf16(1.0f)) == 1.0f);
    CHECK(xs::bf16_to_f32(xs::f32_to_bf16(39.191836f)) == 39.25f);  // sqrt(1536) in bf16, as Gemma's embed scale
    CHECK(std::isnan(xs::bf16_to_f32(xs::f32_to_bf16(std::numeric_limits<float>::quiet_NaN()))));
    const uint16_t big = xs::f32_to_bf16(1.0e6f);
    CHECK(xs::count_f16_overflow(&big, 1) == 1);
}
