#include "quant/fp8.h"

#include <algorithm>
#include <array>
#include <limits>

#include "quant/quant.h"
#include "util/error.h"

namespace xs::quant {

namespace {

constexpr float pow2(int e) {
    float v = 1.0f;
    for (; e > 0; --e) v *= 2.0f;
    for (; e < 0; ++e) v *= 0.5f;
    return v;
}

constexpr std::array<float, 256> e4m3_table() {
    std::array<float, 256> t{};
    for (int code = 0; code < 256; ++code) {
        const int exponent = (code >> 3) & 0xF;
        const int mantissa = code & 0x7;
        float v = exponent == 0 ? static_cast<float>(mantissa) * pow2(-9)
                                : static_cast<float>(8 + mantissa) * pow2(exponent - 10);
        if (exponent == 15 && mantissa == 7) v = std::numeric_limits<float>::quiet_NaN();
        t[static_cast<size_t>(code)] = (code & 0x80) != 0 ? -v : v;
    }
    return t;
}

constexpr std::array<float, 256> kE4m3 = e4m3_table();

}  // namespace

float e4m3_to_f32(uint8_t code) { return kE4m3[code]; }

void fp8_rows(const TensorView& v, int64_t first, int64_t count, float* out) {
    XS_CHECK(v.dtype == DType::F8 && v.shape.size() >= 2 && !v.fp8.empty(),
             "an fp8 weight needs its codes paired with their scales (the weight store does that), got ",
             dtype_name(v.dtype), " ", dims_to_string(v.shape));
    const int64_t cols = v.shape.back();
    XS_CHECK(first >= 0 && count >= 0 && first + count <= matrix_rows(v.shape), "fp8: rows ", first, " to ",
             first + count, " of a ", dims_to_string(v.shape), " weight");
    const int64_t grid_cols = (cols + kFp8Block - 1) / kFp8Block;
    int64_t begin = 0;  // the first row of `run`
    for (const Fp8Run& run : v.fp8) {
        const int64_t end = std::min(first + count, begin + run.rows);
        for (int64_t r = std::max(first, begin); r < end; ++r) {
            const uint8_t* codes = run.codes + (r - begin) * cols;
            const float* scales = run.scales + (r - begin) / kFp8Block * grid_cols;
            float* o = out + (r - first) * cols;
            for (int64_t c0 = 0; c0 < cols; c0 += kFp8Block) {
                const float scale = scales[c0 / kFp8Block];
                const int64_t c1 = std::min(cols, c0 + kFp8Block);
                for (int64_t c = c0; c < c1; ++c) o[c] = kE4m3[codes[c]] * scale;
            }
        }
        begin += run.rows;
    }
}

}  // namespace xs::quant
