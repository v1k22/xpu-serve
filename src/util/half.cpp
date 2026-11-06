#include "util/half.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace xs {

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: normalize
            int e = -1;
            do {
                ++e;
                mant <<= 1;
            } while ((mant & 0x400u) == 0);
            bits = sign | static_cast<uint32_t>(127 - 15 - e) << 23 | (mant & 0x3ffu) << 13;
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | (exp + (127 - 15)) << 23 | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t abs = bits & 0x7fffffffu;
    if (abs > 0x7f800000u) return static_cast<uint16_t>(sign | 0x7e00u);  // NaN
    if (abs >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7c00u);  // rounds to >= 65520: inf
    if (abs < 0x38800000u) {                                              // below the f16 normal range
        if (abs < 0x33000000u) return sign;                               // rounds to zero
        const uint32_t exp = abs >> 23;
        const uint32_t mant = (abs & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126 - exp;  // 14..24
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1);
        const uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    const uint32_t rounded = abs + 0xfffu + ((abs >> 13) & 1u);
    return static_cast<uint16_t>(sign | ((rounded - 0x38000000u) >> 13));
}

namespace {
template <typename Fn>
void parallel_chunks(size_t n, Fn&& fn) {
    constexpr size_t kMinPerThread = 1u << 20;
    const size_t hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t threads = std::min(hw, std::max<size_t>(1, n / kMinPerThread));
    if (threads <= 1) {
        fn(size_t{0}, n);
        return;
    }
    std::vector<std::thread> pool;
    const size_t chunk = (n + threads - 1) / threads;
    for (size_t t = 0; t < threads; ++t) {
        const size_t begin = t * chunk;
        const size_t end = std::min(n, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([&fn, begin, end] { fn(begin, end); });
    }
    for (auto& th : pool) th.join();
}
}  // namespace

void convert_bf16_to_f16(const uint16_t* src, uint16_t* dst, size_t n) {
    parallel_chunks(n, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) dst[i] = f32_to_f16(bf16_to_f32(src[i]));
    });
}

void convert_bf16_to_f32(const uint16_t* src, float* dst, size_t n) {
    parallel_chunks(n, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) dst[i] = bf16_to_f32(src[i]);
    });
}

void convert_f16_to_f32(const uint16_t* src, float* dst, size_t n) {
    parallel_chunks(n, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) dst[i] = f16_to_f32(src[i]);
    });
}

void convert_f32_to_f16(const float* src, uint16_t* dst, size_t n) {
    parallel_chunks(n, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) dst[i] = f32_to_f16(src[i]);
    });
}

size_t count_f16_overflow(const uint16_t* src, size_t n) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        const float f = bf16_to_f32(src[i]);
        if (std::isfinite(f) && std::fabs(f) >= 65520.0f) ++count;
    }
    return count;
}

}  // namespace xs
