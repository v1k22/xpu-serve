#pragma once

// The TurboQuant KV formats inside a kernel (quant/kv_turbo.h): a sub-group of 16 lanes holds one group of 128
// values, lane l the values 8 l .. 8 l + 7 in registers. Shared by the kernels that write a cache in these formats,
// decode it and attend over it.

#include <cstdint>

#include <sycl/sycl.hpp>

#include "quant/kv_turbo.h"

namespace xs::sycl_be::turbo {

using namespace xs::quant::turbo;

// The same value in every lane: the lanes' v added pairwise, lanes 1 apart, then 2, 4 and 8 apart.
inline float sum_lanes(sycl::sub_group sg, float v) {
    for (int m = 1; m < kLanes; m <<= 1)
        v += sycl::permute_group_by_xor(sg, v, m);
    return v;
}

// v[e] = -v[e] where bit e of `signs` is set.
inline void flip(float* v, uint32_t signs) {
    for (int e = 0; e < kLaneValues; ++e)
        v[e] = ((signs >> e) & 1u) != 0 ? -v[e] : v[e];
}

// The Walsh-Hadamard transform of the sub-group's group, unnormalized: the spans inside a lane, then between lanes.
inline void butterflies(sycl::sub_group sg, int lane, float* v) {
    for (int h = 1; h < kLaneValues; h *= 2) {
        for (int i = 0; i < kLaneValues; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                const float a = v[j];
                const float b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
    for (int m = 1; m < kLanes; m <<= 1) {
        for (int e = 0; e < kLaneValues; ++e) {
            const float other = sycl::permute_group_by_xor(sg, v[e], m);
            v[e] = (lane & m) != 0 ? other - v[e] : v[e] + other;
        }
    }
}

// The sub-group's group rotated in place, or rotated back.
inline void rotate(sycl::sub_group sg, int lane, float* v, bool inverse) {
    const uint32_t before = lane_signs(kSignsBefore[0], kSignsBefore[1], lane);
    const uint32_t after = lane_signs(kSignsAfter[0], kSignsAfter[1], lane);
    flip(v, inverse ? after : before);
    butterflies(sg, lane, v);
    for (int e = 0; e < kLaneValues; ++e)
        v[e] *= kInvSqrtGroup;
    flip(v, inverse ? before : after);
}

// A lane's 8 codes of a group, as the bytes that hold them (quant/kv_turbo.h's layout).
template <int Bits>
struct LaneCodes {
    uint32_t low = 0;   // 4 bits: the codes, a nibble each. 3 and 2 bits: each code's low two bits
    uint32_t high = 0;  // 3 bits: each code's third bit

    static LaneCodes load(const uint8_t* group, int lane) {
        LaneCodes c;
        if constexpr (Bits == 4) {
            c.low = *reinterpret_cast<const uint32_t*>(group + 4 * lane);
        } else {
            c.low = *reinterpret_cast<const uint16_t*>(group + 2 * lane);
            if constexpr (Bits == 3) c.high = group[kGroup / 4 + lane];
        }
        return c;
    }
    void store(uint8_t* group, int lane) const {
        if constexpr (Bits == 4) {
            *reinterpret_cast<uint32_t*>(group + 4 * lane) = low;
        } else {
            *reinterpret_cast<uint16_t*>(group + 2 * lane) = static_cast<uint16_t>(low);
            if constexpr (Bits == 3) group[kGroup / 4 + lane] = static_cast<uint8_t>(high);
        }
    }
    uint32_t code(int e) const {
        if constexpr (Bits == 4) {
            return (low >> (4 * e)) & 15u;
        } else if constexpr (Bits == 3) {
            return ((low >> (2 * e)) & 3u) | (((high >> e) & 1u) << 2);
        } else {
            return (low >> (2 * e)) & 3u;
        }
    }
    void set(int e, uint32_t code) {
        if constexpr (Bits == 4) {
            low |= code << (4 * e);
        } else {
            low |= (code & 3u) << (2 * e);
            high |= (code >> 2) << e;
        }
    }
};

}  // namespace xs::sycl_be::turbo
