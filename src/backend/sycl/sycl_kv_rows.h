#pragma once

// How the per-row attention kernel (attention_rows, sycl_kernels.cpp) reads the keys or the values of a KV cache: one
// reader per storage format. A sub-group of 16 lanes covers a row of D = 16 * DPL values, each lane DPL of them in
// registers. A reader says which (`dim`), and
//   load(row, lane, x)       reads the lane's values of a row into x and returns the factor they are still to be
//                            multiplied by (a row's scale, applied once to what is computed from x)
//   to_cache / from_cache    a query brought into the domain the rows are stored in, a sum of rows brought out of it
// `row` counts (key, KV head) pairs.

#include <cstdint>

#include <sycl/sycl.hpp>

#include "backend/sycl/sycl_kv_turbo.h"

namespace xs::sycl_be {

inline constexpr int kRowLanes = 16;

// Float values [rows, D], f16 or f32; lane l holds the values l, l + 16, ...
template <typename T, int DPL>
struct FloatRows {
    const T* data = nullptr;

    static int64_t dim(int lane, int e) { return lane + kRowLanes * e; }
    float load(int64_t row, int lane, float* x) const {
        const T* r = data + row * (DPL * kRowLanes) + lane;
        for (int e = 0; e < DPL; ++e)
            x[e] = static_cast<float>(r[kRowLanes * e]);
        return 1.0f;
    }
    static void to_cache(sycl::sub_group, int, float*) {}
    static void from_cache(sycl::sub_group, int, float*) {}
};

// int8 codes [rows, D] and an f32 scale per row. From 4 values a lane, each lane holds runs of 4 consecutive bytes:
// one 32-bit load, 64 bytes a sub-group.
template <int DPL>
struct I8Rows {
    static constexpr bool kPacked = DPL % 4 == 0;
    const int8_t* codes = nullptr;
    const float* scales = nullptr;

    static int64_t dim(int lane, int e) {
        if constexpr (kPacked) {
            return (e / 4) * (4 * kRowLanes) + lane * 4 + e % 4;
        } else {
            return lane + kRowLanes * e;
        }
    }
    float load(int64_t row, int lane, float* x) const {
        const int8_t* r = codes + row * (DPL * kRowLanes);
        if constexpr (kPacked) {
            for (int g = 0; g < DPL / 4; ++g) {
                const sycl::vec<int8_t, 4> c =
                    *reinterpret_cast<const sycl::vec<int8_t, 4>*>(r + g * 4 * kRowLanes + lane * 4);
                for (int i = 0; i < 4; ++i)
                    x[g * 4 + i] = static_cast<float>(c[i]);
            }
        } else {
            for (int e = 0; e < DPL; ++e)
                x[e] = static_cast<float>(r[lane + kRowLanes * e]);
        }
        return scales[row];
    }
    static void to_cache(sycl::sub_group, int, float*) {}
    static void from_cache(sycl::sub_group, int, float*) {}
};

// A TurboQuant format (quant/kv_turbo.h): D / 128 groups a row, each its codes and an f16 norm; lane l holds the
// values 8 l .. 8 l + 7 of every group. The rows are stored rotated.
template <int Bits, int DPL>
struct TurboRows {
    static constexpr int kGroups = DPL / turbo::kLaneValues;
    const uint8_t* codes = nullptr;
    const half* norms = nullptr;

    static int64_t dim(int lane, int e) {
        return (e / turbo::kLaneValues) * turbo::kGroup + lane * turbo::kLaneValues + e % turbo::kLaneValues;
    }
    float load(int64_t row, int lane, float* x) const {
        for (int g = 0; g < kGroups; ++g) {
            const int64_t group = row * kGroups + g;
            const auto c = turbo::LaneCodes<Bits>::load(codes + group * turbo::group_bytes(Bits), lane);
            const float norm = static_cast<float>(norms[group]);
            for (int e = 0; e < turbo::kLaneValues; ++e)
                x[g * turbo::kLaneValues + e] = turbo::kTables<Bits>.centroid[c.code(e)] * norm;
        }
        return 1.0f;
    }
    static void to_cache(sycl::sub_group sg, int lane, float* x) {
        for (int g = 0; g < kGroups; ++g)
            turbo::rotate(sg, lane, x + g * turbo::kLaneValues, false);
    }
    static void from_cache(sycl::sub_group sg, int lane, float* x) {
        for (int g = 0; g < kGroups; ++g)
            turbo::rotate(sg, lane, x + g * turbo::kLaneValues, true);
    }
};

}  // namespace xs::sycl_be
