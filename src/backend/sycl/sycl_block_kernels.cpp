#include "backend/sycl/sycl_block_kernels.h"

#include <algorithm>
#include <array>

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include "backend/sycl/sycl_kernels.h"
#include "quant/block_tables.h"
#include "util/error.h"

namespace xs::sycl_be {

namespace syclex = sycl::ext::oneapi::experimental;

namespace {

constexpr int kSubGroup = 16;
constexpr int kSgPerWg = 8;
constexpr int kRows = 2;   // output rows a sub-group computes
constexpr int kUnit = 32;  // columns a lane decodes at a time; a unit's four quarters of 8 share its loads

// What the kernels capture: the streams, the bytes a row takes in each, the table.
struct Streams {
    const uint8_t* s[kBlockStreams];
    int64_t stride[kBlockStreams];
    const uint32_t* grid;
};
struct RowPtr {
    const uint8_t* p[kBlockStreams];
};
inline RowPtr row_ptr(const Streams& w, int64_t row) {
    RowPtr r;
    for (int i = 0; i < kBlockStreams; ++i)
        r.p[i] = w.s[i] + row * w.stride[i];
    return r;
}

template <typename T>
inline T load(const uint8_t* p) {
    return *reinterpret_cast<const T*>(p);
}
inline float f16_at(const uint8_t* p) {
    return static_cast<float>(load<half>(p));
}
// 1 + code / 2^bits, written into the mantissa: a code read with no int-to-float conversion.
inline float unit_float(uint32_t code, int bits) {
    return sycl::bit_cast<float>((code << (23 - bits)) | 0x3F800000u);
}
inline float signed_float(uint32_t magnitude, uint32_t negative) {
    return sycl::bit_cast<float>(sycl::bit_cast<uint32_t>(static_cast<float>(magnitude)) | (negative << 31));
}
// The eight sign bits a 7-bit sign code stands for: the eighth makes their number even (ggml's ksigns_iq2xs).
inline uint32_t signs_of(uint32_t code7) {
    return code7 | ((sycl::popcount(code7) & 1u) << 7);
}
// ggml's kvalues_iq4nl[code], its sixteen bytes held in four words.
inline float iq4_value(uint32_t code) {
    const uint32_t hi = code >> 2;
    const uint32_t word = hi == 0 ? 0xBFAD9881u : hi == 1 ? 0xF6EADDCFu : hi == 2 ? 0x26190D01u : 0x71594535u;
    return static_cast<float>(static_cast<int8_t>(word >> (8 * (code & 3))));
}

// A group of columns: weight = s * level - m.
struct Affine {
    float s;
    float m;
};

// Scale and minimum of 32-column group j of a Q4_K / Q5_K block: twelve bytes hold eight 6-bit pairs.
inline Affine k4_affine(const uint8_t* hdr, int j) {
    const float d = f16_at(hdr);
    const float dmin = f16_at(hdr + 2);
    const uint8_t* q = hdr + 4;
    const uint32_t a = q[j];
    const uint32_t b = q[j + 4];
    const uint32_t c = q[j & 3];
    const bool lo = j < 4;
    const uint32_t sc = lo ? (a & 63u) : ((b & 0x0Fu) | ((c >> 6) << 4));
    const uint32_t mn = lo ? (b & 63u) : ((b >> 4) | ((a >> 6) << 4));
    return {d * static_cast<float>(sc), dmin * static_cast<float>(mn)};
}

// One specialization per format. Fields: what a unit (32 columns of a row) loads. codes(): the unsigned codes of
// quarter t (kTrick: weight = s * (code - kOff) - m, read through unit_float), or levels(): its signed levels as
// floats (weight = s * level). affine(): the s and m of the unit's half g, in ggml's operations; kGroups says whether
// the halves differ.
template <BlockFormat F>
struct Fmt;

template <>
struct Fmt<BlockFormat::Q4_K> {
    static constexpr bool kTrick = true;
    static constexpr int kBits = 4;
    static constexpr int kOff = 0;
    static constexpr bool kMin = true;
    static constexpr int kGroups = 1;
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        const uint8_t* hdr;
        int j;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u), r.p[1] + 16 * (u >> 3), u & 7};
    }
    static void codes(const Fields& f, int t, uint32_t* c) {
        const uint32_t w = f.lo[t];
        for (int i = 0; i < 8; ++i)
            c[i] = (w >> (4 * i)) & 15u;
    }
    static Affine affine(const Fields& f, int) { return k4_affine(f.hdr, f.j); }
};

template <>
struct Fmt<BlockFormat::Q5_K> {
    static constexpr bool kTrick = true;
    static constexpr int kBits = 5;
    static constexpr int kOff = 0;
    static constexpr bool kMin = true;
    static constexpr int kGroups = 1;
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        uint32_t hi;
        const uint8_t* hdr;
        int j;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u), load<uint32_t>(r.p[1] + 4 * u), r.p[2] + 16 * (u >> 3),
                u & 7};
    }
    static void codes(const Fields& f, int t, uint32_t* c) {
        const uint32_t w = f.lo[t];
        const uint32_t h = f.hi >> (8 * t);
        for (int i = 0; i < 8; ++i)
            c[i] = ((w >> (4 * i)) & 15u) | (((h >> i) & 1u) << 4);
    }
    static Affine affine(const Fields& f, int) { return k4_affine(f.hdr, f.j); }
};

template <>
struct Fmt<BlockFormat::Q6_K> {
    static constexpr bool kTrick = true;
    static constexpr int kBits = 6;
    static constexpr int kOff = 32;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        sycl::vec<uint32_t, 2> hi;
        uint32_t scales;  // two i8
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u), load<sycl::vec<uint32_t, 2>>(r.p[1] + 8 * u),
                load<uint16_t>(r.p[2] + 2 * u), f16_at(r.p[3] + 2 * (u >> 3))};
    }
    static void codes(const Fields& f, int t, uint32_t* c) {
        const uint32_t w = f.lo[t];
        const uint32_t h = f.hi[t >> 1] >> (16 * (t & 1));
        for (int i = 0; i < 8; ++i)
            c[i] = ((w >> (4 * i)) & 15u) | (((h >> (2 * i)) & 3u) << 4);
    }
    static Affine affine(const Fields& f, int g) {
        return {f.d * static_cast<float>(static_cast<int8_t>(f.scales >> (8 * g))), 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::Q3_K> {
    static constexpr bool kTrick = true;
    static constexpr int kBits = 3;
    static constexpr int kOff = 4;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    struct Fields {
        sycl::vec<uint32_t, 2> lo;
        uint32_t hi;
        const uint8_t* scales;  // sixteen 6-bit scales in twelve bytes
        float d;
        int j;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 2>>(r.p[0] + 8 * u), load<uint32_t>(r.p[1] + 4 * u), r.p[2] + 12 * (u >> 3),
                f16_at(r.p[3] + 2 * (u >> 3)), u & 7};
    }
    static void codes(const Fields& f, int t, uint32_t* c) {
        const uint32_t w = f.lo[t >> 1] >> (16 * (t & 1));
        const uint32_t h = f.hi >> (8 * t);
        for (int i = 0; i < 8; ++i)
            c[i] = ((w >> (2 * i)) & 3u) | (((h >> i) & 1u) << 2);
    }
    static Affine affine(const Fields& f, int g) {
        const int is = 2 * f.j + g;
        const uint32_t low = (static_cast<uint32_t>(f.scales[is & 7]) >> (4 * (is >> 3))) & 15u;
        const uint32_t high = (static_cast<uint32_t>(f.scales[8 + (is & 3)]) >> (2 * (is >> 2))) & 3u;
        return {f.d * static_cast<float>(static_cast<int32_t>(low | (high << 4)) - 32), 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::Q2_K> {
    static constexpr bool kTrick = true;
    static constexpr int kBits = 2;
    static constexpr int kOff = 0;
    static constexpr bool kMin = true;
    static constexpr int kGroups = 2;
    struct Fields {
        sycl::vec<uint32_t, 2> lo;
        uint32_t scales;  // two bytes: scale in the low nibble, minimum in the high
        float d;
        float dmin;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        const uint8_t* dd = r.p[2] + 4 * (u >> 3);
        return {load<sycl::vec<uint32_t, 2>>(r.p[0] + 8 * u), load<uint16_t>(r.p[1] + 2 * u), f16_at(dd),
                f16_at(dd + 2)};
    }
    static void codes(const Fields& f, int t, uint32_t* c) {
        const uint32_t w = f.lo[t >> 1] >> (16 * (t & 1));
        for (int i = 0; i < 8; ++i)
            c[i] = (w >> (2 * i)) & 3u;
    }
    static Affine affine(const Fields& f, int g) {
        const uint32_t sc = (f.scales >> (8 * g)) & 255u;
        return {f.d * static_cast<float>(sc & 15u), f.dmin * static_cast<float>(sc >> 4)};
    }
};

template <>
struct Fmt<BlockFormat::IQ4_NL> {
    static constexpr bool kTrick = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u), f16_at(r.p[1] + 2 * u)};
    }
    static void levels(const Fields& f, const uint32_t*, int t, float* lv) {
        const uint32_t w = f.lo[t];
        for (int i = 0; i < 8; ++i)
            lv[i] = iq4_value((w >> (4 * i)) & 15u);
    }
    static Affine affine(const Fields& f, int) { return {f.d, 0.0f}; }
};

template <>
struct Fmt<BlockFormat::IQ4_XS> {
    static constexpr bool kTrick = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        float dl;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        const uint8_t* hdr = r.p[1] + 8 * (u >> 3);  // d, scales_h, scales_l[4]
        const int j = u & 7;
        const uint32_t high = load<uint16_t>(hdr + 2);
        const uint32_t low = hdr[4 + (j >> 1)];
        const uint32_t ls = ((low >> (4 * (j & 1))) & 15u) | (((high >> (2 * j)) & 3u) << 4);
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u),
                f16_at(hdr) * static_cast<float>(static_cast<int32_t>(ls) - 32)};
    }
    static void levels(const Fields& f, const uint32_t*, int t, float* lv) {
        const uint32_t w = f.lo[t];
        for (int i = 0; i < 8; ++i)
            lv[i] = iq4_value((w >> (4 * i)) & 15u);
    }
    static Affine affine(const Fields& f, int) { return {f.dl, 0.0f}; }
};

template <>
struct Fmt<BlockFormat::IQ3_XXS> {
    static constexpr bool kTrick = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    struct Fields {
        sycl::vec<uint32_t, 2> idx;
        uint32_t aux;  // four 7-bit sign codes, then the 4-bit scale
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 2>>(r.p[0] + 8 * u), load<uint32_t>(r.p[1] + 4 * u),
                f16_at(r.p[2] + 2 * (u >> 3))};
    }
    static void levels(const Fields& f, const uint32_t* grid, int t, float* lv) {
        const uint32_t signs = signs_of((f.aux >> (7 * t)) & 127u);
        const uint32_t pair = f.idx[t >> 1] >> (16 * (t & 1));
        for (int h = 0; h < 2; ++h) {
            const uint32_t g = grid[(pair >> (8 * h)) & 255u];
            for (int i = 0; i < 4; ++i)
                lv[4 * h + i] = signed_float((g >> (8 * i)) & 255u, (signs >> (4 * h + i)) & 1u);
        }
    }
    static Affine affine(const Fields& f, int) { return {f.d * (0.5f + static_cast<float>(f.aux >> 28)) * 0.5f, 0.0f}; }
};

template <>
struct Fmt<BlockFormat::IQ3_S> {
    static constexpr bool kTrick = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    struct Fields {
        sycl::vec<uint32_t, 2> idx;
        uint32_t signs;
        uint32_t qh;     // bit 8 of each of the unit's eight indexes
        uint32_t scale;  // 4 bits
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 2>>(r.p[0] + 8 * u), load<uint32_t>(r.p[1] + 4 * u), r.p[2][u],
                (static_cast<uint32_t>(r.p[3][u >> 1]) >> (4 * (u & 1))) & 15u, f16_at(r.p[4] + 2 * (u >> 3))};
    }
    static void levels(const Fields& f, const uint32_t* grid, int t, float* lv) {
        const uint32_t signs = f.signs >> (8 * t);
        const uint32_t pair = f.idx[t >> 1] >> (16 * (t & 1));
        for (int h = 0; h < 2; ++h) {
            const uint32_t g = grid[((pair >> (8 * h)) & 255u) | (((f.qh >> (2 * t + h)) & 1u) << 8)];
            for (int i = 0; i < 4; ++i)
                lv[4 * h + i] = signed_float((g >> (8 * i)) & 255u, (signs >> (4 * h + i)) & 1u);
        }
    }
    static Affine affine(const Fields& f, int) {
        return {f.d * static_cast<float>(1 + 2 * static_cast<int32_t>(f.scale)), 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::IQ2_XXS> {
    static constexpr bool kTrick = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    struct Fields {
        uint32_t idx;
        uint32_t aux;  // four 7-bit sign codes, then the 4-bit scale
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        const auto v = load<sycl::vec<uint32_t, 2>>(r.p[0] + 8 * u);
        return {v[0], v[1], f16_at(r.p[1] + 2 * (u >> 3))};
    }
    static void levels(const Fields& f, const uint32_t* grid, int t, float* lv) {
        const uint32_t signs = signs_of((f.aux >> (7 * t)) & 127u);
        const auto g =
            load<sycl::vec<uint32_t, 2>>(reinterpret_cast<const uint8_t*>(grid) + 8 * ((f.idx >> (8 * t)) & 255u));
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < 4; ++i)
                lv[4 * h + i] = signed_float((g[h] >> (8 * i)) & 255u, (signs >> (4 * h + i)) & 1u);
        }
    }
    static Affine affine(const Fields& f, int) {
        return {f.d * (0.5f + static_cast<float>(f.aux >> 28)) * 0.25f, 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::IQ2_S> {
    static constexpr bool kTrick = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    struct Fields {
        uint32_t idx;
        uint32_t signs;
        uint32_t qh;      // bits 8-9 of each of the unit's four indexes
        uint32_t scales;  // 4 bits a half
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<uint32_t>(r.p[0] + 4 * u), load<uint32_t>(r.p[1] + 4 * u), r.p[2][u], r.p[3][u],
                f16_at(r.p[4] + 2 * (u >> 3))};
    }
    static void levels(const Fields& f, const uint32_t* grid, int t, float* lv) {
        const uint32_t signs = f.signs >> (8 * t);
        const uint32_t index = ((f.idx >> (8 * t)) & 255u) | (((f.qh >> (2 * t)) & 3u) << 8);
        const auto g = load<sycl::vec<uint32_t, 2>>(reinterpret_cast<const uint8_t*>(grid) + 8 * index);
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < 4; ++i)
                lv[4 * h + i] = signed_float((g[h] >> (8 * i)) & 255u, (signs >> (4 * h + i)) & 1u);
        }
    }
    static Affine affine(const Fields& f, int g) {
        return {f.d * (0.5f + static_cast<float>((f.scales >> (4 * g)) & 15u)) * 0.25f, 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::IQ2_XS> {
    static constexpr bool kTrick = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    struct Fields {
        sycl::vec<uint32_t, 2> q;  // four u16: a 9-bit index under a 7-bit sign code
        uint32_t scales;           // 4 bits a half
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 2>>(r.p[0] + 8 * u), r.p[1][u], f16_at(r.p[2] + 2 * (u >> 3))};
    }
    static void levels(const Fields& f, const uint32_t* grid, int t, float* lv) {
        const uint32_t q = (f.q[t >> 1] >> (16 * (t & 1))) & 0xFFFFu;
        const uint32_t signs = signs_of(q >> 9);
        const auto g = load<sycl::vec<uint32_t, 2>>(reinterpret_cast<const uint8_t*>(grid) + 8 * (q & 511u));
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < 4; ++i)
                lv[4 * h + i] = signed_float((g[h] >> (8 * i)) & 255u, (signs >> (4 * h + i)) & 1u);
        }
    }
    static Affine affine(const Fields& f, int g) {
        return {f.d * (0.5f + static_cast<float>((f.scales >> (4 * g)) & 15u)) * 0.25f, 0.0f};
    }
};

// IQ1: a grid value is -1, 0 or 1, and a group of 8 adds or subtracts an eighth to all of its own.
inline void iq1_levels(const uint32_t* grid, uint32_t index, bool minus, float* lv) {
    const auto g = load<sycl::vec<uint32_t, 2>>(reinterpret_cast<const uint8_t*>(grid) + 8 * index);
    const float delta = minus ? -quant::kIq1Delta : quant::kIq1Delta;
    for (int h = 0; h < 2; ++h) {
        for (int i = 0; i < 4; ++i)
            lv[4 * h + i] = static_cast<float>(static_cast<int8_t>(g[h] >> (8 * i))) + delta;
    }
}

template <>
struct Fmt<BlockFormat::IQ1_S> {
    static constexpr bool kTrick = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    struct Fields {
        uint32_t idx;
        uint32_t qh;  // four 3-bit index tops, the 3-bit scale, the shift's sign
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<uint32_t>(r.p[0] + 4 * u), load<uint16_t>(r.p[1] + 2 * u), f16_at(r.p[2] + 2 * (u >> 3))};
    }
    static void levels(const Fields& f, const uint32_t* grid, int t, float* lv) {
        iq1_levels(grid, ((f.idx >> (8 * t)) & 255u) | (((f.qh >> (3 * t)) & 7u) << 8), (f.qh & 0x8000u) != 0, lv);
    }
    static Affine affine(const Fields& f, int) {
        return {f.d * static_cast<float>(2 * static_cast<int32_t>((f.qh >> 12) & 7u) + 1), 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::IQ1_M> {
    static constexpr bool kTrick = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    struct Fields {
        uint32_t idx;
        uint32_t qh;      // a nibble a quarter: three index tops and the shift's sign
        uint32_t scales;  // 3 bits a half
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        // the block's f16 scale is spread over the top nibbles of its four u16 of scales
        const auto sc = load<sycl::vec<uint16_t, 4>>(r.p[2] + 8 * (u >> 3));
        const auto bits =
            static_cast<uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        const int j = u & 7;
        return {load<uint32_t>(r.p[0] + 4 * u), load<uint16_t>(r.p[1] + 2 * u),
                (static_cast<uint32_t>(sc[j >> 1]) >> (6 * (j & 1))) & 63u,
                static_cast<float>(sycl::bit_cast<half>(bits))};
    }
    static void levels(const Fields& f, const uint32_t* grid, int t, float* lv) {
        const uint32_t nibble = f.qh >> (4 * t);
        iq1_levels(grid, ((f.idx >> (8 * t)) & 255u) | ((nibble & 7u) << 8), (nibble & 8u) != 0, lv);
    }
    static Affine affine(const Fields& f, int g) {
        return {f.d * static_cast<float>(2 * static_cast<int32_t>((f.scales >> (3 * g)) & 7u) + 1), 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::Q5_0> {
    static constexpr bool kTrick = true;
    static constexpr int kBits = 5;
    static constexpr int kOff = 16;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        uint32_t hi;
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u), load<uint32_t>(r.p[1] + 4 * u), f16_at(r.p[2] + 2 * u)};
    }
    static void codes(const Fields& f, int t, uint32_t* c) {
        const uint32_t w = f.lo[t];
        const uint32_t h = f.hi >> (8 * t);
        for (int i = 0; i < 8; ++i)
            c[i] = ((w >> (4 * i)) & 15u) | (((h >> i) & 1u) << 4);
    }
    static Affine affine(const Fields& f, int) { return {f.d, 0.0f}; }
};

template <>
struct Fmt<BlockFormat::Q5_1> {
    static constexpr bool kTrick = true;
    static constexpr int kBits = 5;
    static constexpr int kOff = 0;
    static constexpr bool kMin = true;
    static constexpr int kGroups = 1;
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        uint32_t hi;
        float d;
        float m;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u), load<uint32_t>(r.p[1] + 4 * u), f16_at(r.p[2] + 4 * u),
                f16_at(r.p[2] + 4 * u + 2)};
    }
    static void codes(const Fields& f, int t, uint32_t* c) {
        const uint32_t w = f.lo[t];
        const uint32_t h = f.hi >> (8 * t);
        for (int i = 0; i < 8; ++i)
            c[i] = ((w >> (4 * i)) & 15u) | (((h >> i) & 1u) << 4);
    }
    static Affine affine(const Fields& f, int) { return {f.d, -f.m}; }  // w = d code + m
};

// Quarter t of a unit for the GEMV: wv with a and b (per half) such that weight = a * wv + b.
template <BlockFormat F>
inline void fast_quarter(const typename Fmt<F>::Fields& f, const uint32_t* grid, int t, float* wv) {
    if constexpr (Fmt<F>::kTrick) {
        uint32_t c[8];
        Fmt<F>::codes(f, t, c);
        for (int i = 0; i < 8; ++i)
            wv[i] = unit_float(c[i], Fmt<F>::kBits);
    } else {
        Fmt<F>::levels(f, grid, t, wv);
    }
}
template <BlockFormat F>
inline void fast_affine(const typename Fmt<F>::Fields& f, float (&a)[2], float (&b)[2]) {
    for (int g = 0; g < 2; ++g) {
        if (g == 1 && Fmt<F>::kGroups == 1) {
            a[1] = a[0];
            b[1] = b[0];
            break;
        }
        const Affine af = Fmt<F>::affine(f, g);
        if constexpr (Fmt<F>::kTrick) {  // s (code - off) - m with code = 2^bits (wv - 1)
            constexpr float kOne = static_cast<float>(1 << Fmt<F>::kBits);
            a[g] = af.s * kOne;
            b[g] = -af.m - af.s * (kOne + static_cast<float>(Fmt<F>::kOff));
        } else {
            a[g] = af.s;
            b[g] = 0.0f;
        }
    }
}
// The eight weights of quarter t as ggml computes them.
template <BlockFormat F>
inline void exact_quarter(const typename Fmt<F>::Fields& f, const uint32_t* grid, int t, float* v) {
    float lv[8];
    if constexpr (Fmt<F>::kTrick) {
        uint32_t c[8];
        Fmt<F>::codes(f, t, c);
        for (int i = 0; i < 8; ++i)
            lv[i] = static_cast<float>(static_cast<int32_t>(c[i]) - Fmt<F>::kOff);
    } else {
        Fmt<F>::levels(f, grid, t, lv);
    }
    const Affine af = Fmt<F>::affine(f, t >> 1);
    for (int i = 0; i < 8; ++i)
        v[i] = Fmt<F>::kMin ? af.s * lv[i] - af.m : af.s * lv[i];
}

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// A sub-group takes kRows rows; a lane takes units lane, lane + 16, ... of each, so a unit's x loads once for the
// rows and its codes decode once for the M inputs. Over a half unit dot(w, x) = a dot(wv, x) + b sum(x).
template <BlockFormat F, int M>
void gemv_block_t(sycl::queue& q, const float* x, const Streams& w, float* y, int64_t m, int64_t n, int64_t k) {
    const int64_t wgs = ceil_div(n, kSgPerWg * kRows);
    const auto units = static_cast<int32_t>(k / kUnit);
    const auto inputs = static_cast<int>(m);
    syclex::nd_launch(
        q, sycl::nd_range<1>(static_cast<size_t>(wgs * kSgPerWg * kSubGroup), kSgPerWg * kSubGroup),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = static_cast<int>(sg.get_local_linear_id());
            const int64_t row0 =
                (static_cast<int64_t>(it.get_group(0)) * kSgPerWg + static_cast<int64_t>(sg.get_group_linear_id())) *
                kRows;
            RowPtr rp[kRows];
            for (int r = 0; r < kRows; ++r)
                rp[r] = row_ptr(w, std::min<int64_t>(row0 + r, n - 1));
            const float* xs[M];
            for (int mm = 0; mm < M; ++mm)
                xs[mm] = x + std::min(mm, inputs - 1) * k;  // unused inputs repeat the last
            float acc[kRows][M] = {};
            for (int32_t u = lane; u < units; u += kSubGroup) {
                float wv[kRows][kUnit];
                float a[kRows][2];
                float b[kRows][2];
                for (int r = 0; r < kRows; ++r) {
                    const auto f = Fmt<F>::fields(rp[r], u);
                    for (int t = 0; t < 4; ++t)
                        fast_quarter<F>(f, w.grid, t, &wv[r][8 * t]);
                    fast_affine<F>(f, a[r], b[r]);
                }
                for (int mm = 0; mm < M; ++mm) {
                    float xv[kUnit];
                    const auto* xp =
                        reinterpret_cast<const sycl::vec<float, 4>*>(xs[mm] + static_cast<int64_t>(u) * kUnit);
                    for (int i = 0; i < kUnit / 4; ++i) {
                        const sycl::vec<float, 4> v = xp[i];
                        for (int j = 0; j < 4; ++j)
                            xv[4 * i + j] = v[j];
                    }
                    float xsum[2] = {0.0f, 0.0f};
                    if constexpr (Fmt<F>::kTrick) {
                        for (int i = 0; i < kUnit; ++i)
                            xsum[i >> 4] += xv[i];
                    }
                    for (int r = 0; r < kRows; ++r) {
                        float dot[2] = {0.0f, 0.0f};
                        for (int i = 0; i < kUnit; ++i)
                            dot[i >> 4] += wv[r][i] * xv[i];
                        acc[r][mm] += a[r][0] * dot[0] + a[r][1] * dot[1];
                        if constexpr (Fmt<F>::kTrick) acc[r][mm] += b[r][0] * xsum[0] + b[r][1] * xsum[1];
                    }
                }
            }
            for (int r = 0; r < kRows; ++r) {
                for (int mm = 0; mm < M; ++mm) {
                    const float sum = sycl::reduce_over_group(sg, acc[r][mm], sycl::plus<float>());
                    if (lane == 0 && row0 + r < n && mm < inputs) y[mm * n + row0 + r] = sum;
                }
            }
        });
}

template <BlockFormat F>
void gemv_block_f(sycl::queue& q, const float* x, const Streams& w, float* y, int64_t m, int64_t n, int64_t k) {
    if (m == 1) return gemv_block_t<F, 1>(q, x, w, y, m, n, k);
    if (m == 2) return gemv_block_t<F, 2>(q, x, w, y, m, n, k);
    if (m <= 4) return gemv_block_t<F, 4>(q, x, w, y, m, n, k);
    gemv_block_t<F, 8>(q, x, w, y, m, n, k);
}

// One work item per 8 columns: neighbouring items write neighbouring bytes, so a sub-group's stores cover whole
// cache lines (sycl_kernels.cpp dequantize_t).
template <BlockFormat F, typename T>
void dequantize_block_t(sycl::queue& q, const Streams& w, T* out, int64_t n, int64_t k) {
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(n), static_cast<size_t>(k / 8)), [=](sycl::item<2> it) {
        const auto r = static_cast<int64_t>(it[0]);
        const auto c8 = static_cast<int32_t>(it[1]);
        const auto f = Fmt<F>::fields(row_ptr(w, r), c8 >> 2);
        float v[8];
        exact_quarter<F>(f, w.grid, c8 & 3, v);
        sycl::vec<T, 8> o;
        for (int j = 0; j < 8; ++j)
            o[j] = static_cast<T>(v[j]);
        *reinterpret_cast<sycl::vec<T, 8>*>(out + r * k + static_cast<int64_t>(c8) * 8) = o;
    });
}

template <BlockFormat F, typename I>
void gather_block_t(sycl::queue& q, const Streams& w, const I* ids, float* out, int64_t count, int64_t width) {
    syclex::parallel_for(q, sycl::range<2>(static_cast<size_t>(count), static_cast<size_t>(width / 8)),
                         [=](sycl::item<2> it) {
                             const auto i = static_cast<int64_t>(it[0]);
                             const auto c8 = static_cast<int32_t>(it[1]);
                             const auto f = Fmt<F>::fields(row_ptr(w, static_cast<int64_t>(ids[i])), c8 >> 2);
                             float v[8];
                             exact_quarter<F>(f, w.grid, c8 & 3, v);
                             float* o = out + i * width + static_cast<int64_t>(c8) * 8;
                             for (int j = 0; j < 8; ++j)
                                 o[j] = v[j];
                         });
}

Streams streams_of(const BlockWeight& w, int64_t k) {
    XS_CHECK(block_on_device(w.format), "sycl: no kernel reads the block format ", block_format_name(w.format));
    XS_CHECK(k % 256 == 0, "sycl: a ", block_format_name(w.format), " weight needs a multiple of 256 columns, got ", k);
    Streams s;
    for (int i = 0; i < kBlockStreams; ++i)
        s.s[i] = w.s[i];
    block_stream_rows(w.format, k, s.stride);
    s.grid = w.grid;
    return s;
}

// f.template operator()<F>() for the format.
template <typename Fn>
void dispatch(BlockFormat format, Fn&& f) {
    switch (format) {
        case BlockFormat::Q4_K:
            return f.template operator()<BlockFormat::Q4_K>();
        case BlockFormat::Q5_K:
            return f.template operator()<BlockFormat::Q5_K>();
        case BlockFormat::Q6_K:
            return f.template operator()<BlockFormat::Q6_K>();
        case BlockFormat::Q3_K:
            return f.template operator()<BlockFormat::Q3_K>();
        case BlockFormat::Q2_K:
            return f.template operator()<BlockFormat::Q2_K>();
        case BlockFormat::IQ4_NL:
            return f.template operator()<BlockFormat::IQ4_NL>();
        case BlockFormat::IQ4_XS:
            return f.template operator()<BlockFormat::IQ4_XS>();
        case BlockFormat::IQ3_XXS:
            return f.template operator()<BlockFormat::IQ3_XXS>();
        case BlockFormat::IQ3_S:
            return f.template operator()<BlockFormat::IQ3_S>();
        case BlockFormat::IQ2_XXS:
            return f.template operator()<BlockFormat::IQ2_XXS>();
        case BlockFormat::IQ2_S:
            return f.template operator()<BlockFormat::IQ2_S>();
        case BlockFormat::IQ2_XS:
            return f.template operator()<BlockFormat::IQ2_XS>();
        case BlockFormat::IQ1_S:
            return f.template operator()<BlockFormat::IQ1_S>();
        case BlockFormat::IQ1_M:
            return f.template operator()<BlockFormat::IQ1_M>();
        case BlockFormat::Q5_0:
            return f.template operator()<BlockFormat::Q5_0>();
        case BlockFormat::Q5_1:
            return f.template operator()<BlockFormat::Q5_1>();
        default:
            fail("sycl: no kernel reads the block format ", block_format_name(format));
    }
}

template <size_t N>
std::span<const uint32_t> words(const std::array<uint32_t, N>& table) {
    return {table.data(), N};
}
// A u64 entry as two words, the low one first: the layout of the table's bytes on the (little-endian) device.
template <size_t N>
std::span<const uint32_t> words(const std::array<uint64_t, N>& table) {
    return {reinterpret_cast<const uint32_t*>(table.data()), 2 * N};
}

}  // namespace

std::span<const uint32_t> block_grid(BlockFormat f) {
    switch (f) {
        case BlockFormat::IQ3_XXS:
            return words(quant::iq3xxs_grid);
        case BlockFormat::IQ3_S:
            return words(quant::iq3s_grid);
        case BlockFormat::IQ2_XXS:
            return words(quant::iq2xxs_grid);
        case BlockFormat::IQ2_S:
            return words(quant::iq2s_grid);
        case BlockFormat::IQ2_XS:
            return words(quant::iq2xs_grid);
        case BlockFormat::IQ1_S:
        case BlockFormat::IQ1_M:
            return words(quant::iq1s_grid);
        default:
            return {};
    }
}

void gemv_block(sycl::queue& q, const float* x, const BlockWeight& w, float* y, int64_t m, int64_t n, int64_t k) {
    XS_CHECK(m >= 1 && m <= kGemvMaxRows, "sycl gemv_block: ", m, " rows");
    const Streams s = streams_of(w, k);
    dispatch(w.format, [&]<BlockFormat F>() { gemv_block_f<F>(q, x, s, y, m, n, k); });
}

template <typename T>
void dequantize_block(sycl::queue& q, const BlockWeight& w, T* out, int64_t n, int64_t k) {
    const Streams s = streams_of(w, k);
    dispatch(w.format, [&]<BlockFormat F>() { dequantize_block_t<F, T>(q, s, out, n, k); });
}

template <typename I>
void gather_block(sycl::queue& q, const BlockWeight& table, const I* ids, float* out, int64_t count, int64_t width) {
    const Streams s = streams_of(table, width);
    dispatch(table.format, [&]<BlockFormat F>() { gather_block_t<F, I>(q, s, ids, out, count, width); });
}

template void dequantize_block<half>(sycl::queue&, const BlockWeight&, half*, int64_t, int64_t);
template void dequantize_block<float>(sycl::queue&, const BlockWeight&, float*, int64_t, int64_t);
template void gather_block<int32_t>(sycl::queue&, const BlockWeight&, const int32_t*, float*, int64_t, int64_t);
template void gather_block<int64_t>(sycl::queue&, const BlockWeight&, const int64_t*, float*, int64_t, int64_t);

}  // namespace xs::sycl_be
