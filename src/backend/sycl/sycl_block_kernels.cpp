#include "backend/sycl/sycl_block_kernels.h"

#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include "backend/sycl/sycl_kernels.h"
#include "backend/sycl/sycl_xmx.h"
#include "quant/block_tables.h"
#include "util/error.h"

namespace xs::sycl_be {

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

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

// The value of a 4-bit float code (E2M1), doubled: 0, 1, 2, 3, 4, 6, 8, 12 by its low three bits, a nibble each, and
// its sign.
inline float fp4_level(uint32_t code) {
    return signed_float((0xC8643210u >> (4 * (code & 7u))) & 15u, code >> 3);
}
// Half of an unsigned E4M3 code, as ggml reads an NVFP4 scale: (1 + m / 8) 2^(e - 8), below e = 1 the subnormal
// m 2^-10; 0 and 0x7f are 0, the top bit is not looked at.
inline float ue4m3_half(uint32_t x) {
    const uint32_t e = (x >> 3) & 15u;
    const uint32_t m = x & 7u;
    const float normal = sycl::bit_cast<float>(((e + 119u) << 23) | (m << 20));
    const float v = e == 0 ? static_cast<float>(m) * 0x1p-10f : normal;
    return x == 0x7Fu ? 0.0f : v;
}
// 2^(x - 128), ggml's power of two of an MXFP4 block (a subnormal below x = 2).
inline float e8m0_half(uint32_t x) {
    return sycl::bit_cast<float>(x < 2 ? 0x00200000u << x : (x - 1u) << 23);
}

// The two words of an 8-byte entry of a table of u64.
inline void grid_pair(const uint32_t* grid, uint32_t index, uint32_t* g) {
    const auto v = load<sycl::vec<uint32_t, 2>>(reinterpret_cast<const uint8_t*>(grid) + 8 * index);
    g[0] = v[0];
    g[1] = v[1];
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

// One specialization per format. kSuperblock: the bytes 256 columns of a row take in each stream (the layout's,
// sycl_block_layout.h). Fields: what a unit (32 columns of a row) loads, read at r.p[i] + bytes * u in stream i, or at
// the start of its block's run of a stream: so a copy of a block's run of each stream reads as the streams do with u
// below 8. The values of quarter t (8 columns)
// come in one of three forms. kTrick: unsigned codes, weight = s * (code - kOff) - m, their kLowBits from low() and
// kHighBits from high() (code i in bits [L i, L i + L) of each). kGrid: magnitudes from the format's table, four bytes
// a word for columns 0-3 and 4-7 (grid()), with the eight sign bits beside them; weight = s * level. Otherwise
// levels(): the signed levels as floats. affine(): the s and m of the unit's half g, in ggml's operations; kGroups says
// whether the halves differ.
template <BlockFormat F>
struct Fmt;

template <>
struct Fmt<BlockFormat::Q4_K> {
    static constexpr bool kTrick = true;
    static constexpr bool kGrid = false;
    static constexpr int kLowBits = 4;
    static constexpr int kHighBits = 0;
    static constexpr int kOff = 0;
    static constexpr bool kMin = true;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {128, 16};
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        const uint8_t* hdr;
        int j;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u), r.p[1] + 16 * (u >> 3), u & 7};
    }
    static uint32_t low(const Fields& f, int t) { return f.lo[t]; }
    static uint32_t high(const Fields&, int) { return 0; }
    static Affine affine(const Fields& f, int) { return k4_affine(f.hdr, f.j); }
};

template <>
struct Fmt<BlockFormat::Q5_K> {
    static constexpr bool kTrick = true;
    static constexpr bool kGrid = false;
    static constexpr int kLowBits = 4;
    static constexpr int kHighBits = 1;
    static constexpr int kOff = 0;
    static constexpr bool kMin = true;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {128, 32, 16};
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
    static uint32_t low(const Fields& f, int t) { return f.lo[t]; }
    static uint32_t high(const Fields& f, int t) { return f.hi >> (8 * t); }
    static Affine affine(const Fields& f, int) { return k4_affine(f.hdr, f.j); }
};

template <>
struct Fmt<BlockFormat::Q6_K> {
    static constexpr bool kTrick = true;
    static constexpr bool kGrid = false;
    static constexpr int kLowBits = 4;
    static constexpr int kHighBits = 2;
    static constexpr int kOff = 32;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {128, 64, 16, 2};
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
    static uint32_t low(const Fields& f, int t) { return f.lo[t]; }
    static uint32_t high(const Fields& f, int t) { return f.hi[t >> 1] >> (16 * (t & 1)); }
    static Affine affine(const Fields& f, int g) {
        return {f.d * static_cast<float>(static_cast<int8_t>(f.scales >> (8 * g))), 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::Q3_K> {
    static constexpr bool kTrick = true;
    static constexpr bool kGrid = false;
    static constexpr int kLowBits = 2;
    static constexpr int kHighBits = 1;
    static constexpr int kOff = 4;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {64, 32, 12, 2};
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
    static uint32_t low(const Fields& f, int t) { return f.lo[t >> 1] >> (16 * (t & 1)); }
    static uint32_t high(const Fields& f, int t) { return f.hi >> (8 * t); }
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
    static constexpr bool kGrid = false;
    static constexpr int kLowBits = 2;
    static constexpr int kHighBits = 0;
    static constexpr int kOff = 0;
    static constexpr bool kMin = true;
    static constexpr int kGroups = 2;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {64, 16, 4};
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
    static uint32_t low(const Fields& f, int t) { return f.lo[t >> 1] >> (16 * (t & 1)); }
    static uint32_t high(const Fields&, int) { return 0; }
    static Affine affine(const Fields& f, int g) {
        const uint32_t sc = (f.scales >> (8 * g)) & 255u;
        return {f.d * static_cast<float>(sc & 15u), f.dmin * static_cast<float>(sc >> 4)};
    }
};

template <>
struct Fmt<BlockFormat::IQ4_NL> {
    static constexpr bool kTrick = false;
    static constexpr bool kGrid = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {128, 16};
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
    static constexpr bool kGrid = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {128, 8};
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
    static constexpr bool kGrid = true;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {64, 32, 2};
    struct Fields {
        sycl::vec<uint32_t, 2> idx;
        uint32_t aux;  // four 7-bit sign codes, then the 4-bit scale
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 2>>(r.p[0] + 8 * u), load<uint32_t>(r.p[1] + 4 * u),
                f16_at(r.p[2] + 2 * (u >> 3))};
    }
    static uint32_t grid_words(const Fields& f, const uint32_t* grid, int t, uint32_t* g) {
        const uint32_t pair = f.idx[t >> 1] >> (16 * (t & 1));
        g[0] = grid[pair & 255u];
        g[1] = grid[(pair >> 8) & 255u];
        return signs_of((f.aux >> (7 * t)) & 127u);
    }
    static Affine affine(const Fields& f, int) { return {f.d * (0.5f + static_cast<float>(f.aux >> 28)) * 0.5f, 0.0f}; }
};

template <>
struct Fmt<BlockFormat::IQ3_S> {
    static constexpr bool kTrick = false;
    static constexpr bool kGrid = true;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {64, 32, 8, 4, 2};
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
    static uint32_t grid_words(const Fields& f, const uint32_t* grid, int t, uint32_t* g) {
        const uint32_t pair = f.idx[t >> 1] >> (16 * (t & 1));
        for (int h = 0; h < 2; ++h)
            g[h] = grid[((pair >> (8 * h)) & 255u) | (((f.qh >> (2 * t + h)) & 1u) << 8)];
        return f.signs >> (8 * t);
    }
    static Affine affine(const Fields& f, int) {
        return {f.d * static_cast<float>(1 + 2 * static_cast<int32_t>(f.scale)), 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::IQ2_XXS> {
    static constexpr bool kTrick = false;
    static constexpr bool kGrid = true;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {64, 2};
    struct Fields {
        uint32_t idx;
        uint32_t aux;  // four 7-bit sign codes, then the 4-bit scale
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        const auto v = load<sycl::vec<uint32_t, 2>>(r.p[0] + 8 * u);
        return {v[0], v[1], f16_at(r.p[1] + 2 * (u >> 3))};
    }
    static uint32_t grid_words(const Fields& f, const uint32_t* grid, int t, uint32_t* g) {
        grid_pair(grid, (f.idx >> (8 * t)) & 255u, g);
        return signs_of((f.aux >> (7 * t)) & 127u);
    }
    static Affine affine(const Fields& f, int) {
        return {f.d * (0.5f + static_cast<float>(f.aux >> 28)) * 0.25f, 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::IQ2_S> {
    static constexpr bool kTrick = false;
    static constexpr bool kGrid = true;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {32, 32, 8, 8, 2};
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
    static uint32_t grid_words(const Fields& f, const uint32_t* grid, int t, uint32_t* g) {
        grid_pair(grid, ((f.idx >> (8 * t)) & 255u) | (((f.qh >> (2 * t)) & 3u) << 8), g);
        return f.signs >> (8 * t);
    }
    static Affine affine(const Fields& f, int g) {
        return {f.d * (0.5f + static_cast<float>((f.scales >> (4 * g)) & 15u)) * 0.25f, 0.0f};
    }
};

template <>
struct Fmt<BlockFormat::IQ2_XS> {
    static constexpr bool kTrick = false;
    static constexpr bool kGrid = true;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {64, 8, 2};
    struct Fields {
        sycl::vec<uint32_t, 2> q;  // four u16: a 9-bit index under a 7-bit sign code
        uint32_t scales;           // 4 bits a half
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 2>>(r.p[0] + 8 * u), r.p[1][u], f16_at(r.p[2] + 2 * (u >> 3))};
    }
    static uint32_t grid_words(const Fields& f, const uint32_t* grid, int t, uint32_t* g) {
        const uint32_t q = (f.q[t >> 1] >> (16 * (t & 1))) & 0xFFFFu;
        grid_pair(grid, q & 511u, g);
        return signs_of(q >> 9);
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
    static constexpr bool kGrid = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {32, 16, 2};
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
    static constexpr bool kGrid = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {32, 16, 8};
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
    static constexpr bool kGrid = false;
    static constexpr int kLowBits = 4;
    static constexpr int kHighBits = 1;
    static constexpr int kOff = 16;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {128, 32, 16};
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        uint32_t hi;
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u), load<uint32_t>(r.p[1] + 4 * u), f16_at(r.p[2] + 2 * u)};
    }
    static uint32_t low(const Fields& f, int t) { return f.lo[t]; }
    static uint32_t high(const Fields& f, int t) { return f.hi >> (8 * t); }
    static Affine affine(const Fields& f, int) { return {f.d, 0.0f}; }
};

template <>
struct Fmt<BlockFormat::Q5_1> {
    static constexpr bool kTrick = true;
    static constexpr bool kGrid = false;
    static constexpr int kLowBits = 4;
    static constexpr int kHighBits = 1;
    static constexpr int kOff = 0;
    static constexpr bool kMin = true;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {128, 32, 32};
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
    static uint32_t low(const Fields& f, int t) { return f.lo[t]; }
    static uint32_t high(const Fields& f, int t) { return f.hi >> (8 * t); }
    static Affine affine(const Fields& f, int) { return {f.d, -f.m}; }  // w = d code + m
};

template <>
struct Fmt<BlockFormat::NVFP4> {
    static constexpr bool kTrick = false;
    static constexpr bool kGrid = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 2;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {128, 16};
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        uint32_t scales;  // two bytes: each half's E4M3 code
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u), load<uint16_t>(r.p[1] + 2 * u)};
    }
    static void levels(const Fields& f, const uint32_t*, int t, float* lv) {
        const uint32_t w = f.lo[t];
        for (int i = 0; i < 8; ++i)
            lv[i] = fp4_level((w >> (4 * i)) & 15u);
    }
    static Affine affine(const Fields& f, int g) { return {ue4m3_half((f.scales >> (8 * g)) & 255u), 0.0f}; }
};

template <>
struct Fmt<BlockFormat::MXFP4> {
    static constexpr bool kTrick = false;
    static constexpr bool kGrid = false;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {128, 8};
    struct Fields {
        sycl::vec<uint32_t, 4> lo;
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 4>>(r.p[0] + 16 * u), e8m0_half(r.p[1][u])};
    }
    static void levels(const Fields& f, const uint32_t*, int t, float* lv) {
        const uint32_t w = f.lo[t];
        for (int i = 0; i < 8; ++i)
            lv[i] = fp4_level((w >> (4 * i)) & 15u);
    }
    static Affine affine(const Fields& f, int) { return {f.d, 0.0f}; }
};

template <>
struct Fmt<BlockFormat::Q2_0> {
    static constexpr bool kTrick = true;
    static constexpr bool kGrid = false;
    static constexpr int kLowBits = 2;
    static constexpr int kHighBits = 0;
    static constexpr int kOff = 1;
    static constexpr bool kMin = false;
    static constexpr int kGroups = 1;
    static constexpr std::array<int, kBlockStreams> kSuperblock = {64, 8};
    struct Fields {
        sycl::vec<uint32_t, 2> lo;
        float d;
    };
    static Fields fields(const RowPtr& r, int32_t u) {
        return {load<sycl::vec<uint32_t, 2>>(r.p[0] + 8 * u), f16_at(r.p[1] + 2 * (u >> 1))};
    }
    static uint32_t low(const Fields& f, int t) { return f.lo[t >> 1] >> (16 * (t & 1)); }
    static uint32_t high(const Fields&, int) { return 0; }
    static Affine affine(const Fields& f, int) { return {f.d, 0.0f}; }
};

// The codes of quarter t of a kTrick format.
template <BlockFormat F>
inline void codes(const typename Fmt<F>::Fields& f, int t, uint32_t* c) {
    constexpr int L = Fmt<F>::kLowBits;
    constexpr int H = Fmt<F>::kHighBits;
    const uint32_t w = Fmt<F>::low(f, t);
    const uint32_t h = Fmt<F>::high(f, t);
    for (int i = 0; i < 8; ++i)
        c[i] = ((w >> (L * i)) & ((1u << L) - 1u)) | (((h >> (H * i)) & ((1u << H) - 1u)) << L);
}
// The signed levels of quarter t of any format but a kTrick one.
template <BlockFormat F>
inline void levels(const typename Fmt<F>::Fields& f, const uint32_t* grid, int t, float* lv) {
    if constexpr (Fmt<F>::kGrid) {
        uint32_t g[2];
        const uint32_t signs = Fmt<F>::grid_words(f, grid, t, g);
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < 4; ++i)
                lv[4 * h + i] = signed_float((g[h] >> (8 * i)) & 255u, (signs >> (4 * h + i)) & 1u);
        }
    } else {
        Fmt<F>::levels(f, grid, t, lv);
    }
}

// Quarter t of a unit for the GEMV: wv with a and b (per half) such that weight = a * wv + b.
template <BlockFormat F>
inline void fast_quarter(const typename Fmt<F>::Fields& f, const uint32_t* grid, int t, float* wv) {
    if constexpr (Fmt<F>::kTrick) {
        uint32_t c[8];
        codes<F>(f, t, c);
        for (int i = 0; i < 8; ++i)
            wv[i] = unit_float(c[i], Fmt<F>::kLowBits + Fmt<F>::kHighBits);
    } else {
        levels<F>(f, grid, t, wv);
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
            constexpr float kOne = static_cast<float>(1 << (Fmt<F>::kLowBits + Fmt<F>::kHighBits));
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
        codes<F>(f, t, c);
        for (int i = 0; i < 8; ++i)
            lv[i] = static_cast<float>(static_cast<int32_t>(c[i]) - Fmt<F>::kOff);
    } else {
        levels<F>(f, grid, t, lv);
    }
    const Affine af = Fmt<F>::affine(f, t >> 1);
    for (int i = 0; i < 8; ++i)
        v[i] = Fmt<F>::kMin ? af.s * lv[i] - af.m : af.s * lv[i];
}

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// Whether a format's GEMV with m inputs takes a unit a quarter at a time (8 decoded values live at once instead of
// 32) or whole: whichever read more weights a second on an Arc Pro B60. The difference is the registers a decoder
// leaves free: Q5_K gains a third from quarters at one input, the K-quants lose a seventh with several.
constexpr bool by_quarters(BlockFormat f, int m) {
    switch (f) {
        case BlockFormat::Q4_K:
        case BlockFormat::Q5_K:
            return m == 1;
        case BlockFormat::IQ4_NL:
        case BlockFormat::IQ4_XS:
        case BlockFormat::IQ3_XXS:
        case BlockFormat::IQ2_XXS:
        case BlockFormat::IQ2_XS:
        case BlockFormat::IQ2_S:
        case BlockFormat::IQ1_S:
            return true;
        default:
            return false;
    }
}

// acc[r][mm] += row rp[r] of w . xs[mm] over this lane's units: a lane takes units lane, lane + 16, ... of each row, so
// a unit's x loads once for the rows and its codes decode once for the M inputs. Over a half unit
// dot(w, x) = a dot(wv, x) + b sum(x).
template <BlockFormat F, int M>
inline void block_rows_dot(const Streams& w, const RowPtr (&rp)[kRows], const float* const (&xs)[M], int lane,
                           int32_t units, float (&acc)[kRows][M]) {
    for (int32_t u = lane; u < units; u += kSubGroup) {
        float a[kRows][2];
        float b[kRows][2];
        if constexpr (by_quarters(F, M)) {
            float dot[kRows][M][2] = {};
            float xsum[M][2] = {};
            const typename Fmt<F>::Fields f[kRows] = {Fmt<F>::fields(rp[0], u), Fmt<F>::fields(rp[1], u)};
            for (int r = 0; r < kRows; ++r)
                fast_affine<F>(f[r], a[r], b[r]);
            for (int t = 0; t < 4; ++t) {
                float wv[kRows][8];
                for (int r = 0; r < kRows; ++r)
                    fast_quarter<F>(f[r], w.grid, t, wv[r]);
                for (int mm = 0; mm < M; ++mm) {
                    const auto* xp =
                        reinterpret_cast<const sycl::vec<float, 4>*>(xs[mm] + static_cast<int64_t>(u) * kUnit + 8 * t);
                    const sycl::vec<float, 4> x0 = xp[0];
                    const sycl::vec<float, 4> x1 = xp[1];
                    if constexpr (Fmt<F>::kTrick) {
                        xsum[mm][t >> 1] += ((x0[0] + x0[1]) + (x0[2] + x0[3])) + ((x1[0] + x1[1]) + (x1[2] + x1[3]));
                    }
                    for (int r = 0; r < kRows; ++r) {
                        float d = 0.0f;
                        for (int i = 0; i < 4; ++i)
                            d += wv[r][i] * x0[i] + wv[r][4 + i] * x1[i];
                        dot[r][mm][t >> 1] += d;
                    }
                }
            }
            for (int r = 0; r < kRows; ++r) {
                for (int mm = 0; mm < M; ++mm) {
                    acc[r][mm] += a[r][0] * dot[r][mm][0] + a[r][1] * dot[r][mm][1];
                    if constexpr (Fmt<F>::kTrick) acc[r][mm] += b[r][0] * xsum[mm][0] + b[r][1] * xsum[mm][1];
                }
            }
        } else {
            float wv[kRows][kUnit];
            for (int r = 0; r < kRows; ++r) {
                const auto f = Fmt<F>::fields(rp[r], u);
                for (int t = 0; t < 4; ++t)
                    fast_quarter<F>(f, w.grid, t, &wv[r][8 * t]);
                fast_affine<F>(f, a[r], b[r]);
            }
            for (int mm = 0; mm < M; ++mm) {
                float xv[kUnit];
                const auto* xp = reinterpret_cast<const sycl::vec<float, 4>*>(xs[mm] + static_cast<int64_t>(u) * kUnit);
                for (int i = 0; i < kUnit / 4; ++i) {
                    const sycl::vec<float, 4> v = xp[i];
                    for (int j = 0; j < 4; ++j)
                        xv[4 * i + j] = v[j];
                }
                float sum[2] = {0.0f, 0.0f};
                if constexpr (Fmt<F>::kTrick) {
                    for (int i = 0; i < kUnit; ++i)
                        sum[i >> 4] += xv[i];
                }
                for (int r = 0; r < kRows; ++r) {
                    float d[2] = {0.0f, 0.0f};
                    for (int i = 0; i < kUnit; ++i)
                        d[i >> 4] += wv[r][i] * xv[i];
                    acc[r][mm] += a[r][0] * d[0] + a[r][1] * d[1];
                    if constexpr (Fmt<F>::kTrick) acc[r][mm] += b[r][0] * sum[0] + b[r][1] * sum[1];
                }
            }
        }
    }
}

// A sub-group takes kRows rows of the weight (block_rows_dot).
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
            block_rows_dot<F, M>(w, rp, xs, lane, units, acc);
            for (int r = 0; r < kRows; ++r) {
                for (int mm = 0; mm < M; ++mm) {
                    const float sum = sycl::reduce_over_group(sg, acc[r][mm], sycl::plus<float>());
                    if (lane == 0 && row0 + r < n && mm < inputs) y[mm * n + row0 + r] = sum;
                }
            }
        });
}

// ExpertMatMul over a bank (rows = E * N), a GEMV per (token, slot) pair on the pair's expert: one work-group per pair
// and kSgPerWg * kRows of the expert's rows.
template <BlockFormat F, typename I>
void expert_block_pairs_t(sycl::queue& q, const float* x, bool per_slot, const Streams& w, const I* idx, float* out,
                          int64_t pairs, int64_t slots, int64_t experts, int64_t n, int64_t k) {
    const int64_t blocks = ceil_div(n, kSgPerWg * kRows);
    const auto units = static_cast<int32_t>(k / kUnit);
    syclex::nd_launch(
        q, sycl::nd_range<1>(static_cast<size_t>(pairs * blocks * kSgPerWg * kSubGroup), kSgPerWg * kSubGroup),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = static_cast<int>(sg.get_local_linear_id());
            const int64_t p = static_cast<int64_t>(it.get_group(0)) / blocks;
            const int64_t block = static_cast<int64_t>(it.get_group(0)) % blocks;
            const int64_t row0 = (block * kSgPerWg + static_cast<int64_t>(sg.get_group_linear_id())) * kRows;
            const int64_t base = expert_of(idx, p, experts) * n;
            RowPtr rp[kRows];
            for (int r = 0; r < kRows; ++r)
                rp[r] = row_ptr(w, base + std::min<int64_t>(row0 + r, n - 1));
            const float* const xs[1] = {x + (per_slot ? p : p / slots) * k};
            float acc[kRows][1] = {};
            block_rows_dot<F, 1>(w, rp, xs, lane, units, acc);
            for (int r = 0; r < kRows; ++r) {
                const float sum = sycl::reduce_over_group(sg, acc[r][0], sycl::plus<float>());
                if (lane == 0 && row0 + r < n) out[p * n + row0 + r] = sum;
            }
        });
}

// The same with the pairs grouped by expert (expert_group): one work-group per (expert with pairs, block of rows); it
// walks the expert's pairs kGemvMaxRows at a time, so an expert's rows are read once per that many pairs.
template <BlockFormat F>
void expert_block_grouped_t(sycl::queue& q, const float* x, bool per_slot, const Streams& w, float* out, int64_t slots,
                            int64_t experts, int64_t n, int64_t k, const int32_t* grouping, int64_t max_active) {
    constexpr int M = static_cast<int>(kGemvMaxRows);
    const int64_t blocks = ceil_div(n, kSgPerWg * kRows);
    const auto units = static_cast<int32_t>(k / kUnit);
    const int32_t* offsets = expert_offsets(grouping, experts);
    const int32_t* order = expert_order(grouping, experts);
    const int32_t* active = expert_active(grouping, experts);
    const int32_t* n_active = expert_active_count(grouping, experts);
    syclex::nd_launch(
        q, sycl::nd_range<1>(static_cast<size_t>(max_active * blocks * kSgPerWg * kSubGroup), kSgPerWg * kSubGroup),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
            const int64_t a = static_cast<int64_t>(it.get_group(0)) / blocks;
            if (a >= *n_active) return;  // uniform across the work-group
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = static_cast<int>(sg.get_local_linear_id());
            const int64_t block = static_cast<int64_t>(it.get_group(0)) % blocks;
            const int64_t row0 = (block * kSgPerWg + static_cast<int64_t>(sg.get_group_linear_id())) * kRows;
            const int64_t e = active[a];
            const int64_t begin = offsets[e];
            const int64_t end = offsets[e + 1];
            RowPtr rp[kRows];
            for (int r = 0; r < kRows; ++r)
                rp[r] = row_ptr(w, e * n + std::min<int64_t>(row0 + r, n - 1));
            for (int64_t c = begin; c < end; c += M) {
                int64_t ps[M];
                const float* xs[M];
                for (int mm = 0; mm < M; ++mm) {
                    ps[mm] = order[std::min<int64_t>(c + mm, end - 1)];
                    xs[mm] = x + (per_slot ? ps[mm] : ps[mm] / slots) * k;
                }
                float acc[kRows][M] = {};
                block_rows_dot<F, M>(w, rp, xs, lane, units, acc);
                for (int r = 0; r < kRows; ++r) {
                    for (int mm = 0; mm < M; ++mm) {
                        const float sum = sycl::reduce_over_group(sg, acc[r][mm], sycl::plus<float>());
                        if (lane == 0 && c + mm < end && row0 + r < n) out[ps[mm] * n + row0 + r] = sum;
                    }
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

// The GEMV on the matrix engines (gemv_block_xmx), M <= 8: gemv_quant_xmx's design over the block formats. A
// sub-group owns 16 output features, one per lane (the DPAS B operand's columns), and a slice of K. Per block of 256
// columns a lane loads its own feature's run of each stream whole, a load or two each (per lane every load is a row
// of its own, so few and wide loads are what keeps the memory busy), then per unit writes fast_quarter's values as
// f16 into two B operands: 1 + code / 2^bits, or the levels, small integers (IQ1's in eighths), so exact. Each
// multiplies an 8 x 16 tile of x in f16 (a 2D block load; rows past M read as zero), and the unit's scale applies to
// the sum, with x's sum over those columns times the affine's constant where the values are codes. The decoding is
// the work per weight, and every input shares it: the cost hardly grows with M.
constexpr int kXmxCols = 16;  // columns of a DPAS, and of the sums of x beside it

// Columns 16 h .. 16 h + 15 of a unit as a B operand: word i holds columns 2 i (low half) and 2 i + 1.
// IQ4's B operand words come from a table in local memory of every byte's two levels: computing a level from its
// code (a select among four words, then two conversions) costs more than the lookup.
template <BlockFormat F>
constexpr bool kPairTable = F == BlockFormat::IQ4_NL || F == BlockFormat::IQ4_XS;
inline uint32_t iq4_pair(uint32_t byte) {
    const auto biased = [](uint32_t code) { return static_cast<uint32_t>(static_cast<int32_t>(iq4_value(code)) + 128); };
    constexpr auto kBias = static_cast<_Float16>(1024 + 128);
    const uint32_t p = biased(byte & 15u) | (biased(byte >> 4) << 16) | 0x64006400u;
    return __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2_v, p) - h2_v{kBias, kBias});
}
// Codes 2 i and 2 i + 1 of a kTrick quarter as a B operand word, each in the top bits of the mantissa of 1: the f16
// 1 + code / 2^bits, fast_quarter's value. A multiply puts a pair's low bits (and one its high bits) 16 bits apart.
template <BlockFormat F>
inline int unit_pair(uint32_t low, uint32_t high, int i) {
    constexpr int L = Fmt<F>::kLowBits;
    constexpr int H = Fmt<F>::kHighBits;
    constexpr int S = 10 - L - H;
    constexpr uint32_t kLow = (1u << L) - 1u;
    const uint32_t lo = (low >> (2 * L * i)) & ((1u << (2 * L)) - 1u);
    uint32_t p = (lo * ((1u << S) | (1u << (16 + S - L)))) & ((kLow << S) | (kLow << (16 + S)));
    if constexpr (H > 0) {
        constexpr uint32_t kHigh = (1u << H) - 1u;
        const uint32_t hi = (high >> (2 * H * i)) & ((1u << (2 * H)) - 1u);
        p |= (hi * ((1u << (S + L)) | (1u << (16 + S + L - H)))) & ((kHigh << (S + L)) | (kHigh << (16 + S + L)));
    }
    return static_cast<int>(p | 0x3C003C00u);
}
template <BlockFormat F, typename Table>
inline int8_v b_operand(const typename Fmt<F>::Fields& f, const uint32_t* grid, const Table& pairs, int h) {
    int8_v b;
    for (int q = 0; q < 2; ++q) {
        const int t = 2 * h + q;
        if constexpr (kPairTable<F>) {
            for (int i = 0; i < 4; ++i)
                b[4 * q + i] = static_cast<int>(pairs[(f.lo[t] >> (8 * i)) & 255u]);
        } else if constexpr (Fmt<F>::kTrick) {
            const uint32_t low = Fmt<F>::low(f, t);
            const uint32_t high = Fmt<F>::high(f, t);
            for (int i = 0; i < 4; ++i)
                b[4 * q + i] = unit_pair<F>(low, high, i);
        } else {  // these levels' float conversions run faster than building their f16 bits from integers
            float lv[8];
            levels<F>(f, grid, t, lv);
            for (int i = 0; i < 4; ++i)
                b[4 * q + i] = __builtin_bit_cast(
                    int, h2_v{static_cast<_Float16>(lv[2 * i]), static_cast<_Float16>(lv[2 * i + 1])});
        }
    }
    return b;
}

// `bytes` (a whole number of words, or 2) from p into words, as few and wide loads as its alignment allows: p's offset
// in its stream is a multiple of `bytes`, and every stream's rows start 64-byte aligned.
template <int B>
inline void load_words(const uint8_t* p, uint32_t* w) {
    if constexpr (B % 64 == 0) {
        for (int i = 0; i < B / 64; ++i) {
            const auto v = load<sycl::vec<uint32_t, 16>>(p + 64 * i);
            for (int j = 0; j < 16; ++j)
                w[16 * i + j] = v[j];
        }
    } else if constexpr (B == 32 || B == 16 || B == 8) {
        const auto v = load<sycl::vec<uint32_t, B / 4>>(p);
        for (int j = 0; j < B / 4; ++j)
            w[j] = v[j];
    } else if constexpr (B == 2) {
        w[0] = load<uint16_t>(p);
    } else {
        for (int j = 0; j < B / 4; ++j)
            w[j] = load<uint32_t>(p + 4 * j);
    }
}

// A DPAS over R rows of x: its accumulator and A operand, an element a row.
template <int R>
struct XmxRows;
template <>
struct XmxRows<1> {
    using Acc = float;
    using A = short;
    static Acc dpas(Acc acc, A a, int8_v b) { return __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_1(acc, a, b); }
    static float get(Acc v, int) { return v; }
    static void set(Acc& v, int, float x) { v = x; }
    static void set(A& v, int, short x) { v = x; }
};
template <typename AccT, typename AT>
struct XmxRowsOf {
    using Acc = AccT;
    using A = AT;
    static float get(const Acc& v, int r) { return v[r]; }
    static void set(Acc& v, int r, float x) { v[r] = x; }
    static void set(A& v, int r, short x) { v[r] = x; }
};
template <>
struct XmxRows<2> : XmxRowsOf<float2_v, short2_v> {
    static Acc dpas(Acc acc, A a, int8_v b) { return __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_2(acc, a, b); }
};
template <>
struct XmxRows<4> : XmxRowsOf<float4_v, short4_v> {
    static Acc dpas(Acc acc, A a, int8_v b) { return __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_4(acc, a, b); }
};
template <>
struct XmxRows<8> : XmxRowsOf<float8_v, short8_v> {
    static Acc dpas(Acc acc, A a, int8_v b) { return __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(acc, a, b); }
};

// acc[c] += a unit of this lane's row (fields f) . tile c of R rows of x, for C tiles. tile(h, c) is the A operand of
// the unit's half h (16 columns of tile c's rows, in f16) and sum(h, c, tile, s) is s plus those rows' sums of the
// f16 values. The B operands are fast_quarter's values, exact in f16 and decoded once for the tiles, so over a half
// dot(w, x) = a dot + b sum(x) as in block_rows_dot.
template <BlockFormat F, int R, int C, typename Table, typename TileFn, typename SumFn>
inline void xmx_unit(const typename Fmt<F>::Fields& f, const uint32_t* grid, const Table& pairs, TileFn&& tile,
                     SumFn&& sum, typename XmxRows<R>::Acc (&acc)[C]) {
    using T = XmxRows<R>;
    using Acc = typename T::Acc;
    constexpr bool kSums = Fmt<F>::kTrick;  // b is zero for the others
    constexpr bool kHalves = Fmt<F>::kGroups == 2;  // a scale per half; else both halves into one sum
    float a[2];
    float b[2];
    fast_affine<F>(f, a, b);
    const auto add = [&](int g, int c, const Acc& dot, const Acc& s) {
        for (int r = 0; r < R; ++r) {
            const float base = kSums ? sycl::fma(b[g], T::get(s, r), T::get(acc[c], r)) : T::get(acc[c], r);
            T::set(acc[c], r, sycl::fma(a[g], T::get(dot, r), base));
        }
    };
    Acc dot[C];
    Acc s[C];
    for (int h = 0; h < 2; ++h) {
        const int8_v bq = b_operand<F, Table>(f, grid, pairs, h);
        for (int c = 0; c < C; ++c) {
            if (h == 0 || kHalves) {
                dot[c] = Acc(0.0f);
                s[c] = Acc(0.0f);
            }
            const typename T::A x = tile(h, c);
            dot[c] = T::dpas(dot[c], x, bq);
            if constexpr (kSums) s[c] = sum(h, c, x, s[c]);
            if (h == 1 || kHalves) add(h == 1 && !kHalves ? 0 : h, c, dot[c], s[c]);
        }
    }
}

// A block's run of each stream of a row, loaded whole into `copy` (kWords words): `r` then reads the copy as the
// streams read, with units 0 to 7.
template <BlockFormat F>
struct BlockRun {
    static constexpr std::array<int, kBlockStreams> kBytes = Fmt<F>::kSuperblock;
    static constexpr int kStreams = kBytes[4] ? 5 : kBytes[3] ? 4 : kBytes[2] ? 3 : 2;
    static constexpr int words(int i) { return (kBytes[i] + 3) / 4; }
    static constexpr int offset(int i) {
        int o = 0;
        for (int j = 0; j < i; ++j)
            o += words(j);
        return o;
    }
    static constexpr int kWords = offset(kBlockStreams);

    static void load(const RowPtr& rp, int32_t b, uint32_t* copy, RowPtr& r) {
        [&]<int... I>(std::integer_sequence<int, I...>) {
            ((load_words<kBytes[I]>(rp.p[I] + b * kBytes[I], copy + offset(I)),
              r.p[I] = reinterpret_cast<const uint8_t*>(copy + offset(I))),
             ...);
        }(std::make_integer_sequence<int, kStreams>{});
    }
};

// Sub-groups a work-group of 16 output features splits K between: about 8 per compute unit streaming the weight, each
// slice of K at least two blocks of 256 columns.
int xmx_slices(int64_t wgs, int32_t blocks, int64_t compute_units) {
    int slices = 1;
    while (slices < kQuantGemmMaxSlices && wgs * slices < 8 * compute_units && blocks / (slices * 2) >= 2)
        slices *= 2;
    return slices;
}

template <BlockFormat F>
struct GemvBlockXmx {
    using T = XmxRows<8>;

    Streams w;
    float* y;
    const half* xh;   // [M, K]
    const float* xs;  // [K / 16][8]
    int64_t m, n, k;
    int32_t slice_blocks;
    sycl::local_accessor<float, 1> slm;  // every slice's sums: [slice][8 rows][16 lanes]
    sycl::local_accessor<uint32_t, 1> pairs;  // kPairTable: a byte's two levels as a B operand word

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<kSubGroup>, intelex::grf_size<256>};
    }

    short8_v x_tile(int col) const {
        const int width = static_cast<int>(k * 2) - 1;  // x as an [M, K] f16 surface
        return __builtin_bit_cast(short8_v, __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
                                                reinterpret_cast<long>(xh), width, static_cast<int>(m) - 1, width,
                                                int2_v{col, 0}));
    }

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int slice = static_cast<int>(sg.get_group_linear_id());
        const int slices = static_cast<int>(sg.get_group_linear_range());
        const int64_t row = static_cast<int64_t>(it.get_group(0)) * kSubGroup + lane;  // this lane's output feature
        const RowPtr rp = row_ptr(w, std::min(row, n - 1));
        if constexpr (kPairTable<F>) {
            for (size_t i = it.get_local_linear_id(); i < 256; i += it.get_local_range(0))
                pairs[i] = iq4_pair(static_cast<uint32_t>(i));
            sycl::group_barrier(it.get_group());
        }
        const auto blocks = static_cast<int32_t>(k / 256);
        const int32_t b_lo = std::min(blocks, slice * slice_blocks);
        const int32_t b_hi = std::min(blocks, b_lo + slice_blocks);
        float8_v acc[1] = {0.0f};
        for (int32_t b = b_lo; b < b_hi; ++b) {
            uint32_t copy[BlockRun<F>::kWords];  // its eight units then decode from registers
            RowPtr r;
            BlockRun<F>::load(rp, b, copy, r);
            #pragma unroll
            for (int32_t j = 0; j < 8; ++j) {
                const int32_t u = b * 8 + j;
                xmx_unit<F, 8, 1>(
                    Fmt<F>::fields(r, j), w.grid, pairs,
                    [&](int h, int) { return x_tile(u * kUnit + kXmxCols * h); },
                    [&](int h, int, short8_v, float8_v s) {
                        return s + *reinterpret_cast<const float8_v*>(xs + (static_cast<int64_t>(u) * 2 + h) * 8);
                    },
                    acc);
            }
        }
        // The slices add up in local memory; element r of acc at lane l is y[r][row].
        for (int r2 = 0; r2 < 8; ++r2)
            slm[(slice * 8 + r2) * kSubGroup + lane] = acc[0][r2];
        sycl::group_barrier(it.get_group());
        if (slice == 0 && row < n) {
            for (int r2 = 0; r2 < 8 && r2 < m; ++r2) {
                float sum = 0.0f;
                for (int s2 = 0; s2 < slices; ++s2)
                    sum += slm[(s2 * 8 + r2) * kSubGroup + lane];
                y[r2 * n + row] = sum;
            }
        }
    }
};

template <BlockFormat F>
void gemv_block_xmx_t(sycl::queue& q, const Streams& w, const half* xh, const float* xs, float* y, int64_t m,
                      int64_t n, int64_t k, int64_t compute_units) {
    for (int i = 0; i < kBlockStreams; ++i) {
        XS_CHECK(w.stride[i] == Fmt<F>::kSuperblock[i] * (k / 256), "sycl gemv_block_xmx: stream ", i, " of ",
                 block_format_name(F), " is not the layout's");
    }
    const int64_t wgs = ceil_div(n, kSubGroup);
    const auto blocks = static_cast<int32_t>(k / 256);
    const int slices = xmx_slices(wgs, blocks, compute_units);
    const auto slice_blocks = static_cast<int32_t>(ceil_div(blocks, slices));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(static_cast<size_t>(slices * 8 * kSubGroup)), h);
        sycl::local_accessor<uint32_t, 1> pairs(sycl::range<1>(kPairTable<F> ? 256 : 1), h);
        const auto local = static_cast<size_t>(slices * kSubGroup);
        syclex::nd_launch(h, sycl::nd_range<1>(static_cast<size_t>(wgs) * local, local),
                          GemvBlockXmx<F>{w, y, xh, xs, m, n, k, slice_blocks, slm, pairs});
    });
}

// The same GEMV for more rows (gemm_block_xmx), M <= 8 C: per unit a lane's B operands multiply C tiles of 8 rows of
// x, so the weight is read and decoded once for all of them and never written out in f16. x's sums come from a DPAS
// with ones: x in f16 is all the kernel reads beside the weight. y in f32 or (half_y) f16.
template <BlockFormat F, int C>
struct GemmBlockXmx {
    Streams w;
    void* y;
    const half* xh;  // [M, K]
    int64_t m, n, k;
    int32_t slice_blocks;
    bool half_y, add;
    sycl::local_accessor<float, 1> slm;  // every slice's sums: [slice][8 C rows][16 lanes]
    sycl::local_accessor<uint32_t, 1> pairs;  // kPairTable: a byte's two levels as a B operand word

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<kSubGroup>, intelex::grf_size<256>};
    }

    void operator()(sycl::nd_item<1> it) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int slice = static_cast<int>(sg.get_group_linear_id());
        const int slices = static_cast<int>(sg.get_group_linear_range());
        const int64_t row = static_cast<int64_t>(it.get_group(0)) * kSubGroup + lane;  // this lane's output feature
        const RowPtr rp = row_ptr(w, std::min(row, n - 1));
        if constexpr (kPairTable<F>) {
            for (size_t i = it.get_local_linear_id(); i < 256; i += it.get_local_range(0))
                pairs[i] = iq4_pair(static_cast<uint32_t>(i));
            sycl::group_barrier(it.get_group());
        }
        const int width = static_cast<int>(k * 2) - 1;  // x as an [M, K] f16 surface; rows past M read as zero
        const int height = static_cast<int>(m) - 1;
        const auto base = reinterpret_cast<long>(xh);
        const int8_v ones = 0x3C003C00;
        const auto blocks = static_cast<int32_t>(k / 256);
        const int32_t b_lo = std::min(blocks, slice * slice_blocks);
        const int32_t b_hi = std::min(blocks, b_lo + slice_blocks);
        float8_v acc[C];
        for (int c = 0; c < C; ++c)
            acc[c] = 0.0f;
        for (int32_t b = b_lo; b < b_hi; ++b) {
            uint32_t copy[BlockRun<F>::kWords];
            RowPtr r;
            BlockRun<F>::load(rp, b, copy, r);
            #pragma unroll
            for (int32_t j = 0; j < 8; ++j) {
                const int32_t u = b * 8 + j;
                xmx_unit<F, 8, C>(
                    Fmt<F>::fields(r, j), w.grid, pairs,
                    [&](int h, int c) {
                        return __builtin_bit_cast(short8_v, __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
                                                                base, width, height, width,
                                                                int2_v{u * kUnit + kXmxCols * h, 8 * c}));
                    },
                    [&](int, int, short8_v a, float8_v s) {
                        return __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(s, a, ones);
                    },
                    acc);
            }
        }
        for (int r2 = 0; r2 < 8 * C; ++r2)
            slm[(slice * 8 * C + r2) * kSubGroup + lane] = acc[r2 / 8][r2 % 8];
        sycl::group_barrier(it.get_group());
        if (slice == 0 && row < n) {
            for (int r2 = 0; r2 < 8 * C && r2 < m; ++r2) {
                float sum = 0.0f;
                for (int s2 = 0; s2 < slices; ++s2)
                    sum += slm[(s2 * 8 * C + r2) * kSubGroup + lane];
                if (half_y) {
                    static_cast<half*>(y)[r2 * n + row] = static_cast<half>(sum);
                } else {
                    float* out = static_cast<float*>(y) + r2 * n + row;
                    *out = add ? *out + sum : sum;
                }
            }
        }
    }
};

template <BlockFormat F, int C>
void gemm_block_xmx_t(sycl::queue& q, const Streams& w, const half* xh, void* y, bool half_y, int64_t m, int64_t n,
                      int64_t k, int64_t compute_units, bool add) {
    const int64_t wgs = ceil_div(n, kSubGroup);
    const auto blocks = static_cast<int32_t>(k / 256);
    const int slices = xmx_slices(wgs, blocks, compute_units);
    const auto slice_blocks = static_cast<int32_t>(ceil_div(blocks, slices));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(static_cast<size_t>(slices * 8 * C * kSubGroup)), h);
        sycl::local_accessor<uint32_t, 1> pairs(sycl::range<1>(kPairTable<F> ? 256 : 1), h);
        const auto local = static_cast<size_t>(slices * kSubGroup);
        syclex::nd_launch(h, sycl::nd_range<1>(static_cast<size_t>(wgs) * local, local),
                          GemmBlockXmx<F, C>{w, y, xh, m, n, k, slice_blocks, half_y, add, slm, pairs});
    });
}

// ExpertMatMul on the matrix engines: the pairs of an expert as the rows of one GEMV, R at a time, so its weights are
// read once for them. A work-group owns 16 rows of a pair's expert, a lane each, and splits K between its sub-groups
// as gemv_block_xmx does; it runs for the first pair of every R that read the expert and takes the rest of those R
// with it, found by walking the ids (few: up to kExpertPairsDirect). x is read in f32, each tile converted where it
// is used and summed by a DPAS with ones: no launch before this one, whose cost a decode step's eight pairs would
// feel.
template <BlockFormat F, int R, typename I>
struct ExpertBlockXmx {
    using T = XmxRows<R>;
    using Acc = typename T::Acc;

    Streams w;
    const float* x;
    const I* idx;
    float* out;
    int64_t pairs, slots, experts, n, k, row_groups;
    bool per_slot;
    int32_t slice_blocks;
    sycl::local_accessor<float, 1> slm;  // every slice's sums: [slice][R rows][16 lanes]
    sycl::local_accessor<uint32_t, 1> table;  // kPairTable: a byte's two levels as a B operand word

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<kSubGroup>, intelex::grf_size<256>};
    }

    static typename T::A x_tile(const float* const (&xs)[R], int col) {
        typename T::A a;
        for (int r = 0; r < R; ++r) {
            const auto v = sycl::bit_cast<float>(
                __builtin_IB_simd_block_read_1_global(reinterpret_cast<const unsigned*>(xs[r] + col)));
            T::set(a, r, __builtin_bit_cast(short, static_cast<_Float16>(v)));
        }
        return a;
    }

    void operator()(sycl::nd_item<1> it) const {
        const auto wg = static_cast<int64_t>(it.get_group(0));
        const int64_t p = wg / row_groups;
        const int64_t e = expert_of(idx, p, experts);
        int64_t before = 0;
        for (int64_t q = 0; q < p; ++q)
            before += expert_of(idx, q, experts) == e ? 1 : 0;
        if (before % R != 0) return;  // uniform across the work-group: an earlier pair's takes this one
        int64_t ps[R];  // the pairs of the R rows; rows past the last repeat it
        for (int r = 0; r < R; ++r)
            ps[r] = p;
        int count = 1;
        for (int64_t q = p + 1; q < pairs && count < R; ++q) {
            if (expert_of(idx, q, experts) != e) continue;
            for (int r = 0; r < R; ++r)
                ps[r] = r >= count ? q : ps[r];
            ++count;
        }
        const float* xs[R];
        for (int r = 0; r < R; ++r)
            xs[r] = x + (per_slot ? ps[r] : ps[r] / slots) * k;

        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int slice = static_cast<int>(sg.get_group_linear_id());
        const int slices = static_cast<int>(sg.get_group_linear_range());
        const int64_t row = (wg % row_groups) * kSubGroup + lane;  // this lane's row of the expert
        const RowPtr rp = row_ptr(w, e * n + std::min(row, n - 1));
        if constexpr (kPairTable<F>) {
            for (size_t i = it.get_local_linear_id(); i < 256; i += it.get_local_range(0))
                table[i] = iq4_pair(static_cast<uint32_t>(i));
            sycl::group_barrier(it.get_group());
        }
        const int8_v ones = 0x3C003C00;
        Acc acc[1] = {Acc(0.0f)};
        const auto unit = [&](const RowPtr& r, int32_t j, int32_t u) {
            xmx_unit<F, R, 1>(
                Fmt<F>::fields(r, j), w.grid, table,
                [&](int h, int) { return x_tile(xs, u * kUnit + kXmxCols * h); },
                [&](int, int, typename T::A a, Acc s) { return T::dpas(s, a, ones); }, acc);
        };
        const auto units = static_cast<int32_t>(k / kUnit);
        const int32_t blocks = units / 8;
        const int32_t b_lo = std::min(blocks, slice * slice_blocks);
        const int32_t b_hi = std::min(blocks, b_lo + slice_blocks);
        for (int32_t b = b_lo; b < b_hi; ++b) {
            uint32_t copy[BlockRun<F>::kWords];
            RowPtr r;
            BlockRun<F>::load(rp, b, copy, r);
            #pragma unroll
            for (int32_t j = 0; j < 8; ++j)
                unit(r, j, b * 8 + j);
        }
        if (slice == slices - 1) {  // the units past the last whole block, from the streams
            for (int32_t u = blocks * 8; u < units; ++u)
                unit(rp, u, u);
        }
        for (int r = 0; r < R; ++r)
            slm[(slice * R + r) * kSubGroup + lane] = T::get(acc[0], r);
        sycl::group_barrier(it.get_group());
        if (slice == 0 && row < n) {
            for (int r = 0; r < count; ++r) {
                float sum = 0.0f;
                for (int s2 = 0; s2 < slices; ++s2)
                    sum += slm[(s2 * R + r) * kSubGroup + lane];
                out[ps[r] * n + row] = sum;
            }
        }
    }
};

template <BlockFormat F, int R, typename I>
void expert_block_xmx_t(sycl::queue& q, const float* x, bool per_slot, const Streams& w, const I* idx, float* out,
                        int64_t pairs, int64_t slots, int64_t experts, int64_t n, int64_t k, int64_t compute_units) {
    const int64_t row_groups = ceil_div(n, kSubGroup);
    const int64_t wgs = pairs * row_groups;
    const auto blocks = static_cast<int32_t>(k / 256);
    const int slices = xmx_slices(wgs, blocks, compute_units);
    const auto slice_blocks = static_cast<int32_t>(ceil_div(blocks, slices));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(static_cast<size_t>(slices * R * kSubGroup)), h);
        sycl::local_accessor<uint32_t, 1> table(sycl::range<1>(kPairTable<F> ? 256 : 1), h);
        const auto local = static_cast<size_t>(slices * kSubGroup);
        syclex::nd_launch(h, sycl::nd_range<1>(static_cast<size_t>(wgs) * local, local),
                          ExpertBlockXmx<F, R, I>{w, x, idx, out, pairs, slots, experts, n, k, row_groups, per_slot,
                                                  slice_blocks, slm, table});
    });
}

// Columns 16 h .. 16 h + 15 of a unit as a B operand of the weights themselves, rounded to f16: the values
// dequantize_block writes, so their product with x needs no sums of x beside it and adds straight into an accumulator.
template <BlockFormat F>
inline int8_v weight_operand(const typename Fmt<F>::Fields& f, const uint32_t* grid, int h) {
    int8_v b;
    for (int q = 0; q < 2; ++q) {
        float v[8];
        exact_quarter<F>(f, grid, 2 * h + q, v);
        for (int i = 0; i < 4; ++i) {
            b[4 * q + i] = __builtin_bit_cast(
                int, h2_v{static_cast<_Float16>(v[2 * i]), static_cast<_Float16>(v[2 * i + 1])});
        }
    }
    return b;
}

// ExpertMatMul over many pairs on the matrix engines (expert_gemm_block_xmx). The pairs come grouped by expert, their
// rows of x gathered in that order in f16, and cut into passes: up to kPassRows rows of one expert. A work-group owns
// a pass and 16 rows of its expert, a lane each, and splits K between its sub-groups. Per 16 columns a lane decodes
// its row's weights once into a B operand (weight_operand) and every tile of 8 of the pass's rows multiplies it, 32
// rows of x a load: the bank is read once per pass and never written out, where a GEMM per expert first writes every
// expert that has a pair out in f16. A pass runs the body compiled for its tiles (1, 2, 4 or 8): what a pass costs
// beside decoding its weights is loading its rows of x, so the pairs left over after an expert's whole passes cost
// their own rows and not 64.
constexpr int kPassTiles = 8;
constexpr int kPassRows = 8 * kPassTiles;
constexpr int kPassFields = 3;  // a pass in the table: its expert, its first row, its rows

template <BlockFormat F>
struct ExpertGemmBlockXmx {
    Streams w;
    const half* xh;         // [pairs, K]: row i is the input of pair order[i]
    const int32_t* order;
    const int32_t* passes;  // their number, then kPassFields each
    float* out;
    int64_t pairs, n, k, row_groups;
    int32_t slice_blocks;
    sycl::local_accessor<float, 1> slm;  // every slice's sums: [slice][rows][16 lanes]

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<kSubGroup>, intelex::grf_size<256>};
    }

    template <int MT>
    void pass(const sycl::nd_item<1>& it, int64_t expert, int32_t first, int32_t rows) const {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int slice = static_cast<int>(sg.get_group_linear_id());
        const int slices = static_cast<int>(sg.get_group_linear_range());
        const int64_t row = (static_cast<int64_t>(it.get_group(0)) % row_groups) * kSubGroup + lane;
        const RowPtr rp = row_ptr(w, expert * n + std::min(row, n - 1));
        const int width = static_cast<int>(k * 2) - 1;  // x as a [pairs, K] f16 surface; rows past it read as zero
        const int height = static_cast<int>(pairs) - 1;
        const auto base = reinterpret_cast<long>(xh);
        float8_v acc[MT];
        #pragma unroll
        for (int t = 0; t < MT; ++t)
            acc[t] = 0.0f;
        const auto unit = [&](const RowPtr& r, int32_t j, int32_t u) {
            const auto f = Fmt<F>::fields(r, j);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int8_v b = weight_operand<F>(f, w.grid, h);
                const int col = u * kUnit + kXmxCols * h;
                if constexpr (MT == 1) {
                    const short8_v a = __builtin_bit_cast(
                        short8_v, __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(base, width, height, width,
                                                                                    int2_v{col, first}));
                    acc[0] = __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(acc[0], a, b);
                } else {
                    #pragma unroll
                    for (int t0 = 0; t0 < MT; t0 += 4) {
                        const ushort32_v a4 = __builtin_IB_subgroup_block_read_flat_u16_m32k16v1(
                            base, width, height, width, int2_v{col, first + 8 * t0});
                        #pragma unroll
                        for (int t = t0; t < t0 + 4 && t < MT; ++t) {
                            ushort8_v a;
                            #pragma unroll
                            for (int r2 = 0; r2 < 8; ++r2)
                                a[r2] = a4[8 * (t - t0) + r2];
                            const short8_v a8 = __builtin_bit_cast(short8_v, a);
                            acc[t] = __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(acc[t], a8, b);
                        }
                    }
                }
            }
        };
        const auto units = static_cast<int32_t>(k / kUnit);
        const int32_t blocks = units / 8;
        const int32_t b_lo = std::min(blocks, slice * slice_blocks);
        const int32_t b_hi = std::min(blocks, b_lo + slice_blocks);
        for (int32_t b = b_lo; b < b_hi; ++b) {
            uint32_t copy[BlockRun<F>::kWords];  // its eight units then decode from registers
            RowPtr r;
            BlockRun<F>::load(rp, b, copy, r);
            #pragma unroll
            for (int32_t j = 0; j < 8; ++j)
                unit(r, j, b * 8 + j);
        }
        if (slice == slices - 1) {  // the units past the last whole block, from the streams
            for (int32_t u = blocks * 8; u < units; ++u)
                unit(rp, u, u);
        }
        // The slices add up in local memory; element r of acc[t] at lane l is row 8 t + r of the pass, feature `row`.
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            #pragma unroll
            for (int r2 = 0; r2 < 8; ++r2)
                slm[(slice * kPassRows + 8 * t + r2) * kSubGroup + lane] = acc[t][r2];
        }
        sycl::group_barrier(it.get_group());
        if (slice == 0 && row < n) {
            for (int32_t r2 = 0; r2 < rows; ++r2) {
                float sum = 0.0f;
                for (int s2 = 0; s2 < slices; ++s2)
                    sum += slm[(s2 * kPassRows + r2) * kSubGroup + lane];
                out[static_cast<int64_t>(order[first + r2]) * n + row] = sum;
            }
        }
    }

    void operator()(sycl::nd_item<1> it) const {
        const int64_t p = static_cast<int64_t>(it.get_group(0)) / row_groups;
        if (p >= passes[0]) return;  // uniform across the work-group, as is the pass's body
        const int32_t* entry = passes + 1 + p * kPassFields;
        const int64_t expert = entry[0];
        const int32_t first = entry[1];
        const int32_t rows = entry[2];
        if (rows <= 8) {
            pass<1>(it, expert, first, rows);
        } else if (rows <= 16) {
            pass<2>(it, expert, first, rows);
        } else if (rows <= 32) {
            pass<4>(it, expert, first, rows);
        } else {
            pass<kPassTiles>(it, expert, first, rows);
        }
    }
};

// The passes of a grouping (expert_group): an expert's rows kPassRows at a time, its last pass the rows left over.
int64_t expert_max_passes(int64_t pairs, int64_t experts) {
    return pairs / kPassRows + std::min(pairs, experts);
}

void expert_passes(sycl::queue& q, const int32_t* grouping, int64_t experts, int32_t* passes) {
    const int32_t* offsets = expert_offsets(grouping, experts);
    syclex::single_task(q, [=] {
        int32_t count = 0;
        for (int64_t e = 0; e < experts; ++e) {
            for (int32_t first = offsets[e]; first < offsets[e + 1]; first += kPassRows) {
                int32_t* entry = passes + 1 + count * kPassFields;
                entry[0] = static_cast<int32_t>(e);
                entry[1] = first;
                entry[2] = std::min<int32_t>(kPassRows, offsets[e + 1] - first);
                ++count;
            }
        }
        passes[0] = count;
    });
}

template <BlockFormat F>
void expert_gemm_block_xmx_t(sycl::queue& q, const Streams& w, const half* xh, const int32_t* order,
                             const int32_t* passes, float* out, int64_t pairs, int64_t experts, int64_t n, int64_t k,
                             int64_t compute_units) {
    const int64_t row_groups = ceil_div(n, kSubGroup);
    const auto blocks = static_cast<int32_t>(k / 256);
    // every pair on one expert makes the fewest passes
    const int slices = xmx_slices(ceil_div(pairs, kPassRows) * row_groups, blocks, compute_units);
    const auto slice_blocks = static_cast<int32_t>(ceil_div(blocks, slices));
    const int64_t wgs = expert_max_passes(pairs, experts) * row_groups;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(static_cast<size_t>(slices * kPassRows * kSubGroup)), h);
        const auto local = static_cast<size_t>(slices * kSubGroup);
        syclex::nd_launch(h, sycl::nd_range<1>(static_cast<size_t>(wgs) * local, local),
                          ExpertGemmBlockXmx<F>{w, xh, order, passes, out, pairs, n, k, row_groups, slice_blocks, slm});
    });
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
    XS_CHECK(block_whole_rows(w.format, k), "sycl: a ", block_format_name(w.format), " weight needs a multiple of ",
             block_layout(w.format).weights, " columns, got ", k);
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
        case BlockFormat::NVFP4:
            return f.template operator()<BlockFormat::NVFP4>();
        case BlockFormat::MXFP4:
            return f.template operator()<BlockFormat::MXFP4>();
        case BlockFormat::Q2_0:
            return f.template operator()<BlockFormat::Q2_0>();
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

bool gemv_block_xmx_fits(const BlockWeight& w, int64_t k) {
    return block_on_device(w.format) && k % 256 == 0 && k * 2 < (int64_t{1} << 24);
}

size_t gemv_block_xmx_scratch(int64_t m, int64_t k) {
    return xmx_prep_bytes(m, k, kXmxCols);
}

void gemv_block_xmx(sycl::queue& q, const float* x, const BlockWeight& w, float* y, int64_t m, int64_t n, int64_t k,
                    int64_t compute_units, void* scratch) {
    XS_CHECK(m >= 1 && m <= kGemvMaxRows && gemv_block_xmx_fits(w, k) && block_load_aligned(scratch),
             "sycl gemv_block_xmx: a ", block_format_name(w.format), " weight of K ", k, " at ", m, " rows");
    const Streams s = streams_of(w, k);
    xmx_prep(q, x, nullptr, scratch, m, k, kXmxCols);
    const auto* xh = static_cast<const half*>(scratch);
    const auto* xs = reinterpret_cast<const float*>(static_cast<const char*>(scratch) + xmx_x_bytes(m, k));
    dispatch(w.format, [&]<BlockFormat F>() { gemv_block_xmx_t<F>(q, s, xh, xs, y, m, n, k, compute_units); });
}

template <typename Y>
void gemm_block_xmx(sycl::queue& q, const half* x, const BlockWeight& w, Y* y, int64_t m, int64_t n, int64_t k,
                    int64_t compute_units, bool add) {
    constexpr bool kHalfY = std::is_same_v<Y, half>;
    XS_CHECK(m > kGemvMaxRows && m <= kQuantGemmMaxRows && gemv_block_xmx_fits(w, k) && block_load_aligned(x) &&
                 !(add && kHalfY),
             "sycl gemm_block_xmx: a ", block_format_name(w.format), " weight of K ", k, " at ", m, " rows");
    const Streams s = streams_of(w, k);
    // Four tiles a pass, so past 32 rows the weight is read twice: eight tiles' accumulators do not fit the registers
    // (spilled, one pass costs more than dequantizing the weight).
    constexpr int64_t kPassRows = 32;
    dispatch(w.format, [&]<BlockFormat F>() {
        for (int64_t r = 0; r < m; r += kPassRows) {
            const int64_t rows = std::min(kPassRows, m - r);
            const auto run = [&]<int C>() {
                gemm_block_xmx_t<F, C>(q, s, x + r * k, y + r * n, kHalfY, rows, n, k, compute_units, add);
            };
            rows <= 16 ? run.template operator()<2>() : run.template operator()<4>();
        }
    });
}

template <typename I>
void expert_matmul_block(sycl::queue& q, const float* x, bool per_slot, const BlockWeight& w, const I* idx, float* out,
                         int64_t tokens, int64_t slots, int64_t experts, int64_t n, int64_t k, int32_t* grouping,
                         int64_t xmx_units) {
    const Streams s = streams_of(w, k);
    const int64_t pairs = tokens * slots;
    if (pairs <= kExpertPairsDirect && xmx_units > 0) {
        // Rows of a GEMV, by the tokens: few tokens share few experts, and rows no pair fills still cost their DPAS
        // (measured on Inkling-2-Mini's banks: one row a GEMV at one token, two up to four, four beyond).
        dispatch(w.format, [&]<BlockFormat F>() {
            const auto run = [&]<int R>() {
                expert_block_xmx_t<F, R, I>(q, x, per_slot, s, idx, out, pairs, slots, experts, n, k, xmx_units);
            };
            tokens == 1 ? run.template operator()<1>()
                        : tokens <= 4 ? run.template operator()<2>() : run.template operator()<4>();
        });
        return;
    }
    if (pairs <= kExpertPairsDirect || grouping == nullptr) {
        dispatch(w.format, [&]<BlockFormat F>() {
            expert_block_pairs_t<F, I>(q, x, per_slot, s, idx, out, pairs, slots, experts, n, k);
        });
        return;
    }
    expert_group(q, idx, pairs, experts, grouping);
    dispatch(w.format, [&]<BlockFormat F>() {
        expert_block_grouped_t<F>(q, x, per_slot, s, out, slots, experts, n, k, grouping, std::min(pairs, experts));
    });
}

bool expert_gemm_block_xmx_fits(const BlockWeight& w, int64_t pairs, int64_t k) {
    return block_on_device(w.format) && block_whole_rows(w.format, k) && k % kUnit == 0 && k * 2 < (int64_t{1} << 24) &&
           pairs < (int64_t{1} << 24);
}

namespace {

struct ExpertGemmScratch {
    size_t passes, x, bytes;  // where the passes and x in f16 start, after the grouping
    ExpertGemmScratch(int64_t pairs, int64_t experts, int64_t k) {
        const auto align = [](size_t v) { return (v + 63) / 64 * 64; };
        passes = align(expert_grouping_bytes(pairs, experts));
        x = passes + align(static_cast<size_t>(1 + expert_max_passes(pairs, experts) * kPassFields) * sizeof(int32_t));
        bytes = x + align(static_cast<size_t>(pairs * k) * sizeof(half));
    }
};

}  // namespace

size_t expert_gemm_block_xmx_scratch(int64_t pairs, int64_t experts, int64_t k) {
    return ExpertGemmScratch(pairs, experts, k).bytes;
}

template <typename I>
void expert_gemm_block_xmx(sycl::queue& q, const float* x, bool per_slot, const BlockWeight& w, const I* idx,
                           float* out, int64_t tokens, int64_t slots, int64_t experts, int64_t n, int64_t k,
                           int64_t compute_units, void* scratch) {
    const int64_t pairs = tokens * slots;
    XS_CHECK(expert_gemm_block_xmx_fits(w, pairs, k) && block_load_aligned(scratch), "sycl expert_gemm_block_xmx: a ",
             block_format_name(w.format), " bank of K ", k, " at ", pairs, " pairs");
    const Streams s = streams_of(w, k);
    const ExpertGemmScratch at(pairs, experts, k);
    auto* grouping = static_cast<int32_t*>(scratch);
    auto* passes = reinterpret_cast<int32_t*>(static_cast<char*>(scratch) + at.passes);
    auto* xh = reinterpret_cast<half*>(static_cast<char*>(scratch) + at.x);
    expert_group(q, idx, pairs, experts, grouping);
    const int32_t* order = expert_order(grouping, experts);
    expert_passes(q, grouping, experts, passes);
    expert_gather_rows(q, x, per_slot, slots, order, pairs, k, xh);
    dispatch(w.format, [&]<BlockFormat F>() {
        expert_gemm_block_xmx_t<F>(q, s, xh, order, passes, out, pairs, experts, n, k, compute_units);
    });
}

BlockWeight block_rows(const BlockWeight& w, int64_t first, int64_t k) {
    int64_t stride[kBlockStreams];
    block_stream_rows(w.format, k, stride);
    BlockWeight rows = w;
    for (int i = 0; i < kBlockStreams; ++i) {
        if (rows.s[i] != nullptr) rows.s[i] += first * stride[i];
    }
    return rows;
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

template void expert_matmul_block<int32_t>(sycl::queue&, const float*, bool, const BlockWeight&, const int32_t*, float*,
                                           int64_t, int64_t, int64_t, int64_t, int64_t, int32_t*, int64_t);
template void expert_matmul_block<int64_t>(sycl::queue&, const float*, bool, const BlockWeight&, const int64_t*, float*,
                                           int64_t, int64_t, int64_t, int64_t, int64_t, int32_t*, int64_t);
template void expert_gemm_block_xmx<int32_t>(sycl::queue&, const float*, bool, const BlockWeight&, const int32_t*,
                                             float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, void*);
template void expert_gemm_block_xmx<int64_t>(sycl::queue&, const float*, bool, const BlockWeight&, const int64_t*,
                                             float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, void*);
template void gemm_block_xmx<float>(sycl::queue&, const half*, const BlockWeight&, float*, int64_t, int64_t, int64_t,
                                    int64_t, bool);
template void gemm_block_xmx<half>(sycl::queue&, const half*, const BlockWeight&, half*, int64_t, int64_t, int64_t,
                                   int64_t, bool);
template void dequantize_block<half>(sycl::queue&, const BlockWeight&, half*, int64_t, int64_t);
template void dequantize_block<float>(sycl::queue&, const BlockWeight&, float*, int64_t, int64_t);
template void gather_block<int32_t>(sycl::queue&, const BlockWeight&, const int32_t*, float*, int64_t, int64_t);
template void gather_block<int64_t>(sycl::queue&, const BlockWeight&, const int64_t*, float*, int64_t, int64_t);

}  // namespace xs::sycl_be
