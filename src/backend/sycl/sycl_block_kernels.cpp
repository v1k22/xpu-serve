#include "backend/sycl/sycl_block_kernels.h"

#include <algorithm>
#include <array>
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
                            const auto* xp = reinterpret_cast<const sycl::vec<float, 4>*>(
                                xs[mm] + static_cast<int64_t>(u) * kUnit + 8 * t);
                            const sycl::vec<float, 4> x0 = xp[0];
                            const sycl::vec<float, 4> x1 = xp[1];
                            if constexpr (Fmt<F>::kTrick) {
                                xsum[mm][t >> 1] +=
                                    ((x0[0] + x0[1]) + (x0[2] + x0[3])) + ((x1[0] + x1[1]) + (x1[2] + x1[3]));
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
                        const auto* xp =
                            reinterpret_cast<const sycl::vec<float, 4>*>(xs[mm] + static_cast<int64_t>(u) * kUnit);
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

// The GEMV on the matrix engines (gemv_block_xmx), M <= 8: gemv_quant_xmx's design over the block formats. A
// sub-group owns 16 output features, one per lane (the DPAS B operand's columns), and a slice of K. Per block of 256
// columns a lane loads its own feature's run of each stream whole, a load or two each (per lane every load is a row
// of its own, so few and wide loads are what keeps the memory busy), then per unit writes its codes less their
// offset, or its levels, as f16 into two B operands: small integers (IQ1's in eighths), so exact. Each
// multiplies an 8 x 16 tile of x in f16 (a 2D block load; rows past M read as zero), and the unit's scale applies to
// the sum, less its minimum times x's sum over those columns where the format has one. The decoding is the work per
// weight, and every input shares it: the cost hardly grows with M.
constexpr int kXmxCols = 16;  // columns of a DPAS, and of the sums of x a minimum multiplies

// Codes 2 i and 2 i + 1 of a kTrick quarter less kOff, as a B operand word (two f16, column 2 i in the low half): a
// code below 1024 under the exponent of 1024 is exactly the f16 1024 + code, so one packed subtract makes the pair.
template <BlockFormat F>
inline int code_pair(uint32_t low, uint32_t high, int i) {
    constexpr int L = Fmt<F>::kLowBits;
    constexpr int H = Fmt<F>::kHighBits;
    constexpr uint32_t kLow = (1u << L) - 1u;
    const uint32_t lo = (low >> (2 * L * i)) & ((1u << (2 * L)) - 1u);
    uint32_t p = (lo | (lo << (16 - L))) & (kLow | (kLow << 16));
    if constexpr (H > 0) {
        constexpr uint32_t kHigh = (1u << H) - 1u;
        const uint32_t hi = (high >> (2 * H * i)) & ((1u << (2 * H)) - 1u);
        p |= ((hi | (hi << (16 - H))) & (kHigh | (kHigh << 16))) << L;
    }
    constexpr auto kBias = static_cast<_Float16>(1024 + Fmt<F>::kOff);
    return __builtin_bit_cast(int, __builtin_bit_cast(h2_v, p | 0x64006400u) - h2_v{kBias, kBias});
}
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
                b[4 * q + i] = code_pair<F>(low, high, i);
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

template <BlockFormat F>
struct GemvBlockXmx {
    using Sums = sycl::vec<float, 8>;
    static constexpr std::array<int, kBlockStreams> kBytes = Fmt<F>::kSuperblock;
    static constexpr int kStreams = kBytes[4] ? 5 : kBytes[3] ? 4 : kBytes[2] ? 3 : 2;
    static constexpr int words(int i) { return (kBytes[i] + 3) / 4; }
    static constexpr int kWords = words(0) + words(1) + words(2) + words(3) + words(4);

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
    Sums x_sums(int32_t u, int h) const {
        return *reinterpret_cast<const Sums*>(xs + (static_cast<int64_t>(u) * 2 + h) * 8);
    }

    // Unit u (columns 32 u .. 32 u + 31) into acc, its fields read from `r`.
    void unit(const RowPtr& r, int32_t j, int32_t u, float8_v& acc) const {
        const auto bop = [&](const auto& f, int h) { return b_operand<F>(f, w.grid, pairs, h); };
        const auto f = Fmt<F>::fields(r, j);
        if constexpr (Fmt<F>::kGroups == 1) {  // one scale for the unit: both halves into one sum
            float8_v dot = 0.0f;
            for (int h = 0; h < 2; ++h)
                dot = __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(dot, x_tile(u * kUnit + kXmxCols * h), bop(f, h));
            const Affine af = Fmt<F>::affine(f, 0);
            if constexpr (Fmt<F>::kMin) {
                const Sums s = x_sums(u, 0) + x_sums(u, 1);
                for (int r2 = 0; r2 < 8; ++r2)
                    acc[r2] = sycl::fma(af.s, dot[r2], sycl::fma(-af.m, s[r2], acc[r2]));
            } else {
                for (int r2 = 0; r2 < 8; ++r2)
                    acc[r2] = sycl::fma(af.s, dot[r2], acc[r2]);
            }
        } else {
            for (int h = 0; h < 2; ++h) {
                const float8_v dot = __builtin_IB_sub_group16_fdpas_f_f_hf_hf_8_8(
                    float8_v(0.0f), x_tile(u * kUnit + kXmxCols * h), bop(f, h));
                const Affine af = Fmt<F>::affine(f, h);
                if constexpr (Fmt<F>::kMin) {
                    const Sums s = x_sums(u, h);
                    for (int r2 = 0; r2 < 8; ++r2)
                        acc[r2] = sycl::fma(af.s, dot[r2], sycl::fma(-af.m, s[r2], acc[r2]));
                } else {
                    for (int r2 = 0; r2 < 8; ++r2)
                        acc[r2] = sycl::fma(af.s, dot[r2], acc[r2]);
                }
            }
        }
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
        float8_v acc = 0.0f;
        for (int32_t b = b_lo; b < b_hi; ++b) {
            // a block's run of each stream, then its eight units decode from registers
            uint32_t copy[kWords];
            RowPtr r;
            [&]<int... I>(std::integer_sequence<int, I...>) {
                ((load_words<kBytes[I]>(rp.p[I] + b * kBytes[I], copy + offset(I)),
                  r.p[I] = reinterpret_cast<const uint8_t*>(copy + offset(I))),
                 ...);
            }(std::make_integer_sequence<int, kStreams>{});
            #pragma unroll
            for (int32_t j = 0; j < 8; ++j)
                unit(r, j, b * 8 + j, acc);
        }
        // The slices add up in local memory; element r of acc at lane l is y[r][row].
        for (int r2 = 0; r2 < 8; ++r2)
            slm[(slice * 8 + r2) * kSubGroup + lane] = acc[r2];
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

    static constexpr int offset(int i) {
        int o = 0;
        for (int j = 0; j < i; ++j)
            o += words(j);
        return o;
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
    // About 8 sub-groups per compute unit streaming the weight, each slice of K at least two blocks of 256 columns.
    int slices = 1;
    while (slices < kQuantGemmMaxSlices && wgs * slices < 8 * compute_units && blocks / (slices * 2) >= 2)
        slices *= 2;
    const auto slice_blocks = static_cast<int32_t>(ceil_div(blocks, slices));
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(static_cast<size_t>(slices * 8 * kSubGroup)), h);
        sycl::local_accessor<uint32_t, 1> pairs(sycl::range<1>(kPairTable<F> ? 256 : 1), h);
        const auto local = static_cast<size_t>(slices * kSubGroup);
        syclex::nd_launch(h, sycl::nd_range<1>(static_cast<size_t>(wgs) * local, local),
                          GemvBlockXmx<F>{w, y, xh, xs, m, n, k, slice_blocks, slm, pairs});
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
