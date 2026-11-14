// A KV cache's quantized formats on the device: rows written in a format as they are appended, and decoded for the
// attention paths that read floats. The per-row attention kernel reads the codes itself (sycl_kv_rows.h).

#include "backend/sycl/sycl_kernels.h"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include "backend/sycl/sycl_kv_turbo.h"
#include "util/error.h"

namespace xs::sycl_be {

namespace syclex = sycl::ext::oneapi::experimental;
using quant::KvFormat;

namespace {

constexpr int kSubGroup = 16;
constexpr int kSgPerWg = 8;

// One sub-group per unit of work (`units` of them): f(nd_item, unit) in every lane of the units that exist.
template <typename F>
void launch_sub_groups(sycl::queue& q, int64_t units, F f) {
    const int64_t wgs = (units + kSgPerWg - 1) / kSgPerWg;
    syclex::nd_launch(q, sycl::nd_range<1>(static_cast<size_t>(wgs * kSgPerWg * kSubGroup), kSgPerWg * kSubGroup),
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
        const int64_t unit = static_cast<int64_t>(it.get_group(0)) * kSgPerWg +
                             static_cast<int64_t>(it.get_sub_group().get_group_linear_id());
        if (unit < units) f(it, unit);
    });
}

// A row per sub-group, rounded as quant::quantize_kv_row_i8.
void quantize_i8(sycl::queue& q, const float* x, int8_t* codes, float* scales, int64_t rows, int64_t width) {
    launch_sub_groups(q, rows, [=](sycl::nd_item<1> it, int64_t r) {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const float* xr = x + r * width;
        float amax = 0.0f;
        for (int64_t i = lane; i < width; i += kSubGroup)
            amax = sycl::fmax(amax, sycl::fabs(xr[i]));
        amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        int8_t* cr = codes + r * width;
        for (int64_t i = lane; i < width; i += kSubGroup)
            cr[i] = static_cast<int8_t>(sycl::clamp(sycl::rint(xr[i] * inv), -127.0f, 127.0f));
        if (lane == 0) scales[r] = amax / 127.0f;
    });
}

// A group of 128 values per sub-group: quant::turbo::quantize_group's arithmetic, each lane on its 8 values.
template <int Bits>
void quantize_turbo(sycl::queue& q, const float* x, uint8_t* codes, half* norms, int64_t groups) {
    launch_sub_groups(q, groups, [=](sycl::nd_item<1> it, int64_t g) {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const float* xl = x + g * turbo::kGroup + lane * turbo::kLaneValues;
        float v[turbo::kLaneValues];
        float n = 0.0f;
        for (int e = 0; e < turbo::kLaneValues; ++e) {
            v[e] = xl[e];
            n = sycl::fma(v[e], v[e], n);
        }
        n = turbo::sum_lanes(sg, n);
        turbo::flip(v, turbo::lane_signs(turbo::kSignsBefore[0], turbo::kSignsBefore[1], lane));
        turbo::butterflies(sg, lane, v);
        turbo::flip(v, turbo::lane_signs(turbo::kSignsAfter[0], turbo::kSignsAfter[1], lane));
        turbo::LaneCodes<Bits> packed;
        float r = 0.0f;
        for (int e = 0; e < turbo::kLaneValues; ++e) {
            const bool negative = v[e] < 0.0f;
            const float sq = v[e] * v[e];
            uint32_t m = 0;
            for (int j = 0; j < turbo::kMagnitudes<Bits> - 1; ++j) {
                const float edge = turbo::edge_sq<Bits>(j) * n;
                m += (negative ? sq > edge : sq >= edge) ? 1u : 0u;
            }
            packed.set(e, turbo::code_of<Bits>(negative, m));
            r += turbo::kTables<Bits>.mag_sq[m];
        }
        r = turbo::sum_lanes(sg, r);
        packed.store(codes + g * turbo::group_bytes(Bits), lane);
        if (lane == 0) norms[g] = static_cast<half>(n > 0.0f ? sycl::sqrt(n / r) : 0.0f);
    });
}

template <typename CT>
void dequantize_i8(sycl::queue& q, const int8_t* codes, const float* scales, CT* out, int64_t rows, int64_t width) {
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(rows * width)), [=](sycl::id<1> idx) {
        const int64_t i = static_cast<int64_t>(idx[0]);
        out[i] = static_cast<CT>(static_cast<float>(codes[i]) * scales[i / width]);
    });
}

template <int Bits, typename CT>
void dequantize_turbo(sycl::queue& q, const uint8_t* codes, const half* norms, CT* out, int64_t groups) {
    launch_sub_groups(q, groups, [=](sycl::nd_item<1> it, int64_t g) {
        const int lane = static_cast<int>(it.get_sub_group().get_local_linear_id());
        const auto c = turbo::LaneCodes<Bits>::load(codes + g * turbo::group_bytes(Bits), lane);
        const float norm = static_cast<float>(norms[g]);
        CT* o = out + g * turbo::kGroup + lane * turbo::kLaneValues;
        for (int e = 0; e < turbo::kLaneValues; ++e)
            o[e] = static_cast<CT>(turbo::kTables<Bits>.centroid[c.code(e)] * norm);
    });
}

void check_turbo_width(int64_t width) {
    XS_CHECK(width % turbo::kGroup == 0, "sycl: a TurboQuant KV format needs rows of a multiple of ", turbo::kGroup,
             " values, got ", width);
}

}  // namespace

bool kv_formats_supported(const quant::KvFormats& f) {
    if (f.k == f.v) return true;
    return f.k == KvFormat::I8 && quant::turbo_bits(f.v) != 0;
}

void quantize_kv(sycl::queue& q, KvFormat format, const float* x, uint8_t* codes, uint8_t* scales, int64_t rows,
                 int64_t width) {
    if (format == KvFormat::I8) {
        return quantize_i8(q, x, reinterpret_cast<int8_t*>(codes), reinterpret_cast<float*>(scales), rows, width);
    }
    check_turbo_width(width);
    auto* norms = reinterpret_cast<half*>(scales);
    const int64_t groups = rows * (width / turbo::kGroup);
    switch (quant::turbo_bits(format)) {
        case 4: return quantize_turbo<4>(q, x, codes, norms, groups);
        case 3: return quantize_turbo<3>(q, x, codes, norms, groups);
        case 2: return quantize_turbo<2>(q, x, codes, norms, groups);
        default: fail("sycl quantize_kv: ", quant::kv_format_name(format), " is not a quantized KV format");
    }
}

template <typename CT>
void dequantize_kv(sycl::queue& q, const KvStream& s, CT* out, int64_t rows, int64_t width) {
    if (s.format == KvFormat::I8) {
        return dequantize_i8(q, reinterpret_cast<const int8_t*>(s.codes), reinterpret_cast<const float*>(s.scales), out,
                             rows, width);
    }
    check_turbo_width(width);
    const auto* norms = reinterpret_cast<const half*>(s.scales);
    const int64_t groups = rows * (width / turbo::kGroup);
    switch (quant::turbo_bits(s.format)) {
        case 4: return dequantize_turbo<4>(q, s.codes, norms, out, groups);
        case 3: return dequantize_turbo<3>(q, s.codes, norms, out, groups);
        case 2: return dequantize_turbo<2>(q, s.codes, norms, out, groups);
        default: fail("sycl dequantize_kv: ", quant::kv_format_name(s.format), " is not a quantized KV format");
    }
}

void kv_rotate(sycl::queue& q, const float* x, float* y, int64_t rows, int64_t width, bool inverse) {
    check_turbo_width(width);
    launch_sub_groups(q, rows * (width / turbo::kGroup), [=](sycl::nd_item<1> it, int64_t g) {
        const sycl::sub_group sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int64_t at = g * turbo::kGroup + lane * turbo::kLaneValues;
        float v[turbo::kLaneValues];
        for (int e = 0; e < turbo::kLaneValues; ++e)
            v[e] = x[at + e];
        turbo::rotate(sg, lane, v, inverse);
        for (int e = 0; e < turbo::kLaneValues; ++e)
            y[at + e] = v[e];
    });
}

template void dequantize_kv<half>(sycl::queue&, const KvStream&, half*, int64_t, int64_t);
template void dequantize_kv<float>(sycl::queue&, const KvStream&, float*, int64_t, int64_t);

}  // namespace xs::sycl_be
