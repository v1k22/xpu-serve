#include "backend/sycl/sycl_conv.h"

#include <cstdint>
#include <type_traits>

#include "util/error.h"

namespace xs::sycl_be {

namespace {

namespace syclex = sycl::ext::oneapi::experimental;

size_t align_up(size_t n) { return (n + 255) & ~size_t{255}; }

// col [N * HO * WO, CG * KH * KW]: row (n, i, j) holds group g's patch under output (i, j), zeros where the kernel
// reaches into the padding. Indices within a row and a patch are 32-bit (64-bit division is emulated on the device).
template <typename T>
void unfold(sycl::queue& q, const float* x, T* col, const Conv2dShape& s, int64_t g) {
    const graph::Conv2dAttrs a = s.attrs;
    const auto i32 = [](int64_t v) { return static_cast<int32_t>(v); };
    const int32_t cg = i32(s.c / a.groups), khw = i32(s.kh * s.kw), kw = i32(s.kw), wo = i32(s.wo);
    const int32_t positions = i32(s.ho * s.wo), h = i32(s.h), w = i32(s.w);
    const int32_t sh = i32(a.stride_h), sw = i32(a.stride_w), dh = i32(a.dilation_h), dw = i32(a.dilation_w);
    const int32_t pt = i32(a.pad_top), pl = i32(a.pad_left);
    const int64_t c = s.c, k = int64_t{cg} * khw, channel0 = g * cg;
    const sycl::range<2> range(static_cast<size_t>(s.n * positions), static_cast<size_t>(k));
    syclex::parallel_for(q, range, [=](sycl::id<2> id) {
        const auto row = static_cast<int32_t>(id[0]), col_k = static_cast<int32_t>(id[1]);
        const int32_t b = row / positions, p = row - b * positions, i = p / wo, j = p - i * wo;
        const int32_t ci = col_k / khw, uv = col_k - ci * khw, u = uv / kw, v = uv - u * kw;
        const int32_t y = i * sh + u * dh - pt, z = j * sw + v * dw - pl;
        const bool inside = y >= 0 && y < h && z >= 0 && z < w;
        const float value = inside ? x[((b * c + channel0 + ci) * h + y) * w + z] : 0.0f;
        col[static_cast<int64_t>(row) * k + col_k] = static_cast<T>(value);
    });
}

}  // namespace

template <typename W>
void conv2d(sycl::queue& q, Scratch& scratch, const float* x, const W* w, const float* bias, float* out,
            const Conv2dShape& s) {
    using T = std::conditional_t<std::is_same_v<W, half>, half, float>;
    const int64_t groups = s.attrs.groups, cg = s.c / groups, og = s.o / groups;
    const int64_t k = cg * s.kh * s.kw, positions = s.ho * s.wo, rows = s.n * positions;
    XS_CHECK(rows < INT32_MAX && k < INT32_MAX, "sycl conv2d: ", rows, " output positions of ", k,
             " values each is past the 32-bit indices it computes with");
    const size_t col_bytes = align_up(static_cast<size_t>(rows * k) * sizeof(T));
    char* base = static_cast<char*>(scratch.get(col_bytes + static_cast<size_t>(rows * og) * sizeof(float)));
    T* col = reinterpret_cast<T*>(base);
    float* gemm = reinterpret_cast<float*>(base + col_bytes);  // [N * HO * WO, OG]
    for (int64_t g = 0; g < groups; ++g) {
        unfold(q, x, col, s, g);
        if constexpr (std::is_same_v<W, half>) {
            matmul_half_x(q, col, w + g * og * k, gemm, rows, og, k);
        } else {
            matmul(q, scratch, col, w + g * og * k, gemm, rows, og, k);
        }
        const int64_t o = s.o;
        const sycl::range<3> range(static_cast<size_t>(s.n), static_cast<size_t>(og), static_cast<size_t>(positions));
        syclex::parallel_for(q, range, [=](sycl::id<3> id) {
            const auto b = static_cast<int64_t>(id[0]), oc = static_cast<int64_t>(id[1]);
            const auto p = static_cast<int64_t>(id[2]);
            const int64_t channel = g * og + oc;
            out[(b * o + channel) * positions + p] =
                gemm[(b * positions + p) * og + oc] + (bias != nullptr ? bias[channel] : 0.0f);
        });
    }
}

template void conv2d<half>(sycl::queue&, Scratch&, const float*, const half*, const float*, float*, const Conv2dShape&);
template void conv2d<float>(sycl::queue&, Scratch&, const float*, const float*, const float*, float*,
                            const Conv2dShape&);

}  // namespace xs::sycl_be
