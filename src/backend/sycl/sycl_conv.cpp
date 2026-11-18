#include "backend/sycl/sycl_conv.h"

#include <type_traits>

namespace xs::sycl_be {

namespace {

namespace syclex = sycl::ext::oneapi::experimental;

size_t align_up(size_t n) { return (n + 255) & ~size_t{255}; }

// col [N * HO * WO, CG * KH * KW]: row (n, i, j) holds group g's patch under output (i, j), zeros where the kernel
// reaches into the padding.
template <typename T>
void unfold(sycl::queue& q, const float* x, T* col, const Conv2dShape& s, int64_t g) {
    const graph::Conv2dAttrs a = s.attrs;
    const int64_t cg = s.c / a.groups, k = cg * s.kh * s.kw, positions = s.ho * s.wo;
    const int64_t c = s.c, h = s.h, w = s.w, kh = s.kh, kw = s.kw, wo = s.wo;
    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(s.n * positions * k)), [=](sycl::id<1> id) {
        const auto idx = static_cast<int64_t>(id[0]);
        const int64_t col_k = idx % k, row = idx / k;
        const int64_t b = row / positions, p = row % positions;
        const int64_t i = p / wo, j = p % wo;
        const int64_t ci = col_k / (kh * kw), u = (col_k / kw) % kh, v = col_k % kw;
        const int64_t y = i * a.stride_h + u * a.dilation_h - a.pad_top;
        const int64_t z = j * a.stride_w + v * a.dilation_w - a.pad_left;
        const bool inside = y >= 0 && y < h && z >= 0 && z < w;
        const float value = inside ? x[((b * c + g * cg + ci) * h + y) * w + z] : 0.0f;
        col[idx] = static_cast<T>(value);
    });
}

}  // namespace

template <typename W>
void conv2d(sycl::queue& q, Scratch& scratch, const float* x, const W* w, const float* bias, float* out,
            const Conv2dShape& s) {
    using T = std::conditional_t<std::is_same_v<W, half>, half, float>;
    const int64_t groups = s.attrs.groups, cg = s.c / groups, og = s.o / groups;
    const int64_t k = cg * s.kh * s.kw, positions = s.ho * s.wo, rows = s.n * positions;
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
        syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(rows * og)), [=](sycl::id<1> id) {
            const auto idx = static_cast<int64_t>(id[0]);
            const int64_t row = idx / og, oc = idx % og;
            const int64_t b = row / positions, p = row % positions, channel = g * og + oc;
            out[(b * o + channel) * positions + p] = gemm[idx] + (bias != nullptr ? bias[channel] : 0.0f);
        });
    }
}

template void conv2d<half>(sycl::queue&, Scratch&, const float*, const half*, const float*, float*, const Conv2dShape&);
template void conv2d<float>(sycl::queue&, Scratch&, const float*, const float*, const float*, float*,
                            const Conv2dShape&);

}  // namespace xs::sycl_be
