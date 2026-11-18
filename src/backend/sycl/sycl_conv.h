#pragma once

// Conv2d of the SYCL backend (graph.h): each group's input unfolded into rows of patches, one GEMM, then the result
// laid out NCHW with the bias added.

#include <cstdint>

#include "backend/sycl/sycl_matmul.h"
#include "graph/graph.h"

namespace xs::sycl_be {

struct Conv2dShape {
    int64_t n = 0, c = 0, h = 0, w = 0;  // input
    int64_t o = 0, kh = 0, kw = 0;       // weight [O, C / groups, KH, KW]
    int64_t ho = 0, wo = 0;              // output
    graph::Conv2dAttrs attrs;
};

// out [N, O, HO, WO] from x [N, C, H, W]; w in f16 or f32, bias [O] or null.
template <typename W>
void conv2d(sycl::queue& q, Scratch& scratch, const float* x, const W* w, const float* bias, float* out,
            const Conv2dShape& s);

}  // namespace xs::sycl_be
