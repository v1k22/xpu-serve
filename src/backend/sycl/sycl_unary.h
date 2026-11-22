#pragma once

// The unary ops of the SYCL backend's elementwise kernels, as device code: by the op known at compile time, or chosen
// at run time (an elementwise program, sycl_elementwise.h).

#include <sycl/sycl.hpp>

#include "backend/sycl/sycl_kernels.h"

namespace xs::sycl_be {

// `a` is the scale factor, softcap or lower bound, `b` the upper bound.
template <UnaryOp Op>
inline float apply_unary(float x, float a, float b) {
    if constexpr (Op == UnaryOp::Scale) {
        return x * a;
    } else if constexpr (Op == UnaryOp::Tanh) {
        return sycl::tanh(x);
    } else if constexpr (Op == UnaryOp::GeluTanh) {
        // 1 + tanh(u) = 2 - 2 / (exp(2u) + 1): the hardware exponential instead of sycl::tanh's long sequence (the
        // product with the gate is a large pass), absolute error near 1e-7.
        const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return x * (1.0f - 1.0f / (sycl::native::exp(2.0f * inner) + 1.0f));
    } else if constexpr (Op == UnaryOp::Gelu) {
        return 0.5f * x * (1.0f + sycl::erf(x * 0.7071067811865476f));
    } else if constexpr (Op == UnaryOp::Softcap) {
        return a * sycl::tanh(x / a);
    } else if constexpr (Op == UnaryOp::Sigmoid) {
        return 1.0f / (1.0f + sycl::exp(-x));
    } else if constexpr (Op == UnaryOp::Silu) {
        return x / (1.0f + sycl::exp(-x));
    } else if constexpr (Op == UnaryOp::Softplus) {
        return x > 20.0f ? x : sycl::log1p(sycl::exp(x));  // torch's threshold: identity above 20
    } else {
        return sycl::fmin(sycl::fmax(x, a), b);
    }
}

inline float apply_unary(UnaryOp op, float x, float a, float b) {
    switch (op) {
        case UnaryOp::Scale: return apply_unary<UnaryOp::Scale>(x, a, b);
        case UnaryOp::Tanh: return apply_unary<UnaryOp::Tanh>(x, a, b);
        case UnaryOp::GeluTanh: return apply_unary<UnaryOp::GeluTanh>(x, a, b);
        case UnaryOp::Gelu: return apply_unary<UnaryOp::Gelu>(x, a, b);
        case UnaryOp::Softcap: return apply_unary<UnaryOp::Softcap>(x, a, b);
        case UnaryOp::Sigmoid: return apply_unary<UnaryOp::Sigmoid>(x, a, b);
        case UnaryOp::Silu: return apply_unary<UnaryOp::Silu>(x, a, b);
        case UnaryOp::Softplus: return apply_unary<UnaryOp::Softplus>(x, a, b);
        case UnaryOp::Clamp: return apply_unary<UnaryOp::Clamp>(x, a, b);
    }
    return x;
}

}  // namespace xs::sycl_be
