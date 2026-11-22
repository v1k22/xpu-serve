#pragma once

// An elementwise program: a tree of elementwise ops over one output shape run as one kernel, each element's operands
// read through a map and the values between the ops kept in a few registers (graph/elementwise_groups.h finds the
// trees). The program lives in device memory and the kernel reads it as it runs: indexing a kernel argument by a
// step's number would copy the argument into per-thread memory, which costs more than the launches it saves.

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "backend/sycl/sycl_kernels.h"

namespace xs::sycl_be {

inline constexpr int kEwLeaves = 6;
inline constexpr int kEwSteps = 12;
inline constexpr int kEwRegs = 8;  // three bits of a step's word each

// Element e of the output reads (e / inner) * outer + (e % inner) * step + offset of a leaf: a value of the output's
// shape is inner 1, outer 1, step 0; one repeating every P elements inner P, outer 0, step 1; one held across blocks
// of B elements inner B, outer 1, step 0; a slice of `span` elements of every `stride` from `begin` inner span,
// outer stride, step 1, offset begin.
struct EwLeaf {
    int32_t inner = 1;
    int32_t outer = 1;
    int32_t step = 0;
    int32_t offset = 0;
    int32_t f16 = 0;
};

enum class EwOp : uint8_t { Add, Mul, Div, Unary };

// Register dst = op(register a, register b), or unary(register a; p, q). Leaf l is loaded into register l.
struct EwStep {
    EwOp op = EwOp::Add;
    UnaryOp unary = UnaryOp::Scale;
    uint8_t a = 0;
    uint8_t b = 0;
    uint8_t dst = 0;
    float p = 0.0f;
    float q = 0.0f;
};

// As the kernel reads it: each step's op, unary, a, b and dst packed in a word (pack_step), beside its p and q.
struct EwProgram {
    EwLeaf leaf[kEwLeaves];
    uint32_t code[kEwSteps];
    float p[kEwSteps];
    float q[kEwSteps];
};

inline uint32_t pack_step(const EwStep& s) {
    return static_cast<uint32_t>(s.op) | static_cast<uint32_t>(s.unary) << 2 | static_cast<uint32_t>(s.a) << 6 |
           static_cast<uint32_t>(s.b) << 9 | static_cast<uint32_t>(s.dst) << 12;
}

// The registers of a tree's steps, `operands[i]` = {a, b} as leaves (-1 - leaf) or earlier steps (their index), b
// unused where `unary[i]`: each step's dst is the lowest register free once its operands are read. Nothing when the
// tree needs more than kEwRegs or has more than kEwLeaves leaves or kEwSteps steps.
std::optional<std::vector<EwStep>> allocate_registers(int leaves, const std::vector<std::pair<int, int>>& operands,
                                                      const std::vector<bool>& unary);

// y[e] for e < n (n below 2^31) by `program`, in device memory, of `leaves` leaves and `steps` steps (the output is
// the last step's dst); `data[l]` is leaf l's memory, f32 or f16 as its map says. Y = float or half.
template <typename Y>
void elementwise(sycl::queue& q, const EwProgram* program, int leaves, int steps, const void* const* data, Y* y,
                 int64_t n);

}  // namespace xs::sycl_be
