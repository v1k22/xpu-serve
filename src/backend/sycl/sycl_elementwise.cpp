#include "backend/sycl/sycl_elementwise.h"

#include <array>
#include <limits>

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include "backend/sycl/sycl_unary.h"
#include "util/error.h"

namespace xs::sycl_be {

namespace syclex = sycl::ext::oneapi::experimental;

std::optional<std::vector<EwStep>> allocate_registers(int leaves, const std::vector<std::pair<int, int>>& operands,
                                                      const std::vector<bool>& unary) {
    const auto steps = static_cast<int>(operands.size());
    if (leaves > kEwLeaves || steps > kEwSteps || leaves > kEwRegs) return std::nullopt;
    // A value: leaf l is l, step i is leaves + i. Its last reader, and the register it holds.
    const auto value = [&](int ref) { return ref < 0 ? -1 - ref : leaves + ref; };
    std::vector<int> last(static_cast<size_t>(leaves + steps), -1);
    for (int i = 0; i < steps; ++i) {
        const auto [a, b] = operands[static_cast<size_t>(i)];
        last[static_cast<size_t>(value(a))] = i;
        if (!unary[static_cast<size_t>(i)]) last[static_cast<size_t>(value(b))] = i;
    }
    std::vector<int> reg(static_cast<size_t>(leaves + steps), -1);
    std::array<bool, kEwRegs> busy{};
    for (int l = 0; l < leaves; ++l) {
        reg[static_cast<size_t>(l)] = l;
        busy[static_cast<size_t>(l)] = true;
    }
    std::vector<EwStep> out(static_cast<size_t>(steps));
    for (int i = 0; i < steps; ++i) {
        const auto [a, b] = operands[static_cast<size_t>(i)];
        EwStep& s = out[static_cast<size_t>(i)];
        s.a = static_cast<uint8_t>(reg[static_cast<size_t>(value(a))]);
        s.b = unary[static_cast<size_t>(i)] ? s.a : static_cast<uint8_t>(reg[static_cast<size_t>(value(b))]);
        for (const int v : {value(a), unary[static_cast<size_t>(i)] ? value(a) : value(b)}) {
            if (last[static_cast<size_t>(v)] == i) busy[static_cast<size_t>(reg[static_cast<size_t>(v)])] = false;
        }
        int dst = 0;
        while (dst < kEwRegs && busy[static_cast<size_t>(dst)]) ++dst;
        if (dst == kEwRegs) return std::nullopt;
        busy[static_cast<size_t>(dst)] = true;
        reg[static_cast<size_t>(leaves + i)] = dst;
        s.dst = static_cast<uint8_t>(dst);
    }
    return out;
}

// The registers are a small array the compiler keeps in the thread's GRF and indexes in place. (Chains of selects
// over it, the program as a kernel argument indexed by step, or its words read ahead into an array put arrays in
// per-thread memory or cost more than they saved.)
template <typename Y>
void elementwise(sycl::queue& q, const EwProgram* program, int leaves, int steps, const void* const* data, Y* y,
                 int64_t n) {
    XS_CHECK(n <= std::numeric_limits<int32_t>::max(), "sycl: an elementwise program of ", n, " elements");
    XS_CHECK(leaves >= 1 && leaves <= kEwLeaves && steps >= 1 && steps <= kEwSteps, "sycl: an elementwise program of ",
             leaves, " leaves and ", steps, " steps");
    static_assert(kEwLeaves == 6);
    const void* d[kEwLeaves] = {};
    for (int l = 0; l < leaves; ++l) d[l] = data[l];
    const void *d0 = d[0], *d1 = d[1], *d2 = d[2], *d3 = d[3], *d4 = d[4], *d5 = d[5];

    syclex::parallel_for(q, sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> idx) {
        const auto e = static_cast<int32_t>(idx[0]);
        float r[kEwRegs];
#pragma unroll
        for (int l = 0; l < kEwLeaves; ++l) {
            if (l < leaves) {
                const EwLeaf f = program->leaf[l];
                const void* src = l == 0 ? d0 : l == 1 ? d1 : l == 2 ? d2 : l == 3 ? d3 : l == 4 ? d4 : d5;
                const int32_t at = f.inner == 1 ? e * f.outer + f.offset
                                                : (e / f.inner) * f.outer + (e % f.inner) * f.step + f.offset;
                r[l] = f.f16 != 0 ? static_cast<float>(static_cast<const half*>(src)[at])
                                  : static_cast<const float*>(src)[at];
            }
        }
        int dst = 0;
        for (int s = 0; s < steps; ++s) {
            const uint32_t c = program->code[s];
            const float a = r[c >> 6 & 7];
            const float b = r[c >> 9 & 7];
            float v;
            switch (static_cast<EwOp>(c & 3)) {
                case EwOp::Add: v = a + b; break;
                case EwOp::Mul: v = a * b; break;
                case EwOp::Div: v = a / b; break;
                default: v = apply_unary(static_cast<UnaryOp>(c >> 2 & 15), a, program->p[s], program->q[s]); break;
            }
            dst = static_cast<int>(c >> 12 & 7);
            r[dst] = v;
        }
        y[e] = static_cast<Y>(r[dst]);
    });
}

template void elementwise<float>(sycl::queue&, const EwProgram*, int, int, const void* const*, float*, int64_t);
template void elementwise<half>(sycl::queue&, const EwProgram*, int, int, const void* const*, half*, int64_t);

}  // namespace xs::sycl_be
