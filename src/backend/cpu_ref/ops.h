#pragma once

// The reference kernels, one per graph op. Float operands are f32 tensors with concrete shapes; outputs are freshly
// allocated. Semantics follow the OpKind comments in graph/graph.h.

#include <cstdint>

#include "backend/cpu_ref/thread_pool.h"
#include "graph/graph.h"
#include "tensor/tensor.h"

namespace xs::cpu_ref {

HostTensor gather(const HostTensor& table, const HostTensor& ids);
HostTensor matmul(ThreadPool& pool, const HostTensor& x, const HostTensor& w);
HostTensor add(const HostTensor& a, const HostTensor& b);
HostTensor mul(const HostTensor& a, const HostTensor& b);
HostTensor scale(const HostTensor& x, float factor);
HostTensor tanh(const HostTensor& x);
HostTensor gelu_tanh(const HostTensor& x);
HostTensor softcap(const HostTensor& x, float cap);
// `weight` may be null (no scaling).
HostTensor rms_norm(const HostTensor& x, const HostTensor* weight, float eps);
// Any dtype; `shape` is concrete.
HostTensor reshape(const HostTensor& x, const Dims& shape);
// Any dtype; begin/end as in graph::SliceAttrs, resolved against the runtime dim.
HostTensor slice(const HostTensor& x, int axis, int64_t begin, int64_t end);
HostTensor rope(const HostTensor& x, const HostTensor& positions, const HostTensor& inv_freq);
HostTensor attention(ThreadPool& pool, const HostTensor& q, const HostTensor& k_all, const HostTensor& v_all,
                     const graph::AttentionAttrs& attrs);

}  // namespace xs::cpu_ref
