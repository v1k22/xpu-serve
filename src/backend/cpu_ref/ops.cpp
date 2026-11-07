#include "backend/cpu_ref/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <vector>

#include "util/error.h"

namespace xs::cpu_ref {

namespace {

constexpr int64_t kMatmulColumnsPerItem = 16;  // weight rows a work item keeps hot in cache across all x rows

void check_f32(const HostTensor& t, const char* what) {
    XS_CHECK(t.dtype() == DType::F32, what, " must be f32, got ", dtype_name(t.dtype()));
}

int64_t int_at(const HostTensor& t, int64_t i) {
    switch (t.dtype()) {
        case DType::I32:
            return t.data<int32_t>()[i];
        case DType::I64:
            return t.data<int64_t>()[i];
        default:
            fail("expected an integer tensor, got ", dtype_name(t.dtype()));
    }
}

template <typename Fn>
HostTensor map_unary(const HostTensor& x, Fn fn) {
    check_f32(x, "elementwise operand");
    HostTensor out(DType::F32, x.shape());
    std::transform(x.data<float>(), x.data<float>() + x.numel(), out.data<float>(), fn);
    return out;
}

// numpy broadcasting: shapes align on the right; a dim of 1 stretches.
Dims broadcast_shape(const Dims& a, const Dims& b) {
    const size_t rank = std::max(a.size(), b.size());
    Dims out(rank);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
        const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
        XS_CHECK(da == db || da == 1 || db == 1, "cannot broadcast ", dims_to_string(a), " with ", dims_to_string(b));
        out[i] = da == 1 ? db : da;
    }
    return out;
}

// Row-major element strides of `shape` aligned to `out`, with stride 0 on broadcast dims.
std::vector<int64_t> broadcast_strides(const Dims& shape, const Dims& out) {
    std::vector<int64_t> strides(out.size(), 0);
    int64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        const size_t o = i + (out.size() - shape.size());
        strides[o] = shape[i] == 1 ? 0 : stride;
        stride *= shape[i];
    }
    return strides;
}

template <typename Fn>
HostTensor map_binary(const HostTensor& a, const HostTensor& b, Fn fn) {
    check_f32(a, "elementwise operand");
    check_f32(b, "elementwise operand");
    const Dims out_shape = broadcast_shape(a.shape(), b.shape());
    HostTensor out(DType::F32, out_shape);
    const std::vector<int64_t> sa = broadcast_strides(a.shape(), out_shape);
    const std::vector<int64_t> sb = broadcast_strides(b.shape(), out_shape);
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    float* po = out.data<float>();
    std::vector<int64_t> index(out_shape.size(), 0);
    int64_t oa = 0;
    int64_t ob = 0;
    for (int64_t i = 0; i < out.numel(); ++i) {
        po[i] = fn(pa[oa], pb[ob]);
        for (size_t d = out_shape.size(); d-- > 0;) {  // odometer increment of the output index
            oa += sa[d];
            ob += sb[d];
            if (++index[d] < out_shape[d]) break;
            oa -= sa[d] * out_shape[d];
            ob -= sb[d] * out_shape[d];
            index[d] = 0;
        }
    }
    return out;
}

// f32 products are exact in double; four partial sums keep the loop pipelined.
double dot(const float* a, const float* b, int64_t n) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += static_cast<double>(a[i]) * b[i];
        s1 += static_cast<double>(a[i + 1]) * b[i + 1];
        s2 += static_cast<double>(a[i + 2]) * b[i + 2];
        s3 += static_cast<double>(a[i + 3]) * b[i + 3];
    }
    for (; i < n; ++i)
        s0 += static_cast<double>(a[i]) * b[i];
    return (s0 + s1) + (s2 + s3);
}

// Softmax-weighted sum of v rows for one query head; keys [first, last] are the visible ones.
void attend_one(const float* q, const HostTensor& k_all, const HostTensor& v_all, int64_t kv_head, int64_t first,
                int64_t last, const graph::AttentionAttrs& attrs, float* out) {
    const int64_t kv_heads = k_all.dim(1);
    const int64_t head_dim = k_all.dim(2);
    const float* k = k_all.data<float>();
    const float* v = v_all.data<float>();
    std::vector<double> scores(static_cast<size_t>(last - first + 1));
    double max_score = -std::numeric_limits<double>::infinity();
    for (int64_t j = first; j <= last; ++j) {
        double s = attrs.scale * dot(q, k + (j * kv_heads + kv_head) * head_dim, head_dim);
        if (attrs.softcap > 0.0f) s = attrs.softcap * std::tanh(s / attrs.softcap);
        scores[static_cast<size_t>(j - first)] = s;
        max_score = std::max(max_score, s);
    }
    double denom = 0;
    for (double& s : scores) {
        s = std::exp(s - max_score);
        denom += s;
    }
    std::vector<double> acc(static_cast<size_t>(head_dim), 0.0);
    for (int64_t j = first; j <= last; ++j) {
        const double p = scores[static_cast<size_t>(j - first)] / denom;
        const float* vj = v + (j * kv_heads + kv_head) * head_dim;
        for (int64_t d = 0; d < head_dim; ++d)
            acc[static_cast<size_t>(d)] += p * vj[d];
    }
    for (int64_t d = 0; d < head_dim; ++d)
        out[d] = static_cast<float>(acc[static_cast<size_t>(d)]);
}

}  // namespace

HostTensor gather(const HostTensor& table, const HostTensor& ids) {
    check_f32(table, "gather table");
    const int64_t rows = table.dim(0);
    const int64_t width = table.dim(1);
    HostTensor out(DType::F32, {ids.numel(), width});
    for (int64_t i = 0; i < ids.numel(); ++i) {
        const int64_t id = int_at(ids, i);
        XS_CHECK(id >= 0 && id < rows, "gather id ", id, " at index ", i, " is outside the table of ", rows, " rows");
        std::memcpy(out.data<float>() + i * width, table.data<float>() + id * width,
                    static_cast<size_t>(width) * sizeof(float));
    }
    return out;
}

HostTensor matmul(ThreadPool& pool, const HostTensor& x, const HostTensor& w) {
    check_f32(x, "matmul x");
    check_f32(w, "matmul weight");
    const int64_t n = w.dim(0);
    const int64_t k = w.dim(1);
    XS_CHECK(x.rank() >= 1 && x.shape().back() == k, "matmul x ", dims_to_string(x.shape()), " does not match weight ",
             dims_to_string(w.shape()));
    const int64_t m = x.numel() / k;
    Dims out_shape = x.shape();
    out_shape.back() = n;
    HostTensor out(DType::F32, out_shape);
    const float* px = x.data<float>();
    const float* pw = w.data<float>();
    float* po = out.data<float>();
    const int64_t items = (n + kMatmulColumnsPerItem - 1) / kMatmulColumnsPerItem;
    pool.parallel_for(items, [&](int64_t item) {
        const int64_t col_end = std::min(n, (item + 1) * kMatmulColumnsPerItem);
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = item * kMatmulColumnsPerItem; col < col_end; ++col) {
                po[row * n + col] = static_cast<float>(dot(px + row * k, pw + col * k, k));
            }
        }
    });
    return out;
}

HostTensor add(const HostTensor& a, const HostTensor& b) {
    return map_binary(a, b, [](float x, float y) { return x + y; });
}

HostTensor mul(const HostTensor& a, const HostTensor& b) {
    return map_binary(a, b, [](float x, float y) { return x * y; });
}

HostTensor scale(const HostTensor& x, float factor) {
    return map_unary(x, [factor](float v) { return v * factor; });
}

HostTensor tanh(const HostTensor& x) {
    return map_unary(x, [](float v) { return static_cast<float>(std::tanh(static_cast<double>(v))); });
}

HostTensor gelu_tanh(const HostTensor& x) {
    return map_unary(x, [](float v) {
        const double d = v;
        const double inner = std::sqrt(2.0 / std::numbers::pi) * (d + 0.044715 * d * d * d);
        return static_cast<float>(0.5 * d * (1.0 + std::tanh(inner)));
    });
}

HostTensor softcap(const HostTensor& x, float cap) {
    return map_unary(x, [cap](float v) {
        return static_cast<float>(static_cast<double>(cap) * std::tanh(static_cast<double>(v) / cap));
    });
}

HostTensor clamp(const HostTensor& x, float lo, float hi) {
    return map_unary(x, [lo, hi](float v) { return std::min(std::max(v, lo), hi); });
}

HostTensor rms_norm(const HostTensor& x, const HostTensor* weight, float eps) {
    check_f32(x, "rms_norm x");
    const int64_t width = x.shape().back();
    if (weight != nullptr) {
        check_f32(*weight, "rms_norm weight");
        XS_CHECK(weight->numel() == width, "rms_norm weight has ", weight->numel(), " elements for rows of ", width);
    }
    HostTensor out(DType::F32, x.shape());
    for (int64_t row = 0; row < x.numel() / width; ++row) {
        const float* in = x.data<float>() + row * width;
        float* o = out.data<float>() + row * width;
        double sum_sq = 0;
        for (int64_t i = 0; i < width; ++i)
            sum_sq += static_cast<double>(in[i]) * in[i];
        const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(width) + eps);
        for (int64_t i = 0; i < width; ++i) {
            const double w = weight != nullptr ? weight->data<float>()[i] : 1.0;
            o[i] = static_cast<float>(in[i] * inv_rms * w);
        }
    }
    return out;
}

HostTensor reshape(const HostTensor& x, const Dims& shape) {
    return x.clone().reshaped(shape);
}

HostTensor slice(const HostTensor& x, int axis, int64_t begin, int64_t end) {
    XS_CHECK(axis >= 0 && static_cast<size_t>(axis) < x.rank(), "slice axis ", axis, " out of range for ",
             dims_to_string(x.shape()));
    const int64_t dim = x.dim(static_cast<size_t>(axis));
    const int64_t b = begin < 0 ? begin + dim : begin;
    const int64_t e = end == graph::kSliceEnd ? dim : (end < 0 ? end + dim : end);
    XS_CHECK(b >= 0 && b < e && e <= dim, "slice [", begin, ", ", end, ") out of range for runtime dim ", dim);
    int64_t outer = 1;
    for (int i = 0; i < axis; ++i)
        outer *= x.dim(static_cast<size_t>(i));
    const int64_t inner = x.numel() / (outer * dim);
    Dims out_shape = x.shape();
    out_shape[static_cast<size_t>(axis)] = e - b;
    HostTensor out(x.dtype(), out_shape);
    const size_t elem = dtype_size(x.dtype());
    const size_t chunk = static_cast<size_t>((e - b) * inner) * elem;
    const auto* src = static_cast<const unsigned char*>(x.raw());
    auto* dst = static_cast<unsigned char*>(out.raw());
    for (int64_t o = 0; o < outer; ++o) {
        std::memcpy(dst + static_cast<size_t>(o) * chunk, src + static_cast<size_t>((o * dim + b) * inner) * elem,
                    chunk);
    }
    return out;
}

HostTensor transpose(const HostTensor& x, const std::vector<int>& perm) {
    const size_t rank = x.rank();
    XS_CHECK(perm.size() == rank, "transpose of ", dims_to_string(x.shape()), " with ", perm.size(), " axes");
    Dims out_shape(rank);
    for (size_t i = 0; i < rank; ++i)
        out_shape[i] = x.dim(static_cast<size_t>(perm[i]));
    // Element stride in x of each output axis.
    std::vector<int64_t> in_strides(rank, 1);
    for (size_t d = rank; d-- > 1;)
        in_strides[d - 1] = in_strides[d] * x.dim(d);
    std::vector<int64_t> strides(rank);
    for (size_t i = 0; i < rank; ++i)
        strides[i] = in_strides[static_cast<size_t>(perm[i])];
    HostTensor out(x.dtype(), out_shape);
    const size_t elem = dtype_size(x.dtype());
    const auto* src = static_cast<const unsigned char*>(x.raw());
    auto* dst = static_cast<unsigned char*>(out.raw());
    std::vector<int64_t> index(rank, 0);
    int64_t offset = 0;
    for (int64_t i = 0; i < out.numel(); ++i) {
        std::memcpy(dst + static_cast<size_t>(i) * elem, src + static_cast<size_t>(offset) * elem, elem);
        for (size_t d = rank; d-- > 0;) {  // odometer increment of the output index
            offset += strides[d];
            if (++index[d] < out_shape[d]) break;
            offset -= strides[d] * out_shape[d];
            index[d] = 0;
        }
    }
    return out;
}

HostTensor rope(const HostTensor& x, const HostTensor& positions, const HostTensor& inv_freq) {
    check_f32(x, "rope x");
    check_f32(inv_freq, "rope inv_freq");
    const int64_t tokens = x.dim(0);
    const int64_t heads = x.dim(1);
    const int64_t dim = x.dim(2);
    const int64_t sections = positions.rank() == 2 ? positions.dim(1) : 1;
    const int64_t width = dim / sections;
    const int64_t half = width / 2;
    XS_CHECK(positions.numel() == tokens * sections, "rope has ", positions.numel(), " positions for ", tokens,
             " tokens of ", sections, " sections");
    XS_CHECK(inv_freq.numel() == half, "rope inv_freq has ", inv_freq.numel(), " entries, expected ", half);
    HostTensor out(DType::F32, x.shape());
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t s = 0; s < sections; ++s) {
            const double pos = static_cast<double>(int_at(positions, t * sections + s));
            for (int64_t i = 0; i < half; ++i) {
                const double angle = pos * inv_freq.data<float>()[i];
                const double c = std::cos(angle);
                const double sn = std::sin(angle);
                for (int64_t h = 0; h < heads; ++h) {
                    const int64_t base = (t * heads + h) * dim + s * width;
                    const double x1 = x.data<float>()[base + i];
                    const double x2 = x.data<float>()[base + half + i];
                    out.data<float>()[base + i] = static_cast<float>(x1 * c - x2 * sn);
                    out.data<float>()[base + half + i] = static_cast<float>(x2 * c + x1 * sn);
                }
            }
        }
    }
    return out;
}

HostTensor attention(ThreadPool& pool, const HostTensor& q, const HostTensor& k_all, const HostTensor& v_all,
                     const graph::AttentionAttrs& attrs) {
    check_f32(q, "attention q");
    check_f32(k_all, "attention k");
    check_f32(v_all, "attention v");
    const int64_t tokens = q.dim(0);
    const int64_t q_heads = q.dim(1);
    const int64_t head_dim = q.dim(2);
    const int64_t kv_len = k_all.dim(0);
    const int64_t kv_heads = k_all.dim(1);
    XS_CHECK(!attrs.causal || kv_len >= tokens, "attention has ", kv_len, " keys for ", tokens, " queries");
    XS_CHECK(k_all.shape() == v_all.shape() && k_all.dim(2) == head_dim && q_heads % kv_heads == 0,
             "attention shapes q ", dims_to_string(q.shape()), ", k ", dims_to_string(k_all.shape()), ", v ",
             dims_to_string(v_all.shape()), " do not fit");
    const int64_t group = q_heads / kv_heads;
    HostTensor out(DType::F32, q.shape());
    pool.parallel_for(tokens * q_heads, [&](int64_t item) {
        const int64_t t = item / q_heads;
        const int64_t h = item % q_heads;
        const int64_t pos = kv_len - tokens + t;
        const int64_t first = attrs.sliding_window > 0 ? std::max<int64_t>(0, pos - attrs.sliding_window + 1) : 0;
        const int64_t last = attrs.causal ? pos : kv_len - 1;
        const int64_t offset = (t * q_heads + h) * head_dim;
        attend_one(q.data<float>() + offset, k_all, v_all, h / group, first, last, attrs, out.data<float>() + offset);
    });
    return out;
}

}  // namespace xs::cpu_ref
