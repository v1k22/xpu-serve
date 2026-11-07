#pragma once

// Backend-agnostic helpers for backend tests: an in-memory WeightStore, seeded random tensors, fp64 tensors and a
// numpy-style closeness check. Any backend's tests may use them.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "graph/graph.h"
#include "quant/quant.h"
#include "tensor/tensor.h"
#include "util/half.h"

namespace xs::test {

// Weights held in memory as HostTensors.
class MemoryWeights final : public WeightStore {
  public:
    void add(const std::string& name, HostTensor t) {
        tensors_[name] = std::move(t);
        views_[name] = tensors_[name].view();
    }
    // A Q8 / Q4 weight, as a pack holds it. tensor(name) is its dequantized f32 value (what a backend must compute
    // with).
    void add_quantized(const std::string& name, quant::Quantized q) {
        HostTensor dense(DType::F32, {q.rows, q.cols});
        quantized_[name] = std::move(q);
        views_[name] = quantized_[name].view();
        quant::dequantize_rows(views_[name], 0, dense.dim(0), dense.data<float>());
        tensors_[name] = std::move(dense);
    }
    const TensorView* find(const std::string& name) const override {
        const auto it = views_.find(name);
        return it == views_.end() ? nullptr : &it->second;
    }
    const HostTensor& tensor(const std::string& name) const { return tensors_.at(name); }
    std::vector<std::string> names() const override {
        std::vector<std::string> out;
        for (const auto& [name, view] : views_)
            out.push_back(name);
        return out;
    }

  private:
    std::map<std::string, HostTensor> tensors_;
    std::map<std::string, quant::Quantized> quantized_;
    std::map<std::string, TensorView> views_;
};

// Deterministic uniform values in [-range, range], stored as `dtype` (f32, f16 or bf16).
class TensorRng {
  public:
    explicit TensorRng(uint32_t seed) : engine_(seed) {}

    HostTensor uniform(DType dtype, Dims shape, float range = 1.0f) {
        HostTensor t(dtype, std::move(shape));
        std::uniform_real_distribution<float> dist(-range, range);
        for (int64_t i = 0; i < t.numel(); ++i) {
            const float v = dist(engine_);
            switch (dtype) {
                case DType::F32:
                    t.data<float>()[i] = v;
                    break;
                case DType::F16:
                    t.data<uint16_t>()[i] = f32_to_f16(v);
                    break;
                case DType::BF16:
                    t.data<uint16_t>()[i] = f32_to_bf16(v);
                    break;
                default:
                    throw std::invalid_argument("TensorRng::uniform: unsupported dtype");
            }
        }
        return t;
    }

    HostTensor ints(DType dtype, Dims shape, int64_t lo, int64_t hi) {
        HostTensor t(dtype, std::move(shape));
        std::uniform_int_distribution<int64_t> dist(lo, hi);
        for (int64_t i = 0; i < t.numel(); ++i) {
            if (dtype == DType::I64) {
                t.data<int64_t>()[i] = dist(engine_);
            } else {
                t.data<int32_t>()[i] = static_cast<int32_t>(dist(engine_));
            }
        }
        return t;
    }

  private:
    std::mt19937 engine_;
};

template <typename T>
HostTensor make_tensor(DType dtype, Dims shape, const std::vector<T>& values) {
    return HostTensor::from_vector(dtype, std::move(shape), values);
}

inline HostTensor arange_i32(int32_t begin, int32_t end) {
    std::vector<int32_t> v;
    for (int32_t i = begin; i < end; ++i)
        v.push_back(i);
    return make_tensor(DType::I32, {static_cast<int64_t>(v.size())}, v);
}

// Dense fp64 tensor for reference computations.
struct Tensor64 {
    Dims shape;
    std::vector<double> data;

    Tensor64() = default;
    explicit Tensor64(Dims s) : shape(std::move(s)), data(static_cast<size_t>(xs::numel(shape)), 0.0) {}
    Tensor64(Dims s, std::vector<double> values) : shape(std::move(s)), data(std::move(values)) {}
    int64_t numel() const { return static_cast<int64_t>(data.size()); }
    double& operator[](int64_t i) { return data[static_cast<size_t>(i)]; }
    double operator[](int64_t i) const { return data[static_cast<size_t>(i)]; }
};

inline Tensor64 to_f64(const HostTensor& t) {
    Tensor64 out(t.shape());
    const std::vector<float> f = t.to_f32();
    for (size_t i = 0; i < f.size(); ++i)
        out.data[i] = f[i];
    return out;
}

inline std::vector<int64_t> to_ints(const HostTensor& t) {
    std::vector<int64_t> out(static_cast<size_t>(t.numel()));
    for (int64_t i = 0; i < t.numel(); ++i) {
        out[static_cast<size_t>(i)] = t.dtype() == DType::I64 ? t.data<int64_t>()[i] : t.data<int32_t>()[i];
    }
    return out;
}

// Largest |actual - expected|; infinity on a shape mismatch.
inline double max_abs_diff(const HostTensor& actual, const Tensor64& expected) {
    if (actual.shape() != expected.shape) return INFINITY;
    const std::vector<float> a = actual.to_f32();
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::abs(a[i] - expected.data[i]));
    return worst;
}

// numpy.allclose: |actual - expected| <= atol + rtol * |expected| everywhere, shapes equal, no NaN.
inline bool all_close(const HostTensor& actual, const Tensor64& expected, double atol, double rtol) {
    if (actual.shape() != expected.shape) return false;
    const std::vector<float> a = actual.to_f32();
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = expected.data[i];
        if (!(std::abs(a[i] - e) <= atol + rtol * std::abs(e))) return false;
    }
    return true;
}

inline bool all_close(const HostTensor& actual, const HostTensor& expected, double atol, double rtol) {
    return all_close(actual, to_f64(expected), atol, rtol);
}

// Compiles `graph` once and runs a single step.
inline TensorMap run_once(Backend& backend, const graph::Graph& graph, const WeightStore& weights,
                          const TensorMap& inputs, const SessionOptions& options = {}) {
    const std::unique_ptr<Session> session = backend.compile(graph, weights, options);
    TensorMap outputs;
    session->run(inputs, outputs);
    return outputs;
}

// Rows [begin, end) along axis 0.
inline HostTensor rows(const HostTensor& t, int64_t begin, int64_t end) {
    const int64_t width = t.numel() / t.dim(0);
    Dims shape = t.shape();
    shape[0] = end - begin;
    HostTensor out(t.dtype(), shape);
    const size_t elem = dtype_size(t.dtype());
    std::memcpy(out.raw(), static_cast<const unsigned char*>(t.raw()) + static_cast<size_t>(begin * width) * elem,
                out.nbytes());
    return out;
}

}  // namespace xs::test
