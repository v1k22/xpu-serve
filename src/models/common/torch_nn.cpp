#include "models/common/torch_nn.h"

#include <algorithm>

#include "util/error.h"

namespace xs::torch_nn {

Value linear(Graph& g, const WeightStore& w, Value x, const std::string& stem) {
    return g.add(g.matmul(x, g.weight(w, stem + ".weight")), g.weight(w, stem + ".bias"));
}

Value layer_norm(Graph& g, const WeightStore& w, Value x, const std::string& stem, float eps) {
    return g.layer_norm(x, g.weight(w, stem + ".weight"), g.weight(w, stem + ".bias"), eps);
}

Value layer_norm_without_bias(Graph& g, const WeightStore& w, Value x, const std::string& stem, float eps) {
    const Value weight = g.weight(w, stem + ".weight");
    const Dims& shape = g.type(weight).shape;
    XS_CHECK(shape.size() == 1, "'", stem, ".weight' is ", dims_to_string(shape), ", a LayerNorm weight is a vector");
    HostTensor zero(DType::F32, shape);
    std::fill_n(zero.data<float>(), zero.numel(), 0.0f);
    return g.layer_norm(x, weight, g.constant(std::move(zero)), eps);
}

}  // namespace xs::torch_nn
