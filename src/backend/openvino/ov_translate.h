#pragma once

// Step graph -> ov::Model. Every graph value becomes an f32 (or integer) OpenVINO value with the same logical shape;
// kT becomes a dynamic dimension. Each KV slot becomes two ReadValue -> Concat(axis 2) -> Assign variables holding
// [1, Hkv, L, D], the layout the plugins' stateful KV-cache passes recognize.

#include <memory>
#include <string>
#include <vector>

#include <openvino/core/model.hpp>

#include "backend/openvino/ov_attention.h"
#include "backend/openvino/ov_weights.h"
#include "graph/graph.h"
#include "io/safetensors.h"

namespace xs::ov_backend {

struct TranslateOptions {
    DType precision = DType::F16;  // weight precision for bf16 sources
    AttentionImpl attention = AttentionImpl::Auto;
};

struct InputSpec {
    std::string name;
    DType dtype = DType::F32;
    Dims shape;  // may contain graph::kT
};

struct Translation {
    std::shared_ptr<ov::Model> model;
    std::vector<InputSpec> inputs;     // in model parameter order
    std::vector<std::string> outputs;  // in model result order
    WeightBytes weight_bytes;
};

Translation translate(const graph::Graph& graph, const WeightStore& weights, const TranslateOptions& options);

}  // namespace xs::ov_backend
