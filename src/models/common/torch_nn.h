#pragma once

// torch.nn modules as graph ops, over a checkpoint's weights under the names torch's state_dict gives them (the stem
// of a module, e.g. "encoder.layer.0.output.dense"): what the encoders and the heads written in torch share.

#include <string>

#include "graph/graph.h"
#include "io/safetensors.h"

namespace xs::torch_nn {

using graph::Graph;
using graph::Value;

// nn.Linear under `stem`: x @ <stem>.weight^T + <stem>.bias. A Linear built with bias=False is a plain Graph::matmul.
Value linear(Graph& g, const WeightStore& w, Value x, const std::string& stem);

// nn.LayerNorm under `stem` over the last dim: <stem>.weight and <stem>.bias.
Value layer_norm(Graph& g, const WeightStore& w, Value x, const std::string& stem, float eps);

// nn.LayerNorm(bias=False) under `stem`: <stem>.weight only (the graph's LayerNorm with a zero bias).
Value layer_norm_without_bias(Graph& g, const WeightStore& w, Value x, const std::string& stem, float eps);

}  // namespace xs::torch_nn
