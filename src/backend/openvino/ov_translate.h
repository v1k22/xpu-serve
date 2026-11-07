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

// How GatedDelta nodes are lowered: PagedGatedDeltaNet (optimized GPU kernel; the state lives in a session-owned
// table updated in place) or GatedDeltaNet (reference kernel; the state is a ReadValue/Assign variable).
enum class GatedDeltaImpl { Paged, Ref };

struct TranslateOptions {
    DType precision = DType::F16;  // weight precision for bf16 sources
    AttentionImpl attention = AttentionImpl::Auto;
    GatedDeltaImpl gated_delta = GatedDeltaImpl::Paged;
};

// Model parameters: graph inputs, fed by the caller, and backend-owned ones the session fills itself.
enum class InputRole {
    Graph,
    StateTable,          // [1, heads, value_dim, key_dim] f32, a recurrent state the kernel updates in place
    SubsequenceBegins,   // [2] i32: 0, T
    BlockIndices,        // [2] i32: read the state from block 0, write it back to block 0
    BlockIndicesBegins,  // [2] i32: 0, 2
    ProcessedTokens,     // [1] i32: tokens before this step
    CacheInterval,       // [1] i32: 0 (no intermediate state snapshots)
};

struct InputSpec {
    std::string name;
    DType dtype = DType::F32;
    Dims shape;  // may contain graph::kT
    InputRole role = InputRole::Graph;
};

struct Translation {
    std::shared_ptr<ov::Model> model;
    std::vector<InputSpec> inputs;     // in model parameter order
    std::vector<std::string> outputs;  // in model result order
    WeightBytes weight_bytes;
    bool stateful = true;     // false for a graph without slots: runs do not accumulate tokens
    bool fixed_state = false;  // conv or recurrent state: cannot be cut back to a prefix
};

Translation translate(const graph::Graph& graph, const WeightStore& weights, const TranslateOptions& options);

}  // namespace xs::ov_backend
