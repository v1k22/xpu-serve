#pragma once

// Small helpers for building OpenVINO subgraphs: constants, naming, and the runtime-info marks that steer the
// plugins' precision passes.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>

#include "tensor/dtype.h"

namespace xs::ov_backend {

using OvOutput = ov::Output<ov::Node>;

ov::element::Type to_ov_type(DType t);
DType from_ov_type(const ov::element::Type& t);

OvOutput i64_constant(const std::vector<int64_t>& values);
OvOutput i64_scalar(int64_t value);
OvOutput f32_scalar(float value);

// Sets the node's friendly name to "<label>/<part>" (or just <label> when part is empty) and returns its output 0.
OvOutput named(const std::shared_ptr<ov::Node>& node, const std::string& label, const std::string& part = {});

// Keeps the node in f32 when the plugin otherwise lowers the model to f16 (DisableFP16Compression, "precise").
void keep_in_f32(const std::shared_ptr<ov::Node>& node);
// Marks a Convert(f16 constant -> f32) as weight decompression so plugins keep the constant in f16 instead of
// constant-folding it to f32.
void mark_decompression(const std::shared_ptr<ov::Node>& convert);

}  // namespace xs::ov_backend
