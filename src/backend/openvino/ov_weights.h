#pragma once

#include <cstddef>
#include <string>

#include "backend/openvino/ov_builders.h"
#include "tensor/tensor.h"

namespace xs::ov_backend {

// Host bytes behind the weight constants of one model.
struct WeightBytes {
    size_t shared = 0;     // constants that point straight into the WeightStore (no copy)
    size_t converted = 0;  // buffers allocated here for bf16 sources converted to the session precision
};

// Turns one stored weight into a value of f32 element type for the model:
//   * bf16 is converted on the host to `precision` (F16 or F32) into a buffer the constant owns;
//   * f16 and f32 (and integer) data are shared with the store without a copy, so the store must outlive the
//     session (the CPU plugin keeps pointing at it after compile);
//   * an f16 constant is followed by a Convert marked as decompression, so plugins keep it in f16.
OvOutput make_weight(const TensorView& view, DType precision, const std::string& name, WeightBytes& bytes);

// Copies a small graph constant (bf16 becomes f32).
OvOutput make_constant(const HostTensor& value, const std::string& label);

}  // namespace xs::ov_backend
