#include "backend/openvino/ov_session.h"

#include <algorithm>
#include <cstring>

#include <openvino/core/except.hpp>

#include "util/error.h"
#include "util/half.h"

namespace xs::ov_backend {

namespace {

template <typename Fn>
void rethrow_openvino(const char* what, Fn&& fn) {
    try {
        fn();
    } catch (const ov::Exception& e) {
        fail("openvino ", what, " failed: ", e.what());
    }
}

// Keeps the first `keep` positions along axis 2 of a [1, H, L, D] state tensor.
ov::Tensor slice_state(const ov::Tensor& state, int64_t keep) {
    const ov::Shape& shape = state.get_shape();
    XS_CHECK(shape.size() == 4 && static_cast<int64_t>(shape[2]) >= keep, "kv state has shape ", shape.to_string(),
             ", cannot keep ", keep, " positions");
    ov::Shape kept = shape;
    kept[2] = static_cast<size_t>(keep);
    ov::Tensor out(state.get_element_type(), kept);
    const size_t row_bytes = shape[3] * state.get_element_type().size();
    const size_t src_head_bytes = shape[2] * row_bytes;
    const size_t dst_head_bytes = kept[2] * row_bytes;
    const auto* src = static_cast<const uint8_t*>(state.data());
    auto* dst = static_cast<uint8_t*>(out.data());
    for (size_t h = 0; h < shape[0] * shape[1]; ++h) {
        std::memcpy(dst + h * dst_head_bytes, src + h * src_head_bytes, dst_head_bytes);
    }
    return out;
}

HostTensor to_host(const ov::Tensor& t) {
    const ov::Shape& s = t.get_shape();
    const Dims shape(s.begin(), s.end());
    if (t.get_element_type() == ov::element::f16) {
        HostTensor out(DType::F32, shape);
        convert_f16_to_f32(static_cast<const uint16_t*>(t.data()), out.data<float>(), t.get_size());
        return out;
    }
    HostTensor out(from_ov_type(t.get_element_type()), shape);
    std::memcpy(out.raw(), t.data(), t.get_byte_size());
    return out;
}

}  // namespace

OvSession::OvSession(ov::CompiledModel compiled, std::vector<InputSpec> inputs, std::vector<std::string> outputs,
                     int64_t max_context)
    : compiled_(std::move(compiled)),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      max_context_(max_context) {
    rethrow_openvino("create_infer_request", [&] { request_ = compiled_.create_infer_request(); });
}

int64_t OvSession::validate_inputs(const TensorMap& inputs) const {
    for (const auto& [name, tensor] : inputs) {
        const bool known =
            std::any_of(inputs_.begin(), inputs_.end(), [&](const InputSpec& s) { return s.name == name; });
        XS_CHECK(known, "input '", name, "' is not an input of this graph");
    }
    int64_t tokens = -1;
    std::string tokens_source;
    for (const InputSpec& spec : inputs_) {
        const auto it = inputs.find(spec.name);
        XS_CHECK(it != inputs.end(), "graph input '", spec.name, "' is missing");
        const HostTensor& t = it->second;
        XS_CHECK(t.dtype() == spec.dtype, "input '", spec.name, "' has dtype ", dtype_name(t.dtype()), ", expected ",
                 dtype_name(spec.dtype));
        XS_CHECK(t.rank() == spec.shape.size(), "input '", spec.name, "' has shape ", dims_to_string(t.shape()),
                 ", expected ", graph::shape_to_string(spec.shape));
        for (size_t axis = 0; axis < spec.shape.size(); ++axis) {
            const int64_t expected = spec.shape[axis];
            const int64_t actual = t.dim(axis);
            if (expected != graph::kT) {
                XS_CHECK(actual == expected, "input '", spec.name, "' has shape ", dims_to_string(t.shape()),
                         ", expected ", graph::shape_to_string(spec.shape));
            } else if (tokens < 0) {
                tokens = actual;
                tokens_source = spec.name;
            } else {
                XS_CHECK(actual == tokens, "input '", spec.name, "' has T = ", actual, " but input '", tokens_source,
                         "' has T = ", tokens);
            }
        }
    }
    return std::max<int64_t>(tokens, 0);
}

void OvSession::bind_inputs(const TensorMap& inputs) {
    for (size_t i = 0; i < inputs_.size(); ++i) {
        const HostTensor& t = inputs.at(inputs_[i].name);
        const ov::Shape shape(t.shape().begin(), t.shape().end());
        // The host tensor is only read, and only during this synchronous infer.
        request_.set_input_tensor(i, ov::Tensor(to_ov_type(t.dtype()), shape, const_cast<void*>(t.raw())));
    }
}

void OvSession::read_outputs(TensorMap& outputs) {
    outputs.clear();
    for (size_t i = 0; i < outputs_.size(); ++i)
        outputs[outputs_[i]] = to_host(request_.get_output_tensor(i));
}

void OvSession::run(const TensorMap& inputs, TensorMap& outputs) {
    const int64_t tokens = validate_inputs(inputs);
    XS_CHECK(num_cached_ + tokens <= max_context_, "step of ", tokens, " tokens on top of ", num_cached_,
             " cached exceeds max_context ", max_context_);
    rethrow_openvino("infer", [&] {
        bind_inputs(inputs);
        request_.infer();
        read_outputs(outputs);
    });
    num_cached_ += tokens;
}

void OvSession::reset() {
    rethrow_openvino("reset_state", [&] { request_.reset_state(); });
    num_cached_ = 0;
}

void OvSession::truncate(int64_t n) {
    XS_CHECK(n >= 0 && n <= num_cached_, "truncate to ", n, " tokens, but ", num_cached_, " are cached");
    if (n == num_cached_) return;
    if (n == 0) {
        reset();
        return;
    }
    rethrow_openvino("truncate", [&] {
        for (ov::VariableState& state : request_.query_state())
            state.set_state(slice_state(state.get_state(), n));
    });
    num_cached_ = n;
}

}  // namespace xs::ov_backend
