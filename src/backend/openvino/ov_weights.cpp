#include "backend/openvino/ov_weights.h"

#include <memory>

#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>

#include "util/error.h"
#include "util/half.h"
#include "util/log.h"

namespace xs::ov_backend {

namespace {

ov::Shape to_static_shape(const Dims& dims) {
    return ov::Shape(dims.begin(), dims.end());
}

// Constant over `data`; `owner` (may be null) keeps the memory alive as long as the constant.
std::shared_ptr<ov::op::v0::Constant> shared_constant(DType dtype, const Dims& shape, const void* data,
                                                      std::shared_ptr<void> owner) {
    return std::make_shared<ov::op::v0::Constant>(to_ov_type(dtype), to_static_shape(shape), data, std::move(owner));
}

std::shared_ptr<ov::op::v0::Constant> convert_bf16(const TensorView& view, DType precision, const std::string& name) {
    const auto* src = static_cast<const uint16_t*>(view.data);
    const size_t n = static_cast<size_t>(view.numel());
    auto buffer = std::make_shared<HostTensor>(precision, view.shape);
    if (precision == DType::F16) {
        if (const size_t overflow = count_f16_overflow(src, n); overflow > 0) {
            log::warn("weight '", name, "': ", overflow, " bf16 values exceed the f16 range and become inf");
        }
        convert_bf16_to_f16(src, buffer->data<uint16_t>(), n);
    } else {
        convert_bf16_to_f32(src, buffer->data<float>(), n);
    }
    return shared_constant(precision, view.shape, buffer->raw(), buffer);
}

OvOutput to_f32(const std::shared_ptr<ov::op::v0::Constant>& constant, const std::string& label) {
    if (constant->get_element_type() != ov::element::f16) return constant;
    auto convert = std::make_shared<ov::op::v0::Convert>(constant, ov::element::f32);
    mark_decompression(convert);
    return named(convert, label, "decompress");
}

}  // namespace

OvOutput make_weight(const TensorView& view, DType precision, const std::string& name, WeightBytes& bytes) {
    XS_CHECK(precision == DType::F16 || precision == DType::F32, "weight precision must be f16 or f32, got ",
             dtype_name(precision));
    std::shared_ptr<ov::op::v0::Constant> constant;
    if (view.dtype == DType::BF16) {
        constant = convert_bf16(view, precision, name);
        bytes.converted += constant->get_byte_size();
    } else if (view.dtype == DType::Bool) {
        fail("weight '", name, "' has dtype bool, which no op accepts");
    } else {
        constant = shared_constant(view.dtype, view.shape, view.data, nullptr);
        bytes.shared += view.nbytes();
    }
    constant->set_friendly_name(name);
    return to_f32(constant, name);
}

OvOutput make_constant(const HostTensor& value, const std::string& label) {
    std::shared_ptr<ov::op::v0::Constant> constant;
    if (value.dtype() == DType::BF16) {
        const HostTensor f32 = HostTensor::from_vector(DType::F32, value.shape(), value.to_f32());
        constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, to_static_shape(f32.shape()), f32.raw());
    } else {
        constant = std::make_shared<ov::op::v0::Constant>(to_ov_type(value.dtype()), to_static_shape(value.shape()),
                                                          value.raw());
    }
    constant->set_friendly_name(label);
    return to_f32(constant, label);
}

}  // namespace xs::ov_backend
