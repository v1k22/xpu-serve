#include "backend/openvino/ov_builders.h"

#include <openvino/core/runtime_attribute.hpp>
#include <openvino/op/constant.hpp>

#include "util/error.h"

namespace xs::ov_backend {

namespace {

// The plugins look these marks up by their RTTI key ("precise_0", "decompression_0"); the classes that own them live
// in OpenVINO's developer API, which the release package does not ship, so we declare look-alikes with the same keys.
class KeepInF32Mark : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("precise", "0", ov::RuntimeAttribute);
    bool is_copyable() const override { return false; }
};

class DecompressionMark : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("decompression", "0", ov::RuntimeAttribute);
    bool is_copyable() const override { return false; }
};

}  // namespace

ov::element::Type to_ov_type(DType t) {
    switch (t) {
        case DType::F32:
            return ov::element::f32;
        case DType::F16:
            return ov::element::f16;
        case DType::BF16:
            return ov::element::bf16;
        case DType::I64:
            return ov::element::i64;
        case DType::I32:
            return ov::element::i32;
        case DType::I8:
            return ov::element::i8;
        case DType::U8:
            return ov::element::u8;
        case DType::Bool:
            return ov::element::boolean;
    }
    fail("unknown dtype ", static_cast<int>(t));
}

DType from_ov_type(const ov::element::Type& t) {
    if (t == ov::element::f32) return DType::F32;
    if (t == ov::element::f16) return DType::F16;
    if (t == ov::element::bf16) return DType::BF16;
    if (t == ov::element::i64) return DType::I64;
    if (t == ov::element::i32) return DType::I32;
    if (t == ov::element::i8) return DType::I8;
    if (t == ov::element::u8) return DType::U8;
    if (t == ov::element::boolean) return DType::Bool;
    fail("OpenVINO element type ", t.get_type_name(), " has no xs::DType");
}

OvOutput i64_constant(const std::vector<int64_t>& values) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{values.size()}, values);
}

OvOutput i64_scalar(int64_t value) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{value});
}

OvOutput f32_scalar(float value) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{value});
}

OvOutput named(const std::shared_ptr<ov::Node>& node, const std::string& label, const std::string& part) {
    node->set_friendly_name(part.empty() ? label : label + "/" + part);
    return node->output(0);
}

void keep_in_f32(const std::shared_ptr<ov::Node>& node) {
    node->get_rt_info()[KeepInF32Mark::get_type_info_static()] = KeepInF32Mark{};
}

void mark_decompression(const std::shared_ptr<ov::Node>& convert) {
    convert->get_rt_info()[DecompressionMark::get_type_info_static()] = DecompressionMark{};
}

}  // namespace xs::ov_backend
