#include "tensor/tensor.h"

#include <cstdlib>
#include <cstring>

#include "util/error.h"
#include "util/half.h"

namespace xs {

int64_t numel(const Dims& dims) {
    int64_t n = 1;
    for (int64_t d : dims) {
        XS_CHECK(d >= 0, "numel of a shape with a symbolic or negative dim: ", dims_to_string(dims));
        n *= d;
    }
    return n;
}

std::string dims_to_string(const Dims& dims) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(dims[i]);
    }
    return s + "]";
}

HostTensor::HostTensor(DType dtype, Dims shape) : dtype_(dtype), shape_(std::move(shape)) {
    const size_t bytes = std::max<size_t>(nbytes(), 1);
    void* p = std::calloc(1, bytes);
    if (p == nullptr) fail("out of host memory allocating ", bytes, " bytes for a ", dims_to_string(shape_), " tensor");
    storage_ = std::shared_ptr<void>(p, std::free);
    data_ = p;
}

HostTensor HostTensor::borrow(DType dtype, Dims shape, void* data) {
    HostTensor t;
    t.dtype_ = dtype;
    t.shape_ = std::move(shape);
    t.data_ = data;
    return t;
}

HostTensor HostTensor::clone() const {
    HostTensor t(dtype_, shape_);
    if (nbytes() > 0) std::memcpy(t.data_, data_, nbytes());
    return t;
}

HostTensor HostTensor::reshaped(Dims shape) const {
    XS_CHECK(xs::numel(shape) == numel(), "reshape ", dims_to_string(shape_), " -> ", dims_to_string(shape));
    HostTensor t = *this;
    t.shape_ = std::move(shape);
    return t;
}

std::vector<float> HostTensor::to_f32() const {
    const size_t n = static_cast<size_t>(numel());
    std::vector<float> out(n);
    switch (dtype_) {
        case DType::F32: std::memcpy(out.data(), data_, n * sizeof(float)); break;
        case DType::F16: convert_f16_to_f32(data<uint16_t>(), out.data(), n); break;
        case DType::BF16: convert_bf16_to_f32(data<uint16_t>(), out.data(), n); break;
        case DType::I64:
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(data<int64_t>()[i]);
            break;
        case DType::I32:
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(data<int32_t>()[i]);
            break;
        case DType::I8:
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(data<int8_t>()[i]);
            break;
        case DType::U8:
        case DType::Bool:
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(data<uint8_t>()[i]);
            break;
        case DType::Q8:
        case DType::Q4:
        case DType::Block:
            fail("a host tensor never holds quantized codes");
    }
    return out;
}

}  // namespace xs
