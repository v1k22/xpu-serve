#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor/block_format.h"
#include "tensor/dtype.h"

namespace xs {

using Dims = std::vector<int64_t>;

int64_t numel(const Dims& dims);
std::string dims_to_string(const Dims& dims);

// Non-owning, read-only view of dense row-major data (e.g. a tensor inside a memory-mapped safetensors file).
// The scales and biases of a quantized (Q8 / Q4) weight [rows, cols]: one f16 of each per `group` columns of a row,
// [rows, cols / group] row-major. Q8: w = scale * q (q int8). Q4: w = scale * q + bias (q in [0, 15]; two per byte,
// the low nibble is the even column).
// AWQ packs add input_scales (f32 [cols]): the weight a matmul uses is dequant(w)[r, c] * input_scales[c].
struct QuantParams {
    int64_t group = 0;
    const void* scales = nullptr;        // f16
    const void* biases = nullptr;        // f16, Q4 only
    const float* input_scales = nullptr;  // f32 [cols], or null
};

// A run of rows of an FP8 weight (dtype F8): E4M3 codes [rows, cols], a byte each, and f32 scales, one per block of
// `block_rows` rows and `block_cols` columns, [ceil(rows / block_rows), ceil(cols / block_cols)] row-major:
// w = e4m3(code) * scale. A scale per row is a block of one row and all its columns.
inline constexpr int64_t kFp8Block = 128;
inline constexpr int64_t kMxfp8Block = 32;  // columns of a row that share a scale in MXFP8
struct Fp8Run {
    const uint8_t* codes = nullptr;
    const float* scales = nullptr;
    int64_t rows = 0;
    int64_t block_rows = kFp8Block;
    int64_t block_cols = kFp8Block;
};

// A run of rows of an NVFP4 weight (dtype F4): E2M1 codes [rows, cols], two a byte (the low nibble is the even
// column), an E4M3 code per kFp4Block columns of a row as their scale, [rows, cols / kFp4Block], and one factor for the
// run: w = e2m1(code) * e4m3(scale) * factor.
inline constexpr int64_t kFp4Block = 16;
struct Fp4Run {
    const uint8_t* codes = nullptr;
    const uint8_t* scales = nullptr;
    float factor = 1.0f;
    int64_t rows = 0;
};

struct TensorView {
    DType dtype = DType::F32;
    Dims shape;
    const void* data = nullptr;   // dense values or codes; null for an F8 or F4 weight
    QuantParams quant;            // dtype Q8 / Q4 only
    std::span<const Fp8Run> fp8;  // an F8 weight: the rows of its matrix [numel / cols, cols], run after run
    std::span<const Fp4Run> fp4;  // an F4 weight: likewise
    BlockFormat block = BlockFormat::Q4_0;  // dtype Block only: `data` is the matrix's rows as blocks of this format

    int64_t numel() const { return xs::numel(shape); }
    size_t nbytes() const {
        return dtype == DType::Block ? block_storage_bytes(block, numel()) : storage_bytes(dtype, numel());
    }
};

// Owning (or borrowing) dense row-major host tensor. Copies share storage; use clone() for a deep copy.
class HostTensor {
public:
    HostTensor() = default;
    // Allocates zero-initialized storage.
    HostTensor(DType dtype, Dims shape);
    // Borrows external memory; the caller keeps it alive.
    static HostTensor borrow(DType dtype, Dims shape, void* data);

    template <typename T>
    static HostTensor from_vector(DType dtype, Dims shape, const std::vector<T>& values);

    DType dtype() const { return dtype_; }
    const Dims& shape() const { return shape_; }
    int64_t dim(size_t i) const { return shape_.at(i); }
    size_t rank() const { return shape_.size(); }
    int64_t numel() const { return xs::numel(shape_); }
    size_t nbytes() const { return static_cast<size_t>(numel()) * dtype_size(dtype_); }
    bool empty() const { return data_ == nullptr; }

    void* raw() { return data_; }
    const void* raw() const { return data_; }
    template <typename T>
    T* data() { return static_cast<T*>(data_); }
    template <typename T>
    const T* data() const { return static_cast<const T*>(data_); }
    template <typename T>
    std::span<T> span() { return {data<T>(), static_cast<size_t>(numel())}; }
    template <typename T>
    std::span<const T> span() const { return {data<T>(), static_cast<size_t>(numel())}; }

    TensorView view() const { return {dtype_, shape_, data_}; }
    HostTensor clone() const;
    // Same storage, new shape with the same element count.
    HostTensor reshaped(Dims shape) const;
    // Values as f32 regardless of storage dtype (f32/f16/bf16/ints).
    std::vector<float> to_f32() const;

private:
    DType dtype_ = DType::F32;
    Dims shape_;
    std::shared_ptr<void> storage_;
    void* data_ = nullptr;
};

using TensorMap = std::map<std::string, HostTensor>;

// Rows [first, first + count) of a tensor's leading axis, copied.
HostTensor rows_of(const HostTensor& t, int64_t first, int64_t count);
// Tensors of one dtype whose shapes agree past the leading axis, their rows one after another.
HostTensor concat_rows(std::span<const HostTensor> parts);

template <typename T>
HostTensor HostTensor::from_vector(DType dtype, Dims shape, const std::vector<T>& values) {
    HostTensor t(dtype, std::move(shape));
    if (static_cast<size_t>(t.numel()) != values.size() || sizeof(T) != dtype_size(dtype)) {
        throw std::invalid_argument("HostTensor::from_vector: size or element type mismatch");
    }
    std::copy(values.begin(), values.end(), t.data<T>());
    return t;
}

}  // namespace xs
