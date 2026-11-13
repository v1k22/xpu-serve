#pragma once

// GGUF files (llama.cpp's model format), read in place from a memory map: the header's metadata and tensor table, and
// each tensor's bytes. Versions 2 and 3, little-endian (the layout is the same; version 3 only allows big-endian
// files, which are refused). Refused by name: other versions, big-endian files, a file that is one part of a split
// model (split.count > 1), tensor types the engine cannot hold, and anything the file's size does not cover.
//
// A tensor's dimensions are listed innermost first (ggml's ne): a matrix [rows, cols] in row-major order is listed
// as {cols, rows}. TensorInfo::shape() gives the row-major order the rest of the engine uses.

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "tensor/tensor.h"
#include "util/mmap_file.h"

namespace xs::gguf {

// Metadata value types, numbered as the file stores them.
enum class ValueType : uint32_t {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
};
std::string_view value_type_name(ValueType t);  // "u32", "string", "array", ...

// ggml's name of a tensor type code ("F32", "Q4_K", "TQ1_0"); "type <n>" for a code it never had.
std::string ggml_type_name(uint32_t type);

// One metadata value, viewed where the file holds it: a scalar, a string, or an array of either (arrays nest).
// Accessors check the type and name the key in their errors.
class Value {
public:
    ValueType type() const { return type_; }
    std::string_view key() const { return key_; }

    // Scalars. as_int takes every integer type (a u64 above the int64 range is refused); as_float takes f32 and f64.
    int64_t as_int() const;
    double as_float() const;
    bool as_bool() const;
    std::string_view as_string() const;

    // Arrays.
    ValueType element_type() const;
    uint64_t size() const;
    // A string array as views into the file, in one pass (a vocabulary of a quarter million entries costs one vector
    // of views, nothing copied).
    std::vector<std::string_view> strings() const;
    std::vector<int64_t> ints() const;   // an array of any integer type
    std::vector<double> floats() const;  // an f32 or f64 array
    std::vector<bool> bools() const;
    std::vector<Value> elements() const;  // any array, element by element (the way to read nested arrays)

    // The same type (and element type) and the same bytes as `other`, whatever the two keys: one value, exactly.
    bool same_as(const Value& other) const;

    Value() = default;

private:
    friend class File;
    Value(std::string_view key, ValueType type, ValueType element_type, uint64_t count, const std::byte* data,
          size_t bytes)
        : key_(key), type_(type), element_type_(element_type), count_(count), data_(data), bytes_(bytes) {}
    [[noreturn]] void wrong_type(std::string_view wanted) const;
    void need_array() const;

    std::string_view key_;
    ValueType type_ = ValueType::U8;
    ValueType element_type_ = ValueType::U8;  // arrays
    uint64_t count_ = 0;                      // arrays
    const std::byte* data_ = nullptr;         // a scalar's bytes, a string's characters, an array's first element
    size_t bytes_ = 0;                        // of all that
};

struct TensorInfo {
    std::string name;
    std::vector<int64_t> dims;              // as the file lists them: dims[0] is the number of columns
    uint32_t ggml_type = 0;                 // the file's type code
    DType dtype = DType::F32;               // F32, F16, BF16, or Block
    BlockFormat block = BlockFormat::Q4_0;  // dtype Block only
    uint64_t offset = 0;                    // from the start of the data section
    size_t nbytes = 0;
    const std::byte* data = nullptr;  // into the map

    int64_t numel() const;
    int64_t cols() const { return dims.front(); }
    int64_t rows() const { return numel() / cols(); }
    Dims shape() const;  // row-major, outermost first: dims reversed
    std::string type_name() const { return ggml_type_name(ggml_type); }
    // The tensor as a view of `shape` (which must hold numel() values, and for a block format end in cols()).
    TensorView view(Dims shape) const;
};

class File {
public:
    explicit File(const std::filesystem::path& path);
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    const std::filesystem::path& path() const { return map_.path(); }
    uint32_t version() const { return version_; }
    uint32_t alignment() const { return alignment_; }
    uint64_t data_offset() const { return data_offset_; }
    uint64_t size() const { return map_.size(); }

    // Metadata in file order.
    const std::vector<Value>& metadata() const { return values_; }
    const Value* find(std::string_view key) const;
    const Value& get(std::string_view key) const;  // a named error when absent
    int64_t get_int(std::string_view key) const { return get(key).as_int(); }
    double get_float(std::string_view key) const { return get(key).as_float(); }
    std::string_view get_string(std::string_view key) const { return get(key).as_string(); }

    // Tensors in file order.
    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* find_tensor(std::string_view name) const;

private:
    MappedFile map_;
    uint32_t version_ = 0;
    uint32_t alignment_ = 32;
    uint64_t data_offset_ = 0;
    std::vector<Value> values_;
    std::map<std::string_view, size_t, std::less<>> value_index_;
    std::vector<TensorInfo> tensors_;
    std::map<std::string, size_t, std::less<>> tensor_index_;
};

// The file's first four bytes are GGUF's magic.
bool has_gguf_magic(const std::filesystem::path& path);

}  // namespace xs::gguf
