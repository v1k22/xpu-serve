#include "io/gguf.h"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>

#include "util/error.h"

namespace xs::gguf {
namespace {

constexpr std::array<char, 4> kMagic = {'G', 'G', 'U', 'F'};
constexpr uint32_t kMaxDims = 4;  // ggml's GGML_MAX_DIMS
constexpr int kMaxNesting = 8;    // arrays of arrays of ...

// Bytes of one value of a fixed-size type; 0 for strings and arrays.
size_t fixed_size(ValueType t) {
    switch (t) {
        case ValueType::U8:
        case ValueType::I8:
        case ValueType::Bool:
            return 1;
        case ValueType::U16:
        case ValueType::I16:
            return 2;
        case ValueType::U32:
        case ValueType::I32:
        case ValueType::F32:
            return 4;
        case ValueType::U64:
        case ValueType::I64:
        case ValueType::F64:
            return 8;
        case ValueType::String:
        case ValueType::Array:
            return 0;
    }
    return 0;
}

bool is_integer(ValueType t) {
    switch (t) {
        case ValueType::U8:
        case ValueType::I8:
        case ValueType::U16:
        case ValueType::I16:
        case ValueType::U32:
        case ValueType::I32:
        case ValueType::U64:
        case ValueType::I64:
            return true;
        default:
            return false;
    }
}

template <typename T>
T load(const std::byte* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));  // little-endian, like every host this builds for
    return v;
}

// An integer of type `t` at `p`; nullopt for a u64 beyond the int64 range.
std::optional<int64_t> load_int(ValueType t, const std::byte* p) {
    switch (t) {
        case ValueType::U8:
            return load<uint8_t>(p);
        case ValueType::I8:
            return load<int8_t>(p);
        case ValueType::U16:
            return load<uint16_t>(p);
        case ValueType::I16:
            return load<int16_t>(p);
        case ValueType::U32:
            return load<uint32_t>(p);
        case ValueType::I32:
            return load<int32_t>(p);
        case ValueType::I64:
            return load<int64_t>(p);
        case ValueType::U64: {
            const auto v = load<uint64_t>(p);
            if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return std::nullopt;
            return static_cast<int64_t>(v);
        }
        default:
            return std::nullopt;
    }
}

// Bounds-checked reading of the header: every read names what it reads, and a file too short for it is refused as
// truncated.
class Cursor {
public:
    Cursor(const std::byte* begin, size_t size, const std::filesystem::path& path)
        : begin_(begin), size_(size), path_(path) {}

    size_t pos() const { return pos_; }
    const std::byte* here() const { return begin_ + pos_; }
    size_t left() const { return size_ - pos_; }

    void need(uint64_t n, std::string_view what) const {
        if (n > left()) {
            fail("GGUF '", path_.string(), "' is truncated: ", what, " at byte ", pos_, " needs ", n,
                 " bytes, the file has ", left(), " left");
        }
    }
    void skip(uint64_t n, std::string_view what) {
        need(n, what);
        pos_ += static_cast<size_t>(n);
    }
    template <typename T>
    T read(std::string_view what) {
        need(sizeof(T), what);
        const T v = load<T>(here());
        pos_ += sizeof(T);
        return v;
    }
    // `count` entries of at least `smallest` bytes each must fit in what is left.
    void need_entries(uint64_t count, size_t smallest, std::string_view what) const {
        if (count > left() / smallest) {
            fail("GGUF '", path_.string(), "' is truncated: ", what, " at byte ", pos_, " declares ", count,
                 " entries, more than the ", left(), " bytes left can hold");
        }
    }
    std::string_view string(std::string_view what) {
        const auto n = read<uint64_t>(what);
        need(n, what);
        const std::string_view s(reinterpret_cast<const char*>(here()), static_cast<size_t>(n));
        pos_ += static_cast<size_t>(n);
        return s;
    }
    const std::filesystem::path& path() const { return path_; }

private:
    const std::byte* begin_;
    size_t size_;
    size_t pos_ = 0;
    const std::filesystem::path& path_;
};

ValueType read_type(Cursor& c, std::string_view key) {
    const auto t = c.read<uint32_t>("a value type");
    if (t > static_cast<uint32_t>(ValueType::F64)) {
        fail("GGUF '", c.path().string(), "': metadata '", key, "' has unknown value type ", t);
    }
    return static_cast<ValueType>(t);
}

struct Parsed {
    ValueType type;
    ValueType element = ValueType::U8;
    uint64_t count = 0;
    const std::byte* data = nullptr;
    size_t bytes = 0;
};

// The value of `type` at the cursor, which moves past it. An array's elements are all walked, so that whatever
// accessor reads them later stays inside the file.
Parsed parse_value(Cursor& c, ValueType type, std::string_view key, int depth) {
    Parsed p{type};
    const std::string what = str_cat("the value of '", key, "'");
    if (const size_t size = fixed_size(type)) {
        p.data = c.here();
        p.bytes = size;
        c.skip(size, what);
        return p;
    }
    if (type == ValueType::String) {
        const std::string_view s = c.string(what);
        p.data = reinterpret_cast<const std::byte*>(s.data());
        p.bytes = s.size();
        return p;
    }
    if (depth >= kMaxNesting) fail("GGUF '", c.path().string(), "': metadata '", key, "' nests arrays too deeply");
    p.element = read_type(c, key);
    p.count = c.read<uint64_t>(what);
    p.data = c.here();
    const size_t start = c.pos();
    if (const size_t size = fixed_size(p.element)) {
        c.need_entries(p.count, size, what);
        c.skip(p.count * size, what);
    } else {
        // A string is at least its 8-byte length, an array its type and count.
        c.need_entries(p.count, p.element == ValueType::String ? 8 : 12, what);
        for (uint64_t i = 0; i < p.count; ++i) {
            if (p.element == ValueType::String) {
                c.string(what);
            } else {
                parse_value(c, ValueType::Array, key, depth + 1);
            }
        }
    }
    p.bytes = c.pos() - start;
    return p;
}

// What the engine holds a tensor of a ggml type as: F32, F16, BF16, or a block format kept byte for byte.
struct Storage {
    DType dtype;
    BlockFormat block = BlockFormat::Q4_0;
};
std::optional<Storage> storage_of(uint32_t type) {
    switch (type) {
        case 0:
            return Storage{DType::F32};
        case 1:
            return Storage{DType::F16};
        case 30:
            return Storage{DType::BF16};
        default:
            break;
    }
    if (const auto block = block_format_from_name(ggml_type_name(type))) return Storage{DType::Block, *block};
    return std::nullopt;
}

std::string supported_types() {
    std::string list = "F32, F16, BF16";
    for (int i = 0; i < kBlockFormats; ++i)
        list += str_cat(", ", block_format_name(static_cast<BlockFormat>(i)));
    return list;
}

}  // namespace

std::string_view value_type_name(ValueType t) {
    switch (t) {
        case ValueType::U8:
            return "u8";
        case ValueType::I8:
            return "i8";
        case ValueType::U16:
            return "u16";
        case ValueType::I16:
            return "i16";
        case ValueType::U32:
            return "u32";
        case ValueType::I32:
            return "i32";
        case ValueType::F32:
            return "f32";
        case ValueType::Bool:
            return "bool";
        case ValueType::String:
            return "string";
        case ValueType::Array:
            return "array";
        case ValueType::U64:
            return "u64";
        case ValueType::I64:
            return "i64";
        case ValueType::F64:
            return "f64";
    }
    return "?";
}

std::string ggml_type_name(uint32_t type) {
    // ggml.h's enum ggml_type; the codes it no longer reads keep their names, so a refusal can say what a file holds.
    static constexpr std::array<const char*, 44> kNames = {
        "F32",        "F16",        "Q4_0",       "Q4_1",  "Q4_2",     "Q4_3",     "Q5_0",     "Q5_1",    "Q8_0",
        "Q8_1",       "Q2_K",       "Q3_K",       "Q4_K",  "Q5_K",     "Q6_K",     "Q8_K",     "IQ2_XXS", "IQ2_XS",
        "IQ3_XXS",    "IQ1_S",      "IQ4_NL",     "IQ3_S", "IQ2_S",    "IQ4_XS",   "I8",       "I16",     "I32",
        "I64",        "F64",        "IQ1_M",      "BF16",  "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8", "TQ1_0",   "TQ2_0",
        "IQ4_NL_4_4", "IQ4_NL_4_8", "IQ4_NL_8_8", "MXFP4", "NVFP4",    "Q1_0",     "Q2_0",     "F8_E4M3",
    };
    if (type < kNames.size()) return kNames[type];
    return str_cat("type ", type);
}

void Value::wrong_type(std::string_view wanted) const {
    std::string got(value_type_name(type_));
    if (type_ == ValueType::Array) got = str_cat("array of ", value_type_name(element_type_));
    fail("GGUF metadata '", key_, "' is ", got, ", not ", wanted);
}

void Value::need_array() const {
    if (type_ != ValueType::Array) wrong_type("an array");
}

int64_t Value::as_int() const {
    if (!is_integer(type_)) wrong_type("an integer");
    const auto v = load_int(type_, data_);
    if (!v) fail("GGUF metadata '", key_, "' is ", load<uint64_t>(data_), ", beyond the int64 range");
    return *v;
}

double Value::as_float() const {
    if (type_ == ValueType::F32) return load<float>(data_);
    if (type_ == ValueType::F64) return load<double>(data_);
    wrong_type("a float");
}

bool Value::as_bool() const {
    if (type_ != ValueType::Bool) wrong_type("a bool");
    return load<uint8_t>(data_) != 0;
}

std::string_view Value::as_string() const {
    if (type_ != ValueType::String) wrong_type("a string");
    return {reinterpret_cast<const char*>(data_), bytes_};
}

ValueType Value::element_type() const {
    need_array();
    return element_type_;
}

uint64_t Value::size() const {
    need_array();
    return count_;
}

std::vector<std::string_view> Value::strings() const {
    need_array();
    if (element_type_ != ValueType::String) wrong_type("an array of strings");
    std::vector<std::string_view> out;
    out.reserve(static_cast<size_t>(count_));
    const std::byte* p = data_;
    for (uint64_t i = 0; i < count_; ++i) {
        const auto n = static_cast<size_t>(load<uint64_t>(p));
        out.emplace_back(reinterpret_cast<const char*>(p + 8), n);
        p += 8 + n;
    }
    return out;
}

std::vector<int64_t> Value::ints() const {
    need_array();
    if (!is_integer(element_type_)) wrong_type("an array of integers");
    const size_t size = fixed_size(element_type_);
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(count_));
    for (uint64_t i = 0; i < count_; ++i) {
        const auto v = load_int(element_type_, data_ + i * size);
        if (!v) fail("GGUF metadata '", key_, "' element ", i, " is beyond the int64 range");
        out.push_back(*v);
    }
    return out;
}

std::vector<double> Value::floats() const {
    need_array();
    if (element_type_ != ValueType::F32 && element_type_ != ValueType::F64) wrong_type("an array of floats");
    std::vector<double> out;
    out.reserve(static_cast<size_t>(count_));
    for (uint64_t i = 0; i < count_; ++i) {
        out.push_back(element_type_ == ValueType::F32 ? load<float>(data_ + i * 4) : load<double>(data_ + i * 8));
    }
    return out;
}

std::vector<bool> Value::bools() const {
    need_array();
    if (element_type_ != ValueType::Bool) wrong_type("an array of bools");
    std::vector<bool> out;
    out.reserve(static_cast<size_t>(count_));
    for (uint64_t i = 0; i < count_; ++i)
        out.push_back(load<uint8_t>(data_ + i) != 0);
    return out;
}

std::vector<Value> Value::elements() const {
    need_array();
    std::vector<Value> out;
    out.reserve(static_cast<size_t>(count_));
    const std::filesystem::path none;
    Cursor c(data_, bytes_, none);  // parse_value walked these bytes already: they are in the file
    for (uint64_t i = 0; i < count_; ++i) {
        const Parsed p = parse_value(c, element_type_, key_, 1);
        out.push_back(Value(key_, p.type, p.element, p.count, p.data, p.bytes));
    }
    return out;
}

int64_t TensorInfo::numel() const {
    int64_t n = 1;
    for (const int64_t d : dims)
        n *= d;
    return n;
}

Dims TensorInfo::shape() const {
    return Dims(dims.rbegin(), dims.rend());
}

TensorView TensorInfo::view(Dims shape) const {
    XS_CHECK(xs::numel(shape) == numel() && (dtype != DType::Block || shape.back() == cols()), "GGUF tensor '", name,
             "' ", dims_to_string(this->shape()), " cannot be viewed as ", dims_to_string(shape));
    TensorView v;
    v.dtype = dtype;
    v.shape = std::move(shape);
    v.data = data;
    v.block = block;
    return v;
}

File::File(const std::filesystem::path& path) : map_(path) {
    const std::string where = str_cat("GGUF '", path.string(), "'");
    Cursor c(map_.data(), map_.size(), map_.path());
    if (map_.size() < kMagic.size() || std::memcmp(map_.data(), kMagic.data(), kMagic.size()) != 0) {
        fail(path.string(), " is not a GGUF file (its first bytes are not 'GGUF')");
    }
    c.skip(kMagic.size(), "the magic");
    version_ = c.read<uint32_t>("the version");
    if (version_ != 2 && version_ != 3) {
        if ((version_ & 0xFFFF) == 0) fail(where, " is big-endian; only little-endian GGUF files are read");
        fail(where, " is GGUF version ", version_, "; versions 2 and 3 are read");
    }
    const auto n_tensors = c.read<uint64_t>("the tensor count");
    const auto n_values = c.read<uint64_t>("the metadata count");

    // A key, its type and value take at least 13 bytes; a tensor's name, dimensions, type and offset 32.
    c.need_entries(n_values, 13, "the metadata");
    values_.reserve(static_cast<size_t>(n_values));
    for (uint64_t i = 0; i < n_values; ++i) {
        const std::string_view key = c.string("a metadata key");
        const ValueType type = read_type(c, key);
        const Parsed p = parse_value(c, type, key, 0);
        if (!value_index_.emplace(key, values_.size()).second) fail(where, ": metadata '", key, "' appears twice");
        values_.push_back(Value(key, p.type, p.element, p.count, p.data, p.bytes));
    }
    if (const Value* split = find("split.count"); split != nullptr && split->as_int() > 1) {
        fail(where, " is one part of a model split into ", split->as_int(),
             " files (split.count); merge the parts into one file first");
    }
    if (const Value* a = find("general.alignment")) {
        if (a->type() != ValueType::U32) a->wrong_type("a u32");
        alignment_ = static_cast<uint32_t>(a->as_int());
        if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) {
            fail(where, ": general.alignment ", alignment_, " is not a power of two");
        }
    }

    c.need_entries(n_tensors, 32, "the tensor table");
    tensors_.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo t;
        t.name = std::string(c.string("a tensor name"));
        const auto n_dims = c.read<uint32_t>("a tensor's dimension count");
        if (n_dims == 0 || n_dims > kMaxDims) {
            fail(where, ": tensor '", t.name, "' has ", n_dims, " dimensions (1 to ", kMaxDims, " are read)");
        }
        int64_t numel = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            const auto n = c.read<uint64_t>("a tensor dimension");
            if (n == 0 || n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                static_cast<int64_t>(n) > std::numeric_limits<int64_t>::max() / numel) {
                fail(where, ": tensor '", t.name, "' has dimension ", n, ", which this reader refuses");
            }
            t.dims.push_back(static_cast<int64_t>(n));
            numel *= static_cast<int64_t>(n);
        }
        t.ggml_type = c.read<uint32_t>("a tensor type");
        t.offset = c.read<uint64_t>("a tensor offset");
        const auto storage = storage_of(t.ggml_type);
        if (!storage) {
            fail(where, ": tensor '", t.name, "' has type ", ggml_type_name(t.ggml_type),
                 ", which xpu-serve does not read (it reads ", supported_types(), ")");
        }
        t.dtype = storage->dtype;
        t.block = storage->block;
        if (t.dtype == DType::Block) {
            const BlockLayout l = block_layout(t.block);
            if (t.cols() % l.weights != 0) {
                fail(where, ": tensor '", t.name, "' (", t.type_name(), ") has rows of ", t.cols(),
                     " columns, not a whole number of blocks of ", l.weights);
            }
            t.nbytes = block_storage_bytes(t.block, numel);
        } else {
            if (static_cast<uint64_t>(numel) > map_.size()) {
                fail(where, " is truncated: tensor '", t.name, "' holds ", numel, " values, more than the file's ",
                     map_.size(), " bytes");
            }
            t.nbytes = storage_bytes(t.dtype, numel);
        }
        if (!tensor_index_.emplace(t.name, tensors_.size()).second) {
            fail(where, ": tensor '", t.name, "' appears twice");
        }
        tensors_.push_back(std::move(t));
    }

    data_offset_ = (static_cast<uint64_t>(c.pos()) + alignment_ - 1) / alignment_ * alignment_;
    const uint64_t data_size = map_.size() > data_offset_ ? map_.size() - data_offset_ : 0;
    for (TensorInfo& t : tensors_) {
        if (t.offset % alignment_ != 0) {
            fail(where, ": tensor '", t.name, "' starts at offset ", t.offset, ", not a multiple of the alignment ",
                 alignment_);
        }
        if (t.offset > data_size || t.nbytes > data_size - t.offset) {
            fail(where, " is truncated: tensor '", t.name, "' needs bytes ", data_offset_ + t.offset, " to ",
                 data_offset_ + t.offset + t.nbytes, ", the file has ", map_.size());
        }
        t.data = map_.data() + data_offset_ + t.offset;
    }
}

const Value* File::find(std::string_view key) const {
    const auto it = value_index_.find(key);
    return it == value_index_.end() ? nullptr : &values_[it->second];
}

const Value& File::get(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) fail("GGUF '", path().string(), "' has no metadata '", key, "'");
    return *v;
}

const TensorInfo* File::find_tensor(std::string_view name) const {
    const auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

bool has_gguf_magic(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::array<char, 4> head{};
    return in.read(head.data(), head.size()) && head == kMagic;
}

}  // namespace xs::gguf
