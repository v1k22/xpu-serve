#pragma once

// GGUF files written by tests: little-endian, version 3 unless asked otherwise, metadata of every value type, tensors
// placed at the file's alignment. And the two block formats a test can encode by hand, Q8_0 and Q4_0, the way ggml's
// reference quantizers write them, with the values ggml's dequantization reads back.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "io/gguf.h"
#include "util/half.h"

namespace xs::test {

// ggml type codes the tests write.
inline constexpr uint32_t kGgmlF32 = 0, kGgmlF16 = 1, kGgmlQ4_0 = 2, kGgmlQ8_0 = 8, kGgmlQ4_K = 12, kGgmlBF16 = 30,
                          kGgmlTQ1_0 = 34;

template <typename T>
std::string pod_bytes(const T& v) {
    std::string s(sizeof(T), '\0');
    std::memcpy(s.data(), &v, sizeof(T));
    return s;
}

template <typename T>
std::string pod_bytes(const std::vector<T>& v) {
    std::string s(v.size() * sizeof(T), '\0');
    if (!v.empty()) std::memcpy(s.data(), v.data(), s.size());
    return s;
}

inline std::string gguf_string(std::string_view text) {
    return pod_bytes(static_cast<uint64_t>(text.size())) + std::string(text);
}

class GgufWriter {
public:
    using T = gguf::ValueType;

    // A value of `type` whose bytes (after the type) are `payload`.
    GgufWriter& raw(std::string key, T type, std::string payload) {
        kvs_.push_back({std::move(key), static_cast<uint32_t>(type), std::move(payload)});
        return *this;
    }
    GgufWriter& u8(std::string key, uint8_t v) { return raw(std::move(key), T::U8, pod_bytes(v)); }
    GgufWriter& i8(std::string key, int8_t v) { return raw(std::move(key), T::I8, pod_bytes(v)); }
    GgufWriter& u16(std::string key, uint16_t v) { return raw(std::move(key), T::U16, pod_bytes(v)); }
    GgufWriter& i16(std::string key, int16_t v) { return raw(std::move(key), T::I16, pod_bytes(v)); }
    GgufWriter& u32(std::string key, uint32_t v) { return raw(std::move(key), T::U32, pod_bytes(v)); }
    GgufWriter& i32(std::string key, int32_t v) { return raw(std::move(key), T::I32, pod_bytes(v)); }
    GgufWriter& u64(std::string key, uint64_t v) { return raw(std::move(key), T::U64, pod_bytes(v)); }
    GgufWriter& i64(std::string key, int64_t v) { return raw(std::move(key), T::I64, pod_bytes(v)); }
    GgufWriter& f32(std::string key, float v) { return raw(std::move(key), T::F32, pod_bytes(v)); }
    GgufWriter& f64(std::string key, double v) { return raw(std::move(key), T::F64, pod_bytes(v)); }
    GgufWriter& boolean(std::string key, bool v) { return raw(std::move(key), T::Bool, pod_bytes<uint8_t>(v)); }
    GgufWriter& string(std::string key, std::string_view v) { return raw(std::move(key), T::String, gguf_string(v)); }

    // An array of `count` elements of `element` whose bytes are `elements`.
    GgufWriter& array(std::string key, T element, uint64_t count, const std::string& elements) {
        return raw(std::move(key), T::Array, pod_bytes(static_cast<uint32_t>(element)) + pod_bytes(count) + elements);
    }
    GgufWriter& strings(std::string key, const std::vector<std::string>& v) {
        std::string bytes;
        for (const std::string& s : v)
            bytes += gguf_string(s);
        return array(std::move(key), T::String, v.size(), bytes);
    }
    GgufWriter& i32s(std::string key, const std::vector<int32_t>& v) {
        return array(std::move(key), T::I32, v.size(), pod_bytes(v));
    }
    GgufWriter& f32s(std::string key, const std::vector<float>& v) {
        return array(std::move(key), T::F32, v.size(), pod_bytes(v));
    }
    GgufWriter& bools(std::string key, const std::vector<bool>& v) {
        std::string bytes;
        for (const bool b : v)
            bytes += static_cast<char>(b ? 1 : 0);
        return array(std::move(key), T::Bool, v.size(), bytes);
    }

    // general.alignment, which also places the tensors.
    GgufWriter& alignment(uint32_t a) {
        alignment_ = a;
        return u32("general.alignment", a);
    }

    // A tensor: dims as the file lists them (columns first), a ggml type code, its bytes.
    GgufWriter& tensor(std::string name, std::vector<int64_t> dims, uint32_t type, std::string bytes) {
        tensors_.push_back({std::move(name), std::move(dims), type, std::move(bytes)});
        return *this;
    }
    GgufWriter& f32_tensor(std::string name, std::vector<int64_t> dims, const std::vector<float>& values) {
        return tensor(std::move(name), std::move(dims), kGgmlF32, pod_bytes(values));
    }

    std::string bytes(uint32_t version = 3) const {
        std::string out = "GGUF" + pod_bytes(version) + pod_bytes(static_cast<uint64_t>(tensors_.size())) +
                          pod_bytes(static_cast<uint64_t>(kvs_.size()));
        for (const Kv& kv : kvs_)
            out += gguf_string(kv.key) + pod_bytes(kv.type) + kv.payload;
        uint64_t offset = 0;
        std::vector<uint64_t> offsets;
        for (const Tensor& t : tensors_) {
            out += gguf_string(t.name) + pod_bytes(static_cast<uint32_t>(t.dims.size()));
            for (const int64_t d : t.dims)
                out += pod_bytes(static_cast<uint64_t>(d));
            out += pod_bytes(t.type) + pod_bytes(offset);
            offsets.push_back(offset);
            offset = padded(offset + t.bytes.size());
        }
        out.resize(padded(out.size()), '\0');
        const size_t data = out.size();
        for (size_t i = 0; i < tensors_.size(); ++i) {
            out.resize(data + offsets[i], '\0');
            out += tensors_[i].bytes;
        }
        return out;
    }

    void write(const std::filesystem::path& path, uint32_t version = 3) const {
        const std::string b = bytes(version);
        std::ofstream(path, std::ios::binary).write(b.data(), static_cast<std::streamsize>(b.size()));
    }

private:
    uint64_t padded(uint64_t n) const { return (n + alignment_ - 1) / alignment_ * alignment_; }

    struct Kv {
        std::string key;
        uint32_t type;
        std::string payload;
    };
    struct Tensor {
        std::string name;
        std::vector<int64_t> dims;
        uint32_t type;
        std::string bytes;
    };
    std::vector<Kv> kvs_;
    std::vector<Tensor> tensors_;
    uint32_t alignment_ = 32;
};

// ggml's quantize_row_q8_0_ref: per 32 values an f16 scale d = max |x| / 127, then round(x / d) as int8. `decoded`
// gets what ggml's dequantize_row_q8_0 reads back: q * d with d as stored.
inline std::string encode_q8_0(const float* x, int64_t n, float* decoded) {
    std::string out;
    for (int64_t b = 0; b < n / 32; ++b) {
        const float* v = x + b * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j)
            amax = std::max(amax, std::fabs(v[j]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = f32_to_f16(d);
        out += pod_bytes(dh);
        for (int j = 0; j < 32; ++j) {
            const auto q = static_cast<int8_t>(std::round(v[j] * id));
            out += static_cast<char>(q);
            decoded[b * 32 + j] = static_cast<float>(q) * f16_to_f32(dh);
        }
    }
    return out;
}

// ggml's quantize_row_q4_0_ref: per 32 values d = (the value of largest magnitude) / -8 as f16, codes
// min(15, (int8)(x / d + 8.5)), the low nibbles of the 16 bytes holding values 0..15 and the high ones 16..31.
// `decoded` gets (q - 8) * d, as ggml's dequantize_row_q4_0 reads it.
inline std::string encode_q4_0(const float* x, int64_t n, float* decoded) {
    std::string out;
    for (int64_t b = 0; b < n / 32; ++b) {
        const float* v = x + b * 32;
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < 32; ++j) {
            if (amax < std::fabs(v[j])) {
                amax = std::fabs(v[j]);
                max = v[j];
            }
        }
        const float d = max / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = f32_to_f16(d);
        out += pod_bytes(dh);
        const float scale = f16_to_f32(dh);
        for (int j = 0; j < 16; ++j) {
            const auto lo = static_cast<uint8_t>(std::min<int>(15, static_cast<int8_t>(v[j] * id + 8.5f)));
            const auto hi = static_cast<uint8_t>(std::min<int>(15, static_cast<int8_t>(v[j + 16] * id + 8.5f)));
            out += static_cast<char>(lo | (hi << 4));
            decoded[b * 32 + j] = static_cast<float>(static_cast<int>(lo) - 8) * scale;
            decoded[b * 32 + j + 16] = static_cast<float>(static_cast<int>(hi) - 8) * scale;
        }
    }
    return out;
}

}  // namespace xs::test
