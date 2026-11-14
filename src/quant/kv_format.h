#pragma once

// How a session stores its KV cache: the keys' format and the values' (SessionOptions::kv, `--kv-dtype`).

#include <cstdint>
#include <string>
#include <string_view>

namespace xs::quant {

enum class KvFormat : uint8_t {
    Float,   // the precision's float: f16, or f32 at f32 precision
    I8,      // int8 codes and one f32 scale per (token, KV head): quantize_kv_row_i8 (quant.h)
    Turbo4,  // TurboQuant: a 4, 3 or 2 bit code per value and one f16 norm per 128 values (kv_turbo.h)
    Turbo3,
    Turbo2,
};

struct KvFormats {
    KvFormat k = KvFormat::Float;
    KvFormat v = KvFormat::Float;
    bool operator==(const KvFormats&) const = default;
};

// "auto" (Float), "i8", "turbo4", "turbo3", "turbo2".
std::string_view kv_format_name(KvFormat f);
// One name for keys and values alike, or "<keys>:<values>". Throws naming what it does not know.
KvFormats parse_kv_formats(std::string_view text);
// "turbo4", or "i8:turbo4" when keys and values differ.
std::string kv_formats_name(const KvFormats& f);

// Bits per value of a TurboQuant format, 0 for the others.
constexpr int turbo_bits(KvFormat f) {
    return f == KvFormat::Turbo4 ? 4 : f == KvFormat::Turbo3 ? 3 : f == KvFormat::Turbo2 ? 2 : 0;
}

// What one row (a token's head, `head_dim` values) of a quantized format takes: bytes of codes, and bytes of its
// scales beside them (I8: one f32; TurboQuant: one f16 norm per 128 values). Not for Float.
struct KvRowBytes {
    int64_t codes = 0;
    int64_t scales = 0;
};
KvRowBytes kv_row_bytes(KvFormat f, int64_t head_dim);

// Whether rows of `head_dim` values can be stored in `f` (TurboQuant rotates groups of 128 values).
constexpr bool kv_format_fits(KvFormat f, int64_t head_dim) { return turbo_bits(f) == 0 || head_dim % 128 == 0; }

// Replaces each row of x [rows, width] by the value `f` stores for it (Float: unchanged). What a backend's cache
// holds, in the original domain: the reference every backend's attention over that cache is tested against.
void round_trip_kv(KvFormat f, float* x, int64_t rows, int64_t width);

}  // namespace xs::quant
