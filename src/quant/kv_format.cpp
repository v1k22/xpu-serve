#include "quant/kv_format.h"

#include "quant/kv_turbo.h"
#include "quant/quant.h"
#include "util/error.h"

namespace xs::quant {

namespace {

constexpr KvFormat kAll[] = {KvFormat::Float, KvFormat::I8, KvFormat::Turbo4, KvFormat::Turbo3, KvFormat::Turbo2};

KvFormat parse_one(std::string_view name, std::string_view whole) {
    for (const KvFormat f : kAll) {
        if (name == kv_format_name(f)) return f;
    }
    fail("KV cache format must be auto, i8, turbo4, turbo3 or turbo2, or <keys>:<values> of those, got '", whole, "'");
}

}  // namespace

std::string_view kv_format_name(KvFormat f) {
    switch (f) {
        case KvFormat::Float: return "auto";
        case KvFormat::I8: return "i8";
        case KvFormat::Turbo4: return "turbo4";
        case KvFormat::Turbo3: return "turbo3";
        case KvFormat::Turbo2: return "turbo2";
    }
    return "?";
}

KvFormats parse_kv_formats(std::string_view text) {
    const size_t colon = text.find(':');
    if (colon == std::string_view::npos) {
        const KvFormat f = parse_one(text, text);
        return {f, f};
    }
    return {parse_one(text.substr(0, colon), text), parse_one(text.substr(colon + 1), text)};
}

std::string kv_formats_name(const KvFormats& f) {
    if (f.k == f.v) return std::string(kv_format_name(f.k));
    return str_cat(kv_format_name(f.k), ":", kv_format_name(f.v));
}

KvRowBytes kv_row_bytes(KvFormat f, int64_t head_dim) {
    if (f == KvFormat::I8) return {head_dim, static_cast<int64_t>(sizeof(float))};
    const int bits = turbo_bits(f);
    XS_CHECK(bits != 0, "kv_row_bytes: the float format's size is the backend's");
    XS_CHECK(kv_format_fits(f, head_dim), "KV format ", kv_format_name(f), " needs a head_dim that is a multiple of ",
             turbo::kGroup, ", got ", head_dim);
    const int64_t groups = head_dim / turbo::kGroup;
    return {groups * turbo::group_bytes(bits), groups * static_cast<int64_t>(sizeof(uint16_t))};
}

void round_trip_kv(KvFormat f, float* x, int64_t rows, int64_t width) {
    if (f == KvFormat::Float) return;
    if (f == KvFormat::I8) return round_trip_kv_i8(x, rows, width);
    turbo::round_trip(x, rows, width, turbo_bits(f));
}

}  // namespace xs::quant
