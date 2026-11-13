#include "util/base64.h"

#include <array>

namespace xs {

namespace {

constexpr int8_t kInvalid = -1;
constexpr int8_t kSkip = -2;

constexpr std::string_view kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

constexpr std::array<int8_t, 256> make_table() {
    std::array<int8_t, 256> t{};
    for (auto& v : t)
        v = kInvalid;
    for (size_t i = 0; i < kAlphabet.size(); ++i)
        t[static_cast<unsigned char>(kAlphabet[i])] = static_cast<int8_t>(i);
    for (const char c : {' ', '\t', '\r', '\n'})
        t[static_cast<unsigned char>(c)] = kSkip;
    return t;
}

constexpr std::array<int8_t, 256> kTable = make_table();

}  // namespace

std::optional<std::vector<uint8_t>> base64_decode(std::string_view text) {
    std::vector<uint8_t> out;
    out.reserve(text.size() / 4 * 3);
    uint32_t group = 0;
    int filled = 0;  // sextets in `group`
    size_t padding = 0;
    for (const char c : text) {
        if (c == '=') {
            ++padding;
            continue;
        }
        const int8_t v = kTable[static_cast<unsigned char>(c)];
        if (v == kSkip) continue;
        if (v == kInvalid || padding > 0) return std::nullopt;  // padding only at the end
        group = group << 6 | static_cast<uint32_t>(v);
        if (++filled == 4) {
            out.push_back(static_cast<uint8_t>(group >> 16));
            out.push_back(static_cast<uint8_t>(group >> 8));
            out.push_back(static_cast<uint8_t>(group));
            group = 0;
            filled = 0;
        }
    }
    if (padding > 2 || (padding > 0 && filled + static_cast<int>(padding) != 4)) return std::nullopt;
    switch (filled) {
        case 0:
            break;
        case 2:
            out.push_back(static_cast<uint8_t>(group >> 4));
            break;
        case 3:
            out.push_back(static_cast<uint8_t>(group >> 10));
            out.push_back(static_cast<uint8_t>(group >> 2));
            break;
        default:
            return std::nullopt;  // one leftover sextet cannot encode a byte
    }
    return out;
}

std::string base64_encode(std::span<const uint8_t> bytes) {
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 3 <= bytes.size(); i += 3) {
        const uint32_t group = uint32_t{bytes[i]} << 16 | uint32_t{bytes[i + 1]} << 8 | bytes[i + 2];
        for (const int shift : {18, 12, 6, 0})
            out.push_back(kAlphabet[group >> shift & 63]);
    }
    if (const size_t rest = bytes.size() - i; rest > 0) {
        const uint32_t group = uint32_t{bytes[i]} << 16 | (rest == 2 ? uint32_t{bytes[i + 1]} << 8 : 0);
        out.push_back(kAlphabet[group >> 18 & 63]);
        out.push_back(kAlphabet[group >> 12 & 63]);
        out.push_back(rest == 2 ? kAlphabet[group >> 6 & 63] : '=');
        out.push_back('=');
    }
    return out;
}

}  // namespace xs
