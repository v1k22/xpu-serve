#include "tokenizer/byte_level.h"

#include <array>

#include "tokenizer/utf8.h"

namespace xs::tok::byte_level {
namespace {
// Printable Latin-1 bytes map to themselves; the others, in byte order, to U+0100, U+0101, ...
const std::array<char32_t, 256>& alphabet() {
    static const std::array<char32_t, 256> table = [] {
        std::array<char32_t, 256> t{};
        char32_t next = 0x100;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            t[b] = printable ? static_cast<char32_t>(b) : next++;
        }
        return t;
    }();
    return table;
}
}  // namespace

char32_t byte_to_char(uint8_t byte) { return alphabet()[byte]; }

std::optional<uint8_t> char_to_byte(char32_t c) {
    if (c < 0x100) {
        if (alphabet()[c] == c) return static_cast<uint8_t>(c);
        return std::nullopt;
    }
    static const std::array<int16_t, 0x44> high = [] {
        std::array<int16_t, 0x44> t{};
        t.fill(-1);
        for (int b = 0; b < 256; ++b) {
            if (alphabet()[b] >= 0x100) t[alphabet()[b] - 0x100] = static_cast<int16_t>(b);
        }
        return t;
    }();
    if (c - 0x100 >= high.size() || high[c - 0x100] < 0) return std::nullopt;
    return static_cast<uint8_t>(high[c - 0x100]);
}

std::string byte_piece(uint8_t byte) {
    std::string out;
    utf8::append(out, byte_to_char(byte));
    return out;
}

}  // namespace xs::tok::byte_level
