#include "tokenizer/utf8.h"

#include <cstdint>

namespace xs::tok::utf8 {
namespace {
bool is_continuation(uint8_t b) { return (b & 0xC0) == 0x80; }
}  // namespace

size_t valid_char_length(std::string_view s, size_t pos) {
    const auto b0 = static_cast<uint8_t>(s[pos]);
    if (b0 < 0x80) return 1;
    size_t len = 0;
    uint8_t lo = 0x80;  // allowed range of the second byte; excludes overlongs, surrogates and > U+10FFFF
    uint8_t hi = 0xBF;
    if (b0 >= 0xC2 && b0 <= 0xDF) {
        len = 2;
    } else if (b0 >= 0xE0 && b0 <= 0xEF) {
        len = 3;
        if (b0 == 0xE0) lo = 0xA0;
        if (b0 == 0xED) hi = 0x9F;
    } else if (b0 >= 0xF0 && b0 <= 0xF4) {
        len = 4;
        if (b0 == 0xF0) lo = 0x90;
        if (b0 == 0xF4) hi = 0x8F;
    } else {
        return 0;
    }
    if (pos + len > s.size()) return 0;
    const auto b1 = static_cast<uint8_t>(s[pos + 1]);
    if (b1 < lo || b1 > hi) return 0;
    for (size_t i = 2; i < len; ++i) {
        if (!is_continuation(static_cast<uint8_t>(s[pos + i]))) return 0;
    }
    return len;
}

bool is_valid(std::string_view s) {
    for (size_t pos = 0; pos < s.size();) {
        const size_t n = valid_char_length(s, pos);
        if (n == 0) return false;
        pos += n;
    }
    return true;
}

void append(std::string& out, char32_t c) {
    if (c < 0x80) {
        out.push_back(static_cast<char>(c));
    } else if (c < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (c >> 6)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (c >> 12)));
        out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (c >> 18)));
        out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
}

std::string lossy(std::string_view s) {
    if (is_valid(s)) return std::string(s);
    std::string out;
    out.reserve(s.size() + 8);
    for (size_t pos = 0; pos < s.size();) {
        if (const size_t n = valid_char_length(s, pos); n != 0) {
            out.append(s.substr(pos, n));
            pos += n;
            continue;
        }
        // Skip the lead byte and every continuation byte that could still have been part of a valid sequence.
        const auto b0 = static_cast<uint8_t>(s[pos]);
        size_t len = 0;
        uint8_t lo = 0x80;
        uint8_t hi = 0xBF;
        if (b0 >= 0xC2 && b0 <= 0xDF) {
            len = 2;
        } else if (b0 >= 0xE0 && b0 <= 0xEF) {
            len = 3;
            if (b0 == 0xE0) lo = 0xA0;
            if (b0 == 0xED) hi = 0x9F;
        } else if (b0 >= 0xF0 && b0 <= 0xF4) {
            len = 4;
            if (b0 == 0xF0) lo = 0x90;
            if (b0 == 0xF4) hi = 0x8F;
        }
        size_t skip = 1;
        if (len > 0 && pos + 1 < s.size()) {
            const auto b1 = static_cast<uint8_t>(s[pos + 1]);
            if (b1 >= lo && b1 <= hi) {
                skip = 2;
                while (skip < len && pos + skip < s.size() && is_continuation(static_cast<uint8_t>(s[pos + skip]))) {
                    ++skip;
                }
            }
        }
        out.append("\xEF\xBF\xBD");
        pos += skip;
    }
    return out;
}

char32_t decode_at(std::string_view s, size_t pos) {
    const size_t len = valid_char_length(s, pos);
    const auto b0 = static_cast<uint8_t>(s[pos]);
    if (len == 0) return U'\uFFFD';
    if (len == 1) return b0;
    static constexpr uint8_t kLeadMask[] = {0, 0, 0x1F, 0x0F, 0x07};
    char32_t c = b0 & kLeadMask[len];
    for (size_t i = 1; i < len; ++i) c = (c << 6) | (static_cast<uint8_t>(s[pos + i]) & 0x3F);
    return c;
}

size_t previous_char_start(std::string_view s, size_t end) {
    size_t start = end - 1;
    while (start > 0 && end - start < 4 && is_continuation(static_cast<uint8_t>(s[start]))) --start;
    return valid_char_length(s, start) == end - start ? start : end - 1;
}

bool is_whitespace(char32_t c) {
    if (c >= 0x09 && c <= 0x0D) return true;
    if (c >= 0x2000 && c <= 0x200A) return true;
    switch (c) {
        case 0x20:
        case 0x85:
        case 0xA0:
        case 0x1680:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            return false;
    }
}

std::string_view trim_whitespace_left(std::string_view s) {
    size_t pos = 0;
    while (pos < s.size() && is_whitespace(decode_at(s, pos))) pos += char_length(s, pos);
    return s.substr(pos);
}

std::string_view trim_whitespace_right(std::string_view s) {
    size_t end = s.size();
    while (end > 0) {
        const size_t start = previous_char_start(s, end);
        if (!is_whitespace(decode_at(s, start))) break;
        end = start;
    }
    return s.substr(0, end);
}

}  // namespace xs::tok::utf8
