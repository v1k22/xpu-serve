#pragma once

// UTF-8 helpers for the tokenizer. Input text is not trusted to be valid UTF-8: an invalid byte counts as a one-byte
// "character" so it still reaches byte fallback instead of being dropped.

#include <cstddef>
#include <string>
#include <string_view>

namespace xs::tok::utf8 {

// Length of the valid UTF-8 sequence starting at s[pos], or 0 when the bytes there are not one.
size_t valid_char_length(std::string_view s, size_t pos);

// Length of the character starting at s[pos]: its UTF-8 length, or 1 for an invalid byte.
inline size_t char_length(std::string_view s, size_t pos) {
    const size_t n = valid_char_length(s, pos);
    return n == 0 ? 1 : n;
}

bool is_valid(std::string_view s);

// Appends the UTF-8 encoding of c (a Unicode scalar value).
void append(std::string& out, char32_t c);

// s with each invalid sequence replaced by U+FFFD, one per maximal prefix of a valid sequence (Rust's
// String::from_utf8_lossy, the WHATWG rule).
std::string lossy(std::string_view s);

// Code point of the character at s[pos]; an invalid byte decodes to U+FFFD.
char32_t decode_at(std::string_view s, size_t pos);

// Start of the character that ends at s[end] (exclusive); end must be > 0.
size_t previous_char_start(std::string_view s, size_t end);

// Unicode White_Space, the set Rust's char::is_whitespace uses (and so Hugging Face tokenizers).
bool is_whitespace(char32_t c);

std::string_view trim_whitespace_left(std::string_view s);
std::string_view trim_whitespace_right(std::string_view s);

}  // namespace xs::tok::utf8
