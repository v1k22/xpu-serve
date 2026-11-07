#pragma once

// The GPT-2 byte-to-character alphabet of byte-level BPE (bytes_to_unicode): every byte maps to one printable
// character, so BPE runs on bytes while vocab pieces stay valid UTF-8.

#include <cstdint>
#include <optional>
#include <string>

namespace xs::tok::byte_level {

char32_t byte_to_char(uint8_t byte);

// The byte a character stands for; nullopt for a character outside the alphabet.
std::optional<uint8_t> char_to_byte(char32_t c);

// The UTF-8 vocab piece of one byte.
std::string byte_piece(uint8_t byte);

}  // namespace xs::tok::byte_level
