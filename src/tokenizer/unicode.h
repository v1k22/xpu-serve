#pragma once

// Unicode properties and NFC normalization for the tokenizer, from the generated tables in unicode_tables.cpp (aligned
// with what Hugging Face tokenizers uses, see the generator).

#include <cstdint>
#include <string>
#include <string_view>

namespace xs::tok::unicode {

enum class Category : uint8_t { kOther, kLetter, kMark, kNumber };

// General category class as the regex \p{L} / \p{M} / \p{N} sees it.
Category category(char32_t c);

uint8_t combining_class(char32_t c);

// Canonical composition (NFC). Bytes that are not valid UTF-8 pass through unchanged.
std::string nfc(std::string_view text);

}  // namespace xs::tok::unicode
