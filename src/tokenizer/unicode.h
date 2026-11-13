#pragma once

// Unicode properties and NFC / NFD normalization for the tokenizer, from the generated tables in unicode_tables.cpp
// (aligned with what Hugging Face tokenizers uses, see the generator).

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

// Canonical decomposition (NFD) of a sequence of code points, in place.
void nfd(std::u32string& chars);

// The character classes Hugging Face's BERT normalizer and pre-tokenizer test (bert.rs), as bits.
enum BertClass : uint8_t {
    kBertControl = 1,         // is_control: unicode_categories' Other, minus tab, newline and carriage return
    kBertChinese = 2,         // is_chinese_char: the CJK blocks bert.rs lists
    kBertNonspacingMark = 4,  // removed by strip_accents (Mn); set only on characters NFD leaves alone
    kBertPunctuation = 8,     // is_bert_punc: ASCII punctuation or unicode_categories' Punctuation
};
bool is_bert(char32_t c, BertClass bert_class);

// Appends the lowercase of c (one to three code points): Rust's char::to_lowercase, which knows no context (a final
// sigma lowercases to σ).
void append_lowercase(char32_t c, std::u32string& out);

}  // namespace xs::tok::unicode
