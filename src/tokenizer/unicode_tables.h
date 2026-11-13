#pragma once

// Generated Unicode data (src/tokenizer/unicode_tables.cpp, written by tests/parity/tokenizer/gen_unicode_tables.py).
// Every table is sorted by code point.

#include <cstddef>
#include <cstdint>

namespace xs::tok::unicode::tables {

struct CodePointRange {
    char32_t first;
    char32_t last;  // inclusive
};

// value: 1 = letter (L), 2 = mark (M), 3 = number (N).
struct CategoryRange {
    char32_t first;
    char32_t last;
    uint8_t value;
};

struct CombiningRange {
    char32_t first;
    char32_t last;
    uint8_t ccc;  // canonical combining class
};

// One level of a canonical decomposition (Hangul syllables excluded: they decompose algorithmically).
struct Decomposition {
    char32_t code_point;
    char32_t first;
    char32_t second;  // 0 for a singleton decomposition
    uint8_t composes;  // 1 when first + second recompose to code_point (a primary composite)
};

extern const CategoryRange kCategoryRanges[];
extern const size_t kCategoryRangeCount;
extern const CombiningRange kCombiningRanges[];
extern const size_t kCombiningRangeCount;
extern const Decomposition kDecompositions[];
extern const size_t kDecompositionCount;
// Code points whose NFC quick check is not Yes, or whose combining class is not 0.
extern const CodePointRange kNfcUnsafeRanges[];
extern const size_t kNfcUnsafeRangeCount;

struct BertClassRange {
    char32_t first;
    char32_t last;
    uint8_t classes;  // unicode::BertClass bits
};

// Code points first, first + stride, ..., last lowercase to code point + delta (Rust's char::to_lowercase).
struct LowercaseRun {
    char32_t first;
    char32_t last;
    int32_t delta;
    uint8_t stride;
};

// A code point whose lowercase is several code points (0-padded).
struct LowercaseExpansion {
    char32_t code_point;
    char32_t lower[3];
};

extern const BertClassRange kBertClassRanges[];
extern const size_t kBertClassRangeCount;
extern const LowercaseRun kLowercaseRuns[];
extern const size_t kLowercaseRunCount;
extern const LowercaseExpansion kLowercaseExpansions[];
extern const size_t kLowercaseExpansionCount;

}  // namespace xs::tok::unicode::tables
