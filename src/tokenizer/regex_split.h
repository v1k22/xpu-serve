#pragma once

// Hand-written matchers for the regexes that byte-level BPE tokenizers use in their Split pre-tokenizer, and the GPT-2
// regex of the ByteLevel pre-tokenizer (no regex engine). Each supported pattern is a list of alternatives tried in
// order at every position, with the leftmost-first backtracking semantics of the Oniguruma engine Hugging Face
// tokenizers uses.

#include <cstddef>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace xs::tok {

// The Split patterns of the byte-level BPE tokenizers matched here, as tokenizer.json writes them.
namespace split_regex {
// Qwen2Tokenizer's (Qwen2, Qwen3, Qwen3-ASR).
inline constexpr std::string_view kQwen2 =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// Qwen3.5's: letter runs take combining marks too.
inline constexpr std::string_view kQwen35 =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+)"
    R"(|\s+(?!\S)|\s+)";
// Llama 3's (LFM3): up to three digits a number.
inline constexpr std::string_view kLlama3 =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+)"
    R"(|\s+(?!\S)|\s+)";
// o200k_base's (GPT-4o; Inkling 2): a word is uppercase letters then lowercase ones, with a contraction on it;
// punctuation takes the slashes after it.
inline constexpr std::string_view kO200k =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?)"
    R"(|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?)"
    R"(|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
}  // namespace split_regex

class RegexSplitter {
public:
    // The matcher for `pattern`, or nullptr when the pattern is not one of the supported ones.
    static std::unique_ptr<RegexSplitter> for_pattern(std::string_view pattern);
    // The ByteLevel pre-tokenizer's regex: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    static std::unique_ptr<RegexSplitter> gpt2();
    virtual ~RegexSplitter() = default;

    // Byte ranges [begin, end) of the successive non-overlapping matches in `text`, left to right.
    virtual std::vector<std::pair<size_t, size_t>> find_matches(std::string_view text) const = 0;
};

}  // namespace xs::tok
