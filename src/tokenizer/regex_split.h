#pragma once

// Hand-written matchers for the regexes that byte-level BPE tokenizers use in their Split pre-tokenizer (no regex
// engine). Each supported pattern is a list of alternatives tried in order at every position, with the leftmost-first
// backtracking semantics of the Oniguruma engine Hugging Face tokenizers uses.

#include <cstddef>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace xs::tok {

class RegexSplitter {
public:
    // The matcher for `pattern`, or nullptr when the pattern is not one of the supported ones.
    static std::unique_ptr<RegexSplitter> for_pattern(std::string_view pattern);
    virtual ~RegexSplitter() = default;

    // Byte ranges [begin, end) of the successive non-overlapping matches in `text`, left to right.
    virtual std::vector<std::pair<size_t, size_t>> find_matches(std::string_view text) const = 0;
};

}  // namespace xs::tok
