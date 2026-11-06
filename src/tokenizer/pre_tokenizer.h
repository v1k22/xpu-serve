#pragma once

// tokenizer.json pre-tokenizers: Split (string pattern, every split behavior) and Sequence. Others (ByteLevel,
// Metaspace, Split with a Regex) are refused by name; byte-level BPE adds them.

#include <memory>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace xs::tok {

class PreTokenizer {
public:
    virtual ~PreTokenizer() = default;
    // Splits each piece further; pieces are views into the caller's text, and empty pieces are dropped.
    virtual std::vector<std::string_view> split(const std::vector<std::string_view>& pieces) const = 0;
};

// Returns nullptr for a null pre-tokenizer; refuses unsupported types by name.
std::unique_ptr<PreTokenizer> make_pre_tokenizer(const nlohmann::json& config);

}  // namespace xs::tok
