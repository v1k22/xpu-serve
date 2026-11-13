#pragma once

// tokenizer.json pre-tokenizers: Split (a string pattern, or one of the regexes in regex_split.h; every split
// behavior), ByteLevel (without its own regex), BertPreTokenizer and Sequence. Others are refused by name.

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
    // True when the pieces must be mapped to the byte-level alphabet before the model (a final ByteLevel step). The
    // mapping itself happens in the model, which then works on bytes.
    virtual bool byte_level() const { return false; }
};

// Returns nullptr for a null pre-tokenizer; refuses unsupported types by name.
std::unique_ptr<PreTokenizer> make_pre_tokenizer(const nlohmann::json& config);

}  // namespace xs::tok
