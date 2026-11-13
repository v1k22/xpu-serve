#pragma once

// tokenizer.json pre-tokenizers: Split (a string pattern, or one of the regexes in regex_split.h; every split
// behavior), ByteLevel (add_prefix_space; use_regex: the GPT-2 regex), Metaspace (prepend_scheme always or never),
// BertPreTokenizer and Sequence. Others are refused by name.

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace xs::tok {

class PreTokenizer {
public:
    virtual ~PreTokenizer() = default;
    // Splits each piece further, as Hugging Face pre-tokenizes each split on its own; some also rewrite a piece
    // (Metaspace's replacement and prefix, ByteLevel's prefix space). Empty pieces, given or made, are dropped.
    virtual std::vector<std::string> split(std::vector<std::string> pieces) const = 0;
    // True when the pieces must be mapped to the byte-level alphabet before the model (a final ByteLevel step). The
    // mapping itself happens in the model, which then works on bytes.
    virtual bool byte_level() const { return false; }
};

// Returns nullptr for a null pre-tokenizer; refuses unsupported types by name.
std::unique_ptr<PreTokenizer> make_pre_tokenizer(const nlohmann::json& config);

}  // namespace xs::tok
