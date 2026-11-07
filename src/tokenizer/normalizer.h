#pragma once

// tokenizer.json normalizers: Replace (string pattern), Prepend, Strip, NFC and Sequence.

#include <memory>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace xs::tok {

class Normalizer {
public:
    virtual ~Normalizer() = default;
    virtual std::string normalize(std::string_view text) const = 0;
};

// Returns nullptr for a null normalizer; refuses unsupported types by name.
std::unique_ptr<Normalizer> make_normalizer(const nlohmann::json& config);

// Replaces every non-overlapping occurrence of `from` (non-empty), scanning left to right.
std::string replace_all(std::string_view text, std::string_view from, std::string_view to);

}  // namespace xs::tok
