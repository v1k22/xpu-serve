#include "tokenizer/normalizer.h"

#include <vector>

#include "tokenizer/json_fields.h"
#include "tokenizer/unicode.h"
#include "tokenizer/utf8.h"
#include "util/error.h"

namespace xs::tok {
namespace {
constexpr std::string_view kWhat = "normalizer";

class ReplaceNormalizer final : public Normalizer {
public:
    ReplaceNormalizer(std::string from, std::string to) : from_(std::move(from)), to_(std::move(to)) {}
    std::string normalize(std::string_view text) const override { return replace_all(text, from_, to_); }

private:
    std::string from_;
    std::string to_;
};

class PrependNormalizer final : public Normalizer {
public:
    explicit PrependNormalizer(std::string prefix) : prefix_(std::move(prefix)) {}
    std::string normalize(std::string_view text) const override {
        if (text.empty()) return {};
        std::string out = prefix_;
        out.append(text);
        return out;
    }

private:
    std::string prefix_;
};

class StripNormalizer final : public Normalizer {
public:
    StripNormalizer(bool left, bool right) : left_(left), right_(right) {}
    std::string normalize(std::string_view text) const override {
        if (left_) text = utf8::trim_whitespace_left(text);
        if (right_) text = utf8::trim_whitespace_right(text);
        return std::string(text);
    }

private:
    bool left_;
    bool right_;
};

class NfcNormalizer final : public Normalizer {
public:
    std::string normalize(std::string_view text) const override { return unicode::nfc(text); }
};

// Hugging Face's BertNormalizer (normalizers/bert.rs), its steps in its order. strip_accents is NFD without the
// nonspacing marks; lowercase maps each character on its own.
class BertNormalizer final : public Normalizer {
public:
    BertNormalizer(bool clean_text, bool handle_chinese_chars, bool strip_accents, bool lowercase)
        : clean_text_(clean_text),
          handle_chinese_chars_(handle_chinese_chars),
          strip_accents_(strip_accents),
          lowercase_(lowercase) {}

    std::string normalize(std::string_view text) const override {
        // Rust strings are valid UTF-8: an invalid sequence reads as U+FFFD, as with String::from_utf8_lossy.
        const std::string valid = utf8::lossy(text);
        std::u32string chars;
        chars.reserve(valid.size());
        for (size_t pos = 0; pos < valid.size(); pos += utf8::char_length(valid, pos)) {
            chars.push_back(utf8::decode_at(valid, pos));
        }
        if (clean_text_) {
            std::erase_if(chars,
                          [](char32_t c) { return c == 0 || c == 0xFFFD || unicode::is_bert(c, unicode::kBertControl); });
            for (char32_t& c : chars) {
                if (utf8::is_whitespace(c)) c = U' ';
            }
        }
        if (handle_chinese_chars_) {
            std::u32string spaced;
            spaced.reserve(chars.size());
            for (const char32_t c : chars) {
                if (unicode::is_bert(c, unicode::kBertChinese)) {
                    spaced += {U' ', c, U' '};
                } else {
                    spaced.push_back(c);
                }
            }
            chars = std::move(spaced);
        }
        if (strip_accents_) {
            unicode::nfd(chars);
            std::erase_if(chars, [](char32_t c) { return unicode::is_bert(c, unicode::kBertNonspacingMark); });
        }
        if (lowercase_) {
            std::u32string lowered;
            lowered.reserve(chars.size());
            for (const char32_t c : chars) unicode::append_lowercase(c, lowered);
            chars = std::move(lowered);
        }
        std::string out;
        out.reserve(chars.size());
        for (const char32_t c : chars) utf8::append(out, c);
        return out;
    }

private:
    bool clean_text_;
    bool handle_chinese_chars_;
    bool strip_accents_;
    bool lowercase_;
};

std::unique_ptr<Normalizer> make_bert(const nlohmann::json& config) {
    const bool lowercase = bool_field(config, "lowercase", kWhat);
    return std::make_unique<BertNormalizer>(bool_field(config, "clean_text", kWhat),
                                            bool_field(config, "handle_chinese_chars", kWhat),
                                            bool_field_or(config, "strip_accents", lowercase, kWhat), lowercase);
}

class SequenceNormalizer final : public Normalizer {
public:
    explicit SequenceNormalizer(std::vector<std::unique_ptr<Normalizer>> steps) : steps_(std::move(steps)) {}
    std::string normalize(std::string_view text) const override {
        std::string out(text);
        for (const auto& step : steps_) out = step->normalize(out);
        return out;
    }

private:
    std::vector<std::unique_ptr<Normalizer>> steps_;
};

std::unique_ptr<Normalizer> make_sequence(const nlohmann::json& config) {
    if (!config.contains("normalizers") || !config.at("normalizers").is_array()) {
        fail("tokenizer.json normalizer 'Sequence' has no 'normalizers' list");
    }
    std::vector<std::unique_ptr<Normalizer>> steps;
    for (const auto& step : config.at("normalizers")) {
        auto normalizer = make_normalizer(step);
        if (normalizer) steps.push_back(std::move(normalizer));
    }
    return std::make_unique<SequenceNormalizer>(std::move(steps));
}
}  // namespace

std::string replace_all(std::string_view text, std::string_view from, std::string_view to) {
    std::string out;
    out.reserve(text.size());
    size_t pos = 0;
    for (size_t hit = text.find(from); hit != std::string_view::npos; hit = text.find(from, pos)) {
        out.append(text.substr(pos, hit - pos));
        out.append(to);
        pos = hit + from.size();
    }
    out.append(text.substr(pos));
    return out;
}

std::unique_ptr<Normalizer> make_normalizer(const nlohmann::json& config) {
    if (config.is_null()) return nullptr;
    const std::string type = component_type(config, kWhat);
    if (type == "Replace") {
        return std::make_unique<ReplaceNormalizer>(string_pattern(config, kWhat),
                                                   string_field(config, "content", kWhat));
    }
    if (type == "Prepend") return std::make_unique<PrependNormalizer>(string_field(config, "prepend", kWhat));
    if (type == "Strip") {
        return std::make_unique<StripNormalizer>(bool_field_or(config, "strip_left", false, kWhat),
                                                 bool_field_or(config, "strip_right", false, kWhat));
    }
    if (type == "NFC") return std::make_unique<NfcNormalizer>();
    if (type == "BertNormalizer") return make_bert(config);
    if (type == "Sequence") return make_sequence(config);
    refuse_component(kWhat, type);
}

}  // namespace xs::tok
