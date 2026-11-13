#include "tokenizer/pre_tokenizer.h"

#include <algorithm>
#include <string>
#include <utility>

#include "tokenizer/json_fields.h"
#include "tokenizer/normalizer.h"
#include "tokenizer/regex_split.h"
#include "tokenizer/unicode.h"
#include "tokenizer/utf8.h"
#include "util/error.h"

namespace xs::tok {
namespace {
constexpr std::string_view kWhat = "pre_tokenizer";

enum class SplitBehavior { kRemoved, kIsolated, kMergedWithPrevious, kMergedWithNext, kContiguous };

SplitBehavior parse_behavior(const std::string& name) {
    if (name == "Removed") return SplitBehavior::kRemoved;
    if (name == "Isolated") return SplitBehavior::kIsolated;
    if (name == "MergedWithPrevious") return SplitBehavior::kMergedWithPrevious;
    if (name == "MergedWithNext") return SplitBehavior::kMergedWithNext;
    if (name == "Contiguous") return SplitBehavior::kContiguous;
    fail("tokenizer.json pre_tokenizer 'Split' behavior '", name, "' is not supported");
}

struct Span {
    size_t begin;
    size_t end;
    bool is_match;
};

// Alternating non-match / match spans covering the whole text (Hugging Face's Pattern::find_matches).
std::vector<Span> to_spans(std::string_view text, const std::vector<std::pair<size_t, size_t>>& matches, bool invert) {
    std::vector<Span> spans;
    size_t pos = 0;
    for (const auto& [begin, end] : matches) {
        if (begin > pos) spans.push_back({pos, begin, invert});
        spans.push_back({begin, end, !invert});
        pos = end;
    }
    if (pos < text.size() || spans.empty()) spans.push_back({pos, text.size(), invert});
    return spans;
}

std::vector<std::pair<size_t, size_t>> find_string(std::string_view text, std::string_view pattern) {
    std::vector<std::pair<size_t, size_t>> matches;
    for (size_t hit = text.find(pattern); hit != std::string_view::npos;) {
        matches.emplace_back(hit, hit + pattern.size());
        hit = text.find(pattern, hit + pattern.size());
    }
    return matches;
}

// Folds spans per behavior, mirroring Hugging Face's NormalizedString::split.
std::vector<Span> apply_behavior(std::vector<Span> spans, SplitBehavior behavior) {
    std::vector<Span> out;
    bool previous_match = false;
    switch (behavior) {
        case SplitBehavior::kIsolated:
            return spans;
        case SplitBehavior::kRemoved:
            std::erase_if(spans, [](const Span& s) { return s.is_match; });
            return spans;
        case SplitBehavior::kMergedWithPrevious:
            for (const Span& s : spans) {
                if (s.is_match && !previous_match && !out.empty()) {
                    out.back().end = s.end;
                } else {
                    out.push_back(s);
                }
                previous_match = s.is_match;
            }
            return out;
        case SplitBehavior::kMergedWithNext:
            for (auto it = spans.rbegin(); it != spans.rend(); ++it) {
                if (it->is_match && !previous_match && !out.empty()) {
                    out.back().begin = it->begin;
                } else {
                    out.push_back(*it);
                }
                previous_match = it->is_match;
            }
            std::reverse(out.begin(), out.end());
            return out;
        case SplitBehavior::kContiguous:
            for (const Span& s : spans) {
                if (s.is_match == previous_match && !out.empty()) {
                    out.back().end = s.end;
                } else {
                    out.push_back(s);
                }
                previous_match = s.is_match;
            }
            return out;
    }
    return out;
}

// Appends the non-empty spans of `piece` to `out`.
void append_spans(std::string_view piece, const std::vector<Span>& spans, std::vector<std::string>& out) {
    for (const Span& s : spans) {
        if (s.end > s.begin) out.emplace_back(piece.substr(s.begin, s.end - s.begin));
    }
}

class SplitPreTokenizer final : public PreTokenizer {
public:
    // Exactly one of `pattern` (non-empty) and `regex` is set.
    SplitPreTokenizer(std::string pattern, std::unique_ptr<RegexSplitter> regex, SplitBehavior behavior, bool invert)
        : pattern_(std::move(pattern)), regex_(std::move(regex)), behavior_(behavior), invert_(invert) {}

    std::vector<std::string> split(std::vector<std::string> pieces) const override {
        std::vector<std::string> out;
        for (const std::string& piece : pieces) {
            const auto matches = regex_ ? regex_->find_matches(piece) : find_string(piece, pattern_);
            append_spans(piece, apply_behavior(to_spans(piece, matches, invert_), behavior_), out);
        }
        return out;
    }

private:
    std::string pattern_;
    std::unique_ptr<RegexSplitter> regex_;
    SplitBehavior behavior_;
    bool invert_;
};

// Hugging Face's ByteLevel: with add_prefix_space, a space before each piece that does not start with one; with
// use_regex, the GPT-2 regex's matches and what lies between them as pieces. Marks the pieces as byte-level.
class ByteLevelPreTokenizer final : public PreTokenizer {
public:
    ByteLevelPreTokenizer(bool add_prefix_space, std::unique_ptr<RegexSplitter> regex)
        : add_prefix_space_(add_prefix_space), regex_(std::move(regex)) {}

    std::vector<std::string> split(std::vector<std::string> pieces) const override {
        std::vector<std::string> out;
        for (std::string& piece : pieces) {
            if (piece.empty()) continue;
            if (add_prefix_space_ && !piece.starts_with(' ')) piece.insert(0, 1, ' ');
            if (regex_) {
                append_spans(piece, to_spans(piece, regex_->find_matches(piece), false), out);
            } else {
                out.push_back(std::move(piece));
            }
        }
        return out;
    }
    bool byte_level() const override { return true; }

private:
    bool add_prefix_space_;
    std::unique_ptr<RegexSplitter> regex_;  // null without use_regex
};

// Hugging Face's Metaspace: spaces become the replacement character, which (prepend_scheme always) is put before each
// piece that does not start with it; with split, each replacement character starts a new piece.
class MetaspacePreTokenizer final : public PreTokenizer {
public:
    MetaspacePreTokenizer(std::string replacement, bool prepend, bool split)
        : replacement_(std::move(replacement)), prepend_(prepend), split_(split) {}

    std::vector<std::string> split(std::vector<std::string> pieces) const override {
        std::vector<std::string> out;
        for (const std::string& piece : pieces) {
            if (piece.empty()) continue;
            std::string text = replace_all(piece, " ", replacement_);
            if (prepend_ && !text.starts_with(replacement_)) text.insert(0, replacement_);
            if (split_) {
                const auto spans = to_spans(text, find_string(text, replacement_), false);
                append_spans(text, apply_behavior(spans, SplitBehavior::kMergedWithNext), out);
            } else {
                out.push_back(std::move(text));
            }
        }
        return out;
    }

private:
    std::string replacement_;  // one character
    bool prepend_;
    bool split_;
};

// Hugging Face's BertPreTokenizer: whitespace (Rust's char::is_whitespace) ends a piece and is dropped, and each
// punctuation character (bert.rs is_bert_punc) is a piece of its own.
class BertPreTokenizer final : public PreTokenizer {
public:
    std::vector<std::string> split(std::vector<std::string> pieces) const override {
        std::vector<std::string> out;
        for (const std::string& piece : pieces) {
            size_t word = 0;  // start of the piece being read
            for (size_t pos = 0; pos < piece.size();) {
                const char32_t c = utf8::decode_at(piece, pos);
                const size_t len = utf8::char_length(piece, pos);
                const bool space = utf8::is_whitespace(c);
                if (space || unicode::is_bert(c, unicode::kBertPunctuation)) {
                    if (pos > word) out.emplace_back(piece, word, pos - word);
                    if (!space) out.emplace_back(piece, pos, len);
                    word = pos + len;
                }
                pos += len;
            }
            if (piece.size() > word) out.emplace_back(piece, word);
        }
        return out;
    }
};

class SequencePreTokenizer final : public PreTokenizer {
public:
    explicit SequencePreTokenizer(std::vector<std::unique_ptr<PreTokenizer>> steps) : steps_(std::move(steps)) {
        for (size_t i = 0; i + 1 < steps_.size(); ++i) {
            if (steps_[i]->byte_level()) fail("tokenizer.json pre_tokenizer 'ByteLevel' must be the last step");
        }
    }

    std::vector<std::string> split(std::vector<std::string> pieces) const override {
        for (const auto& step : steps_) pieces = step->split(std::move(pieces));
        return pieces;
    }
    bool byte_level() const override { return !steps_.empty() && steps_.back()->byte_level(); }

private:
    std::vector<std::unique_ptr<PreTokenizer>> steps_;
};

std::unique_ptr<PreTokenizer> make_split(const nlohmann::json& config) {
    const SplitBehavior behavior = parse_behavior(string_field(config, "behavior", kWhat));
    const bool invert = bool_field_or(config, "invert", false, kWhat);
    if (const auto regex = regex_pattern(config, kWhat)) {
        auto splitter = RegexSplitter::for_pattern(*regex);
        if (!splitter) fail("tokenizer.json pre_tokenizer 'Split' Regex '", *regex, "' is not supported");
        return std::make_unique<SplitPreTokenizer>(std::string(), std::move(splitter), behavior, invert);
    }
    return std::make_unique<SplitPreTokenizer>(string_pattern(config, kWhat), nullptr, behavior, invert);
}

// trim_offsets only moves offsets, which encode does not return.
std::unique_ptr<PreTokenizer> make_byte_level(const nlohmann::json& config) {
    const bool add_prefix_space = bool_field_or(config, "add_prefix_space", true, kWhat);
    const bool use_regex = bool_field_or(config, "use_regex", true, kWhat);
    return std::make_unique<ByteLevelPreTokenizer>(add_prefix_space, use_regex ? RegexSplitter::gpt2() : nullptr);
}

// Hugging Face reads the older add_prefix_space: false as prepend_scheme never, and refuses it beside another scheme.
std::unique_ptr<PreTokenizer> make_metaspace(const nlohmann::json& config) {
    std::string replacement = string_field(config, "replacement", kWhat);
    if (replacement.empty() || utf8::valid_char_length(replacement, 0) != replacement.size()) {
        fail("tokenizer.json pre_tokenizer 'Metaspace': 'replacement' must be exactly one character");
    }
    std::string scheme = "always";
    if (config.contains("prepend_scheme") && !config.at("prepend_scheme").is_null()) {
        scheme = string_field(config, "prepend_scheme", kWhat);
    }
    if (!bool_field_or(config, "add_prefix_space", true, kWhat) && scheme != "never") {
        fail("tokenizer.json pre_tokenizer 'Metaspace': add_prefix_space false does not match prepend_scheme '",
             scheme, "'");
    }
    if (scheme != "always" && scheme != "never") {
        fail("tokenizer.json pre_tokenizer 'Metaspace' prepend_scheme '", scheme, "' is not supported");
    }
    return std::make_unique<MetaspacePreTokenizer>(std::move(replacement), scheme == "always",
                                                   bool_field_or(config, "split", true, kWhat));
}

std::unique_ptr<PreTokenizer> make_sequence(const nlohmann::json& config) {
    if (!config.contains("pretokenizers") || !config.at("pretokenizers").is_array()) {
        fail("tokenizer.json pre_tokenizer 'Sequence' has no 'pretokenizers' list");
    }
    std::vector<std::unique_ptr<PreTokenizer>> steps;
    for (const auto& step : config.at("pretokenizers")) {
        auto pre_tokenizer = make_pre_tokenizer(step);
        if (pre_tokenizer) steps.push_back(std::move(pre_tokenizer));
    }
    return std::make_unique<SequencePreTokenizer>(std::move(steps));
}
}  // namespace

std::unique_ptr<PreTokenizer> make_pre_tokenizer(const nlohmann::json& config) {
    if (config.is_null()) return nullptr;
    const std::string type = component_type(config, kWhat);
    if (type == "Split") return make_split(config);
    if (type == "ByteLevel") return make_byte_level(config);
    if (type == "Metaspace") return make_metaspace(config);
    if (type == "BertPreTokenizer") return std::make_unique<BertPreTokenizer>();
    if (type == "Sequence") return make_sequence(config);
    refuse_component(kWhat, type);
}

}  // namespace xs::tok
