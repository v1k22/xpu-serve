#include "tokenizer/pre_tokenizer.h"

#include <algorithm>
#include <string>
#include <utility>

#include "tokenizer/json_fields.h"
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

class SplitPreTokenizer final : public PreTokenizer {
public:
    // Exactly one of `pattern` (non-empty) and `regex` is set.
    SplitPreTokenizer(std::string pattern, std::unique_ptr<RegexSplitter> regex, SplitBehavior behavior, bool invert)
        : pattern_(std::move(pattern)), regex_(std::move(regex)), behavior_(behavior), invert_(invert) {}

    std::vector<std::string_view> split(const std::vector<std::string_view>& pieces) const override {
        std::vector<std::string_view> out;
        for (std::string_view piece : pieces) {
            const auto matches = regex_ ? regex_->find_matches(piece) : find_string(piece, pattern_);
            for (const Span& s : apply_behavior(to_spans(piece, matches, invert_), behavior_)) {
                if (s.end > s.begin) out.push_back(piece.substr(s.begin, s.end - s.begin));
            }
        }
        return out;
    }

private:
    std::string pattern_;
    std::unique_ptr<RegexSplitter> regex_;
    SplitBehavior behavior_;
    bool invert_;
};

// Only marks the pieces as byte-level; with use_regex false it does not split.
class ByteLevelPreTokenizer final : public PreTokenizer {
public:
    std::vector<std::string_view> split(const std::vector<std::string_view>& pieces) const override { return pieces; }
    bool byte_level() const override { return true; }
};

// Hugging Face's BertPreTokenizer: whitespace (Rust's char::is_whitespace) ends a piece and is dropped, and each
// punctuation character (bert.rs is_bert_punc) is a piece of its own.
class BertPreTokenizer final : public PreTokenizer {
public:
    std::vector<std::string_view> split(const std::vector<std::string_view>& pieces) const override {
        std::vector<std::string_view> out;
        for (const std::string_view piece : pieces) {
            size_t word = 0;  // start of the piece being read
            for (size_t pos = 0; pos < piece.size();) {
                const char32_t c = utf8::decode_at(piece, pos);
                const size_t len = utf8::char_length(piece, pos);
                const bool space = utf8::is_whitespace(c);
                if (space || unicode::is_bert(c, unicode::kBertPunctuation)) {
                    if (pos > word) out.push_back(piece.substr(word, pos - word));
                    if (!space) out.push_back(piece.substr(pos, len));
                    word = pos + len;
                }
                pos += len;
            }
            if (piece.size() > word) out.push_back(piece.substr(word));
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

    std::vector<std::string_view> split(const std::vector<std::string_view>& pieces) const override {
        std::vector<std::string_view> out = pieces;
        for (const auto& step : steps_) out = step->split(out);
        return out;
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

std::unique_ptr<PreTokenizer> make_byte_level(const nlohmann::json& config) {
    if (bool_field_or(config, "add_prefix_space", true, kWhat)) {
        fail("tokenizer.json pre_tokenizer 'ByteLevel' with add_prefix_space is not supported");
    }
    if (bool_field_or(config, "use_regex", true, kWhat)) {
        fail("tokenizer.json pre_tokenizer 'ByteLevel' with use_regex is not supported");
    }
    return std::make_unique<ByteLevelPreTokenizer>();
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
    if (type == "BertPreTokenizer") return std::make_unique<BertPreTokenizer>();
    if (type == "Sequence") return make_sequence(config);
    refuse_component(kWhat, type);
}

}  // namespace xs::tok
