#include "tokenizer/pre_tokenizer.h"

#include <algorithm>
#include <string>
#include <utility>

#include "tokenizer/json_fields.h"
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
std::vector<Span> find_matches(std::string_view text, std::string_view pattern, bool invert) {
    std::vector<Span> spans;
    size_t pos = 0;
    for (size_t hit = text.find(pattern); hit != std::string_view::npos; hit = text.find(pattern, pos)) {
        if (hit > pos) spans.push_back({pos, hit, invert});
        spans.push_back({hit, hit + pattern.size(), !invert});
        pos = hit + pattern.size();
    }
    if (pos < text.size() || spans.empty()) spans.push_back({pos, text.size(), invert});
    return spans;
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
    SplitPreTokenizer(std::string pattern, SplitBehavior behavior, bool invert)
        : pattern_(std::move(pattern)), behavior_(behavior), invert_(invert) {}

    std::vector<std::string_view> split(const std::vector<std::string_view>& pieces) const override {
        std::vector<std::string_view> out;
        for (std::string_view piece : pieces) {
            for (const Span& s : apply_behavior(find_matches(piece, pattern_, invert_), behavior_)) {
                if (s.end > s.begin) out.push_back(piece.substr(s.begin, s.end - s.begin));
            }
        }
        return out;
    }

private:
    std::string pattern_;
    SplitBehavior behavior_;
    bool invert_;
};

class SequencePreTokenizer final : public PreTokenizer {
public:
    explicit SequencePreTokenizer(std::vector<std::unique_ptr<PreTokenizer>> steps) : steps_(std::move(steps)) {}

    std::vector<std::string_view> split(const std::vector<std::string_view>& pieces) const override {
        std::vector<std::string_view> out = pieces;
        for (const auto& step : steps_) out = step->split(out);
        return out;
    }

private:
    std::vector<std::unique_ptr<PreTokenizer>> steps_;
};

std::unique_ptr<PreTokenizer> make_split(const nlohmann::json& config) {
    return std::make_unique<SplitPreTokenizer>(string_pattern(config, kWhat),
                                               parse_behavior(string_field(config, "behavior", kWhat)),
                                               bool_field_or(config, "invert", false, kWhat));
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
    if (type == "Sequence") return make_sequence(config);
    refuse_component(kWhat, type);
}

}  // namespace xs::tok
