#include "tokenizer/normalizer.h"

#include <vector>

#include "tokenizer/json_fields.h"
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
    if (type == "Sequence") return make_sequence(config);
    refuse_component(kWhat, type);
}

}  // namespace xs::tok
