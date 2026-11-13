#include "tokenizer/post_processor.h"

#include <climits>
#include <string>
#include <unordered_map>
#include <utility>

#include "tokenizer/json_fields.h"
#include "util/error.h"

namespace xs::tok {
namespace {

// Hugging Face's TemplateProcessing for one sequence: the special tokens its `single` template puts before and after
// the sequence ($A), with the ids `special_tokens` lists for them (never looked up in the vocab).
class TemplateProcessing final : public PostProcessor {
public:
    TemplateProcessing(std::vector<int32_t> before, std::vector<int32_t> after)
        : before_(std::move(before)), after_(std::move(after)) {}

    std::vector<int32_t> single(std::vector<int32_t> ids) const override {
        ids.insert(ids.begin(), before_.begin(), before_.end());
        ids.insert(ids.end(), after_.begin(), after_.end());
        return ids;
    }

private:
    std::vector<int32_t> before_;
    std::vector<int32_t> after_;
};

[[noreturn]] void refuse_template(const std::string& why) {
    fail("tokenizer.json post_processor 'TemplateProcessing': ", why);
}

// special_tokens: {"<name>": {"id": "<name>", "ids": [...], "tokens": [...]}}, looked up by key.
std::unordered_map<std::string, std::vector<int32_t>> special_token_ids(const nlohmann::json& config) {
    if (!config.contains("special_tokens") || !config.at("special_tokens").is_object()) {
        refuse_template("no 'special_tokens' object");
    }
    std::unordered_map<std::string, std::vector<int32_t>> tokens;
    for (const auto& [name, token] : config.at("special_tokens").items()) {
        if (!token.is_object() || !token.contains("ids") || !token.at("ids").is_array()) {
            refuse_template("special token '" + name + "' has no 'ids' list");
        }
        std::vector<int32_t>& ids = tokens[name];
        for (const auto& id : token.at("ids")) {
            if (!id.is_number_unsigned() || id.get<uint64_t>() > INT32_MAX) {
                refuse_template("special token '" + name + "' has an invalid id");
            }
            ids.push_back(id.get<int32_t>());
        }
    }
    return tokens;
}

std::unique_ptr<PostProcessor> make_template(const nlohmann::json& config) {
    const auto special = special_token_ids(config);
    if (!config.contains("single") || !config.at("single").is_array()) refuse_template("no 'single' template");
    std::vector<int32_t> before;
    std::vector<int32_t> after;
    int sequences = 0;
    for (const auto& piece : config.at("single")) {
        const bool object = piece.is_object() && piece.size() == 1;
        if (object && piece.contains("Sequence") && piece.at("Sequence").is_object()) {
            const auto& sequence = piece.at("Sequence");
            const std::string id = sequence.contains("id") && sequence.at("id").is_string()
                                       ? sequence.at("id").get<std::string>()
                                       : std::string();
            if (id != "A") refuse_template("the single template names sequence '" + id + "' (one sequence is A)");
            ++sequences;
        } else if (object && piece.contains("SpecialToken") && piece.at("SpecialToken").is_object()) {
            const std::string name = string_field(piece.at("SpecialToken"), "id", "post_processor");
            const auto it = special.find(name);
            if (it == special.end()) refuse_template("special token '" + name + "' is not in 'special_tokens'");
            std::vector<int32_t>& side = sequences == 0 ? before : after;
            side.insert(side.end(), it->second.begin(), it->second.end());
        } else {
            refuse_template("a single template piece is neither a Sequence nor a SpecialToken");
        }
    }
    // Hugging Face copies every $A, even with add_special_tokens=False, which encode() would then not match.
    if (sequences != 1) refuse_template("the single template must hold sequence A exactly once");
    return std::make_unique<TemplateProcessing>(std::move(before), std::move(after));
}

}  // namespace

std::unique_ptr<PostProcessor> make_post_processor(const nlohmann::json& config) {
    if (config.is_null()) return nullptr;
    const std::string type = component_type(config, "post_processor");
    if (type == "TemplateProcessing") return make_template(config);
    if (type == "ByteLevel") return nullptr;
    refuse_component("post_processor", type);
}

}  // namespace xs::tok
