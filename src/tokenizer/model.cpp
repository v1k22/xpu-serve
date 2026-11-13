#include "tokenizer/model.h"

#include <algorithm>
#include <climits>
#include <string>

#include "tokenizer/bpe_model.h"
#include "tokenizer/json_fields.h"
#include "util/error.h"

namespace xs::tok {

Vocab::Vocab(const nlohmann::json& config, std::string_view model) {
    if (!config.contains("vocab") || !config.at("vocab").is_object()) {
        fail("tokenizer.json ", model, " model has no 'vocab'");
    }
    const nlohmann::json& vocab = config.at("vocab");
    size_t total_bytes = 0;
    int64_t max_id = -1;
    for (const auto& [piece, value] : vocab.items()) {
        if (!value.is_number_integer() || value.get<int64_t>() < 0 || value.get<int64_t>() > INT32_MAX) {
            fail("tokenizer.json ", model, " vocab entry '", piece, "' has an invalid id");
        }
        total_bytes += piece.size();
        max_id = std::max(max_id, value.get<int64_t>());
    }
    storage_.reserve(total_bytes);  // reserved up front so the views taken below stay valid
    pieces_.assign(static_cast<size_t>(max_id + 1), std::nullopt);
    ids_.reserve(vocab.size());
    for (const auto& [piece, value] : vocab.items()) {
        const auto id_value = value.get<int32_t>();
        const size_t offset = storage_.size();
        storage_.insert(storage_.end(), piece.begin(), piece.end());
        const std::string_view view(storage_.data() + offset, piece.size());
        if (pieces_[id_value]) fail("tokenizer.json ", model, " vocab id ", id_value, " is assigned twice");
        pieces_[id_value] = view;
        ids_.emplace(view, id_value);
    }
}

std::optional<int32_t> Vocab::id(std::string_view piece) const {
    const auto it = ids_.find(piece);
    if (it == ids_.end()) return std::nullopt;
    return it->second;
}

std::optional<std::string_view> Vocab::piece(int32_t id) const {
    if (id < 0 || id >= id_limit()) return std::nullopt;
    return pieces_[id];
}

std::unique_ptr<Model> make_model(const nlohmann::json& config, bool byte_level) {
    const std::string type = component_type(config, "model");
    if (type == "BPE") return std::make_unique<BpeModel>(config, byte_level);
    refuse_component("model", type);
}

}  // namespace xs::tok
