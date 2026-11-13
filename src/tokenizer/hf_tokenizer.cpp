// The Tokenizer behind Tokenizer::load: a Hugging Face tokenizer.json pipeline (added tokens -> normalizer ->
// pre-tokenizer -> model; the post-processor on request; decoder on the way back). tokenizer.json's truncation and
// padding are not applied: callers own sequence lengths.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "models/config_util.h"
#include "tokenizer/added_vocabulary.h"
#include "tokenizer/decoder.h"
#include "tokenizer/model.h"
#include "tokenizer/normalizer.h"
#include "tokenizer/post_processor.h"
#include "tokenizer/pre_tokenizer.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"
#include "util/mmap_file.h"

namespace xs::tok {
namespace {

// tokenizer_config.json writes a special token as "<bos>" or as {"content": "<bos>", ...}.
std::optional<std::string> config_token(const nlohmann::json& config, const char* key) {
    if (!config.contains(key)) return std::nullopt;
    const auto& value = config.at(key);
    if (value.is_string()) return value.get<std::string>();
    if (value.is_object() && value.contains("content") && value.at("content").is_string()) {
        return value.at("content").get<std::string>();
    }
    return std::nullopt;
}

// tokenizer.json added_tokens plus the tokenizer_config.json added_tokens_decoder entries it lacks: transformers
// registers both (Qwen3.5 lists its audio and TTS markers only in tokenizer_config.json).
nlohmann::json merged_added_tokens(const nlohmann::json& tokenizer_json, const nlohmann::json& tokenizer_config) {
    nlohmann::json added = tokenizer_json.value("added_tokens", nlohmann::json::array());
    if (!tokenizer_config.contains("added_tokens_decoder") || !added.is_array()) return added;
    const auto& decoder = tokenizer_config.at("added_tokens_decoder");
    if (!decoder.is_object()) fail("tokenizer_config.json 'added_tokens_decoder' must be an object");
    std::unordered_set<int64_t> known;
    for (const auto& entry : added) {
        if (entry.is_object() && entry.contains("id") && entry.at("id").is_number_integer()) {
            known.insert(entry.at("id").get<int64_t>());
        }
    }
    for (const auto& [key, entry] : decoder.items()) {
        int64_t id = 0;
        try {
            size_t used = 0;
            id = std::stoll(key, &used);
            if (used != key.size()) throw std::invalid_argument(key);
        } catch (const std::exception&) {
            fail("tokenizer_config.json added_tokens_decoder key '", key, "' is not a token id");
        }
        if (known.contains(id) || !entry.is_object()) continue;
        nlohmann::json token = entry;
        token["id"] = id;
        added.push_back(std::move(token));
    }
    return added;
}

class HfTokenizer final : public Tokenizer {
public:
    HfTokenizer(const nlohmann::json& tokenizer_json, const nlohmann::json& tokenizer_config)
        : normalizer_(make_normalizer(tokenizer_json.value("normalizer", nlohmann::json()))),
          added_(merged_added_tokens(tokenizer_json, tokenizer_config), normalizer_.get()),
          pre_tokenizer_(make_pre_tokenizer(tokenizer_json.value("pre_tokenizer", nlohmann::json()))),
          model_(make_model(tokenizer_json.contains("model") ? tokenizer_json.at("model") : nlohmann::json(),
                            pre_tokenizer_ && pre_tokenizer_->byte_level())),
          post_processor_(make_post_processor(tokenizer_json.value("post_processor", nlohmann::json()))),
          decoder_(make_decoder(tokenizer_json.value("decoder", nlohmann::json()))) {
        id_limit_ = model_->id_limit();
        for (const AddedToken& token : added_.tokens()) {
            id_limit_ = std::max(id_limit_, token.id + 1);
            added_by_id_.emplace(token.id, &token);
            added_by_content_.emplace(token.content, token.id);
            if (token.special) special_ids_.insert(token.id);
        }
        bos_id_ = config_token_id(tokenizer_config, "bos_token");
        eos_id_ = config_token_id(tokenizer_config, "eos_token");
    }

    std::vector<int32_t> encode(std::string_view text, bool parse_special) const override {
        std::vector<int32_t> ids;
        for (const TextSegment& raw : added_.split(text, false, parse_special)) {
            if (raw.token) {
                ids.push_back(*raw.token);
                continue;
            }
            const std::string normalized = normalizer_ ? normalizer_->normalize(raw.text) : std::string(raw.text);
            for (const TextSegment& segment : split_normalized(normalized, parse_special)) {
                if (segment.token) {
                    ids.push_back(*segment.token);
                } else {
                    encode_plain(segment.text, ids);
                }
            }
        }
        return ids;
    }

    std::vector<int32_t> with_special_tokens(std::vector<int32_t> ids) const override {
        return post_processor_ ? post_processor_->single(std::move(ids)) : ids;
    }

    std::string decode(std::span<const int32_t> ids, bool skip_special) const override {
        std::vector<std::string> tokens;
        tokens.reserve(ids.size());
        for (const int32_t id : ids) {
            if (skip_special && is_special(id)) continue;
            tokens.push_back(token_text(id));
        }
        if (!decoder_) return join_with_spaces(tokens);  // Hugging Face's behavior without a decoder
        decoder_->decode(tokens);
        std::string text;
        for (const auto& token : tokens) text += token;
        return text;
    }

    int32_t vocab_size() const override { return id_limit_; }

    std::string token_text(int32_t id) const override {
        if (const auto it = added_by_id_.find(id); it != added_by_id_.end()) return it->second->pattern;
        const auto piece = model_->piece(id);
        if (!piece) fail("token id ", id, " is not in the vocabulary (size ", id_limit_, ")");
        return std::string(*piece);
    }

    std::optional<int32_t> token_id(std::string_view piece) const override {
        const auto it = added_by_content_.find(std::string(piece));
        if (it != added_by_content_.end()) return it->second;
        return model_->id(piece);
    }

    bool is_special(int32_t id) const override { return special_ids_.contains(id); }
    std::optional<int32_t> bos_id() const override { return bos_id_; }
    std::optional<int32_t> eos_id() const override { return eos_id_; }

private:
    std::vector<TextSegment> split_normalized(std::string_view normalized, bool parse_special) const {
        if (added_.has_normalized_tokens()) return added_.split(normalized, true, parse_special);
        if (normalized.empty()) return {};
        return {TextSegment{normalized, std::nullopt}};
    }

    void encode_plain(std::string_view text, std::vector<int32_t>& ids) const {
        if (!pre_tokenizer_) {
            model_->tokenize(text, ids);
            return;
        }
        for (const std::string_view piece : pre_tokenizer_->split({text})) model_->tokenize(piece, ids);
    }

    std::optional<int32_t> config_token_id(const nlohmann::json& config, const char* key) const {
        const auto content = config_token(config, key);
        if (!content) return std::nullopt;
        const auto id = token_id(*content);
        if (!id) fail("tokenizer_config.json ", key, " '", *content, "' is not in the vocabulary");
        return id;
    }

    static std::string join_with_spaces(const std::vector<std::string>& tokens) {
        std::string text;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) text += ' ';
            text += tokens[i];
        }
        return text;
    }

    std::unique_ptr<Normalizer> normalizer_;
    AddedVocabulary added_;
    std::unique_ptr<PreTokenizer> pre_tokenizer_;
    std::unique_ptr<Model> model_;
    std::unique_ptr<PostProcessor> post_processor_;
    std::unique_ptr<Decoder> decoder_;
    int32_t id_limit_ = 0;
    std::unordered_map<int32_t, const AddedToken*> added_by_id_;  // points into added_, which never changes
    std::unordered_map<std::string, int32_t> added_by_content_;
    std::unordered_set<int32_t> special_ids_;
    std::optional<int32_t> bos_id_;
    std::optional<int32_t> eos_id_;
};

// Parses from a memory map: several times faster than nlohmann's stream parser on the 30+ MB Gemma file.
nlohmann::json read_tokenizer_json(const std::filesystem::path& path) {
    const MappedFile file(path);
    const auto* begin = reinterpret_cast<const char*>(file.data());
    try {
        return nlohmann::json::parse(begin, begin + file.size());
    } catch (const nlohmann::json::exception& e) {
        fail("'", path.string(), "' is not valid JSON: ", e.what());
    }
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::load(const std::filesystem::path& model_dir) {
    const auto tokenizer_path = model_dir / "tokenizer.json";
    if (!std::filesystem::exists(tokenizer_path)) fail("no tokenizer.json in '", model_dir.string(), "'");
    const nlohmann::json tokenizer_json = read_tokenizer_json(tokenizer_path);
    if (!tokenizer_json.is_object()) fail("'", tokenizer_path.string(), "' is not a JSON object");
    const auto config_path = model_dir / "tokenizer_config.json";
    const nlohmann::json tokenizer_config =
        std::filesystem::exists(config_path) ? cfg::read_json(config_path) : nlohmann::json::object();
    return std::make_unique<HfTokenizer>(tokenizer_json, tokenizer_config);
}

}  // namespace xs::tok
