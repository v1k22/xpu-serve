#include "convert/gguf/tokenizer.h"

#include <algorithm>
#include <array>
#include <optional>
#include <set>
#include <string_view>

#include "util/error.h"
#include "util/log.h"

namespace xs::gguf_import {
namespace {

// llama.cpp's token types (gguf TokenType).
enum TokenType : int64_t { kNormal = 1, kUnknown = 2, kControl = 3, kUserDefined = 4, kUnused = 5, kByte = 6 };

std::string token_type_name(int64_t t) {
    switch (t) {
        case kNormal:
            return "normal";
        case kUnknown:
            return "unknown";
        case kControl:
            return "control";
        case kUserDefined:
            return "user-defined";
        case kUnused:
            return "unused";
        case kByte:
            return "byte";
        default:
            return str_cat("type ", t);
    }
}

constexpr std::string_view kByteLevel =
    R"json({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})json";

// What a tokenizer.ggml.pre value stands for: the parts of the original tokenizer.json the file does not hold.
struct PreTokenizer {
    std::string_view name;
    std::string_view normalizer, pre_tokenizer, post_processor, decoder;  // JSON
    bool nfc;  // the normalizer is NFC; a file's tokenizer.ggml.normalizer.nfc, when present, must agree
    // Added tokens llama.cpp's converter types control although the original does not mark them special: it types
    // every added token that looks like <|...|> so.
    std::vector<std::string_view> not_special;
};

const std::vector<PreTokenizer>& pre_tokenizers() {
    static const std::vector<PreTokenizer> table = {
        // Qwen3.5, Qwen3.6, Qwen3.8: Qwen/Qwen3.5-4B's tokenizer.json.
        {"qwen35",
         R"json({"type": "NFC"})json",
         R"json({"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]})json",
         kByteLevel,
         kByteLevel,
         true,
         {"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"}},
    };
    return table;
}

const PreTokenizer& find_pre_tokenizer(std::string_view name) {
    std::string known;
    for (const PreTokenizer& p : pre_tokenizers()) {
        if (p.name == name) return p;
        known += str_cat(known.empty() ? "" : ", ", p.name);
    }
    fail("GGUF tokenizer.ggml.pre '", name, "' is not supported (supported: ", known,
         "); the file does not hold its pre-tokenizer, only that name");
}

// tokenizer.ggml.* keys the import reads or knows to change nothing it writes; any other is refused by name.
constexpr std::array<std::string_view, 30> kKnownKeys = {
    "model",
    "pre",
    "tokens",
    "token_type",
    "merges",
    "scores",
    "bos_token_id",
    "eos_token_id",
    "eot_token_id",
    "eom_token_id",
    "unknown_token_id",
    "seperator_token_id",
    "padding_token_id",
    "mask_token_id",
    "add_bos_token",
    "add_eos_token",
    "add_sep_token",
    "fim_pre_token_id",
    "fim_suf_token_id",
    "fim_mid_token_id",
    "fim_pad_token_id",
    "fim_rep_token_id",
    "fim_sep_token_id",
    "prefix_token_id",
    "suffix_token_id",
    "middle_token_id",
    "normalizer.nfc",
    "normalizer.lowercase",
    "normalizer.strip_accents",
    "clean_spaces",
};

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.substr(0, prefix.size()) == prefix;
}

void check_keys(const gguf::File& file, const PreTokenizer& pre) {
    for (const gguf::Value& v : file.metadata()) {
        const std::string_view key = v.key();
        if (starts_with(key, "tokenizer.ggml.")) {
            const std::string_view rest = key.substr(15);
            if (std::find(kKnownKeys.begin(), kKnownKeys.end(), rest) == kKnownKeys.end()) {
                fail("GGUF metadata '", key,
                     "' is not read by this importer, and a tokenizer key may change how "
                     "text is tokenized");
            }
        } else if (starts_with(key, "tokenizer.") && !starts_with(key, "tokenizer.chat_template") &&
                   key != "tokenizer.huggingface.json") {
            fail("GGUF metadata '", key, "' is not read by this importer (a tokenizer other than gpt2)");
        }
    }
    const auto agree = [&](std::string_view key, bool table) {
        if (const gguf::Value* v = file.find(key); v != nullptr && v->as_bool() != table) {
            fail("GGUF ", key, " is ", v->as_bool() ? "true" : "false", ", which disagrees with the tokenizer ",
                 "tokenizer.ggml.pre '", pre.name, "' stands for");
        }
    };
    agree("tokenizer.ggml.normalizer.nfc", pre.nfc);
    agree("tokenizer.ggml.normalizer.lowercase", false);
    agree("tokenizer.ggml.normalizer.strip_accents", false);
}

nlohmann::json parse(std::string_view text) {
    return nlohmann::json::parse(text);
}

}  // namespace

TokenizerFiles tokenizer_files(const gguf::File& file) {
    const std::string_view model = file.get_string("tokenizer.ggml.model");
    if (model != "gpt2") fail("GGUF tokenizer.ggml.model '", model, "' is not supported (supported: gpt2)");
    const PreTokenizer& pre = find_pre_tokenizer(file.get_string("tokenizer.ggml.pre"));
    check_keys(file, pre);

    const std::vector<std::string_view> tokens = file.get("tokenizer.ggml.tokens").strings();
    const std::vector<int64_t> types = file.get("tokenizer.ggml.token_type").ints();
    if (types.size() != tokens.size()) {
        fail("GGUF tokenizer.ggml.token_type has ", types.size(), " entries for ", tokens.size(), " tokens");
    }
    const std::set<std::string_view> not_special(pre.not_special.begin(), pre.not_special.end());

    TokenizerFiles out;
    nlohmann::json vocab = nlohmann::json::object();
    nlohmann::json added = nlohmann::json::array();
    nlohmann::ordered_json decoder = nlohmann::ordered_json::object();
    std::vector<bool> present(tokens.size(), false);
    for (size_t id = 0; id < tokens.size(); ++id) {
        const std::string text(tokens[id]);
        switch (types[id]) {
            case kNormal:
                if (vocab.contains(text)) {
                    fail("GGUF tokenizer: token '", text, "' is both id ", vocab.at(text).get<int64_t>(), " and ", id);
                }
                vocab[text] = id;
                break;
            case kControl:
            case kUserDefined: {
                const bool special = types[id] == kControl && !not_special.contains(tokens[id]);
                added.push_back({{"id", id},
                                 {"content", text},
                                 {"single_word", false},
                                 {"lstrip", false},
                                 {"rstrip", false},
                                 {"normalized", false},
                                 {"special", special}});
                decoder[std::to_string(id)] = {{"content", text}, {"lstrip", false},      {"normalized", false},
                                               {"rstrip", false}, {"single_word", false}, {"special", special}};
                break;
            }
            case kUnused:
                if (text != str_cat("[PAD", id, "]")) {
                    fail("GGUF tokenizer: token ", id, " '", text, "' is unused; only the converter's [PAD", id,
                         "] placeholders are expected to be");
                }
                continue;
            default:
                fail("GGUF tokenizer: token ", id, " '", text, "' has type ", token_type_name(types[id]),
                     ", which a gpt2 tokenizer does not have");
        }
        present[id] = true;
    }

    nlohmann::json merges = nlohmann::json::array();
    for (const std::string_view merge : file.get("tokenizer.ggml.merges").strings()) {
        const size_t space = merge.find(' ');
        if (space == std::string_view::npos || merge.find(' ', space + 1) != std::string_view::npos) {
            fail("GGUF tokenizer.ggml.merges entry '", merge, "' is not two pieces separated by one space");
        }
        merges.push_back(std::string(merge));
    }

    out.tokenizer = {{"version", "1.0"},
                     {"truncation", nullptr},
                     {"padding", nullptr},
                     {"added_tokens", std::move(added)},
                     {"normalizer", parse(pre.normalizer)},
                     {"pre_tokenizer", parse(pre.pre_tokenizer)},
                     {"post_processor", parse(pre.post_processor)},
                     {"decoder", parse(pre.decoder)}};
    out.tokenizer["model"] = {{"type", "BPE"},
                              {"dropout", nullptr},
                              {"unk_token", nullptr},
                              {"continuing_subword_prefix", ""},
                              {"end_of_word_suffix", ""},
                              {"fuse_unk", false},
                              {"byte_fallback", false},
                              {"ignore_merges", false}};
    out.tokenizer["model"]["vocab"] = std::move(vocab);
    out.tokenizer["model"]["merges"] = std::move(merges);

    // A special token's text, from its id; `required` ones must be a token of the vocabulary.
    const auto token_text = [&](std::string_view key, bool required) -> std::optional<std::string> {
        const gguf::Value* v = file.find(key);
        if (v == nullptr) {
            if (required) fail("GGUF '", file.path().string(), "' names no ", key, " (the end of a reply)");
            return std::nullopt;
        }
        const int64_t id = v->as_int();
        if (id >= 0 && static_cast<size_t>(id) < tokens.size() && present[static_cast<size_t>(id)]) {
            return std::string(tokens[static_cast<size_t>(id)]);
        }
        if (required) fail("GGUF ", key, " ", id, " is not a token of the vocabulary");
        log::warn("gguf: ", key, " ", id, " is not a token of the vocabulary; left out");
        return std::nullopt;
    };
    out.config["added_tokens_decoder"] = std::move(decoder);
    const auto bos = token_text("tokenizer.ggml.bos_token_id", false);
    out.config["bos_token"] = bos ? nlohmann::ordered_json(*bos) : nlohmann::ordered_json(nullptr);
    out.config["eos_token"] = *token_text("tokenizer.ggml.eos_token_id", true);
    for (const auto& [key, name] :
         {std::pair{"unknown_token_id", "unk_token"}, std::pair{"seperator_token_id", "sep_token"},
          std::pair{"padding_token_id", "pad_token"}, std::pair{"mask_token_id", "mask_token"}}) {
        if (const auto text = token_text(str_cat("tokenizer.ggml.", key), false)) out.config[name] = *text;
    }
    for (const auto& [key, name] :
         {std::pair{"add_bos_token", "add_bos_token"}, std::pair{"add_eos_token", "add_eos_token"},
          std::pair{"clean_spaces", "clean_up_tokenization_spaces"}}) {
        if (const gguf::Value* v = file.find(str_cat("tokenizer.ggml.", key))) out.config[name] = v->as_bool();
    }

    out.stop_ids.push_back(static_cast<int32_t>(file.get_int("tokenizer.ggml.eos_token_id")));
    for (const char* key : {"tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
        if (const auto text = token_text(key, false)) {
            const auto id = static_cast<int32_t>(file.get_int(key));
            if (std::find(out.stop_ids.begin(), out.stop_ids.end(), id) == out.stop_ids.end())
                out.stop_ids.push_back(id);
        }
    }
    return out;
}

std::string chat_template(const gguf::File& file) {
    for (const gguf::Value& v : file.metadata()) {
        if (starts_with(v.key(), "tokenizer.chat_template.") || v.key() == "tokenizer.chat_templates") {
            log::warn("gguf: ", v.key(),
                      " (a named chat template) is not imported; the engine renders with the "
                      "default one, tokenizer.chat_template");
        }
    }
    const gguf::Value* t = file.find("tokenizer.chat_template");
    if (t == nullptr) {
        fail("GGUF '", file.path().string(), "' has no tokenizer.chat_template; the engine renders every prompt with ",
             "the model's chat template");
    }
    return std::string(t->as_string());
}

}  // namespace xs::gguf_import
