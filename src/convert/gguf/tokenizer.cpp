#include "convert/gguf/tokenizer.h"

#include <algorithm>
#include <array>
#include <optional>
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

// A tokenizer llama.cpp's converter writes: the parts of the original tokenizer.json the file does not hold, and how
// the file's token types read back as the original's vocabulary and added tokens.
struct TokenizerKind {
    std::string_view model;  // tokenizer.ggml.model
    std::string_view pre;    // tokenizer.ggml.pre; empty when the model implies it (llama.cpp then ignores the key)
    std::string_view normalizer, pre_tokenizer, post_processor, decoder;  // JSON
    std::string_view bpe;  // the BPE model's settings (JSON, without vocab and merges)
    bool nfc;              // the normalizer is NFC; a file's tokenizer.ggml.normalizer.nfc, when present, must agree
    // Added tokens are also pieces of the BPE vocabulary.
    bool added_in_vocab;
    // Control tokens the original does not mark special.
    std::vector<std::string_view> control_not_special;
    // User-defined tokens the original marks special.
    std::vector<std::string_view> user_defined_special;
    // Normal tokens the original adds as special tokens.
    std::vector<std::string_view> normal_special;
    // Tokens that end a reply besides the file's eos (and eot, eom) token: those llama.cpp ends generation at by text
    // for this tokenizer, which the original names in generation_config.json.
    std::vector<std::string_view> end_of_generation;
    // When not empty: the only control tokens the original marks special (control_not_special is then unused).
    std::vector<std::string_view> special_controls;
    // Normal tokens the original adds as tokens that are not special; null: none.
    bool (*normal_added)(std::string_view text) = nullptr;
    // Added tokens the original strips the spaces before (lstrip): a BERT-style mask token.
    std::vector<std::string_view> lstrip;
    // The original's added tokens that are not special are normalized: matched in the normalized text, after the
    // special ones (ModernBERT's runs of spaces then never take the spaces its mask token strips).
    bool plain_added_normalized = false;
    // When not empty: the post-processor is a BERT-style template, the file's bos token (its [CLS]) and separator
    // around a sequence, whose special tokens are the ones of these tokenizer.ggml.<name>_token_id keys; the
    // post_processor above is then unused, and tokenizer_config.json names the bos token cls_token too.
    std::vector<std::string_view> bert_template;

    std::string name() const { return std::string(pre.empty() ? model : pre); }
};

bool listed(const std::vector<std::string_view>& list, std::string_view text) {
    return std::find(list.begin(), list.end(), text) != list.end();
}

// Two or more U+2581: the runs of spaces mmBERT's tokenizer adds as tokens, which the converter types normal.
bool metaspace_run(std::string_view text) {
    constexpr std::string_view kSpace = "▁";
    if (text.size() < 2 * kSpace.size() || text.size() % kSpace.size() != 0) return false;
    for (size_t i = 0; i < text.size(); i += kSpace.size()) {
        if (text.substr(i, kSpace.size()) != kSpace) return false;
    }
    return true;
}

const std::vector<TokenizerKind>& tokenizer_kinds() {
    static const std::vector<TokenizerKind> table = {
        // Qwen3.5, Qwen3.6, Qwen3.8: Qwen/Qwen3.5-4B's tokenizer.json. The converter types every added token that
        // looks like <|...|> control.
        {"gpt2",
         "qwen35",
         R"json({"type": "NFC"})json",
         R"json({"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]})json",
         kByteLevel,
         kByteLevel,
         R"json({"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false})json",
         true,
         false,
         {"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"},
         {},
         {},
         {}},
        // Gemma 4: google/gemma-4-E2B-it's tokenizer.json (BPE over the text with spaces as U+2581, byte fallback;
        // every id in model.vocab, the added tokens included). The converter (Gemma4Model.set_vocab) types the
        // markers a chat parser reads user-defined so that llama.cpp prints them, and types control only the tokens
        // the publisher's tokenizer_config.json names: <eos> is typed normal when it names <turn|> as eos_token.
        // llama.cpp ends generation at <eos>, <turn|> and <|tool_response>, the original's generation_config.json
        // eos_token_id.
        {"gemma4",
         "",
         R"json({"type": "Replace", "pattern": {"String": " "}, "content": "▁"})json",
         R"json({"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false})json",
         R"json({"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}], "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}], "special_tokens": {}})json",
         R"json({"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]})json",
         R"json({"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false})json",
         false,
         true,
         {},
         {"<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|\"|>"},
         {"<eos>"},
         {"<eos>", "<turn|>", "<|tool_response>"}},
        // ModernBERT (answerdotai/ModernBERT-large's tokenizer.json, Laya's): NFC, ByteLevel with the GPT-2 regex,
        // byte-level BPE; [CLS] ... [SEP] around a sequence. Its added tokens below id 50280 are also vocabulary
        // pieces, those above are not; as added tokens only, they encode alike.
        {.model = "gpt2",
         .pre = "modern-bert",
         .normalizer = R"json({"type": "NFC"})json",
         .pre_tokenizer =
             R"json({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true})json",
         .post_processor = "",
         .decoder =
             R"json({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true})json",
         .bpe =
             R"json({"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false})json",
         .nfc = true,
         .added_in_vocab = false,
         .lstrip = {"[MASK]"},
         .plain_added_normalized = true,
         .bert_template = {"bos", "mask", "padding", "seperator", "unknown"}},
        // mmBERT (jhu-clsp/mmBERT's tokenizer.json, Julia-1's and Laya multilingual's): Gemma's vocabulary, spaces as
        // U+2581 and a Metaspace split, BPE with byte fallback; <bos> ... <eos> around a sequence. The converter types
        // the byte pieces and the runs of U+2581 (added tokens in the original) normal, and the unused placeholders
        // control like the seven special tokens.
        {.model = "gpt2",
         .pre = "mmbert",
         .normalizer = R"json({"type": "Replace", "pattern": {"String": " "}, "content": "▁"})json",
         .pre_tokenizer =
             R"json({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true})json",
         .post_processor = "",
         .decoder =
             R"json({"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]})json",
         .bpe =
             R"json({"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false})json",
         .nfc = false,
         .added_in_vocab = true,
         .special_controls = {"<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<start_of_turn>", "<end_of_turn>"},
         .normal_added = metaspace_run,
         .lstrip = {"<mask>"},
         .bert_template = {"bos", "seperator"}},
    };
    return table;
}

// The table's tokenizer models, or its pre-tokenizer names, as a list for an error.
std::string supported(bool models) {
    std::vector<std::string_view> names;
    for (const TokenizerKind& k : tokenizer_kinds()) {
        const std::string_view name = models ? k.model : k.pre;
        if (!name.empty() && !listed(names, name)) names.push_back(name);
    }
    std::string out;
    for (const std::string_view name : names)
        out += str_cat(out.empty() ? "" : ", ", name);
    return out;
}

// The kind of the file's tokenizer: by tokenizer.ggml.model, and for a model that does not imply its pre-tokenizer,
// by tokenizer.ggml.pre.
const TokenizerKind& find_kind(const gguf::File& file) {
    const std::string_view model = file.get_string("tokenizer.ggml.model");
    const gguf::Value* pre = file.find("tokenizer.ggml.pre");
    bool model_known = false;
    for (const TokenizerKind& k : tokenizer_kinds()) {
        if (k.model != model) continue;
        model_known = true;
        if (k.pre.empty()) {
            if (pre != nullptr && pre->as_string() != model) {
                fail("GGUF tokenizer.ggml.pre '", pre->as_string(), "' disagrees with tokenizer.ggml.model '", model,
                     "', which implies its pre-tokenizer");
            }
            return k;
        }
        if (pre == nullptr)
            fail("GGUF tokenizer.ggml.model '", model, "' needs a tokenizer.ggml.pre; the file has none");
        if (k.pre == pre->as_string()) return k;
    }
    if (!model_known) {
        fail("GGUF tokenizer.ggml.model '", model, "' is not supported (supported: ", supported(true), ")");
    }
    fail("GGUF tokenizer.ggml.pre '", pre->as_string(), "' is not supported (supported: ", supported(false),
         "); the file does not hold its pre-tokenizer, only that name");
}

// tokenizer.ggml.* keys the import reads or knows to change nothing it writes; any other is refused by name.
// (token_type_count is a model's number of token types, which its architecture checks.)
constexpr std::array<std::string_view, 33> kKnownKeys = {
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
    "add_space_prefix",
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
    "token_type_count",
    "metaspace.prepend_scheme",
};

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.substr(0, prefix.size()) == prefix;
}

void check_keys(const gguf::File& file, const TokenizerKind& kind) {
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
            fail("GGUF metadata '", key, "' is not read by this importer (a tokenizer it does not read)");
        }
    }
    const auto agree = [&](std::string_view key, bool table) {
        if (const gguf::Value* v = file.find(key); v != nullptr && v->as_bool() != table) {
            fail("GGUF ", key, " is ", v->as_bool() ? "true" : "false", ", which disagrees with the tokenizer '",
                 kind.name(), "' stands for");
        }
    };
    agree("tokenizer.ggml.normalizer.nfc", kind.nfc);
    agree("tokenizer.ggml.normalizer.lowercase", false);
    agree("tokenizer.ggml.normalizer.strip_accents", false);
    agree("tokenizer.ggml.add_space_prefix", false);
    if (const gguf::Value* v = file.find("tokenizer.ggml.metaspace.prepend_scheme")) {
        const nlohmann::json pre = nlohmann::json::parse(kind.pre_tokenizer);
        if (pre.value("type", "") != "Metaspace" || pre.value("prepend_scheme", "") != v->as_string()) {
            fail("GGUF tokenizer.ggml.metaspace.prepend_scheme '", v->as_string(), "' disagrees with the tokenizer '",
                 kind.name(), "' stands for");
        }
    }
}

// "<0xXX>": a byte fallback piece.
bool is_byte_piece(std::string_view text) {
    const auto hex = [](char c) { return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f'); };
    return text.size() == 6 && starts_with(text, "<0x") && hex(text[3]) && hex(text[4]) && text[5] == '>';
}

nlohmann::json parse(std::string_view text) {
    return nlohmann::json::parse(text);
}

}  // namespace

TokenizerFiles tokenizer_files(const gguf::File& file) {
    const TokenizerKind& kind = find_kind(file);
    check_keys(file, kind);
    nlohmann::json bpe = parse(kind.bpe);

    const std::vector<std::string_view> tokens = file.get("tokenizer.ggml.tokens").strings();
    const std::vector<int64_t> types = file.get("tokenizer.ggml.token_type").ints();
    if (types.size() != tokens.size()) {
        fail("GGUF tokenizer.ggml.token_type has ", types.size(), " entries for ", tokens.size(), " tokens");
    }

    TokenizerFiles out;
    nlohmann::json vocab = nlohmann::json::object();
    nlohmann::json added = nlohmann::json::array();
    nlohmann::ordered_json decoder = nlohmann::ordered_json::object();
    std::vector<bool> present(tokens.size(), false);
    const auto add_piece = [&](const std::string& text, size_t id) {
        if (vocab.contains(text)) {
            fail("GGUF tokenizer: token '", text, "' is both id ", vocab.at(text).get<int64_t>(), " and ", id);
        }
        vocab[text] = id;
    };
    const auto add_token = [&](const std::string& text, size_t id, bool special) {
        const bool lstrip = listed(kind.lstrip, text);
        const bool normalized = !special && kind.plain_added_normalized;
        added.push_back({{"id", id},
                         {"content", text},
                         {"single_word", false},
                         {"lstrip", lstrip},
                         {"rstrip", false},
                         {"normalized", normalized},
                         {"special", special}});
        decoder[std::to_string(id)] = {{"content", text}, {"lstrip", lstrip},     {"normalized", normalized},
                                       {"rstrip", false}, {"single_word", false}, {"special", special}};
    };
    for (size_t id = 0; id < tokens.size(); ++id) {
        const std::string text(tokens[id]);
        switch (types[id]) {
            case kNormal:
                add_piece(text, id);
                if (listed(kind.normal_special, text)) {
                    add_token(text, id, true);
                } else if (kind.normal_added != nullptr && kind.normal_added(text)) {
                    add_token(text, id, false);
                }
                break;
            case kByte:
                if (!bpe.value("byte_fallback", false) || !is_byte_piece(text)) {
                    fail("GGUF tokenizer: token ", id, " '", text, "' has type byte, which the tokenizer '",
                         kind.name(), "' does not have");
                }
                add_piece(text, id);
                break;
            case kControl:
            case kUserDefined: {
                const bool control_special = kind.special_controls.empty() ? !listed(kind.control_not_special, text)
                                                                           : listed(kind.special_controls, text);
                const bool special =
                    types[id] == kControl ? control_special : listed(kind.user_defined_special, text);
                add_token(text, id, special);
                if (kind.added_in_vocab) add_piece(text, id);
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
                     ", which the tokenizer '", kind.name(), "' does not have");
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
    if (const auto unk = token_text("tokenizer.ggml.unknown_token_id", false);
        unk && bpe["unk_token"].is_string() && bpe["unk_token"] != *unk) {
        fail("GGUF tokenizer.ggml.unknown_token_id is '", *unk, "', which disagrees with the tokenizer '", kind.name(),
             "' stands for (", bpe["unk_token"].get<std::string>(), ")");
    }

    nlohmann::json post_processor;
    if (kind.bert_template.empty()) {
        post_processor = parse(kind.post_processor);
    } else {  // [CLS] A [SEP] and [CLS] A [SEP] B [SEP], every token of type 0, over the file's tokens
        const auto special = [&](std::string_view name) {
            const std::string key = str_cat("tokenizer.ggml.", name, "_token_id");
            const std::optional<std::string> text = token_text(key, false);
            if (!text) {
                fail("GGUF '", file.path().string(), "' names no ", key, ", which the tokenizer '", kind.name(),
                     "' has");
            }
            return std::pair{*text, file.get_int(key)};
        };
        const std::string cls = special("bos").first, sep = special("seperator").first;
        const auto piece = [](const char* kind_of, const std::string& id) {
            return nlohmann::json{{kind_of, {{"id", id}, {"type_id", 0}}}};
        };
        nlohmann::json tokens_by_text = nlohmann::json::object();
        for (const std::string_view name : kind.bert_template) {
            const auto [text, id] = special(name);
            tokens_by_text[text] = {{"id", text}, {"ids", {id}}, {"tokens", {text}}};
        }
        post_processor = {{"type", "TemplateProcessing"},
                          {"single", {piece("SpecialToken", cls), piece("Sequence", "A"), piece("SpecialToken", sep)}},
                          {"pair",
                           {piece("SpecialToken", cls), piece("Sequence", "A"), piece("SpecialToken", sep),
                            piece("Sequence", "B"), piece("SpecialToken", sep)}},
                          {"special_tokens", std::move(tokens_by_text)}};
    }
    out.tokenizer = {{"version", "1.0"},
                     {"truncation", nullptr},
                     {"padding", nullptr},
                     {"added_tokens", std::move(added)},
                     {"normalizer", parse(kind.normalizer)},
                     {"pre_tokenizer", parse(kind.pre_tokenizer)},
                     {"post_processor", std::move(post_processor)},
                     {"decoder", parse(kind.decoder)}};
    bpe["vocab"] = std::move(vocab);
    bpe["merges"] = std::move(merges);
    out.tokenizer["model"] = std::move(bpe);

    out.config["added_tokens_decoder"] = std::move(decoder);
    const auto bos = token_text("tokenizer.ggml.bos_token_id", false);
    out.config["bos_token"] = bos ? nlohmann::ordered_json(*bos) : nlohmann::ordered_json(nullptr);
    out.config["eos_token"] = *token_text("tokenizer.ggml.eos_token_id", true);
    if (!kind.bert_template.empty()) out.config["cls_token"] = out.config["bos_token"];
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

    const auto add_stop = [&](int32_t id) {
        if (std::find(out.stop_ids.begin(), out.stop_ids.end(), id) == out.stop_ids.end()) out.stop_ids.push_back(id);
    };
    add_stop(static_cast<int32_t>(file.get_int("tokenizer.ggml.eos_token_id")));
    for (const char* key : {"tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
        if (token_text(key, false)) add_stop(static_cast<int32_t>(file.get_int(key)));
    }
    for (const std::string_view text : kind.end_of_generation) {
        for (size_t id = 0; id < tokens.size(); ++id) {
            if (present[id] && tokens[id] == text) add_stop(static_cast<int32_t>(id));
        }
    }
    return out;
}

std::string chat_template(const gguf::File& file, const std::map<std::string, std::string, std::less<>>& unused) {
    constexpr std::string_view kNamed = "tokenizer.chat_template.";
    for (const gguf::Value& v : file.metadata()) {
        if (!starts_with(v.key(), kNamed)) continue;  // tokenizer.chat_templates only lists their names
        if (const auto it = unused.find(v.key().substr(kNamed.size())); it != unused.end()) {
            log::info("gguf: ", v.key(), " is not imported: ", it->second);
        } else {
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
