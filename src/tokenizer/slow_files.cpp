#include "tokenizer/slow_files.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizer/json_fields.h"
#include "tokenizer/regex_split.h"
#include "tokenizer/tokenizer_json.h"
#include "util/error.h"
#include "util/mmap_file.h"

namespace xs::tok {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;

// Qwen2Tokenizer's own values of the special tokens tokenizer_config.json leaves out (one it writes as null is none).
constexpr std::pair<const char*, const char*> kQwen2Defaults[] = {
    {"unk_token", "<|endoftext|>"}, {"eos_token", "<|endoftext|>"}, {"pad_token", "<|endoftext|>"}};

// A token as tokenizer_config.json writes it: "<x>", or an AddedToken {"content": "<x>", ...}.
std::optional<std::string> token_content(const json& value) {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_object() && value.contains("content") && value.at("content").is_string()) {
        return value.at("content").get<std::string>();
    }
    return std::nullopt;
}

bool is_set(const json& config, const char* key) {
    const auto it = config.find(key);
    return it != config.end() && it->is_boolean() && it->get<bool>();
}

json read_merges(const fs::path& path) {
    const MappedFile file(path);
    const std::string_view text(reinterpret_cast<const char*>(file.data()), file.size());
    json merges = json::array();
    size_t number = 0;
    for (size_t start = 0; start < text.size();) {
        size_t end = text.find('\n', start);
        if (end == std::string_view::npos) end = text.size();
        std::string_view line = text.substr(start, end - start);
        start = end + 1;
        ++number;
        if (line.ends_with('\r')) line.remove_suffix(1);
        if (line.starts_with("#version")) continue;
        const size_t space = line.find(' ');
        if (space == 0 || space == std::string_view::npos || space + 1 == line.size() ||
            line.find(' ', space + 1) != std::string_view::npos) {
            fail("tokenizer: '", path.string(), "' line ", number, " is not two pieces separated by one space");
        }
        merges.push_back(std::string(line));
    }
    return merges;
}

// The tokens transformers counts special (all_special_tokens): the *_token keys' and extra_special_tokens' (a list,
// or names to tokens), or additional_special_tokens' when extra_special_tokens is no list of them.
std::vector<std::string> special_tokens(const json& config) {
    std::vector<std::string> out;
    const auto add = [&](const json& value) {
        const auto content = token_content(value);
        if (content && std::find(out.begin(), out.end(), *content) == out.end()) out.push_back(*content);
    };
    for (const auto& [key, value] : config.items()) {
        if (key.ends_with("_token")) add(value);
    }
    const json extra = config.value("extra_special_tokens", json());
    if (extra.is_object()) {
        for (const auto& [name, value] : extra.items()) add(value);
    }
    const json list = extra.is_array() && !extra.empty() ? extra : config.value("additional_special_tokens", json());
    if (list.is_array()) {
        for (const json& value : list) add(value);
    }
    return out;
}

int64_t token_id(const std::string& key, const fs::path& config_path) {
    try {
        size_t used = 0;
        const int64_t id = std::stoll(key, &used);
        if (used == key.size() && id >= 0) return id;
    } catch (const std::exception&) {
    }
    fail("tokenizer: '", config_path.string(), "' added_tokens_decoder key '", key, "' is not a token id");
}

// tokenizer.json added_tokens, by id: added_tokens_decoder's entries, then the special tokens as Qwen2Tokenizer adds
// them (AddedToken(content, special=True)).
json added_tokens(const json& config, const json& vocab, const fs::path& config_path) {
    std::map<int64_t, json> by_id;
    const json decoder = config.value("added_tokens_decoder", json::object());
    if (!decoder.is_object()) fail("tokenizer: '", config_path.string(), "' added_tokens_decoder is not an object");
    for (const auto& [key, entry] : decoder.items()) {
        if (!entry.is_object() || !entry.contains("content") || !entry.at("content").is_string()) {
            fail("tokenizer: '", config_path.string(), "' added_tokens_decoder entry ", key, " has no string content");
        }
        const int64_t id = token_id(key, config_path);
        json token = entry;
        token["id"] = id;
        by_id.emplace(id, std::move(token));
    }

    // tokenizers gives an added token its vocabulary piece's id, else the next id past the vocabulary and the added
    // tokens before it.
    const auto vocab_id = [&](const std::string& content) -> std::optional<int64_t> {
        const auto it = vocab.find(content);
        if (it == vocab.end() || !it->is_number_integer()) return std::nullopt;
        return it->get<int64_t>();
    };
    std::unordered_map<std::string, int64_t> added;
    int64_t next = static_cast<int64_t>(vocab.size());
    for (const auto& [id, token] : by_id) {
        const std::string content = token.at("content").get<std::string>();
        const int64_t expected = vocab_id(content).value_or(next);
        if (id != expected) {
            fail("tokenizer: '", config_path.string(), "' added_tokens_decoder gives '", content, "' id ", id,
                 ", where transformers adds it as id ", expected);
        }
        if (!added.emplace(content, id).second) {
            fail("tokenizer: '", config_path.string(), "' added_tokens_decoder lists '", content, "' twice");
        }
        next = std::max(next, id + 1);
    }

    for (const std::string& content : special_tokens(config)) {
        const auto it = added.find(content);
        const std::optional<int64_t> id = it != added.end() ? std::optional(it->second) : vocab_id(content);
        if (!id) {
            fail("tokenizer: '", config_path.string(), "' names '", content,
                 "' a special token, which neither its added_tokens_decoder nor vocab.json has");
        }
        by_id[*id] = {{"id", *id},       {"content", content}, {"single_word", false}, {"lstrip", false},
                      {"rstrip", false}, {"normalized", false}, {"special", true}};
    }

    json out = json::array();
    for (auto& [id, token] : by_id) out.push_back(std::move(token));
    return out;
}

}  // namespace

SlowTokenizerFiles read_slow_files(const fs::path& dir, const json& tokenizer_config) {
    const json tokenizer_class =
        tokenizer_config.is_object() ? tokenizer_config.value("tokenizer_class", json()) : json();
    const std::string name = tokenizer_class.is_string() ? tokenizer_class.get<std::string>() : "";
    if (name != "Qwen2Tokenizer" && name != "Qwen2TokenizerFast") {
        fail("tokenizer: ", dir.string(), " has no tokenizer.json, and its tokenizer_class ",
             name.empty() ? "(none in a tokenizer_config.json)" : name,
             " is not one xpu-serve builds from vocab.json and merges.txt (Qwen2Tokenizer is)");
    }
    const fs::path config_path = dir / "tokenizer_config.json";
    for (const char* key : {"add_bos_token", "add_eos_token", "clean_up_tokenization_spaces"}) {
        if (is_set(tokenizer_config, key)) {
            fail("tokenizer: '", config_path.string(), "' sets ", key, ", which a ", name,
                 " built from vocab.json and merges.txt does not support");
        }
    }
    for (const char* file : {"vocab.json", "merges.txt"}) {
        if (!fs::exists(dir / file)) {
            fail("tokenizer: ", dir.string(), " has no tokenizer.json, and no ", file, " for its ", name);
        }
    }

    SlowTokenizerFiles out;
    out.config = tokenizer_config;
    for (const auto& [key, value] : kQwen2Defaults) {
        if (!out.config.contains(key)) out.config[key] = value;
    }
    json vocab = read_json_file(dir / "vocab.json");
    if (!vocab.is_object()) fail("tokenizer: '", (dir / "vocab.json").string(), "' is not an object of pieces and ids");
    json added = added_tokens(out.config, vocab, config_path);
    const json prefix_space = out.config.value("add_prefix_space", json());
    if (!prefix_space.is_null() && !prefix_space.is_boolean()) {
        fail("tokenizer: '", config_path.string(), "' add_prefix_space must be a boolean");
    }
    const bool add_prefix_space = prefix_space.is_boolean() && prefix_space.get<bool>();
    out.tokenizer = make_tokenizer_json(qwen_byte_level_pipeline(split_regex::kQwen2, add_prefix_space),
                                        std::move(vocab), read_merges(dir / "merges.txt"), std::move(added));
    return out;
}

}  // namespace xs::tok
