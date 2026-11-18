#pragma once

// A tokenizer shipped as transformers' "slow" files, vocab.json, merges.txt and tokenizer_config.json, without a
// tokenizer.json: the tokenizer.json AutoTokenizer builds from them when it loads the directory, by its
// tokenizer_config.json's tokenizer_class. Built here: Qwen2Tokenizer (Qwen2TokenizerFast reads the same), Qwen's
// byte-level BPE over Qwen2's Split regex (tokenizer_json.h); any other class is refused by name.

#include <filesystem>

#include <nlohmann/json.hpp>

namespace xs::tok {

// What save_pretrained would write for the tokenizer transformers loads from the slow files.
struct SlowTokenizerFiles {
    nlohmann::json tokenizer;  // tokenizer.json
    nlohmann::json config;     // tokenizer_config.json, with the class's defaults for the special tokens it leaves out
};

// merges.txt is read as tokenizers reads it: a merge per line by rank, two pieces separated by one space; lines that
// start with "#version" skipped, any other line refused by its number. The added tokens are added_tokens_decoder's,
// with its flags: transformers adds them in id order, each at its vocabulary piece's id or the next free one, and a
// key that is not that id is refused. Every token tokenizer_config.json names special (the *_token keys, with the
// class's defaults, and extra_special_tokens or additional_special_tokens) is an added special token with no other
// flag, as Qwen2Tokenizer adds it; one neither added nor a piece of vocab.json is refused, where transformers would
// give it a new id. Also refused: add_bos_token, add_eos_token and clean_up_tokenization_spaces set.
SlowTokenizerFiles read_slow_files(const std::filesystem::path& dir, const nlohmann::json& tokenizer_config);

}  // namespace xs::tok
