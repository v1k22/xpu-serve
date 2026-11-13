#pragma once

// A GGUF file's tokenizer as the Hugging Face files the engine reads (tokenizer.json, tokenizer_config.json), and its
// chat template.
//
// The file holds the vocabulary, each token's type, the merges and the special tokens' ids; the normalizer and
// pre-tokenizer are not in it, only a name for them (tokenizer.ggml.model, and for a byte-level BPE tokenizer
// tokenizer.ggml.pre), so they come from a table of the tokenizers this importer knows. Token types: normal and byte
// tokens are the BPE vocabulary; control tokens are added special tokens and user-defined ones added tokens that are
// not special (Qwen's <think> and <tool_call> stay non-special, so the output parser sees them), each with the
// exceptions the table names where the converter's typing departs from the original's (and the added tokens' lstrip
// and normalized flags it does not keep); unused ones are the converter's [PAD<id>] placeholders for ids the original
// tokenizer does not have, and are left out. A BERT-style post-processor ([CLS] ... [SEP]) is built from the file's
// own special tokens.

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/gguf.h"

namespace xs::gguf_import {

struct TokenizerFiles {
    nlohmann::json tokenizer;       // tokenizer.json
    nlohmann::ordered_json config;  // tokenizer_config.json: added_tokens_decoder and the named special tokens
    // config.json's eos_token_id: eos, then eot and eom when the file names them, then the tokens the table says end a
    // reply for this tokenizer.
    std::vector<int32_t> stop_ids;
};

// Refuses by name a tokenizer model or pre-tokenizer the table does not have, tokenizer.ggml.* keys it does not know
// or that disagree with the table, token types the tokenizer does not have, and special-token ids outside the
// vocabulary.
TokenizerFiles tokenizer_files(const gguf::File& file);

// tokenizer.chat_template; a named error when the file has none (the engine renders every prompt with it). Named
// templates (tokenizer.chat_template.<name>) are not imported: a warning for each, but for those `unused` gives the
// reason for (by name).
std::string chat_template(const gguf::File& file, const std::map<std::string, std::string, std::less<>>& unused = {});

}  // namespace xs::gguf_import
