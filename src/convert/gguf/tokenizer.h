#pragma once

// A GGUF file's tokenizer as the Hugging Face files the engine reads (tokenizer.json, tokenizer_config.json), and its
// chat template.
//
// The file holds the vocabulary, each token's type, the merges and the special tokens' ids; the normalizer and
// pre-tokenizer are not in it, only a name for them (tokenizer.ggml.pre), so they come from a table of the names this
// importer knows. Token types: normal tokens are the BPE vocabulary; control tokens are added special tokens and
// user-defined ones added tokens that are not special (Qwen's <think> and <tool_call> stay non-special, so the output
// parser sees them); unused ones are the converter's [PAD<id>] placeholders for ids the original tokenizer does not
// have, and are left out.

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/gguf.h"

namespace xs::gguf_import {

struct TokenizerFiles {
    nlohmann::json tokenizer;       // tokenizer.json
    nlohmann::ordered_json config;  // tokenizer_config.json: added_tokens_decoder and the named special tokens
    std::vector<int32_t> stop_ids;  // eos, then eot and eom when the file names them: config.json's eos_token_id
};

// Refuses by name a tokenizer model other than gpt2 (byte-level BPE), a tokenizer.ggml.pre it has no table entry for,
// tokenizer.ggml.* keys it does not know, token types gpt2 does not have, and special-token ids outside the
// vocabulary.
TokenizerFiles tokenizer_files(const gguf::File& file);

// tokenizer.chat_template; a named error when the file has none (the engine renders every prompt with it).
std::string chat_template(const gguf::File& file);

}  // namespace xs::gguf_import
