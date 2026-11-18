#pragma once

// A tokenizer.json built in memory, for a tokenizer that does not come as one: a checkpoint's slow files
// (slow_files.h) and a GGUF file's vocabulary (convert/gguf/tokenizer.h). Tokenizer::load reads what this builds as it
// reads a tokenizer.json file.

#include <string_view>

#include <nlohmann/json.hpp>

namespace xs::tok {

// What a tokenizer.json holds besides its vocabulary, merges and added tokens.
struct TokenizerPipeline {
    nlohmann::json normalizer, pre_tokenizer, post_processor, decoder;
    nlohmann::json model;  // the model's settings, without "vocab" and "merges"
};

// Qwen's byte-level BPE, as transformers' Qwen2Converter builds it from a Qwen2Tokenizer: NFC; a Split (Isolated) by
// `split_regex` (one of regex_split.h's) then ByteLevel without the GPT-2 regex; BPE without unk, byte fallback or
// ignore_merges; ByteLevel post-processor and decoder. ByteLevel's other flags are written as Qwen's own tokenizer.json
// files write them: the converter's (trim_offsets, the decoder's) move offsets only, never ids or text.
TokenizerPipeline qwen_byte_level_pipeline(std::string_view split_regex, bool add_prefix_space = false);

// The tokenizer.json of `pipeline` over `vocab` (piece -> id), `merges` (by rank: "left right" or [left, right]) and
// `added_tokens` (tokenizer.json's entries), without truncation or padding.
nlohmann::json make_tokenizer_json(TokenizerPipeline pipeline, nlohmann::json vocab, nlohmann::json merges,
                                   nlohmann::json added_tokens);

}  // namespace xs::tok
