#pragma once

// What the real-file GGUF checks share: the file (XS_GGUF_FILE) and the Hugging Face checkpoint directory it was
// converted from, the tokenizer files the import writes from it, and comparisons with the checkpoint's.

#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string_view>

#include "convert/gguf/tokenizer.h"
#include "io/gguf.h"
#include "models/qwen3_5/qwen3_5.h"
#include "tokenizer/tokenizer.h"

namespace xs::test {

struct RealGguf {
    std::filesystem::path gguf;
    std::filesystem::path hf;
};

// XS_GGUF_FILE and the directory in `dir_variable`, when both are set and the file's general.architecture is
// `architecture`; nullopt otherwise (the checks then skip).
inline std::optional<RealGguf> real_gguf(const char* dir_variable, std::string_view architecture) {
    const char* g = std::getenv("XS_GGUF_FILE");
    const char* h = std::getenv(dir_variable);
    if (g == nullptr || *g == '\0' || h == nullptr || *h == '\0') return std::nullopt;
    if (gguf::File(g).get_string("general.architecture") != architecture) return std::nullopt;
    return RealGguf{g, h};
}

// The tokenizer files the import writes from `file`, into `dir`.
inline void write_gguf_tokenizer(const gguf::File& file, const std::filesystem::path& dir) {
    const gguf_import::TokenizerFiles t = gguf_import::tokenizer_files(file);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "tokenizer.json") << t.tokenizer.dump();
    std::ofstream(dir / "tokenizer_config.json") << t.config.dump(1);
}

// What the Qwen3.5 decoder's code reads of a config from a GGUF file (`a`) and of its checkpoint's (`b`): the same,
// but the value heads' order (a GGUF file's are tiled) and the MTP head (which the caller checks).
inline void check_decoder_config(const qwen3_5::Config& a, const qwen3_5::Config& b) {
    CHECK(a.vocab_size == b.vocab_size);
    CHECK(a.hidden_size == b.hidden_size);
    CHECK(a.intermediate_size == b.intermediate_size);
    CHECK(a.num_experts == b.num_experts);
    CHECK(a.num_layers == b.num_layers);
    CHECK(static_cast<float>(a.rms_norm_eps) == static_cast<float>(b.rms_norm_eps));
    CHECK(a.rms_norm_eps == b.rms_norm_eps);
    CHECK(a.max_position == b.max_position);
    CHECK(a.layer_kinds == b.layer_kinds);
    CHECK(a.num_heads == b.num_heads);
    CHECK(a.num_kv_heads == b.num_kv_heads);
    CHECK(a.head_dim == b.head_dim);
    CHECK(a.attn_output_gate == b.attn_output_gate);
    CHECK(a.rope_theta == b.rope_theta);
    CHECK(a.rotary_dim == b.rotary_dim);
    CHECK(a.linear_key_heads == b.linear_key_heads);
    CHECK(a.linear_value_heads == b.linear_value_heads);
    CHECK(a.linear_key_dim == b.linear_key_dim);
    CHECK(a.linear_value_dim == b.linear_value_dim);
    CHECK(a.conv_kernel == b.conv_kernel);
    CHECK(a.value_head_order == qwen3_5::ValueHeadOrder::Tiled);
    CHECK(b.value_head_order == qwen3_5::ValueHeadOrder::Grouped);
}

// The ids whose text or special flag differ between two tokenizers of one vocabulary size.
inline int differing_tokens(const tok::Tokenizer& a, const tok::Tokenizer& b) {
    REQUIRE(a.vocab_size() == b.vocab_size());
    int differ = 0;
    for (int32_t id = 0; id < b.vocab_size(); ++id) {
        if (a.token_text(id) != b.token_text(id) || a.is_special(id) != b.is_special(id)) ++differ;
    }
    return differ;
}

}  // namespace xs::test
