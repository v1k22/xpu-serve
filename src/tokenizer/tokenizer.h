#pragma once

// Hugging Face tokenizer.json tokenizers, implemented natively (no Python, no Rust).
// Scope: BPE models as used by Gemma (Replace normalizer, Split or Metaspace pre-tokenizer, byte fallback, fuse_unk),
// byte-level BPE as used by Qwen (NFC, regex Split, ByteLevel) and ModernBERT (ByteLevel with the GPT-2 regex), and
// WordPiece as used by BERT-family encoders (BertNormalizer, BertPreTokenizer, WordPiece decoder), with added/special
// tokens (normalized ones matched in normalized text) and TemplateProcessing post-processors.
// Unsupported components are refused by name. tokenizer.json's truncation and padding are not applied: callers own
// sequence lengths.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace xs::tok {

class Tokenizer {
public:
    // Loads <dir>/tokenizer.json (and tokenizer_config.json when present, for bos/eos and special-token names).
    static std::unique_ptr<Tokenizer> load(const std::filesystem::path& model_dir);
    virtual ~Tokenizer() = default;

    // Encodes text. With parse_special, added tokens written in the text (e.g. "<|turn>") become their ids; without
    // it they are tokenized as plain text. Never adds bos/eos (chat templates do that).
    virtual std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const = 0;
    // The ids with the special tokens tokenizer.json's post-processor puts around one sequence (TemplateProcessing's
    // `single` template, e.g. [CLS] ... [SEP]); unchanged when the post-processor adds none.
    // with_special_tokens(encode(text)) is Hugging Face's encode(text, add_special_tokens=True).
    virtual std::vector<int32_t> with_special_tokens(std::vector<int32_t> ids) const = 0;
    // Decodes ids to UTF-8 (invalid byte sequences become U+FFFD). skip_special drops special tokens.
    virtual std::string decode(std::span<const int32_t> ids, bool skip_special = false) const = 0;

    virtual int32_t vocab_size() const = 0;
    // The vocab piece, verbatim; for an added token matched in normalized text, its content normalized (as Hugging
    // Face's id_to_token gives it, while token_id takes the content).
    virtual std::string token_text(int32_t id) const = 0;
    virtual std::optional<int32_t> token_id(std::string_view piece) const = 0;
    virtual bool is_special(int32_t id) const = 0;
    virtual std::optional<int32_t> bos_id() const = 0;
    virtual std::optional<int32_t> eos_id() const = 0;
};

// Incremental detokenizer: turns a token stream into UTF-8 text deltas without ever splitting a multi-byte character
// (byte-fallback and byte-level tokens may carry partial characters).
class StreamDecoder {
public:
    explicit StreamDecoder(const Tokenizer& tokenizer, bool skip_special = false);
    // Returns the text that became final with this token (may be empty).
    std::string push(int32_t id);
    // Returns whatever is still pending (incomplete bytes become U+FFFD).
    std::string flush();

private:
    const Tokenizer& tokenizer_;
    bool skip_special_;
    std::vector<int32_t> context_;  // the last emitted chunk, decoded again for decoders that look back
    std::vector<int32_t> pending_;  // tokens whose text is not final yet
    size_t emitted_ = 0;            // bytes of the pending tokens' text already returned
};

}  // namespace xs::tok
