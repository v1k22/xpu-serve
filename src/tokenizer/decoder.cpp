#include "tokenizer/decoder.h"

#include <optional>
#include <utility>

#include "tokenizer/byte_level.h"
#include "tokenizer/json_fields.h"
#include "tokenizer/normalizer.h"
#include "tokenizer/utf8.h"
#include "util/error.h"

namespace xs::tok {
namespace {
constexpr std::string_view kWhat = "decoder";
constexpr std::string_view kReplacementChar = "\xEF\xBF\xBD";

class ReplaceDecoder final : public Decoder {
public:
    ReplaceDecoder(std::string from, std::string to) : from_(std::move(from)), to_(std::move(to)) {}
    void decode(std::vector<std::string>& tokens) const override {
        for (auto& token : tokens) token = replace_all(token, from_, to_);
    }

private:
    std::string from_;
    std::string to_;
};

// "<0xAB>" -> 0xAB; anything else -> nullopt.
std::optional<uint8_t> byte_token_value(const std::string& token) {
    if (token.size() != 6 || token.compare(0, 3, "<0x") != 0 || token[5] != '>') return std::nullopt;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    const int hi = hex(token[3]);
    const int lo = hex(token[4]);
    if (hi < 0 || lo < 0) return std::nullopt;
    return static_cast<uint8_t>(hi * 16 + lo);
}

// A run of byte tokens becomes its UTF-8 text when the whole run is valid; otherwise every byte becomes U+FFFD
// (Hugging Face's ByteFallback is all-or-nothing per run, not a lossy decode).
class ByteFallbackDecoder final : public Decoder {
public:
    void decode(std::vector<std::string>& tokens) const override {
        std::vector<std::string> out;
        out.reserve(tokens.size());
        std::string run;
        size_t run_tokens = 0;
        auto flush_run = [&] {
            if (run_tokens == 0) return;
            if (utf8::is_valid(run)) {
                out.push_back(std::move(run));
            } else {
                for (size_t i = 0; i < run_tokens; ++i) out.emplace_back(kReplacementChar);
            }
            run.clear();
            run_tokens = 0;
        };
        for (auto& token : tokens) {
            if (const auto byte = byte_token_value(token)) {
                run.push_back(static_cast<char>(*byte));
                ++run_tokens;
                continue;
            }
            flush_run();
            out.push_back(std::move(token));
        }
        flush_run();
        tokens = std::move(out);
    }
};

// Maps each token's characters back to bytes (a token with a character outside the alphabet keeps its own UTF-8
// bytes, as in Hugging Face) and joins everything into one string, invalid sequences becoming U+FFFD.
class ByteLevelDecoder final : public Decoder {
public:
    void decode(std::vector<std::string>& tokens) const override {
        std::string bytes;
        for (const auto& token : tokens) {
            const size_t start = bytes.size();
            bool mapped = true;
            for (size_t pos = 0; pos < token.size() && mapped;) {
                const size_t len = utf8::valid_char_length(token, pos);
                const auto byte = len == 0 ? std::nullopt : byte_level::char_to_byte(utf8::decode_at(token, pos));
                if (byte) bytes.push_back(static_cast<char>(*byte));
                mapped = byte.has_value();
                pos += len;
            }
            if (!mapped) {
                bytes.resize(start);
                bytes += token;
            }
        }
        tokens.assign(1, utf8::lossy(bytes));
    }
};

class FuseDecoder final : public Decoder {
public:
    void decode(std::vector<std::string>& tokens) const override {
        std::string fused;
        for (const auto& token : tokens) fused += token;
        tokens.assign(1, std::move(fused));
    }
};

// Removes up to `start` leading and `stop` trailing copies of one character from each token.
class StripDecoder final : public Decoder {
public:
    StripDecoder(std::string content, size_t start, size_t stop)
        : content_(std::move(content)), start_(start), stop_(stop) {}
    void decode(std::vector<std::string>& tokens) const override {
        for (auto& token : tokens) {
            std::string_view view = token;
            for (size_t i = 0; i < start_ && view.starts_with(content_); ++i) view.remove_prefix(content_.size());
            for (size_t i = 0; i < stop_ && view.ends_with(content_); ++i) view.remove_suffix(content_.size());
            token = std::string(view);
        }
    }

private:
    std::string content_;
    size_t start_;
    size_t stop_;
};

// Hugging Face's WordPiece decoder: each token after the first loses the continuing-subword prefix, or else gains a
// leading space. Cleanup removes the spaces tokenization put before punctuation and contractions, token by token as
// Hugging Face does (so " do not" changes only within one token).
class WordPieceDecoder final : public Decoder {
public:
    WordPieceDecoder(std::string prefix, bool cleanup) : prefix_(std::move(prefix)), cleanup_(cleanup) {}
    void decode(std::vector<std::string>& tokens) const override {
        static constexpr std::pair<std::string_view, std::string_view> kCleanups[] = {
            {" .", "."},   {" ?", "?"},           {" !", "!"},   {" ,", ","},     {" ' ", "'"},   {" n't", "n't"},
            {" 'm", "'m"}, {" do not", " don't"}, {" 's", "'s"}, {" 've", "'ve"}, {" 're", "'re"},
        };
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string& token = tokens[i];
            if (i != 0) {
                if (token.starts_with(prefix_)) {
                    token.erase(0, prefix_.size());
                } else {
                    token.insert(0, 1, ' ');
                }
            }
            if (!cleanup_) continue;
            for (const auto& [from, to] : kCleanups) token = replace_all(token, from, to);
        }
    }

private:
    std::string prefix_;
    bool cleanup_;
};

class SequenceDecoder final : public Decoder {
public:
    explicit SequenceDecoder(std::vector<std::unique_ptr<Decoder>> steps) : steps_(std::move(steps)) {}
    void decode(std::vector<std::string>& tokens) const override {
        for (const auto& step : steps_) step->decode(tokens);
    }

private:
    std::vector<std::unique_ptr<Decoder>> steps_;
};

size_t count_field(const nlohmann::json& config, const char* key) {
    if (!config.contains(key) || !config.at(key).is_number_unsigned()) {
        fail("tokenizer.json decoder 'Strip': field '", key, "' must be a non-negative integer");
    }
    return config.at(key).get<size_t>();
}

std::unique_ptr<Decoder> make_strip(const nlohmann::json& config) {
    std::string content = string_field(config, "content", kWhat);
    if (content.empty() || utf8::valid_char_length(content, 0) != content.size()) {
        fail("tokenizer.json decoder 'Strip': 'content' must be exactly one character");
    }
    const size_t start = count_field(config, "start");
    const size_t stop = count_field(config, "stop");
    return std::make_unique<StripDecoder>(std::move(content), start, stop);
}

std::unique_ptr<Decoder> make_sequence(const nlohmann::json& config) {
    if (!config.contains("decoders") || !config.at("decoders").is_array()) {
        fail("tokenizer.json decoder 'Sequence' has no 'decoders' list");
    }
    std::vector<std::unique_ptr<Decoder>> steps;
    for (const auto& step : config.at("decoders")) {
        auto decoder = make_decoder(step);
        if (decoder) steps.push_back(std::move(decoder));
    }
    return std::make_unique<SequenceDecoder>(std::move(steps));
}
}  // namespace

std::unique_ptr<Decoder> make_decoder(const nlohmann::json& config) {
    if (config.is_null()) return nullptr;
    const std::string type = component_type(config, kWhat);
    if (type == "Replace") {
        return std::make_unique<ReplaceDecoder>(string_pattern(config, kWhat), string_field(config, "content", kWhat));
    }
    if (type == "ByteFallback") return std::make_unique<ByteFallbackDecoder>();
    if (type == "Fuse") return std::make_unique<FuseDecoder>();
    if (type == "ByteLevel") return std::make_unique<ByteLevelDecoder>();
    if (type == "Strip") return make_strip(config);
    if (type == "WordPiece") {
        return std::make_unique<WordPieceDecoder>(string_field(config, "prefix", kWhat),
                                                  bool_field(config, "cleanup", kWhat));
    }
    if (type == "Sequence") return make_sequence(config);
    refuse_component(kWhat, type);
}

}  // namespace xs::tok
