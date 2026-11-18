#pragma once

// Test doubles for the server: a byte tokenizer, a line-per-message chat template and a scripted engine.

#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat/chat_template.h"
#include "engine/service.h"
#include "image/image.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"

namespace xs::test {

// One token per byte (id = byte value).
class ByteTokenizer final : public tok::Tokenizer {
public:
    std::vector<int32_t> encode(std::string_view text, bool) const override {
        return {text.begin(), text.end()};
    }
    std::vector<int32_t> with_special_tokens(std::vector<int32_t> ids) const override { return ids; }
    std::string decode(std::span<const int32_t> ids, bool) const override {
        std::string out;
        for (int32_t id : ids) out.push_back(char(id));
        return out;
    }
    int32_t vocab_size() const override { return 256; }
    std::string token_text(int32_t id) const override { return std::string(1, char(id)); }
    std::optional<int32_t> token_id(std::string_view piece) const override {
        if (piece.size() != 1) return std::nullopt;
        return int32_t(static_cast<unsigned char>(piece[0]));
    }
    bool is_special(int32_t) const override { return false; }
    std::optional<int32_t> bos_id() const override { return std::nullopt; }
    std::optional<int32_t> eos_id() const override { return std::nullopt; }

    static std::vector<int32_t> tokens(std::string_view text) { return {text.begin(), text.end()}; }
};

// The byte LineTemplate renders for an image part, and FakeService's image placeholder token.
inline constexpr char kImageByte = '\x7f';

// "<role>: <content>\n" per message, then "[tools:N]" and "[think]" markers so tests can see what reached it.
// List content renders its text parts and kImageByte per image part. A message whose content is "BOOM" makes
// rendering fail.
class LineTemplate final : public chat::ChatTemplate {
public:
    std::string render(const nlohmann::json& messages, const nlohmann::json& tools, bool add_generation_prompt,
                       const nlohmann::json& extra) const override {
        std::string out;
        for (const auto& m : messages) {
            std::string content;
            if (m.contains("content") && m["content"].is_array()) {
                for (const auto& part : m["content"]) {
                    content += part["type"] == "image" ? std::string(1, kImageByte) : part.value("text", "");
                }
            } else {
                content = m.value("content", "");
            }
            if (content == "BOOM") fail("template exploded");
            out += m["role"].get<std::string>() + ": " + content + "\n";
        }
        if (!tools.is_null()) out += "[tools:" + std::to_string(tools.size()) + "]";
        if (extra.value("enable_thinking", false)) out += "[think]";
        if (add_generation_prompt) out += "assistant:";
        return out;
    }
};

// Replies with `reply` one byte-token at a time, one request at a time, like the real engine.
class FakeService final : public InferenceService {
public:
    FakeService() {
        card_.id = "fake-model";
        card_.arch = "fake";
        card_.dialect = "plain";
        card_.context_length = 4096;
        card_.default_sampling.temperature = 0.7f;
    }

    const ModelCard& model() const override { return card_; }
    const tok::Tokenizer& tokenizer() const override { return tokenizer_; }
    const chat::ChatTemplate& chat_template() const override { return template_; }

    GenerateResult generate(const GenerateRequest& request, const TokenCallback& on_token) override {
        std::lock_guard lock(serial_);
        last_request_ = request;
        ++calls_;
        GenerateResult result;
        result.stats.prompt_tokens = int(request.prompt.size());
        result.stats.prefill_ms = 5.0;
        if (!error.empty()) {
            result.finish = FinishReason::Error;
            result.error = error;
            return result;
        }
        for (int32_t token : ByteTokenizer::tokens(reply)) {
            if (int(result.tokens.size()) >= request.max_new_tokens) {
                result.finish = FinishReason::Length;
                break;
            }
            std::this_thread::sleep_for(token_delay);
            result.tokens.push_back(token);
            if (!on_token(token)) {
                result.finish = FinishReason::Cancelled;
                cancelled = true;
                break;
            }
        }
        result.stats.generated_tokens = int(result.tokens.size());
        result.stats.decode_ms = 10.0;
        return result;
    }

    bool accepts(Modality m) const override { return images && m == Modality::Image; }
    // Decodes for real (so bad images fail as they would), then stands in two soft tokens per image.
    MediaInput prepare_media(Modality m, std::span<const uint8_t> encoded) const override {
        if (!accepts(m)) return InferenceService::prepare_media(m, encoded);
        const image::Image img = image::decode(encoded);
        MediaInput item;
        item.key = image::content_hash(img);
        item.num_tokens = 2;
        return item;
    }
    std::vector<int32_t> expand_media(std::vector<int32_t> prompt, const std::vector<MediaInput>& media) const override {
        std::vector<int32_t> out;
        size_t k = 0;
        for (int32_t t : prompt) {
            if (t != kImageByte) {
                out.push_back(t);
                continue;
            }
            if (k == media.size()) fail("more image placeholders than images");
            out.insert(out.end(), static_cast<size_t>(media[k++].num_tokens), t);
        }
        if (k != media.size()) fail("fewer image placeholders than images");
        return out;
    }

    bool transcribes() const override { return transcriber; }
    // Hands `reply` to on_text three bytes at a time, as a transcript decoding.
    TranscriptionResult transcribe(const TranscriptionRequest& request, const TextCallback& on_text) override {
        if (!transcriber) return InferenceService::transcribe(request, on_text);
        {
            std::lock_guard lock(serial_);
            last_transcription_ = request;
        }
        TranscriptionResult r;
        if (!error.empty()) {
            r.finish = FinishReason::Error;
            r.error = error;
            return r;
        }
        for (size_t at = 0; at < reply.size(); at += 3) {
            if (on_text && !on_text(std::string_view(reply).substr(at, 3))) {
                r.finish = FinishReason::Cancelled;
                return r;
            }
        }
        r.text = reply;
        r.language = "English";
        r.seconds = 1.5;
        r.pieces = 1;
        r.stats.prompt_tokens = 30;
        r.stats.generated_tokens = 7;
        return r;
    }
    TranscriptionRequest last_transcription() const {
        std::lock_guard lock(serial_);
        return last_transcription_;
    }

    GenerateRequest last_request() const {
        std::lock_guard lock(serial_);
        return last_request_;
    }
    int calls() const {
        std::lock_guard lock(serial_);
        return calls_;
    }
    ModelCard& card() { return card_; }

    // Set before requests are sent.
    std::string reply = "Hello!";
    std::string error;  // non-empty: every generation fails with this message
    bool images = false;  // accept image parts
    bool transcriber = false;  // a speech recognition model
    std::chrono::milliseconds token_delay{0};
    std::atomic<bool> cancelled = false;

private:
    ModelCard card_;
    ByteTokenizer tokenizer_;
    LineTemplate template_;
    mutable std::mutex serial_;
    GenerateRequest last_request_;
    TranscriptionRequest last_transcription_;
    int calls_ = 0;
};

}  // namespace xs::test
