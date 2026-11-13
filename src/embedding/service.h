#pragma once

// An embedding model behind a server: the checkpoint's tokenizer for text, and one device thread that embeds requests
// in arrival order. The server is written against this class with any functions that tokenize and embed, so it is
// tested without a model.

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>

#include "embedding/runner.h"
#include "tokenizer/tokenizer.h"

namespace xs {

// The ids an embedding model reads `text` as: the tokenizer's encoding (special tokens written in the text become
// theirs) inside the special tokens its post-processor puts around one sequence, as sentence-transformers tokenizes.
std::vector<int32_t> embedding_tokens(const tok::Tokenizer& tokenizer, std::string_view text);

struct EmbeddingCall {
    EmbeddingResult result;
    double queue_ms = 0;  // from submission until the device thread took it
};

class EmbeddingService {
public:
    using TokenizeFn = std::function<std::vector<int32_t>(std::string_view)>;
    using EmbedFn = std::function<EmbeddingResult(const std::vector<std::vector<int32_t>>&)>;

    // `tokenize` runs on the caller's thread and must be safe to call from several at once; `embed` runs on the
    // service's thread.
    EmbeddingService(EmbeddingSpec spec, TokenizeFn tokenize, EmbedFn embed);
    static std::unique_ptr<EmbeddingService> create(std::unique_ptr<tok::Tokenizer> tokenizer,
                                                    std::unique_ptr<EmbeddingRunner> runner);
    ~EmbeddingService();
    EmbeddingService(const EmbeddingService&) = delete;
    EmbeddingService& operator=(const EmbeddingService&) = delete;

    const EmbeddingSpec& spec() const { return spec_; }
    // The ids the model reads `text` as (embedding_tokens).
    std::vector<int32_t> tokenize(std::string_view text) const { return tokenize_(text); }
    // Blocks until the sequences are embedded. Throws what `embed` threw. Thread-safe.
    EmbeddingCall embed(const std::vector<std::vector<int32_t>>& sequences);

private:
    struct Job;
    void loop();

    EmbeddingSpec spec_;
    TokenizeFn tokenize_;
    EmbedFn embed_;
    std::unique_ptr<tok::Tokenizer> tokenizer_;  // when created from one
    std::unique_ptr<EmbeddingRunner> runner_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Job*> queue_;
    bool stopping_ = false;
    std::thread thread_;
};

}  // namespace xs
