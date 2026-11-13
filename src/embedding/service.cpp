#include "embedding/service.h"

#include <chrono>
#include <future>

#include "util/error.h"

namespace xs {

std::vector<int32_t> embedding_tokens(const tok::Tokenizer& tokenizer, std::string_view text) {
    return tokenizer.with_special_tokens(tokenizer.encode(text));
}

struct EmbeddingService::Job {
    const std::vector<std::vector<int32_t>>& sequences;
    std::chrono::steady_clock::time_point submitted;
    std::promise<EmbeddingCall> done;
};

EmbeddingService::EmbeddingService(EmbeddingSpec spec, TokenizeFn tokenize, EmbedFn embed)
    : spec_(std::move(spec)),
      tokenize_(std::move(tokenize)),
      embed_(std::move(embed)),
      thread_([this] { loop(); }) {}

std::unique_ptr<EmbeddingService> EmbeddingService::create(std::unique_ptr<tok::Tokenizer> tokenizer,
                                                           std::unique_ptr<EmbeddingRunner> runner) {
    const tok::Tokenizer* t = tokenizer.get();
    EmbeddingRunner* r = runner.get();
    auto service = std::make_unique<EmbeddingService>(
        r->spec(), [t](std::string_view text) { return embedding_tokens(*t, text); },
        [r](const std::vector<std::vector<int32_t>>& sequences) { return r->embed(sequences); });
    service->tokenizer_ = std::move(tokenizer);
    service->runner_ = std::move(runner);
    return service;
}

EmbeddingService::~EmbeddingService() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    thread_.join();
}

EmbeddingCall EmbeddingService::embed(const std::vector<std::vector<int32_t>>& sequences) {
    Job job{sequences, std::chrono::steady_clock::now(), {}};
    std::future<EmbeddingCall> done = job.done.get_future();
    {
        std::lock_guard lock(mutex_);
        if (stopping_) fail("the embedding service is stopping");
        queue_.push_back(&job);
    }
    wake_.notify_one();
    return done.get();
}

void EmbeddingService::loop() {
    for (;;) {
        Job* job = nullptr;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) return;  // stopping and drained
            job = queue_.front();
            queue_.pop_front();
        }
        EmbeddingCall call;
        call.queue_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - job->submitted).count();
        try {
            call.result = embed_(job->sequences);
            job->done.set_value(std::move(call));
        } catch (...) {
            job->done.set_exception(std::current_exception());
        }
    }
}

}  // namespace xs
