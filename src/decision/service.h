#pragma once

// A decision model behind a server: one device thread that takes requests in arrival order. The server is written
// against this class with any function that answers a request, so it is tested without a model.

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "decision/runner.h"

namespace xs {

struct DecisionCall {
    DecisionResult result;
    double queue_ms = 0;  // from submission until the device thread took it
};

class DecisionService {
public:
    using DecideFn = std::function<DecisionResult(const DecisionRequest&)>;

    // `decide` runs on the service's thread.
    DecisionService(DecisionSpec spec, int64_t max_tokens, DecideFn decide);
    static std::unique_ptr<DecisionService> create(std::unique_ptr<DecisionRunner> runner);
    ~DecisionService();
    DecisionService(const DecisionService&) = delete;
    DecisionService& operator=(const DecisionService&) = delete;

    const DecisionSpec& spec() const { return spec_; }
    int64_t max_tokens() const { return max_tokens_; }  // the longest sequence a request may need
    // Blocks until the request has run. Throws what `decide` threw (DecisionRefused for a request the model cannot
    // take). Thread-safe.
    DecisionCall decide(const DecisionRequest& request);

private:
    struct Job;
    void loop();

    DecisionSpec spec_;
    int64_t max_tokens_ = 0;
    DecideFn decide_;
    std::unique_ptr<DecisionRunner> runner_;  // when created from one
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Job*> queue_;
    bool stopping_ = false;
    std::thread thread_;
};

}  // namespace xs
