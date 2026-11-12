#include "decision/service.h"

#include <chrono>
#include <future>

#include "util/error.h"

namespace xs {

struct DecisionService::Job {
    const DecisionRequest& request;
    std::chrono::steady_clock::time_point submitted;
    std::promise<DecisionCall> done;
};

DecisionService::DecisionService(DecisionSpec spec, int64_t max_tokens, DecideFn decide)
    : spec_(std::move(spec)), max_tokens_(max_tokens), decide_(std::move(decide)), thread_([this] { loop(); }) {}

std::unique_ptr<DecisionService> DecisionService::create(std::unique_ptr<DecisionRunner> runner) {
    DecisionRunner* r = runner.get();
    auto service = std::make_unique<DecisionService>(
        r->spec(), r->max_tokens(), [r](const DecisionRequest& request) { return r->decide(request); });
    service->runner_ = std::move(runner);
    return service;
}

DecisionService::~DecisionService() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    thread_.join();
}

DecisionCall DecisionService::decide(const DecisionRequest& request) {
    Job job{request, std::chrono::steady_clock::now(), {}};
    std::future<DecisionCall> done = job.done.get_future();
    {
        std::lock_guard lock(mutex_);
        if (stopping_) fail("the decision service is stopping");
        queue_.push_back(&job);
    }
    wake_.notify_one();
    return done.get();
}

void DecisionService::loop() {
    for (;;) {
        Job* job = nullptr;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) return;  // stopping and drained
            job = queue_.front();
            queue_.pop_front();
        }
        DecisionCall call;
        call.queue_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - job->submitted).count();
        try {
            call.result = decide_(job->request);
            job->done.set_value(std::move(call));
        } catch (...) {
            job->done.set_exception(std::current_exception());
        }
    }
}

}  // namespace xs
