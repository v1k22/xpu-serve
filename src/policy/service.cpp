#include "policy/service.h"

#include <chrono>
#include <future>

#include "util/log.h"

namespace xs {

struct PolicyService::Job {
    Observation obs;
    ActOptions options;
    std::chrono::steady_clock::time_point submitted;
    std::promise<PolicyCall> done;
};

PolicyService::PolicyService(PolicySpec spec, ActFn act, std::function<void()> warm_up)
    : spec_(std::move(spec)), act_(std::move(act)), warm_up_(std::move(warm_up)), thread_([this] { loop(); }) {}

std::unique_ptr<PolicyService> PolicyService::create(std::unique_ptr<PolicyRunner> runner, int cameras) {
    PolicyRunner* r = runner.get();
    const auto n = static_cast<int>(r->spec().cameras.size());
    XS_CHECK(cameras >= 0 && cameras <= n, "warm-up cameras must be 1 to ", n, " (0: all), got ", cameras);
    std::vector<int> all;
    for (int c = 0; c < (cameras > 0 ? cameras : n); ++c) all.push_back(c);
    auto service = std::make_unique<PolicyService>(
        r->spec(), [r](const Observation& obs, const ActOptions& o) { return r->act(obs, o); },
        [r, all] {
            const PolicySpec& s = r->spec();
            const double ms = r->warm_up(synthetic_observation(s, all, "warm up"), s.denoise_steps);
            log::info("policy ", s.id, ": warmed up (", static_cast<int>(all.size()), " cameras, ", s.denoise_steps,
                      " flow steps) in ", static_cast<int>(ms), " ms");
        });
    service->runner_ = std::move(runner);
    return service;
}

PolicyService::~PolicyService() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    thread_.join();
}

bool PolicyService::ready() const {
    std::lock_guard lock(mutex_);
    return ready_;
}

PolicyCall PolicyService::act(Observation obs, ActOptions options) {
    auto job = std::make_shared<Job>();
    job->obs = std::move(obs);
    job->options = options;
    job->submitted = std::chrono::steady_clock::now();
    std::future<PolicyCall> done = job->done.get_future();
    {
        std::lock_guard lock(mutex_);
        if (stopping_) fail("policy service is stopping");
        if (pending_) pending_->done.set_exception(std::make_exception_ptr(Superseded()));
        pending_ = job;
    }
    wake_.notify_all();
    return done.get();
}

void PolicyService::loop() {
    try {
        if (warm_up_) warm_up_();
    } catch (const std::exception& e) {
        log::error("policy warm-up failed: ", e.what());
    }
    {
        std::lock_guard lock(mutex_);
        ready_ = true;
    }
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [this] { return stopping_ || pending_; });
            if (!pending_) return;  // stopping, nothing left
            job = std::move(pending_);
        }
        PolicyCall call;
        call.queue_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - job->submitted).count();
        try {
            call.result = act_(job->obs, job->options);
            job->done.set_value(std::move(call));
        } catch (...) {
            job->done.set_exception(std::current_exception());
        }
    }
}

}  // namespace xs
