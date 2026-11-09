#pragma once

// A policy behind a server: one device thread and a one-slot queue in which the latest observation wins. A robot
// streams observations; one that is still waiting when a newer one arrives is answered "superseded" instead of
// being run late (the running call is never interrupted). The thread warms up first; until then ready() is false and
// calls wait.

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include "policy/runner.h"
#include "util/error.h"

namespace xs {

// A waiting call that a newer one replaced.
class Superseded : public Error {
public:
    Superseded() : Error("superseded by a newer observation before it ran") {}
};

struct PolicyCall {
    ActResult result;
    double queue_ms = 0;  // from submission until the device thread took it
};

class PolicyService {
public:
    using ActFn = std::function<ActResult(const Observation&, const ActOptions&)>;

    // `act` runs on the service's thread, after `warm_up`.
    PolicyService(PolicySpec spec, ActFn act, std::function<void()> warm_up);
    // A PolicyRunner, warmed up at the checkpoint's flow steps on its first `cameras` cameras (0: all); other camera
    // subsets compile on their first call.
    static std::unique_ptr<PolicyService> create(std::unique_ptr<PolicyRunner> runner, int cameras = 0);
    ~PolicyService();
    PolicyService(const PolicyService&) = delete;
    PolicyService& operator=(const PolicyService&) = delete;

    const PolicySpec& spec() const { return spec_; }
    bool ready() const;
    // Blocks until the call has run. Throws Superseded if a newer call replaced it while it waited, or what `act`
    // threw.
    PolicyCall act(Observation obs, ActOptions options);

private:
    struct Job;
    void loop();

    PolicySpec spec_;
    ActFn act_;
    std::function<void()> warm_up_;
    std::unique_ptr<PolicyRunner> runner_;  // when created from one
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::shared_ptr<Job> pending_;
    bool ready_ = false;
    bool stopping_ = false;
    std::thread thread_;
};

}  // namespace xs
