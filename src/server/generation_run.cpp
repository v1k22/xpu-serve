#include "server/generation_run.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>

namespace xs::server {
namespace {

constexpr auto kAlivePollInterval = std::chrono::milliseconds(100);

// Deltas from the generating thread to the HTTP thread.
class DeltaChannel {
public:
    void push(chat::ParsedDelta delta) {
        {
            std::lock_guard lock(mutex_);
            queue_.push_back(std::move(delta));
        }
        ready_.notify_one();
    }
    void close() {
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
        }
        ready_.notify_one();
    }
    // Waits up to `timeout` for output. Returns false once the channel is closed and drained.
    bool wait_take(std::deque<chat::ParsedDelta>& out, std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        ready_.wait_for(lock, timeout, [this] { return closed_ || !queue_.empty(); });
        out = std::exchange(queue_, {});
        return !(closed_ && out.empty());
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<chat::ParsedDelta> queue_;
    bool closed_ = false;
};

GenerateResult generate_or_error(InferenceService& service, const GenerateRequest& request,
                                 const TokenCallback& on_token) {
    try {
        return service.generate(request, on_token);
    } catch (const std::exception& e) {
        GenerateResult failed;
        failed.finish = FinishReason::Error;
        failed.error = e.what();
        return failed;
    }
}

}  // namespace

GenerationOutcome run_generation(InferenceService& service, const GenerateRequest& request, TokenOutput& output,
                                 const std::function<bool(const chat::ParsedDelta&)>& on_delta,
                                 const std::function<bool()>& client_alive) {
    std::atomic<bool> client_gone = false;
    DeltaChannel channel;
    GenerateResult result;

    const TokenCallback on_token = [&](int32_t token) {
        if (client_gone.load()) return false;
        chat::ParsedDelta delta = output.push(token);
        if (!delta.empty()) channel.push(std::move(delta));
        return !output.stop_string_hit();
    };
    std::jthread worker([&] {
        result = generate_or_error(service, request, on_token);
        if (chat::ParsedDelta rest = output.finish(); !rest.empty()) channel.push(std::move(rest));
        channel.close();
    });

    std::deque<chat::ParsedDelta> batch;
    auto next_alive_check = std::chrono::steady_clock::now() + kAlivePollInterval;
    while (channel.wait_take(batch, kAlivePollInterval)) {
        for (const chat::ParsedDelta& delta : batch) {
            if (!client_gone && !on_delta(delta)) client_gone = true;
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_alive_check) {
            if (!client_gone && !client_alive()) client_gone = true;
            next_alive_check = now + kAlivePollInterval;
        }
    }
    worker.join();
    return {std::move(result), output.stop_string_hit(), client_gone.load()};
}

}  // namespace xs::server
