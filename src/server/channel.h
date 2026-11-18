#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

namespace xs::server {

// Output of a generating thread (deltas, transcript text) handed to the HTTP thread that writes it.
template <typename T>
class Channel {
public:
    void push(T item) {
        {
            std::lock_guard lock(mutex_);
            queue_.push_back(std::move(item));
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
    bool wait_take(std::deque<T>& out, std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        ready_.wait_for(lock, timeout, [this] { return closed_ || !queue_.empty(); });
        out = std::exchange(queue_, {});
        return !(closed_ && out.empty());
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<T> queue_;
    bool closed_ = false;
};

}  // namespace xs::server
