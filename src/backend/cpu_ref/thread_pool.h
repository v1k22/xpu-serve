#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace xs::cpu_ref {

// A fixed set of worker threads that run one parallel_for at a time. Shared by every session of a backend.
class ThreadPool {
  public:
    explicit ThreadPool(size_t num_workers);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Calls fn(i) for every i in [0, n) on the workers and the calling thread; returns when all calls are done.
    // fn must not throw. Concurrent callers are serialized.
    void parallel_for(int64_t n, const std::function<void(int64_t)>& fn);

    size_t num_threads() const { return workers_.size() + 1; }

  private:
    void worker_loop();
    void run_items();

    std::mutex job_mutex_;  // one parallel_for at a time
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    const std::function<void(int64_t)>* job_ = nullptr;
    int64_t job_size_ = 0;
    std::atomic<int64_t> next_item_{0};
    size_t running_ = 0;
    uint64_t generation_ = 0;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
};

}  // namespace xs::cpu_ref
