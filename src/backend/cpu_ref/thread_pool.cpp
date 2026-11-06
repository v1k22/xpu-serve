#include "backend/cpu_ref/thread_pool.h"

namespace xs::cpu_ref {

ThreadPool::ThreadPool(size_t num_workers) {
    workers_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; ++i)
        workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        const std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    start_cv_.notify_all();
    for (std::thread& t : workers_)
        t.join();
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t)>& fn) {
    if (n <= 0) return;
    if (n == 1 || workers_.empty()) {
        for (int64_t i = 0; i < n; ++i)
            fn(i);
        return;
    }
    const std::lock_guard job_lock(job_mutex_);
    {
        const std::lock_guard lock(mutex_);
        job_ = &fn;
        job_size_ = n;
        next_item_.store(0);
        running_ = workers_.size();
        ++generation_;
    }
    start_cv_.notify_all();
    run_items();
    std::unique_lock lock(mutex_);
    done_cv_.wait(lock, [this] { return running_ == 0; });
    job_ = nullptr;
}

void ThreadPool::run_items() {
    for (int64_t i = next_item_.fetch_add(1); i < job_size_; i = next_item_.fetch_add(1))
        (*job_)(i);
}

void ThreadPool::worker_loop() {
    uint64_t seen = 0;
    while (true) {
        {
            std::unique_lock lock(mutex_);
            start_cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
            if (stopping_) return;
            seen = generation_;
        }
        run_items();
        {
            const std::lock_guard lock(mutex_);
            --running_;
        }
        done_cv_.notify_one();
    }
}

}  // namespace xs::cpu_ref
