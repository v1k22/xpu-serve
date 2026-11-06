#pragma once

#include <chrono>

namespace xs {

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    void reset() { start_ = std::chrono::steady_clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

}  // namespace xs
