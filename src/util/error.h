#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace xs {

// The one exception type the project throws. Messages name the thing that is wrong and why, so a refusal reads as a
// diagnosis ("model_type 'foo' is not supported; supported: gemma4"), never as a bare failure.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template <typename... Args>
std::string str_cat(Args&&... args) {
    std::ostringstream os;
    (os << ... << std::forward<Args>(args));
    return os.str();
}

template <typename... Args>
[[noreturn]] void fail(Args&&... args) {
    throw Error(str_cat(std::forward<Args>(args)...));
}

}  // namespace xs

// XS_CHECK(cond, "message parts", ...): throws xs::Error naming the failed condition.
#define XS_CHECK(cond, ...)                                                     \
    do {                                                                        \
        if (!(cond)) ::xs::fail("check failed: (" #cond "): ", __VA_ARGS__);    \
    } while (0)
