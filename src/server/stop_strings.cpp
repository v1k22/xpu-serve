#include "server/stop_strings.h"

#include <algorithm>
#include <utility>

namespace xs::server {

StopStringMatcher::StopStringMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {
    std::erase_if(stops_, [](const std::string& s) { return s.empty(); });
}

std::string StopStringMatcher::push(std::string_view text) {
    if (stopped_) return {};
    pending_ += text;
    size_t match = std::string::npos;
    for (const std::string& stop : stops_) match = std::min(match, pending_.find(stop));
    if (match != std::string::npos) {
        stopped_ = true;
        std::string released = pending_.substr(0, match);
        pending_.clear();
        return released;
    }
    const size_t keep = held_back_length();
    std::string released = pending_.substr(0, pending_.size() - keep);
    pending_.erase(0, pending_.size() - keep);
    return released;
}

std::string StopStringMatcher::flush() {
    if (stopped_) return {};
    return std::exchange(pending_, {});
}

// Length of the longest suffix of pending_ that is a proper prefix of some stop string.
size_t StopStringMatcher::held_back_length() const {
    size_t longest = 0;
    for (const std::string& stop : stops_) {
        const size_t max_len = std::min(pending_.size(), stop.size() - 1);
        for (size_t len = max_len; len > longest; --len) {
            if (std::string_view(pending_).substr(pending_.size() - len) == std::string_view(stop).substr(0, len)) {
                longest = len;
                break;
            }
        }
    }
    return longest;
}

}  // namespace xs::server
