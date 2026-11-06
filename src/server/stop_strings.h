#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace xs::server {

// Finds OpenAI `stop` strings in streamed text. Text that could still be the start of a stop string is held back
// until it can no longer match, so nothing at or past a match is ever released.
class StopStringMatcher {
public:
    explicit StopStringMatcher(std::vector<std::string> stops);  // empty strings are ignored

    // Returns the text that is now safe to emit. On a match it returns the text before the match and stopped()
    // becomes true; every later call returns nothing.
    std::string push(std::string_view text);
    // End of stream: releases whatever is still held back (nothing once stopped).
    std::string flush();
    bool stopped() const { return stopped_; }

private:
    size_t held_back_length() const;

    std::vector<std::string> stops_;
    std::string pending_;
    bool stopped_ = false;
};

}  // namespace xs::server
