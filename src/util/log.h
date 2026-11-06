#pragma once

#include <string_view>

#include "util/error.h"

namespace xs::log {

enum class Level { Debug = 0, Info = 1, Warn = 2, Error = 3 };

void set_level(Level level);
Level level();
// Parses "debug", "info", "warn", "error"; throws on anything else.
Level parse_level(std::string_view name);

// Writes one line to stderr with a timestamp and level tag. Thread-safe.
void write(Level level, std::string_view message);

template <typename... Args>
void debug(Args&&... args) {
    if (level() <= Level::Debug) write(Level::Debug, str_cat(std::forward<Args>(args)...));
}
template <typename... Args>
void info(Args&&... args) {
    if (level() <= Level::Info) write(Level::Info, str_cat(std::forward<Args>(args)...));
}
template <typename... Args>
void warn(Args&&... args) {
    if (level() <= Level::Warn) write(Level::Warn, str_cat(std::forward<Args>(args)...));
}
template <typename... Args>
void error(Args&&... args) {
    write(Level::Error, str_cat(std::forward<Args>(args)...));
}

}  // namespace xs::log
