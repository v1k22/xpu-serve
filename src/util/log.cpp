#include "util/log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>

namespace xs::log {
namespace {
std::atomic<Level> g_level{Level::Info};
std::mutex g_mutex;

const char* tag(Level level) {
    switch (level) {
        case Level::Debug: return "debug";
        case Level::Info: return "info ";
        case Level::Warn: return "warn ";
        case Level::Error: return "error";
    }
    return "?";
}
}  // namespace

void set_level(Level level) { g_level.store(level); }
Level level() { return g_level.load(); }

Level parse_level(std::string_view name) {
    if (name == "debug") return Level::Debug;
    if (name == "info") return Level::Info;
    if (name == "warn") return Level::Warn;
    if (name == "error") return Level::Error;
    fail("unknown log level '", name, "' (expected debug, info, warn or error)");
}

void write(Level lvl, std::string_view message) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t secs = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
    localtime_r(&secs, &tm);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%H:%M:%S", &tm);
    std::lock_guard<std::mutex> lock(g_mutex);
    std::fprintf(stderr, "%s.%03lld %s %.*s\n", stamp, static_cast<long long>(ms), tag(lvl),
                 static_cast<int>(message.size()), message.data());
}

}  // namespace xs::log
