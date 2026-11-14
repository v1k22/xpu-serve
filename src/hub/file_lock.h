#pragma once

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <filesystem>

#include "util/error.h"
#include "util/log.h"

namespace xs::hub {

// An exclusive flock on a file (created if missing), held for the object's life: what huggingface_hub takes on a
// blob it writes, so neither tool writes a file the other is writing.
class FileLock {
public:
    explicit FileLock(const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) fail("cannot open the lock file ", path.string());
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            log::info("waiting for another process that holds ", path.string());
            if (::flock(fd_, LOCK_EX) != 0) fail("cannot lock ", path.string());
        }
    }
    ~FileLock() {
        if (fd_ >= 0) ::close(fd_);
    }
    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

private:
    int fd_ = -1;
};

}  // namespace xs::hub
