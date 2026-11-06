#include "util/mmap_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>

#include "util/error.h"

namespace xs {

MappedFile::MappedFile(const std::filesystem::path& path) : path_(path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) fail("cannot open '", path.string(), "': ", std::strerror(errno));
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        const int err = errno;
        ::close(fd);
        fail("cannot stat '", path.string(), "': ", std::strerror(err));
    }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ > 0) {
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            const int err = errno;
            ::close(fd);
            fail("cannot mmap '", path.string(), "': ", std::strerror(err));
        }
        data_ = static_cast<const std::byte*>(p);
    }
    ::close(fd);  // the mapping keeps the file alive
}

MappedFile::~MappedFile() { release(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : path_(std::move(other.path_)), data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        release();
        path_ = std::move(other.path_);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void MappedFile::release() noexcept {
    if (data_ != nullptr) ::munmap(const_cast<std::byte*>(data_), size_);
    data_ = nullptr;
    size_ = 0;
}

}  // namespace xs
