#pragma once

#include <cstddef>
#include <filesystem>

namespace xs {

// Read-only memory map of a whole file. Move-only.
class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const { return data_; }
    size_t size() const { return size_; }
    const std::filesystem::path& path() const { return path_; }

private:
    void release() noexcept;

    std::filesystem::path path_;
    const std::byte* data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace xs
