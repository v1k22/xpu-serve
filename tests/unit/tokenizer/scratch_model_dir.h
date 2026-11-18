#pragma once

// A scratch model directory holding one tokenizer.json, or any files, for tokenizer tests that load hand-written or
// edited ones.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#include <nlohmann/json.hpp>

namespace xs::test {

// Removed on destruction.
class ScratchModelDir {
public:
    explicit ScratchModelDir(const nlohmann::json& tokenizer_json)
        : ScratchModelDir(std::map<std::string, std::string>{{"tokenizer.json", tokenizer_json.dump()}}) {}
    // The files by name, each with its contents.
    explicit ScratchModelDir(const std::map<std::string, std::string>& files)
        : path_(std::filesystem::temp_directory_path() /
                ("xs_tok_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)))) {
        std::filesystem::create_directories(path_);
        for (const auto& [name, text] : files) std::ofstream(path_ / name, std::ios::binary) << text;
    }
    ~ScratchModelDir() { std::filesystem::remove_all(path_); }
    ScratchModelDir(const ScratchModelDir&) = delete;
    ScratchModelDir& operator=(const ScratchModelDir&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace xs::test
