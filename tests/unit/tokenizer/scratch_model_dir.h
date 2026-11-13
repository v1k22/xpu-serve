#pragma once

// A scratch model directory holding one tokenizer.json, for tokenizer tests that load a hand-written or edited file.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace xs::test {

// Removed on destruction.
class ScratchModelDir {
public:
    explicit ScratchModelDir(const nlohmann::json& tokenizer_json)
        : path_(std::filesystem::temp_directory_path() /
                ("xs_tok_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)))) {
        std::filesystem::create_directories(path_);
        std::ofstream(path_ / "tokenizer.json") << tokenizer_json.dump();
    }
    ~ScratchModelDir() { std::filesystem::remove_all(path_); }
    ScratchModelDir(const ScratchModelDir&) = delete;
    ScratchModelDir& operator=(const ScratchModelDir&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace xs::test
