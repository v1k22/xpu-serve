#pragma once

// What the real-file GGUF checks share: the file (XS_GGUF_FILE) and the Hugging Face checkpoint directory it was
// converted from, and the tokenizer files the import writes from it.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string_view>

#include "convert/gguf/tokenizer.h"
#include "io/gguf.h"

namespace xs::test {

struct RealGguf {
    std::filesystem::path gguf;
    std::filesystem::path hf;
};

// XS_GGUF_FILE and the directory in `dir_variable`, when both are set and the file's general.architecture is
// `architecture`; nullopt otherwise (the checks then skip).
inline std::optional<RealGguf> real_gguf(const char* dir_variable, std::string_view architecture) {
    const char* g = std::getenv("XS_GGUF_FILE");
    const char* h = std::getenv(dir_variable);
    if (g == nullptr || *g == '\0' || h == nullptr || *h == '\0') return std::nullopt;
    if (gguf::File(g).get_string("general.architecture") != architecture) return std::nullopt;
    return RealGguf{g, h};
}

// The tokenizer files the import writes from `file`, into `dir`.
inline void write_gguf_tokenizer(const gguf::File& file, const std::filesystem::path& dir) {
    const gguf_import::TokenizerFiles t = gguf_import::tokenizer_files(file);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "tokenizer.json") << t.tokenizer.dump();
    std::ofstream(dir / "tokenizer_config.json") << t.config.dump(1);
}

}  // namespace xs::test
