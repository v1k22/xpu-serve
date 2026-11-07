#pragma once

// `xpu-serve convert`: a Hugging Face checkpoint in, an xpu-serve pack out (DESIGN.md section 6). File to file: the
// model's own step graph decides what to quantize (weights it reads as matmul weights, and gather tables, which go to
// q8 because a tied LM head reads them too); every other tensor and every other file (config, tokenizer, chat
// template, generation config) is copied unchanged, and xpu_pack.json records each quantized tensor's format.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "tensor/dtype.h"

namespace xs {

struct ConvertOptions {
    std::filesystem::path source;
    std::filesystem::path output;     // created; must not exist or be empty
    DType format = DType::Q4;         // matmul weights: Q4 or Q8
    int64_t group = 64;               // columns per scale: 32, 64 or 128
    int64_t min_elements = 1 << 20;   // smaller matrices stay as they are (they cost little and are often sensitive)
    std::vector<std::string> q8_names;  // matmul weights whose name contains one of these take q8 instead of q4
    std::string converter = "xpu-serve";  // recorded in the manifest, e.g. "xpu-serve 0.4.0"
};

struct ConvertReport {
    int64_t q4 = 0;    // tensors written as q4
    int64_t q8 = 0;    // tensors written as q8
    int64_t kept = 0;  // tensors copied as they were
    uint64_t source_bytes = 0;
    uint64_t output_bytes = 0;
    int shards = 0;
};

ConvertReport convert_model(const ConvertOptions& options);

}  // namespace xs
