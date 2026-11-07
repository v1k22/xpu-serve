#pragma once

// `xpu-serve convert`: a Hugging Face checkpoint in, an xpu-serve pack out (DESIGN.md section 6). File to file: the
// model's own step graphs decide what to quantize: the text decoder's matmul weights take the requested format, its
// gather tables q8 (a tied LM head reads them too), a vision tower's matmul weights q8, and tables the model reads on
// the host (Model::host_tables) q8. Every other tensor and every other file (config, tokenizer, chat template,
// generation config) is copied unchanged, and xpu_pack.json records each quantized tensor's format.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "tensor/dtype.h"

namespace xs {

enum class QuantMethod : uint8_t {
    Rtn,  // round to nearest
    Awq,  // q4 matmul weights: activation-aware scale and clip search from calibration (quant::quantize_awq)
};

struct ConvertOptions {
    std::filesystem::path source;
    std::filesystem::path output;     // created; must not exist or be empty
    DType format = DType::Q4;         // matmul weights: Q4 or Q8
    int64_t group = 64;               // columns per scale: 32, 64 or 128
    int64_t min_elements = 1 << 20;   // smaller matrices stay as they are (they cost little and are often sensitive)
    std::vector<std::string> q8_names;  // matmul weights whose name contains one of these take q8 instead of q4
    QuantMethod method = QuantMethod::Rtn;
    std::string calibration_text;     // AWQ: text to run through the model; empty = builtin_calibration_text()
    std::vector<int32_t> calibration_ids;  // AWQ: token ids to use instead of tokenizing text (tests)
    std::string backend;              // AWQ calibration backend; empty = the first of this build
    std::string device;
    int64_t calibration_tokens = 8192;
    std::string converter = "xpu-serve";  // recorded in the manifest, e.g. "xpu-serve 0.4.0"
};

struct ConvertReport {
    int64_t q4 = 0;    // tensors written as q4
    int64_t q8 = 0;    // tensors written as q8
    int64_t kept = 0;  // tensors copied as they were
    uint64_t source_bytes = 0;
    uint64_t output_bytes = 0;
    int shards = 0;
    int64_t awq = 0;           // q4 tensors quantized with AWQ
    double awq_gain = 0;       // mean of 1 - error / round-to-nearest error over them (the AWQ objective)
};

ConvertReport convert_model(const ConvertOptions& options);

}  // namespace xs
