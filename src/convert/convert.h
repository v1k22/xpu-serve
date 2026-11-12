#pragma once

// `xpu-serve convert`: a Hugging Face checkpoint (a language or decision model, or a LeRobot policy) in, an xpu-serve
// pack out (DESIGN.md section 6). File to file: the model's own step graphs decide what to quantize, by one rule for every model: a matmul weight
// takes the requested format when every matmul reading it is bandwidth-bound (a step-dependent number of rows, as in
// decode, or at most kBandwidthRows rows per call, as in a policy's action expert), because a smaller weight is then a
// faster matmul; weights of compute-bound matmuls (an image encoder, a policy's prefix) stay as they are unless the
// scope is All (then a language model's vision tower takes q8, as it is not quality-measured at q4). Gather tables
// take q8 (a tied LM head reads them too), and so do tables a language model reads on the host
// (Model::host_tables). Every other tensor and every other file (config, tokenizer, chat template, processor
// configs) is copied unchanged, and xpu_pack.json records each quantized tensor's format.
//
// A drafter checkpoint beside a model (a block drafter, an assistant) converts the same way, into a pack of its own:
// the model it drafts for (ConvertOptions::target, a checkpoint or a pack) is loaded with it, and the programs that
// read the drafter's weights decide their formats.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <map>

#include "graph/graph.h"
#include "io/safetensors.h"
#include "tensor/dtype.h"

namespace xs {

enum class QuantMethod : uint8_t {
    Rtn,  // round to nearest
    Awq,  // q4 matmul weights: activation-aware scale and clip search from calibration (quant::quantize_awq)
};

// Which matmul weights take the requested format (the rule above).
enum class QuantScope : uint8_t {
    Bandwidth,  // matmuls that read their weight more than they compute: a speed-up
    All,        // every matmul weight: the smallest pack
};
inline constexpr int64_t kBandwidthRows = 64;

struct ConvertOptions {
    std::filesystem::path source;
    std::filesystem::path target;     // set: `source` is a drafter checkpoint of this model
    std::filesystem::path output;     // created; must not exist or be empty
    DType format = DType::Q4;         // matmul weights: Q4 or Q8
    int64_t group = 64;               // columns per scale: 32, 64 or 128
    int64_t min_elements = 1 << 20;   // smaller matrices stay as they are (they cost little and are often sensitive)
    std::vector<std::string> q8_names;  // matmul weights whose name contains one of these take q8 instead of q4
    QuantScope scope = QuantScope::Bandwidth;
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

// The formats the rule above gives the weights `g` reads: matmul weights at `matmul_format` within `options.scope`,
// gather tables at q8; weights left out stay as they are.
std::map<std::string, DType> plan_graph_formats(const graph::Graph& g, const WeightStore& weights, DType matmul_format,
                                                const ConvertOptions& options);

}  // namespace xs
