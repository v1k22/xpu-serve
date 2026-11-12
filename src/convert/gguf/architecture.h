#pragma once

// What a GGUF architecture's importer provides (one file per architecture, e.g. qwen35.cpp): from the file's metadata
// the config.json the model code reads, and for every tensor of the file the pack tensors it becomes, under the names
// the model code reads. The import driver (gguf_import.cpp) writes them: a tensor in a block format byte for byte, a
// float tensor as stored or through a value-by-value transform. A quantized tensor is never decoded, re-encoded or
// reordered; it can only be cut into runs of whole rows.

#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/gguf.h"
#include "tensor/tensor.h"

namespace xs::gguf_import {

// A float transform, value by value, that undoes one llama.cpp's converter applied. Only a float tensor (F32, F16,
// BF16) can take one; the result is written as F32.
struct FloatMap {
    std::string_view what;  // for errors, e.g. "A_log = log(-ssm_a)"
    float (*apply)(float);
};

// One tensor of the pack.
struct PackTensor {
    std::string name;  // the name the model code reads
    Dims shape;        // its shape there, row-major; the last dim is the source's columns
    const gguf::TensorInfo* source = nullptr;
    int64_t first_row = 0;          // the source rows it takes: numel(shape) / shape.back() of them from this one
    const FloatMap* map = nullptr;  // null: the values as stored
};

struct ImportPlan {
    nlohmann::ordered_json config;    // config.json, without eos_token_id (the import adds the tokenizer's)
    std::vector<PackTensor> tensors;  // with `derived`, every row of every tensor of the file exactly once
    // Tensors the converter computed from the config (not the checkpoint's), which the architecture checked against
    // the config it writes: not imported, since the model code computes them from config.json.
    std::vector<const gguf::TensorInfo*> derived;
};

class Architecture {
public:
    virtual ~Architecture() = default;
    virtual std::string_view name() const = 0;  // general.architecture
    // Refuses by name what it does not implement: metadata keys of its namespace it does not know, tensors it has no
    // name for, shapes that disagree with the metadata.
    virtual ImportPlan plan(const gguf::File& file) const = 0;
};

const Architecture& qwen35();

// The importer of general.architecture `name`; a named error listing the supported ones otherwise.
const Architecture& find_architecture(std::string_view name);

// A metadata value of type f32 as the shortest decimal that reads back as the same float (what a config.json written
// by Python holds: 1e-06, not 9.99999997e-07); an f64 or an integer as it is.
double config_number(const gguf::Value& v);

// The values of rows [first, first + rows) of a float tensor (F32, F16, BF16), as f32.
std::vector<float> float_rows(const gguf::TensorInfo& t, int64_t first, int64_t rows);

}  // namespace xs::gguf_import
