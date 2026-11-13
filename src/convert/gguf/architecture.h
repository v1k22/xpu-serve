#pragma once

// What a GGUF architecture's importer provides (one file per architecture: qwen35.cpp, gemma4.cpp, clef.cpp): from the
// file's metadata the config.json the model code reads (and any other JSON file of the pack), and for every tensor of
// the file the pack tensors it becomes, under the names the model code reads. The import driver (gguf_import.cpp)
// writes them: a tensor in a block format byte for byte, a float tensor as stored or through a value-by-value
// transform. A quantized tensor is never decoded, re-encoded or reordered; it can only be cut into runs of whole rows.
//
// A model's MTP head can come in a file of its own (`convert --mtp`): the plan then draws its tensors from two files,
// and is the plan of a model file that held the head.

#include <functional>
#include <map>
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
    nlohmann::ordered_json config;  // config.json, without eos_token_id (the import adds the tokenizer's)
    // The pack's other JSON files, by file name (a decision model's head: joint_head_config.json).
    std::map<std::string, nlohmann::ordered_json> files;
    // Named chat templates of the file (tokenizer.chat_template.<name>) the model has no use for, with the reason; left
    // out of the pack like any named template, but not warned about.
    std::map<std::string, std::string, std::less<>> unused_templates;
    // With `derived` and `copies`, every row of every tensor of the files exactly once.
    std::vector<PackTensor> tensors;
    // Tensors the converter computed from the config (not the checkpoint's), which the architecture checked against
    // the config it writes: not imported, since the model code computes them from config.json.
    std::vector<const gguf::TensorInfo*> derived;
    // Tensors of an MTP head's file that repeat one of the model's (llama.cpp's draft model carries its own embedding,
    // output head and final norm), which the architecture checked against the model's shapes: not imported, since
    // the head reads the model's.
    std::vector<const gguf::TensorInfo*> copies;
    // A drafter beside a target model: it is served with the target's tokenizer, chat template and stop tokens, so its
    // pack holds config.json and the weights only.
    bool drafter = false;
};

class Architecture {
public:
    virtual ~Architecture() = default;
    virtual std::string_view name() const = 0;  // general.architecture
    // Refuses by name what it does not implement: metadata keys of its namespace it does not know, tensors it has no
    // name for, shapes that disagree with the metadata.
    virtual ImportPlan plan(const gguf::File& file) const = 0;
    // The plan of `file`, a model without its MTP head, with the head taken from `mtp`, a file of its own: the plan
    // of a file that held both. Refuses by name a model file that has a head, and an MTP file of another architecture,
    // of another model (metadata that disagrees with the file's) or with tensors beyond the head and the copies.
    // The default refuses it by name, for an architecture that has no such files.
    virtual ImportPlan plan_with_mtp(const gguf::File& file, const gguf::File& mtp) const;
};

const Architecture& qwen35();
const Architecture& gemma4();
const Architecture& gemma4_assistant();
const Architecture& clef();

// The importer of general.architecture `name`; a named error listing the supported ones otherwise.
const Architecture& find_architecture(std::string_view name);

// A metadata value of type f32 as the shortest decimal that reads back as the same float (what a config.json written
// by Python holds: 1e-06, not 9.99999997e-07); an f64 or an integer as it is.
double config_number(const gguf::Value& v);

// The values of rows [first, first + rows) of a float tensor (F32, F16, BF16), as f32.
std::vector<float> float_rows(const gguf::TensorInfo& t, int64_t first, int64_t rows);

}  // namespace xs::gguf_import
