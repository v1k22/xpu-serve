#pragma once

// A model = config + weights + the step graph its architecture code builds. One implementation per architecture
// family (models/<arch>/); every size of the family is served by the same code.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "engine/types.h"
#include "graph/graph.h"
#include "io/safetensors.h"
#include "tensor/tensor.h"

namespace xs {

struct ModelInfo {
    std::string arch;     // e.g. "gemma4"
    std::string id;       // display name, the model directory name by default
    int64_t vocab_size = 0;
    int64_t max_position = 0;     // longest context the model supports
    std::string dialect;          // chat::make_output_parser dialect
    std::vector<int32_t> stop_token_ids;
    SamplingParams default_sampling;  // generation_config.json
};

class Model {
public:
    virtual ~Model() = default;
    virtual const ModelInfo& info() const = 0;
    virtual const graph::Graph& graph() const = 0;
    virtual const WeightStore& weights() const = 0;
    // Fills every graph input for a step that feeds `tokens` at absolute positions start_pos, start_pos + 1, ...
    virtual void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, TensorMap& inputs) const = 0;
};

// Reads <dir>/config.json and dispatches on model_type. Throws xs::Error naming unsupported architectures.
std::unique_ptr<Model> load_model(const std::filesystem::path& dir);
std::vector<std::string> supported_model_types();

}  // namespace xs
