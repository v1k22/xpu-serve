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
#include "image/image.h"
#include "io/safetensors.h"
#include "tensor/tensor.h"

namespace xs {

struct ModelInfo {
    std::string arch;     // e.g. "gemma4"
    std::string id;       // display name, the model directory name by default
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;      // width of a token embedding (and of an image soft token's row)
    int64_t max_position = 0;     // longest context the model supports
    std::string dialect;          // chat::make_output_parser dialect
    std::vector<int32_t> stop_token_ids;
    SamplingParams default_sampling;  // generation_config.json
};

// The image side of a vision-language model: preprocessing, the encoder graph and the prompt layout of an image.
class VisionEncoder {
public:
    virtual ~VisionEncoder() = default;
    // Run once per image (stateless). Its inputs are what prepare() fills; its output "embeddings" is [N, hidden]
    // f32 whose first MediaInput::num_tokens rows are the image's soft tokens (the rest pads to a static shape).
    virtual const graph::Graph& graph() const = 0;
    // Decoded pixels -> encoder inputs and soft-token count (key is left for the caller). Throws xs::Error for an
    // image the model cannot take.
    virtual MediaInput prepare(const image::Image& img) const = 0;
    // The token the chat template renders once per image.
    virtual int32_t placeholder_token() const = 0;
    // What one placeholder becomes in the prompt for an image of `num_tokens` soft tokens.
    virtual std::vector<int32_t> token_run(int64_t num_tokens) const = 0;
    // The token at every soft-token position of a run.
    virtual int32_t soft_token() const = 0;
};

class Model {
public:
    virtual ~Model() = default;
    virtual const ModelInfo& info() const = 0;
    virtual const graph::Graph& graph() const = 0;
    virtual const WeightStore& weights() const = 0;
    // Fills every graph input for a step that feeds `tokens` at absolute positions start_pos, start_pos + 1, ...
    // `embeddings` is empty, or holds one entry per token: a row of the model's hidden size that replaces that
    // token's embedding (an image's soft token), or nullptr for an ordinary token.
    virtual void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos,
                                std::span<const float* const> embeddings, TensorMap& inputs) const = 0;
    // Null for a text-only model, or when vision was not loaded.
    virtual const VisionEncoder* vision() const { return nullptr; }
};

struct LoadOptions {
    bool vision = true;  // load the vision encoder when the checkpoint has one
};

// Reads <dir>/config.json and dispatches on model_type. Throws xs::Error naming unsupported architectures.
std::unique_ptr<Model> load_model(const std::filesystem::path& dir, const LoadOptions& options = {});
std::vector<std::string> supported_model_types();

}  // namespace xs
