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

// Speculative decoding with a model's own drafter (an MTP head in the checkpoint, or an assistant checkpoint): the
// programs it adds to the step graph and how the drafter is fed. The engine compiles them into the step graph's
// session after it, in the order prefill, verify, draft (engine/spec/programs.h); they share its weights by name and
// its state slots by label. "hidden" is whatever the drafter takes from the target (the same width in every program).
class Speculation {
public:
    virtual ~Speculation() = default;
    // The target with "logits" of the last row and "hidden" of every row: a prefill chunk.
    virtual const graph::Graph& prefill_graph() const = 0;
    // The target with "logits" and "hidden" of every row: verifying a pending token and its drafts.
    virtual const graph::Graph& verify_graph() const = 0;
    // One drafter step over pairs (the target's hidden row of position i, the token at i + 1): "logits" [1, n] and
    // "hidden" [1, width] of the last pair (the hidden the next chained step takes), where the logits are over the
    // vocabulary or, when the graph has an I32 "candidates" output [n], over the tokens it lists.
    virtual const graph::Graph& draft_graph() const = 0;
    // Inputs of draft_graph. depth 0: the pairs at positions start_pos, start_pos + 1, ... (`hidden` holds one row
    // per token). depth d > 0: one chained step, the d-th draft after the pair at start_pos, from the token drafted
    // before it and the drafter's own "hidden".
    virtual void prepare_draft(std::span<const int32_t> tokens, std::span<const float> hidden, int64_t start_pos,
                               int depth, TensorMap& inputs) const = 0;
    // The drafter keeps its own KV over the sequence: every pair goes through it once, in order (catching up on
    // the pairs a verify accepted). Otherwise a round starts from the last pair alone.
    virtual bool drafter_keeps_state() const = 0;
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
    // Weights the model reads on the host one row at a time (embedding tables kept out of device memory, like Gemma
    // 4's per-layer embeddings). They never appear in a graph, so convert asks for them here; they may be Q8 / Q4.
    virtual std::vector<std::string> host_tables() const { return {}; }
    // Null unless the model was loaded with LoadOptions::speculation and has a drafter.
    virtual const Speculation* speculation() const { return nullptr; }
};

struct LoadOptions {
    bool vision = true;        // load the vision encoder when the checkpoint has one
    bool speculation = false;  // build the speculative decoding programs when the model has a drafter
    std::filesystem::path drafter;  // a drafter checkpoint beside the model (a Gemma 4 assistant), for speculation
};

// Reads <dir>/config.json and dispatches on model_type. Throws xs::Error naming unsupported architectures.
std::unique_ptr<Model> load_model(const std::filesystem::path& dir, const LoadOptions& options = {});
std::vector<std::string> supported_model_types();

}  // namespace xs
