#pragma once

// A model = config + weights + the step graph its architecture code builds. One implementation per architecture
// family (models/<arch>/); every size of the family is served by the same code.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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

// A drafter that proposes one token a run (an MTP head, an assistant): Speculation::draft_graph takes pairs (the
// target's "hidden" row of position i, the token at i + 1) and gives "logits" [1, n] and "hidden" [1, width] of the
// last pair (the hidden the next chained step takes), where the logits are over the vocabulary or, when the graph has
// an I32 "candidates" output [n], over the tokens it lists. The target's prefill and verify programs output "hidden":
// whatever the drafter takes of them, one row per token.
class ChainDrafting {
public:
    virtual ~ChainDrafting() = default;
    // Inputs of the draft graph. depth 0: the pairs at positions start_pos, start_pos + 1, ... (`hidden` holds one
    // row per token). depth d > 0: one chained step, the d-th draft after the pair at start_pos, from the token
    // drafted before it and the drafter's own "hidden".
    virtual void prepare_draft(std::span<const int32_t> tokens, std::span<const float> hidden, int64_t start_pos,
                               int depth, TensorMap& inputs) const = 0;
    // The drafter keeps its own KV over the sequence: every pair goes through it once, in order (catching up on
    // the pairs a verify accepted). Otherwise a round starts from the last pair alone.
    virtual bool drafter_keeps_state() const = 0;
};

// A drafter that proposes a block in one run (DFlash, DSpine): block_drafts() tokens after a pending one. Nothing is
// handed over: the target's prefill and verify programs append what the drafter reads of them to state slots of its
// own (they count the same tokens as the target's), and Speculation::draft_graph, fed the pending token at its
// position, appends its block to those slots; the engine cuts them back after the run. It gives "drafts" [n] i32, or
// "candidates" [n, K] i32 with "scores" [n, K, K] f32: scores[i][j][k] ranks candidate k of draft i given that draft
// i - 1 was its candidate j (row j = 0 for the first draft), and the drafts follow the best of each row in turn.
class BlockDrafting {
public:
    virtual ~BlockDrafting() = default;
    virtual int64_t block_drafts() const = 0;
    // Inputs of the draft graph for the block after `token`, which sits at `position`.
    virtual void prepare_block(int32_t token, int64_t position, TensorMap& inputs) const = 0;
};

// Speculative decoding with a model's drafter (an MTP head in the checkpoint, an assistant or a block drafter beside
// it): the programs it adds to the step graph. The engine compiles them into the step graph's session after it, in
// the order prefill, verify, draft (engine/spec/drafter.h); they share its weights by name and its state slots by
// label.
class Speculation {
public:
    virtual ~Speculation() = default;
    // The target with "logits" of the last row: a prefill chunk.
    virtual const graph::Graph& prefill_graph() const = 0;
    // The target with "logits" of every row: verifying a pending token and its drafts.
    virtual const graph::Graph& verify_graph() const = 0;
    // The drafter's program: what it takes and gives is its way of drafting, exactly one of the two below.
    virtual const graph::Graph& draft_graph() const = 0;
    virtual const ChainDrafting* chain() const { return nullptr; }
    virtual const BlockDrafting* block() const { return nullptr; }
    // What drafts: "mtp", "assistant", "dflash", "dflash2", "dspine".
    virtual std::string_view drafter_name() const = 0;
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
    // A drafter checkpoint beside the model, for speculation: a Gemma 4 assistant, a block drafter (DFlash, DSpine).
    std::filesystem::path drafter;
    // A drafter that reads the target's head (an MTP head) picks among this many of its rows, chosen by a one-bit
    // sketch of the head and re-scored exactly (blocks::shortlist_head); 0: it reads the whole head.
    int64_t draft_shortlist = 64;
    // The drafts a round will ask for, when known (0: the drafter's own count): a block drafter whose rows do not see
    // the rows after them runs no more than these.
    int64_t drafts = 0;
};

// Reads <dir>/config.json and dispatches on model_type. Throws xs::Error naming unsupported architectures.
std::unique_ptr<Model> load_model(const std::filesystem::path& dir, const LoadOptions& options = {});
std::vector<std::string> supported_model_types();

}  // namespace xs
