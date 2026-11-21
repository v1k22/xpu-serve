#pragma once

// Inkling 2's block drafter (DSpark; BlockDrafting, models/model.h): the checkpoint's own, its weights in
// mtp.safetensors and its settings the target's mtp_config. A few decoder layers of the target's width fill a block
// in one pass, the pending token's row and then rows of one mask vector; a recurrent head then picks the drafts one
// after another, each from the token before it.
//
// It reads the target through hidden states at a few of its layer visits (a looped layer has a visit per pass): each
// drafter layer mixes them with weights of its own (the softmax of its row of fuse_weight), norms the mix and
// projects it into the context keys and values its block attends to, with projections kept for that (k_proj_ctx,
// v_proj_ctx). Those depend on the target alone, so the target's programs append them (append_context) to KV slots
// the block program shares. The head's own mix, at the last position the target ran, comes to the block program as
// an input (kContextRow): the target's programs output it for every row.
//
// A block sees all of itself and, in a local layer, the window of context before its start. Its rows are always the
// trained block's: the layers are bidirectional, so a shorter block would be other hidden states.
// Reference: llama.cpp's DSpark for Inkling 2 (common/dspark.cpp, as Unsloth's builds carry it) and vLLM's port.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph/graph.h"
#include "io/safetensors.h"
#include "models/model.h"

namespace xs::dspark {

// The output of a target's prefill and verify programs, and the input of the block program, that carries the
// head's context: [rows, hidden] and [1, hidden].
inline constexpr const char* kContextRow = "draft_context";

struct Config {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t block_size = 0;            // the pending token and block_size - 1 drafts
    int64_t markov_rank = 0;           // width of the head's state
    std::vector<int64_t> windows;      // per layer: the context before a block it sees (0: all of it)
    std::vector<int64_t> target_ids;   // the target's layer visits whose input is read; the last, its count of
                                       // visits, stands for its final normed state
    double rms_norm_eps = 1e-6;
    double rope_theta = 0;
};

// What a block reads of its target, in the draft program's graph.
struct Target {
    graph::Value embed;       // the token table, [rows, hidden]
    graph::Value embed_norm;  // the weight of the norm the target puts on an embedded token
    graph::Value head;        // the output layer, [rows, hidden]
    int64_t tokens = 0;       // the head's first rows that are tokens (the rest are padding)
};

// `mtp_config` of an Inkling 2 checkpoint's config.json, for a target of `hidden` and `visits` layer visits; refuses
// by name what is not implemented.
Config parse_config(const nlohmann::json& mtp_config, int64_t hidden, int64_t visits);

class Drafter final : public BlockDrafting {
public:
    // `weights` holds the drafter's tensors under `prefix` and outlives the drafter. `drafts` > 0: a round asks for
    // that many, and the head runs that many steps (the block keeps its rows).
    Drafter(Config config, const WeightStore& weights, std::string prefix, int64_t drafts = 0);

    const Config& config() const { return cfg_; }

    // In a target's prefill or verify program: appends the step's tokens to each layer's context. taps[i] is the
    // target's state for target_ids[i] ([kT, hidden]); `positions` are the tokens'. Returns the head's context of
    // every row, [kT, hidden]: the program's kContextRow output.
    graph::Value append_context(graph::Graph& g, std::span<const graph::Value> taps, graph::Value positions) const;
    // The draft program.
    void build_block(graph::Graph& g, const Target& target) const;

    int64_t block_drafts() const override { return drafts_; }
    void prepare_block(int32_t token, int64_t position, TensorMap& inputs) const override;
    std::span<const std::string> target_rows() const override { return target_rows_; }

private:
    graph::Value weight(graph::Graph& g, const std::string& name) const;
    graph::Value bias(graph::Graph& g, const std::string& name) const;
    graph::Value norm(graph::Graph& g, graph::Value x, const std::string& name) const;
    graph::Value inv_freq(graph::Graph& g) const;
    std::string slot_label(int64_t layer) const;
    // Each layer's mix of the taps and the head's, [kT, hidden] each: num_layers + 1 of them.
    std::vector<graph::Value> mixes(graph::Graph& g, std::span<const graph::Value> taps) const;

    Config cfg_;
    const WeightStore& weights_;
    std::string prefix_;
    int64_t drafts_ = 0;
    std::vector<std::string> target_rows_{kContextRow};
};

}  // namespace xs::dspark
