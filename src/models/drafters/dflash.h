#pragma once

// Block drafters of the DFlash family (BlockDrafting, models/model.h): DFlash, DFlash 2 and DSpine. A few decoder
// layers of the target's width fill a block of mask tokens after a pending token in one pass. They read the target
// through its hidden states: the outputs of a few of its layers, concatenated and projected (fc, hidden_norm), are the
// context every drafter layer attends to beside its own block. The context's keys and values depend on the target
// alone, so the target's own programs append them (append_context) to KV slots the block program shares.
//   DFlash    the block's argmax, position by position
//   DFlash 2  two-tap dynamic convolutions around attention and MLP, a block that sees all of itself, and a selector
//             that scores each position's candidates against the one chosen before
//   DSpine    after every layer each position passes a code of what it predicts to the next one (a causal block),
//             and the last of those writes is redone with the token actually chosen before
// Reference: github.com/z-lab/dflash (dflash/model.py), github.com/Tencent/DSpine (dspine_mlx).

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph/graph.h"
#include "io/safetensors.h"
#include "models/model.h"

namespace xs::dflash {

enum class Variant : uint8_t { DFlash, DFlash2, DSpine };

struct Config {
    Variant variant = Variant::DFlash;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t vocab_size = 0;
    int64_t num_target_layers = 0;          // of the target it was trained for
    std::vector<int64_t> target_layer_ids;  // whose outputs make the context, 0-based
    int64_t block_size = 0;                 // the pending token and block_size - 1 drafts
    int64_t mask_token_id = 0;
    bool causal = true;          // a block row sees the rows before it; else all of its block
    int64_t sliding_window = 0;  // keys this far behind a row are not seen (0: all are)
    double rms_norm_eps = 1e-6;
    double rope_theta = 0;
    // DFlash 2
    int64_t conv_taps = 0;
    int64_t conv_group = 0;
    int64_t selector_rank = 0;
    // DSpine: the code bridge
    int64_t code_size = 0;
    int64_t bridge_rank = 0;
    double bridge_eps = 0;
    // DFlash 2 and DSpine: candidates per position
    int64_t top_k = 0;
};

// Parses a drafter checkpoint's config.json; refuses by name what is not implemented.
Config parse_config(const nlohmann::json& config);
std::string_view variant_name(Variant v);  // "dflash", "dflash2", "dspine"

// The checkpoint's tensors (its stacked matrices also one by one, under "<name>.<index>").
std::unique_ptr<WeightStore> open_weights(const std::filesystem::path& dir);

class Drafter final : public BlockDrafting {
public:
    // `weights` holds the drafter's tensors under `prefix` and outlives the drafter. `drafts` > 0: a round asks for
    // that many, and a causal block (whose rows are the same whatever follows them) stops there; a block that sees
    // itself whole always runs the rows it was trained with.
    Drafter(Config config, const WeightStore& weights, std::string prefix, int64_t drafts = 0);

    const Config& config() const { return cfg_; }

    // In a target's prefill or verify program: appends the step's tokens to the drafter's context. layers[i] is the
    // target's hidden state after its layer target_layer_ids[i] ([kT, hidden]); `positions` are the tokens'.
    void append_context(graph::Graph& g, std::span<const graph::Value> layers, graph::Value positions) const;
    // The draft program. `embed` and `lm_head` are the target's ([vocab, hidden] weights of `g`).
    void build_block(graph::Graph& g, graph::Value embed, graph::Value lm_head) const;

    int64_t block_drafts() const override { return rows_ - 1; }
    void prepare_block(int32_t token, int64_t position, TensorMap& inputs) const override;

private:
    // What the tail of a DSpine block redoes: the last write's input, its scale and its gate queries, per row.
    struct LastWrite {
        graph::Value raw, rms, query;
    };
    graph::Value weight(graph::Graph& g, const std::string& name) const;
    HostTensor host_f32(const std::string& name) const;
    graph::Value norm(graph::Graph& g, graph::Value x, const std::string& name) const;
    graph::Value inv_freq(graph::Graph& g) const;
    std::string slot_label(int64_t layer) const;
    graph::Value attention(graph::Graph& g, int64_t layer, graph::Value x, graph::Value pos, graph::Value freq) const;
    // One half (0: before, 1: after the sublayer) of a grouped dynamic causal convolution over the block's rows.
    graph::Value convolve(graph::Graph& g, const std::string& conv, graph::Value x, graph::Value kernels, int half) const;
    // DSpine's write after `layer`: every row but the pending token's takes what the row before predicts.
    graph::Value bridge_write(graph::Graph& g, int64_t layer, graph::Value h, graph::Value anchor_code,
                              graph::Value rest, LastWrite* last) const;
    void selector_outputs(graph::Graph& g, graph::Value rows, graph::Value logits, graph::Value anchor) const;
    void refine_outputs(graph::Graph& g, graph::Value logits, graph::Value lm_head, graph::Value anchor,
                        const LastWrite& last) const;

    Config cfg_;
    const WeightStore& weights_;
    std::string prefix_;
    int64_t rows_ = 0;  // of a block: the pending token and the drafts
};

}  // namespace xs::dflash
