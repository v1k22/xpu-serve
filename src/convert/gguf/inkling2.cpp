// GGUF architectures "inkling2" and "inkling2_loop": Inkling 2 Mini as two converters write it, imported for
// models/inkling2/ as a pack in the released checkpoint's layout (names under model.llm.). "inkling2" is the layout
// of Unsloth's llama.cpp builds, "inkling2_loop" mainline llama.cpp's (conversion/inkling2.py, Inkling2LoopModel):
// the same tensors, with the differences a Flavor names (its keys, how it counts layers, the loop's tensors).
//
// What the converters changed, and what the import does about it:
// - names: blk.<i>.* for layer i (each layer once: the passes of the loop share its tensors), token_embd, output*,
//   loop_*; mapped back to the checkpoint's names. A layer's attention query is the checkpoint's
//   attn_res.attn_queries.<i - 1>.
// - inkling2's block_count counts layer visits (a looped layer once per pass) and its per-layer arrays have an entry
//   per visit; inkling2_loop's count layers: read back as the layers and local_layer_ids.
// - the messages' queries: inkling2 keeps the checkpoint's two tensors; inkling2_loop gives each looped layer its four
//   (those of its query, key, value and MLP messages, loop_msg_query), kept as layers.<i>.message_queries, which the
//   model reads in their place.
// - the experts' fused gate and up bank has its rows in halves (the gate rows, then the up rows) where the checkpoint
//   alternates them. A block format's rows cannot be moved, so config.json says so (gate_up_rows) and the model reads
//   the bank as it is.
// - a bank the release holds in NVFP4 is kept as NVFP4 blocks without each expert's factor, which is a tensor of its
//   own (<bank>.scale): kept as <bank>_factor, which the model multiplies the expert's output by. A file that
//   requantized such a bank kept the factor too.
// - the stored taps of the full-attention layers' convolutions are flipped and transposed, the profiles of the
//   relative bias transposed: put back.
// - convolution kernels lost their middle dimension of 1: put back (a shape).
// - the vision tower is not in the file (llama.cpp writes it to an mmproj file of its own): the pack is the text model.

#include <array>
#include <cmath>
#include <set>

#include "convert/gguf/architecture.h"
#include "convert/gguf/layout.h"
#include "util/error.h"

namespace xs::gguf_import {
namespace {

const std::string kPrefix = "model.llm.";

// What tells the two layouts apart.
struct Flavor {
    std::string_view name;  // general.architecture, and its metadata's namespace
    bool per_visit;         // block_count and the per-layer arrays count layer visits, not layers
    std::string_view passes, loop_start, loop_end, loop_gate_rank;  // the loop's keys
    // A looped layer's message gates (attention's query, key, value, the MLP's), as the file names them:
    // loop_<word><suffix>down.weight.
    std::array<std::string_view, 4> gates;
    std::string_view gate_suffix;
};

const Flavor kUnsloth{"inkling2", true, "num_loops", "loop.core_start", "loop.core_end", "loop.gate_rank",
                      {"attn", "k", "v", "mlp"}, "_"};
const Flavor kMainline{"inkling2_loop", false, "loop.count", "loop.start", "loop.end", "loop.gate_rank",
                       {"q", "k", "v", "ffn"}, "_ctrl_"};
// The same gates as the checkpoint names them: controllers.<word>.down_weight.
constexpr std::array<std::string_view, 4> kCheckpointGates = {"attn", "k", "v", "mlp"};

// The keys both layouts write, beside a layout's own loop keys.
const std::set<std::string_view> kSharedKeys = {
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "expert_feed_forward_length",
    "expert_count",
    "expert_used_count",
    "expert_weights_scale",
    "vocab_size",
    "vocab_size_unpadded",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.key_length_swa",
    "attention.value_length_swa",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "attention.relative_length",
    "attention.relative_extent",
    "attention.qk_clip",
    "attention.relative_clip",
    "attention.log_scale_floor",
    "attention.log_scale_alpha",
    "rope.dimension_count",
    "shortconv.l_cache",
    "activation.situ_beta",
    "activation.situ_linear_beta",
    "attn_res.block_size",
    "final_logit_softcapping",
};

std::set<std::string_view> known_keys(const Flavor& fl) {
    std::set<std::string_view> keys = kSharedKeys;
    keys.insert({fl.passes, fl.loop_start, fl.loop_end, fl.loop_gate_rank});
    if (!fl.per_visit) keys.insert("loop.varying");
    return keys;
}

std::string key(const Flavor& fl, std::string_view rest) {
    return str_cat(fl.name, ".", rest);
}

struct Dimensions {
    int64_t vocab = 0, tokens = 0, hidden = 0, layers = 0;
    int64_t heads = 0, kv_heads = 0, head_dim = 0;
    int64_t rel_width = 0, rel_extent = 0, window = 0, conv_taps = 0, block_layers = 0;
    int64_t experts = 0, experts_per_token = 0, expert_width = 0;
    int64_t loop_start = 0, loop_end = 0, passes = 0, gate_rank = 0;
    std::vector<bool> sliding;  // per layer

    bool looped(int64_t i) const { return i >= loop_start && i < loop_end; }
    int64_t visits() const { return layers + (passes - 1) * (loop_end - loop_start); }
};

// The layer a visit runs: the layers before the loop, the looped ones once per pass, the layers after it.
int64_t layer_of_visit(const Dimensions& d, int64_t visit) {
    const int64_t span = d.loop_end - d.loop_start;
    if (visit < d.loop_start) return visit;
    if (visit < d.loop_start + d.passes * span) return d.loop_start + (visit - d.loop_start) % span;
    return visit - (d.passes - 1) * span;
}

// A value every entry of a per-layer array (`entries`: of layers or of visits) must agree on.
int64_t same_per_entry(const gguf::File& f, const Flavor& fl, std::string_view rest, int64_t entries) {
    const std::vector<int64_t> values = f.get(key(fl, rest)).ints();
    if (static_cast<int64_t>(values.size()) != entries) {
        fail(fl.name, ": ", key(fl, rest), " has ", values.size(), " entries for ", entries,
             fl.per_visit ? " layer visits" : " layers");
    }
    for (const int64_t v : values) {
        if (v != values[0] || v <= 0) {
            fail(fl.name, ": ", key(fl, rest), " holds ", v, " beside ", values[0],
                 ": layers of different head counts are not supported");
        }
    }
    return values[0];
}

Dimensions dimensions(const gguf::File& f, const Flavor& fl) {
    for (const std::string_view loop : {fl.passes, fl.loop_start, fl.loop_end, fl.loop_gate_rank}) {
        if (f.find(key(fl, loop)) == nullptr) {
            fail(fl.name, ": the file has no ", key(fl, loop), ": an Inkling 2 without looped layers is not supported");
        }
    }
    if (const gguf::Value* v = f.find(key(fl, "loop.varying")); v != nullptr && v->as_bool()) {
        fail(fl.name, ": ", key(fl, "loop.varying"), " true (a variable number of passes) is not supported");
    }
    Dimensions d;
    const int64_t count = positive(f, fl.name, key(fl, "block_count"));
    d.passes = positive(f, fl.name, key(fl, fl.passes));
    d.loop_start = positive(f, fl.name, key(fl, fl.loop_start));
    d.loop_end = positive(f, fl.name, key(fl, fl.loop_end));
    d.gate_rank = positive(f, fl.name, key(fl, fl.loop_gate_rank));
    d.layers = fl.per_visit ? count - (d.passes - 1) * (d.loop_end - d.loop_start) : count;
    if (d.loop_end <= d.loop_start || d.layers < d.loop_end) {
        fail(fl.name, ": ", count, fl.per_visit ? " layer visits" : " layers", " do not hold ", d.passes,
             " passes over layers ", d.loop_start, " to ", d.loop_end);
    }
    const int64_t entries = fl.per_visit ? d.visits() : d.layers;  // of a per-layer array
    d.hidden = positive(f, fl.name, key(fl, "embedding_length"));
    d.vocab = positive(f, fl.name, key(fl, "vocab_size"));
    d.tokens = positive(f, fl.name, key(fl, "vocab_size_unpadded"));
    if (f.get("tokenizer.ggml.tokens").size() != static_cast<uint64_t>(d.tokens) || d.tokens > d.vocab) {
        fail(fl.name, ": the tokenizer has ", f.get("tokenizer.ggml.tokens").size(), " tokens, ",
             key(fl, "vocab_size_unpadded"), " is ", d.tokens, " and ", key(fl, "vocab_size"), " ", d.vocab);
    }
    d.heads = same_per_entry(f, fl, "attention.head_count", entries);
    d.kv_heads = same_per_entry(f, fl, "attention.head_count_kv", entries);
    d.head_dim = positive(f, fl.name, key(fl, "attention.key_length"));
    for (const char* same : {"attention.value_length", "attention.key_length_swa", "attention.value_length_swa"}) {
        if (const int64_t v = positive(f, fl.name, key(fl, same)); v != d.head_dim) {
            fail(fl.name, ": ", key(fl, same), " ", v, " differs from attention.key_length ", d.head_dim,
                 ", which is not supported");
        }
    }
    if (const int64_t rope = f.get_int(key(fl, "rope.dimension_count")); rope != 0) {
        fail(fl.name, ": ", key(fl, "rope.dimension_count"), " is ", rope, "; the model has no rotary positions");
    }
    d.rel_width = positive(f, fl.name, key(fl, "attention.relative_length"));
    d.rel_extent = positive(f, fl.name, key(fl, "attention.relative_extent"));
    d.window = positive(f, fl.name, key(fl, "attention.sliding_window"));
    d.conv_taps = positive(f, fl.name, key(fl, "shortconv.l_cache"));
    d.block_layers = positive(f, fl.name, key(fl, "attn_res.block_size"));
    d.experts = positive(f, fl.name, key(fl, "expert_count"));
    d.experts_per_token = positive(f, fl.name, key(fl, "expert_used_count"));
    d.expert_width = positive(f, fl.name, key(fl, "expert_feed_forward_length"));
    if (const int64_t ffn = positive(f, fl.name, key(fl, "feed_forward_length")); ffn != d.expert_width) {
        fail(fl.name, ": ", key(fl, "feed_forward_length"), " ", ffn, " differs from the experts' ", d.expert_width);
    }

    const std::vector<bool> pattern = f.get(key(fl, "attention.sliding_window_pattern")).bools();
    if (static_cast<int64_t>(pattern.size()) != entries) {
        fail(fl.name, ": attention.sliding_window_pattern has ", pattern.size(), " entries for ", entries,
             fl.per_visit ? " layer visits" : " layers");
    }
    d.sliding.assign(static_cast<size_t>(d.layers), false);
    std::vector<bool> seen(static_cast<size_t>(d.layers), false);
    for (int64_t v = 0; v < entries; ++v) {
        const auto layer = static_cast<size_t>(fl.per_visit ? layer_of_visit(d, v) : v);
        if (seen[layer] && d.sliding[layer] != pattern[static_cast<size_t>(v)]) {
            fail(fl.name, ": attention.sliding_window_pattern gives layer ", layer, " two kinds on two passes");
        }
        seen[layer] = true;
        d.sliding[layer] = pattern[static_cast<size_t>(v)];
    }
    return d;
}

nlohmann::ordered_json text_config(const gguf::File& f, const Flavor& fl, const Dimensions& d) {
    const auto number = [&](std::string_view rest) { return config_number(f.get(key(fl, rest))); };
    nlohmann::ordered_json t;
    t["model_type"] = "inkling2_loop_text";
    t["model_max_length"] = positive(f, fl.name, key(fl, "context_length"));
    t["hidden_size"] = d.hidden;
    t["num_hidden_layers"] = d.layers;
    t["vocab_size"] = d.vocab;
    t["unpadded_vocab_size"] = d.tokens;
    t["num_attention_heads"] = d.heads;
    t["num_key_value_heads"] = d.kv_heads;
    t["head_dim"] = d.head_dim;
    t["swa_num_attention_heads"] = d.heads;
    t["swa_num_key_value_heads"] = d.kv_heads;
    t["swa_head_dim"] = d.head_dim;
    t["sliding_window_size"] = d.window;
    t["d_rel"] = d.rel_width;
    t["rel_extent"] = d.rel_extent;
    t["q_bias"] = false;
    t["o_bias"] = false;
    t["log_scaling_n_floor"] = positive(f, fl.name, key(fl, "attention.log_scale_floor"));
    t["log_scaling_alpha"] = number("attention.log_scale_alpha");
    t["rms_norm_eps"] = number("attention.layer_norm_rms_epsilon");
    t["use_value_norm_no_gamma"] = true;
    t["qk_l2_clip_threshold"] = number("attention.qk_clip");
    t["rel_l2_clip_threshold"] = number("attention.relative_clip");
    nlohmann::ordered_json local = nlohmann::ordered_json::array();
    for (int64_t i = 0; i < d.layers; ++i) {
        if (d.sliding[static_cast<size_t>(i)]) local.push_back(i);
    }
    t["local_layer_ids"] = std::move(local);
    t["sconv_kernel_size"] = d.conv_taps;
    t["final_logit_softcapping"] = number("final_logit_softcapping");
    t["n_layers_per_attn_res_block"] = d.block_layers;
    t["attn_res_normalize_values"] = true;
    t["n_routed_experts"] = d.experts;
    t["num_experts_per_tok"] = d.experts_per_token;
    t["n_shared_experts"] = 0;
    t["intermediate_size"] = d.expert_width;
    t["mlp_gate_softcap"] = number("activation.situ_beta");
    t["mlp_up_softcap"] = number("activation.situ_linear_beta");
    t["route_scale"] = number("expert_weights_scale");
    t["use_gate_bias"] = true;
    t["gate_up_rows"] = "halves";
    t["loop_config"] = {{"core_start", d.loop_start},
                        {"core_end", d.loop_end},
                        {"num_passes", d.passes},
                        {"gate_rank", d.gate_rank},
                        {"is_varloop", false}};
    return t;
}

// The profiles of bias by distance: stored [extent, width], the checkpoint's [width, extent].
const FloatReorder kTranspose{"a transpose", [](const std::vector<float>& v, const Dims& stored) {
                                  const int64_t rows = stored[0], cols = stored[1];
                                  std::vector<float> out(v.size());
                                  for (int64_t r = 0; r < rows; ++r) {
                                      for (int64_t c = 0; c < cols; ++c)
                                          out[static_cast<size_t>(c * rows + r)] = v[static_cast<size_t>(r * cols + c)];
                                  }
                                  return out;
                              }};
// A dynamic convolution's stored taps: [channels, taps] with the current token's tap last in the file, [taps,
// channels] with it first in the checkpoint.
const FloatReorder kTaps{"a transpose and a flip of the taps", [](const std::vector<float>& v, const Dims& stored) {
                             const int64_t channels = stored[0], taps = stored[1];
                             std::vector<float> out(v.size());
                             for (int64_t t = 0; t < taps; ++t) {
                                 for (int64_t c = 0; c < channels; ++c) {
                                     out[static_cast<size_t>(t * channels + c)] =
                                         v[static_cast<size_t>(c * taps + (taps - 1 - t))];
                                 }
                             }
                             return out;
                         }};

Entries layer(const Flavor& fl, const Dimensions& d, int64_t i) {
    const bool sliding = d.sliding[static_cast<size_t>(i)];
    const int64_t h = d.hidden, q = d.heads * d.head_dim, kv = d.kv_heads * d.head_dim, taps = d.conv_taps;
    const std::string L = kPrefix + "layers." + std::to_string(i) + ".";
    const std::string A = L + "attn.";
    const std::string X = L + "mlp.experts.";
    Entries e;
    e["attn_q.weight"] = one({q, h}, A + "wq_du.weight");
    e["attn_k.weight"] = one({kv, h}, A + "wk_dv.weight");
    e["attn_v.weight"] = one({kv, h}, A + "wv_dv.weight");
    e["attn_output.weight"] = one({h, q}, A + "wo_ud.weight");
    e["attn_gate.weight"] = one({q, h}, A + "gate_attn.weight");
    e["attn_rel.weight"] = one({d.heads * d.rel_width, h}, A + "wr_du.weight");
    e["attn_sink_proj.weight"] = one({d.heads, h}, A + "attn_sink_proj.weight");
    e["attn_q_clip.weight"] = one({d.heads, d.head_dim}, A + "q_l2_clip.scale");
    e["attn_rel_clip.weight"] = one({d.heads, d.rel_width}, A + "rel_l2_clip.scale");
    // A sliding layer's bias reaches as far as its window, a full layer's as far as the extent.
    const int64_t extent = sliding ? d.window : d.rel_extent;
    e["attn_rel_bias.weight"] =
        reordered({extent, d.rel_width}, A + "rel_logits_proj.proj", {d.rel_width, extent}, &kTranspose);
    for (const char* which : {"k", "v"}) {
        const std::string conv = str_cat("attn_", which, "_conv");
        const std::string name = str_cat(A, which, "_sconv.");
        if (sliding) {
            e[conv + ".weight"] = Entry{{kv, taps}, {Piece{name + "weight", {kv, 1, taps}}}};
        } else {
            e[conv + ".weight"] = reordered({kv, taps}, name + "static_w", {taps, kv}, &kTaps);
            e[conv + "_proj.weight"] = one({d.kv_heads * taps, h}, name + "weight_proj");
        }
    }
    if (sliding) {
        e["attn_norm.weight"] = one({h}, L + "attn_norm.weight");
    } else {
        for (const char* which : {"q", "k", "v"}) {
            e[str_cat("attn_", which, "_inp_norm.weight")] = one({h}, str_cat(L, which, "_input_norm.weight"));
        }
    }
    if (i > 0) {  // the first layer reads the embedded tokens as they are
        const std::string name = kPrefix + "attn_res.attn_queries." + std::to_string(i - 1);
        e["attn_res_query.weight"] = sliding ? one({h}, name) : one({3, h}, name);
    }
    e["attn_post_conv.weight"] = Entry{{h, taps}, {Piece{L + "attn_sconv.weight", {h, 1, taps}}}};
    e["ffn_post_conv.weight"] = Entry{{h, taps}, {Piece{L + "mlp_sconv.weight", {h, 1, taps}}}};
    e["ffn_res_query.weight"] = one({h}, kPrefix + "attn_res.mlp_queries." + std::to_string(i));
    e["ffn_norm.weight"] = one({h}, L + "mlp_norm.weight");
    e["ffn_gate_inp.weight"] = one({d.experts, h}, L + "mlp.gate.weight");
    e["exp_probs_b.bias"] = one({d.experts}, L + "mlp.gate.bias");
    e["ffn_gate_up_exps.weight"] = one({d.experts, 2 * d.expert_width, h}, X + "w13_weight");
    e["ffn_down_exps.weight"] = one({d.experts, h, d.expert_width}, X + "w2_weight");
    e["ffn_gate_up_exps.scale"] = one({d.experts}, X + "w13_weight_factor");
    e["ffn_down_exps.scale"] = one({d.experts}, X + "w2_weight_factor");
    e["ffn_gate_up_exps.scale"].optional = e["ffn_down_exps.scale"].optional = true;
    if (d.looped(i)) {
        for (size_t m = 0; m < kCheckpointGates.size(); ++m) {
            const std::string file = str_cat("loop_", fl.gates[m], fl.gate_suffix);
            const std::string name = str_cat(L, "controllers.", kCheckpointGates[m], ".");
            e[file + "down.weight"] = one({d.gate_rank, 2 * h}, name + "down_weight");
            e[file + "forget.weight"] = one({h, d.gate_rank}, name + "forget_weight");
            e[file + "gain.weight"] = one({h, d.gate_rank}, name + "gain_weight");
        }
        if (!fl.per_visit) e["loop_msg_query.weight"] = one({4, h}, L + "message_queries");
    }
    return e;
}

Layout layout(const Flavor& fl, const Dimensions& d) {
    Layout l;
    const int64_t h = d.hidden, message_queries = 2 * (d.loop_end - d.loop_start);
    l.top["token_embd.weight"] = one({d.vocab, h}, kPrefix + "embed.weight");
    l.top["token_embd_norm.weight"] = one({h}, kPrefix + "embed_norm.weight");
    l.top["output.weight"] = one({d.vocab, h}, kPrefix + "unembed.weight");
    l.top["output_norm.weight"] = one({h}, kPrefix + "norm.weight");
    l.top["output_res_query.weight"] = one({h}, kPrefix + "attn_res.output_query");
    if (fl.per_visit) {
        l.top["loop_res_query.weight"] = one({message_queries, h}, kPrefix + "middle_cycle_cross_pass_queries");
        l.top["loop_kv_res_query.weight"] = one({message_queries, h}, kPrefix + "middle_cycle_cross_pass_kv_queries");
    }
    for (int64_t i = 0; i < d.layers; ++i)
        l.blocks.push_back(layer(fl, d, i));
    l.block_kind = [d](int64_t i) {
        return str_cat(d.sliding[static_cast<size_t>(i)] ? "sliding" : "full", " attention",
                       d.looped(i) ? ", looped" : "");
    };
    return l;
}

// ---- The drafter (DSpark), a file of its own: general.architecture "dflash", in one of two layouts.
//
// Unsloth's (dflash.dspark.markov_rank) keeps the checkpoint's tensors. Mainline llama.cpp's (conversion/inkling2.py,
// Inkling2DSparkModel; dflash.has_markov_context) changed some, and the import puts back what it can:
// - fuse_weight's rows through their softmax, a tensor per reader (blk.<i>.ctx_mix, markov_ctx_mix): their logs, a
//   row of fuse_weight each (the softmax of a row's logs is the row), kept per reader.
// - the head's hidden_out folded through the target's output layer: markov_w2 = unembed x hidden_out and its bias
//   unembed x bias, over the tokens' rows. The fold cannot be undone without the output layer the converter read, so
//   it is kept (markov_head.fold.*) and the drafter adds it to the logits.
// - markov_w1 without its padding rows (a padding token is never drafted), mask_embed copied to each draft's row
//   (the drafter checks the rows are one vector), and the target's embedding norm copied in (not imported).
// - attention.sliding_window one more than the checkpoint's (what llama.cpp's window counts).

constexpr std::string_view kDrafter = "dflash";
// Where a pack keeps the drafter's tensors, under the names of the checkpoint's mtp.safetensors.
const std::string kDrafterPrefix = "mtp.";

const std::set<std::string_view> kDrafterKeys = {
    "block_count", "context_length", "embedding_length", "feed_forward_length", "attention.head_count",
    "attention.head_count_kv", "attention.key_length", "attention.value_length", "attention.layer_norm_rms_epsilon",
    "attention.causal", "attention.sliding_window", "attention.sliding_window_pattern", "rope.freq_base",
    "rope.dimension_count", "vocab_size", "vocab_size_unpadded", "block_size", "target_layers", "dspark.markov_rank",
    "sample_from_anchor", "has_confidence_head", "final_logit_softcapping", "has_markov_context",
};

std::string drafter_key(std::string_view rest) {
    return str_cat(kDrafter, ".", rest);
}

struct DrafterDimensions {
    bool mainline = false;  // mainline llama.cpp's layout
    int64_t layers = 0, hidden = 0, width = 0, heads = 0, kv_heads = 0, head_dim = 0, rank = 0, taps = 0, block = 0;
};

// A row of fuse_weight from its softmax.
const FloatMap kLogOfMix{"a row of fuse_weight = log(its softmax)", [](float a) { return std::log(a); }};

// The drafter file `f` against the model's: mtp_config as the released config.json has it. Refuses a drafter of
// another kind or another model by name.
nlohmann::ordered_json drafter_config(const gguf::File& f, const gguf::File& model, const Flavor& fl,
                                      const Dimensions& d, DrafterDimensions& out) {
    const std::string_view arch = f.get_string("general.architecture");
    out.mainline = f.find(drafter_key("has_markov_context")) != nullptr;
    if (arch != kDrafter || (!out.mainline && f.find(drafter_key("dspark.markov_rank")) == nullptr)) {
        fail(fl.name, ": --mtp: the file is not Inkling 2's drafter (general.architecture '", arch, "'; expected '",
             kDrafter, "' with ", drafter_key("dspark.markov_rank"), " or ", drafter_key("has_markov_context"), ")");
    }
    refuse_unknown_keys(f, kDrafter, kDrafterKeys);
    const auto positive_of = [&](std::string_view rest) { return positive(f, kDrafter, drafter_key(rest)); };
    const auto refuse_true = [&](std::string_view rest) {
        if (const gguf::Value* v = f.find(drafter_key(rest)); v != nullptr && v->as_bool()) {
            fail(fl.name, ": --mtp: ", drafter_key(rest), " true is not supported");
        }
    };
    refuse_true("attention.causal");
    refuse_true("sample_from_anchor");
    refuse_true("has_confidence_head");
    if (out.mainline && !f.get(drafter_key("has_markov_context")).as_bool()) {
        fail(fl.name, ": --mtp: ", drafter_key("has_markov_context"), " false (a head without the target's context) ",
             "is not Inkling 2's drafter");
    }

    out.layers = positive_of("block_count");
    out.hidden = positive_of("embedding_length");
    out.width = positive_of("feed_forward_length");
    out.heads = positive_of("attention.head_count");
    out.kv_heads = positive_of("attention.head_count_kv");
    out.head_dim = positive_of("attention.key_length");
    out.block = positive_of("block_size");
    if (out.mainline) {  // the width of markov_w1's rows
        const gguf::TensorInfo* table = f.find_tensor("markov_w1.weight");
        XS_CHECK(table != nullptr && table->dims.size() == 2, fl.name, ": --mtp: the file lacks markov_w1.weight");
        out.rank = table->shape().back();
    } else {
        out.rank = positive_of("dspark.markov_rank");
    }
    const std::vector<int64_t> taps = f.get(drafter_key("target_layers")).ints();
    out.taps = static_cast<int64_t>(taps.size());
    // The model the drafter was made for: the same width, vocabulary and softcap, and as many layer visits as the
    // last state it reads names.
    const int64_t visits = d.visits();
    const auto same = [&](std::string_view what, double mine, double models) {
        if (mine != models) {
            fail(fl.name, ": --mtp: the drafter's ", what, " is ", mine, ", the model's ", models,
                 ": it drafts for another model");
        }
    };
    same("embedding_length", static_cast<double>(out.hidden), static_cast<double>(d.hidden));
    if (!out.mainline) {  // mainline's: the rows of markov_w1 and markov_w2, which its layout checks
        same("vocab_size", static_cast<double>(positive_of("vocab_size")), static_cast<double>(d.vocab));
        same("vocab_size_unpadded", static_cast<double>(positive_of("vocab_size_unpadded")),
             static_cast<double>(d.tokens));
    }
    same("final_logit_softcapping", f.get_float(drafter_key("final_logit_softcapping")),
         model.get_float(key(fl, "final_logit_softcapping")));
    same("last target layer", static_cast<double>(taps.empty() ? -1 : taps.back()), static_cast<double>(visits));
    if (f.get_int(drafter_key("attention.value_length")) != out.head_dim ||
        f.get_int(drafter_key("rope.dimension_count")) != out.head_dim) {
        fail(fl.name, ": --mtp: attention.value_length and rope.dimension_count must equal attention.key_length ",
             out.head_dim);
    }
    const std::vector<gguf::Value> pattern = f.get(drafter_key("attention.sliding_window_pattern")).elements();
    if (static_cast<int64_t>(pattern.size()) != out.layers) {
        fail(fl.name, ": --mtp: attention.sliding_window_pattern has ", pattern.size(), " entries for ", out.layers,
             " layers");
    }
    std::vector<int64_t> local;
    for (size_t i = 0; i < pattern.size(); ++i) {
        if (pattern[i].as_bool()) local.push_back(static_cast<int64_t>(i));
    }

    nlohmann::ordered_json c;
    c["hidden_size"] = out.hidden;
    c["num_hidden_layers"] = out.layers;
    c["num_attention_heads"] = out.heads;
    c["num_key_value_heads"] = out.kv_heads;
    c["head_dim"] = out.head_dim;
    c["intermediate_size"] = out.width;
    c["max_position_embeddings"] = positive_of("context_length");
    c["q_bias"] = false;
    c["o_bias"] = false;
    c["rms_norm_eps"] = config_number(f.get(drafter_key("attention.layer_norm_rms_epsilon")));
    c["rope_theta"] = config_number(f.get(drafter_key("rope.freq_base")));
    c["local_layer_ids"] = local;
    const int64_t window = positive_of("attention.sliding_window") - (out.mainline ? 1 : 0);
    XS_CHECK(window > 0, fl.name, ": --mtp: ", drafter_key("attention.sliding_window"), " leaves no window");
    c["sliding_window_size"] = window;
    c["block_size"] = out.block;
    c["markov_rank"] = out.rank;
    c["markov_head_type"] = "rnn_hidden";  // the one head the converter writes (joint_proj, dspark_hidden_out)
    c["target_layer_ids"] = taps;
    return c;
}

// The drafter's file in its layout (`d`), for a target of `model`.
Layout drafter_layout(const DrafterDimensions& d, const Dimensions& model) {
    const std::string P = kDrafterPrefix;
    const std::string H = P + "markov_head.";
    const int64_t h = d.hidden, q = d.heads * d.head_dim, kv = d.kv_heads * d.head_dim, rank = d.rank;
    Layout l;
    l.top["output_norm.weight"] = one({h}, P + "norm.weight");
    l.top["joint_proj.weight"] = one({3 * rank, 2 * rank + 2 * h}, H + "rnn_proj.weight");
    l.top["joint_proj.bias"] = one({3 * rank}, H + "rnn_proj.bias");
    if (d.mainline) {
        l.top["markov_ctx_mix.weight"] = one({d.taps}, H + "fuse_weight", &kLogOfMix);
        l.top["markov_ctx_norm.weight"] = one({h}, H + "dflare_norm.weight");
        l.top["markov_w1.weight"] = one({model.tokens, rank}, H + "markov_w1.weight");
        l.top["markov_w2.weight"] = one({model.tokens, rank}, H + "fold.weight");
        l.top["markov_w2.bias"] = one({model.tokens}, H + "fold.bias");
        l.top["mask_embd.weight"] = one({d.block - 1, h}, P + "mask_embed");
    } else {
        l.top["dspark_fuse.weight"] = one({d.layers + 1, d.taps}, P + "fuse_weight");
        l.top["dspark_norm.weight"] = one({h}, H + "dflare_norm.weight");
        l.top["markov_w1.weight"] = one({model.vocab, rank}, H + "markov_w1.weight");
        l.top["dspark_hidden_out.weight"] = one({h, rank}, H + "hidden_out.weight");
        l.top["dspark_hidden_out.bias"] = one({h}, H + "hidden_out.bias");
        l.top["mask_embd.weight"] = one({h}, P + "mask_embed");
    }
    for (int64_t i = 0; i < d.layers; ++i) {
        const std::string L = P + "layers." + std::to_string(i) + ".";
        const std::string A = L + "self_attn.";
        Entries e;
        e["attn_norm.weight"] = one({h}, L + "input_layernorm.weight");
        e["attn_q.weight"] = one({q, h}, A + "q_proj.weight");
        e["attn_k.weight"] = one({kv, h}, A + "k_proj.weight");
        e["attn_v.weight"] = one({kv, h}, A + "v_proj.weight");
        e["attn_output.weight"] = one({h, q}, A + "o_proj.weight");
        e["attn_q_norm.weight"] = one({d.head_dim}, A + "q_norm.weight");
        e["attn_k_norm.weight"] = one({d.head_dim}, A + "k_norm.weight");
        e["attn_k_ctx.weight"] = one({kv, h}, A + "k_proj_ctx.weight");
        e["attn_v_ctx.weight"] = one({kv, h}, A + "v_proj_ctx.weight");
        e["attn_k_ctx_norm.weight"] = one({d.head_dim}, A + "k_norm_ctx.weight");
        e[d.mainline ? "ctx_norm.weight" : "dflare_norm.weight"] = one({h}, L + "dflare_norm.weight");
        if (d.mainline) e["ctx_mix.weight"] = one({d.taps}, L + "fuse_weight", &kLogOfMix);
        e["ffn_norm.weight"] = one({h}, L + "post_attention_layernorm.weight");
        e["ffn_gate.weight"] = one({d.width, h}, L + "mlp.gate_proj.weight");
        e["ffn_up.weight"] = one({d.width, h}, L + "mlp.up_proj.weight");
        e["ffn_down.weight"] = one({h, d.width}, L + "mlp.down_proj.weight");
        l.blocks.push_back(std::move(e));
    }
    l.block_kind = [](int64_t) { return std::string("a drafter layer"); };
    return l;
}

// The vision tower's file (llama.cpp's mmproj): general.architecture clip, its projector type inkling2v.
constexpr std::string_view kProjector = "clip";
constexpr std::string_view kProjectorType = "inkling2v";
// Where a pack keeps the tower, under the checkpoint's names.
const std::string kVisionPrefix = "model.visual.";

// image_size is llama.cpp's own bound on an image; the processor here pads an image and does not resize it.
const std::set<std::string_view> kProjectorKeys = {
    "has_vision_encoder", "vision.projection_dim", "vision.image_size", "vision.patch_size",
    "vision.embedding_length", "vision.feed_forward_length", "vision.block_count", "vision.attention.head_count",
    "vision.image_mean", "vision.image_std", "vision.projector_type", "vision.attention.layer_norm_epsilon",
    "vision.spatial_merge_size",
};

struct VisionDimensions {
    int64_t hidden = 0, width = 0, layers = 0, heads = 0, patch = 0, merge = 0, grid = 0, projection = 0, out = 0;
};

// The patch embedding: stored as a convolution's weight [hidden, channels, y, x], the checkpoint's a linear layer's
// over a patch as (y, x, channel).
const FloatReorder kPatchLinear{"a convolution's weight as a linear layer's",
                                [](const std::vector<float>& v, const Dims& stored) {
                                    const int64_t n = stored[0], c = stored[1], h = stored[2], w = stored[3];
                                    std::vector<float> out(v.size());
                                    for (int64_t o = 0; o < n; ++o) {
                                        for (int64_t ch = 0; ch < c; ++ch) {
                                            for (int64_t y = 0; y < h; ++y) {
                                                for (int64_t x = 0; x < w; ++x) {
                                                    out[static_cast<size_t>(((o * h + y) * w + x) * c + ch)] =
                                                        v[static_cast<size_t>(((o * c + ch) * h + y) * w + x)];
                                                }
                                            }
                                        }
                                    }
                                    return out;
                                }};

// The projector file `f` against the model's: vision_config as the released config.json has it. Refuses by name a
// file that is not Inkling 2's tower, or another model's.
nlohmann::ordered_json vision_config(const gguf::File& f, const Flavor& fl, const Dimensions& d,
                                     VisionDimensions& v) {
    const std::string_view arch = f.get_string("general.architecture");
    const gguf::Value* type = f.find("clip.vision.projector_type");
    if (arch != kProjector || type == nullptr || type->as_string() != kProjectorType) {
        fail(fl.name, ": --mmproj: the file is not Inkling 2's vision tower (general.architecture '", arch,
             "', clip.vision.projector_type '", type != nullptr ? type->as_string() : std::string_view("none"),
             "'; expected '", kProjector, "' and '", kProjectorType, "')");
    }
    refuse_unknown_keys(f, kProjector, kProjectorKeys);
    const auto positive_of = [&](std::string_view rest) {
        return positive(f, kProjector, str_cat("clip.vision.", rest));
    };
    v.hidden = positive_of("embedding_length");
    v.width = positive_of("feed_forward_length");
    v.layers = positive_of("block_count");
    v.heads = positive_of("attention.head_count");
    v.patch = positive_of("patch_size");
    v.merge = positive_of("spatial_merge_size");
    v.out = positive_of("projection_dim");
    XS_CHECK(v.out == d.hidden, fl.name, ": --mmproj: the tower projects to ", v.out, ", the model's hidden size is ",
             d.hidden, ": another model's file");
    // The processor's constants are the checkpoint's; a file that says otherwise is not read as it means.
    for (const char* key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
        const gguf::Value* a = f.find(key);
        XS_CHECK(a != nullptr, fl.name, ": --mmproj: the file has no ", key);
        for (const double m : a->floats()) {
            XS_CHECK(m == 0.5, fl.name, ": --mmproj: ", key, " is not 0.5 a channel");
        }
    }
    const gguf::TensorInfo* table = f.find_tensor("v.position_embd.weight");
    const gguf::TensorInfo* first = f.find_tensor("mm.1.weight");
    XS_CHECK(table != nullptr && first != nullptr && table->dims.size() == 2 && first->dims.size() == 2, fl.name,
             ": --mmproj: the file lacks v.position_embd.weight or mm.1.weight");
    v.grid = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(table->rows()))));
    XS_CHECK(v.grid * v.grid == table->rows(), fl.name, ": --mmproj: v.position_embd.weight has ", table->rows(),
             " rows, not a square grid");
    v.projection = first->rows();
    nlohmann::ordered_json c;
    c["model_type"] = "inkling2_vision";
    c["text_hidden_size"] = v.out;
    c["native_patch_size"] = v.patch;
    c["adapter_merge_size"] = v.merge;
    c["projection_hidden_dim"] = v.projection;
    c["projector_bias"] = false;
    c["hidden_size"] = v.hidden;
    c["num_hidden_layers"] = v.layers;
    c["num_attention_heads"] = v.heads;
    c["intermediate_size"] = v.width;
    c["posemb_grid_size"] = v.grid;
    c["layer_norm_eps"] = config_number(f.get("clip.vision.attention.layer_norm_epsilon"));
    return c;
}

Layout vision_layout(const VisionDimensions& v) {
    const std::string B = kVisionPrefix + "backbone.";
    const int64_t h = v.hidden;
    Layout l;
    l.top["v.class_embd"] = one({h}, B + "embeddings.class_embedding");
    l.top["v.class_position_embd.weight"] = one({h}, B + "embeddings.cls_position_embedding");
    l.top["v.patch_embd.weight"] = reordered({h, 3, v.patch, v.patch}, B + "embeddings.patch_embedding.weight",
                                             {h, 3 * v.patch * v.patch}, &kPatchLinear);
    l.top["v.position_embd.weight"] = one({v.grid * v.grid, h}, B + "embeddings.position_embedding");
    l.top["v.pre_ln.weight"] = one({h}, B + "ln_pre.weight");
    l.top["v.pre_ln.bias"] = one({h}, B + "ln_pre.bias");
    for (int64_t i = 0; i < v.layers; ++i) {
        const std::string from = "v.blk." + std::to_string(i) + ".", to = B + "layers." + std::to_string(i) + ".";
        for (const auto& [a, b] : {std::pair{"ln1", "ln_1"}, std::pair{"ln2", "ln_2"}}) {
            l.top[from + a + ".weight"] = one({h}, to + b + ".weight");
            l.top[from + a + ".bias"] = one({h}, to + b + ".bias");
        }
        l.top[from + "ls1.weight"] = one({h}, to + "ls_1.gamma");
        l.top[from + "ls2.weight"] = one({h}, to + "ls_2.gamma");
        l.top[from + "ffn_up.weight"] = one({v.width, h}, to + "mlp.c_fc.weight");
        l.top[from + "ffn_up.bias"] = one({v.width}, to + "mlp.c_fc.bias");
        l.top[from + "ffn_down.weight"] = one({h, v.width}, to + "mlp.c_proj.weight");
        l.top[from + "ffn_down.bias"] = one({h}, to + "mlp.c_proj.bias");
        for (const auto& [a, b] : {std::pair{"attn_q", "q_proj"}, std::pair{"attn_k", "k_proj"},
                                   std::pair{"attn_v", "v_proj"}, std::pair{"attn_out", "out_proj"}}) {
            l.top[from + a + ".weight"] = one({h, h}, to + "self_attn." + b + ".weight");
            l.top[from + a + ".bias"] = one({h}, to + "self_attn." + b + ".bias");
        }
    }
    l.top["mm.1.weight"] = one({v.projection, v.merge * v.merge * h}, kVisionPrefix + "proj_in.weight");
    l.top["mm.2.weight"] = one({v.out, v.projection}, kVisionPrefix + "proj_out.weight");
    return l;
}

class Inkling2 final : public Architecture {
public:
    explicit Inkling2(const Flavor& flavor) : fl_(flavor) {}
    std::string_view name() const override { return fl_.name; }

    // Names do not depend on the sizes: a layout of nominal ones, whose block i is a layer of each kind in turn,
    // names every tensor a layer can hold.
    std::optional<std::string> adapter_target(std::string_view tensor) const override {
        int64_t block = 0;
        if (tensor.rfind("blk.", 0) == 0) {
            const size_t dot = tensor.find('.', 4);
            const std::string digits(tensor.substr(4, dot == std::string_view::npos ? 0 : dot - 4));
            if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) return std::nullopt;
            block = std::stoll(digits);
        }
        for (const bool sliding : {true, false}) {
            for (const bool looped : {true, false}) {
                Dimensions d;
                d.vocab = d.tokens = d.hidden = d.heads = d.kv_heads = d.head_dim = d.rel_width = d.rel_extent = 1;
                d.window = d.conv_taps = d.block_layers = d.experts = d.experts_per_token = d.expert_width = 1;
                d.gate_rank = 1;
                d.passes = 2;
                d.layers = block + 1;
                d.sliding.assign(static_cast<size_t>(d.layers), sliding);
                d.loop_start = looped ? 0 : d.layers;
                d.loop_end = d.layers;
                if (auto name = whole_tensor(layout(fl_, d), tensor)) return name;
            }
        }
        return std::nullopt;
    }

    ImportPlan plan(const gguf::File& f) const override {
        refuse_unknown_keys(f, fl_.name, known_keys(fl_));
        const Dimensions d = dimensions(f, fl_);
        ImportPlan plan;
        plan.config["architectures"] = {"Inkling2LoopForConditionalGeneration"};
        plan.config["model_type"] = "inkling2_loop";
        plan.config["text_config"] = text_config(f, fl_, d);
        map_tensors(f, fl_.name, layout(fl_, d), plan);
        return plan;
    }

    // The drafter is always a file of its own (its blocks are named as the model's are): its tensors go into the
    // pack beside the model's, under the names a pack of the released checkpoint gives them, and its settings into
    // config.json's mtp_config.
    bool drafter_apart(const gguf::File&) const override { return true; }
    bool reads_projector() const override { return true; }

    void add_projector(ImportPlan& p, const gguf::File& f, const gguf::File& projector) const override {
        VisionDimensions v;
        p.config["vision_config"] = vision_config(projector, fl_, dimensions(f, fl_), v);
        map_tensors(projector, kProjector, vision_layout(v), p);
    }

    ImportPlan plan_with_mtp(const gguf::File& f, const gguf::File& mtp) const override {
        ImportPlan p = plan(f);
        DrafterDimensions dd;
        const Dimensions d = dimensions(f, fl_);
        p.config["mtp_config"] = drafter_config(mtp, f, fl_, d, dd);
        // Mainline's file copies in the target's embedding norm, which the drafter reads from the target.
        std::vector<const gguf::TensorInfo*> own;
        for (const gguf::TensorInfo& t : mtp.tensors()) {
            if (dd.mainline && t.name == "token_embd_norm.weight") {
                XS_CHECK(same_shape(t.shape(), {d.hidden}), fl_.name, ": --mtp: token_embd_norm.weight is ",
                         dims_to_string(t.shape()), ", the model's embedding norm [", d.hidden, "]");
                p.copies.push_back(&t);
            } else {
                own.push_back(&t);
            }
        }
        map_tensors(own, kDrafter, drafter_layout(dd, d), p);
        return p;
    }

private:
    const Flavor& fl_;
};

}  // namespace

const Architecture& inkling2() {
    static const Inkling2 a(kUnsloth);
    return a;
}

const Architecture& inkling2_loop() {
    static const Inkling2 a(kMainline);
    return a;
}

}  // namespace xs::gguf_import
