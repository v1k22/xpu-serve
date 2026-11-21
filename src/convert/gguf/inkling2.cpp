// GGUF architecture "inkling2": Inkling 2 Mini as the converter of Unsloth's llama.cpp builds writes it
// (conversion/inkling2.py), imported for models/inkling2/ as a pack in the released checkpoint's layout (names under
// model.llm.).
//
// What the converter changed, and what the import does about it:
// - names: blk.<i>.* for layer i (each layer once: the passes of the loop share its tensors), token_embd, output*,
//   loop_*; mapped back to the checkpoint's names. A layer's attention query is the checkpoint's
//   attn_res.attn_queries.<i - 1>.
// - block_count counts layer visits (a looped layer once per pass) and the per-layer arrays have an entry per visit:
//   read back as the layers and local_layer_ids.
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
//
// Mainline llama.cpp's layout of the same tensors (general.architecture inkling2_loop, the repository's
// Mainline_Support files) has other names and keys and is not read here.

#include <set>

#include "convert/gguf/architecture.h"
#include "convert/gguf/layout.h"
#include "util/error.h"

namespace xs::gguf_import {
namespace {

constexpr std::string_view kName = "inkling2";
const std::string kPrefix = "model.llm.";

const std::set<std::string_view> kKnownKeys = {
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
    "num_loops",
    "loop.core_start",
    "loop.core_end",
    "loop.gate_rank",
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

std::string key(std::string_view rest) {
    return str_cat(kName, ".", rest);
}

struct Dimensions {
    int64_t vocab = 0, tokens = 0, hidden = 0, layers = 0;
    int64_t heads = 0, kv_heads = 0, head_dim = 0;
    int64_t rel_width = 0, rel_extent = 0, window = 0, conv_taps = 0, block_layers = 0;
    int64_t experts = 0, experts_per_token = 0, expert_width = 0;
    int64_t loop_start = 0, loop_end = 0, passes = 0, gate_rank = 0;
    std::vector<bool> sliding;  // per layer

    bool looped(int64_t i) const { return i >= loop_start && i < loop_end; }
};

// The layer a visit runs: the layers before the loop, the looped ones once per pass, the layers after it.
int64_t layer_of_visit(const Dimensions& d, int64_t visit) {
    const int64_t span = d.loop_end - d.loop_start;
    if (visit < d.loop_start) return visit;
    if (visit < d.loop_start + d.passes * span) return d.loop_start + (visit - d.loop_start) % span;
    return visit - (d.passes - 1) * span;
}

// A value every visit must agree on.
int64_t same_per_visit(const gguf::File& f, std::string_view rest, int64_t visits) {
    const std::vector<int64_t> values = f.get(key(rest)).ints();
    if (static_cast<int64_t>(values.size()) != visits) {
        fail(kName, ": ", key(rest), " has ", values.size(), " entries for ", visits, " layer visits");
    }
    for (const int64_t v : values) {
        if (v != values[0] || v <= 0) {
            fail(kName, ": ", key(rest), " holds ", v, " beside ", values[0],
                 ": layers of different head counts are not supported");
        }
    }
    return values[0];
}

Dimensions dimensions(const gguf::File& f) {
    for (const char* loop : {"num_loops", "loop.core_start", "loop.core_end", "loop.gate_rank"}) {
        if (f.find(key(loop)) == nullptr) {
            fail(kName, ": the file has no ", key(loop), ": an Inkling 2 without looped layers is not supported");
        }
    }
    Dimensions d;
    const int64_t visits = positive(f, kName, key("block_count"));
    d.passes = positive(f, kName, key("num_loops"));
    d.loop_start = positive(f, kName, key("loop.core_start"));
    d.loop_end = positive(f, kName, key("loop.core_end"));
    d.gate_rank = positive(f, kName, key("loop.gate_rank"));
    d.layers = visits - (d.passes - 1) * (d.loop_end - d.loop_start);
    if (d.loop_end <= d.loop_start || d.layers < d.loop_end) {
        fail(kName, ": ", visits, " layer visits do not hold ", d.passes, " passes over layers ", d.loop_start, " to ",
             d.loop_end);
    }
    d.hidden = positive(f, kName, key("embedding_length"));
    d.vocab = positive(f, kName, key("vocab_size"));
    d.tokens = positive(f, kName, key("vocab_size_unpadded"));
    if (f.get("tokenizer.ggml.tokens").size() != static_cast<uint64_t>(d.tokens) || d.tokens > d.vocab) {
        fail(kName, ": the tokenizer has ", f.get("tokenizer.ggml.tokens").size(), " tokens, ",
             key("vocab_size_unpadded"), " is ", d.tokens, " and ", key("vocab_size"), " ", d.vocab);
    }
    d.heads = same_per_visit(f, "attention.head_count", visits);
    d.kv_heads = same_per_visit(f, "attention.head_count_kv", visits);
    d.head_dim = positive(f, kName, key("attention.key_length"));
    for (const char* same : {"attention.value_length", "attention.key_length_swa", "attention.value_length_swa"}) {
        if (const int64_t v = positive(f, kName, key(same)); v != d.head_dim) {
            fail(kName, ": ", key(same), " ", v, " differs from attention.key_length ", d.head_dim,
                 ", which is not supported");
        }
    }
    if (const int64_t rope = f.get_int(key("rope.dimension_count")); rope != 0) {
        fail(kName, ": ", key("rope.dimension_count"), " is ", rope, "; the model has no rotary positions");
    }
    d.rel_width = positive(f, kName, key("attention.relative_length"));
    d.rel_extent = positive(f, kName, key("attention.relative_extent"));
    d.window = positive(f, kName, key("attention.sliding_window"));
    d.conv_taps = positive(f, kName, key("shortconv.l_cache"));
    d.block_layers = positive(f, kName, key("attn_res.block_size"));
    d.experts = positive(f, kName, key("expert_count"));
    d.experts_per_token = positive(f, kName, key("expert_used_count"));
    d.expert_width = positive(f, kName, key("expert_feed_forward_length"));
    if (const int64_t ffn = positive(f, kName, key("feed_forward_length")); ffn != d.expert_width) {
        fail(kName, ": ", key("feed_forward_length"), " ", ffn, " differs from the experts' ", d.expert_width);
    }

    const std::vector<bool> pattern = f.get(key("attention.sliding_window_pattern")).bools();
    if (static_cast<int64_t>(pattern.size()) != visits) {
        fail(kName, ": attention.sliding_window_pattern has ", pattern.size(), " entries for ", visits,
             " layer visits");
    }
    d.sliding.assign(static_cast<size_t>(d.layers), false);
    std::vector<bool> seen(static_cast<size_t>(d.layers), false);
    for (int64_t v = 0; v < visits; ++v) {
        const auto layer = static_cast<size_t>(layer_of_visit(d, v));
        if (seen[layer] && d.sliding[layer] != pattern[static_cast<size_t>(v)]) {
            fail(kName, ": attention.sliding_window_pattern gives layer ", layer, " two kinds on two passes");
        }
        seen[layer] = true;
        d.sliding[layer] = pattern[static_cast<size_t>(v)];
    }
    return d;
}

nlohmann::ordered_json text_config(const gguf::File& f, const Dimensions& d) {
    const auto number = [&](std::string_view rest) { return config_number(f.get(key(rest))); };
    nlohmann::ordered_json t;
    t["model_type"] = "inkling2_loop_text";
    t["model_max_length"] = positive(f, kName, key("context_length"));
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
    t["log_scaling_n_floor"] = positive(f, kName, key("attention.log_scale_floor"));
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

Entries layer(const Dimensions& d, int64_t i) {
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
        for (const char* which : {"attn", "k", "v", "mlp"}) {
            const std::string file = str_cat("loop_", which, "_");
            const std::string name = str_cat(L, "controllers.", which, ".");
            e[file + "down.weight"] = one({d.gate_rank, 2 * h}, name + "down_weight");
            e[file + "forget.weight"] = one({h, d.gate_rank}, name + "forget_weight");
            e[file + "gain.weight"] = one({h, d.gate_rank}, name + "gain_weight");
        }
    }
    return e;
}

Layout layout(const Dimensions& d) {
    Layout l;
    const int64_t h = d.hidden, message_queries = 2 * (d.loop_end - d.loop_start);
    l.top["token_embd.weight"] = one({d.vocab, h}, kPrefix + "embed.weight");
    l.top["token_embd_norm.weight"] = one({h}, kPrefix + "embed_norm.weight");
    l.top["output.weight"] = one({d.vocab, h}, kPrefix + "unembed.weight");
    l.top["output_norm.weight"] = one({h}, kPrefix + "norm.weight");
    l.top["output_res_query.weight"] = one({h}, kPrefix + "attn_res.output_query");
    l.top["loop_res_query.weight"] = one({message_queries, h}, kPrefix + "middle_cycle_cross_pass_queries");
    l.top["loop_kv_res_query.weight"] = one({message_queries, h}, kPrefix + "middle_cycle_cross_pass_kv_queries");
    for (int64_t i = 0; i < d.layers; ++i)
        l.blocks.push_back(layer(d, i));
    l.block_kind = [d](int64_t i) {
        return str_cat(d.sliding[static_cast<size_t>(i)] ? "sliding" : "full", " attention",
                       d.looped(i) ? ", looped" : "");
    };
    return l;
}

class Inkling2 final : public Architecture {
public:
    std::string_view name() const override { return kName; }

    ImportPlan plan(const gguf::File& f) const override {
        refuse_unknown_keys(f, kName, kKnownKeys);
        const Dimensions d = dimensions(f);
        ImportPlan plan;
        plan.config["architectures"] = {"Inkling2LoopForConditionalGeneration"};
        plan.config["model_type"] = "inkling2_loop";
        plan.config["text_config"] = text_config(f, d);
        map_tensors(f, kName, layout(d), plan);
        return plan;
    }
};

}  // namespace

const Architecture& inkling2() {
    static const Inkling2 a;
    return a;
}

}  // namespace xs::gguf_import
