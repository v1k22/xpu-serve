// GGUF architectures "gemma4" and "gemma4-assistant": Gemma 4's text decoder and its assistant (the drafter of
// speculative decoding) as llama.cpp's converter writes them (conversion/gemma.py, Gemma4Model and
// Gemma4AssistantModel), imported for models/gemma4/: the decoder as a text-only checkpoint (Gemma4ForCausalLM, names
// under model.), the assistant as an assistant checkpoint (Gemma4AssistantForCausalLM).
//
// What the converter changed, and what the import does about it:
// - names: blk.<i>.* for layer i, token_embd, per_layer_*, nextn.*_projection and output_norm; mapped back to the
//   Hugging Face names.
// - no value is changed: Gemma 4's norms multiply by w (not 1 + w, as Gemma 3's), and they, the scales and every
//   matrix are stored as the checkpoint holds them, rows and columns in its order.
// - it can store q_proj, k_proj and v_proj as one tensor (attn_qkv): cut back into their runs of rows.
// - rope_freqs.weight is not the checkpoint's: llama.cpp's proportional RoPE for the full-attention layers, a factor
//   of 1 for each rotated pair and 1e30 (no rotation) for the rest. It is checked and read as config.json's
//   partial_rotary_factor, and not imported.
// - the vision and audio towers are not in the file: llama.cpp writes them to an mmproj file of their own
//   (projector type gemma4v), which `convert --mmproj` reads: the vision tower under the checkpoint's names
//   (model.vision_tower.*, model.embed_vision.*), its patch embedding put back from a convolution's layout
//   ([out, channel, y, x] to the checkpoint's [out, y, x, channel]); config.json as the release lays it out
//   (text_config, vision_config, the image tokens) and the release's processor_config.json, with what llama.cpp
//   fixes for the tower (RoPE theta 100, pooling 3 x 3, bicubic resizing, 280 soft tokens an image). An audio tower
//   in the same file (E2B, E4B) is left out: the model serves no audio.
// - Gemma 4 12B's projector file (projector types gemma4uv and gemma4ua, Gemma4UnifiedVisionAudioModel): the
//   embedders without a tower, for images and for audio, under model.vision_embedder.*, model.embed_vision.* and
//   model.embed_audio.*. The converter laid the dense layer's columns and the first LayerNorm out channel-major
//   (an im2col's order) and swapped the position table's first two axes: both put back.
// - 26B-A4B's experts: the router (ffn_gate_inp, its per-dimension scale ffn_gate_inp.scale), the banks (gate and up
//   fused as ffn_gate_up_exps, gate rows first, or apart as ffn_gate_exps and ffn_up_exps; ffn_down_exps) with the
//   router's per-expert scale stored as ffn_down_exps.scale, and the experts' own norms (pre_ffw_norm_2,
//   post_ffw_norm_1, post_ffw_norm_2): mapped back by name, values unchanged.
// - attention_k_eq_v (26B-A4B, 31B): the full-attention layers have no attn_v, their values being their keys.
// - an assistant's centroid head (masked_embedding.*) is left out: llama.cpp's graph scores the whole vocabulary with
//   the tied embedding, which is the assistant without ordered embeddings (use_ordered_embeddings false).

#include <algorithm>
#include <cmath>
#include <set>

#include "convert/gguf/architecture.h"
#include "convert/gguf/layout.h"
#include "util/error.h"

namespace xs::gguf_import {
namespace {

const std::string kPrefix = "model.";  // the decoder of a text-only checkpoint, and of an assistant

// The <architecture>.* keys the import reads; an assistant's file also has the two after them.
const std::set<std::string_view> kKnownKeys = {
    "block_count",
    "context_length",
    "embedding_length",
    "embedding_length_per_layer_input",
    "feed_forward_length",
    "final_logit_softcapping",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.key_length_swa",
    "attention.value_length_swa",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "attention.shared_kv_layers",
    "rope.freq_base",
    "rope.freq_base_swa",
    "rope.dimension_count",
    "rope.dimension_count_swa",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
};
const std::set<std::string_view> kAssistantKeys = {
    "embedding_length_out",  // the target's hidden size
    "nextn_predict_layers",  // every layer drafts
};

struct Dimensions {
    int64_t vocab = 0, hidden = 0, layers = 0, heads = 0;
    int64_t ffn = 0;                            // the MLP's width; a KV-shared layer's is twice it with double_wide
    bool double_wide = false;                   // use_double_wide_mlp
    int64_t head_dim = 0, global_head_dim = 0;  // sliding and full-attention layers
    int64_t kv_heads = 0, global_kv_heads = 0;
    int64_t ple = 0;  // hidden_size_per_layer_input; 0: no per-layer embeddings
    int64_t ple_vocab = 0;
    int64_t shared = 0;         // num_kv_shared_layers: the last ones read an earlier layer's KV (all: an assistant)
    int64_t rotated_pairs = 0;  // of a full-attention head (rope_freqs); global_head_dim / 2 without the tensor
    int64_t backbone = 0;       // an assistant's target's hidden size
    int64_t experts = 0, experts_used = 0, expert_ffn = 0;  // a mixture of experts beside every layer's MLP
    bool k_eq_v = false;        // full-attention layers read their keys as values (no attn_v)
    bool fused_gate_up = true;  // the expert banks' gate and up fused (ffn_gate_up_exps), or apart
    std::vector<bool> sliding;  // per layer
    std::vector<int64_t> layer_ffn;

    int64_t first_shared() const { return layers - shared; }
};

// The layer whose KV shared layer `i` reads, as Gemma 4 has it: the last earlier layer of its kind that computes its
// own; -1 when none does.
int64_t kv_source(const Dimensions& d, int64_t i) {
    for (int64_t j = d.first_shared() - 1; j >= 0; --j) {
        if (d.sliding[static_cast<size_t>(j)] == d.sliding[static_cast<size_t>(i)]) return j;
    }
    return -1;
}

Entries layer(const Dimensions& d, int64_t i) {
    const bool sliding = d.sliding[static_cast<size_t>(i)];
    const bool shared = i >= d.first_shared();
    const int64_t hd = sliding ? d.head_dim : d.global_head_dim;
    const int64_t kv = sliding ? d.kv_heads : d.global_kv_heads;
    const int64_t ffn = d.layer_ffn[static_cast<size_t>(i)];
    const int64_t h = d.hidden;
    const std::string L = kPrefix + "layers." + std::to_string(i) + ".";
    const std::string A = L + "self_attn.";
    Entries e;
    e["attn_norm.weight"] = one({h}, L + "input_layernorm.weight");
    const Piece q{A + "q_proj.weight", {d.heads * hd, h}};
    const Piece k{A + "k_proj.weight", {kv * hd, h}};
    const Piece v{A + "v_proj.weight", {kv * hd, h}};
    e["attn_q.weight"] = Entry{q.shape, {q}};
    e["attn_k.weight"] = Entry{k.shape, {k}, shared};  // a KV-shared layer's own k and v are never read
    if (!(d.k_eq_v && !sliding)) {
        e["attn_v.weight"] = Entry{v.shape, {v}, shared};
        e["attn_qkv.weight"] = Entry{{(d.heads + 2 * kv) * hd, h}, {q, k, v}, true};
    }
    e["attn_output.weight"] = one({h, d.heads * hd}, A + "o_proj.weight");
    e["attn_q_norm.weight"] = one({hd}, A + "q_norm.weight");
    e["attn_k_norm.weight"] = one({hd}, A + "k_norm.weight");
    e["attn_k_norm.weight"].optional = shared;
    e["post_attention_norm.weight"] = one({h}, L + "post_attention_layernorm.weight");
    e["ffn_norm.weight"] = one({h}, L + "pre_feedforward_layernorm.weight");
    e["ffn_gate.weight"] = one({ffn, h}, L + "mlp.gate_proj.weight");
    e["ffn_up.weight"] = one({ffn, h}, L + "mlp.up_proj.weight");
    e["ffn_down.weight"] = one({h, ffn}, L + "mlp.down_proj.weight");
    e["post_ffw_norm.weight"] = one({h}, L + "post_feedforward_layernorm.weight");
    if (d.ple > 0) {
        e["inp_gate.weight"] = one({d.ple, h}, L + "per_layer_input_gate.weight");
        e["proj.weight"] = one({h, d.ple}, L + "per_layer_projection.weight");
        e["post_norm.weight"] = one({h}, L + "post_per_layer_input_norm.weight");
    }
    e["layer_output_scale.weight"] = one({1}, L + "layer_scalar");
    if (d.experts > 0) {
        const int64_t n = d.experts, f = d.expert_ffn;
        e["ffn_gate_inp.weight"] = one({n, h}, L + "router.proj.weight");
        e["ffn_gate_inp.scale"] = one({h}, L + "router.scale");
        e["ffn_down_exps.scale"] = one({n}, L + "router.per_expert_scale");
        if (d.fused_gate_up) {
            e["ffn_gate_up_exps.weight"] = one({n, 2 * f, h}, L + "experts.gate_up_proj");
        } else {
            e["ffn_gate_exps.weight"] = one({n, f, h}, L + "experts.gate_proj");
            e["ffn_up_exps.weight"] = one({n, f, h}, L + "experts.up_proj");
        }
        e["ffn_down_exps.weight"] = one({n, h, f}, L + "experts.down_proj");
        e["pre_ffw_norm_2.weight"] = one({h}, L + "pre_feedforward_layernorm_2.weight");
        e["post_ffw_norm_1.weight"] = one({h}, L + "post_feedforward_layernorm_1.weight");
        e["post_ffw_norm_2.weight"] = one({h}, L + "post_feedforward_layernorm_2.weight");
    }
    return e;
}

// ---- The vision tower's file (llama.cpp's mmproj, projector type gemma4v)

constexpr std::string_view kClip = "clip";
const std::set<std::string_view> kClipKeys = {
    "has_vision_encoder", "has_audio_encoder", "use_gelu", "projector_type", "vision.projector_type",
    "vision.projection_dim", "vision.image_size", "vision.patch_size", "vision.embedding_length",
    "vision.feed_forward_length", "vision.block_count", "vision.attention.head_count", "vision.image_mean",
    "vision.image_std", "vision.attention.layer_norm_epsilon", "vision.projector.scale_factor",
    // The audio tower's keys (E2B, E4B): read by no one here, its tensors left out.
    "audio.projector_type", "audio.projection_dim", "audio.embedding_length", "audio.feed_forward_length",
    "audio.block_count", "audio.attention.head_count", "audio.num_mel_bins", "audio.attention.layer_norm_epsilon",
};

struct VisionDimensions {
    int64_t hidden = 0, width = 0, layers = 0, heads = 0, head_dim = 0, patch = 0, pooling = 3, positions = 0;
    int64_t out = 0;
    double eps = 1e-6;  // as config_number gives it
    bool clipped = false;      // Gemma4ClippableLinear bounds beside every linear (E2B, E4B)
    bool standardize = false;  // std_bias and std_scale (26B-A4B, 31B)
};

// The projector file against the model's hidden size: its dimensions, refusing by name a file that is not a Gemma 4
// vision tower (or another model's), and settings the tower here does not read as the file means them.
VisionDimensions vision_dimensions(const gguf::File& f, std::string_view arch, int64_t hidden) {
    const std::string_view general = f.get_string("general.architecture");
    const gguf::Value* type = f.find("clip.vision.projector_type");
    if (type == nullptr) type = f.find("clip.projector_type");
    if (general != kClip || type == nullptr || type->as_string() != "gemma4v") {
        fail(arch, ": --mmproj: the file is not a Gemma 4 vision tower (general.architecture '", general,
             "', projector type '", type != nullptr ? type->as_string() : std::string_view("none"),
             "'; expected 'clip' and 'gemma4v')");
    }
    refuse_unknown_keys(f, kClip, kClipKeys);
    const auto positive_of = [&](std::string_view rest) { return positive(f, kClip, str_cat("clip.", rest)); };
    VisionDimensions v;
    v.hidden = positive_of("vision.embedding_length");
    v.width = positive_of("vision.feed_forward_length");
    v.layers = positive_of("vision.block_count");
    v.heads = positive_of("vision.attention.head_count");
    v.patch = positive_of("vision.patch_size");
    if (f.find("clip.vision.projector.scale_factor") != nullptr) {
        v.pooling = positive_of("vision.projector.scale_factor");
    }
    v.out = positive_of("vision.projection_dim");
    XS_CHECK(v.out == hidden, arch, ": --mmproj: the tower projects to ", v.out, ", the model's hidden size is ",
             hidden, ": another model's file");
    for (const auto& [key, want] :
         {std::pair{"clip.vision.image_mean", 0.0}, std::pair{"clip.vision.image_std", 1.0}}) {
        if (const gguf::Value* m = f.find(key)) {
            for (const double x : m->floats()) {
                XS_CHECK(x == want, arch, ": --mmproj: ", key, " is not ", want, " a channel (Gemma 4's processor ",
                         "does not normalize)");
            }
        }
    }
    v.eps = config_number(f.get("clip.vision.attention.layer_norm_epsilon"));
    const gguf::TensorInfo* norm = f.find_tensor("v.blk.0.attn_q_norm.weight");
    XS_CHECK(norm != nullptr && norm->dims.size() == 1, arch, ": --mmproj: the file lacks v.blk.0.attn_q_norm.weight");
    v.head_dim = norm->numel();
    XS_CHECK(v.heads * v.head_dim == v.hidden, arch, ": --mmproj: ", v.heads, " heads of ", v.head_dim,
             " for a width of ", v.hidden);
    const gguf::TensorInfo* table = f.find_tensor("v.position_embd.weight");
    XS_CHECK(table != nullptr && table->shape().size() == 3, arch,
             ": --mmproj: the file lacks the position table v.position_embd.weight [2, positions, width]");
    v.positions = table->shape()[1];
    v.clipped = f.find_tensor("v.blk.0.attn_q.input_max") != nullptr;
    v.standardize = f.find_tensor("v.std_bias") != nullptr;
    return v;
}

// vision_config as the release's config.json has it.
nlohmann::ordered_json vision_config(const VisionDimensions& v) {
    nlohmann::ordered_json c;
    c["model_type"] = "gemma4_vision";
    c["hidden_size"] = v.hidden;
    c["intermediate_size"] = v.width;
    c["num_hidden_layers"] = v.layers;
    c["num_attention_heads"] = v.heads;
    c["num_key_value_heads"] = v.heads;
    c["head_dim"] = v.head_dim;
    c["global_head_dim"] = v.head_dim;
    c["patch_size"] = v.patch;
    c["pooling_kernel_size"] = v.pooling;
    c["position_embedding_size"] = v.positions;
    c["default_output_length"] = 280;
    c["rms_norm_eps"] = v.eps;
    c["use_clipped_linears"] = v.clipped;
    c["standardize"] = v.standardize;
    c["attention_bias"] = false;
    c["hidden_activation"] = "gelu_pytorch_tanh";
    c["rope_parameters"] = {{"rope_theta", 100.0}, {"rope_type", "default"}};
    return c;
}

// The release's image processor (processor_config.json), which the file does not hold.
nlohmann::ordered_json processor_config(const VisionDimensions& v) {
    nlohmann::ordered_json p;
    p["do_convert_rgb"] = true;
    p["do_normalize"] = false;
    p["do_rescale"] = true;
    p["do_resize"] = true;
    p["image_mean"] = {0.0, 0.0, 0.0};
    p["image_std"] = {1.0, 1.0, 1.0};
    p["image_processor_type"] = "Gemma4ImageProcessor";
    p["image_seq_length"] = 280;
    p["max_soft_tokens"] = 280;
    p["patch_size"] = v.patch;
    p["pooling_kernel_size"] = v.pooling;
    p["resample"] = 3;  // bicubic
    p["rescale_factor"] = 1.0 / 255.0;
    return {{"image_processor", std::move(p)}, {"processor_class", "Gemma4Processor"}};
}

// The converter wrote the patch embedding [out, y * x * channel] as a convolution's [out, channel, y, x].
std::vector<float> patch_back(const std::vector<float>& stored, const Dims& shape) {
    const int64_t out = shape[0], ch = shape[1], ys = shape[2], xs = shape[3];
    std::vector<float> back(stored.size());
    for (int64_t o = 0; o < out; ++o)
        for (int64_t c = 0; c < ch; ++c)
            for (int64_t y = 0; y < ys; ++y)
                for (int64_t x = 0; x < xs; ++x)
                    back[static_cast<size_t>(((o * ys + y) * xs + x) * ch + c)] =
                        stored[static_cast<size_t>(((o * ch + c) * ys + y) * xs + x)];
    return back;
}
const FloatReorder kPatchBack{"a convolution's [out, channel, y, x] to [out, y, x, channel]", patch_back};

Layout vision_layout(const VisionDimensions& v) {
    const std::string P = "model.vision_tower.";
    const int64_t h = v.hidden;
    Layout l;
    l.top["v.patch_embd.weight"] =
        reordered({h, 3, v.patch, v.patch}, P + "patch_embedder.input_proj.weight", {h, v.patch * v.patch * 3},
                  &kPatchBack);
    l.top["v.position_embd.weight"] = one({2, v.positions, h}, P + "patch_embedder.position_embedding_table");
    l.top["mm.input_projection.weight"] = one({v.out, h}, "model.embed_vision.embedding_projection.weight");
    if (v.standardize) {
        l.top["v.std_bias"] = one({h}, P + "std_bias");
        l.top["v.std_scale"] = one({h}, P + "std_scale");
    }
    for (int64_t i = 0; i < v.layers; ++i) {
        const std::string from = "v.blk." + std::to_string(i) + ".";
        const std::string to = P + "encoder.layers." + std::to_string(i) + ".";
        for (const auto& [a, b] :
             {std::pair{"ln1", "input_layernorm"}, std::pair{"attn_post_norm", "post_attention_layernorm"},
              std::pair{"ln2", "pre_feedforward_layernorm"}, std::pair{"ffn_post_norm", "post_feedforward_layernorm"}}) {
            l.top[from + a + ".weight"] = one({h}, to + b + ".weight");
        }
        l.top[from + "attn_q_norm.weight"] = one({v.head_dim}, to + "self_attn.q_norm.weight");
        l.top[from + "attn_k_norm.weight"] = one({v.head_dim}, to + "self_attn.k_norm.weight");
        const auto linear = [&](const std::string& gguf, const std::string& hf, Dims shape) {
            l.top[from + gguf + ".weight"] = one(shape, to + hf + ".linear.weight");
            if (!v.clipped) return;
            for (const char* bound : {"input_min", "input_max", "output_min", "output_max"}) {
                l.top[from + gguf + "." + bound] = one({1}, to + hf + "." + bound);
            }
        };
        linear("attn_q", "self_attn.q_proj", {h, h});
        linear("attn_k", "self_attn.k_proj", {h, h});
        linear("attn_v", "self_attn.v_proj", {h, h});
        linear("attn_out", "self_attn.o_proj", {h, h});
        linear("ffn_gate", "mlp.gate_proj", {v.width, h});
        linear("ffn_up", "mlp.up_proj", {v.width, h});
        linear("ffn_down", "mlp.down_proj", {h, v.width});
    }
    return l;
}

// ---- Gemma 4 12B's projector file (gemma4uv and gemma4ua)

struct UnifiedDimensions {
    int64_t embed = 0, patch = 0, pooling = 3, positions = 0, out = 0, samples = 0, audio_out = 0;
    double eps = 1e-6, audio_eps = 1e-6;  // as config_number gives them
};

UnifiedDimensions unified_dimensions(const gguf::File& f, std::string_view arch, int64_t hidden) {
    const auto positive_of = [&](std::string_view rest) { return positive(f, kClip, str_cat("clip.", rest)); };
    const gguf::Value* audio = f.find("clip.audio.projector_type");
    if (audio == nullptr || audio->as_string() != "gemma4ua") {
        fail(arch, ": --mmproj: a gemma4uv vision embedder without its gemma4ua audio embedder");
    }
    refuse_unknown_keys(f, kClip, kClipKeys);
    UnifiedDimensions u;
    u.embed = positive_of("vision.embedding_length");
    u.patch = positive_of("vision.patch_size");
    if (f.find("clip.vision.projector.scale_factor") != nullptr) {
        u.pooling = positive_of("vision.projector.scale_factor");
    }
    u.out = positive_of("vision.projection_dim");
    u.samples = positive_of("audio.embedding_length");
    u.audio_out = positive_of("audio.projection_dim");
    XS_CHECK(u.out == hidden && u.audio_out == hidden, arch, ": --mmproj: the embedders project to ", u.out, " and ",
             u.audio_out, ", the model's hidden size is ", hidden, ": another model's file");
    u.eps = config_number(f.get("clip.vision.attention.layer_norm_epsilon"));
    u.audio_eps = config_number(f.get("clip.audio.attention.layer_norm_epsilon"));
    const gguf::TensorInfo* table = f.find_tensor("v.position_embd.weight");
    XS_CHECK(table != nullptr && table->shape().size() == 3 && table->shape()[0] == 2, arch,
             ": --mmproj: the file lacks the position table v.position_embd.weight [2, positions, width]");
    u.positions = table->shape()[1];
    return u;
}

// The converter's order of a merged patch's values (channel, row, column) back to the checkpoint's (row, column,
// channel): value i of the file is value perm(i) of the checkpoint.
int64_t merged_patch_index(int64_t i, int64_t side) {
    const int64_t ch = i / (side * side), row = (i % (side * side)) / side, col = i % side;
    return (row * side + col) * 3 + ch;
}
std::vector<float> patch_columns_back(const std::vector<float>& stored, const Dims& shape) {
    const int64_t rows = shape[0], cols = shape[1];
    const auto side = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(cols / 3))));
    std::vector<float> back(stored.size());
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t i = 0; i < cols; ++i)
            back[static_cast<size_t>(r * cols + merged_patch_index(i, side))] =
                stored[static_cast<size_t>(r * cols + i)];
    return back;
}
std::vector<float> patch_vector_back(const std::vector<float>& stored, const Dims& shape) {
    return patch_columns_back(stored, {1, shape.back()});
}
std::vector<float> position_axes_back(const std::vector<float>& stored, const Dims& shape) {  // [2, P, D] -> [P, 2, D]
    const int64_t axes = shape[0], positions = shape[1], width = shape[2];
    std::vector<float> back(stored.size());
    for (int64_t a = 0; a < axes; ++a)
        for (int64_t p = 0; p < positions; ++p)
            std::copy_n(stored.begin() + static_cast<std::ptrdiff_t>((a * positions + p) * width), width,
                        back.begin() + static_cast<std::ptrdiff_t>((p * axes + a) * width));
    return back;
}
const FloatReorder kPatchColumnsBack{"the dense layer's columns from channel-major", patch_columns_back};
const FloatReorder kPatchVectorBack{"the LayerNorm's values from channel-major", patch_vector_back};
const FloatReorder kPositionAxesBack{"the position table's axes swapped back", position_axes_back};

nlohmann::ordered_json unified_vision_config(const UnifiedDimensions& u) {
    nlohmann::ordered_json c;
    c["model_type"] = "gemma4_unified_vision";
    c["patch_size"] = u.patch;
    c["pooling_kernel_size"] = u.pooling;
    c["mm_embed_dim"] = u.embed;
    c["mm_posemb_size"] = u.positions;
    c["output_proj_dims"] = u.embed;
    c["rms_norm_eps"] = u.eps;
    return c;
}

nlohmann::ordered_json unified_audio_config(const UnifiedDimensions& u) {
    nlohmann::ordered_json c;
    c["model_type"] = "gemma4_unified_audio";
    c["audio_embed_dim"] = u.samples;
    c["output_proj_dims"] = u.samples;
    c["rms_norm_eps"] = u.audio_eps;
    return c;
}

Layout unified_layout(const UnifiedDimensions& u) {
    const std::string P = "model.vision_embedder.";
    const int64_t side = u.patch * u.pooling, merged = side * side * 3, d = u.embed;
    Layout l;
    l.top["v.patch_embd.weight"] = reordered({d, merged}, P + "patch_dense.weight", {d, merged}, &kPatchColumnsBack);
    l.top["v.patch_embd.bias"] = one({d}, P + "patch_dense.bias");
    l.top["v.patch_norm.1.weight"] = reordered({merged}, P + "patch_ln1.weight", {merged}, &kPatchVectorBack);
    l.top["v.patch_norm.1.bias"] = reordered({merged}, P + "patch_ln1.bias", {merged}, &kPatchVectorBack);
    for (const auto& [from, to] : {std::pair{"2", "patch_ln2"}, std::pair{"3", "pos_norm"}}) {
        l.top[str_cat("v.patch_norm.", from, ".weight")] = one({d}, P + to + ".weight");
        l.top[str_cat("v.patch_norm.", from, ".bias")] = one({d}, P + to + ".bias");
    }
    l.top["v.position_embd.weight"] =
        reordered({2, u.positions, d}, P + "pos_embedding", {u.positions, 2, d}, &kPositionAxesBack);
    l.top["mm.input_projection.weight"] = one({u.out, d}, "model.embed_vision.embedding_projection.weight");
    l.top["mm.a.input_projection.weight"] =
        one({u.audio_out, u.samples}, "model.embed_audio.embedding_projection.weight");
    return l;
}

class Gemma4 final : public Architecture {
public:
    explicit Gemma4(bool assistant) : assistant_(assistant) {}

    std::string_view name() const override { return assistant_ ? "gemma4-assistant" : "gemma4"; }

    ImportPlan plan(const gguf::File& f) const override {
        std::set<std::string_view> known = kKnownKeys;
        if (assistant_) known.insert(kAssistantKeys.begin(), kAssistantKeys.end());
        refuse_unknown_keys(f, name(), known);
        const Dimensions d = dimensions(f);
        ImportPlan plan;
        if (assistant_) {
            plan.config["architectures"] = {"Gemma4AssistantForCausalLM"};
            plan.config["model_type"] = "gemma4_assistant";
            plan.config["dtype"] = "bfloat16";
            plan.config["backbone_hidden_size"] = d.backbone;
            plan.config["use_ordered_embeddings"] = false;
            plan.config["tie_word_embeddings"] = f.find_tensor("output.weight") == nullptr;
            plan.config["text_config"] = text_config(f, d);
            plan.kind = ImportPlan::Kind::Drafter;
        } else {
            plan.config["architectures"] = {"Gemma4ForCausalLM"};
            plan.config.update(text_config(f, d));
        }
        map_tensors(f, name(), layout(d), plan);
        return plan;
    }

    bool reads_projector() const override { return !assistant_; }

    // The tower's tensors under the checkpoint's names, and config.json as the release lays it out: the text config,
    // vision_config and the image tokens around it; the release's processor_config.json.
    void add_projector(ImportPlan& p, const gguf::File& f, const gguf::File& projector) const override {
        if (assistant_) Architecture::add_projector(p, f, projector);
        const gguf::Value* type = projector.find("clip.vision.projector_type");
        if (type != nullptr && type->as_string() == "gemma4uv") return add_unified(p, f, projector);
        const VisionDimensions v = vision_dimensions(projector, name(), positive(f, name(), key("embedding_length")));
        nlohmann::ordered_json config;
        config["architectures"] = {"Gemma4ForConditionalGeneration"};
        config["model_type"] = "gemma4";
        config["image_token_id"] = token_id(f, name(), "<|image|>");
        config["boi_token_id"] = token_id(f, name(), "<|image>");
        config["eoi_token_id"] = token_id(f, name(), "<image|>");
        config["vision_soft_tokens_per_image"] = 280;
        config["tie_word_embeddings"] = p.config.value("tie_word_embeddings", true);
        config["text_config"] = std::move(p.config);
        config["vision_config"] = vision_config(v);
        p.config = std::move(config);
        p.files["processor_config.json"] = processor_config(v);
        // The vision tower's tensors; an audio tower's (a.*, mm.a.*) are left out.
        std::vector<const gguf::TensorInfo*> vision;
        for (const gguf::TensorInfo& t : projector.tensors()) {
            (t.name.starts_with("a.") || t.name.starts_with("mm.a.") ? p.left_out : vision).push_back(&t);
        }
        if (!p.left_out.empty()) p.left_out_why = "the audio tower: the model serves no audio";
        map_tensors(vision, kClip, vision_layout(v), p);
    }

private:
    // Gemma 4 12B's: its embedders for images and audio, config.json as the release lays it out
    // (Gemma4UnifiedForConditionalGeneration; text_config of gemma4_unified_text, whose images see themselves whole).
    void add_unified(ImportPlan& p, const gguf::File& f, const gguf::File& projector) const {
        const UnifiedDimensions u = unified_dimensions(projector, name(), positive(f, name(), key("embedding_length")));
        nlohmann::ordered_json text = std::move(p.config);
        text["model_type"] = "gemma4_unified_text";
        nlohmann::ordered_json config;
        config["architectures"] = {"Gemma4UnifiedForConditionalGeneration"};
        config["model_type"] = "gemma4_unified";
        config["image_token_id"] = token_id(f, name(), "<|image|>");
        config["boi_token_id"] = token_id(f, name(), "<|image>");
        config["eoi_token_id"] = token_id(f, name(), "<image|>");
        config["audio_token_id"] = token_id(f, name(), "<|audio|>");
        config["boa_token_id"] = token_id(f, name(), "<|audio>");
        config["eoa_token_index"] = token_id(f, name(), "<audio|>");
        config["tie_word_embeddings"] = text.value("tie_word_embeddings", true);
        config["text_config"] = std::move(text);
        config["vision_config"] = unified_vision_config(u);
        config["audio_config"] = unified_audio_config(u);
        p.config = std::move(config);
        VisionDimensions v;  // the processor's: Gemma 4's image processor, its sides a multiple of the merged patch
        v.patch = u.patch;
        v.pooling = u.pooling;
        p.files["processor_config.json"] = processor_config(v);
        map_tensors(projector, kClip, unified_layout(u), p);
    }

    std::string key(std::string_view rest) const { return str_cat(name(), ".", rest); }

    // A positive per-layer value, written as one number or as an array of one per layer.
    std::vector<int64_t> positive_per_layer(const gguf::File& f, std::string_view rest, int64_t layers) const {
        const gguf::Value& v = f.get(key(rest));
        std::vector<int64_t> out = v.type() == gguf::ValueType::Array
                                       ? v.ints()
                                       : std::vector<int64_t>(static_cast<size_t>(layers), v.as_int());
        if (static_cast<int64_t>(out.size()) != layers) {
            fail(name(), ": ", key(rest), " has ", out.size(), " entries for ", layers, " layers");
        }
        for (const int64_t x : out) {
            if (x <= 0) fail(name(), ": ", key(rest), " holds ", x);
        }
        return out;
    }

    Dimensions dimensions(const gguf::File& f) const {
        Dimensions d;
        const gguf::TensorInfo* embed = f.find_tensor("token_embd.weight");
        if (embed == nullptr) fail(name(), ": the file has no token_embd.weight");
        d.vocab = embed->rows();
        if (f.get("tokenizer.ggml.tokens").size() > static_cast<uint64_t>(d.vocab)) {
            fail(name(), ": the tokenizer has ", f.get("tokenizer.ggml.tokens").size(), " tokens for ", d.vocab,
                 " embedding rows");
        }
        d.hidden = positive(f, name(), key("embedding_length"));
        d.layers = positive(f, name(), key("block_count"));
        const std::vector<bool> pattern = f.get(key("attention.sliding_window_pattern")).bools();
        if (static_cast<int64_t>(pattern.size()) != d.layers) {
            fail(name(), ": attention.sliding_window_pattern has ", pattern.size(), " entries for ", d.layers,
                 " layers");
        }
        d.sliding = pattern;

        const std::vector<int64_t> heads = positive_per_layer(f, "attention.head_count", d.layers);
        if (std::count(heads.begin(), heads.end(), heads[0]) != d.layers) {
            fail(name(), ": attention.head_count differs between layers, which Gemma 4 does not");
        }
        d.heads = heads[0];
        d.head_dim = positive(f, name(), key("attention.key_length_swa"));
        d.global_head_dim = positive(f, name(), key("attention.key_length"));
        for (const auto& [k, v] : {std::pair{"attention.key_length", "attention.value_length"},
                                   std::pair{"attention.key_length_swa", "attention.value_length_swa"}}) {
            if (f.find(key(v)) != nullptr && f.get_int(key(v)) != f.get_int(key(k))) {
                fail(name(), ": ", key(v), " ", f.get_int(key(v)), " differs from ", key(k), " ", f.get_int(key(k)));
            }
        }
        // Each kind of layer has one count of KV heads: the file gives one, or one per layer.
        const std::vector<int64_t> kv = positive_per_layer(f, "attention.head_count_kv", d.layers);
        for (const bool sliding : {true, false}) {
            int64_t n = 0;
            for (int64_t i = 0; i < d.layers; ++i) {
                if (d.sliding[static_cast<size_t>(i)] != sliding) continue;
                if (n != 0 && kv[static_cast<size_t>(i)] != n) {
                    fail(name(), ": attention.head_count_kv differs between ", sliding ? "sliding" : "full",
                         "-attention layers, which Gemma 4 does not");
                }
                n = kv[static_cast<size_t>(i)];
            }
            (sliding ? d.kv_heads : d.global_kv_heads) = n;
        }
        if (d.kv_heads == 0) d.kv_heads = d.global_kv_heads;
        if (d.global_kv_heads == 0) d.global_kv_heads = d.kv_heads;

        d.shared = f.find(key("attention.shared_kv_layers")) ? f.get_int(key("attention.shared_kv_layers")) : 0;
        if (assistant_) {
            // Every layer drafts, over the target's KV.
            for (const char* k : {"attention.shared_kv_layers", "nextn_predict_layers"}) {
                if (f.get_int(key(k)) != d.layers) {
                    fail(name(), ": ", key(k), " is ", f.get_int(key(k)), "; an assistant's ", d.layers,
                         " layers all draft over the target's KV");
                }
            }
            d.backbone = positive(f, name(), key("embedding_length_out"));
        } else {
            target_kv_sharing(d);
        }

        // The MLP's width: one for the layers that compute their own KV, and the same or twice it for the shared
        // ones (use_double_wide_mlp).
        d.layer_ffn = positive_per_layer(f, "feed_forward_length", d.layers);
        d.ffn = d.layer_ffn[0];
        d.double_wide = d.shared > 0 && d.shared < d.layers && d.layer_ffn.back() == 2 * d.ffn;
        for (int64_t i = 0; i < d.layers; ++i) {
            const int64_t want = i >= d.first_shared() && d.double_wide ? 2 * d.ffn : d.ffn;
            if (d.layer_ffn[static_cast<size_t>(i)] != want) {
                fail(name(), ": feed_forward_length of layer ", i, " is ", d.layer_ffn[static_cast<size_t>(i)],
                     "; Gemma 4 has one width, doubled on the KV-shared layers or not (", want, " here)");
            }
        }

        d.ple = f.get_int(key("embedding_length_per_layer_input"));
        if (d.ple < 0 || (assistant_ && d.ple != 0)) {
            fail(name(), ": embedding_length_per_layer_input is ", d.ple);
        }
        if (d.ple > 0) {
            const gguf::TensorInfo* table = f.find_tensor("per_layer_token_embd.weight");
            if (table == nullptr) {
                fail(name(), ": the file has no per_layer_token_embd.weight for its per-layer embeddings (a file ",
                     "written without them, --no-ple, is not supported)");
            }
            d.ple_vocab = table->rows();
        }

        // RoPE: sliding layers rotate their whole head; full-attention layers rotate the pairs rope_freqs gives a
        // factor of 1 (the converter writes rope.dimension_count as the whole head and leaves the rest to the factors).
        if (f.get_int(key("rope.dimension_count")) != d.global_head_dim) {
            fail(name(), ": rope.dimension_count ", f.get_int(key("rope.dimension_count")),
                 " is not the full-attention head size ", d.global_head_dim);
        }
        if (f.find(key("rope.dimension_count_swa")) && f.get_int(key("rope.dimension_count_swa")) != d.head_dim) {
            fail(name(), ": rope.dimension_count_swa ", f.get_int(key("rope.dimension_count_swa")),
                 " is not the sliding-attention head size ", d.head_dim, " (partial RoPE on sliding layers)");
        }
        d.rotated_pairs = d.global_head_dim / 2;
        if (const gguf::TensorInfo* freqs = f.find_tensor("rope_freqs.weight")) {
            d.rotated_pairs = rotated_pairs(*freqs, d.global_head_dim);
        }
        experts_and_values(f, d);
        return d;
    }

    // The mixture of experts beside every layer's MLP (expert_count, its three keys together), and whether the
    // full-attention layers take their keys as values: then none of them has attn_v, and every sliding one does.
    void experts_and_values(const gguf::File& f, Dimensions& d) const {
        const bool any = f.find(key("expert_count")) || f.find(key("expert_used_count")) ||
                         f.find(key("expert_feed_forward_length"));
        if (any) {
            if (assistant_) fail(name(), ": ", key("expert_count"), ": an assistant with experts is not supported");
            d.experts = positive(f, name(), key("expert_count"));
            d.experts_used = positive(f, name(), key("expert_used_count"));
            d.expert_ffn = positive(f, name(), key("expert_feed_forward_length"));
            if (d.experts_used > d.experts) {
                fail(name(), ": ", key("expert_used_count"), " ", d.experts_used, " of ", d.experts, " experts");
            }
            d.fused_gate_up = f.find_tensor("blk.0.ffn_gate_up_exps.weight") != nullptr;
        }
        int64_t without_v = 0, full = 0;
        for (int64_t i = 0; i < d.first_shared(); ++i) {
            const std::string b = str_cat("blk.", i, ".");
            const bool has_v = f.find_tensor(b + "attn_v.weight") != nullptr || f.find_tensor(b + "attn_qkv.weight");
            const bool sliding = d.sliding[static_cast<size_t>(i)];
            full += sliding ? 0 : 1;
            if (!has_v && sliding) {
                fail(name(), ": sliding-attention block ", i, " has no attn_v.weight (only full-attention layers ",
                     "take their keys as values)");
            }
            without_v += has_v ? 0 : 1;
        }
        if (without_v != 0 && without_v != full) {
            fail(name(), ": ", without_v, " of the ", full, " full-attention layers that compute their own KV have no ",
                 "attn_v.weight; Gemma 4 takes the keys as values in all of them (attention_k_eq_v) or none");
        }
        d.k_eq_v = without_v > 0;
    }

    // A decoder's KV-shared layers as llama.cpp's graph reads them (layer first_shared - 2 for a sliding one, - 1 for a
    // full one) and as Gemma 4 does (the last earlier layer of its kind): the file means the model only when the two
    // agree.
    void target_kv_sharing(const Dimensions& d) const {
        if (d.shared < 0 || d.shared >= d.layers) {
            fail(name(), ": attention.shared_kv_layers ", d.shared, " of ", d.layers, " layers");
        }
        if (d.shared > 0 && d.first_shared() < 2) {
            fail(name(), ": attention.shared_kv_layers ", d.shared, " leaves ", d.first_shared(), " layer computing ",
                 "its own KV; llama.cpp needs two");
        }
        for (int64_t i = d.first_shared(); i < d.layers; ++i) {
            const int64_t llama = d.first_shared() - (d.sliding[static_cast<size_t>(i)] ? 2 : 1);
            if (kv_source(d, i) != llama) {
                fail(name(), ": KV-shared layer ", i, " reads layer ", llama, "'s KV in llama.cpp but layer ",
                     kv_source(d, i), "'s in Gemma 4; a file whose layers are laid out so is not supported");
            }
        }
    }

    // The pairs of a full-attention head the converter's rope_freqs leaves rotating: a factor of 1 for each, then
    // 1e30 (a frequency of zero) for the rest.
    int64_t rotated_pairs(const gguf::TensorInfo& t, int64_t head_dim) const {
        if (t.dtype == DType::Block || t.numel() != head_dim / 2) {
            fail(name(), ": rope_freqs.weight is ", t.type_name(), " ", dims_to_string(t.shape()), "; expected ",
                 head_dim / 2, " float factors");
        }
        const std::vector<float> factors = float_rows(t, 0, t.rows());
        int64_t rotated = 0;
        while (rotated < static_cast<int64_t>(factors.size()) && factors[static_cast<size_t>(rotated)] == 1.0f)
            ++rotated;
        for (size_t i = static_cast<size_t>(rotated); i < factors.size(); ++i) {
            if (factors[i] != static_cast<float>(1e30)) {
                fail(name(), ": rope_freqs.weight holds ", factors[i], " at ", i, "; the import reads only the ",
                     "converter's proportional RoPE (factors of 1, then 1e30)");
            }
        }
        return rotated;
    }

    // The fields of a Gemma4TextConfig: the root of a text-only config.json, an assistant's text_config.
    nlohmann::ordered_json text_config(const gguf::File& f, const Dimensions& d) const {
        nlohmann::ordered_json c;
        c["model_type"] = "gemma4_text";
        // Gemma 4 is released in bf16, and the model code rounds its embedding scales to that dtype as transformers
        // does; the file does not say what it was converted from.
        c["dtype"] = "bfloat16";
        c["vocab_size"] = d.vocab;
        c["hidden_size"] = d.hidden;
        c["intermediate_size"] = d.ffn;
        c["use_double_wide_mlp"] = d.double_wide;
        c["num_hidden_layers"] = d.layers;
        nlohmann::ordered_json types = nlohmann::ordered_json::array();
        for (const bool s : d.sliding)
            types.push_back(s ? "sliding_attention" : "full_attention");
        c["layer_types"] = std::move(types);
        c["num_attention_heads"] = d.heads;
        c["num_key_value_heads"] = d.kv_heads;
        if (d.global_kv_heads != d.kv_heads) c["num_global_key_value_heads"] = d.global_kv_heads;
        c["head_dim"] = d.head_dim;
        c["global_head_dim"] = d.global_head_dim;
        c["attention_k_eq_v"] = d.k_eq_v;
        c["attention_bias"] = false;
        c["enable_moe_block"] = d.experts > 0;
        if (d.experts > 0) {
            c["num_experts"] = d.experts;
            c["top_k_experts"] = d.experts_used;
            c["moe_intermediate_size"] = d.expert_ffn;
        }
        c["hidden_activation"] = "gelu_pytorch_tanh";
        c["sliding_window"] = positive(f, name(), key("attention.sliding_window"));
        c["num_kv_shared_layers"] = d.shared;
        // An image's tokens attend to each other both ways in the sliding layers of every Gemma 4 but E2B and E4B
        // (the ones with per-layer embeddings): the releases' use_bidirectional_attention, which llama.cpp applies
        // to every gemma4 file but those two; the file does not say it.
        if (!assistant_ && d.ple == 0) c["use_bidirectional_attention"] = "vision";
        c["hidden_size_per_layer_input"] = d.ple;
        if (d.ple > 0) c["vocab_size_per_layer_input"] = d.ple_vocab;
        c["rms_norm_eps"] = config_number(f.get(key("attention.layer_norm_rms_epsilon")));
        if (const gguf::Value* s = f.find(key("final_logit_softcapping"))) {
            c["final_logit_softcapping"] = config_number(*s);
        } else {
            c["final_logit_softcapping"] = nullptr;
        }
        c["max_position_embeddings"] = positive(f, name(), key("context_length"));
        const gguf::Value& base = f.get(key("rope.freq_base"));
        const gguf::Value* base_swa = f.find(key("rope.freq_base_swa"));
        nlohmann::ordered_json full;
        if (d.rotated_pairs == d.global_head_dim / 2) {
            full = {{"rope_theta", config_number(base)}, {"rope_type", "default"}};
        } else {
            full = {{"partial_rotary_factor",
                     static_cast<double>(2 * d.rotated_pairs) / static_cast<double>(d.global_head_dim)},
                    {"rope_theta", config_number(base)},
                    {"rope_type", "proportional"}};
        }
        c["rope_parameters"] = {
            {"full_attention", std::move(full)},
            {"sliding_attention",
             {{"rope_theta", config_number(base_swa != nullptr ? *base_swa : base)}, {"rope_type", "default"}}}};
        c["tie_word_embeddings"] = f.find_tensor("output.weight") == nullptr;
        for (const auto& [k, field] : {std::pair{"tokenizer.ggml.bos_token_id", "bos_token_id"},
                                       std::pair{"tokenizer.ggml.padding_token_id", "pad_token_id"}}) {
            if (const gguf::Value* v = f.find(k)) c[field] = v->as_int();
        }
        return c;
    }

    Layout layout(const Dimensions& d) const {
        Layout l;
        l.top["token_embd.weight"] = one({d.vocab, d.hidden}, kPrefix + "embed_tokens.weight");
        l.top["output.weight"] = one({d.vocab, d.hidden}, "lm_head.weight");
        l.top["output.weight"].optional = true;
        l.top["output_norm.weight"] = one({d.hidden}, kPrefix + "norm.weight");
        l.top["rope_freqs.weight"] = Entry{{d.global_head_dim / 2}, {}, true};
        if (d.ple > 0) {
            const int64_t width = d.layers * d.ple;
            l.top["per_layer_token_embd.weight"] = one({d.ple_vocab, width}, kPrefix + "embed_tokens_per_layer.weight");
            l.top["per_layer_model_proj.weight"] =
                one({width, d.hidden}, kPrefix + "per_layer_model_projection.weight");
            l.top["per_layer_proj_norm.weight"] = one({d.ple}, kPrefix + "per_layer_projection_norm.weight");
        }
        if (assistant_) {
            // Its input: the target's scaled embedding of a token beside the target's hidden state; its output goes
            // back to the target's width for the next chained draft.
            l.top["nextn.pre_projection.weight"] = one({d.hidden, 2 * d.backbone}, "pre_projection.weight");
            l.top["nextn.post_projection.weight"] = one({d.backbone, d.hidden}, "post_projection.weight");
        }
        for (int64_t i = 0; i < d.layers; ++i)
            l.blocks.push_back(layer(d, i));
        const bool assistant = assistant_;
        l.block_kind = [d, assistant](int64_t i) {
            return str_cat(d.sliding[static_cast<size_t>(i)] ? "sliding" : "full", " attention",
                           assistant               ? ", over the target's KV"
                           : i >= d.first_shared() ? ", KV-shared"
                                                   : "");
        };
        return l;
    }

    bool assistant_;
};

}  // namespace

const Architecture& gemma4() {
    static const Gemma4 arch(false);
    return arch;
}

const Architecture& gemma4_assistant() {
    static const Gemma4 arch(true);
    return arch;
}

}  // namespace xs::gguf_import
