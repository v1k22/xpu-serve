// GGUF architectures "qwen35" and "qwen35moe": the Qwen3.5 family's text models (Qwen3.5, Qwen3.6, Qwen3.8), dense
// and mixture of experts, as llama.cpp's converter writes them (conversion/qwen.py: Qwen3_5TextModel,
// Qwen3_5MoeTextModel), imported for models/qwen3_5/. The two differ in their layers' MLP alone (Qwen35Mlp).
//
// What the converter changed, and what the import does about it:
// - names: blk.<i>.* for layer i, token_embd / output / output_norm; mapped back to the Hugging Face names.
// - a mixture of experts' fused bank, experts.gate_up_proj [E, 2 I, hidden] (an expert's gate rows, then its up rows),
//   is stored as two, ffn_gate_exps and ffn_up_exps [E, I, hidden], which can differ in format: kept apart, as
//   experts.gate_proj and experts.up_proj, which models/qwen3_5/ reads in the fused bank's place (a quantized weight is
//   never re-encoded, and the two halves of an expert are not runs of rows of one tensor).
// - the shared expert's gate, [1, hidden], can be stored 1-D: reshaped.
// - every norm weight but linear attention's (ssm_norm) is stored as 1 + w: written back as w.
// - A_log is stored as ssm_a = -exp(A_log): written back as log(-ssm_a).
// - the conv1d weight [C, 1, K] is stored squeezed, [C, K]: reshaped.
// - linear attention's value heads are stored tiled instead of grouped: kept, and config.json says
//   linear_value_head_order "tiled" (models/qwen3_5/qwen3_5.h), since a quantized weight is never reordered.
// - the converter can store a dense model's in_proj_qkv and in_proj_z as one tensor (attn_qkv_gate), and its in_proj_b
//   and in_proj_a (ssm_beta_alpha): cut back into their runs of rows.
// - the MTP head is one more block per MTP layer (blk.<n> past the decoder's layers, with nextn.* beside the first):
//   mapped to mtp.*.
// - a file can come without the head (nextn_predict_layers 0), its head published apart as llama.cpp's draft model
//   (`convert --mtp`): a file of the same model's metadata with one more block, which holds the head, and the draft
//   model's own token_embd, output and output_norm. The head's block is imported as a file that held it is; the other
//   three are checked against the model's shapes and left out (ImportPlan::copies): the Hugging Face head reads the
//   model's embedding and lm_head (mtp_use_dedicated_embeddings false), and the final norm of its own block
//   (nextn.shared_head_norm, mtp.norm).
//
// - the vision tower is not in the file: llama.cpp writes it to a file of its own (mmproj: general.architecture clip,
//   projector type qwen3vl_merger), which `convert --mmproj` reads into the pack under the checkpoint's names
//   (model.visual.*), with the release's config.json layout around the text config (text_config, vision_config). The
//   converter cuts the patch embedding, a convolution over two frames, into one kernel a frame: kept apart, as
//   patch_embed.proj.frames.<t>.weight, which models/qwen3_vision/ reads in the whole kernel's place.
//
// The decoder is read under any namespace (read_qwen35_decoder, qwen35.h): clef files hold it under clef.*.

#include "convert/gguf/qwen35.h"

#include <charconv>
#include <cmath>

#include "util/error.h"

namespace xs::gguf_import {
namespace {

// A text-only checkpoint's decoder, as Qwen3_5ForCausalLM and Qwen3_5MoeForCausalLM name it.
const std::string kPrefix = "model.";

float minus_one(float x) {
    return x - 1.0f;
}
float log_of_negative(float x) {
    return std::log(-x);
}
const FloatMap kNorm{"w = (1 + w) - 1 of a norm weight", minus_one};
const FloatMap kALog{"A_log = log(-ssm_a)", log_of_negative};

// The decoder's keys the import reads, or knows to change nothing it writes, whatever its layers' MLP. An MTP head's
// (nextn_predict_layers) is the architecture's to allow.
const std::set<std::string_view> kDecoderKeys = {
    "block_count",
    "context_length",
    "embedding_length",
    "vocab_size",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "attention.recurrent_layers",
    "full_attention_interval",
    "rope.freq_base",
    "rope.dimension_count",
    "rope.dimension_sections",  // multimodal RoPE: text tokens rotate as with plain RoPE
    "rope.scaling.type",        // only "none"
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "ssm.time_step_rank",
    "ssm.inner_size",
};

// The keys of each kind of MLP, all of them required.
const std::set<std::string_view> kDenseKeys = {"feed_forward_length"};
const std::set<std::string_view> kExpertKeys = {
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_shared_feed_forward_length",
};

struct Dimensions {
    int64_t vocab, hidden, heads, kv_heads, head_dim;
    int64_t key_heads, value_heads, key_dim, value_dim, conv_kernel;
    int64_t ffn;  // the dense MLP's width; 0 with experts
    // A mixture of experts (experts > 0): how many, how many a token uses, an expert's width and the shared expert's.
    int64_t experts, experts_used, expert_ffn, shared_ffn;
    int64_t key_width() const { return key_heads * key_dim; }
    int64_t value_width() const { return value_heads * value_dim; }
    int64_t conv_width() const { return 2 * key_width() + value_width(); }
};

// A layer's MLP under `M` (".../mlp."): the dense one, or the router, the expert banks and the gated shared expert.
void add_mlp(Entries& e, const Dimensions& d, const std::string& M) {
    if (d.experts == 0) {
        e["ffn_gate.weight"] = one({d.ffn, d.hidden}, M + "gate_proj.weight");
        e["ffn_up.weight"] = one({d.ffn, d.hidden}, M + "up_proj.weight");
        e["ffn_down.weight"] = one({d.hidden, d.ffn}, M + "down_proj.weight");
        return;
    }
    e["ffn_gate_inp.weight"] = one({d.experts, d.hidden}, M + "gate.weight");
    e["ffn_gate_exps.weight"] = one({d.experts, d.expert_ffn, d.hidden}, M + "experts.gate_proj");
    e["ffn_up_exps.weight"] = one({d.experts, d.expert_ffn, d.hidden}, M + "experts.up_proj");
    e["ffn_down_exps.weight"] = one({d.experts, d.hidden, d.expert_ffn}, M + "experts.down_proj");
    e["ffn_gate_shexp.weight"] = one({d.shared_ffn, d.hidden}, M + "shared_expert.gate_proj.weight");
    e["ffn_up_shexp.weight"] = one({d.shared_ffn, d.hidden}, M + "shared_expert.up_proj.weight");
    e["ffn_down_shexp.weight"] = one({d.hidden, d.shared_ffn}, M + "shared_expert.down_proj.weight");
    e["ffn_gate_inp_shexp.weight"] = one({1, d.hidden}, M + "shared_expert_gate.weight");
}

// A decoder layer of either kind: the norms and the MLP.
void add_layer_common(Entries& e, const Dimensions& d, const std::string& L) {
    e["attn_norm.weight"] = one({d.hidden}, L + "input_layernorm.weight", &kNorm);
    e["post_attention_norm.weight"] = one({d.hidden}, L + "post_attention_layernorm.weight", &kNorm);
    add_mlp(e, d, L + "mlp.");
}

// Gated full attention: q_proj holds each head's query and output gate.
Entries full_attention_layer(const Dimensions& d, const std::string& L) {
    Entries e;
    add_layer_common(e, d, L);
    const std::string A = L + "self_attn.";
    e["attn_q.weight"] = one({2 * d.heads * d.head_dim, d.hidden}, A + "q_proj.weight");
    e["attn_k.weight"] = one({d.kv_heads * d.head_dim, d.hidden}, A + "k_proj.weight");
    e["attn_v.weight"] = one({d.kv_heads * d.head_dim, d.hidden}, A + "v_proj.weight");
    e["attn_output.weight"] = one({d.hidden, d.heads * d.head_dim}, A + "o_proj.weight");
    e["attn_q_norm.weight"] = one({d.head_dim}, A + "q_norm.weight", &kNorm);
    e["attn_k_norm.weight"] = one({d.head_dim}, A + "k_norm.weight", &kNorm);
    return e;
}

// Gated DeltaNet; in_proj_qkv / in_proj_z and in_proj_b / in_proj_a stored apart, or in a dense model's file as one
// tensor each (the converter merges them for Qwen3_5TextModel alone, and llama.cpp reads the merged tensors in no
// qwen35moe file).
Entries linear_attention_layer(const Dimensions& d, const std::string& L) {
    Entries e;
    add_layer_common(e, d, L);
    const std::string A = L + "linear_attn.";
    const Piece qkv{A + "in_proj_qkv.weight", {d.conv_width(), d.hidden}};
    const Piece z{A + "in_proj_z.weight", {d.value_width(), d.hidden}};
    const Piece b{A + "in_proj_b.weight", {d.value_heads, d.hidden}};
    const Piece a{A + "in_proj_a.weight", {d.value_heads, d.hidden}};
    e["attn_qkv.weight"] = Entry{qkv.shape, {qkv}};
    e["attn_gate.weight"] = Entry{z.shape, {z}};
    e["ssm_beta.weight"] = Entry{b.shape, {b}};
    e["ssm_alpha.weight"] = Entry{a.shape, {a}};
    if (d.experts == 0) {
        e["attn_qkv_gate.weight"] = Entry{{d.conv_width() + d.value_width(), d.hidden}, {qkv, z}};
        e["ssm_beta_alpha.weight"] = Entry{{2 * d.value_heads, d.hidden}, {b, a}};
    }
    e["ssm_a"] = one({d.value_heads}, A + "A_log", &kALog);
    e["ssm_dt.bias"] = one({d.value_heads}, A + "dt_bias");
    e["ssm_conv1d.weight"] =
        Entry{{d.conv_width(), d.conv_kernel}, {Piece{A + "conv1d.weight", {d.conv_width(), 1, d.conv_kernel}}}};
    e["ssm_norm.weight"] = one({d.value_dim}, A + "norm.weight");
    e["ssm_out.weight"] = one({d.hidden, d.value_width()}, A + "out_proj.weight");
    return e;
}

// MTP layer `i` (block n_layers + i): a full-attention layer; the first also holds the head's input projection and
// norms.
Entries mtp_layer(const Dimensions& d, int64_t i) {
    Entries e = full_attention_layer(d, "mtp.layers." + std::to_string(i) + ".");
    if (i == 0) {
        e["nextn.eh_proj.weight"] = one({d.hidden, 2 * d.hidden}, "mtp.fc.weight");
        e["nextn.enorm.weight"] = one({d.hidden}, "mtp.pre_fc_norm_embedding.weight", &kNorm);
        e["nextn.hnorm.weight"] = one({d.hidden}, "mtp.pre_fc_norm_hidden.weight", &kNorm);
        e["nextn.shared_head_norm.weight"] = one({d.hidden}, "mtp.norm.weight", &kNorm);
    }
    return e;
}

// A metadata value as an error shows it: its type and value (an array of integers: its values; another array: its
// length), or "absent".
std::string shown(const gguf::Value* v) {
    if (v == nullptr) return "absent";
    const std::string_view type = gguf::value_type_name(v->type());
    switch (v->type()) {
        case gguf::ValueType::Array: {
            const gguf::ValueType e = v->element_type();
            const bool integers = e != gguf::ValueType::String && e != gguf::ValueType::Array &&
                                  e != gguf::ValueType::Bool && e != gguf::ValueType::F32 && e != gguf::ValueType::F64;
            if (integers) return str_cat(gguf::value_type_name(e), " ", dims_to_string(v->ints()));
            return str_cat("an array of ", v->size(), " ", gguf::value_type_name(e));
        }
        case gguf::ValueType::String:
            return str_cat("'", v->as_string(), "'");
        case gguf::ValueType::Bool:
            return v->as_bool() ? "true" : "false";
        case gguf::ValueType::F32:
        case gguf::ValueType::F64: {
            char text[32];
            const char* end = std::to_chars(text, text + sizeof(text), config_number(*v)).ptr;
            return str_cat(type, " ", std::string_view(text, static_cast<size_t>(end - text)));
        }
        default:
            return str_cat(type, " ", v->as_int());
    }
}

// The MTP file `head` describes the model of `f` (a file of `blocks` blocks, none of them MTP layers) with one more
// block, the head: the same architecture, and every other <arch>.* key the same in both (the per-block
// attention.recurrent_layers: the model's blocks, then the head's, not recurrent).
void check_head_metadata(const gguf::File& f, const gguf::File& head, std::string_view arch, int64_t blocks) {
    const auto key = [&](std::string_view rest) { return str_cat(arch, ".", rest); };
    if (const std::string_view a = head.get_string("general.architecture"); a != arch) {
        fail(arch, ": the MTP file is general.architecture '", a, "', not ", arch);
    }
    if (const int64_t n = head.get_int(key("nextn_predict_layers")); n != 1) {
        fail(arch, ": the MTP file has nextn_predict_layers ", n, "; an MTP file holds one MTP layer");
    }
    if (const int64_t n = head.get_int(key("block_count")); n != blocks + 1) {
        fail(arch, ": the MTP file has block_count ", n, ", not the model file's ", blocks, " blocks and the head");
    }
    std::set<std::string_view> keys;
    for (const gguf::File* file : {&f, &head}) {
        for (const gguf::Value& v : file->metadata()) {
            if (v.key().starts_with(key(""))) keys.insert(v.key());
        }
    }
    for (const std::string_view k : keys) {
        if (k == key("block_count") || k == key("nextn_predict_layers")) continue;
        const gguf::Value* ours = f.find(k);
        const gguf::Value* theirs = head.find(k);
        bool same = ours != nullptr && theirs != nullptr && ours->same_as(*theirs);
        if (k == key("attention.recurrent_layers") && ours != nullptr && theirs != nullptr) {
            std::vector<bool> list = ours->bools();
            list.push_back(false);
            same = theirs->bools() == list;
        }
        if (!same) {
            fail(arch, ": the MTP file's ", k, " (", shown(theirs), ") differs from the model file's (", shown(ours),
                 "): it is the head of another model");
        }
    }
}

// The tensors the MTP file `head` adds to the model's: its block `layers`, the head, through the model's layout as a
// file that held it; and its own copies of the model's top-level tensors, checked against the model's shapes and left
// out.
void add_head_tensors(const gguf::File& f, const gguf::File& head, std::string_view arch, int64_t layers,
                      const Layout& l, std::vector<const gguf::TensorInfo*>& tensors,
                      std::vector<const gguf::TensorInfo*>& copies) {
    const std::string block = str_cat("blk.", layers, ".");
    for (const gguf::TensorInfo& t : f.tensors()) {
        if (t.name.starts_with(block)) {
            fail(arch, ": tensor '", t.name, "' is not in one of the model file's ", layers, " blocks");
        }
    }
    for (const gguf::TensorInfo& t : head.tensors()) {
        if (t.name.starts_with(block)) {
            tensors.push_back(&t);
        } else if (const auto e = l.top.find(t.name); e != l.top.end()) {
            if (!same_shape(t.shape(), e->second.stored)) {
                fail(arch, ": the MTP file's '", t.name, "' has shape ", dims_to_string(t.shape()), "; the model's is ",
                     dims_to_string(e->second.stored));
            }
            copies.push_back(&t);
        } else {
            fail(arch, ": the MTP file's tensor '", t.name, "' is neither in the head's block ", layers,
                 " nor one of the model's token_embd, output and output_norm");
        }
    }
}

nlohmann::ordered_json config(const gguf::File& f, std::string_view arch, const Dimensions& d, int64_t layers,
                              int64_t mtp, const std::vector<bool>& linear) {
    const auto key = [&](std::string_view rest) { return str_cat(arch, ".", rest); };
    const bool experts = d.experts > 0;
    nlohmann::ordered_json c;
    c["architectures"] = {experts ? "Qwen3_5MoeForCausalLM" : "Qwen3_5ForCausalLM"};
    c["model_type"] = experts ? "qwen3_5_moe_text" : "qwen3_5_text";
    c["vocab_size"] = d.vocab;
    c["hidden_size"] = d.hidden;
    if (experts) {
        c["num_experts"] = d.experts;
        c["num_experts_per_tok"] = d.experts_used;
        c["moe_intermediate_size"] = d.expert_ffn;
        c["shared_expert_intermediate_size"] = d.shared_ffn;
    } else {
        c["intermediate_size"] = d.ffn;
    }
    c["num_hidden_layers"] = layers;
    nlohmann::ordered_json types = nlohmann::ordered_json::array();
    for (const bool l : linear)
        types.push_back(l ? "linear_attention" : "full_attention");
    c["layer_types"] = std::move(types);
    if (f.find(key("full_attention_interval")))
        c["full_attention_interval"] = f.get_int(key("full_attention_interval"));
    c["num_attention_heads"] = d.heads;
    c["num_key_value_heads"] = d.kv_heads;
    c["head_dim"] = d.head_dim;
    c["attn_output_gate"] = true;
    c["attention_bias"] = false;
    c["hidden_act"] = "silu";
    c["linear_num_key_heads"] = d.key_heads;
    c["linear_num_value_heads"] = d.value_heads;
    c["linear_key_head_dim"] = d.key_dim;
    c["linear_value_head_dim"] = d.value_dim;
    c["linear_conv_kernel_dim"] = d.conv_kernel;
    c["linear_value_head_order"] = "tiled";
    c["rms_norm_eps"] = config_number(f.get(key("attention.layer_norm_rms_epsilon")));
    c["max_position_embeddings"] = positive(f, arch, key("context_length"));
    const int64_t rotary =
        f.find(key("rope.dimension_count")) ? positive(f, arch, key("rope.dimension_count")) : d.head_dim;
    if (rotary > d.head_dim || rotary % 2 != 0) {
        fail(arch, ": rope.dimension_count ", rotary, " does not fit head_dim ", d.head_dim);
    }
    nlohmann::ordered_json rope;
    rope["rope_type"] = "default";
    rope["rope_theta"] = config_number(f.get(key("rope.freq_base")));
    rope["partial_rotary_factor"] = static_cast<double>(rotary) / static_cast<double>(d.head_dim);
    if (const gguf::Value* s = f.find(key("rope.dimension_sections"))) {
        std::vector<int64_t> sections = s->ints();
        while (sections.size() > 3 && sections.back() == 0)
            sections.pop_back();
        rope["mrope_section"] = sections;
        rope["mrope_interleaved"] = true;
    }
    c["rope_parameters"] = std::move(rope);
    c["tie_word_embeddings"] = f.find_tensor("output.weight") == nullptr;
    c["mtp_num_hidden_layers"] = mtp;
    c["mtp_use_dedicated_embeddings"] = false;
    return c;
}

Layout layout(const Dimensions& d, int64_t layers, int64_t blocks, const std::vector<bool>& linear) {
    Layout l;
    l.top["token_embd.weight"] = one({d.vocab, d.hidden}, kPrefix + "embed_tokens.weight");
    l.top["output_norm.weight"] = one({d.hidden}, kPrefix + "norm.weight", &kNorm);
    l.top["output.weight"] = one({d.vocab, d.hidden}, "lm_head.weight");
    l.top["output.weight"].optional = true;
    l.blocks.resize(static_cast<size_t>(blocks));
    for (int64_t i = 0; i < blocks; ++i) {
        const std::string L = kPrefix + "layers." + std::to_string(i) + ".";
        l.blocks[static_cast<size_t>(i)] = i >= layers                      ? mtp_layer(d, i - layers)
                                           : linear[static_cast<size_t>(i)] ? linear_attention_layer(d, L)
                                                                            : full_attention_layer(d, L);
    }
    l.block_kind = [layers, linear](int64_t i) -> std::string {
        return i >= layers ? "an MTP layer" : linear[static_cast<size_t>(i)] ? "linear attention" : "full attention";
    };
    return l;
}

// The language model of a file: its decoder, and its MTP head when the file, or a file of its own, holds one.
// The vision tower's file (llama.cpp's mmproj): Qwen3-VL's tower without deepstack layers, as Qwen3.5, 3.6 and 3.8
// have it.
constexpr std::string_view kProjector = "clip";
constexpr std::string_view kProjectorType = "qwen3vl_merger";
const std::string kVisionPrefix = "model.visual.";
// image_size is llama.cpp's own bound on an image; the processor here is the checkpoint's (preprocessor_config.json).
const std::set<std::string_view> kProjectorKeys = {
    "has_vision_encoder", "projector_type", "use_gelu", "vision.projection_dim", "vision.image_size",
    "vision.patch_size", "vision.embedding_length", "vision.feed_forward_length", "vision.block_count",
    "vision.attention.head_count", "vision.image_mean", "vision.image_std", "vision.spatial_merge_size",
    "vision.attention.layer_norm_epsilon", "vision.is_deepstack_layers",
};

struct VisionDimensions {
    int64_t hidden = 0, width = 0, layers = 0, heads = 0, patch = 0, merge = 0, frames = 0, positions = 0, out = 0;
};

// The projector file `f` against the model's hidden size: its dimensions, refusing by name a file that is not this
// family's tower (or another model's) and settings the tower here does not read as the file means them.
VisionDimensions vision_dimensions(const gguf::File& f, std::string_view arch, int64_t hidden) {
    const std::string_view general = f.get_string("general.architecture");
    const gguf::Value* type = f.find("clip.projector_type");
    if (general != kProjector || type == nullptr || type->as_string() != kProjectorType) {
        fail(arch, ": --mmproj: the file is not a Qwen3.5-family vision tower (general.architecture '", general,
             "', clip.projector_type '", type != nullptr ? type->as_string() : std::string_view("none"),
             "'; expected '", kProjector, "' and '", kProjectorType, "')");
    }
    refuse_unknown_keys(f, kProjector, kProjectorKeys);
    const auto positive_of = [&](std::string_view rest) { return positive(f, kProjector, str_cat("clip.", rest)); };
    VisionDimensions v;
    v.hidden = positive_of("vision.embedding_length");
    v.width = positive_of("vision.feed_forward_length");
    v.layers = positive_of("vision.block_count");
    v.heads = positive_of("vision.attention.head_count");
    v.patch = positive_of("vision.patch_size");
    v.merge = positive_of("vision.spatial_merge_size");
    v.out = positive_of("vision.projection_dim");
    XS_CHECK(v.out == hidden, arch, ": --mmproj: the tower projects to ", v.out, ", the model's hidden size is ",
             hidden, ": another model's file");
    if (const gguf::Value* deep = f.find("clip.vision.is_deepstack_layers")) {
        for (const bool layer : deep->bools()) {
            XS_CHECK(!layer, arch, ": --mmproj: the tower has deepstack layers (Qwen3-VL's), which this family's ",
                     "does not");
        }
    }
    const gguf::Value* gelu = f.find("clip.use_gelu");
    XS_CHECK(gelu != nullptr && gelu->as_bool(), arch, ": --mmproj: the tower's blocks are not GELU (clip.use_gelu)");
    for (const char* key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
        const gguf::Value* m = f.find(key);
        XS_CHECK(m != nullptr, arch, ": --mmproj: the file has no ", key);
        for (const double x : m->floats()) XS_CHECK(x == 0.5, arch, ": --mmproj: ", key, " is not 0.5 a channel");
    }
    const double eps = f.get("clip.vision.attention.layer_norm_epsilon").as_float();
    XS_CHECK(std::fabs(eps - 1e-6) < 1e-9, arch, ": --mmproj: layer_norm_epsilon ", eps, " (the tower's is 1e-6)");
    const gguf::TensorInfo* table = f.find_tensor("v.position_embd.weight");
    XS_CHECK(table != nullptr && table->dims.size() == 2, arch, ": --mmproj: the file lacks v.position_embd.weight");
    v.positions = table->rows();
    v.frames = f.find_tensor("v.patch_embd.weight.1") != nullptr ? 2 : 1;
    return v;
}

// vision_config as the release's config.json has it.
nlohmann::ordered_json vision_config(const VisionDimensions& v, bool experts) {
    nlohmann::ordered_json c;
    c["model_type"] = experts ? "qwen3_5_moe" : "qwen3_5";
    c["depth"] = v.layers;
    c["hidden_size"] = v.hidden;
    c["intermediate_size"] = v.width;
    c["num_heads"] = v.heads;
    c["out_hidden_size"] = v.out;
    c["patch_size"] = v.patch;
    c["spatial_merge_size"] = v.merge;
    c["temporal_patch_size"] = v.frames;
    c["in_channels"] = 3;
    c["num_position_embeddings"] = v.positions;
    c["hidden_act"] = "gelu_pytorch_tanh";
    c["deepstack_visual_indexes"] = nlohmann::ordered_json::array();
    return c;
}

// The release's image processor (preprocessor_config.json, the same for Qwen3.5, 3.6 and 3.8), which the file does
// not hold: its patch and merge sizes are the file's.
nlohmann::ordered_json preprocessor_config(const VisionDimensions& v) {
    nlohmann::ordered_json c;
    c["size"] = {{"longest_edge", 16777216}, {"shortest_edge", 65536}};
    c["patch_size"] = v.patch;
    c["temporal_patch_size"] = v.frames;
    c["merge_size"] = v.merge;
    c["image_mean"] = {0.5, 0.5, 0.5};
    c["image_std"] = {0.5, 0.5, 0.5};
    c["processor_class"] = "Qwen3VLProcessor";
    c["image_processor_type"] = "Qwen2VLImageProcessorFast";
    return c;
}

Layout vision_layout(const VisionDimensions& v) {
    const std::string& P = kVisionPrefix;
    const int64_t h = v.hidden;
    Layout l;
    for (int64_t t = 0; t < v.frames; ++t) {
        l.top[t == 0 ? "v.patch_embd.weight" : "v.patch_embd.weight." + std::to_string(t)] =
            one({h, 3, v.patch, v.patch}, P + "patch_embed.proj.frames." + std::to_string(t) + ".weight");
    }
    l.top["v.patch_embd.bias"] = one({h}, P + "patch_embed.proj.bias");
    l.top["v.position_embd.weight"] = one({v.positions, h}, P + "pos_embed.weight");
    for (int64_t i = 0; i < v.layers; ++i) {
        const std::string from = "v.blk." + std::to_string(i) + ".", to = P + "blocks." + std::to_string(i) + ".";
        for (const auto& [a, b] : {std::pair{"ln1", "norm1"}, std::pair{"ln2", "norm2"}}) {
            l.top[from + a + ".weight"] = one({h}, to + b + ".weight");
            l.top[from + a + ".bias"] = one({h}, to + b + ".bias");
        }
        l.top[from + "attn_qkv.weight"] = one({3 * h, h}, to + "attn.qkv.weight");
        l.top[from + "attn_qkv.bias"] = one({3 * h}, to + "attn.qkv.bias");
        l.top[from + "attn_out.weight"] = one({h, h}, to + "attn.proj.weight");
        l.top[from + "attn_out.bias"] = one({h}, to + "attn.proj.bias");
        l.top[from + "ffn_up.weight"] = one({v.width, h}, to + "mlp.linear_fc1.weight");
        l.top[from + "ffn_up.bias"] = one({v.width}, to + "mlp.linear_fc1.bias");
        l.top[from + "ffn_down.weight"] = one({h, v.width}, to + "mlp.linear_fc2.weight");
        l.top[from + "ffn_down.bias"] = one({h}, to + "mlp.linear_fc2.bias");
    }
    const int64_t joined = v.merge * v.merge * h;
    l.top["v.post_ln.weight"] = one({h}, P + "merger.norm.weight");
    l.top["v.post_ln.bias"] = one({h}, P + "merger.norm.bias");
    l.top["mm.0.weight"] = one({joined, joined}, P + "merger.linear_fc1.weight");
    l.top["mm.0.bias"] = one({joined}, P + "merger.linear_fc1.bias");
    l.top["mm.2.weight"] = one({v.out, joined}, P + "merger.linear_fc2.weight");
    l.top["mm.2.bias"] = one({v.out}, P + "merger.linear_fc2.bias");
    return l;
}

// The id of the token spelled `text` in the file's vocabulary.
int32_t token_id(const gguf::File& f, std::string_view arch, std::string_view text) {
    const std::vector<std::string_view> tokens = f.get("tokenizer.ggml.tokens").strings();
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == text) return static_cast<int32_t>(i);
    }
    fail(arch, ": --mmproj: the vocabulary has no ", text);
}

class Qwen35 final : public Architecture {
public:
    Qwen35(std::string_view name, Qwen35Mlp mlp) : name_(name), mlp_(mlp) {}

    std::string_view name() const override { return name_; }

    ImportPlan plan(const gguf::File& f) const override { return plan_from(f, nullptr); }
    ImportPlan plan_with_mtp(const gguf::File& f, const gguf::File& mtp) const override { return plan_from(f, &mtp); }

    // A file without its MTP head (nextn_predict_layers 0: unsloth's smallest Qwen3.8-27B files) takes llama.cpp's
    // draft model.
    bool drafter_apart(const gguf::File& f) const override {
        const gguf::Value* n = f.find(str_cat(name_, ".nextn_predict_layers"));
        return n == nullptr || n->as_int() == 0;
    }
    bool reads_projector() const override { return true; }

    // The tower's tensors under the checkpoint's names, and config.json as the release lays it out: the text config,
    // vision_config and the image token around it.
    void add_projector(ImportPlan& p, const gguf::File& f, const gguf::File& projector) const override {
        const bool experts = mlp_ == Qwen35Mlp::Experts;
        const VisionDimensions v =
            vision_dimensions(projector, name_, positive(f, name_, str_cat(name_, ".embedding_length")));
        nlohmann::ordered_json config;
        config["architectures"] = {experts ? "Qwen3_5MoeForConditionalGeneration" : "Qwen3_5ForConditionalGeneration"};
        config["model_type"] = experts ? "qwen3_5_moe" : "qwen3_5";
        config["image_token_id"] = token_id(f, name_, "<|image_pad|>");
        config["text_config"] = std::move(p.config);
        config["vision_config"] = vision_config(v, experts);
        p.config = std::move(config);
        p.files["preprocessor_config.json"] = preprocessor_config(v);
        map_tensors(projector, kProjector, vision_layout(v), p);
    }

private:
    // The plan of the model file `f`, with its MTP head from the file `head` when one is given.
    ImportPlan plan_from(const gguf::File& f, const gguf::File* head) const {
        const Qwen35Decoder d = read_qwen35_decoder(f, name_, mlp_, {"nextn_predict_layers"}, head);
        ImportPlan plan;
        plan.config = d.config;
        plan.copies = d.copies;
        map_tensors(d.tensors, name_, d.layout, plan);
        return plan;
    }

    std::string_view name_;
    Qwen35Mlp mlp_;
};

}  // namespace

Qwen35Decoder read_qwen35_decoder(const gguf::File& f, std::string_view arch, Qwen35Mlp mlp,
                                  const std::set<std::string_view>& keys, const gguf::File* head) {
    const auto key = [&](std::string_view rest) { return str_cat(arch, ".", rest); };
    const bool experts = mlp == Qwen35Mlp::Experts;
    const std::set<std::string_view>& mlp_keys = experts ? kExpertKeys : kDenseKeys;
    std::set<std::string_view> known = kDecoderKeys;
    known.insert(mlp_keys.begin(), mlp_keys.end());
    known.insert(keys.begin(), keys.end());
    refuse_unknown_keys(f, arch, known);
    if (const gguf::Value* s = f.find(key("rope.scaling.type")); s != nullptr && s->as_string() != "none") {
        fail(arch, ": ", key("rope.scaling.type"), " '", s->as_string(), "' is not supported (supported: none)");
    }

    const gguf::TensorInfo* embed = f.find_tensor("token_embd.weight");
    if (embed == nullptr) fail(arch, ": the file has no token_embd.weight");
    Dimensions d{};
    d.vocab = embed->rows();
    d.hidden = positive(f, arch, key("embedding_length"));
    if (experts) {
        d.experts = positive(f, arch, key("expert_count"));
        d.experts_used = positive(f, arch, key("expert_used_count"));
        d.expert_ffn = positive(f, arch, key("expert_feed_forward_length"));
        d.shared_ffn = positive(f, arch, key("expert_shared_feed_forward_length"));
        if (d.experts_used > d.experts) {
            fail(arch, ": expert_used_count ", d.experts_used, " of expert_count ", d.experts);
        }
    } else {
        d.ffn = positive(f, arch, key("feed_forward_length"));
    }
    d.heads = positive(f, arch, key("attention.head_count"));
    d.kv_heads = positive(f, arch, key("attention.head_count_kv"));
    d.head_dim =
        f.find(key("attention.key_length")) ? positive(f, arch, key("attention.key_length")) : d.hidden / d.heads;
    if (f.find(key("attention.value_length")) && f.get_int(key("attention.value_length")) != d.head_dim) {
        fail(arch, ": attention.value_length ", f.get_int(key("attention.value_length")), " differs from ",
             "key_length ", d.head_dim);
    }
    d.key_heads = positive(f, arch, key("ssm.group_count"));
    d.value_heads = positive(f, arch, key("ssm.time_step_rank"));
    d.key_dim = positive(f, arch, key("ssm.state_size"));
    d.conv_kernel = positive(f, arch, key("ssm.conv_kernel"));
    const int64_t inner = positive(f, arch, key("ssm.inner_size"));
    if (inner % d.value_heads != 0 || d.value_heads % d.key_heads != 0) {
        fail(arch, ": ssm.inner_size ", inner, ", time_step_rank ", d.value_heads, " and group_count ", d.key_heads,
             " are not whole value heads per key head");
    }
    d.value_dim = inner / d.value_heads;
    if (const gguf::Value* v = f.find(key("vocab_size")); v != nullptr && v->as_int() != d.vocab) {
        fail(arch, ": vocab_size ", v->as_int(), " differs from token_embd.weight's ", d.vocab, " rows");
    }
    if (f.get("tokenizer.ggml.tokens").size() > static_cast<uint64_t>(d.vocab)) {
        fail(arch, ": the tokenizer has ", f.get("tokenizer.ggml.tokens").size(), " tokens for ", d.vocab,
             " embedding rows");
    }

    int64_t blocks = positive(f, arch, key("block_count"));
    int64_t mtp = f.find(key("nextn_predict_layers")) ? f.get_int(key("nextn_predict_layers")) : 0;
    if (mtp < 0 || mtp >= blocks) fail(arch, ": nextn_predict_layers ", mtp, " of ", blocks, " blocks");
    if (head != nullptr && mtp != 0) {
        fail(arch, ": --mtp: the model file already has an MTP head (nextn_predict_layers ", mtp, ")");
    }
    const int64_t layers = blocks - mtp;
    // Which layers are linear attention: the file's list, or every one but each full_attention_interval-th (llama.cpp's
    // reading, 4 when the file gives no interval either).
    std::vector<bool> linear(static_cast<size_t>(layers));
    if (const gguf::Value* r = f.find(key("attention.recurrent_layers"))) {
        const std::vector<bool> list = r->bools();
        if (static_cast<int64_t>(list.size()) != blocks) {
            fail(arch, ": attention.recurrent_layers has ", list.size(), " entries for ", blocks, " blocks");
        }
        for (int64_t i = 0; i < blocks; ++i) {
            if (i < layers) {
                linear[static_cast<size_t>(i)] = list[static_cast<size_t>(i)];
            } else if (list[static_cast<size_t>(i)]) {
                fail(arch, ": attention.recurrent_layers marks MTP block ", i, " recurrent");
            }
        }
    } else {
        const int64_t interval =
            f.find(key("full_attention_interval")) ? positive(f, arch, key("full_attention_interval")) : 4;
        for (int64_t i = 0; i < layers; ++i)
            linear[static_cast<size_t>(i)] = (i + 1) % interval != 0;
    }
    if (head != nullptr) {
        check_head_metadata(f, *head, arch, blocks);
        blocks += 1;
        mtp = 1;
    }

    Qwen35Decoder out;
    out.config = config(f, arch, d, layers, mtp, linear);
    out.layout = layout(d, layers, blocks, linear);
    for (const gguf::TensorInfo& t : f.tensors())
        out.tensors.push_back(&t);
    if (head != nullptr) add_head_tensors(f, *head, arch, layers, out.layout, out.tensors, out.copies);
    return out;
}

const Architecture& qwen35() {
    static const Qwen35 arch("qwen35", Qwen35Mlp::Dense);
    return arch;
}

const Architecture& qwen35moe() {
    static const Qwen35 arch("qwen35moe", Qwen35Mlp::Experts);
    return arch;
}

}  // namespace xs::gguf_import
