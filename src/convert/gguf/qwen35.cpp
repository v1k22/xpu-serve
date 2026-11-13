// GGUF architecture "qwen35": the Qwen3.5 family's dense text models (Qwen3.5, Qwen3.6, Qwen3.8) as llama.cpp's
// converter writes them (convert_hf_to_gguf.py, Qwen3_5TextModel), imported for models/qwen3_5/.
//
// What the converter changed, and what the import does about it:
// - names: blk.<i>.* for layer i, token_embd / output / output_norm; mapped back to the Hugging Face names.
// - every norm weight but linear attention's (ssm_norm) is stored as 1 + w: written back as w.
// - A_log is stored as ssm_a = -exp(A_log): written back as log(-ssm_a).
// - the conv1d weight [C, 1, K] is stored squeezed, [C, K]: reshaped.
// - linear attention's value heads are stored tiled instead of grouped: kept, and config.json says
//   linear_value_head_order "tiled" (models/qwen3_5/qwen3_5.h), since a quantized weight is never reordered.
// - the converter can store in_proj_qkv and in_proj_z as one tensor (attn_qkv_gate), and in_proj_b and in_proj_a
//   (ssm_beta_alpha): cut back into their runs of rows.
// - the MTP head is one more block per MTP layer (blk.<n> past the decoder's layers, with nextn.* beside the first):
//   mapped to mtp.*.
// - a file can come without the head (nextn_predict_layers 0), its head published apart as llama.cpp's draft model
//   (`convert --mtp`): a file of the same model's metadata with one more block, which holds the head, and the draft
//   model's own token_embd, output and output_norm. The head's block is imported as a file that held it is; the other
//   three are checked against the model's shapes and left out (ImportPlan::copies): the Hugging Face head reads the
//   model's embedding and lm_head (mtp_use_dedicated_embeddings false), and the final norm of its own block
//   (nextn.shared_head_norm, mtp.norm).

#include <charconv>
#include <cmath>
#include <set>

#include "convert/gguf/architecture.h"
#include "convert/gguf/layout.h"
#include "util/error.h"

namespace xs::gguf_import {
namespace {

constexpr std::string_view kArch = "qwen35";
const std::string kPrefix = "model.";  // a text-only checkpoint's decoder, as Qwen3_5ForCausalLM names it

float minus_one(float x) {
    return x - 1.0f;
}
float log_of_negative(float x) {
    return std::log(-x);
}
const FloatMap kNorm{"w = (1 + w) - 1 of a norm weight", minus_one};
const FloatMap kALog{"A_log = log(-ssm_a)", log_of_negative};

// The qwen35.* keys the import reads, or knows to change nothing it writes.
const std::set<std::string_view> kKnownKeys = {
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
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
    "nextn_predict_layers",
};

struct Dimensions {
    int64_t vocab, hidden, ffn, heads, kv_heads, head_dim;
    int64_t key_heads, value_heads, key_dim, value_dim, conv_kernel;
    int64_t key_width() const { return key_heads * key_dim; }
    int64_t value_width() const { return value_heads * value_dim; }
    int64_t conv_width() const { return 2 * key_width() + value_width(); }
};

// A decoder layer of either kind: the norms and the MLP.
void add_layer_common(Entries& e, const Dimensions& d, const std::string& L) {
    e["attn_norm.weight"] = one({d.hidden}, L + "input_layernorm.weight", &kNorm);
    e["post_attention_norm.weight"] = one({d.hidden}, L + "post_attention_layernorm.weight", &kNorm);
    e["ffn_gate.weight"] = one({d.ffn, d.hidden}, L + "mlp.gate_proj.weight");
    e["ffn_up.weight"] = one({d.ffn, d.hidden}, L + "mlp.up_proj.weight");
    e["ffn_down.weight"] = one({d.hidden, d.ffn}, L + "mlp.down_proj.weight");
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

// Gated DeltaNet; in_proj_qkv / in_proj_z and in_proj_b / in_proj_a stored apart or as one tensor each.
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
    e["attn_qkv_gate.weight"] = Entry{{d.conv_width() + d.value_width(), d.hidden}, {qkv, z}};
    e["ssm_beta.weight"] = Entry{b.shape, {b}};
    e["ssm_alpha.weight"] = Entry{a.shape, {a}};
    e["ssm_beta_alpha.weight"] = Entry{{2 * d.value_heads, d.hidden}, {b, a}};
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

class Qwen35 final : public Architecture {
public:
    std::string_view name() const override { return kArch; }

    ImportPlan plan(const gguf::File& f) const override { return plan_from(f, nullptr); }
    ImportPlan plan_with_mtp(const gguf::File& f, const gguf::File& mtp) const override { return plan_from(f, &mtp); }

private:
    // The plan of the model file `f`, with its MTP head from the file `head` when one is given.
    static ImportPlan plan_from(const gguf::File& f, const gguf::File* head) {
        refuse_unknown_keys(f, kArch, kKnownKeys);
        if (const gguf::Value* s = f.find("qwen35.rope.scaling.type"); s != nullptr && s->as_string() != "none") {
            fail("qwen35: qwen35.rope.scaling.type '", s->as_string(), "' is not supported (supported: none)");
        }

        const gguf::TensorInfo* embed = f.find_tensor("token_embd.weight");
        if (embed == nullptr) fail("qwen35: the file has no token_embd.weight");
        Dimensions d{};
        d.vocab = embed->rows();
        d.hidden = positive(f, kArch, "qwen35.embedding_length");
        d.ffn = positive(f, kArch, "qwen35.feed_forward_length");
        d.heads = positive(f, kArch, "qwen35.attention.head_count");
        d.kv_heads = positive(f, kArch, "qwen35.attention.head_count_kv");
        d.head_dim = f.find("qwen35.attention.key_length") ? positive(f, kArch, "qwen35.attention.key_length")
                                                           : d.hidden / d.heads;
        if (f.find("qwen35.attention.value_length") && f.get_int("qwen35.attention.value_length") != d.head_dim) {
            fail("qwen35: attention.value_length ", f.get_int("qwen35.attention.value_length"), " differs from ",
                 "key_length ", d.head_dim);
        }
        d.key_heads = positive(f, kArch, "qwen35.ssm.group_count");
        d.value_heads = positive(f, kArch, "qwen35.ssm.time_step_rank");
        d.key_dim = positive(f, kArch, "qwen35.ssm.state_size");
        d.conv_kernel = positive(f, kArch, "qwen35.ssm.conv_kernel");
        const int64_t inner = positive(f, kArch, "qwen35.ssm.inner_size");
        if (inner % d.value_heads != 0 || d.value_heads % d.key_heads != 0) {
            fail("qwen35: ssm.inner_size ", inner, ", time_step_rank ", d.value_heads, " and group_count ", d.key_heads,
                 " are not whole value heads per key head");
        }
        d.value_dim = inner / d.value_heads;
        if (const gguf::Value* v = f.find("qwen35.vocab_size"); v != nullptr && v->as_int() != d.vocab) {
            fail("qwen35: vocab_size ", v->as_int(), " differs from token_embd.weight's ", d.vocab, " rows");
        }
        if (f.get("tokenizer.ggml.tokens").size() > static_cast<uint64_t>(d.vocab)) {
            fail("qwen35: the tokenizer has ", f.get("tokenizer.ggml.tokens").size(), " tokens for ", d.vocab,
                 " embedding rows");
        }

        int64_t blocks = positive(f, kArch, "qwen35.block_count");
        int64_t mtp = f.find("qwen35.nextn_predict_layers") ? f.get_int("qwen35.nextn_predict_layers") : 0;
        if (mtp < 0 || mtp >= blocks) fail("qwen35: nextn_predict_layers ", mtp, " of ", blocks, " blocks");
        if (head != nullptr && mtp != 0) {
            fail("qwen35: --mtp: the model file already has an MTP head (nextn_predict_layers ", mtp, ")");
        }
        const int64_t layers = blocks - mtp;
        // Which layers are linear attention: the file's list, or every one but each full_attention_interval-th
        // (llama.cpp's reading, 4 when the file gives no interval either).
        std::vector<bool> linear(static_cast<size_t>(layers));
        if (const gguf::Value* r = f.find("qwen35.attention.recurrent_layers")) {
            const std::vector<bool> list = r->bools();
            if (static_cast<int64_t>(list.size()) != blocks) {
                fail("qwen35: attention.recurrent_layers has ", list.size(), " entries for ", blocks, " blocks");
            }
            for (int64_t i = 0; i < blocks; ++i) {
                if (i < layers) {
                    linear[static_cast<size_t>(i)] = list[static_cast<size_t>(i)];
                } else if (list[static_cast<size_t>(i)]) {
                    fail("qwen35: attention.recurrent_layers marks MTP block ", i, " recurrent");
                }
            }
        } else {
            const int64_t interval =
                f.find("qwen35.full_attention_interval") ? positive(f, kArch, "qwen35.full_attention_interval") : 4;
            for (int64_t i = 0; i < layers; ++i)
                linear[static_cast<size_t>(i)] = (i + 1) % interval != 0;
        }
        if (head != nullptr) {
            check_head_metadata(f, *head, blocks);
            blocks += 1;
            mtp = 1;
        }

        ImportPlan plan;
        plan.config = config(f, d, layers, mtp, linear);
        const Layout l = layout(d, layers, blocks, linear);
        std::vector<const gguf::TensorInfo*> tensors;
        for (const gguf::TensorInfo& t : f.tensors())
            tensors.push_back(&t);
        if (head != nullptr) add_head_tensors(f, *head, layers, l, tensors, plan.copies);
        map_tensors(tensors, kArch, l, plan);
        return plan;
    }

    // The MTP file `head` describes the model of `f` (a file of `blocks` blocks, none of them MTP layers) with one
    // more block, the head: the same architecture, and every other qwen35.* key the same in both (the per-block
    // attention.recurrent_layers: the model's blocks, then the head's, not recurrent).
    static void check_head_metadata(const gguf::File& f, const gguf::File& head, int64_t blocks) {
        if (const std::string_view a = head.get_string("general.architecture"); a != kArch) {
            fail("qwen35: the MTP file is general.architecture '", a, "', not qwen35");
        }
        if (const int64_t n = head.get_int("qwen35.nextn_predict_layers"); n != 1) {
            fail("qwen35: the MTP file has nextn_predict_layers ", n, "; an MTP file holds one MTP layer");
        }
        if (const int64_t n = head.get_int("qwen35.block_count"); n != blocks + 1) {
            fail("qwen35: the MTP file has block_count ", n, ", not the model file's ", blocks, " blocks and the head");
        }
        std::set<std::string_view> keys;
        for (const gguf::File* file : {&f, &head}) {
            for (const gguf::Value& v : file->metadata()) {
                if (v.key().starts_with("qwen35.")) keys.insert(v.key());
            }
        }
        for (const std::string_view key : keys) {
            if (key == "qwen35.block_count" || key == "qwen35.nextn_predict_layers") continue;
            const gguf::Value* ours = f.find(key);
            const gguf::Value* theirs = head.find(key);
            bool same = ours != nullptr && theirs != nullptr && ours->same_as(*theirs);
            if (key == "qwen35.attention.recurrent_layers" && ours != nullptr && theirs != nullptr) {
                std::vector<bool> list = ours->bools();
                list.push_back(false);
                same = theirs->bools() == list;
            }
            if (!same) {
                fail("qwen35: the MTP file's ", key, " (", shown(theirs), ") differs from the model file's (",
                     shown(ours), "): it is the head of another model");
            }
        }
    }

    // The tensors the MTP file `head` adds to the model's: its block `layers`, the head, through the model's layout
    // as a file that held it; and its own copies of the model's top-level tensors, checked against the model's shapes
    // and left out.
    static void add_head_tensors(const gguf::File& f, const gguf::File& head, int64_t layers, const Layout& l,
                                 std::vector<const gguf::TensorInfo*>& tensors,
                                 std::vector<const gguf::TensorInfo*>& copies) {
        const std::string block = str_cat("blk.", layers, ".");
        for (const gguf::TensorInfo& t : f.tensors()) {
            if (t.name.starts_with(block)) {
                fail("qwen35: tensor '", t.name, "' is not in one of the model file's ", layers, " blocks");
            }
        }
        for (const gguf::TensorInfo& t : head.tensors()) {
            if (t.name.starts_with(block)) {
                tensors.push_back(&t);
            } else if (const auto e = l.top.find(t.name); e != l.top.end()) {
                if (t.shape() != e->second.stored) {
                    fail("qwen35: the MTP file's '", t.name, "' has shape ", dims_to_string(t.shape()),
                         "; the model's is ", dims_to_string(e->second.stored));
                }
                copies.push_back(&t);
            } else {
                fail("qwen35: the MTP file's tensor '", t.name, "' is neither in the head's block ", layers,
                     " nor one of the model's token_embd, output and output_norm");
            }
        }
    }

    static nlohmann::ordered_json config(const gguf::File& f, const Dimensions& d, int64_t layers, int64_t mtp,
                                         const std::vector<bool>& linear) {
        nlohmann::ordered_json c;
        c["architectures"] = {"Qwen3_5ForCausalLM"};
        c["model_type"] = "qwen3_5_text";
        c["vocab_size"] = d.vocab;
        c["hidden_size"] = d.hidden;
        c["intermediate_size"] = d.ffn;
        c["num_hidden_layers"] = layers;
        nlohmann::ordered_json types = nlohmann::ordered_json::array();
        for (const bool l : linear)
            types.push_back(l ? "linear_attention" : "full_attention");
        c["layer_types"] = std::move(types);
        if (f.find("qwen35.full_attention_interval")) {
            c["full_attention_interval"] = f.get_int("qwen35.full_attention_interval");
        }
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
        c["rms_norm_eps"] = config_number(f.get("qwen35.attention.layer_norm_rms_epsilon"));
        c["max_position_embeddings"] = positive(f, kArch, "qwen35.context_length");
        const int64_t rotary =
            f.find("qwen35.rope.dimension_count") ? positive(f, kArch, "qwen35.rope.dimension_count") : d.head_dim;
        if (rotary > d.head_dim || rotary % 2 != 0) {
            fail("qwen35: rope.dimension_count ", rotary, " does not fit head_dim ", d.head_dim);
        }
        nlohmann::ordered_json rope;
        rope["rope_type"] = "default";
        rope["rope_theta"] = config_number(f.get("qwen35.rope.freq_base"));
        rope["partial_rotary_factor"] = static_cast<double>(rotary) / static_cast<double>(d.head_dim);
        if (const gguf::Value* s = f.find("qwen35.rope.dimension_sections")) {
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

    static Layout layout(const Dimensions& d, int64_t layers, int64_t blocks, const std::vector<bool>& linear) {
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
            return i >= layers                      ? "an MTP layer"
                   : linear[static_cast<size_t>(i)] ? "linear attention"
                                                    : "full attention";
        };
        return l;
    }
};

}  // namespace

const Architecture& qwen35() {
    static const Qwen35 arch;
    return arch;
}

}  // namespace xs::gguf_import
