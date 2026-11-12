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

#include <cmath>
#include <map>
#include <set>

#include "convert/gguf/architecture.h"
#include "util/error.h"

namespace xs::gguf_import {
namespace {

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

// What one tensor of the file becomes: its stored shape (row-major), and the pack tensors it holds, in row order.
struct Piece {
    std::string name;
    Dims shape;
    const FloatMap* map = nullptr;
};
struct Entry {
    Dims stored;
    std::vector<Piece> pieces;
    bool optional = false;
};
using Entries = std::map<std::string, Entry, std::less<>>;  // by the name after "blk.<i>." (or the whole name)

Entry one(Dims stored, std::string name, const FloatMap* map = nullptr) {
    Dims shape = stored;
    return Entry{std::move(stored), {Piece{std::move(name), std::move(shape), map}}};
}

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

int64_t positive(const gguf::File& f, const std::string& key) {
    const int64_t v = f.get_int(key);
    if (v <= 0) fail("qwen35: ", key, " is ", v);
    return v;
}

class Qwen35 final : public Architecture {
public:
    std::string_view name() const override { return "qwen35"; }

    ImportPlan plan(const gguf::File& f) const override {
        for (const gguf::Value& v : f.metadata()) {
            const std::string_view key = v.key();
            if (key.substr(0, 7) == "qwen35." && !kKnownKeys.contains(key.substr(7))) {
                fail("qwen35: metadata '", key, "' is not read by this importer; a key it does not know may change ",
                     "the model, so the file is refused");
            }
        }
        if (const gguf::Value* s = f.find("qwen35.rope.scaling.type"); s != nullptr && s->as_string() != "none") {
            fail("qwen35: qwen35.rope.scaling.type '", s->as_string(), "' is not supported (supported: none)");
        }

        const gguf::TensorInfo* embed = f.find_tensor("token_embd.weight");
        if (embed == nullptr) fail("qwen35: the file has no token_embd.weight");
        Dimensions d{};
        d.vocab = embed->rows();
        d.hidden = positive(f, "qwen35.embedding_length");
        d.ffn = positive(f, "qwen35.feed_forward_length");
        d.heads = positive(f, "qwen35.attention.head_count");
        d.kv_heads = positive(f, "qwen35.attention.head_count_kv");
        d.head_dim =
            f.find("qwen35.attention.key_length") ? positive(f, "qwen35.attention.key_length") : d.hidden / d.heads;
        if (f.find("qwen35.attention.value_length") && f.get_int("qwen35.attention.value_length") != d.head_dim) {
            fail("qwen35: attention.value_length ", f.get_int("qwen35.attention.value_length"), " differs from ",
                 "key_length ", d.head_dim);
        }
        d.key_heads = positive(f, "qwen35.ssm.group_count");
        d.value_heads = positive(f, "qwen35.ssm.time_step_rank");
        d.key_dim = positive(f, "qwen35.ssm.state_size");
        d.conv_kernel = positive(f, "qwen35.ssm.conv_kernel");
        const int64_t inner = positive(f, "qwen35.ssm.inner_size");
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

        const int64_t blocks = positive(f, "qwen35.block_count");
        const int64_t mtp = f.find("qwen35.nextn_predict_layers") ? f.get_int("qwen35.nextn_predict_layers") : 0;
        if (mtp < 0 || mtp >= blocks) fail("qwen35: nextn_predict_layers ", mtp, " of ", blocks, " blocks");
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
                f.find("qwen35.full_attention_interval") ? positive(f, "qwen35.full_attention_interval") : 4;
            for (int64_t i = 0; i < layers; ++i)
                linear[static_cast<size_t>(i)] = (i + 1) % interval != 0;
        }

        ImportPlan plan;
        plan.config = config(f, d, layers, mtp, linear);
        plan.tensors = tensors(f, d, layers, blocks, linear);
        return plan;
    }

private:
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
        c["max_position_embeddings"] = positive(f, "qwen35.context_length");
        const int64_t rotary =
            f.find("qwen35.rope.dimension_count") ? positive(f, "qwen35.rope.dimension_count") : d.head_dim;
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

    static std::vector<PackTensor> tensors(const gguf::File& f, const Dimensions& d, int64_t layers, int64_t blocks,
                                           const std::vector<bool>& linear) {
        Entries top;
        top["token_embd.weight"] = one({d.vocab, d.hidden}, kPrefix + "embed_tokens.weight");
        top["output_norm.weight"] = one({d.hidden}, kPrefix + "norm.weight", &kNorm);
        top["output.weight"] = one({d.vocab, d.hidden}, "lm_head.weight");
        top["output.weight"].optional = true;
        std::vector<Entries> block(static_cast<size_t>(blocks));
        for (int64_t i = 0; i < blocks; ++i) {
            const std::string L = kPrefix + "layers." + std::to_string(i) + ".";
            block[static_cast<size_t>(i)] = i >= layers                      ? mtp_layer(d, i - layers)
                                            : linear[static_cast<size_t>(i)] ? linear_attention_layer(d, L)
                                                                             : full_attention_layer(d, L);
        }
        const auto kind = [&](int64_t i) {
            return i >= layers                      ? "an MTP layer"
                   : linear[static_cast<size_t>(i)] ? "linear attention"
                                                    : "full attention";
        };

        std::vector<PackTensor> out;
        std::map<std::string, int, std::less<>> made;  // pack tensor name -> how many tensors of the file made it
        for (const gguf::TensorInfo& t : f.tensors()) {
            const Entry* e = nullptr;
            if (t.name.rfind("blk.", 0) == 0) {
                const size_t dot = t.name.find('.', 4);
                const std::string index = t.name.substr(4, dot == std::string::npos ? std::string::npos : dot - 4);
                if (dot == std::string::npos || index.empty() || index.size() > 9 ||
                    index.find_first_not_of("0123456789") != std::string::npos || std::stoll(index) >= blocks) {
                    fail("qwen35: tensor '", t.name, "' is not in one of the file's ", blocks, " blocks");
                }
                const int64_t i = std::stoll(index);
                const Entries& entries = block[static_cast<size_t>(i)];
                const auto it = entries.find(std::string_view(t.name).substr(dot + 1));
                if (it == entries.end()) {
                    fail("qwen35: tensor '", t.name, "' has no counterpart in block ", i, " (", kind(i), ")");
                }
                e = &it->second;
            } else {
                const auto it = top.find(t.name);
                if (it == top.end()) fail("qwen35: tensor '", t.name, "' has no counterpart in the model");
                e = &it->second;
            }
            if (t.shape() != e->stored) {
                fail("qwen35: tensor '", t.name, "' has shape ", dims_to_string(t.shape()), "; the metadata gives ",
                     dims_to_string(e->stored));
            }
            int64_t row = 0;
            for (const Piece& p : e->pieces) {
                out.push_back(PackTensor{p.name, p.shape, &t, row, p.map});
                row += numel(p.shape) / p.shape.back();
                ++made[p.name];
            }
        }

        // Every tensor the model reads, made once (not both from one tensor and from its two halves).
        const auto check = [&](const Entries& entries, const std::string& where) {
            for (const auto& [suffix, e] : entries) {
                for (const Piece& p : e.pieces) {
                    const auto it = made.find(p.name);
                    const int n = it == made.end() ? 0 : it->second;
                    if (n == 0 && !e.optional) fail("qwen35: the file has no tensor for '", p.name, "' (", where, ")");
                    if (n > 1) fail("qwen35: '", p.name, "' comes from more than one tensor of the file (", where, ")");
                }
            }
        };
        check(top, "the model's");
        for (int64_t i = 0; i < blocks; ++i)
            check(block[static_cast<size_t>(i)], str_cat("block ", i, ", ", kind(i)));
        return out;
    }
};

}  // namespace

const Architecture& qwen35() {
    static const Qwen35 arch;
    return arch;
}

}  // namespace xs::gguf_import
