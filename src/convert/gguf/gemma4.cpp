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
// - the vision and audio towers are not in the file (llama.cpp writes them to an mmproj file of its own): the pack is
//   the text model.
// - an assistant's centroid head (masked_embedding.*) is left out: llama.cpp's graph scores the whole vocabulary with
//   the tied embedding, which is the assistant without ordered embeddings (use_ordered_embeddings false).

#include <algorithm>
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
    e["attn_v.weight"] = Entry{v.shape, {v}, shared};
    e["attn_qkv.weight"] = Entry{{(d.heads + 2 * kv) * hd, h}, {q, k, v}, true};
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
    return e;
}

class Gemma4 final : public Architecture {
public:
    explicit Gemma4(bool assistant) : assistant_(assistant) {}

    std::string_view name() const override { return assistant_ ? "gemma4-assistant" : "gemma4"; }

    ImportPlan plan(const gguf::File& f) const override {
        for (const char* moe : {"expert_count", "expert_used_count", "expert_feed_forward_length"}) {
            if (f.find(key(moe)) != nullptr) {
                fail(name(), ": ", key(moe), ": mixture-of-experts Gemma 4 (26B-A4B) is not supported yet");
            }
        }
        std::set<std::string_view> known = kKnownKeys;
        if (assistant_) known.insert(kAssistantKeys.begin(), kAssistantKeys.end());
        refuse_unknown_keys(f, name(), known);
        const Dimensions d = dimensions(f);
        for (int64_t i = 0; i < d.first_shared(); ++i) {
            const std::string b = str_cat("blk.", i, ".");
            if (f.find_tensor(b + "attn_k.weight") != nullptr && f.find_tensor(b + "attn_v.weight") == nullptr) {
                fail(name(), ": block ", i, " has no attn_v.weight: values from the keys (attention_k_eq_v) are not ",
                     "supported yet");
            }
        }
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

private:
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
        return d;
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
        c["attention_k_eq_v"] = false;
        c["attention_bias"] = false;
        c["enable_moe_block"] = false;
        c["hidden_activation"] = "gelu_pytorch_tanh";
        c["sliding_window"] = positive(f, name(), key("attention.sliding_window"));
        c["num_kv_shared_layers"] = d.shared;
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
