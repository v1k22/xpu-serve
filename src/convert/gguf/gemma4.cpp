// GGUF architecture "gemma4": Gemma 4's text decoder as llama.cpp's converter writes it (conversion/gemma.py,
// Gemma4Model), imported for models/gemma4/ as a text-only checkpoint (Gemma4ForCausalLM: names under model.).
//
// What the converter changed, and what the import does about it:
// - names: blk.<i>.* for layer i, token_embd, per_layer_* and output_norm; mapped back to the Hugging Face names.
// - no value is changed: Gemma 4's norms multiply by w (not 1 + w, as Gemma 3's), and they, the scales and every
//   matrix are stored as the checkpoint holds them, rows and columns in its order.
// - it can store q_proj, k_proj and v_proj as one tensor (attn_qkv): cut back into their runs of rows.
// - rope_freqs.weight is not the checkpoint's: llama.cpp's proportional RoPE for the full-attention layers, a factor
//   of 1 for each rotated pair and 1e30 (no rotation) for the rest. It is checked and read as config.json's
//   partial_rotary_factor, and not imported.
// - the vision and audio towers are not in the file (llama.cpp writes them to an mmproj file of its own): the pack is
//   the text model.

#include <algorithm>
#include <set>

#include "convert/gguf/architecture.h"
#include "convert/gguf/layout.h"
#include "util/error.h"

namespace xs::gguf_import {
namespace {

constexpr std::string_view kArch = "gemma4";
const std::string kPrefix = "model.";  // a text-only checkpoint's decoder, as Gemma4ForCausalLM names it

// The gemma4.* keys the import reads.
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

std::string key(std::string_view rest) {
    return str_cat(kArch, ".", rest);
}

// A positive per-layer value, written as one number or as an array of one per layer.
std::vector<int64_t> positive_per_layer(const gguf::File& f, std::string_view rest, int64_t layers) {
    const gguf::Value& v = f.get(key(rest));
    std::vector<int64_t> out =
        v.type() == gguf::ValueType::Array ? v.ints() : std::vector<int64_t>(static_cast<size_t>(layers), v.as_int());
    if (static_cast<int64_t>(out.size()) != layers) {
        fail(kArch, ": ", key(rest), " has ", out.size(), " entries for ", layers, " layers");
    }
    for (const int64_t x : out) {
        if (x <= 0) fail(kArch, ": ", key(rest), " holds ", x);
    }
    return out;
}

struct Dimensions {
    int64_t vocab = 0, hidden = 0, layers = 0, heads = 0;
    int64_t ffn = 0;                            // the MLP's width; a KV-shared layer's is twice it with double_wide
    bool double_wide = false;                   // use_double_wide_mlp
    int64_t head_dim = 0, global_head_dim = 0;  // sliding and full-attention layers
    int64_t kv_heads = 0, global_kv_heads = 0;
    int64_t ple = 0;  // hidden_size_per_layer_input; 0: no per-layer embeddings
    int64_t ple_vocab = 0;
    int64_t shared = 0;         // num_kv_shared_layers: the last ones read an earlier layer's KV
    int64_t rotated_pairs = 0;  // of a full-attention head (rope_freqs); global_head_dim / 2 without the tensor
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
    std::string_view name() const override { return kArch; }

    ImportPlan plan(const gguf::File& f) const override {
        for (const char* moe : {"expert_count", "expert_used_count", "expert_feed_forward_length"}) {
            if (f.find(key(moe)) != nullptr) {
                fail(kArch, ": ", key(moe), ": mixture-of-experts Gemma 4 (26B-A4B) is not supported yet");
            }
        }
        refuse_unknown_keys(f, kArch, kKnownKeys);
        const Dimensions d = dimensions(f);
        for (int64_t i = 0; i < d.first_shared(); ++i) {
            const std::string b = str_cat("blk.", i, ".");
            if (f.find_tensor(b + "attn_k.weight") != nullptr && f.find_tensor(b + "attn_v.weight") == nullptr) {
                fail(kArch, ": block ", i, " has no attn_v.weight: values from the keys (attention_k_eq_v) are not ",
                     "supported yet");
            }
        }
        ImportPlan plan;
        plan.config = config(f, d);
        map_tensors(f, kArch, layout(d), plan);
        return plan;
    }

private:
    static Dimensions dimensions(const gguf::File& f) {
        Dimensions d;
        const gguf::TensorInfo* embed = f.find_tensor("token_embd.weight");
        if (embed == nullptr) fail(kArch, ": the file has no token_embd.weight");
        d.vocab = embed->rows();
        if (f.get("tokenizer.ggml.tokens").size() > static_cast<uint64_t>(d.vocab)) {
            fail(kArch, ": the tokenizer has ", f.get("tokenizer.ggml.tokens").size(), " tokens for ", d.vocab,
                 " embedding rows");
        }
        d.hidden = positive(f, kArch, key("embedding_length"));
        d.layers = positive(f, kArch, key("block_count"));
        const std::vector<bool> pattern = f.get(key("attention.sliding_window_pattern")).bools();
        if (static_cast<int64_t>(pattern.size()) != d.layers) {
            fail(kArch, ": attention.sliding_window_pattern has ", pattern.size(), " entries for ", d.layers,
                 " layers");
        }
        d.sliding = pattern;

        const std::vector<int64_t> heads = positive_per_layer(f, "attention.head_count", d.layers);
        if (std::count(heads.begin(), heads.end(), heads[0]) != d.layers) {
            fail(kArch, ": attention.head_count differs between layers, which Gemma 4 does not");
        }
        d.heads = heads[0];
        d.head_dim = positive(f, kArch, key("attention.key_length_swa"));
        d.global_head_dim = positive(f, kArch, key("attention.key_length"));
        for (const auto& [k, v] : {std::pair{"attention.key_length", "attention.value_length"},
                                   std::pair{"attention.key_length_swa", "attention.value_length_swa"}}) {
            if (f.find(key(v)) != nullptr && f.get_int(key(v)) != f.get_int(key(k))) {
                fail(kArch, ": ", key(v), " ", f.get_int(key(v)), " differs from ", key(k), " ", f.get_int(key(k)));
            }
        }
        // Each kind of layer has one count of KV heads: the file gives one, or one per layer.
        const std::vector<int64_t> kv = positive_per_layer(f, "attention.head_count_kv", d.layers);
        for (const bool sliding : {true, false}) {
            int64_t n = 0;
            for (int64_t i = 0; i < d.layers; ++i) {
                if (d.sliding[static_cast<size_t>(i)] != sliding) continue;
                if (n != 0 && kv[static_cast<size_t>(i)] != n) {
                    fail(kArch, ": attention.head_count_kv differs between ", sliding ? "sliding" : "full",
                         "-attention layers, which Gemma 4 does not");
                }
                n = kv[static_cast<size_t>(i)];
            }
            (sliding ? d.kv_heads : d.global_kv_heads) = n;
        }
        if (d.kv_heads == 0) d.kv_heads = d.global_kv_heads;
        if (d.global_kv_heads == 0) d.global_kv_heads = d.kv_heads;

        d.shared = f.find(key("attention.shared_kv_layers")) ? f.get_int(key("attention.shared_kv_layers")) : 0;
        if (d.shared < 0 || d.shared >= d.layers) {
            fail(kArch, ": attention.shared_kv_layers ", d.shared, " of ", d.layers, " layers");
        }
        if (d.shared > 0 && d.first_shared() < 2) {
            fail(kArch, ": attention.shared_kv_layers ", d.shared, " leaves ", d.first_shared(), " layer computing ",
                 "its own KV; llama.cpp needs two");
        }
        // llama.cpp's graph reads a shared layer's KV from layer first_shared - 2 (sliding) or - 1 (full), Gemma 4
        // from the last earlier layer of its kind: the file means the model only when the two agree.
        for (int64_t i = d.first_shared(); i < d.layers; ++i) {
            const int64_t llama = d.first_shared() - (d.sliding[static_cast<size_t>(i)] ? 2 : 1);
            if (kv_source(d, i) != llama) {
                fail(kArch, ": KV-shared layer ", i, " reads layer ", llama, "'s KV in llama.cpp but layer ",
                     kv_source(d, i), "'s in Gemma 4; a file whose layers are laid out so is not supported");
            }
        }

        // The MLP's width: one for the layers that compute their own KV, and the same or twice it for the shared
        // ones (use_double_wide_mlp).
        d.layer_ffn = positive_per_layer(f, "feed_forward_length", d.layers);
        d.ffn = d.layer_ffn[0];
        const int64_t tail = d.layer_ffn.back();
        d.double_wide = d.shared > 0 && tail == 2 * d.ffn;
        for (int64_t i = 0; i < d.layers; ++i) {
            const int64_t want = i >= d.first_shared() && d.double_wide ? 2 * d.ffn : d.ffn;
            if (d.layer_ffn[static_cast<size_t>(i)] != want) {
                fail(kArch, ": feed_forward_length of layer ", i, " is ", d.layer_ffn[static_cast<size_t>(i)],
                     "; Gemma 4 has one width, doubled on the KV-shared layers or not (", want, " here)");
            }
        }

        d.ple = f.get_int(key("embedding_length_per_layer_input"));
        if (d.ple < 0) fail(kArch, ": embedding_length_per_layer_input is ", d.ple);
        if (d.ple > 0) {
            const gguf::TensorInfo* table = f.find_tensor("per_layer_token_embd.weight");
            if (table == nullptr) {
                fail(kArch, ": the file has no per_layer_token_embd.weight for its per-layer embeddings (a file ",
                     "written without them, --no-ple, is not supported)");
            }
            d.ple_vocab = table->rows();
        }

        // RoPE: sliding layers rotate their whole head; full-attention layers rotate the pairs rope_freqs gives a
        // factor of 1 (the converter writes rope.dimension_count as the whole head and leaves the rest to the factors).
        if (f.get_int(key("rope.dimension_count")) != d.global_head_dim) {
            fail(kArch, ": rope.dimension_count ", f.get_int(key("rope.dimension_count")),
                 " is not the full-attention ", "head size ", d.global_head_dim);
        }
        if (f.find(key("rope.dimension_count_swa")) && f.get_int(key("rope.dimension_count_swa")) != d.head_dim) {
            fail(kArch, ": rope.dimension_count_swa ", f.get_int(key("rope.dimension_count_swa")),
                 " is not the sliding-attention head size ", d.head_dim, " (partial RoPE on sliding layers)");
        }
        d.rotated_pairs = d.global_head_dim / 2;
        if (const gguf::TensorInfo* freqs = f.find_tensor("rope_freqs.weight")) {
            d.rotated_pairs = rotated_pairs(*freqs, d.global_head_dim);
        }
        return d;
    }

    // The pairs of a full-attention head the converter's rope_freqs leaves rotating: a factor of 1 for each, then
    // 1e30 (a frequency of zero) for the rest.
    static int64_t rotated_pairs(const gguf::TensorInfo& t, int64_t head_dim) {
        if (t.dtype == DType::Block || t.numel() != head_dim / 2) {
            fail(kArch, ": rope_freqs.weight is ", t.type_name(), " ", dims_to_string(t.shape()), "; expected ",
                 head_dim / 2, " float factors");
        }
        const std::vector<float> factors = float_rows(t, 0, t.rows());
        int64_t rotated = 0;
        while (rotated < static_cast<int64_t>(factors.size()) && factors[static_cast<size_t>(rotated)] == 1.0f)
            ++rotated;
        for (size_t i = static_cast<size_t>(rotated); i < factors.size(); ++i) {
            if (factors[i] != static_cast<float>(1e30)) {
                fail(kArch, ": rope_freqs.weight holds ", factors[i], " at ", i, "; the import reads only the ",
                     "converter's proportional RoPE (factors of 1, then 1e30)");
            }
        }
        return rotated;
    }

    static nlohmann::ordered_json config(const gguf::File& f, const Dimensions& d) {
        nlohmann::ordered_json c;
        c["architectures"] = {"Gemma4ForCausalLM"};
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
        c["sliding_window"] = positive(f, kArch, key("attention.sliding_window"));
        c["num_kv_shared_layers"] = d.shared;
        c["hidden_size_per_layer_input"] = d.ple;
        if (d.ple > 0) c["vocab_size_per_layer_input"] = d.ple_vocab;
        c["rms_norm_eps"] = config_number(f.get(key("attention.layer_norm_rms_epsilon")));
        if (const gguf::Value* s = f.find(key("final_logit_softcapping"))) {
            c["final_logit_softcapping"] = config_number(*s);
        } else {
            c["final_logit_softcapping"] = nullptr;
        }
        c["max_position_embeddings"] = positive(f, kArch, key("context_length"));
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
        for (const auto& [k, name] : {std::pair{"tokenizer.ggml.bos_token_id", "bos_token_id"},
                                      std::pair{"tokenizer.ggml.padding_token_id", "pad_token_id"}}) {
            if (const gguf::Value* v = f.find(k)) c[name] = v->as_int();
        }
        return c;
    }

    static Layout layout(const Dimensions& d) {
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
        for (int64_t i = 0; i < d.layers; ++i)
            l.blocks.push_back(layer(d, i));
        l.block_kind = [d](int64_t i) {
            return str_cat(d.sliding[static_cast<size_t>(i)] ? "sliding" : "full", " attention",
                           i >= d.first_shared() ? ", KV-shared" : "");
        };
        return l;
    }
};

}  // namespace

const Architecture& gemma4() {
    static const Gemma4 arch;
    return arch;
}

}  // namespace xs::gguf_import
