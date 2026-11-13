// GGUF architecture "clef": Clef decision models (a Qwen3.5-family decoder with a joint schema head) as llama.cpp's
// converter writes them (conversion/clef.py, ClefModel), imported for models/clef/ as the pack a Clef checkpoint
// converts to: the decoder, the head beside it under joint_head., and joint_head_config.json.
//
// What the converter changed, and what the import does about it:
// - the decoder is a qwen35 file's, its keys under clef.* and without an MTP head: read as a qwen35 file is
//   (read_qwen35_decoder), the same transforms undone.
// - the head's attention and MLP blocks are dec.blk.<i>, the evidence layers first (decision.routing_block_count),
//   then the decoder layers (decision.block_count); its other tensors are decision.* and token_types: mapped back to
//   the checkpoint's names.
// - torch keeps an attention's query, key and value projections as one in_proj_weight / in_proj_bias; the converter
//   stores them as three tensors (_q, _k, _v): kept apart, under the names the head's programs read (q_proj, k_proj,
//   v_proj), as a pack of the checkpoint holds them.
// - the three learned scalars are stored as the values the head multiplies by (decision.scales: exp(min(x, ln 100))
//   of prior_logit_scale and of joint_logit_scale, the sigmoid of residual_gate): kept as stored, as joint_head.scales,
//   which the model reads in their place (computing the parameters back would round them again).
// - the head's width and its MLP's are in no key: read from decision.proj_memory and dec.blk.0.ffn_up.
// - the head's LayerNorm eps (clef.attention.layer_norm_epsilon) must be torch's default, which the model uses.
// - tokenizer.chat_template.systemone is llama.cpp's own writing of the prompt for its server: not imported (the model
//   code writes Clef's prompt from the reference, joint_schema_model.py).
// - the vision tower is in llama.cpp's mmproj file, which is not read: the model refuses images.

#include "convert/gguf/qwen35.h"
#include "util/error.h"

namespace xs::gguf_import {
namespace {

constexpr std::string_view kArch = "clef";
const std::string kHead = "joint_head.";  // where the model reads the head's weights

// The clef.* keys beside the decoder's.
const std::set<std::string_view> kHeadKeys = {
    "decision.type",       "decision.routing_block_count", "decision.block_count",
    "decision.head_count", "attention.layer_norm_epsilon",
};

// joint_head_config.json.
struct Head {
    int64_t hidden = 0, width = 0, routing = 0, layers = 0, heads = 0, ffn = 0;
};

// Rows of the 2-D tensor `name`, a width no key gives.
int64_t rows_of(const gguf::File& f, const std::string& name) {
    const gguf::TensorInfo* t = f.find_tensor(name);
    if (t == nullptr) fail(kArch, ": the file has no ", name, ", which gives the decision head's width");
    return t->rows();
}

// The head's tensors, by GGUF name.
void add_head(Entries& top, const Head& h) {
    const int64_t w = h.width;
    const auto norm = [&](const std::string& from, const std::string& to, int64_t n) {
        top[from + ".weight"] = one({n}, kHead + to + ".weight");
        top[from + ".bias"] = one({n}, kHead + to + ".bias");
    };
    const auto linear = [&](const std::string& from, const std::string& to, int64_t out, int64_t in) {
        top[from + ".weight"] = one({out, in}, kHead + to + ".weight");
        top[from + ".bias"] = one({out}, kHead + to + ".bias");
    };
    // `kind` "attn" (self-attention) or "cross_attn" of block `b`, as torch.nn.MultiheadAttention `to`.
    const auto attention = [&](const std::string& b, const std::string& kind, const std::string& to) {
        for (const char* x : {"q", "k", "v"})
            linear(b + kind + "_" + x, to + "." + x + "_proj", w, w);
        linear(b + kind + "_o", to + ".out_proj", w, w);
    };
    const auto ffn = [&](const std::string& b, const std::string& up, const std::string& down) {
        linear(b + "ffn_up", up, h.ffn, w);
        linear(b + "ffn_down", down, w, h.ffn);
    };
    for (int64_t i = 0; i < h.routing; ++i) {
        const std::string b = str_cat("dec.blk.", i, "."), l = str_cat("evidence_layers.", i);
        norm(b + "cross_attn_norm", l + ".query_norm", w);
        norm(b + "cross_attn_norm_kv", l + ".memory_norm", w);
        attention(b, "cross_attn", l + ".attention");
        norm(b + "ffn_norm", l + ".feedforward_norm", w);
        ffn(b, l + ".feedforward.0", l + ".feedforward.3");
    }
    for (int64_t j = 0; j < h.layers; ++j) {
        const std::string b = str_cat("dec.blk.", h.routing + j, "."), l = str_cat("layers.", j);
        norm(b + "attn_norm", l + ".norm1", w);
        attention(b, "attn", l + ".self_attn");
        norm(b + "cross_attn_norm", l + ".norm2", w);
        attention(b, "cross_attn", l + ".multihead_attn");
        norm(b + "ffn_norm", l + ".norm3", w);
        ffn(b, l + ".linear1", l + ".linear2");
    }
    norm("decision.hidden_norm", "hidden_norm", h.hidden);
    for (const char* n : {"option_summary_norm", "field_norm", "option_norm"})
        norm(str_cat("decision.", n), n, w);
    for (const auto& [from, to] :
         {std::pair{"memory", "memory_projection"}, std::pair{"question", "question_projection"},
          std::pair{"option_question", "option_question_projection"}, std::pair{"global", "global_projection"},
          std::pair{"option_context", "option_context_projection"},
          std::pair{"option_lexical", "option_lexical_projection"}}) {
        top[str_cat("decision.proj_", from, ".weight")] = one({w, h.hidden}, kHead + to + ".weight");
    }
    linear("decision.scorer", "residual_scorer.0", w, 4 * w);
    linear("decision.scorer_out", "residual_scorer.3", 1, w);
    top["decision.scales"] = one({3}, kHead + "scales");
    top["token_types.weight"] = one({3, w}, kHead + "type_embedding.weight");  // noul, choice, score
}

class Clef final : public Architecture {
public:
    std::string_view name() const override { return kArch; }

    ImportPlan plan(const gguf::File& f) const override {
        const Qwen35Decoder decoder = read_qwen35_decoder(f, kArch, kHeadKeys);
        if (const std::string_view type = f.get_string("clef.decision.type"); type != "clef") {
            fail("clef: decision.type '", type, "' is not supported (supported: clef)");
        }
        if (const double eps = config_number(f.get("clef.attention.layer_norm_epsilon")); eps != 1e-5) {
            fail("clef: attention.layer_norm_epsilon ", eps, " is not supported: the head's LayerNorm uses 1e-05");
        }
        Head h;
        h.hidden = positive(f, kArch, "clef.embedding_length");
        h.routing = f.get_int("clef.decision.routing_block_count");
        h.layers = f.get_int("clef.decision.block_count");
        if (h.routing < 0 || h.layers < 0) {
            fail("clef: decision.routing_block_count ", h.routing, " and block_count ", h.layers);
        }
        h.heads = positive(f, kArch, "clef.decision.head_count");
        h.width = rows_of(f, "decision.proj_memory.weight");
        h.ffn = rows_of(f, "dec.blk.0.ffn_up.weight");
        if (h.width % h.heads != 0) {
            fail("clef: the decision head's width ", h.width, " is not a multiple of its ", h.heads, " heads");
        }

        ImportPlan plan;
        plan.config = decoder.config;
        plan.files["joint_head_config.json"] = {
            {"hidden_size", h.hidden}, {"width", h.width}, {"routing_layers", h.routing},
            {"layers", h.layers},      {"heads", h.heads}, {"feedforward", h.ffn}};
        plan.unused_templates["systemone"] =
            "llama.cpp's writing of the decision prompt for its server; models/clef writes Clef's prompt itself";
        Layout layout = decoder.layout;
        add_head(layout.top, h);
        map_tensors(decoder.tensors, kArch, layout, plan);
        return plan;
    }
};

}  // namespace

const Architecture& clef() {
    static const Clef arch;
    return arch;
}

}  // namespace xs::gguf_import
