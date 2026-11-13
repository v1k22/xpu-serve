// GGUF architecture "modern-bert" with decision.type "laya": the decision models on ModernBERT as llama.cpp's converter
// writes them (conversion/bert.py, ModernBertDecisionModel), imported for models/laya/ in their original checkpoints'
// layout: Laya's (convaiinnovations/laya: config.json, rl_agent_config.json) or Julia-1's (SupersonicLabs/Julia-1:
// config.json, julia_config.json, inference-policy.json), both with encoder/config.json and the tokenizer in
// tokenizer/.
//
// What the converter changed, and what the import does about it:
// - names: the encoder's layers are blk.0 to blk.<n - 1>, the two head layers the blocks after them (with biases),
//   token_types the question types' embedding, cls.norm, cls and cls.output the scorer; mapped back to the torch
//   names. No value is changed: matrices and norms are stored as the checkpoint holds them (a BF16 file rounds them).
// - left out: the action head (act_head.*) and the temperature buffer, which the answer does not read.
// - the two models are told apart by the prompt template the converter writes for each (tokenizer.chat_template.
//   systemone, from the file's own [CLS], [SEP] and mask tokens), and a file with any other is refused.
// - Laya's temperatures are metadata (decision.temperature.<type>[.<option count>]), read back exactly.
// - what the file does not carry comes from the published checkpoint it was made from: Laya's state budget (max_len,
//   which only goes with its question budget in the published variants: 192 with 512, the root model; 256 with
//   1024, typed-decisions and multilingual), and Julia-1's inference policy (8,192 tokens, a question budget of 512
//   and strict encoding; the file's max_head_tokens is the converter's default of 256, since julia_config.json names
//   none, and only bounds what strict encoding accepts).

#include <map>
#include <set>

#include "convert/gguf/architecture.h"
#include "convert/gguf/layout.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::gguf_import {
namespace {

const char* const kTypes[] = {"choice", "score", "noul"};
const char* const kBuckets[][2] = {{"2", "2"}, {"3_5", "3-5"}, {"6_10", "6-10"}, {"11", "11+"}};  // file, Laya's

std::set<std::string_view> known_keys() {
    static const std::vector<std::string> temperatures = [] {
        std::vector<std::string> keys;
        for (const char* type : kTypes) {
            keys.push_back(str_cat("decision.temperature.", type));
            for (const auto& bucket : kBuckets)
                keys.push_back(str_cat("decision.temperature.", type, ".", bucket[0]));
        }
        return keys;
    }();
    std::set<std::string_view> keys = {"block_count",
                                       "context_length",
                                       "embedding_length",
                                       "feed_forward_length",
                                       "vocab_size",
                                       "hidden_activation",
                                       "attention.head_count",
                                       "attention.causal",
                                       "attention.layer_norm_epsilon",
                                       "attention.layer_norm_rms_epsilon",
                                       "attention.sliding_window",
                                       "attention.sliding_window_pattern",
                                       "rope.freq_base",
                                       "rope.freq_base_swa",
                                       "rope.scaling.type",
                                       "classifier.pooling_type",
                                       "classifier.activation",
                                       "decision.type",
                                       "decision.block_count",
                                       "decision.max_head_tokens"};
    keys.insert(temperatures.begin(), temperatures.end());
    return keys;
}

// The template the converter writes (ModernBertDecisionModel._systemone_template) for Julia-1 or for Laya, around the
// file's own [CLS], [SEP] and mask tokens.
std::string systemone_template(bool julia, const std::string& cls, const std::string& sep, const std::string& mask) {
    const auto text_or_json = [](const std::string& name) {
        return "{{ " + name + " if " + name + " is string else " + name + " | tojson }}";
    };
    const std::string description = text_or_json("o.description");
    const std::string option =
        julia ? "{% if o.description %}" + description + "{% else %}{{ o.key }}{% endif %}"
              : "{% if type == 'choice' %}{{ o.key }}{% if o.description %}: " + description + "{% endif %}" +
                    "{% elif type == 'score' %}level {{ o.key }}: " + description +
                    "{% else %}{{ o.key }}: {% if o.description %}" + description +
                    "{% elif o.key == 'true' %}yes, the statement holds" +
                    "{% else %}no, the statement does not hold{% endif %}{% endif %}";
    return cls + "{{ type }} question: " + text_or_json("instructions") + sep + "{% for o in options %}" + mask + " " +
           option + "{% endfor %}" + sep + text_or_json("state") + sep;
}

struct Dimensions {
    int64_t vocab = 0, hidden = 0, heads = 0, ffn = 0, context = 0;
    int64_t layers = 0;  // the encoder's
    int64_t head_layers = 0;
    int64_t window = 0;   // local_attention
    int64_t pattern = 0;  // every pattern-th layer is global, the first among them
};

Entries encoder_layer(const Dimensions& d, int64_t i) {
    const int64_t h = d.hidden;
    const std::string L = "encoder.layers." + std::to_string(i) + ".";
    Entries e;
    if (i > 0) e["attn_norm.weight"] = one({h}, L + "attn_norm.weight");
    e["attn_qkv.weight"] = one({3 * h, h}, L + "attn.Wqkv.weight");
    e["attn_output.weight"] = one({h, h}, L + "attn.Wo.weight");
    e["ffn_norm.weight"] = one({h}, L + "mlp_norm.weight");
    e["ffn_up.weight"] = one({2 * d.ffn, h}, L + "mlp.Wi.weight");
    e["ffn_down.weight"] = one({h, d.ffn}, L + "mlp.Wo.weight");
    return e;
}

// nn.TransformerEncoderLayer j of the head: torch's names, biases included.
Entries head_layer(const Dimensions& d, int64_t j) {
    const int64_t h = d.hidden;
    const std::string L = "head.layers." + std::to_string(j) + ".";
    Entries e;
    for (const auto& [file, torch] : {std::pair{"attn_norm", "norm1"}, std::pair{"ffn_norm", "norm2"}}) {
        e[str_cat(file, ".weight")] = one({h}, L + torch + ".weight");
        e[str_cat(file, ".bias")] = one({h}, L + torch + ".bias");
    }
    e["attn_qkv.weight"] = one({3 * h, h}, L + "self_attn.in_proj_weight");
    e["attn_qkv.bias"] = one({3 * h}, L + "self_attn.in_proj_bias");
    e["attn_output.weight"] = one({h, h}, L + "self_attn.out_proj.weight");
    e["attn_output.bias"] = one({h}, L + "self_attn.out_proj.bias");
    e["ffn_up.weight"] = one({4 * h, h}, L + "linear1.weight");
    e["ffn_up.bias"] = one({4 * h}, L + "linear1.bias");
    e["ffn_down.weight"] = one({h, 4 * h}, L + "linear2.weight");
    e["ffn_down.bias"] = one({h}, L + "linear2.bias");
    return e;
}

class ModernBert final : public Architecture {
public:
    std::string_view name() const override { return "modern-bert"; }

    ImportPlan plan(const gguf::File& f) const override {
        const gguf::Value* type = f.find(key("decision.type"));
        const std::string decision = type != nullptr ? std::string(type->as_string()) : "";
        if (decision != "laya") {
            fail(name(), ": ", decision.empty() ? "a file without decision.type (an encoder or embedding model)"
                                                : str_cat("decision.type '", decision, "'"),
                 " is not supported (decision.type 'laya' is: Laya, Julia-1)");
        }
        refuse_unknown_keys(f, name(), known_keys());
        const Dimensions d = dimensions(f);
        const bool julia = identify(f);

        ImportPlan plan;
        plan.kind = ImportPlan::Kind::Decision;
        plan.tokenizer_dir = "tokenizer";
        plan.files["encoder/config.json"] = encoder_config(f, d);
        if (julia) {
            julia_files(f, d, plan);
        } else {
            laya_files(f, d, plan);
        }
        map_tensors(f, name(), layout(d), plan);
        return plan;
    }

private:
    std::string key(std::string_view rest) const { return str_cat(name(), ".", rest); }

    Dimensions dimensions(const gguf::File& f) const {
        Dimensions d;
        const int64_t blocks = positive(f, name(), key("block_count"));
        d.head_layers = positive(f, name(), key("decision.block_count"));
        d.layers = blocks - d.head_layers;
        XS_CHECK(d.layers > 0, name(), ": ", key("decision.block_count"), " ", d.head_layers,
                 " leaves no encoder layer");
        d.vocab = positive(f, name(), key("vocab_size"));
        d.hidden = positive(f, name(), key("embedding_length"));
        d.heads = positive(f, name(), key("attention.head_count"));
        d.context = positive(f, name(), key("context_length"));
        d.window = positive(f, name(), key("attention.sliding_window"));
        d.pattern = positive(f, name(), key("attention.sliding_window_pattern"));
        const std::vector<int64_t> ffn = f.get(key("feed_forward_length")).ints();
        XS_CHECK(static_cast<int64_t>(ffn.size()) == blocks, name(), ": ", key("feed_forward_length"), " has ",
                 ffn.size(), " entries for ", blocks, " blocks");
        d.ffn = ffn.front();
        for (int64_t i = 0; i < blocks; ++i) {
            const int64_t want = i < d.layers ? d.ffn : 4 * d.hidden;  // the head's MLP is 4 x its width
            if (ffn[static_cast<size_t>(i)] != want) {
                fail(name(), ": block ", i, " has a feed-forward width of ", ffn[static_cast<size_t>(i)], ", ",
                     i < d.layers ? "the encoder's first layer's " : "the decision head's (4 x its width) ", want);
            }
        }
        if (f.get(key("attention.causal")).as_bool()) fail(name(), ": causal attention is not ModernBERT's");
        for (const char* act : {"hidden_activation", "classifier.activation"}) {
            if (const gguf::Value* v = f.find(key(act)); v != nullptr && v->as_string() != "gelu") {
                fail(name(), ": ", key(act), " '", v->as_string(), "' is not supported (gelu is)");
            }
        }
        if (const gguf::Value* v = f.find(key("rope.scaling.type")); v != nullptr && v->as_string() != "none") {
            fail(name(), ": ", key("rope.scaling.type"), " '", v->as_string(), "' is not supported (none is)");
        }
        if (const gguf::Value* v = f.find("tokenizer.ggml.token_type_count"); v == nullptr || v->as_int() != 3) {
            fail(name(), ": tokenizer.ggml.token_type_count must be 3, one token type per question type");
        }
        return d;
    }

    // Julia-1 (true) or Laya (false), by the prompt template the converter wrote; any other file is refused.
    bool identify(const gguf::File& f) const {
        const gguf::Value* t = f.find("tokenizer.chat_template.systemone");
        if (t == nullptr) fail(name(), ": the file has no tokenizer.chat_template.systemone (the decision prompt)");
        const std::vector<std::string_view> tokens = f.get("tokenizer.ggml.tokens").strings();
        const auto text = [&](const char* key_name) {
            const int64_t id = f.get_int(str_cat("tokenizer.ggml.", key_name, "_token_id"));
            XS_CHECK(id >= 0 && id < static_cast<int64_t>(tokens.size()), name(), ": tokenizer.ggml.", key_name,
                     "_token_id ", id, " is not a token");
            return std::string(tokens[static_cast<size_t>(id)]);
        };
        const std::string cls = text("bos"), sep = text("seperator"), mask = text("mask");
        for (const bool julia : {false, true}) {
            if (t->as_string() == systemone_template(julia, cls, sep, mask)) return julia;
        }
        fail(name(), ": the decision prompt (tokenizer.chat_template.systemone) is neither Laya's nor Julia-1's as ",
             "llama.cpp's converter writes them; the file is refused");
    }

    nlohmann::ordered_json encoder_config(const gguf::File& f, const Dimensions& d) const {
        nlohmann::ordered_json types = nlohmann::ordered_json::array();
        for (int64_t i = 0; i < d.layers; ++i)
            types.push_back(i % d.pattern == 0 ? "full_attention" : "sliding_attention");
        nlohmann::ordered_json c;
        c["architectures"] = {"ModernBertModel"};
        c["model_type"] = "modernbert";
        c["vocab_size"] = d.vocab;
        c["hidden_size"] = d.hidden;
        c["intermediate_size"] = d.ffn;
        c["num_hidden_layers"] = d.layers;
        c["num_attention_heads"] = d.heads;
        c["max_position_embeddings"] = d.context;
        c["hidden_activation"] = "gelu";
        c["norm_eps"] = config_number(f.get(key("attention.layer_norm_epsilon")));
        c["norm_bias"] = false;
        c["attention_bias"] = false;
        c["mlp_bias"] = false;
        c["local_attention"] = d.window;
        c["global_attn_every_n_layers"] = d.pattern;
        c["layer_types"] = std::move(types);
        c["rope_parameters"] = {
            {"full_attention", {{"rope_theta", config_number(f.get(key("rope.freq_base")))}, {"rope_type", "default"}}},
            {"sliding_attention",
             {{"rope_theta", config_number(f.get(key("rope.freq_base_swa")))}, {"rope_type", "default"}}}};
        return c;
    }

    void laya_files(const gguf::File& f, const Dimensions& d, ImportPlan& plan) const {
        const int64_t head = positive(f, name(), key("decision.max_head_tokens"));
        // rl_agent_config.json's max_len of the published checkpoint with this head_max_len.
        const std::map<int64_t, int64_t> max_len = {{192, 512}, {256, 1024}};
        const auto it = max_len.find(head);
        if (it == max_len.end()) {
            fail(name(), ": a Laya file with decision.max_head_tokens ", head, ": the file does not hold the state's ",
                 "budget (max_len), and no published Laya pairs one with that question budget (192: 512, 256: 1024)");
        }
        // The temperatures exactly as the checkpoint's float32 values (rl_agent_config.json writes them in full).
        nlohmann::ordered_json temperature = nlohmann::ordered_json::array();
        nlohmann::ordered_json by_options = nlohmann::ordered_json::object();
        for (const char* type : kTypes) {
            const gguf::Value* v = f.find(key(str_cat("decision.temperature.", type)));
            temperature.push_back(v != nullptr ? v->as_float() : 1.0);  // system_one's default
            for (const auto& bucket : kBuckets) {
                if (const gguf::Value* b = f.find(key(str_cat("decision.temperature.", type, ".", bucket[0])))) {
                    by_options[str_cat(type, ":", bucket[1])] = b->as_float();
                }
            }
        }
        plan.config["model_type"] = "laya";
        plan.config["architectures"] = {"LayaTypedDecisions"};
        nlohmann::ordered_json agent;
        agent["head_layers"] = d.head_layers;
        agent["max_len"] = it->second;
        agent["head_max_len"] = head;
        agent["temperature"] = std::move(temperature);
        agent["temperature_by_options"] = std::move(by_options);
        plan.files["rl_agent_config.json"] = std::move(agent);
        log::info(name(), ": Laya, a state budget of ", it->second, " tokens (its published checkpoint's with a ",
                  "question budget of ", head, ")");
    }

    void julia_files(const gguf::File& f, const Dimensions& d, ImportPlan& plan) const {
        for (const gguf::Value& v : f.metadata()) {
            if (v.key().starts_with(key("decision.temperature."))) {
                fail(name(), ": ", v.key(), ": Julia-1 has no temperature (a plain softmax)");
            }
        }
        if (const int64_t head = positive(f, name(), key("decision.max_head_tokens")); head != 256) {
            fail(name(), ": a Julia-1 file with decision.max_head_tokens ", head, " (llama.cpp's converter writes 256 ",
                 "for it, its default)");
        }
        plan.config["format_version"] = 1;
        plan.config["architecture"] = "JuliaDecisionModel";
        plan.config["julia_config_file"] = "julia_config.json";
        plan.config["encoder_config_file"] = "encoder/config.json";
        plan.config["weights_file"] = "model.safetensors";
        plan.config["tokenizer_directory"] = "tokenizer";
        plan.files["julia_config.json"] = {{"format_version", 1},
                                           {"architecture", "JuliaDecisionModel"},
                                           {"head_layers", d.head_layers},
                                           {"n_act", 2},
                                           {"dropout", 0.1}};
        plan.files["inference-policy.json"] = {
            {"max_length", d.context}, {"head_length", 512}, {"strict_encoding", true}, {"calibration", nullptr}};
    }

    Layout layout(const Dimensions& d) const {
        const int64_t h = d.hidden;
        Layout l;
        l.top["token_embd.weight"] = one({d.vocab, h}, "encoder.embeddings.tok_embeddings.weight");
        l.top["token_embd_norm.weight"] = one({h}, "encoder.embeddings.norm.weight");
        l.top["output_norm.weight"] = one({h}, "encoder.final_norm.weight");
        l.top["token_types.weight"] = one({3, h}, "type_emb.weight");
        l.top["cls.norm.weight"] = one({h}, "scorer.0.weight");
        l.top["cls.norm.bias"] = one({h}, "scorer.0.bias");
        l.top["cls.weight"] = one({h, h}, "scorer.1.weight");
        l.top["cls.bias"] = one({h}, "scorer.1.bias");
        l.top["cls.output.weight"] = one({1, h}, "scorer.3.weight");  // [h] once llama-quantize drops the 1
        l.top["cls.output.bias"] = one({1}, "scorer.3.bias");
        for (int64_t i = 0; i < d.layers; ++i)
            l.blocks.push_back(encoder_layer(d, i));
        for (int64_t j = 0; j < d.head_layers; ++j)
            l.blocks.push_back(head_layer(d, j));
        l.block_kind = [layers = d.layers](int64_t i) {
            return i < layers ? std::string("an encoder layer") : std::string("a layer of the decision head");
        };
        return l;
    }
};

}  // namespace

const Architecture& modern_bert() {
    static const ModernBert a;
    return a;
}

}  // namespace xs::gguf_import
