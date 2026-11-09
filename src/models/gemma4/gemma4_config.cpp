#include "models/config_util.h"
#include "models/gemma4/gemma4.h"
#include "util/error.h"

namespace xs::gemma4 {
namespace {

RopeConfig parse_rope(const nlohmann::json& params, const char* kind, const std::string& where) {
    if (!params.contains(kind)) fail(where, ": rope_parameters has no '", kind, "' entry");
    const auto& p = params.at(kind);
    const std::string w = where + " rope_parameters." + kind;
    RopeConfig r;
    r.theta = cfg::get_float(p, "rope_theta", w);
    const std::string type = cfg::get_string(p, "rope_type", w);
    if (type == "proportional") {
        r.proportional = true;
        r.partial_rotary_factor = p.contains("partial_rotary_factor") ? cfg::get_float(p, "partial_rotary_factor", w) : 1.0;
        if (p.contains("factor") && cfg::get_float(p, "factor", w) != 1.0) fail(w, ": a scaling factor is not supported");
    } else if (type != "default") {
        fail(w, ": rope_type '", type, "' is not supported (supported: default, proportional)");
    }
    return r;
}

// `drafter`: an assistant's decoder, whose layers all read the target's KV.
void refuse_unsupported(const nlohmann::json& t, const std::string& where, bool drafter) {
    if (drafter && cfg::get_bool_or(t, "enable_moe_block", false, where)) {
        fail(where, ": mixture-of-experts layers in an assistant are not supported");
    }
    if (t.contains("use_bidirectional_attention") && !t.at("use_bidirectional_attention").is_null() &&
        t.at("use_bidirectional_attention") != "vision") {
        fail(where, ": use_bidirectional_attention '", t.at("use_bidirectional_attention").dump(),
             "' (embedding models) is not supported");
    }
    if (cfg::get_bool_or(t, "attention_bias", false, where)) fail(where, ": attention_bias is not supported");
    const std::string act = cfg::get_string(t, "hidden_activation", where);
    if (act != "gelu_pytorch_tanh") fail(where, ": hidden_activation '", act, "' is not supported");
}

}  // namespace

int64_t Config::layer_head_dim(int64_t layer) const {
    return layer_kinds.at(static_cast<size_t>(layer)) == AttnKind::Full ? global_head_dim : head_dim;
}

int64_t Config::layer_kv_heads(int64_t layer) const {
    return layer_kinds.at(static_cast<size_t>(layer)) == AttnKind::Full ? num_global_kv_heads : num_kv_heads;
}

int64_t Config::layer_intermediate_size(int64_t layer) const {
    return intermediate_size * (double_wide_mlp && is_kv_shared(layer) ? 2 : 1);
}

int64_t Config::kv_source_layer(int64_t layer) const {
    const AttnKind kind = layer_kinds.at(static_cast<size_t>(layer));
    for (int64_t i = first_kv_shared_layer() - 1; i >= 0; --i) {
        if (layer_kinds[static_cast<size_t>(i)] == kind) return i;
    }
    fail("gemma4: no earlier layer of the same attention kind for KV-shared layer ", layer);
}

namespace {

Config parse_text(const nlohmann::json& config_json, const nlohmann::json& t, const std::string& where, bool drafter) {
    refuse_unsupported(t, where, drafter);

    Config c;
    for (const nlohmann::json* j : {&t, &config_json}) {
        for (const char* key : {"dtype", "torch_dtype"}) {
            if (!j->contains(key) || !j->at(key).is_string()) continue;
            const std::string d = j->at(key).get<std::string>();
            c.checkpoint_dtype = d == "float32" ? DType::F32 : d == "float16" ? DType::F16 : DType::BF16;
        }
    }
    c.vocab_size = cfg::get_int(t, "vocab_size", where);
    c.hidden_size = cfg::get_int(t, "hidden_size", where);
    c.intermediate_size = cfg::get_int(t, "intermediate_size", where);
    c.num_layers = cfg::get_int(t, "num_hidden_layers", where);
    c.num_heads = cfg::get_int(t, "num_attention_heads", where);
    c.num_kv_heads = cfg::get_int(t, "num_key_value_heads", where);
    c.k_eq_v = cfg::get_bool_or(t, "attention_k_eq_v", false, where);
    c.num_global_kv_heads = t.contains("num_global_key_value_heads") && !t.at("num_global_key_value_heads").is_null()
                                ? cfg::get_int(t, "num_global_key_value_heads", where)
                                : c.num_kv_heads;
    c.bidirectional_vision = t.contains("use_bidirectional_attention") && t.at("use_bidirectional_attention") == "vision";
    if (cfg::get_bool_or(t, "enable_moe_block", false, where)) {
        c.num_experts = cfg::get_int(t, "num_experts", where);
        c.top_k_experts = cfg::get_int(t, "top_k_experts", where);
        c.moe_intermediate_size = cfg::get_int(t, "moe_intermediate_size", where);
        XS_CHECK(c.top_k_experts >= 1 && c.top_k_experts <= c.num_experts, where, ": top_k_experts ", c.top_k_experts,
                 " of ", c.num_experts, " experts");
    }
    c.head_dim = cfg::get_int(t, "head_dim", where);
    c.global_head_dim = cfg::get_int_or(t, "global_head_dim", 512, where);
    c.sliding_window = cfg::get_int(t, "sliding_window", where);
    c.num_kv_shared_layers = cfg::get_int_or(t, "num_kv_shared_layers", 0, where);
    c.ple_dim = cfg::get_int_or(t, "hidden_size_per_layer_input", 0, where);
    c.ple_vocab_size = cfg::get_int_or(t, "vocab_size_per_layer_input", c.vocab_size, where);
    c.double_wide_mlp = cfg::get_bool_or(t, "use_double_wide_mlp", false, where);
    c.rms_norm_eps = cfg::get_float(t, "rms_norm_eps", where);
    c.final_logit_softcap = t.contains("final_logit_softcapping") && !t.at("final_logit_softcapping").is_null()
                                ? cfg::get_float(t, "final_logit_softcapping", where)
                                : 0.0;
    c.max_position = cfg::get_int(t, "max_position_embeddings", where);

    if (!t.contains("layer_types") || !t.at("layer_types").is_array()) fail(where, ": missing layer_types");
    for (const auto& lt : t.at("layer_types")) {
        const std::string s = lt.get<std::string>();
        if (s == "sliding_attention") {
            c.layer_kinds.push_back(AttnKind::Sliding);
        } else if (s == "full_attention") {
            c.layer_kinds.push_back(AttnKind::Full);
        } else {
            fail(where, ": unknown layer type '", s, "'");
        }
    }
    XS_CHECK(static_cast<int64_t>(c.layer_kinds.size()) == c.num_layers, where, ": layer_types has ",
             c.layer_kinds.size(), " entries for ", c.num_layers, " layers");

    // Newer transformers write the full-attention layers' head size and KV heads as per-layer overrides.
    if (t.contains("per_layer_config") && t.at("per_layer_config").is_object()) {
        bool seen = false;
        for (const auto& [index, layer] : t.at("per_layer_config").items()) {
            const std::string w = where + " per_layer_config." + index;
            const int64_t i = std::stoll(index);
            XS_CHECK(i >= 0 && i < c.num_layers && c.layer_kinds[static_cast<size_t>(i)] == AttnKind::Full, w,
                     ": overrides are supported for full-attention layers only");
            for (const auto& [key, value] : layer.items()) {
                if (key != "head_dim" && key != "num_key_value_heads") fail(w, ": '", key, "' is not supported");
            }
            const int64_t hd = cfg::get_int_or(layer, "head_dim", c.global_head_dim, w);
            const int64_t kv = cfg::get_int_or(layer, "num_key_value_heads", c.num_global_kv_heads, w);
            XS_CHECK(!seen || (hd == c.global_head_dim && kv == c.num_global_kv_heads), w,
                     ": full-attention layers with different shapes are not supported");
            c.global_head_dim = hd;
            c.num_global_kv_heads = kv;
            seen = true;
        }
    }

    if (!t.contains("rope_parameters")) fail(where, ": missing rope_parameters");
    c.rope_sliding = parse_rope(t.at("rope_parameters"), "sliding_attention", where);
    c.rope_full = parse_rope(t.at("rope_parameters"), "full_attention", where);

    XS_CHECK(c.num_heads % c.num_kv_heads == 0 && c.num_heads % c.num_global_kv_heads == 0, where,
             ": num_attention_heads ", c.num_heads, " is not a multiple of the KV heads (", c.num_kv_heads, ", global ",
             c.num_global_kv_heads, ")");
    if (drafter) {
        XS_CHECK(c.num_kv_shared_layers == c.num_layers, where, ": an assistant's layers must all read the target's KV "
                 "(num_kv_shared_layers ", c.num_kv_shared_layers, " of ", c.num_layers, ")");
        XS_CHECK(c.ple_dim == 0, where, ": per-layer embeddings in an assistant are not supported");
        return c;
    }
    XS_CHECK(c.num_kv_shared_layers >= 0 && c.num_kv_shared_layers < c.num_layers, where,
             ": num_kv_shared_layers out of range");
    for (int64_t i = c.first_kv_shared_layer(); i < c.num_layers; ++i) (void)c.kv_source_layer(i);
    return c;
}

}  // namespace

Config parse_config(const nlohmann::json& config_json) {
    const bool nested = config_json.contains("text_config");
    return parse_text(config_json, nested ? config_json.at("text_config") : config_json,
                      nested ? "config.json text_config" : "config.json", false);
}

AssistantConfig parse_assistant_config(const nlohmann::json& config_json) {
    const std::string where = "assistant config.json";
    const std::string type = cfg::get_string(config_json, "model_type", where);
    if (type != "gemma4_assistant") fail(where, ": model_type '", type, "' is not a Gemma 4 assistant (gemma4_assistant)");
    if (!config_json.contains("text_config")) fail(where, ": missing text_config");
    AssistantConfig a;
    a.text = parse_text(config_json, config_json.at("text_config"), where + " text_config", true);
    a.backbone_hidden_size = cfg::get_int(config_json, "backbone_hidden_size", where);
    a.ordered_embeddings = cfg::get_bool_or(config_json, "use_ordered_embeddings", false, where);
    if (a.ordered_embeddings) {
        a.num_centroids = cfg::get_int(config_json, "num_centroids", where);
        a.centroid_top_k = cfg::get_int(config_json, "centroid_intermediate_top_k", where);
        XS_CHECK(a.num_centroids > 0 && a.text.vocab_size % a.num_centroids == 0 && a.centroid_top_k > 0 &&
                     a.centroid_top_k <= a.num_centroids,
                 where, ": ", a.num_centroids, " centroids (top ", a.centroid_top_k, ") over a vocabulary of ",
                 a.text.vocab_size);
    }
    return a;
}

}  // namespace xs::gemma4
