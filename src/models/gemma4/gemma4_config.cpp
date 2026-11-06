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

void refuse_unsupported(const nlohmann::json& t, const std::string& where) {
    if (cfg::get_bool_or(t, "enable_moe_block", false, where)) {
        fail(where, ": mixture-of-experts layers (enable_moe_block, Gemma 4 26B-A4B) are not supported yet");
    }
    if (cfg::get_bool_or(t, "attention_k_eq_v", false, where)) {
        fail(where, ": attention_k_eq_v (Gemma 4 31B / 26B-A4B global layers) is not supported yet");
    }
    if (t.contains("use_bidirectional_attention") && !t.at("use_bidirectional_attention").is_null()) {
        fail(where, ": use_bidirectional_attention (embedding models) is not supported");
    }
    if (cfg::get_bool_or(t, "attention_bias", false, where)) fail(where, ": attention_bias is not supported");
    const std::string act = cfg::get_string(t, "hidden_activation", where);
    if (act != "gelu_pytorch_tanh") fail(where, ": hidden_activation '", act, "' is not supported");
}

}  // namespace

int64_t Config::layer_head_dim(int64_t layer) const {
    return layer_kinds.at(static_cast<size_t>(layer)) == AttnKind::Full ? global_head_dim : head_dim;
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

Config parse_config(const nlohmann::json& config_json) {
    const bool nested = config_json.contains("text_config");
    const nlohmann::json& t = nested ? config_json.at("text_config") : config_json;
    const std::string where = nested ? "config.json text_config" : "config.json";
    refuse_unsupported(t, where);

    Config c;
    c.vocab_size = cfg::get_int(t, "vocab_size", where);
    c.hidden_size = cfg::get_int(t, "hidden_size", where);
    c.intermediate_size = cfg::get_int(t, "intermediate_size", where);
    c.num_layers = cfg::get_int(t, "num_hidden_layers", where);
    c.num_heads = cfg::get_int(t, "num_attention_heads", where);
    c.num_kv_heads = cfg::get_int(t, "num_key_value_heads", where);
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

    if (!t.contains("rope_parameters")) fail(where, ": missing rope_parameters");
    c.rope_sliding = parse_rope(t.at("rope_parameters"), "sliding_attention", where);
    c.rope_full = parse_rope(t.at("rope_parameters"), "full_attention", where);

    XS_CHECK(c.num_heads % c.num_kv_heads == 0, where, ": num_attention_heads ", c.num_heads,
             " is not a multiple of num_key_value_heads ", c.num_kv_heads);
    XS_CHECK(c.num_kv_shared_layers >= 0 && c.num_kv_shared_layers < c.num_layers, where,
             ": num_kv_shared_layers out of range");
    for (int64_t i = c.first_kv_shared_layer(); i < c.num_layers; ++i) (void)c.kv_source_layer(i);
    return c;
}

}  // namespace xs::gemma4
