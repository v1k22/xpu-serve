#include "models/config_util.h"
#include "models/qwen3_5/qwen3_5.h"
#include "util/error.h"

namespace xs::qwen3_5 {

Config parse_config(const nlohmann::json& config_json) {
    const bool nested = config_json.contains("text_config");
    const nlohmann::json& t = nested ? config_json.at("text_config") : config_json;
    const std::string where = nested ? "config.json text_config" : "config.json";

    if (t.contains("num_experts") || t.contains("num_local_experts")) {
        fail(where, ": mixture-of-experts layers (Qwen3.5 MoE) are not supported yet");
    }
    if (cfg::get_bool_or(t, "attention_bias", false, where)) fail(where, ": attention_bias is not supported");
    const std::string act = cfg::get_string(t, "hidden_act", where);
    if (act != "silu") fail(where, ": hidden_act '", act, "' is not supported");
    if (t.contains("mlp_only_layers") && t.at("mlp_only_layers").is_array() && !t.at("mlp_only_layers").empty()) {
        fail(where, ": mlp_only_layers is not supported");
    }

    Config c;
    c.vocab_size = cfg::get_int(t, "vocab_size", where);
    c.hidden_size = cfg::get_int(t, "hidden_size", where);
    c.intermediate_size = cfg::get_int(t, "intermediate_size", where);
    c.num_layers = cfg::get_int(t, "num_hidden_layers", where);
    c.rms_norm_eps = cfg::get_float(t, "rms_norm_eps", where);
    c.max_position = cfg::get_int(t, "max_position_embeddings", where);
    c.num_heads = cfg::get_int(t, "num_attention_heads", where);
    c.num_kv_heads = cfg::get_int(t, "num_key_value_heads", where);
    c.head_dim = cfg::get_int_or(t, "head_dim", c.hidden_size / c.num_heads, where);
    c.attn_output_gate = cfg::get_bool_or(t, "attn_output_gate", true, where);
    c.linear_key_heads = cfg::get_int(t, "linear_num_key_heads", where);
    c.linear_value_heads = cfg::get_int(t, "linear_num_value_heads", where);
    c.linear_key_dim = cfg::get_int(t, "linear_key_head_dim", where);
    c.linear_value_dim = cfg::get_int(t, "linear_value_head_dim", where);
    c.conv_kernel = cfg::get_int(t, "linear_conv_kernel_dim", where);

    if (!t.contains("layer_types") || !t.at("layer_types").is_array()) fail(where, ": missing layer_types");
    for (const auto& lt : t.at("layer_types")) {
        const std::string s = lt.get<std::string>();
        if (s == "linear_attention") {
            c.layer_kinds.push_back(LayerKind::Linear);
        } else if (s == "full_attention") {
            c.layer_kinds.push_back(LayerKind::Full);
        } else {
            fail(where, ": unknown layer type '", s, "'");
        }
    }
    XS_CHECK(static_cast<int64_t>(c.layer_kinds.size()) == c.num_layers, where, ": layer_types has ",
             c.layer_kinds.size(), " entries for ", c.num_layers, " layers");

    if (!t.contains("rope_parameters")) fail(where, ": missing rope_parameters");
    const nlohmann::json& rope = t.at("rope_parameters");
    const std::string rw = where + " rope_parameters";
    const std::string rope_type = cfg::get_string(rope, "rope_type", rw);
    if (rope_type != "default") fail(rw, ": rope_type '", rope_type, "' is not supported (supported: default)");
    c.rope_theta = cfg::get_float(rope, "rope_theta", rw);
    const double partial = rope.contains("partial_rotary_factor") ? cfg::get_float(rope, "partial_rotary_factor", rw) : 1.0;
    c.rotary_dim = static_cast<int64_t>(static_cast<double>(c.head_dim) * partial);
    // Multimodal RoPE (mrope_section) gives text tokens the same position on all three axes, which is plain RoPE.

    XS_CHECK(c.num_heads % c.num_kv_heads == 0, where, ": num_attention_heads ", c.num_heads,
             " is not a multiple of num_key_value_heads ", c.num_kv_heads);
    XS_CHECK(c.linear_value_heads % c.linear_key_heads == 0, where, ": linear_num_value_heads ", c.linear_value_heads,
             " is not a multiple of linear_num_key_heads ", c.linear_key_heads);
    XS_CHECK(c.rotary_dim > 0 && c.rotary_dim % 2 == 0 && c.rotary_dim <= c.head_dim, where, ": rotary dim ",
             c.rotary_dim, " does not fit head_dim ", c.head_dim);
    XS_CHECK(c.conv_kernel > 1, where, ": linear_conv_kernel_dim must be at least 2");
    return c;
}

}  // namespace xs::qwen3_5
