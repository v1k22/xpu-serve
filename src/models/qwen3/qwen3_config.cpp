#include "models/config_util.h"
#include "models/qwen3/qwen3.h"
#include "util/error.h"

namespace xs::qwen3 {

Config parse_config(const nlohmann::json& t, const std::string& where) {
    if (cfg::get_bool_or(t, "attention_bias", false, where)) fail(where, ": attention_bias is not supported");
    if (cfg::get_bool_or(t, "use_sliding_window", false, where)) fail(where, ": use_sliding_window is not supported");
    if (t.contains("num_experts") && !t.at("num_experts").is_null() && cfg::get_int(t, "num_experts", where) > 0) {
        fail(where, ": num_experts (Qwen3 MoE) is not supported yet");
    }
    const std::string act = cfg::get_string(t, "hidden_act", where);
    if (act != "silu") fail(where, ": hidden_act '", act, "' is not supported (silu is)");

    Config c;
    c.vocab_size = cfg::get_int(t, "vocab_size", where);
    c.hidden_size = cfg::get_int(t, "hidden_size", where);
    c.intermediate_size = cfg::get_int(t, "intermediate_size", where);
    c.num_layers = cfg::get_int(t, "num_hidden_layers", where);
    c.num_heads = cfg::get_int(t, "num_attention_heads", where);
    c.num_kv_heads = cfg::get_int(t, "num_key_value_heads", where);
    c.head_dim = cfg::get_int_or(t, "head_dim", c.hidden_size / c.num_heads, where);
    c.rms_norm_eps = cfg::get_float(t, "rms_norm_eps", where);
    c.max_position = cfg::get_int(t, "max_position_embeddings", where);
    c.tie_word_embeddings = cfg::get_bool_or(t, "tie_word_embeddings", false, where);
    XS_CHECK(c.num_heads % c.num_kv_heads == 0, where, ": num_attention_heads ", c.num_heads,
             " is not a multiple of num_key_value_heads ", c.num_kv_heads);

    // rope_theta at the top (transformers 4) or in rope_parameters (5); rope_scaling may name the RoPE type.
    const nlohmann::json* rope = t.contains("rope_parameters") ? &t.at("rope_parameters") : nullptr;
    const bool has_scaling = t.contains("rope_scaling") && !t.at("rope_scaling").is_null();
    const nlohmann::json* scaling = has_scaling ? &t.at("rope_scaling") : nullptr;
    c.rope_theta = rope != nullptr && rope->contains("rope_theta") ? cfg::get_float(*rope, "rope_theta", where)
                                                                    : cfg::get_float(t, "rope_theta", where);
    for (const nlohmann::json* r : {rope, scaling}) {
        if (r == nullptr) continue;
        for (const char* key : {"rope_type", "type"}) {
            if (!r->contains(key)) continue;
            const std::string type = cfg::get_string(*r, key, where);
            if (type != "default") fail(where, ": RoPE type '", type, "' is not supported (default is)");
        }
        if (r->contains("partial_rotary_factor") && cfg::get_float(*r, "partial_rotary_factor", where) != 1.0) {
            fail(where, ": partial_rotary_factor is not supported");
        }
    }
    return c;
}

}  // namespace xs::qwen3
