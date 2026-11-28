#include "models/config_util.h"
#include "models/llama/llama.h"
#include "util/error.h"

namespace xs::llama {

Config parse_config(const nlohmann::json& t) {
    const std::string where = "config.json";
    if (cfg::get_bool_or(t, "attention_bias", false, where)) fail("llama: attention_bias is not supported");
    if (cfg::get_bool_or(t, "mlp_bias", false, where)) fail("llama: mlp_bias is not supported");
    const std::string act = cfg::get_string(t, "hidden_act", where);
    if (act != "silu") fail("llama: hidden_act '", act, "' is not supported (silu is)");

    Config c;
    c.vocab_size = cfg::get_int(t, "vocab_size", where);
    c.hidden_size = cfg::get_int(t, "hidden_size", where);
    c.intermediate_size = cfg::get_int(t, "intermediate_size", where);
    c.num_layers = cfg::get_int(t, "num_hidden_layers", where);
    c.num_heads = cfg::get_int(t, "num_attention_heads", where);
    c.num_kv_heads = cfg::get_int_or(t, "num_key_value_heads", c.num_heads, where);
    c.head_dim = cfg::get_int_or(t, "head_dim", c.hidden_size / c.num_heads, where);
    c.rms_norm_eps = cfg::get_float(t, "rms_norm_eps", where);
    c.max_position = cfg::get_int(t, "max_position_embeddings", where);
    c.tie_word_embeddings = cfg::get_bool_or(t, "tie_word_embeddings", false, where);
    XS_CHECK(c.num_heads > 0 && c.num_kv_heads > 0 && c.num_heads % c.num_kv_heads == 0, "llama: num_attention_heads ",
             c.num_heads, " is not a multiple of num_key_value_heads ", c.num_kv_heads);
    XS_CHECK(c.head_dim > 0 && c.head_dim % 2 == 0, "llama: head_dim ", c.head_dim, " must be even");

    // RoPE: rope_theta and rope_scaling at the top (the released configs, transformers 4), or one rope_parameters
    // object (transformers 5).
    const nlohmann::json* params = t.contains("rope_parameters") && t.at("rope_parameters").is_object()
                                       ? &t.at("rope_parameters")
                                       : nullptr;
    const nlohmann::json* scaling =
        t.contains("rope_scaling") && !t.at("rope_scaling").is_null() ? &t.at("rope_scaling") : params;
    c.rope_theta = params != nullptr && params->contains("rope_theta") ? cfg::get_float(*params, "rope_theta", where)
                                                                       : cfg::get_float(t, "rope_theta", where);
    for (const nlohmann::json* o : {&t, params}) {
        if (o != nullptr && o->contains("partial_rotary_factor") &&
            cfg::get_float(*o, "partial_rotary_factor", where) != 1.0) {
            fail("llama: partial_rotary_factor is not supported");
        }
    }
    if (scaling != nullptr) {
        std::string type = "default";
        for (const char* key : {"rope_type", "type"}) {
            if (scaling->contains(key)) type = cfg::get_string(*scaling, key, where);
        }
        if (type == "llama3") {
            const std::string w = where + " rope_scaling";
            blocks::Llama3RopeScaling s;
            s.factor = cfg::get_float(*scaling, "factor", w);
            s.low_freq_factor = cfg::get_float(*scaling, "low_freq_factor", w);
            s.high_freq_factor = cfg::get_float(*scaling, "high_freq_factor", w);
            s.original_max_position =
                scaling->contains("original_max_position_embeddings")
                    ? cfg::get_float(*scaling, "original_max_position_embeddings", w)
                    : static_cast<double>(c.max_position);  // transformers' fallback
            XS_CHECK(s.factor > 0 && s.low_freq_factor > 0 && s.high_freq_factor > s.low_freq_factor &&
                         s.original_max_position > 0,
                     "llama: rope_scaling factor ", s.factor, ", low_freq_factor ", s.low_freq_factor,
                     ", high_freq_factor ", s.high_freq_factor, ", original_max_position_embeddings ",
                     s.original_max_position);
            c.llama3_rope = s;
        } else if (type != "default") {
            fail("llama: RoPE type '", type, "' is not supported (default and llama3 are)");
        }
    }
    return c;
}

}  // namespace xs::llama
