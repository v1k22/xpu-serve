#include "models/config_util.h"
#include "models/nomic_bert/nomic_bert.h"
#include "util/error.h"

namespace xs::nomic_bert {
namespace {

bool present(const nlohmann::json& t, const char* key) {
    return t.contains(key) && !t.at(key).is_null();
}

double float_or(const nlohmann::json& t, const char* key, double fallback, const std::string& where) {
    return present(t, key) ? cfg::get_float(t, key, where) : fallback;
}

}  // namespace

// Absent keys take NomicBertConfig's defaults (and GPT2Config's, which it extends).
Config parse_config(const nlohmann::json& t) {
    const std::string where = "config.json";
    const std::string act = present(t, "activation_function") ? cfg::get_string(t, "activation_function", where)
                                                              : "gelu_new";
    if (act != "swiglu") fail(where, ": activation_function '", act, "' is not supported (supported: swiglu)");
    for (const char* key : {"prenorm", "parallel_block", "use_rms_norm", "causal", "rotary_emb_interleaved",
                            "norm_mlp", "rotary_head_dim"}) {
        if (cfg::get_bool_or(t, key, false, where)) fail(where, ": ", key, " is not supported");
    }
    for (const char* key : {"qkv_proj_bias", "mlp_fc1_bias", "mlp_fc2_bias"}) {
        if (cfg::get_bool_or(t, key, true, where)) fail(where, ": ", key, " (a projection with a bias) is not supported");
    }
    for (const char* key : {"rotary_scaling_factor", "rotary_emb_scale_base"}) {
        if (present(t, key)) fail(where, ": ", key, " is not supported (supported: null)");
    }
    if (cfg::get_int_or(t, "moe_every_n_layers", 0, where) > 0) fail(where, ": mixture of experts layers are not supported");
    const double fraction = float_or(t, "rotary_emb_fraction", 0.0, where);
    if (fraction != 1.0) {
        fail(where, ": rotary_emb_fraction ", fraction, " is not supported (supported: 1, the whole head rotary)");
    }
    if (cfg::get_int_or(t, "type_vocab_size", 2, where) < 1) {
        fail(where, ": type_vocab_size 0 (no token type embeddings) is not supported");
    }

    Config c;
    c.hidden_size = cfg::get_int(t, "n_embd", where);
    c.num_layers = cfg::get_int(t, "n_layer", where);
    c.num_heads = cfg::get_int(t, "n_head", where);
    XS_CHECK(c.hidden_size > 0 && c.num_layers > 0 && c.num_heads > 0 && c.hidden_size % c.num_heads == 0, where,
             ": n_embd ", c.hidden_size, " is not a multiple of n_head ", c.num_heads);
    c.head_dim = c.hidden_size / c.num_heads;
    XS_CHECK(c.head_dim % 2 == 0, where, ": head dim ", c.head_dim, " must be even for rotary embeddings");
    if (cfg::get_int_or(t, "num_heads_kv", c.num_heads, where) != c.num_heads) {
        fail(where, ": num_heads_kv (grouped key and value heads) is not supported");
    }
    c.max_positions = present(t, "max_position_embeddings") ? cfg::get_int(t, "max_position_embeddings", where)
                                                            : cfg::get_int(t, "n_positions", where);
    XS_CHECK(c.max_positions > 0, where, ": max_position_embeddings must be positive");
    c.layer_norm_eps = float_or(t, "layer_norm_epsilon", c.layer_norm_eps, where);
    c.rope_theta = float_or(t, "rotary_emb_base", c.rope_theta, where);
    XS_CHECK(c.rope_theta > 0, where, ": rotary_emb_base must be positive");
    return c;
}

}  // namespace xs::nomic_bert
