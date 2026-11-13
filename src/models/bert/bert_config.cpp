#include "models/bert/bert.h"
#include "models/config_util.h"
#include "util/error.h"

namespace xs::bert {

Config parse_config(const nlohmann::json& t) {
    const std::string where = "config.json";
    const std::string act = cfg::get_string(t, "hidden_act", where);
    if (act != "gelu") fail(where, ": hidden_act '", act, "' is not supported (supported: gelu)");
    if (t.contains("position_embedding_type") && !t.at("position_embedding_type").is_null()) {
        const std::string positions = cfg::get_string(t, "position_embedding_type", where);
        if (positions != "absolute") {
            fail(where, ": position_embedding_type '", positions, "' is not supported (supported: absolute)");
        }
    }
    if (cfg::get_bool_or(t, "is_decoder", false, where)) fail(where, ": is_decoder (causal attention) is not supported");
    if (cfg::get_bool_or(t, "add_cross_attention", false, where)) fail(where, ": add_cross_attention is not supported");
    if (cfg::get_int_or(t, "type_vocab_size", 2, where) < 1) {
        fail(where, ": type_vocab_size 0 (no token type embeddings) is not supported");
    }

    Config c;
    c.hidden_size = cfg::get_int(t, "hidden_size", where);
    c.num_layers = cfg::get_int(t, "num_hidden_layers", where);
    c.num_heads = cfg::get_int(t, "num_attention_heads", where);
    XS_CHECK(c.hidden_size > 0 && c.num_layers > 0 && c.num_heads > 0 && c.hidden_size % c.num_heads == 0, where,
             ": hidden_size ", c.hidden_size, " is not a multiple of num_attention_heads ", c.num_heads);
    c.head_dim = c.hidden_size / c.num_heads;
    c.max_positions = cfg::get_int(t, "max_position_embeddings", where);
    XS_CHECK(c.max_positions > 0, where, ": max_position_embeddings must be positive");
    if (t.contains("layer_norm_eps")) c.layer_norm_eps = cfg::get_float(t, "layer_norm_eps", where);
    return c;
}

}  // namespace xs::bert
