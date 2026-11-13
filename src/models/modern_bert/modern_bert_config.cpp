#include "models/config_util.h"
#include "models/modern_bert/modern_bert.h"
#include "util/error.h"

namespace xs::modern_bert {
namespace {

bool present(const nlohmann::json& t, const char* key) {
    return t.contains(key) && !t.at(key).is_null();
}

// The rope_theta of one layer kind's rope_parameters entry (`fallback` when it names none); only the default
// (unscaled) RoPE is implemented.
double rope_theta(const nlohmann::json& entry, double fallback, const std::string& where) {
    if (entry.is_null()) return fallback;
    XS_CHECK(entry.is_object(), where, " is not an object");
    for (auto it = entry.begin(); it != entry.end(); ++it) {
        if (it.key() != "rope_theta" && it.key() != "rope_type") fail(where, ": '", it.key(), "' is not supported");
    }
    const std::string type = present(entry, "rope_type") ? cfg::get_string(entry, "rope_type", where) : "default";
    if (type != "default") fail(where, ": rope_type '", type, "' is not supported (supported: default)");
    return present(entry, "rope_theta") ? cfg::get_float(entry, "rope_theta", where) : fallback;
}

}  // namespace

Config parse_config(const nlohmann::json& t, const std::string& where) {
    const std::string type = cfg::get_string(t, "model_type", where);
    if (type != "modernbert") fail(where, ": model_type '", type, "' is not ModernBERT's (modernbert)");
    const std::string act = present(t, "hidden_activation") ? cfg::get_string(t, "hidden_activation", where) : "gelu";
    if (act != "gelu") fail(where, ": hidden_activation '", act, "' is not supported (supported: gelu)");
    for (const char* key : {"norm_bias", "attention_bias", "mlp_bias"}) {
        if (cfg::get_bool_or(t, key, false, where)) fail(where, ": ", key, " (a bias in the encoder) is not supported");
    }
    if (present(t, "rope_scaling")) fail(where, ": rope_scaling is not supported");

    Config c;
    c.hidden_size = cfg::get_int(t, "hidden_size", where);
    c.num_layers = cfg::get_int(t, "num_hidden_layers", where);
    c.num_heads = cfg::get_int(t, "num_attention_heads", where);
    XS_CHECK(c.hidden_size > 0 && c.num_layers > 0 && c.num_heads > 0 && c.hidden_size % c.num_heads == 0, where,
             ": hidden_size ", c.hidden_size, " is not a multiple of num_attention_heads ", c.num_heads);
    c.head_dim = c.hidden_size / c.num_heads;
    if (present(t, "head_dim") && cfg::get_int(t, "head_dim", where) != c.head_dim) {
        fail(where, ": head_dim ", cfg::get_int(t, "head_dim", where), " is not hidden_size / num_attention_heads");
    }
    XS_CHECK(c.head_dim % 2 == 0, where, ": head dim ", c.head_dim, " must be even for rotary embeddings");
    c.intermediate_size = cfg::get_int(t, "intermediate_size", where);
    c.vocab_size = cfg::get_int(t, "vocab_size", where);
    c.max_positions = cfg::get_int(t, "max_position_embeddings", where);
    XS_CHECK(c.intermediate_size > 0 && c.vocab_size > 0 && c.max_positions > 0, where,
             ": intermediate_size, vocab_size and max_position_embeddings must be positive");
    if (present(t, "norm_eps")) c.norm_eps = cfg::get_float(t, "norm_eps", where);
    c.local_attention = cfg::get_int_or(t, "local_attention", c.local_attention, where);
    XS_CHECK(c.local_attention >= 0, where, ": local_attention ", c.local_attention);

    if (present(t, "layer_types")) {
        const nlohmann::json& types = t.at("layer_types");
        XS_CHECK(types.is_array() && static_cast<int64_t>(types.size()) == c.num_layers, where, ": layer_types has ",
                 types.size(), " entries for ", c.num_layers, " layers");
        for (const auto& k : types) {
            const std::string kind = k.is_string() ? k.get<std::string>() : k.dump();
            if (kind != "full_attention" && kind != "sliding_attention") {
                fail(where, ": layer type '", kind, "' is not supported (full_attention, sliding_attention are)");
            }
            c.local.push_back(kind == "sliding_attention");
        }
    } else {
        const int64_t every = cfg::get_int_or(t, "global_attn_every_n_layers", 3, where);
        XS_CHECK(every > 0, where, ": global_attn_every_n_layers ", every);
        for (int64_t i = 0; i < c.num_layers; ++i)
            c.local.push_back(i % every != 0);
    }

    // An older config's global_rope_theta and local_rope_theta, unless rope_parameters names its own.
    if (present(t, "global_rope_theta")) c.global_rope_theta = cfg::get_float(t, "global_rope_theta", where);
    if (present(t, "local_rope_theta")) c.local_rope_theta = cfg::get_float(t, "local_rope_theta", where);
    if (present(t, "rope_parameters")) {
        const nlohmann::json& rope = t.at("rope_parameters");
        XS_CHECK(rope.is_object(), where, ": rope_parameters is not an object");
        for (auto it = rope.begin(); it != rope.end(); ++it) {
            if (it.key() == "full_attention") {
                c.global_rope_theta =
                    rope_theta(it.value(), c.global_rope_theta, where + " rope_parameters.full_attention");
            } else if (it.key() == "sliding_attention") {
                c.local_rope_theta =
                    rope_theta(it.value(), c.local_rope_theta, where + " rope_parameters.sliding_attention");
            } else {
                fail(where, ": rope_parameters for '", it.key(), "' layers are not supported");
            }
        }
    }
    XS_CHECK(c.global_rope_theta > 0 && c.local_rope_theta > 0, where, ": a rope_theta must be positive");
    return c;
}

}  // namespace xs::modern_bert
