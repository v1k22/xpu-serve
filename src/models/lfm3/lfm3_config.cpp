#include "models/config_util.h"
#include "models/lfm3/lfm3.h"
#include "util/error.h"

namespace xs::lfm3 {
namespace {

// rope_parameters[kind]: Lfm3RotaryEmbedding.compute_default_rope_parameters.
Rope parse_rope(const nlohmann::json& rope_parameters, const char* kind, int64_t head_dim, const std::string& where) {
    const std::string rw = where + " rope_parameters." + kind;
    if (!rope_parameters.is_object() || !rope_parameters.contains(kind) || !rope_parameters.at(kind).is_object()) {
        fail(where, ": rope_parameters has no entry for the ", kind, " layers");
    }
    const nlohmann::json& r = rope_parameters.at(kind);
    const std::string type = r.contains("rope_type") ? cfg::get_string(r, "rope_type", rw) : "default";
    if (type != "default") fail(rw, ": rope_type '", type, "' is not supported (supported: default)");
    Rope rope;
    rope.theta = cfg::get_float(r, "rope_theta", rw);
    const double partial = r.contains("partial_rotary_factor") ? cfg::get_float(r, "partial_rotary_factor", rw) : 1.0;
    rope.pairs = static_cast<int64_t>(partial * static_cast<double>(head_dim)) / 2;
    XS_CHECK(rope.theta > 0 && rope.pairs >= 1 && rope.pairs <= head_dim / 2, rw, ": theta ", rope.theta, " and ",
             rope.pairs, " rotated pairs do not fit head_dim ", head_dim);
    return rope;
}

}  // namespace

const nlohmann::json& text_config(const nlohmann::json& config_json) {
    const std::string type = cfg::get_string(config_json, "model_type", "config.json");
    if (type == "lfm3") return config_json;
    if (type != "lfm3_vl") fail("lfm3: model_type '", type, "' is not supported (lfm3, lfm3_vl are)");
    if (!config_json.contains("text_config") || !config_json.at("text_config").is_object()) {
        fail("config.json: an lfm3_vl checkpoint without a text_config");
    }
    const nlohmann::json& text = config_json.at("text_config");
    const std::string text_type =
        text.contains("model_type") ? cfg::get_string(text, "model_type", "config.json text_config") : "lfm3";
    if (text_type != "lfm3") fail("config.json: text_config model_type '", text_type, "' is not supported (lfm3 is)");
    return text;
}

Config parse_config(const nlohmann::json& t) {
    const std::string where = "config.json";
    if (cfg::get_bool_or(t, "conv_bias", false, where)) fail(where, ": conv_bias is not supported");
    if (!cfg::get_bool_or(t, "use_sandwich_norm", true, where)) {
        fail(where, ": use_sandwich_norm false is not supported (the released checkpoints norm each layer's outputs)");
    }
    // Hugging Face reads the legacy key over the standard one.
    const bool tied =
        cfg::get_bool_or(t, "tie_embedding", cfg::get_bool_or(t, "tie_word_embeddings", true, where), where);
    if (!tied) fail(where, ": an untied output head is not supported (tie_word_embeddings false)");

    Config c;
    c.vocab_size = cfg::get_int(t, "vocab_size", where);
    c.hidden_size = cfg::get_int(t, "hidden_size", where);
    c.intermediate_size = cfg::get_int(t, "intermediate_size", where);
    c.num_layers = cfg::get_int(t, "num_hidden_layers", where);
    c.norm_eps = cfg::get_float(t, "norm_eps", where);
    c.max_position = cfg::get_int(t, "max_position_embeddings", where);
    c.num_heads = cfg::get_int(t, "num_attention_heads", where);
    c.num_kv_heads = cfg::get_int(t, "num_key_value_heads", where);
    XS_CHECK(c.num_heads > 0 && c.num_kv_heads > 0 && c.num_heads % c.num_kv_heads == 0, where,
             ": num_attention_heads ", c.num_heads, " is not a multiple of num_key_value_heads ", c.num_kv_heads);
    c.head_dim = cfg::get_int_or(t, "head_dim", c.hidden_size / c.num_heads, where);
    XS_CHECK(c.head_dim > 0 && c.head_dim % 2 == 0, where, ": head_dim ", c.head_dim, " must be even");
    c.conv_kernel = cfg::get_int_or(t, "conv_L_cache", 7, where);
    XS_CHECK(c.conv_kernel >= 2, where, ": conv_L_cache must be at least 2");

    if (!t.contains("layer_types") || !t.at("layer_types").is_array()) fail(where, ": missing layer_types");
    bool full = false, sliding = false;
    for (const auto& lt : t.at("layer_types")) {
        if (!lt.is_string()) fail(where, ": layer_types must be a list of strings");
        const std::string s = lt.get<std::string>();
        if (s == "conv") {
            c.layer_kinds.push_back(LayerKind::Conv);
        } else if (s == "full_attention") {
            c.layer_kinds.push_back(LayerKind::Full);
            full = true;
        } else if (s == "sliding_attention") {
            c.layer_kinds.push_back(LayerKind::Sliding);
            sliding = true;
        } else {
            fail(where, ": unknown layer type '", s, "'");
        }
    }
    XS_CHECK(static_cast<int64_t>(c.layer_kinds.size()) == c.num_layers, where, ": layer_types has ",
             c.layer_kinds.size(), " entries for ", c.num_layers, " layers");

    if (full || sliding) {
        if (!t.contains("rope_parameters")) fail(where, ": missing rope_parameters");
        if (full) c.full_rope = parse_rope(t.at("rope_parameters"), "full_attention", c.head_dim, where);
        if (sliding) c.sliding_rope = parse_rope(t.at("rope_parameters"), "sliding_attention", c.head_dim, where);
    }
    if (sliding) {
        c.sliding_window = cfg::get_int(t, "sliding_window", where);
        XS_CHECK(c.sliding_window > 0, where, ": sliding_window must be positive");
    }
    return c;
}

}  // namespace xs::lfm3
