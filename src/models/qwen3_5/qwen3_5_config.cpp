#include <algorithm>
#include <cmath>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/qwen3_5/qwen3_5.h"
#include "util/error.h"

namespace xs::qwen3_5 {
namespace {

// SenseNova Loop's loop_settings and mhc (config.json's root), checked as its execution_plan() checks them.
void parse_loop(const nlohmann::json& root, Config& c) {
    const std::string type = root.contains("model_type") && root.at("model_type").is_string()
                                 ? root.at("model_type").get<std::string>()
                                 : "";
    const bool loop = root.contains("loop_settings") && !root.at("loop_settings").is_null();
    const bool mhc = root.contains("mhc") && !root.at("mhc").is_null();
    if (type != "sensenova_loop") {
        if (loop || mhc) fail("config.json: loop_settings and mhc are SenseNova Loop's (model_type sensenova_loop)");
        return;
    }
    if (!loop) fail("config.json: a sensenova_loop checkpoint needs loop_settings");
    const nlohmann::json& l = root.at("loop_settings");
    const std::string lw = "config.json loop_settings";
    const int64_t start = cfg::get_int(l, "start", lw), end = cfg::get_int(l, "end", lw);
    const int64_t count = cfg::get_int(l, "eval_count", lw);
    XS_CHECK(0 <= start && start < end && end <= c.num_layers && count >= 1, lw, ": start ", start, ", end ", end,
             " and eval_count ", count, " do not fit ", c.num_layers, " layers");
    if (!l.contains("counts") || l.at("counts") != nlohmann::json::array({count})) {
        fail(lw, ": counts must be [eval_count] (a fixed loop count; others are not supported)");
    }
    std::vector<int64_t>& plan = c.plan;
    for (int64_t i = 0; i < start; ++i) plan.push_back(i);
    for (int64_t pass = 0; pass < count; ++pass)
        for (int64_t i = start; i < end; ++i) plan.push_back(i);
    for (int64_t i = end; i < c.num_layers; ++i) plan.push_back(i);

    if (!mhc) return;
    const nlohmann::json& m = root.at("mhc");
    const std::string mw = "config.json mhc";
    for (const auto& [key, value] : m.items()) {
        static const std::vector<std::string> known = {"start", "end", "streams", "iterations", "fixed_identity",
                                                       "pre_scale"};
        if (std::find(known.begin(), known.end(), key) == known.end()) fail(mw, ": unknown key '", key, "'");
    }
    c.mhc.start = cfg::get_int(m, "start", mw);
    c.mhc.end = cfg::get_int(m, "end", mw);
    c.mhc.streams = cfg::get_int(m, "streams", mw);
    c.mhc.iterations = static_cast<int>(cfg::get_int(m, "iterations", mw));
    c.mhc.pre_scale = static_cast<float>(m.contains("pre_scale") ? cfg::get_float(m, "pre_scale", mw) : 1.0);
    XS_CHECK(0 <= c.mhc.start && c.mhc.start < c.mhc.end && c.mhc.end <= c.num_layers, mw, ": layers [", c.mhc.start,
             ", ", c.mhc.end, ") do not fit ", c.num_layers);
    XS_CHECK(c.mhc.streams == 2 || c.mhc.streams == 4, mw, ": streams ", c.mhc.streams, " (2 or 4)");
    XS_CHECK(c.mhc.iterations >= 1, mw, ": iterations ", c.mhc.iterations);
    XS_CHECK(std::isfinite(c.mhc.pre_scale) && c.mhc.pre_scale > 0, mw, ": pre_scale ", c.mhc.pre_scale);
    if (!m.contains("fixed_identity") || !m.at("fixed_identity").is_boolean()) {
        fail(mw, ": fixed_identity must be a boolean");
    }
    if (m.at("fixed_identity").get<bool>()) fail(mw, ": fixed_identity (no stream mix) is not supported yet");
    // Inside one phase of the plan: before the loop, inside it or after it.
    const int64_t bounds[] = {0, start, end, c.num_layers};
    bool inside = false;
    for (int b = 0; b < 3; ++b) inside = inside || (bounds[b] <= c.mhc.start && c.mhc.end <= bounds[b + 1]);
    XS_CHECK(inside, mw, ": layers [", c.mhc.start, ", ", c.mhc.end, ") cross a boundary of the loop [", start, ", ",
             end, ")");
}

}  // namespace

Config parse_config(const nlohmann::json& config_json) {
    const bool nested = config_json.contains("text_config");
    const nlohmann::json& t = nested ? config_json.at("text_config") : config_json;
    const std::string where = nested ? "config.json text_config" : "config.json";

    if (t.contains("num_local_experts")) fail(where, ": num_local_experts is not a Qwen3.5 MoE key");
    if (cfg::get_bool_or(t, "attention_bias", false, where)) fail(where, ": attention_bias is not supported");
    const std::string act = cfg::get_string(t, "hidden_act", where);
    if (act != "silu") fail(where, ": hidden_act '", act, "' is not supported");
    // The gate of linear attention's output norm (Qwen3.8 names it; earlier checkpoints imply it): SiLU.
    const std::string gate = t.contains("output_gate_type") ? cfg::get_string(t, "output_gate_type", where) : "swish";
    if (gate != "swish") fail(where, ": output_gate_type '", gate, "' is not supported (supported: swish)");
    if (t.contains("mlp_only_layers") && t.at("mlp_only_layers").is_array() && !t.at("mlp_only_layers").empty()) {
        fail(where, ": mlp_only_layers is not supported");
    }

    Config c;
    c.vocab_size = cfg::get_int(t, "vocab_size", where);
    c.hidden_size = cfg::get_int(t, "hidden_size", where);
    if (t.contains("num_experts")) {
        c.num_experts = cfg::get_int(t, "num_experts", where);
        c.experts_per_token = cfg::get_int(t, "num_experts_per_tok", where);
        c.moe_intermediate_size = cfg::get_int(t, "moe_intermediate_size", where);
        c.shared_expert_intermediate_size = cfg::get_int_or(t, "shared_expert_intermediate_size", 0, where);
        c.norm_topk_prob = cfg::get_bool_or(t, "norm_topk_prob", true, where);
        XS_CHECK(c.experts_per_token >= 1 && c.experts_per_token <= c.num_experts, where, ": num_experts_per_tok ",
                 c.experts_per_token, " of ", c.num_experts, " experts");
        if (cfg::get_int_or(t, "decoder_sparse_step", 1, where) != 1) fail(where, ": decoder_sparse_step is not supported");
    } else {
        c.intermediate_size = cfg::get_int(t, "intermediate_size", where);
    }
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
    const std::string order =
        t.contains("linear_value_head_order") ? cfg::get_string(t, "linear_value_head_order", where) : "grouped";
    if (order == "tiled") {
        c.value_head_order = ValueHeadOrder::Tiled;
    } else if (order != "grouped") {
        fail(where, ": linear_value_head_order '", order, "' is not supported (supported: grouped, tiled)");
    }
    c.mtp_layers = cfg::get_int_or(t, "mtp_num_hidden_layers", 0, where);
    if (cfg::get_bool_or(t, "mtp_use_dedicated_embeddings", false, where)) {
        fail(where, ": mtp_use_dedicated_embeddings is not supported (the MTP head reads the target's embedding)");
    }

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
    // Multimodal RoPE gives text tokens the same position on all three axes, which is plain RoPE: only an image's
    // tokens read the axes the sections assign. Sections that do not cover the rotated frequencies, or that run in
    // turn (no mrope_interleaved), leave none, which the vision tower refuses (Decoder).
    if (rope.contains("mrope_section") && cfg::get_bool_or(rope, "mrope_interleaved", false, rw)) {
        const std::vector<int32_t> section = cfg::get_int_list(rope, "mrope_section", rw);
        const int64_t pairs = c.rotary_dim / 2;
        if (section.size() == 3 && section[0] + section[1] + section[2] == pairs) {
            c.mrope_axes = blocks::interleaved_mrope_axes(section, pairs, rw);
        }
    }

    XS_CHECK(c.num_heads % c.num_kv_heads == 0, where, ": num_attention_heads ", c.num_heads,
             " is not a multiple of num_key_value_heads ", c.num_kv_heads);
    XS_CHECK(c.linear_value_heads % c.linear_key_heads == 0, where, ": linear_num_value_heads ", c.linear_value_heads,
             " is not a multiple of linear_num_key_heads ", c.linear_key_heads);
    XS_CHECK(c.rotary_dim > 0 && c.rotary_dim % 2 == 0 && c.rotary_dim <= c.head_dim, where, ": rotary dim ",
             c.rotary_dim, " does not fit head_dim ", c.head_dim);
    XS_CHECK(c.conv_kernel > 1, where, ": linear_conv_kernel_dim must be at least 2");
    parse_loop(config_json, c);
    return c;
}

}  // namespace xs::qwen3_5
