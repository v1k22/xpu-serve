#include <algorithm>

#include "models/config_util.h"
#include "models/inkling2/inkling2.h"
#include "util/error.h"

namespace xs::inkling2 {
namespace {

// A switch the released checkpoints all set one way: any other value is math this code does not have.
void require(const nlohmann::json& t, const char* key, bool expected, const std::string& where) {
    if (cfg::get_bool_or(t, key, expected, where) != expected) {
        fail(where, ": ", key, " ", expected ? "false" : "true", " is not supported");
    }
}

// The released checkpoints store their experts in NVFP4 and MXFP8 (models/inkling2 reads both); a pack of one keeps
// the same config.json.
void check_quantization(const nlohmann::json& config) {
    if (!config.contains("quantization_config") || config.at("quantization_config").is_null()) return;
    const nlohmann::json& q = config.at("quantization_config");
    const std::string where = "config.json quantization_config";
    const std::string method = cfg::get_string(q, "quant_method", where);
    const std::string algo = q.contains("quant_algo") ? cfg::get_string(q, "quant_algo", where) : "";
    if (method != "modelopt" || algo != "NVFP4") {
        fail(where, ": quant_method '", method, "' with quant_algo '", algo,
             "' is not supported (inkling2_loop: modelopt's NVFP4, the experts in NVFP4 or MXFP8)");
    }
}

}  // namespace

Config parse_config(const nlohmann::json& config_json) {
    check_quantization(config_json);
    if (!config_json.contains("text_config") || !config_json.at("text_config").is_object()) {
        fail("config.json: an inkling2_loop checkpoint without a text_config");
    }
    const nlohmann::json& t = config_json.at("text_config");
    const std::string where = "config.json text_config";
    require(t, "q_bias", false, where);
    require(t, "o_bias", false, where);
    require(t, "use_value_norm_no_gamma", true, where);
    require(t, "attn_res_normalize_values", true, where);
    require(t, "use_gate_bias", true, where);
    if (cfg::get_int_or(t, "n_shared_experts", 0, where) != 0) fail(where, ": shared experts are not supported");

    Config c;
    c.vocab_size = cfg::get_int(t, "vocab_size", where);
    c.output_size = cfg::get_int_or(t, "unpadded_vocab_size", c.vocab_size, where);
    c.hidden_size = cfg::get_int(t, "hidden_size", where);
    c.num_layers = cfg::get_int(t, "num_hidden_layers", where);
    c.norm_eps = cfg::get_float(t, "rms_norm_eps", where);
    c.max_position = cfg::get_int(t, "model_max_length", where);
    c.num_heads = cfg::get_int(t, "num_attention_heads", where);
    c.num_kv_heads = cfg::get_int(t, "num_key_value_heads", where);
    c.head_dim = cfg::get_int(t, "head_dim", where);
    c.sliding_window = cfg::get_int(t, "sliding_window_size", where);
    c.rel_width = cfg::get_int(t, "d_rel", where);
    c.rel_extent = cfg::get_int(t, "rel_extent", where);
    c.qk_clip = cfg::get_float(t, "qk_l2_clip_threshold", where);
    c.rel_clip = cfg::get_float(t, "rel_l2_clip_threshold", where);
    c.log_scale_alpha = cfg::get_float(t, "log_scaling_alpha", where);
    c.log_scale_floor = cfg::get_float(t, "log_scaling_n_floor", where);
    c.conv_kernel = cfg::get_int(t, "sconv_kernel_size", where);
    c.block_layers = cfg::get_int(t, "n_layers_per_attn_res_block", where);
    c.experts = cfg::get_int(t, "n_routed_experts", where);
    c.experts_per_token = cfg::get_int(t, "num_experts_per_tok", where);
    c.expert_width = cfg::get_int(t, "intermediate_size", where);
    c.route_scale = cfg::get_float(t, "route_scale", where);
    c.gate_softcap = cfg::get_float(t, "mlp_gate_softcap", where);
    c.up_softcap = cfg::get_float(t, "mlp_up_softcap", where);
    c.logit_softcap = cfg::get_float(t, "final_logit_softcapping", where);

    XS_CHECK(c.num_layers > 0 && c.hidden_size > 0 && c.head_dim > 0 && c.num_kv_heads > 0 &&
                 c.num_heads % c.num_kv_heads == 0 && c.output_size > 0 && c.output_size <= c.vocab_size,
             where, ": sizes do not fit (", c.num_layers, " layers, hidden ", c.hidden_size, ", ", c.num_heads,
             " heads over ", c.num_kv_heads, " of ", c.head_dim, ", ", c.output_size, " of ", c.vocab_size, " tokens)");
    // The sliding layers' own head counts: transformers reads them apart, the released checkpoints set them equal.
    for (const auto& [key, same] : {std::pair{"swa_num_attention_heads", c.num_heads},
                                    std::pair{"swa_num_key_value_heads", c.num_kv_heads},
                                    std::pair{"swa_head_dim", c.head_dim}}) {
        if (const int64_t v = cfg::get_int_or(t, key, same, where); v != same) {
            fail(where, ": ", key, " ", v, " differs from the full layers' ", same, ", which is not supported");
        }
    }
    XS_CHECK(c.sliding_window > 0 && c.rel_extent > 0 && c.rel_width > 0 && c.qk_clip > 0 && c.rel_clip > 0 &&
                 c.log_scale_floor > 0 && c.conv_kernel > 1 && c.block_layers > 0,
             where, ": window ", c.sliding_window, ", rel_extent ", c.rel_extent, ", d_rel ", c.rel_width,
             ", conv kernel ", c.conv_kernel, " or n_layers_per_attn_res_block ", c.block_layers, " is out of range");
    XS_CHECK(c.experts > 0 && c.experts_per_token > 0 && c.experts_per_token <= c.experts && c.expert_width > 0 &&
                 c.gate_softcap > 0 && c.up_softcap > 0 && c.logit_softcap > 0,
             where, ": ", c.experts_per_token, " of ", c.experts, " experts of width ", c.expert_width,
             " or a softcap is out of range");

    c.sliding.assign(static_cast<size_t>(c.num_layers), false);
    for (const int32_t i : cfg::get_int_list(t, "local_layer_ids", where)) {
        XS_CHECK(i >= 0 && i < c.num_layers, where, ": local_layer_ids names layer ", i, " of ", c.num_layers);
        c.sliding[static_cast<size_t>(i)] = true;
    }

    c.loop.start = c.loop.end = c.num_layers;  // no loop_config: nothing loops
    if (t.contains("loop_config") && !t.at("loop_config").is_null()) {
        const nlohmann::json& l = t.at("loop_config");
        const std::string lw = where + " loop_config";
        if (cfg::get_bool_or(l, "is_varloop", false, lw)) {
            fail(lw, ": a checkpoint with a variable number of passes (is_varloop) is not supported");
        }
        c.loop.start = cfg::get_int(l, "core_start", lw);
        c.loop.end = cfg::get_int(l, "core_end", lw);
        c.loop.passes = cfg::get_int(l, "num_passes", lw);
        c.loop.gate_rank = cfg::get_int(l, "gate_rank", lw);
        // Each pass restarts from the blocks before the loop, and the passes after the first read the blocks a pass
        // appended after its first: both ends sit on block boundaries.
        XS_CHECK(c.loop.passes >= 1 && c.loop.start > 0 && c.loop.start < c.loop.end && c.loop.end <= c.num_layers &&
                     c.loop.start % c.block_layers == 0 && c.loop.end % c.block_layers == 0 && c.loop.gate_rank > 0,
                 lw, ": layers ", c.loop.start, " to ", c.loop.end, " of ", c.num_layers, " in ", c.loop.passes,
                 " passes, blocks of ", c.block_layers, " layers, gate rank ", c.loop.gate_rank, " do not fit");
    }
    return c;
}

}  // namespace xs::inkling2
