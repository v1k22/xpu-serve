#include <algorithm>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/qwen3/qwen3.h"
#include "util/error.h"

namespace xs::qwen3 {

Config parse_config(const nlohmann::json& t, const std::string& where) {
    if (cfg::get_bool_or(t, "attention_bias", false, where)) fail(where, ": attention_bias is not supported");
    if (cfg::get_bool_or(t, "use_sliding_window", false, where)) fail(where, ": use_sliding_window is not supported");
    if (cfg::get_int_or(t, "shared_expert_intermediate_size", 0, where) > 0) {
        fail(where, ": a shared expert (shared_expert_intermediate_size) is not supported");
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
    // num_experts in the released configs; num_local_experts as transformers 5 writes them.
    for (const char* key : {"num_experts", "num_local_experts"}) {
        if (c.num_experts == 0 && t.contains(key) && !t.at(key).is_null()) c.num_experts = cfg::get_int(t, key, where);
    }
    if (c.num_experts > 0) {
        c.experts_per_token = cfg::get_int(t, "num_experts_per_tok", where);
        c.moe_intermediate_size = cfg::get_int(t, "moe_intermediate_size", where);
        c.norm_topk_prob = cfg::get_bool_or(t, "norm_topk_prob", true, where);
        c.decoder_sparse_step = cfg::get_int_or(t, "decoder_sparse_step", 1, where);
        if (t.contains("mlp_only_layers")) {
            for (const int32_t layer : cfg::get_int_list(t, "mlp_only_layers", where)) {
                c.mlp_only_layers.push_back(layer);
            }
        }
        XS_CHECK(c.experts_per_token >= 1 && c.experts_per_token <= c.num_experts && c.decoder_sparse_step >= 1, where,
                 ": num_experts_per_tok ", c.experts_per_token, " of ", c.num_experts, " experts, decoder_sparse_step ",
                 c.decoder_sparse_step);
    }

    // rope_theta at the top (transformers 4) or in rope_parameters (5); rope_scaling may name the RoPE type.
    const nlohmann::json* rope = t.contains("rope_parameters") ? &t.at("rope_parameters") : nullptr;
    const bool has_scaling = t.contains("rope_scaling") && !t.at("rope_scaling").is_null();
    const nlohmann::json* scaling = has_scaling ? &t.at("rope_scaling") : nullptr;
    c.rope_theta = rope != nullptr && rope->contains("rope_theta") ? cfg::get_float(*rope, "rope_theta", where)
                                                                    : cfg::get_float(t, "rope_theta", where);
    for (const nlohmann::json* r : {rope, scaling}) {
        if (r == nullptr) continue;
        if (r->contains("mrope_section") && c.mrope_axes.empty()) {
            const std::vector<int32_t> section = cfg::get_int_list(*r, "mrope_section", where);
            const bool plain_flag = cfg::get_bool_or(*r, "interleaved", false, where);
            const bool interleaved = cfg::get_bool_or(*r, "mrope_interleaved", plain_flag, where);
            // Interleaved (Qwen3-Omni, Qwen3-VL). Without the flag, sections in turn (Qwen2-VL), which is refused
            // when positions on three axes are fed (Decoder).
            std::vector<int32_t> axes = blocks::interleaved_mrope_axes(section, c.head_dim / 2, where);
            if (interleaved) c.mrope_axes = std::move(axes);
        }
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

bool Config::sparse(int64_t layer) const {
    return num_experts > 0 && (layer + 1) % decoder_sparse_step == 0 &&
           std::find(mlp_only_layers.begin(), mlp_only_layers.end(), layer) == mlp_only_layers.end();
}

}  // namespace xs::qwen3
