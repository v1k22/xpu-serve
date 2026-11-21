// Inkling 2 config parsing (inkling2_loop): the Inkling-2-Mini layout, and what it refuses by name.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "models/inkling2/inkling2.h"
#include "util/error.h"

namespace {

using nlohmann::json;

// thinkingmachines/Inkling-2-Mini's config.json, trimmed to what the parser reads.
json inkling2_config() {
    json local = json::array();
    for (int i = 0; i < 24; ++i) {
        if ((i + 1) % 6 != 0) local.push_back(i);
    }
    return {{"model_type", "inkling2_loop"},
            {"eos_token_id", 200006},
            {"quantization_config", {{"quant_method", "modelopt"}, {"quant_algo", "NVFP4"}}},
            {"text_config",
             {{"model_max_length", 1048576}, {"hidden_size", 2304}, {"num_hidden_layers", 24}, {"vocab_size", 201024},
              {"unpadded_vocab_size", 200058}, {"num_attention_heads", 24}, {"num_key_value_heads", 8},
              {"head_dim", 128}, {"swa_num_attention_heads", 24}, {"swa_num_key_value_heads", 8}, {"swa_head_dim", 128},
              {"d_rel", 16}, {"rel_extent", 1024}, {"sliding_window_size", 1024}, {"q_bias", false}, {"o_bias", false},
              {"log_scaling_n_floor", 8192}, {"log_scaling_alpha", 0.1}, {"rms_norm_eps", 1e-6},
              {"use_value_norm_no_gamma", true}, {"qk_l2_clip_threshold", 22.627416997969522},
              {"rel_l2_clip_threshold", 8.0}, {"local_layer_ids", local}, {"sconv_kernel_size", 4},
              {"final_logit_softcapping", 30.0}, {"n_layers_per_attn_res_block", 6},
              {"attn_res_normalize_values", true}, {"n_routed_experts", 128}, {"num_experts_per_tok", 8},
              {"n_shared_experts", 0}, {"intermediate_size", 1152},
              {"mlp_gate_softcap", 4.0}, {"mlp_up_softcap", 25.0}, {"route_scale", 8.0}, {"use_gate_bias", true},
              {"loop_config",
               {{"core_start", 6}, {"core_end", 18}, {"num_passes", 3}, {"gate_rank", 64}, {"is_varloop", false}}}}}};
}

// Whether parsing `config` is refused with a message that names `part`.
bool refuses(const json& config, const std::string& part) {
    try {
        xs::inkling2::parse_config(config);
    } catch (const xs::Error& e) {
        return std::string(e.what()).find(part) != std::string::npos;
    }
    return false;
}

}  // namespace

TEST_CASE("inkling2 config: the Inkling-2-Mini layout") {
    const xs::inkling2::Config c = xs::inkling2::parse_config(inkling2_config());
    CHECK(c.num_layers == 24);
    CHECK(c.vocab_size == 201024);
    CHECK(c.output_size == 200058);
    CHECK(c.num_heads == 24);
    CHECK(c.num_kv_heads == 8);
    CHECK(c.sliding[0]);
    CHECK(c.sliding[4]);
    CHECK_FALSE(c.sliding[5]);  // every sixth layer attends over the whole context
    CHECK_FALSE(c.sliding[23]);
    CHECK(c.sliding_window == 1024);
    CHECK(c.rel_width == 16);
    CHECK(c.rel_extent == 1024);
    CHECK(c.qk_clip == doctest::Approx(22.627417));
    CHECK(c.conv_kernel == 4);
    CHECK(c.block_layers == 6);
    CHECK(c.experts == 128);
    CHECK(c.experts_per_token == 8);
    CHECK(c.expert_width == 1152);
    CHECK(c.route_scale == doctest::Approx(8.0));
    CHECK(c.loop.start == 6);
    CHECK(c.loop.end == 18);
    CHECK(c.loop.passes == 3);
    CHECK(c.loop.gate_rank == 64);
    CHECK(c.applications() == 48);  // 6 + 3 * 12 + 6
}

TEST_CASE("inkling2 config: without a loop every layer runs once; the padded rows default to none") {
    json config = inkling2_config();
    config["text_config"].erase("loop_config");
    config["text_config"].erase("unpadded_vocab_size");
    config.erase("quantization_config");
    const xs::inkling2::Config c = xs::inkling2::parse_config(config);
    CHECK(c.loop.passes == 1);
    CHECK(c.loop.start == 24);
    CHECK(c.applications() == 24);
    CHECK(c.output_size == 201024);
}

TEST_CASE("inkling2 config: refusals by name") {
    const auto with = [](const char* key, json value) {
        json config = inkling2_config();
        config["text_config"][key] = std::move(value);
        return config;
    };
    CHECK(refuses(with("q_bias", true), "q_bias"));
    CHECK(refuses(with("o_bias", true), "o_bias"));
    CHECK(refuses(with("use_value_norm_no_gamma", false), "use_value_norm_no_gamma"));
    CHECK(refuses(with("attn_res_normalize_values", false), "attn_res_normalize_values"));
    CHECK(refuses(with("use_gate_bias", false), "use_gate_bias"));
    CHECK(refuses(with("n_shared_experts", 2), "shared experts"));
    CHECK(refuses(with("swa_head_dim", 64), "swa_head_dim"));
    CHECK(refuses(with("swa_num_key_value_heads", 4), "swa_num_key_value_heads"));
    CHECK(refuses(with("num_key_value_heads", 7), "sizes do not fit"));
    CHECK(refuses(with("num_experts_per_tok", 200), "experts"));
    CHECK(refuses(with("local_layer_ids", json::array({0, 1, 24})), "local_layer_ids"));
    CHECK(refuses(with("unpadded_vocab_size", 300000), "sizes do not fit"));

    json config = inkling2_config();
    config["text_config"]["loop_config"]["is_varloop"] = true;
    CHECK(refuses(config, "is_varloop"));
    config = inkling2_config();
    config["text_config"]["loop_config"]["core_start"] = 4;  // not on a block boundary
    CHECK(refuses(config, "loop_config"));
    config = inkling2_config();
    config["text_config"]["loop_config"]["core_end"] = 30;
    CHECK(refuses(config, "loop_config"));
    config = inkling2_config();
    config["text_config"]["loop_config"]["core_start"] = 0;  // the first layer has no block before it
    CHECK(refuses(config, "loop_config"));

    config = inkling2_config();
    config["quantization_config"]["quant_algo"] = "FP8";
    CHECK(refuses(config, "quant_algo 'FP8'"));
    config["quantization_config"] = {{"quant_method", "bitsandbytes"}};
    CHECK(refuses(config, "bitsandbytes"));
    config = inkling2_config();
    config.erase("text_config");
    CHECK(refuses(config, "text_config"));
}
