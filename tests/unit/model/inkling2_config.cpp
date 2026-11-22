// Inkling 2 config parsing (inkling2_loop): the Inkling-2-Mini layout and its drafter's settings (mtp_config), and
// what each refuses by name.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "models/drafters/dspark.h"
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

TEST_CASE("inkling2 config: a GGUF import's expert banks hold the gate rows, then the up rows") {
    json config = inkling2_config();
    CHECK(xs::inkling2::parse_config(config).gate_up_alternates);
    config["text_config"]["gate_up_rows"] = "halves";
    CHECK_FALSE(xs::inkling2::parse_config(config).gate_up_alternates);
    config["text_config"]["gate_up_rows"] = "alternating";
    CHECK(xs::inkling2::parse_config(config).gate_up_alternates);
    config["text_config"]["gate_up_rows"] = "pairs";
    CHECK(refuses(config, "gate_up_rows 'pairs'"));
}

namespace {

// thinkingmachines/Inkling-2-Mini's mtp_config: its drafter's settings.
json drafter_config() {
    return {{"hidden_size", 2304},        {"num_hidden_layers", 3},        {"num_attention_heads", 24},
            {"num_key_value_heads", 8},   {"head_dim", 128},               {"intermediate_size", 9216},
            {"max_position_embeddings", 1048576}, {"torch_dtype", "bfloat16"}, {"q_bias", false},
            {"o_bias", false},            {"rms_norm_eps", 1e-6},          {"rope_theta", 1000000.0},
            {"local_layer_ids", {0, 2}},  {"sliding_window_size", 1024},   {"block_size", 8},
            {"markov_rank", 256},         {"markov_head_type", "rnn_hidden"}, {"target_layer_ids", {5, 23, 41, 48}}};
}

bool drafter_refuses(const json& config, const std::string& part, int64_t hidden = 2304, int64_t visits = 48) {
    try {
        xs::dspark::parse_config(config, hidden, visits);
    } catch (const xs::Error& e) {
        return std::string(e.what()).find(part) != std::string::npos;
    }
    return false;
}

}  // namespace

TEST_CASE("inkling2 drafter config: the released mtp_config, and what it refuses by name") {
    const xs::dspark::Config c = xs::dspark::parse_config(drafter_config(), 2304, 48);
    CHECK(c.num_layers == 3);
    CHECK(c.num_heads == 24);
    CHECK(c.num_kv_heads == 8);
    CHECK(c.head_dim == 128);
    CHECK(c.intermediate_size == 9216);
    CHECK(c.block_size == 8);
    CHECK(c.markov_rank == 256);
    CHECK(c.windows == std::vector<int64_t>{1024, 0, 1024});  // the middle layer sees the whole context
    CHECK(c.target_ids == std::vector<int64_t>{5, 23, 41, 48});
    CHECK(c.rope_theta == 1e6);

    const auto with = [](const char* key, const json& value) {
        json config = drafter_config();
        config[key] = value;
        return config;
    };
    CHECK(drafter_refuses(with("markov_head_type", "bilinear"), "markov_head_type 'bilinear'"));
    CHECK(drafter_refuses(with("q_bias", true), "q_bias"));
    CHECK(drafter_refuses(with("o_bias", true), "o_bias"));
    CHECK(drafter_refuses(with("hidden_size", 1024), "hidden_size 1024 is not the target's 2304"));
    CHECK(drafter_refuses(with("local_layer_ids", {0, 3}), "local_layer_ids names layer 3"));
    // The states it reads: rising, the last one the target's count of layer visits (its final state).
    CHECK(drafter_refuses(with("target_layer_ids", {5, 23, 41, 47}), "target_layer_ids"));
    CHECK(drafter_refuses(with("target_layer_ids", {23, 5, 41, 48}), "target_layer_ids"));
    CHECK(drafter_refuses(drafter_config(), "target_layer_ids", 2304, 24));  // a target that runs each layer once
    CHECK(drafter_refuses(with("block_size", 1), "sizes do not fit"));
}

TEST_CASE("inkling2 vision config: the released vision_config, and what it refuses by name") {
    json config = inkling2_config();
    config["vision_config"] = {{"model_type", "inkling2_vision"}, {"text_hidden_size", 2304}, {"temporal_patch_size", 2},
                               {"native_patch_size", 20}, {"adapter_merge_size", 2}, {"projection_hidden_dim", 4096},
                               {"projector_bias", false}, {"hidden_size", 1024}, {"num_hidden_layers", 23},
                               {"num_attention_heads", 16}, {"mlp_ratio", 4.0}, {"posemb_grid_size", 32},
                               {"layer_norm_eps", 1e-5}};
    const xs::inkling2::VisionConfig c = xs::inkling2::parse_vision_config(config);
    CHECK(c.hidden == 1024);
    CHECK(c.layers == 23);
    CHECK(c.heads == 16);
    CHECK(c.intermediate == 4096);  // hidden * mlp_ratio
    CHECK(c.patch == 20);
    CHECK(c.merge == 2);
    CHECK(c.grid == 32);
    CHECK(c.projection == 4096);
    CHECK(c.out_hidden == 2304);
    CHECK(c.rope_theta == 10000.0);
    CHECK(c.image_token == 200054);  // <|unused_200054|>, unless config.json names another

    const auto refuses = [&](const char* key, const json& value, const char* what) {
        json changed = config;
        changed["vision_config"][key] = value;
        try {
            xs::inkling2::parse_vision_config(changed);
        } catch (const xs::Error& e) {
            return std::string(e.what()).find(what) != std::string::npos;
        }
        return false;
    };
    CHECK(refuses("projector_bias", true, "projector_bias"));
    CHECK(refuses("hidden_act", "quick_gelu", "hidden_act 'quick_gelu'"));
    CHECK(refuses("rope_parameters", {{"rope_type", "default"}, {"rope_theta", 10000.0}}, "rope_type 'default'"));
    CHECK(refuses("num_attention_heads", 15, "sizes do not fit"));
    config.erase("vision_config");
    CHECK_THROWS_WITH_AS(xs::inkling2::parse_vision_config(config), doctest::Contains("without a vision_config"),
                         xs::Error);
}
