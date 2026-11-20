// Qwen3-Omni's config (qwen3_omni over qwen3 and qwen3_audio): the released 30B-A3B thinker's layout, and what is
// refused by name.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "models/qwen3/qwen3.h"
#include "models/qwen3_omni/qwen3_omni.h"
#include "util/error.h"

namespace {

using nlohmann::json;

// Qwen/Qwen3-Omni-30B-A3B-Instruct's config.json, trimmed to what the parser reads.
json omni_config() {
    const json text = {{"vocab_size", 152064}, {"hidden_size", 2048}, {"intermediate_size", 768},
                       {"num_hidden_layers", 48}, {"num_attention_heads", 32}, {"num_key_value_heads", 4},
                       {"head_dim", 128}, {"rms_norm_eps", 1e-6}, {"max_position_embeddings", 65536},
                       {"hidden_act", "silu"}, {"attention_bias", false}, {"use_sliding_window", false},
                       {"num_experts", 128}, {"num_experts_per_tok", 8}, {"moe_intermediate_size", 768},
                       {"norm_topk_prob", true}, {"decoder_sparse_step", 1}, {"mlp_only_layers", json::array()},
                       {"shared_expert_intermediate_size", 0}, {"rope_theta", 1000000},
                       {"rope_scaling",
                        {{"interleaved", true}, {"mrope_interleaved", true}, {"mrope_section", {24, 20, 20}},
                         {"rope_type", "default"}, {"type", "default"}}},
                       {"tie_word_embeddings", false}};
    const json audio = {{"d_model", 1280}, {"encoder_layers", 32}, {"encoder_attention_heads", 20},
                        {"encoder_ffn_dim", 5120}, {"num_mel_bins", 128}, {"downsample_hidden_size", 480},
                        {"output_dim", 2048}, {"n_window", 50}, {"n_window_infer", 800},
                        {"max_source_positions", 1500}, {"activation_function", "gelu"}, {"scale_embedding", false}};
    return {{"model_type", "qwen3_omni_moe"},
            {"thinker_config",
             {{"text_config", text}, {"audio_config", audio}, {"audio_token_id", 151675}, {"image_token_id", 151655}}}};
}

}  // namespace

TEST_SUITE("qwen3_omni config") {
    TEST_CASE("the 30B-A3B thinker: a Qwen3 decoder of 48 sparse layers, multimodal RoPE read as plain RoPE") {
        const xs::qwen3_omni::Config c = xs::qwen3_omni::parse_config(omni_config());
        CHECK(c.text.num_layers == 48);
        CHECK(c.text.num_experts == 128);
        CHECK(c.text.experts_per_token == 8);
        CHECK(c.text.moe_intermediate_size == 768);
        CHECK(c.text.head_dim == 128);
        CHECK(c.text.rope_theta == 1e6);
        for (int64_t i = 0; i < 48; ++i) CHECK(c.text.sparse(i));
        CHECK(c.audio.d_model == 1280);
        CHECK(c.audio.layers == 32);
        CHECK(c.audio.output_dim == 2048);
        CHECK(c.audio_token == 151675);
    }

    TEST_CASE("refused by name") {
        const auto refused = [](json config, const char* what) {
            CHECK_THROWS_WITH_AS(xs::qwen3_omni::parse_config(config), doctest::Contains(what), xs::Error);
        };
        json c = omni_config();
        c["thinker_config"]["text_config"]["shared_expert_intermediate_size"] = 512;
        refused(c, "shared expert");
        c = omni_config();
        c["thinker_config"]["text_config"]["use_sliding_window"] = true;
        refused(c, "use_sliding_window");
        c = omni_config();
        c["thinker_config"]["text_config"]["rope_scaling"]["rope_type"] = "yarn";
        refused(c, "yarn");
        c = omni_config();
        c["thinker_config"]["audio_config"]["output_dim"] = 1024;
        refused(c, "output_dim");
        c = omni_config();
        c["thinker_config"].erase("audio_config");
        refused(c, "audio_config");
        c = omni_config();
        c.erase("thinker_config");
        refused(c, "thinker_config");
    }
}
