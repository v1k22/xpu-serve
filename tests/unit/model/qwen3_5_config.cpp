// Qwen3.5-family config parsing (qwen3_5): the Qwen3.8-27B layout, and what it refuses by name.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "models/qwen3_5/qwen3_5.h"
#include "util/error.h"

namespace {

using nlohmann::json;

// The text_config of Qwen/Qwen3.8-27B, trimmed to what the parser reads.
json qwen38_config() {
    json t = {{"vocab_size", 248320}, {"hidden_size", 5120}, {"intermediate_size", 17408}, {"num_hidden_layers", 64},
              {"rms_norm_eps", 1e-6}, {"max_position_embeddings", 262144}, {"num_attention_heads", 24},
              {"num_key_value_heads", 4}, {"head_dim", 256}, {"attn_output_gate", true}, {"linear_num_key_heads", 16},
              {"linear_num_value_heads", 48}, {"linear_key_head_dim", 128}, {"linear_value_head_dim", 128},
              {"linear_conv_kernel_dim", 4}, {"hidden_act", "silu"}, {"output_gate_type", "swish"},
              {"tie_word_embeddings", false}, {"mtp_num_hidden_layers", 1}, {"mtp_use_dedicated_embeddings", false},
              {"rope_parameters",
               {{"mrope_interleaved", true}, {"mrope_section", {11, 11, 10}}, {"partial_rotary_factor", 0.25},
                {"rope_theta", 10000000}, {"rope_type", "default"}}}};
    json types = json::array();
    for (int i = 0; i < 64; ++i)
        types.push_back((i + 1) % 4 == 0 ? "full_attention" : "linear_attention");
    t["layer_types"] = types;
    return {{"model_type", "qwen3_5"}, {"text_config", t}};
}

}  // namespace

TEST_CASE("qwen3_5 config: the Qwen3.8-27B layout") {
    const xs::qwen3_5::Config c = xs::qwen3_5::parse_config(qwen38_config());
    CHECK(c.num_layers == 64);
    CHECK(c.layer_kinds[2] == xs::qwen3_5::LayerKind::Linear);
    CHECK(c.layer_kinds[3] == xs::qwen3_5::LayerKind::Full);
    CHECK(c.rotary_dim == 64);  // a quarter of head_dim 256
    CHECK(c.rope_theta == doctest::Approx(1e7));
    CHECK(c.attn_output_gate);
    CHECK(c.linear_value_heads == 3 * c.linear_key_heads);
    CHECK(c.mtp_layers == 1);
}

TEST_CASE("qwen3_5 config: refuses what it does not implement, by name") {
    struct Case {
        const char* key;
        json value;
    };
    for (const Case& k : {Case{"output_gate_type", "sigmoid"}, Case{"hidden_act", "gelu"}, Case{"attention_bias", true},
                          Case{"linear_num_value_heads", 40}, Case{"mtp_use_dedicated_embeddings", true}}) {
        CAPTURE(k.key);
        json config = qwen38_config();
        config["text_config"][k.key] = k.value;
        CHECK_THROWS_WITH_AS(xs::qwen3_5::parse_config(config), doctest::Contains(k.key), xs::Error);
    }
}

TEST_CASE("qwen3_5 config: linear_value_head_order, grouped unless the config says tiled") {
    json config = qwen38_config();
    CHECK(xs::qwen3_5::parse_config(config).value_head_order == xs::qwen3_5::ValueHeadOrder::Grouped);
    config["text_config"]["linear_value_head_order"] = "tiled";
    CHECK(xs::qwen3_5::parse_config(config).value_head_order == xs::qwen3_5::ValueHeadOrder::Tiled);
    config["text_config"]["linear_value_head_order"] = "grouped";
    CHECK(xs::qwen3_5::parse_config(config).value_head_order == xs::qwen3_5::ValueHeadOrder::Grouped);
    config["text_config"]["linear_value_head_order"] = "interleaved";
    CHECK_THROWS_WITH_AS(xs::qwen3_5::parse_config(config),
                         doctest::Contains("linear_value_head_order 'interleaved' is not supported"), xs::Error);
}
