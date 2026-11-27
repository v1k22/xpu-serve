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

TEST_CASE("qwen3_5 config: the multimodal RoPE axis of each rotated frequency, interleaved") {
    const xs::qwen3_5::Config c = xs::qwen3_5::parse_config(qwen38_config());
    REQUIRE(c.mrope_axes.size() == 32);
    // Sections (11, 11, 10): height at 1, 4, ..., 31, width at 2, 5, ..., 29, time elsewhere (Qwen3-VL's layout).
    for (int32_t i = 0; i < 32; ++i) {
        CAPTURE(i);
        const int32_t axis = i % 3 == 1 && i < 33 ? 1 : i % 3 == 2 && i < 30 ? 2 : 0;
        CHECK(c.mrope_axes[static_cast<size_t>(i)] == axis);
    }
    // Sections in turn, or ones that do not cover the rotated frequencies, leave none (text reads one position).
    json in_turn = qwen38_config();
    in_turn["text_config"]["rope_parameters"].erase("mrope_interleaved");
    CHECK(xs::qwen3_5::parse_config(in_turn).mrope_axes.empty());
    json short_sections = qwen38_config();
    short_sections["text_config"]["rope_parameters"]["mrope_section"] = {8, 8, 8};
    CHECK(xs::qwen3_5::parse_config(short_sections).mrope_axes.empty());
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

namespace {

// SenseNova-Loop-9B's loop_settings and mhc on a 32-layer text_config.
json sensenova_loop_config() {
    json config = qwen38_config();
    config["model_type"] = "sensenova_loop";
    json types = json::array();
    for (int i = 0; i < 32; ++i)
        types.push_back((i + 1) % 4 == 0 ? "full_attention" : "linear_attention");
    config["text_config"]["num_hidden_layers"] = 32;
    config["text_config"]["layer_types"] = types;
    config["loop_settings"] = {{"counts", {2}}, {"end", 28}, {"eval_count", 2}, {"start", 4}};
    config["mhc"] = {{"end", 28}, {"fixed_identity", false}, {"iterations", 20}, {"pre_scale", 1.0}, {"start", 4},
                     {"streams", 2}};
    return config;
}

}  // namespace

TEST_CASE("qwen3_5 config: SenseNova Loop's plan (layers 4 to 27 twice) and hyper-connections") {
    const xs::qwen3_5::Config c = xs::qwen3_5::parse_config(sensenova_loop_config());
    REQUIRE(c.plan.size() == 56);
    CHECK(c.plan[3] == 3);
    CHECK(c.plan[4] == 4);
    CHECK(c.plan[27] == 27);
    CHECK(c.plan[28] == 4);  // the second pass
    CHECK(c.plan[51] == 27);
    CHECK(c.plan[52] == 28);
    CHECK(c.plan[55] == 31);
    CHECK(c.looped());
    CHECK(c.mhc.start == 4);
    CHECK(c.mhc.end == 28);
    CHECK(c.mhc.streams == 2);
    CHECK(c.mhc.iterations == 20);
    CHECK(c.mhc.pre_scale == 1.0f);
    // a plain Qwen3.5 checkpoint has neither
    const xs::qwen3_5::Config q = xs::qwen3_5::parse_config(qwen38_config());
    CHECK(q.plan.empty());
    CHECK(q.mhc.streams == 0);
    CHECK_FALSE(q.looped());
}

TEST_CASE("qwen3_5 config: SenseNova Loop settings it does not implement, refused by name") {
    struct Case {
        const char* object;
        const char* key;
        json value;
        const char* message;
    };
    for (const Case& k : {Case{"loop_settings", "counts", json::array({1, 2}), "counts"},
                          Case{"loop_settings", "end", 40, "loop_settings"},
                          Case{"loop_settings", "eval_count", 0, "loop_settings"},
                          Case{"mhc", "streams", 3, "streams"},
                          Case{"mhc", "iterations", 0, "iterations"},
                          Case{"mhc", "fixed_identity", true, "fixed_identity"},
                          Case{"mhc", "pre_scale", -1.0, "pre_scale"},
                          Case{"mhc", "end", 29, "cross"},
                          Case{"mhc", "alpha", 1.0, "alpha"}}) {
        CAPTURE(k.object);
        CAPTURE(k.key);
        json config = sensenova_loop_config();
        config[k.object][k.key] = k.value;
        CHECK_THROWS_WITH_AS(xs::qwen3_5::parse_config(config), doctest::Contains(k.message), xs::Error);
    }
    json missing = sensenova_loop_config();
    missing.erase("loop_settings");
    CHECK_THROWS_WITH_AS(xs::qwen3_5::parse_config(missing), doctest::Contains("loop_settings"), xs::Error);
    json other = qwen38_config();
    other["mhc"] = sensenova_loop_config()["mhc"];
    CHECK_THROWS_WITH_AS(xs::qwen3_5::parse_config(other), doctest::Contains("sensenova_loop"), xs::Error);
}
