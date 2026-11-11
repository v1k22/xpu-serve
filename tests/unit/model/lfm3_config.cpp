// LFM3 config parsing (lfm3): the LFM3-1.7B layout, and what it refuses by name.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "models/lfm3/lfm3.h"
#include "util/error.h"

namespace {

using nlohmann::json;
using xs::lfm3::LayerKind;

// LiquidAI/LFM3-1.7B's config.json, trimmed to what the parser reads.
json lfm3_config() {
    json c = {{"model_type", "lfm3"}, {"vocab_size", 128000}, {"hidden_size", 2048}, {"intermediate_size", 7680},
              {"num_hidden_layers", 24}, {"num_attention_heads", 16}, {"num_key_value_heads", 4}, {"head_dim", 128},
              {"norm_eps", 1e-5}, {"max_position_embeddings", 262144}, {"conv_L_cache", 7}, {"conv_bias", false},
              {"sliding_window", 256}, {"tie_word_embeddings", true}, {"use_sandwich_norm", true},
              {"rope_parameters",
               {{"full_attention",
                 {{"partial_rotary_factor", 0.5}, {"rope_theta", 1000000.0}, {"rope_type", "default"}}},
                {"sliding_attention", {{"rope_theta", 10000}, {"rope_type", "default"}}}}}};
    json types = json::array();
    for (int i = 0; i < 24; ++i)
        types.push_back(i % 4 == 2 ? "full_attention" : i % 4 == 0 && i > 0 ? "sliding_attention" : "conv");
    c["layer_types"] = types;
    return c;
}

}  // namespace

TEST_CASE("lfm3 config: the LFM3-1.7B layout") {
    const xs::lfm3::Config c = xs::lfm3::parse_config(lfm3_config());
    CHECK(c.num_layers == 24);
    CHECK(c.layer_kinds[0] == LayerKind::Conv);
    CHECK(c.layer_kinds[2] == LayerKind::Full);
    CHECK(c.layer_kinds[4] == LayerKind::Sliding);
    CHECK(c.layer_kinds[23] == LayerKind::Conv);
    CHECK(c.head_dim == 128);
    CHECK(c.sliding_window == 256);
    CHECK(c.conv_kernel == 7);
    CHECK(c.full_rope.theta == doctest::Approx(1e6));
    CHECK(c.full_rope.pairs == 32);  // half of a head's 64 pairs rotate
    CHECK(c.sliding_rope.theta == doctest::Approx(1e4));
    CHECK(c.sliding_rope.pairs == 64);
}

TEST_CASE("lfm3 config: defaults of absent keys") {
    json config = lfm3_config();
    config.erase("head_dim");
    config.erase("conv_L_cache");
    config["rope_parameters"]["full_attention"].erase("rope_type");
    const xs::lfm3::Config c = xs::lfm3::parse_config(config);
    CHECK(c.head_dim == 128);  // hidden_size / num_attention_heads
    CHECK(c.conv_kernel == 7);
    CHECK(c.full_rope.pairs == 32);
}

TEST_CASE("lfm3 config: a model without sliding layers needs neither their window nor their RoPE") {
    json config = lfm3_config();
    for (auto& type : config["layer_types"])
        if (type == "sliding_attention") type = "conv";
    config.erase("sliding_window");
    config["rope_parameters"].erase("sliding_attention");
    const xs::lfm3::Config c = xs::lfm3::parse_config(config);
    CHECK(c.sliding_window == 0);
    CHECK(c.full_rope.pairs == 32);
}

TEST_CASE("lfm3 config: refuses what it does not implement, by name") {
    struct Case {
        const char* key;
        json value;
        const char* names;  // what the error must name
    };
    for (const Case& k : {Case{"conv_bias", true, "conv_bias"}, Case{"use_sandwich_norm", false, "use_sandwich_norm"},
                          Case{"tie_word_embeddings", false, "tie_word_embeddings"},
                          Case{"tie_embedding", false, "tie_word_embeddings"},
                          Case{"num_key_value_heads", 5, "num_key_value_heads"}, Case{"head_dim", 127, "head_dim"},
                          Case{"conv_L_cache", 1, "conv_L_cache"}, Case{"sliding_window", nullptr, "sliding_window"},
                          Case{"rope_parameters", nullptr, "rope_parameters"},
                          Case{"layer_types", json::array({"conv"}), "layer_types"}}) {
        CAPTURE(k.key);
        json config = lfm3_config();
        config[k.key] = k.value;
        CHECK_THROWS_WITH_AS(xs::lfm3::parse_config(config), doctest::Contains(k.names), xs::Error);
    }
    json config = lfm3_config();
    config["layer_types"][3] = "linear_attention";
    CHECK_THROWS_WITH_AS(xs::lfm3::parse_config(config), doctest::Contains("linear_attention"), xs::Error);
    config = lfm3_config();
    config["rope_parameters"]["full_attention"]["rope_type"] = "yarn";
    CHECK_THROWS_WITH_AS(xs::lfm3::parse_config(config), doctest::Contains("yarn"), xs::Error);
    config = lfm3_config();
    config["rope_parameters"].erase("sliding_attention");
    CHECK_THROWS_WITH_AS(xs::lfm3::parse_config(config), doctest::Contains("sliding_attention"), xs::Error);
}
