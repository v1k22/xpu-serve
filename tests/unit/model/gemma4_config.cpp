#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "models/gemma4/gemma4.h"
#include "util/error.h"

namespace {
// The text_config of google/gemma-4-E2B-it, trimmed to what the parser reads.
nlohmann::json e2b_config() {
    nlohmann::json t = {
        {"vocab_size", 262144}, {"hidden_size", 1536}, {"intermediate_size", 6144}, {"num_hidden_layers", 35},
        {"num_attention_heads", 8}, {"num_key_value_heads", 1}, {"head_dim", 256}, {"global_head_dim", 512},
        {"sliding_window", 512}, {"num_kv_shared_layers", 20}, {"hidden_size_per_layer_input", 256},
        {"vocab_size_per_layer_input", 262144}, {"use_double_wide_mlp", true}, {"rms_norm_eps", 1e-6},
        {"final_logit_softcapping", 30.0}, {"max_position_embeddings", 131072},
        {"hidden_activation", "gelu_pytorch_tanh"}, {"attention_k_eq_v", false}, {"enable_moe_block", false},
        {"rope_parameters",
         {{"full_attention", {{"partial_rotary_factor", 0.25}, {"rope_theta", 1000000.0}, {"rope_type", "proportional"}}},
          {"sliding_attention", {{"rope_theta", 10000.0}, {"rope_type", "default"}}}}}};
    nlohmann::json types = nlohmann::json::array();
    for (int i = 0; i < 35; ++i) types.push_back((i + 1) % 5 == 0 ? "full_attention" : "sliding_attention");
    t["layer_types"] = types;
    return {{"model_type", "gemma4"}, {"text_config", t}};
}
}  // namespace

TEST_CASE("gemma4 config: E2B layout") {
    const auto c = xs::gemma4::parse_config(e2b_config());
    CHECK(c.first_kv_shared_layer() == 15);
    CHECK(c.layer_head_dim(4) == 512);
    CHECK(c.layer_head_dim(3) == 256);
    CHECK(c.layer_intermediate_size(14) == 6144);
    CHECK(c.layer_intermediate_size(15) == 12288);  // double-wide MLP on KV-shared layers
    CHECK(c.kv_source_layer(15) == 13);             // last sliding layer before the shared tail
    CHECK(c.kv_source_layer(19) == 14);             // last full layer before the shared tail
    CHECK(c.rope_full.proportional);
    CHECK(c.rope_full.partial_rotary_factor == doctest::Approx(0.25));
}

TEST_CASE("gemma4 config: unsupported variants are refused by name") {
    auto moe = e2b_config();
    moe["text_config"]["enable_moe_block"] = true;
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_config(moe), doctest::Contains("mixture-of-experts"), xs::Error);
    auto keqv = e2b_config();
    keqv["text_config"]["attention_k_eq_v"] = true;
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_config(keqv), doctest::Contains("attention_k_eq_v"), xs::Error);
    auto rope = e2b_config();
    rope["text_config"]["rope_parameters"]["sliding_attention"]["rope_type"] = "yarn";
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_config(rope), doctest::Contains("yarn"), xs::Error);
    auto missing = e2b_config();
    missing["text_config"].erase("head_dim");
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_config(missing), doctest::Contains("head_dim"), xs::Error);
}
