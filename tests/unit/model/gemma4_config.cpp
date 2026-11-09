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

TEST_CASE("gemma4 config: 26B-A4B layout (experts beside the MLP, K = V global layers)") {
    auto j = e2b_config();
    auto& t = j["text_config"];
    t.update({{"hidden_size", 2816}, {"intermediate_size", 2112}, {"num_hidden_layers", 30}, {"num_attention_heads", 16},
              {"num_key_value_heads", 8}, {"num_global_key_value_heads", 2}, {"num_kv_shared_layers", 0},
              {"hidden_size_per_layer_input", 0}, {"use_double_wide_mlp", false}, {"attention_k_eq_v", true},
              {"enable_moe_block", true}, {"num_experts", 128}, {"top_k_experts", 8}, {"moe_intermediate_size", 704},
              {"sliding_window", 1024}, {"use_bidirectional_attention", "vision"}});
    nlohmann::json types = nlohmann::json::array();
    for (int i = 0; i < 30; ++i) types.push_back((i + 1) % 6 == 0 ? "full_attention" : "sliding_attention");
    t["layer_types"] = types;
    const auto c = xs::gemma4::parse_config(j);
    CHECK(c.num_experts == 128);
    CHECK(c.top_k_experts == 8);
    CHECK(c.moe_intermediate_size == 704);
    CHECK(c.k_eq_v);
    CHECK(c.layer_kv_heads(4) == 8);
    CHECK(c.layer_kv_heads(5) == 2);
    CHECK(c.bidirectional_vision);
    // The same shapes as newer transformers write them: per-layer overrides of the full-attention layers.
    t.erase("num_global_key_value_heads");
    t["global_head_dim"] = 256;
    t["per_layer_config"] = {{"5", {{"head_dim", 512}, {"num_key_value_heads", 2}}},
                             {"11", {{"head_dim", 512}, {"num_key_value_heads", 2}}}};
    const auto p = xs::gemma4::parse_config(j);
    CHECK(p.layer_head_dim(5) == 512);
    CHECK(p.layer_kv_heads(11) == 2);
    CHECK(p.layer_kv_heads(4) == 8);
    t["per_layer_config"]["4"] = {{"head_dim", 128}};
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_config(j), doctest::Contains("full-attention layers only"), xs::Error);
}

TEST_CASE("gemma4 config: unsupported variants are refused by name") {
    auto bidirectional = e2b_config();
    bidirectional["text_config"]["use_bidirectional_attention"] = "all";
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_config(bidirectional), doctest::Contains("use_bidirectional_attention"),
                         xs::Error);
    auto moe = e2b_config();
    moe["text_config"]["enable_moe_block"] = true;  // without its expert keys
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_config(moe), doctest::Contains("num_experts"), xs::Error);
    auto rope = e2b_config();
    rope["text_config"]["rope_parameters"]["sliding_attention"]["rope_type"] = "yarn";
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_config(rope), doctest::Contains("yarn"), xs::Error);
    auto missing = e2b_config();
    missing["text_config"].erase("head_dim");
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_config(missing), doctest::Contains("head_dim"), xs::Error);
}

TEST_CASE("gemma4 assistant config: every layer reads the target's KV; refusals by name") {
    auto text = e2b_config()["text_config"];
    text.update({{"hidden_size", 256}, {"intermediate_size", 2048}, {"num_hidden_layers", 4}, {"num_attention_heads", 4},
                 {"num_kv_shared_layers", 4}, {"hidden_size_per_layer_input", 0}, {"use_double_wide_mlp", false},
                 {"layer_types", {"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"}}});
    const nlohmann::json a = {{"model_type", "gemma4_assistant"}, {"backbone_hidden_size", 1536},
                              {"use_ordered_embeddings", true}, {"num_centroids", 2048},
                              {"centroid_intermediate_top_k", 32}, {"text_config", text}};
    const auto c = xs::gemma4::parse_assistant_config(a);
    CHECK(c.backbone_hidden_size == 1536);
    CHECK(c.text.num_layers == 4);
    CHECK(c.ordered_embeddings);
    CHECK(c.centroid_top_k == 32);
    auto partial = a;
    partial["text_config"]["num_kv_shared_layers"] = 2;
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_assistant_config(partial), doctest::Contains("target's KV"), xs::Error);
    auto other = a;
    other["model_type"] = "gemma4";
    CHECK_THROWS_WITH_AS(xs::gemma4::parse_assistant_config(other), doctest::Contains("gemma4_assistant"), xs::Error);
}
