// The Qwen3 decoder's config (qwen3): a mixture of experts, which of its layers are sparse, and what is refused by
// name.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "models/qwen3/qwen3.h"
#include "util/error.h"

namespace {

using nlohmann::json;

// A Qwen3 MoE text config (Qwen3-Omni's thinker's, six layers).
json moe_config() {
    return {{"vocab_size", 152064}, {"hidden_size", 2048}, {"intermediate_size", 768}, {"num_hidden_layers", 6},
            {"num_attention_heads", 32}, {"num_key_value_heads", 4}, {"head_dim", 128}, {"rms_norm_eps", 1e-6},
            {"max_position_embeddings", 65536}, {"hidden_act", "silu"}, {"num_experts", 128},
            {"num_experts_per_tok", 8}, {"moe_intermediate_size", 768}, {"norm_topk_prob", true},
            {"decoder_sparse_step", 1}, {"mlp_only_layers", json::array()}, {"rope_theta", 1000000},
            {"rope_scaling",
             {{"interleaved", true}, {"mrope_section", {24, 20, 20}}, {"rope_type", "default"}}}};
}

}  // namespace

TEST_SUITE("qwen3 config") {
    TEST_CASE("which layers are sparse: decoder_sparse_step and mlp_only_layers, as Qwen3MoeDecoderLayer") {
        json t = moe_config();
        t["decoder_sparse_step"] = 2;
        t["mlp_only_layers"] = {3};
        const xs::qwen3::Config c = xs::qwen3::parse_config(t, "text_config");
        CHECK_FALSE(c.sparse(0));
        CHECK(c.sparse(1));
        CHECK_FALSE(c.sparse(2));
        CHECK_FALSE(c.sparse(3));  // listed as dense
        CHECK(c.sparse(5));
        t["num_experts"] = 0;
        CHECK_FALSE(xs::qwen3::parse_config(t, "text_config").sparse(1));
    }

    TEST_CASE("a mixture of experts is read; a dense config has none; refusals by name") {
        const xs::qwen3::Config c = xs::qwen3::parse_config(moe_config(), "text_config");
        CHECK(c.num_experts == 128);
        CHECK(c.experts_per_token == 8);
        CHECK(c.moe_intermediate_size == 768);
        CHECK(c.norm_topk_prob);
        json dense = moe_config();
        dense.erase("num_experts");
        CHECK(xs::qwen3::parse_config(dense, "text_config").num_experts == 0);
        json bad = moe_config();
        bad["num_experts_per_tok"] = 200;
        CHECK_THROWS_WITH_AS(xs::qwen3::parse_config(bad, "text_config"), doctest::Contains("num_experts_per_tok"),
                             xs::Error);
        bad = moe_config();
        bad["shared_expert_intermediate_size"] = 512;
        CHECK_THROWS_WITH_AS(xs::qwen3::parse_config(bad, "text_config"), doctest::Contains("shared expert"),
                             xs::Error);
    }
}
