// Llama 3's config (models/llama): the released layout (rope_theta and rope_scaling at the top) and transformers 5's
// (one rope_parameters object), Llama 3's RoPE scaling against transformers' frequencies, and refusals by name.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include "models/llama/llama.h"
#include "util/error.h"

namespace {

using nlohmann::json;

// Llama-3.2-1B-Instruct's config.json.
json released() {
    return json::parse(R"({"architectures": ["LlamaForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
        "bos_token_id": 128000, "eos_token_id": [128001, 128008, 128009], "head_dim": 64, "hidden_act": "silu",
        "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 8192, "max_position_embeddings": 131072,
        "mlp_bias": false, "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 16,
        "num_key_value_heads": 8, "pretraining_tp": 1, "rms_norm_eps": 1e-05,
        "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                         "original_max_position_embeddings": 8192, "rope_type": "llama3"},
        "rope_theta": 500000.0, "tie_word_embeddings": true, "vocab_size": 128256})");
}

}  // namespace

TEST_SUITE("llama config") {
    TEST_CASE("the released layout and transformers 5's give the same config") {
        const xs::llama::Config c = xs::llama::parse_config(released());
        CHECK(c.num_layers == 16);
        CHECK(c.num_kv_heads == 8);
        CHECK(c.head_dim == 64);
        CHECK(c.tie_word_embeddings);
        CHECK(c.rope_theta == 500000.0);
        REQUIRE(c.llama3_rope.has_value());
        CHECK(c.llama3_rope->factor == 32.0);
        CHECK(c.llama3_rope->original_max_position == 8192.0);
        json v5 = released();
        v5.erase("rope_theta");
        v5["rope_parameters"] = v5["rope_scaling"];
        v5["rope_parameters"]["rope_theta"] = 500000.0;
        v5.erase("rope_scaling");
        const xs::llama::Config d = xs::llama::parse_config(v5);
        CHECK(d.rope_theta == 500000.0);
        REQUIRE(d.llama3_rope.has_value());
        CHECK(d.llama3_rope->high_freq_factor == 4.0);
        json plain = released();
        plain.erase("rope_scaling");
        plain.erase("head_dim");
        const xs::llama::Config p = xs::llama::parse_config(plain);
        CHECK_FALSE(p.llama3_rope.has_value());
        CHECK(p.head_dim == 64);  // hidden_size / num_attention_heads
    }

    // transformers 5.23's _compute_llama3_parameters for this config (float32), at a few of the 32 frequencies: kept
    // (0, 14), blended (15, 16, 17), divided by the factor (18, 31).
    TEST_CASE("Llama 3's RoPE scaling: the frequencies transformers computes") {
        const xs::llama::Config c = xs::llama::parse_config(released());
        const xs::HostTensor f = xs::blocks::llama3_rope_inv_freq(c.rope_theta, c.head_dim, *c.llama3_rope);
        REQUIRE(f.numel() == 32);
        const float* v = f.data<float>();
        const std::pair<int, double> expected[] = {
            {0, 1.0},          {14, 0.0032114461064338684}, {15, 0.0012905480107292533}, {16, 0.000429556705057621},
            {17, 9.708286233944818e-05}, {18, 1.9461638657958247e-05}, {31, 9.418306490260875e-08}};
        for (const auto& [i, want] : expected) {
            CAPTURE(i);
            CHECK(v[i] == doctest::Approx(want).epsilon(1e-5));
        }
    }

    TEST_CASE("refused by name: biases, other activations and RoPE types, a partial rotary factor") {
        const auto refused = [](const json& c) {
            try {
                xs::llama::parse_config(c);
            } catch (const xs::Error& e) {
                return std::string(e.what());
            }
            return std::string("(accepted)");
        };
        json c = released();
        c["attention_bias"] = true;
        CHECK(refused(c).find("attention_bias") != std::string::npos);
        c = released();
        c["mlp_bias"] = true;
        CHECK(refused(c).find("mlp_bias") != std::string::npos);
        c = released();
        c["hidden_act"] = "gelu";
        CHECK(refused(c).find("hidden_act") != std::string::npos);
        c = released();
        c["rope_scaling"] = {{"rope_type", "yarn"}, {"factor", 4.0}};
        CHECK(refused(c).find("'yarn'") != std::string::npos);
        c = released();
        c["partial_rotary_factor"] = 0.5;
        CHECK(refused(c).find("partial_rotary_factor") != std::string::npos);
        c = released();
        c["rope_scaling"].erase("low_freq_factor");
        CHECK(refused(c).find("low_freq_factor") != std::string::npos);
    }
}
