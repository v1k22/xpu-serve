// ModernBERT config parsing (modernbert): an older config's keys, and what it refuses by name.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/modern_bert/modern_bert.h"
#include "util/error.h"

namespace {

using nlohmann::json;

// A small config in the form older ModernBERT configs take (global_attn_every_n_layers, global_rope_theta).
json older_config() {
    return json::parse(R"({"model_type": "modernbert", "hidden_size": 64, "num_hidden_layers": 3,
        "num_attention_heads": 2, "intermediate_size": 32, "vocab_size": 100, "max_position_embeddings": 512,
        "global_attn_every_n_layers": 3, "local_attention": 128, "global_rope_theta": 150000.0})");
}

json with(json j, const char* key, json value) {
    j[key] = std::move(value);
    return j;
}

std::string refusal(const json& j) {
    try {
        xs::modern_bert::parse_config(j, "encoder/config.json");
    } catch (const xs::Error& e) {
        return e.what();
    }
    return "accepted";
}

}  // namespace

TEST_CASE("modern_bert config: older keys read as transformers reads them") {
    const xs::modern_bert::Config c = xs::modern_bert::parse_config(older_config(), "encoder/config.json");
    CHECK(c.head_dim == 32);
    CHECK(c.local == std::vector<bool>{false, true, true});
    CHECK(c.global_rope_theta == 150000.0);
    CHECK(c.local_rope_theta == 10000.0);  // ModernBertConfig's default
    CHECK(c.local_attention == 128);
    CHECK(c.norm_eps == 1e-5);
    // rope_parameters, when present, names each kind's base.
    const xs::modern_bert::Config d = xs::modern_bert::parse_config(
        with(older_config(), "rope_parameters", {{"sliding_attention", {{"rope_theta", 20000.0}}}}), "c");
    CHECK(d.global_rope_theta == 150000.0);
    CHECK(d.local_rope_theta == 20000.0);
}

TEST_CASE("modern_bert config: what the encoder does not implement is refused by name") {
    CHECK(refusal(with(older_config(), "norm_bias", true)) ==
          "encoder/config.json: norm_bias (a bias in the encoder) is not supported");
    CHECK(refusal(with(older_config(), "mlp_bias", true)) ==
          "encoder/config.json: mlp_bias (a bias in the encoder) is not supported");
    CHECK(refusal(with(older_config(), "hidden_activation", "silu")) ==
          "encoder/config.json: hidden_activation 'silu' is not supported (supported: gelu)");
    CHECK(refusal(with(older_config(), "rope_parameters", {{"full_attention", {{"rope_type", "yarn"}}}})) ==
          "encoder/config.json rope_parameters.full_attention: rope_type 'yarn' is not supported (supported: default)");
    CHECK(refusal(with(older_config(), "rope_scaling", {{"type", "linear"}})) ==
          "encoder/config.json: rope_scaling is not supported");
    CHECK(refusal(with(older_config(), "layer_types", {"full_attention", "chunked_attention", "full_attention"})) ==
          "encoder/config.json: layer type 'chunked_attention' is not supported (full_attention, sliding_attention "
          "are)");
    CHECK(refusal(with(older_config(), "model_type", "bert")) ==
          "encoder/config.json: model_type 'bert' is not ModernBERT's (modernbert)");
    CHECK(refusal(with(older_config(), "head_dim", 16)) ==
          "encoder/config.json: head_dim 16 is not hidden_size / num_attention_heads");
}
