// Qwen3-Omni's config (qwen3_omni over qwen3, qwen3_audio and qwen3_vision): the released 30B-A3B thinker's layout,
// Qwen2VLImageProcessor's sizes and pixel bounds, and what is refused by name.
#include <doctest/doctest.h>

#include <fstream>
#include <nlohmann/json.hpp>

#include "models/model.h"
#include "models/qwen3/qwen3.h"
#include "models/qwen3_omni/qwen3_omni.h"
#include "models/qwen3_vision/qwen3_vision.h"
#include "tiny_checkpoints.h"
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
    const json vision = {{"depth", 27}, {"hidden_size", 1152}, {"intermediate_size", 4304}, {"num_heads", 16},
                         {"out_hidden_size", 2048}, {"patch_size", 16}, {"spatial_merge_size", 2},
                         {"temporal_patch_size", 2}, {"in_channels", 3}, {"hidden_act", "gelu_pytorch_tanh"},
                         {"deepstack_visual_indexes", {8, 16, 24}}, {"num_position_embeddings", 2304}};
    return {{"model_type", "qwen3_omni_moe"},
            {"thinker_config",
             {{"text_config", text}, {"audio_config", audio}, {"vision_config", vision}, {"audio_token_id", 151675},
              {"image_token_id", 151655}}}};
}

// Transformers' smart_resize (Qwen2VLImageProcessor), as Python computes it: round() takes halves to even.
std::pair<int64_t, int64_t> resized(int64_t h, int64_t w, int64_t max_pixels = 12845056) {
    return xs::qwen3_vision::smart_resize(h, w, 32, 3136, max_pixels);
}

}  // namespace

TEST_SUITE("qwen3_omni config") {
    TEST_CASE("the 30B-A3B thinker: a Qwen3 decoder of 48 sparse layers, interleaved multimodal RoPE, the tower") {
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
        // mrope_section [24, 20, 20], interleaved: frequency i reads height when i % 3 == 1 and width when
        // i % 3 == 2, up to three times its section, else time.
        REQUIRE(c.text.mrope_axes.size() == 64);
        const std::vector<int32_t> head(c.text.mrope_axes.begin(), c.text.mrope_axes.begin() + 6);
        CHECK(head == std::vector<int32_t>{0, 1, 2, 0, 1, 2});
        CHECK(c.text.mrope_axes[58] == 1);
        CHECK(c.text.mrope_axes[59] == 2);
        for (size_t i = 60; i < 64; ++i) CHECK(c.text.mrope_axes[i] == 0);
        REQUIRE(c.vision.has_value());
        CHECK(c.vision->depth == 27);
        CHECK(c.vision->deepstack == std::vector<int64_t>{8, 16, 24});
        CHECK(c.image_token == 151655);
    }

    TEST_CASE("smart_resize: a side a multiple of 32, rounded half to even, within the pixel bounds") {
        CHECK(resized(360, 640) == std::pair<int64_t, int64_t>{352, 640});
        CHECK(resized(80, 80) == std::pair<int64_t, int64_t>{64, 64});     // 2.5 rounds to 2, not 3
        CHECK(resized(48, 200) == std::pair<int64_t, int64_t>{64, 192});   // 1.5 rounds to 2
        CHECK(resized(30, 30) == std::pair<int64_t, int64_t>{64, 64});     // up to the least pixels
        CHECK(resized(2048, 2048, 1048576) == std::pair<int64_t, int64_t>{1024, 1024});
        CHECK_THROWS_WITH_AS(resized(1, 201), doctest::Contains("aspect ratio"), xs::Error);
    }

    TEST_CASE("the processor's pixel bounds: preprocessor_config.json's, lowered by --image-max-pixels") {
        const xs::test::TempDir dir("qwen3_omni_processor");
        std::filesystem::create_directories(dir.path);
        const xs::qwen3_vision::Config tower = *xs::qwen3_omni::parse_config(omni_config()).vision;
        std::ofstream(dir.path / "preprocessor_config.json")
            << json{{"min_pixels", 3136}, {"max_pixels", 12845056}, {"patch_size", 16}, {"merge_size", 2},
                    {"temporal_patch_size", 2}, {"image_mean", {0.5, 0.5, 0.5}}, {"image_std", {0.5, 0.5, 0.5}}}
                   .dump();
        CHECK(xs::qwen3_vision::processor_spec(dir.path, tower).max_pixels == 12845056);
        CHECK(xs::qwen3_vision::processor_spec(dir.path, tower, 1048576).max_pixels == 1048576);
        CHECK(xs::qwen3_vision::processor_spec(dir.path, tower, 1 << 30).max_pixels == 12845056);
        CHECK_THROWS_WITH_AS(xs::qwen3_vision::processor_spec(dir.path, tower, 1000), doctest::Contains("below"),
                             xs::Error);
        std::ofstream(dir.path / "preprocessor_config.json") << json{{"patch_size", 14}}.dump();
        CHECK_THROWS_WITH_AS(xs::qwen3_vision::processor_spec(dir.path, tower), doctest::Contains("patch_size"),
                             xs::Error);
        // A model whose images are not sized by a pixel budget refuses the bound, before reading any weight.
        std::ofstream(dir.path / "config.json") << json{{"model_type", "gemma4"}}.dump();
        CHECK_THROWS_WITH_AS(xs::load_model(dir.path, xs::LoadOptions{.image_max_pixels = 1048576}),
                             doctest::Contains("pixel budget"), xs::Error);
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
        c["thinker_config"]["vision_config"]["hidden_act"] = "silu";
        refused(c, "hidden_act");
        c = omni_config();
        c["thinker_config"]["vision_config"]["out_hidden_size"] = 1024;
        refused(c, "out_hidden_size");
        c = omni_config();
        c["thinker_config"]["vision_config"]["deepstack_visual_indexes"] = {8, 30};
        refused(c, "deepstack_visual_indexes");
        c = omni_config();
        c["thinker_config"].erase("audio_config");
        refused(c, "audio_config");
        c = omni_config();
        c.erase("thinker_config");
        refused(c, "thinker_config");
    }
}
