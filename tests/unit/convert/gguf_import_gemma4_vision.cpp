// GGUF import of a Gemma 4 vision tower from llama.cpp's projector file (mmproj: general.architecture clip, projector
// type gemma4v) beside a tiny gemma4 GGUF of 26B-A4B's kind (tiny_gemma4_gguf.h): config.json as the release lays it
// out (text_config with its bidirectional image attention, vision_config, the image tokens), the release's
// processor_config.json, every tower tensor under the checkpoint's name with the file's values (the patch embedding
// put back from a convolution's layout; E2B's clipping bounds; 26B-A4B's standardization); an audio tower in the same
// file left out; the tower over the imported weights gives the rows of the checkpoint's layout (planted: the
// convolution's layout read as it is); refusals by name.
#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "convert/gguf/gguf_import.h"
#include "convert/import_test_util.h"
#include "convert/tiny_gemma4_gguf.h"
#include "convert/tiny_gemma4_projector.h"
#include "image/image.h"
#include "io/gguf.h"
#include "io/host_weights.h"
#include "io/safetensors.h"
#include "models/gemma4/gemma4.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::GgufWriter;
using xs::test::has;
using xs::test::TempDir;
using xs::test::kG4HeadDim;
using xs::test::kG4Layers;
using xs::test::kG4Patch;
using xs::test::kG4Positions;
using xs::test::kG4Tower;
using xs::test::kG4Width;

namespace {

nlohmann::json read_json(const fs::path& path) {
    std::ifstream in(path);
    return nlohmann::json::parse(in);
}

// The tiny vocabulary's ids the test spells as Gemma 4's image tokens (unused added tokens of it).
constexpr int32_t kBoi = 4, kImage = 9, kEoi = 10;

// The tiny 26B-A4B-kind model's file, its vocabulary with the image tokens, and a projector for it in `dir`; the
// projector as `change` leaves it.
template <typename Change>
gguf_import::ImportOptions files(const fs::path& dir, xs::test::TinyGemma4Tower kind, Change change) {
    GgufWriter text = xs::test::build_tiny_gemma4_gguf(dir / "src", 11, xs::test::tiny_gemma4_moe_gguf_model());
    const nlohmann::json tokenizer = xs::test::tiny_gemma4_tokenizer_json();
    std::vector<std::string> tokens(tokenizer["model"]["vocab"].size());
    for (const auto& item : tokenizer["model"]["vocab"].items()) {
        tokens[item.value().template get<size_t>()] = item.key();
    }
    tokens[kBoi] = "<|image>";
    tokens[kImage] = "<|image|>";
    tokens[kEoi] = "<image|>";
    text.drop_value("tokenizer.ggml.tokens").strings("tokenizer.ggml.tokens", tokens);
    text.write(dir / "tiny.gguf");
    GgufWriter p = xs::test::tiny_gemma4_projector(xs::test::tiny_gemma4_moe_gguf_model().hidden, kind);
    change(p);
    p.write(dir / "mmproj.gguf");
    gguf_import::ImportOptions o;
    o.source = dir / "tiny.gguf";
    o.mmproj = dir / "mmproj.gguf";
    o.output = dir / "pack";
    return o;
}

std::vector<float> f32_values(const TensorView& v) {
    std::vector<float> out(static_cast<size_t>(v.numel()));
    if (v.dtype == DType::F32) {
        std::memcpy(out.data(), v.data, out.size() * 4);
    } else {
        const auto* h = static_cast<const uint16_t*>(v.data);
        for (size_t i = 0; i < out.size(); ++i) out[i] = v.dtype == DType::F16 ? f16_to_f32(h[i]) : bf16_to_f32(h[i]);
    }
    return out;
}

HostTensor copy(const TensorView& v) {
    HostTensor h(v.dtype, v.shape);
    std::memcpy(h.raw(), v.data, h.nbytes());
    return h;
}

// The tower's rows for `image` over `weights`, on cpu-ref.
std::vector<float> tower_rows(const fs::path& pack, const WeightStore& weights, const image::Image& image) {
    const nlohmann::json config = read_json(pack / "config.json");
    const int64_t hidden = config.at("text_config").at("hidden_size").get<int64_t>();
    const auto vc = gemma4::parse_vision_config(config, pack, hidden);
    REQUIRE(vc.has_value());
    const auto tower = gemma4::make_vision_encoder(*vc, weights, "model.");
    SessionOptions so;
    so.precision = DType::F32;
    so.max_context = int64_t{1} << 16;
    const auto session = make_backend("cpu-ref")->compile(tower->programs(), weights, so);
    const MediaInput m = tower->prepare(image);
    std::vector<float> rows;
    for (const EncoderRun& run : m.runs) {
        TensorMap out;
        session->run(run.program, run.inputs, out);
        const HostTensor& e = out.at("embeddings");
        rows.insert(rows.end(), e.data<float>(), e.data<float>() + run.rows * e.shape().back());
    }
    return rows;
}

image::Image test_image() {
    image::Image img;
    img.width = 96;
    img.height = 48;
    img.rgb.resize(static_cast<size_t>(img.width * img.height * 3));
    for (size_t i = 0; i < img.rgb.size(); ++i) img.rgb[i] = static_cast<uint8_t>((i * 37 + i / 7) % 251);
    return img;
}

}  // namespace

TEST_SUITE("gguf import gemma4 vision") {
    TEST_CASE("gemma4 --mmproj: the tower under the checkpoint's names, config.json as the release lays it out") {
        const TempDir work("xs_gguf_gemma4_vision_1");
        const gguf_import::ImportOptions o = files(work.path, {}, [](GgufWriter&) {});
        gguf_import::import_gguf(o);
        const int64_t hidden = xs::test::tiny_gemma4_moe_gguf_model().hidden;

        const nlohmann::json config = read_json(o.output / "config.json");
        CHECK(config.at("model_type") == "gemma4");
        CHECK(config.at("architectures") == nlohmann::json::array({"Gemma4ForConditionalGeneration"}));
        CHECK(config.at("image_token_id") == kImage);
        CHECK(config.at("boi_token_id") == kBoi);
        CHECK(config.at("eoi_token_id") == kEoi);
        const nlohmann::json& t = config.at("text_config");
        CHECK(t.at("model_type") == "gemma4_text");
        CHECK(t.at("enable_moe_block") == true);
        CHECK(t.at("use_bidirectional_attention") == "vision");  // a model without per-layer embeddings
        const nlohmann::json& v = config.at("vision_config");
        CHECK(v.at("hidden_size") == kG4Tower);
        CHECK(v.at("intermediate_size") == kG4Width);
        CHECK(v.at("num_hidden_layers") == kG4Layers);
        CHECK(v.at("head_dim") == kG4HeadDim);
        CHECK(v.at("position_embedding_size") == kG4Positions);
        CHECK(v.at("standardize") == true);
        CHECK(v.at("use_clipped_linears") == false);
        CHECK(v.at("pooling_kernel_size") == 3);
        const nlohmann::json processor = read_json(o.output / "processor_config.json").at("image_processor");
        CHECK(processor.at("patch_size") == kG4Patch);
        CHECK(processor.at("max_soft_tokens") == 280);
        CHECK(processor.at("do_normalize") == false);

        // The tensors under their names with the file's values; the patch embedding put back to [out, y, x, channel].
        const auto pack = SafetensorsWeights::open_dir(o.output);
        const gguf::File file(o.mmproj);
        for (const auto& [from, to] :
             std::vector<std::pair<std::string, std::string>>{
                 {"v.position_embd.weight", "model.vision_tower.patch_embedder.position_embedding_table"},
                 {"v.blk.1.attn_q.weight", "model.vision_tower.encoder.layers.1.self_attn.q_proj.linear.weight"},
                 {"v.blk.0.ln2.weight", "model.vision_tower.encoder.layers.0.pre_feedforward_layernorm.weight"},
                 {"v.blk.0.attn_post_norm.weight",
                  "model.vision_tower.encoder.layers.0.post_attention_layernorm.weight"},
                 {"v.std_scale", "model.vision_tower.std_scale"},
                 {"mm.input_projection.weight", "model.embed_vision.embedding_projection.weight"}}) {
            CAPTURE(to);
            REQUIRE(pack->contains(to));
            const TensorView got = pack->get(to);
            const gguf::TensorInfo* info = file.find_tensor(from);
            REQUIRE(info != nullptr);
            CHECK(std::memcmp(got.data, info->view(got.shape).data, got.nbytes()) == 0);
        }
        CHECK(pack->get("model.vision_tower.patch_embedder.position_embedding_table").shape ==
              (Dims{2, kG4Positions, kG4Tower}));
        CHECK(pack->get("model.embed_vision.embedding_projection.weight").shape == (Dims{hidden, kG4Tower}));
        const TensorView patch = pack->get("model.vision_tower.patch_embedder.input_proj.weight");
        REQUIRE(patch.shape == (Dims{kG4Tower, kG4Patch * kG4Patch * 3}));
        const std::vector<float> stored =
            f32_values(file.find_tensor("v.patch_embd.weight")->view({kG4Tower, 3, kG4Patch, kG4Patch}));
        const std::vector<float> back = f32_values(patch);
        const int64_t p = kG4Patch;
        for (const auto& [o_, y, x, c] : {std::array<int64_t, 4>{0, 0, 0, 0}, std::array<int64_t, 4>{1, 2, 3, 1},
                                          std::array<int64_t, 4>{kG4Tower - 1, p - 1, p - 2, 2}}) {
            CAPTURE(o_);
            CHECK(back[static_cast<size_t>(((o_ * p + y) * p + x) * 3 + c)] ==
                  stored[static_cast<size_t>(((o_ * 3 + c) * p + y) * p + x)]);
        }
    }

    TEST_CASE("gemma4 --mmproj: E2B's clipped linears come with their bounds; an audio tower beside them is left out") {
        const TempDir work("xs_gguf_gemma4_vision_2");
        const gguf_import::ImportOptions o =
            files(work.path, {.standardize = false, .audio = true}, [](GgufWriter&) {});
        gguf_import::import_gguf(o);
        const nlohmann::json v = read_json(o.output / "config.json").at("vision_config");
        CHECK(v.at("use_clipped_linears") == true);
        CHECK(v.at("standardize") == false);
        const auto pack = SafetensorsWeights::open_dir(o.output);
        CHECK(pack->contains("model.vision_tower.encoder.layers.0.mlp.down_proj.output_max"));
        CHECK_FALSE(pack->contains("model.vision_tower.std_bias"));
        for (const std::string& name : pack->names()) {
            CAPTURE(name);
            CHECK(name.find("audio") == std::string::npos);
        }
    }

    TEST_CASE("gemma4 --mmproj: the tower over the imported weights gives the rows of the checkpoint's layout") {
        const TempDir work("xs_gguf_gemma4_vision_3");
        const gguf_import::ImportOptions o = files(work.path, {}, [](GgufWriter&) {});
        gguf_import::import_gguf(o);
        const auto pack = SafetensorsWeights::open_dir(o.output);
        const image::Image img = test_image();
        const std::vector<float> rows = tower_rows(o.output, *pack, img);
        REQUIRE(!rows.empty());

        // The same weights, the patch embedding written here from the file's convolution as transformers holds it.
        const gguf::File file(o.mmproj);
        const auto with_patch = [&](bool as_checkpoint) {
            HostWeights w;
            for (const std::string& name : pack->names()) {
                if (name.rfind("model.vision_tower.", 0) == 0 || name.rfind("model.embed_vision.", 0) == 0) {
                    w.add(name, copy(pack->get(name)));
                }
            }
            const int64_t p = kG4Patch;
            const std::vector<float> stored =
                f32_values(file.find_tensor("v.patch_embd.weight")->view({kG4Tower, 3, p, p}));
            HostTensor patch(DType::F32, {kG4Tower, p * p * 3});
            float* dst = patch.data<float>();
            for (int64_t i = 0; i < kG4Tower * p * p * 3; ++i) dst[i] = stored[static_cast<size_t>(i)];
            if (as_checkpoint) {
                for (int64_t oc = 0; oc < kG4Tower; ++oc)
                    for (int64_t c = 0; c < 3; ++c)
                        for (int64_t y = 0; y < p; ++y)
                            for (int64_t x = 0; x < p; ++x)
                                dst[((oc * p + y) * p + x) * 3 + c] =
                                    stored[static_cast<size_t>(((oc * 3 + c) * p + y) * p + x)];
            }
            w.add("model.vision_tower.patch_embedder.input_proj.weight", std::move(patch));
            return w;
        };
        const std::vector<float> expected = tower_rows(o.output, with_patch(true), img);
        REQUIRE(expected.size() == rows.size());
        double diff = 0, planted = 0, scale = 0;
        const std::vector<float> convolution = tower_rows(o.output, with_patch(false), img);
        for (size_t i = 0; i < rows.size(); ++i) {
            diff = std::max(diff, static_cast<double>(std::fabs(rows[i] - expected[i])));
            planted = std::max(planted, static_cast<double>(std::fabs(convolution[i] - expected[i])));
            scale = std::max(scale, static_cast<double>(std::fabs(expected[i])));
        }
        CHECK(diff == 0.0);
        CHECK(planted > 1e-3 * scale);  // planted: the convolution's layout read as the checkpoint's
    }

    TEST_CASE("gemma4 --mmproj: refusals name what is wrong") {
        const TempDir work("xs_gguf_gemma4_vision_4");
        const auto refused = [&](auto change) {
            try {
                gguf_import::import_gguf(files(work.path / "r", {}, change));
            } catch (const Error& e) {
                fs::remove_all(work.path / "r");
                return std::string(e.what());
            }
            fs::remove_all(work.path / "r");
            return std::string("(accepted)");
        };
        CHECK(refused([](GgufWriter&) {}) == "(accepted)");
        CHECK(has(refused([](GgufWriter& w) {
                  w.drop_value("clip.vision.projector_type").string("clip.vision.projector_type", "qwen3vl_merger");
              }),
                  "the file is not a Gemma 4 vision tower"));
        CHECK(has(refused([](GgufWriter& w) {
                  w.drop_value("clip.vision.projection_dim").u32("clip.vision.projection_dim", 999);
              }),
                  "the tower projects to 999"));
        CHECK(has(refused([](GgufWriter& w) {
                  w.drop_value("clip.vision.image_mean").f32s("clip.vision.image_mean", {0.5f, 0.5f, 0.5f});
              }),
                  "clip.vision.image_mean is not 0 a channel"));
        CHECK(has(refused([](GgufWriter& w) { w.u32("clip.vision.window_size", 4); }),
                  "metadata 'clip.vision.window_size' is not read by this importer"));
        CHECK(has(refused([](GgufWriter& w) { w.drop_tensor("v.blk.1.ffn_up.weight"); }),
                  "model.vision_tower.encoder.layers.1.mlp.up_proj.linear.weight"));
    }
}
