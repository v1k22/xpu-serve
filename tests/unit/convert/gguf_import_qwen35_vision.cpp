// GGUF import of a Qwen3.5-family vision tower from llama.cpp's projector file (mmproj: general.architecture clip,
// projector type qwen3vl_merger) beside the tiny qwen35 GGUF (tiny_gguf.h): config.json as the release lays it out
// (text_config, vision_config, the image token), the image processor's file, every tower tensor under the
// checkpoint's name with the file's bytes (the patch embedding a kernel a frame); the tower over those kernels gives
// the rows it gives over the whole kernel; the projector found beside the model file as -hf finds it in a repository;
// refusals by name.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "convert/gguf/gguf_import.h"
#include "convert/import_test_util.h"
#include "convert/tiny_gguf.h"
#include "convert/tiny_qwen35_projector.h"
#include "hub/model_source.h"
#include "image/image.h"
#include "io/host_weights.h"
#include "io/safetensors.h"
#include "models/qwen3_vision/qwen3_vision.h"
#include "util/half.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::GgufWriter;
using xs::test::has;
using xs::test::TempDir;
using xs::test::kHeads;
using xs::test::kLayers;
using xs::test::kMerge;
using xs::test::kPatch;
using xs::test::kPositions;
using xs::test::kTower;
using xs::test::kWidth;

namespace {

HostTensor copy(const TensorView& v) {
    HostTensor h(v.dtype, v.shape);
    std::memcpy(h.raw(), v.data, h.nbytes());
    return h;
}

nlohmann::json read_json(const fs::path& path) {
    std::ifstream in(path);
    return nlohmann::json::parse(in);
}

// The tiny model's file and a projector for it in `dir`; the projector as `change` leaves it.
template <typename Change>
gguf_import::ImportOptions files(const fs::path& dir, Change change) {
    xs::test::build_tiny_gguf(dir / "src", 11).write(dir / "tiny.gguf");
    GgufWriter p = xs::test::tiny_qwen35_projector(xs::test::tiny_gguf_model().hidden);
    change(p);
    p.write(dir / "mmproj.gguf");
    gguf_import::ImportOptions o;
    o.source = dir / "tiny.gguf";
    o.mmproj = dir / "mmproj.gguf";
    o.output = dir / "pack";
    return o;
}

// The tower's rows for `image` over `weights`, on cpu-ref.
std::vector<float> tower_rows(const fs::path& pack, const WeightStore& weights, const image::Image& image) {
    const nlohmann::json config = read_json(pack / "config.json");
    const qwen3_vision::Config c = qwen3_vision::parse_config(config.at("vision_config"), "vision_config");
    const auto tower = qwen3_vision::make_encoder(c, qwen3_vision::processor_spec(pack, c), weights, "model.visual.",
                                                  config.at("image_token_id").get<int32_t>());
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
        rows.insert(rows.end(), e.data<float>(), e.data<float>() + e.numel());
    }
    return rows;
}

}  // namespace

TEST_CASE("qwen35 --mmproj: the tower under the checkpoint's names, config.json as the release lays it out") {
    const TempDir work("xs_gguf_qwen35_vision_1");
    const gguf_import::ImportOptions o = files(work.path, [](GgufWriter&) {});
    gguf_import::import_gguf(o);
    const int64_t hidden = xs::test::tiny_gguf_model().hidden;

    const nlohmann::json config = read_json(o.output / "config.json");
    CHECK(config.at("model_type") == "qwen3_5");
    CHECK(config.at("architectures") == nlohmann::json::array({"Qwen3_5ForConditionalGeneration"}));
    CHECK(config.at("text_config").at("model_type") == "qwen3_5_text");
    CHECK(config.at("image_token_id") == 274);  // <|image_pad|>, the tiny vocabulary's last added token
    const nlohmann::json& v = config.at("vision_config");
    CHECK(v.at("depth") == kLayers);
    CHECK(v.at("hidden_size") == kTower);
    CHECK(v.at("intermediate_size") == kWidth);
    CHECK(v.at("num_heads") == kHeads);
    CHECK(v.at("out_hidden_size") == hidden);
    CHECK(v.at("temporal_patch_size") == 2);
    CHECK(v.at("num_position_embeddings") == kPositions);
    CHECK(v.at("deepstack_visual_indexes").empty());
    const nlohmann::json pre = read_json(o.output / "preprocessor_config.json");
    CHECK(pre.at("patch_size") == kPatch);
    CHECK(pre.at("merge_size") == kMerge);
    CHECK(pre.at("size").at("shortest_edge") == 65536);

    // Every tensor of the projector under its name, with the file's bytes.
    const auto pack = SafetensorsWeights::open_dir(o.output);
    const gguf::File file(o.mmproj);
    const std::vector<std::pair<std::string, std::string>> names = {
        {"v.patch_embd.weight", "model.visual.patch_embed.proj.frames.0.weight"},
        {"v.patch_embd.weight.1", "model.visual.patch_embed.proj.frames.1.weight"},
        {"v.position_embd.weight", "model.visual.pos_embed.weight"},
        {"v.blk.1.attn_qkv.weight", "model.visual.blocks.1.attn.qkv.weight"},
        {"v.blk.0.ffn_down.bias", "model.visual.blocks.0.mlp.linear_fc2.bias"},
        {"v.post_ln.weight", "model.visual.merger.norm.weight"},
        {"mm.2.weight", "model.visual.merger.linear_fc2.weight"},
    };
    for (const auto& [from, to] : names) {
        CAPTURE(to);
        REQUIRE(pack->contains(to));
        const TensorView t = pack->get(to);
        const gguf::TensorInfo* info = file.find_tensor(from);
        REQUIRE(info != nullptr);
        CHECK(std::memcmp(t.data, info->view(t.shape).data, t.nbytes()) == 0);
    }
    CHECK(pack->get("model.visual.patch_embed.proj.frames.0.weight").shape == (Dims{kTower, 3, kPatch, kPatch}));
    CHECK(pack->get("model.visual.blocks.0.attn.qkv.weight").shape == (Dims{3 * kTower, kTower}));
    CHECK(pack->get("model.visual.merger.linear_fc2.weight").shape == (Dims{hidden, kMerge * kMerge * kTower}));
}

TEST_CASE("qwen35 --mmproj: the tower over a kernel a frame gives the rows it gives over the whole kernel") {
    const TempDir work("xs_gguf_qwen35_vision_2");
    const gguf_import::ImportOptions o = files(work.path, [](GgufWriter&) {});
    gguf_import::import_gguf(o);
    const auto pack = SafetensorsWeights::open_dir(o.output);

    // The same weights with the frames' kernels joined into the checkpoint's [hidden, channels, frames, y, x].
    HostWeights whole;
    for (const std::string& name : pack->names()) {
        if (name.rfind("model.visual.", 0) != 0 || name.find("patch_embed.proj.frames.") != std::string::npos) continue;
        whole.add(name, copy(pack->get(name)));
    }
    const TensorView f0 = pack->get("model.visual.patch_embed.proj.frames.0.weight");
    const TensorView f1 = pack->get("model.visual.patch_embed.proj.frames.1.weight");
    const int64_t plane = kPatch * kPatch;
    HostTensor kernel(f0.dtype, {kTower, 3, 2, kPatch, kPatch});
    const auto* a = static_cast<const uint16_t*>(f0.data);
    const auto* b = static_cast<const uint16_t*>(f1.data);
    auto* k = static_cast<uint16_t*>(kernel.raw());
    for (int64_t oc = 0; oc < kTower * 3; ++oc) {
        std::memcpy(k + (2 * oc) * plane, a + oc * plane, static_cast<size_t>(plane) * 2);
        std::memcpy(k + (2 * oc + 1) * plane, b + oc * plane, static_cast<size_t>(plane) * 2);
    }
    whole.add("model.visual.patch_embed.proj.weight", std::move(kernel));

    image::Image img;
    img.width = 64;
    img.height = 96;
    img.rgb.resize(static_cast<size_t>(img.width * img.height * 3));
    for (size_t i = 0; i < img.rgb.size(); ++i) img.rgb[i] = static_cast<uint8_t>((i * 37 + i / 7) % 251);
    const std::vector<float> framed = tower_rows(o.output, *pack, img), joined = tower_rows(o.output, whole, img);
    REQUIRE(framed.size() == joined.size());
    REQUIRE(!framed.empty());
    double diff = 0, scale = 0;
    for (size_t i = 0; i < framed.size(); ++i) {
        diff = std::max(diff, static_cast<double>(std::fabs(framed[i] - joined[i])));
        scale = std::max(scale, static_cast<double>(std::fabs(joined[i])));
    }
    CHECK(diff <= 1e-5 * scale);
    // Planted: the second frame's kernel left out gives other rows (an image's two frames are copies, so the test
    // tells the frames' sum from one frame's).
    HostWeights one_frame;
    for (const std::string& name : pack->names()) {
        if (name.rfind("model.visual.", 0) != 0) continue;
        HostTensor t = copy(pack->get(name));
        if (name.find("frames.1.") != std::string::npos) std::memset(t.raw(), 0, t.nbytes());
        one_frame.add(name, std::move(t));
    }
    const std::vector<float> half = tower_rows(o.output, one_frame, img);
    double planted = 0;
    for (size_t i = 0; i < half.size(); ++i) {
        planted = std::max(planted, static_cast<double>(std::fabs(half[i] - joined[i])));
    }
    CHECK(planted > 100 * diff);
}

TEST_CASE("qwen35 by its path: the projector beside the model file goes into its pack") {
    const TempDir work("xs_gguf_qwen35_vision_by_path");
    fs::create_directories(work.path / "models");
    xs::test::build_tiny_gguf(work.path / "src", 11).write(work.path / "models" / "Tiny-Q8_0.gguf");
    xs::test::tiny_qwen35_projector(xs::test::tiny_gguf_model().hidden)
        .write(work.path / "models" / "mmproj-BF16.gguf");
    hub::SourceOptions o;
    o.cache = work.path / "xs";
    const hub::ModelSource m =
        hub::resolve_model({.path = (work.path / "models" / "Tiny-Q8_0.gguf").string()}, hub::FileRole::Model, o);
    const nlohmann::json config = read_json(m.dir / "config.json");
    CHECK(config.at("model_type") == "qwen3_5");
    CHECK(config.contains("vision_config"));
    CHECK(fs::exists(m.dir / "preprocessor_config.json"));
}

TEST_CASE("qwen35 --mmproj: refusals name what is wrong") {
    struct Case {
        const char* what;
        std::function<void(GgufWriter&)> change;
        const char* error;
    };
    const std::vector<Case> cases = {
        {"another projector",
         [](GgufWriter& w) { w.drop_value("clip.projector_type").string("clip.projector_type", "qwen25vl_merger"); },
         "is not a Qwen3.5-family vision tower"},
        {"deepstack layers",
         [](GgufWriter& w) {
             w.drop_value("clip.vision.is_deepstack_layers").bools("clip.vision.is_deepstack_layers", {true, false});
         },
         "deepstack"},
        {"another model's tower",
         [](GgufWriter& w) { w.drop_value("clip.vision.projection_dim").u32("clip.vision.projection_dim", 96); },
         "another model's"},
        {"another normalization",
         [](GgufWriter& w) {
             w.drop_value("clip.vision.image_mean").f32s("clip.vision.image_mean", {0.48f, 0.46f, 0.41f});
         },
         "image_mean"},
        {"an unknown key", [](GgufWriter& w) { w.u32("clip.vision.window_size", 112); }, "window_size"},
    };
    for (const Case& c : cases) {
        CAPTURE(c.what);
        const TempDir work("xs_gguf_qwen35_vision_3");
        const gguf_import::ImportOptions o = files(work.path, c.change);
        std::string error;
        try {
            gguf_import::import_gguf(o);
        } catch (const Error& e) {
            error = e.what();
        }
        CHECK_MESSAGE(has(error, c.error), error);
    }
}
