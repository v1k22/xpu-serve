// A LoRA adapter in llama.cpp's GGUF form (convert/gguf/adapter.h): its tensors under the names the model code reads,
// its scale, and the files it refuses.
#include <doctest/doctest.h>

#include <filesystem>

#include "convert/gguf/adapter.h"
#include "io/gguf_test_util.h"

namespace fs = std::filesystem;
using xs::test::GgufWriter;

namespace {

// An inkling2_loop adapter of rank 2: layer 3's attention output [5 <- 4], layer 0's expert bank (3 experts of
// [6 <- 2]) and the embedding table (vocab 7, width 5). Dims as the file lists them: columns first.
GgufWriter adapter() {
    GgufWriter w;
    w.string("general.architecture", "inkling2_loop").string("general.type", "adapter");
    w.string("adapter.type", "lora").f32("adapter.lora_alpha", 4.0f);
    const auto values = [](int64_t n, float base) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = base + 0.25f * static_cast<float>(i);
        return v;
    };
    w.f32_tensor("blk.3.attn_output.weight.lora_a", {4, 2}, values(8, 1.0f));
    w.f32_tensor("blk.3.attn_output.weight.lora_b", {2, 5}, values(10, -1.0f));
    w.f32_tensor("blk.0.ffn_down_exps.weight.lora_a", {2, 2, 3}, values(12, 0.5f));
    w.f32_tensor("blk.0.ffn_down_exps.weight.lora_b", {2, 6, 3}, values(36, 0.0f));
    w.f32_tensor("token_embd.weight.lora_a", {2, 7}, values(14, 2.0f));
    w.f32_tensor("token_embd.weight.lora_b", {2, 5}, values(10, 3.0f));
    return w;
}

struct File {
    fs::path path = fs::temp_directory_path() / "xs_test_adapter.gguf";
    explicit File(const GgufWriter& w) { w.write(path); }
    ~File() { fs::remove(path); }
};

}  // namespace

TEST_CASE("gguf adapter: a LoRA file's changes under the checkpoint's names, scaled by alpha / rank") {
    const File f(adapter());
    const xs::gguf_import::Adapter a = xs::gguf_import::read_lora(f.path, 0.5f);
    CHECK(a.architecture == "inkling2_loop");
    REQUIRE(a.changes.size() == 3);
    const std::string wo = "model.llm.layers.3.attn.wo_ud.weight";
    const std::string w2 = "model.llm.layers.0.mlp.experts.w2_weight";
    const std::string embed = "model.llm.embed.weight";
    for (const std::string& name : {wo, w2, embed}) {
        CAPTURE(name);
        REQUIRE(a.changes.count(name) == 1);
        CHECK(a.changes.at(name).scale == doctest::Approx(0.5 * 4.0 / 2.0));
    }
    // Row-major, outermost first: A [r, in], B [out, r]; a bank's per expert; a table's A by token.
    CHECK(a.tensors->get(a.changes.at(wo).a).shape == xs::Dims{2, 4});
    CHECK(a.tensors->get(a.changes.at(wo).b).shape == xs::Dims{5, 2});
    CHECK(a.tensors->get(a.changes.at(w2).a).shape == xs::Dims{3, 2, 2});
    CHECK(a.tensors->get(a.changes.at(w2).b).shape == xs::Dims{3, 6, 2});
    CHECK(a.tensors->get(a.changes.at(embed).a).shape == xs::Dims{7, 2});
    const xs::TensorView& b = a.tensors->get(a.changes.at(wo).b);
    CHECK(static_cast<const float*>(b.data)[3] == doctest::Approx(-0.25f));
}

TEST_CASE("gguf adapter: other files, a tensor without its pair and one the architecture cannot name are refused") {
    {
        GgufWriter w = adapter();
        w.drop_value("general.type").string("general.type", "model");
        const File f(w);
        CHECK_THROWS_WITH(xs::gguf_import::read_lora(f.path), doctest::Contains("is not an adapter"));
    }
    {
        GgufWriter w = adapter();
        w.drop_value("adapter.type").string("adapter.type", "control_vector");
        const File f(w);
        CHECK_THROWS_WITH(xs::gguf_import::read_lora(f.path), doctest::Contains("only LoRA adapters"));
    }
    {
        GgufWriter w = adapter();
        w.drop_tensor("blk.3.attn_output.weight.lora_b");
        const File f(w);
        CHECK_THROWS_WITH(xs::gguf_import::read_lora(f.path), doctest::Contains("without .lora_b"));
    }
    {
        GgufWriter w = adapter();
        w.f32_tensor("blk.0.attn_q_clip.weight.lora_a", {2, 2}, std::vector<float>(4, 1.0f));
        w.f32_tensor("blk.0.attn_q_clip.weight.lora_b", {2, 2}, std::vector<float>(4, 1.0f));
        w.f32_tensor("blk.0.nonsense.weight.lora_a", {2, 2}, std::vector<float>(4, 1.0f));
        w.f32_tensor("blk.0.nonsense.weight.lora_b", {2, 2}, std::vector<float>(4, 1.0f));
        const File f(w);
        CHECK_THROWS_WITH(xs::gguf_import::read_lora(f.path), doctest::Contains("blk.0.nonsense.weight"));
    }
}
