#pragma once

// A tiny projector file of a Qwen3.5-family vision tower (llama.cpp's mmproj: general.architecture clip, projector type
// qwen3vl_merger, Qwen3-VL's tower without deepstack layers), as its converter writes one: matrices f16, norms and
// biases f32, dims in ggml's order, the patch embedding a kernel a frame. For the qwen35 import's tests and the hub's.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "io/gguf_test_util.h"
#include "util/half.h"

namespace xs::test {

inline constexpr int64_t kTower = 32, kWidth = 64, kLayers = 2, kHeads = 2, kPatch = 16, kMerge = 2;
inline constexpr int64_t kPositions = 64;

// Deterministic values in [-scale, scale).
inline std::vector<float> projector_noise(int64_t n, uint32_t seed, float scale) {
    std::vector<float> v(static_cast<size_t>(n));
    uint32_t x = seed * 2654435761u + 1;
    for (float& f : v) {
        x = x * 1664525u + 1013904223u;
        f = scale * (static_cast<float>(x >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f);
    }
    return v;
}

inline std::string f16_bytes(const std::vector<float>& v) {
    std::string out(v.size() * 2, '\0');
    for (size_t i = 0; i < v.size(); ++i) {
        const uint16_t h = f32_to_f16(v[i]);
        std::memcpy(out.data() + 2 * i, &h, 2);
    }
    return out;
}

// A projector file of a tower projecting to `out` (the model's hidden size), as llama.cpp's converter writes one:
// matrices f16, norms and biases f32, dims in ggml's order.
inline GgufWriter tiny_qwen35_projector(int64_t out) {
    GgufWriter w;
    w.string("general.architecture", "clip").string("general.type", "mmproj");
    w.boolean("clip.has_vision_encoder", true).string("clip.projector_type", "qwen3vl_merger");
    w.boolean("clip.use_gelu", true);
    w.u32("clip.vision.projection_dim", static_cast<uint32_t>(out)).u32("clip.vision.image_size", 768);
    w.u32("clip.vision.patch_size", kPatch).u32("clip.vision.embedding_length", kTower);
    w.u32("clip.vision.feed_forward_length", kWidth).u32("clip.vision.block_count", kLayers);
    w.u32("clip.vision.attention.head_count", kHeads).u32("clip.vision.spatial_merge_size", kMerge);
    w.f32s("clip.vision.image_mean", {0.5f, 0.5f, 0.5f}).f32s("clip.vision.image_std", {0.5f, 0.5f, 0.5f});
    w.f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
    w.bools("clip.vision.is_deepstack_layers", std::vector<bool>(kLayers, false));
    uint32_t seed = 1;
    const auto matrix = [&](const std::string& name, int64_t rows, int64_t cols) {
        w.tensor(name, {cols, rows}, kGgmlF16, f16_bytes(projector_noise(rows * cols, seed++, 0.2f)));
    };
    const auto vec = [&](const std::string& name, int64_t n, float base) {
        std::vector<float> v = projector_noise(n, seed++, 0.2f);
        for (float& x : v) x += base;
        w.f32_tensor(name, {n}, v);
    };
    for (const char* frame : {"v.patch_embd.weight", "v.patch_embd.weight.1"}) {
        w.tensor(frame, {kPatch, kPatch, 3, kTower}, kGgmlF16,
                 f16_bytes(projector_noise(kTower * 3 * kPatch * kPatch, seed++, 0.05f)));
    }
    vec("v.patch_embd.bias", kTower, 0.0f);
    w.f32_tensor("v.position_embd.weight", {kTower, kPositions},
                 projector_noise(kTower * kPositions, seed++, 0.2f));
    for (int64_t i = 0; i < kLayers; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        vec(b + "ln1.weight", kTower, 1.0f);
        vec(b + "ln1.bias", kTower, 0.0f);
        vec(b + "ln2.weight", kTower, 1.0f);
        vec(b + "ln2.bias", kTower, 0.0f);
        matrix(b + "attn_qkv.weight", 3 * kTower, kTower);
        vec(b + "attn_qkv.bias", 3 * kTower, 0.0f);
        matrix(b + "attn_out.weight", kTower, kTower);
        vec(b + "attn_out.bias", kTower, 0.0f);
        matrix(b + "ffn_up.weight", kWidth, kTower);
        vec(b + "ffn_up.bias", kWidth, 0.0f);
        matrix(b + "ffn_down.weight", kTower, kWidth);
        vec(b + "ffn_down.bias", kTower, 0.0f);
    }
    const int64_t joined = kMerge * kMerge * kTower;
    vec("v.post_ln.weight", kTower, 1.0f);
    vec("v.post_ln.bias", kTower, 0.0f);
    matrix("mm.0.weight", joined, joined);
    vec("mm.0.bias", joined, 0.0f);
    matrix("mm.2.weight", out, joined);
    vec("mm.2.bias", out, 0.0f);
    return w;
}

}  // namespace xs::test
