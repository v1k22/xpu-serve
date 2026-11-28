#pragma once

// A tiny projector file of a Gemma 4 vision tower (llama.cpp's mmproj: general.architecture clip, projector type
// gemma4v), as its converter writes one (conversion/gemma.py, Gemma4VisionAudioModel): matrices f16, norms f32, dims
// in ggml's order, the patch embedding as a convolution's [out, channel, y, x], the position table f32. Either kind of
// tower: E2B's (each linear with its clipping bounds) or 26B-A4B's (std_bias and std_scale after the pooling); with
// `audio`, an audio tower's keys and tensors beside it, as E2B's file holds them.

#include <cstdint>
#include <string>
#include <vector>

#include "convert/tiny_qwen35_projector.h"  // projector_noise, f16_bytes
#include "io/gguf_test_util.h"

namespace xs::test {

inline constexpr int64_t kG4Tower = 64, kG4Width = 96, kG4Layers = 2, kG4Heads = 2, kG4HeadDim = 32;
inline constexpr int64_t kG4Patch = 16, kG4Positions = 256;  // rows a side: past any patch position of an image

struct TinyGemma4Tower {
    bool standardize = true;  // 26B-A4B's kind; else E2B's clipped linears
    bool audio = false;
};

inline GgufWriter tiny_gemma4_projector(int64_t out, TinyGemma4Tower kind = {}) {
    GgufWriter w;
    w.string("general.architecture", "clip").string("general.type", "mmproj");
    w.boolean("clip.has_vision_encoder", true).string("clip.vision.projector_type", "gemma4v");
    w.u32("clip.vision.projection_dim", static_cast<uint32_t>(out)).u32("clip.vision.image_size", 224);
    w.u32("clip.vision.patch_size", kG4Patch).u32("clip.vision.embedding_length", kG4Tower);
    w.u32("clip.vision.feed_forward_length", kG4Width).u32("clip.vision.block_count", kG4Layers);
    w.u32("clip.vision.attention.head_count", kG4Heads);
    w.f32s("clip.vision.image_mean", {0.0f, 0.0f, 0.0f}).f32s("clip.vision.image_std", {1.0f, 1.0f, 1.0f});
    w.f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
    uint32_t seed = 3;
    const auto matrix = [&](const std::string& name, int64_t rows, int64_t cols) {
        w.tensor(name, {cols, rows}, kGgmlF16, f16_bytes(projector_noise(rows * cols, seed++, 0.2f)));
    };
    const auto vec = [&](const std::string& name, int64_t n, float base) {
        std::vector<float> v = projector_noise(n, seed++, 0.2f);
        for (float& x : v) x += base;
        w.f32_tensor(name, {n}, v);
    };
    const auto linear = [&](const std::string& name, int64_t rows, int64_t cols) {
        matrix(name + ".weight", rows, cols);
        if (kind.standardize) return;
        w.f32_tensor(name + ".input_min", {1}, {-3.0f}).f32_tensor(name + ".input_max", {1}, {3.0f});
        w.f32_tensor(name + ".output_min", {1}, {-2.0f}).f32_tensor(name + ".output_max", {1}, {2.0f});
    };
    w.f32_tensor("v.patch_embd.weight", {kG4Patch, kG4Patch, 3, kG4Tower},
                 projector_noise(kG4Tower * 3 * kG4Patch * kG4Patch, seed++, 0.05f));
    w.f32_tensor("v.position_embd.weight", {kG4Tower, kG4Positions, 2},
                 projector_noise(2 * kG4Positions * kG4Tower, seed++, 0.2f));
    for (int64_t i = 0; i < kG4Layers; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        for (const char* n : {"ln1", "attn_post_norm", "ln2", "ffn_post_norm"}) vec(b + n + ".weight", kG4Tower, 1.0f);
        vec(b + "attn_q_norm.weight", kG4HeadDim, 1.0f);
        vec(b + "attn_k_norm.weight", kG4HeadDim, 1.0f);
        linear(b + "attn_q", kG4Tower, kG4Tower);
        linear(b + "attn_k", kG4Tower, kG4Tower);
        linear(b + "attn_v", kG4Tower, kG4Tower);
        linear(b + "attn_out", kG4Tower, kG4Tower);
        linear(b + "ffn_gate", kG4Width, kG4Tower);
        linear(b + "ffn_up", kG4Width, kG4Tower);
        linear(b + "ffn_down", kG4Tower, kG4Width);
    }
    matrix("mm.input_projection.weight", out, kG4Tower);
    if (kind.standardize) {
        vec("v.std_bias", kG4Tower, 0.0f);
        vec("v.std_scale", kG4Tower, 0.5f);
    }
    if (kind.audio) {
        w.boolean("clip.has_audio_encoder", true).string("clip.audio.projector_type", "gemma4a");
        w.u32("clip.audio.projection_dim", static_cast<uint32_t>(out)).u32("clip.audio.embedding_length", 32);
        w.u32("clip.audio.num_mel_bins", 128);
        matrix("a.blk.0.attn_q.weight", 32, 32);
        matrix("mm.a.input_projection.weight", out, 32);
    }
    return w;
}

}  // namespace xs::test
