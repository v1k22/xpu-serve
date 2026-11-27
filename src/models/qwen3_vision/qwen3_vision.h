#pragma once

// Qwen3-VL's vision tower, Qwen3-Omni's and the Qwen3.5 family's too (transformers Qwen3VLVisionModel,
// Qwen3OmniMoeVisionEncoder, Qwen3_5VisionModel: the same tower without deepstack rows; the image processor is
// Qwen2VLImageProcessor's). An image is resized to a multiple of 32 pixels a side within the processor's pixel bounds,
// cut into 16 x 16 patches of two (identical) frames, and every 2 x 2 block of patches becomes one token for the
// decoder. The tower: the patches' projection, a learned position table interpolated to the image's grid, pre-
// LayerNorm blocks with 2-D rotary positions and attention over all of the image's patches, and mergers that join a
// block's four patches: the last one gives the token's row, and those after some blocks ("deepstack") give rows the
// decoder adds to its first layers' output at the image's tokens. Weights under either naming: Qwen3-Omni's mergers
// (ln_q, mlp.0, mlp.2) or Qwen3-VL's (norm, linear_fc1, linear_fc2); the patch embedding as one kernel, or one a frame
// (patch_embed.proj.frames.<t>.weight, a GGUF import's).
//
// The encoder program runs with T = the image's tokens, each with its 4 patches as rows ([T, 4, ...]): its output
// "embeddings" [T, (1 + deepstack) * out_hidden] is each token's merged row and then its deepstack rows.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/model.h"

namespace xs::qwen3_vision {

struct Config {
    int64_t depth = 0;
    int64_t hidden = 0;          // hidden_size
    int64_t intermediate = 0;    // intermediate_size
    int64_t heads = 0;
    int64_t out_hidden = 0;      // the decoder's width
    int64_t patch = 0;           // patch_size
    int64_t merge = 0;           // spatial_merge_size
    int64_t temporal_patch = 0;  // temporal_patch_size
    int64_t channels = 3;
    std::vector<int64_t> deepstack;  // deepstack_visual_indexes: blocks after which a merger gives decoder rows
    double rope_theta = 10000.0;
};

// A vision_config (`where` names it in errors); variants that change the math are refused by name.
Config parse_config(const nlohmann::json& vision_config, const std::string& where);

// Qwen2VLImageProcessor's settings from preprocessor_config.json.
struct Processor {
    int64_t min_pixels = 56 * 56;
    int64_t max_pixels = 28 * 28 * 1280;
    float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
};
// `max_pixels`: a lower bound than the file's (0: the file's).
Processor processor_spec(const std::filesystem::path& dir, const Config& c, int64_t max_pixels = 0);

// Qwen2VL's smart_resize: the size an image is resized to, (height, width), multiples of `factor`.
std::pair<int64_t, int64_t> smart_resize(int64_t height, int64_t width, int64_t factor, int64_t min_pixels,
                                         int64_t max_pixels);

// The tower over `weights` under `prefix` ("thinker.visual."); `image_token` is what the chat template renders once
// per image and what the prompt holds once per merged token.
std::unique_ptr<VisionEncoder> make_encoder(const Config& c, const Processor& p, const WeightStore& weights,
                                            std::string prefix, int32_t image_token);

}  // namespace xs::qwen3_vision
