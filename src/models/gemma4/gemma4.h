#pragma once

// Gemma 4 (Hugging Face model_type "gemma4" / "gemma4_text"). Reference: transformers models/gemma4/
// modeling_gemma4.py (Gemma4TextModel and its layers; Gemma4VisionModel, Gemma4MultimodalEmbedder) and
// image_processing_gemma4.py (Gemma4ImageProcessor).
//
// Served: the dense E-series text layout (E2B, E4B): sliding + full attention, per-layer embeddings (PLE) and a
// KV-shared tail; and the vision tower of those checkpoints. Mixture-of-experts layers and K=V attention (26B-A4B,
// 31B) are refused by name; their standardized vision towers load as text-only with a warning. Audio is not served.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/model.h"

namespace xs::gemma4 {

enum class AttnKind { Sliding, Full };

struct RopeConfig {
    double theta = 10000.0;
    double partial_rotary_factor = 1.0;  // "proportional" RoPE rotates this share of each head
    bool proportional = false;
};

struct Config {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;         // sliding layers
    int64_t global_head_dim = 0;  // full-attention layers
    int64_t sliding_window = 0;
    int64_t num_kv_shared_layers = 0;
    int64_t ple_dim = 0;          // hidden_size_per_layer_input; 0 = no per-layer embeddings
    int64_t ple_vocab_size = 0;
    bool double_wide_mlp = false;
    double rms_norm_eps = 1e-6;
    double final_logit_softcap = 0.0;
    int64_t max_position = 0;
    std::vector<AttnKind> layer_kinds;
    RopeConfig rope_sliding;
    RopeConfig rope_full;

    int64_t first_kv_shared_layer() const { return num_layers - num_kv_shared_layers; }
    bool is_kv_shared(int64_t layer) const { return num_kv_shared_layers > 0 && layer >= first_kv_shared_layer(); }
    int64_t layer_head_dim(int64_t layer) const;
    int64_t layer_intermediate_size(int64_t layer) const;
    // The earlier layer whose KV a shared layer reads: the last non-shared layer of the same kind.
    int64_t kv_source_layer(int64_t layer) const;
};

// Parses config.json (the text_config object of a multimodal config, or the root of a text-only one).
Config parse_config(const nlohmann::json& config_json);

struct VisionConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t patch_size = 0;
    int64_t pooling_kernel = 0;           // k: soft tokens average k x k patches
    int64_t position_embedding_size = 0;  // rows per axis of the patch position table
    int64_t max_soft_tokens = 0;          // per image (image processor); fixes the patch budget
    double rms_norm_eps = 1e-6;
    double rope_theta = 100.0;
    bool clipped_linears = false;
    float rescale_factor = 1.0f / 255.0f;  // pixel scale before the tower's own 2 * (x - 0.5)
    int64_t text_hidden_size = 0;
    int32_t image_token_id = -1;  // the placeholder the chat template renders, and every soft token
    int32_t boi_token_id = -1;
    int32_t eoi_token_id = -1;
};

// The vision config of a multimodal checkpoint (config.json vision_config + processor_config.json image_processor),
// nullopt when there is none. Throws xs::Error naming what is not supported.
std::optional<VisionConfig> parse_vision_config(const nlohmann::json& config_json,
                                                const std::filesystem::path& model_dir, int64_t text_hidden_size);

// The image size the processor resizes to: the largest one with the aspect ratio kept that fits the patch budget
// and has sides divisible by pooling_kernel * patch_size (get_aspect_ratio_preserving_size in transformers).
// Returns {height, width}. Throws xs::Error for images that cannot fit.
std::pair<int64_t, int64_t> image_target_size(int64_t height, int64_t width, const VisionConfig& vc);

// The vision encoder over the checkpoint's weights; `prefix` is where vision_tower. and embed_vision. live.
std::unique_ptr<VisionEncoder> make_vision_encoder(const VisionConfig& vc, const WeightStore& weights,
                                                   const std::string& prefix);

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options);

}  // namespace xs::gemma4
