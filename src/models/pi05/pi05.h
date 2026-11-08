#pragma once

// π0.5 (LeRobot policy type "pi05"). Reference: lerobot src/lerobot/policies/pi05/ (modeling_pi05.py, processor_pi05.py)
// and pi_gemma.py, a port of Physical Intelligence's openpi.
//
// A PaliGemma prefix (SigLIP So400m/14 per camera -> 256 tokens each, then the prompt, through Gemma 2B; bidirectional,
// prompt padding masked) and a Gemma 300M action expert that runs once per flow step over the action tokens, reading
// every prefix layer's keys and values. The flow timestep conditions the expert through adaptive RMSNorm (scale, shift
// and a residual gate per norm), so each step's modulation is a constant of the graph. The robot state enters the
// prompt as 256-bin integers. Serving: the checkpoint as LeRobot writes it, plus the PaliGemma tokenizer.json in the
// same directory; normalization statistics are refused by name until they are supported.

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/policy.h"

namespace xs::pi05 {

struct GemmaDims {
    int64_t width = 0;
    int64_t depth = 0;
    int64_t mlp = 0;
    int64_t heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
};

struct Config {
    std::vector<CameraSpec> cameras;
    int64_t image_size = 224;
    int64_t state_dim = 0;
    int64_t action_dim = 0;
    int64_t max_action_dim = 32;
    int64_t horizon = 50;  // chunk_size
    int steps = 10;
    double min_period = 4e-3;
    double max_period = 4.0;
    int64_t prompt_tokens = 200;  // tokenizer_max_length: the prompt is padded (and truncated) to this
    GemmaDims vlm;
    GemmaDims expert;
};

// Parses config.json; `preprocessor` is policy_preprocessor.json when present (checked for normalization statistics).
Config parse_config(const nlohmann::json& config, const nlohmann::json* preprocessor);

std::unique_ptr<Policy> load(const std::filesystem::path& dir, const nlohmann::json& config);

}  // namespace xs::pi05
