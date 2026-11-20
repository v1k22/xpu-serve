#pragma once

// Qwen3-Omni's thinker (model_type "qwen3_omni_moe": Qwen/Qwen3-Omni-30B-A3B-Instruct, -Thinking, -Captioner).
// Reference: transformers models/qwen3_omni_moe/ (Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor).
//
// Qwen's audio encoder (models/qwen3_audio, at its larger size), Qwen3-VL's vision tower (models/qwen3_vision) and
// a Qwen3 mixture-of-experts decoder: text, audio and images in, text out. The decoder reads one token per encoder row
// where the chat template put <|audio_pad|> or <|image_pad|>, with multimodal RoPE: positions on three axes (time,
// height, width), an image's tokens at (0, row, column) from its first position (MediaInput::positions), text and
// audio one position a token; and the vision tower's deepstack rows added to its first layers' output at the image's
// tokens. Video and speech out (the talker and Code2Wav, which the checkpoint also holds) are not supported yet.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>

#include <nlohmann/json.hpp>

#include "models/model.h"
#include "models/qwen3/qwen3.h"
#include "models/qwen3_audio/qwen3_audio.h"
#include "models/qwen3_vision/qwen3_vision.h"

namespace xs::qwen3_omni {

struct Config {
    qwen3::Config text;
    qwen3_audio::Config audio;
    std::optional<qwen3_vision::Config> vision;
    int32_t audio_token = 0;  // <|audio_pad|>
    int32_t image_token = 0;  // <|image_pad|>
};

// config.json: the thinker_config's text_config, audio_config and vision_config.
Config parse_config(const nlohmann::json& config_json);

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options);

}  // namespace xs::qwen3_omni
