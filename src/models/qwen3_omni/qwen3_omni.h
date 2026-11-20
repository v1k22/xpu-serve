#pragma once

// Qwen3-Omni's thinker (model_type "qwen3_omni_moe": Qwen/Qwen3-Omni-30B-A3B-Instruct, -Thinking, -Captioner).
// Reference: transformers models/qwen3_omni_moe/ (Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor).
//
// Qwen's audio encoder (models/qwen3_audio, at its larger size) and a Qwen3 mixture-of-experts decoder: text and
// audio in, text out. The decoder reads one token per encoder row where the chat template put <|audio_pad|>; with text
// and audio only, every token takes one position on all three axes of its multimodal RoPE, which is plain RoPE.
// Images and video (Qwen3-VL's vision tower, positions on three axes) and speech out (the talker and Code2Wav, which
// the checkpoint also holds) are not supported yet: the vision tower is not loaded, so image parts are refused, and
// the talker is not read.

#include <cstdint>
#include <filesystem>
#include <memory>

#include <nlohmann/json.hpp>

#include "models/model.h"
#include "models/qwen3/qwen3.h"
#include "models/qwen3_audio/qwen3_audio.h"

namespace xs::qwen3_omni {

struct Config {
    qwen3::Config text;
    qwen3_audio::Config audio;
    int32_t audio_token = 0;  // <|audio_pad|>
};

// config.json: the thinker_config's text_config and audio_config.
Config parse_config(const nlohmann::json& config_json);

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options);

}  // namespace xs::qwen3_omni
