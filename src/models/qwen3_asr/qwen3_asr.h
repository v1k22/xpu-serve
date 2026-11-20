#pragma once

// Qwen3-ASR (model_type "qwen3_asr": Qwen/Qwen3-ASR-1.7B, -0.6B). Reference: transformers models/qwen3_asr/
// (Qwen3ASRForConditionalGeneration, Qwen3ASREncoder, Qwen3ASRFeatureExtractor) and vLLM's qwen3_asr.py.
//
// Qwen's audio encoder (models/qwen3_audio) and a Qwen3 decoder. The decoder reads one token per encoder row where the
// chat template put <|audio_pad|>, and answers "language <Name><asr_text><transcript>". Checkpoints come as released
// (weights under thinker.*, a thinker_config) or converted for transformers (model.*, flat configs). The forced aligner
// (Qwen3-ForcedAligner) is refused by name.

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/model.h"
#include "models/qwen3/qwen3.h"
#include "models/qwen3_audio/qwen3_audio.h"

namespace xs::qwen3_asr {

struct Config {
    qwen3::Config text;
    qwen3_audio::Config audio;
    int32_t audio_token = 0;  // <|audio_pad|>
    bool released_layout = true;  // weights under thinker.* (else transformers': model.*)
    std::vector<std::string> languages;  // support_languages (empty: the 30 Qwen3-ASR names)
};

// config.json (released: thinker_config holds audio_config, text_config and the audio token; converted: flat).
Config parse_config(const nlohmann::json& config_json);

// Qwen3-ASR's transcription format: "language <Name><asr_text>" heads a reply (and forces a language in the prompt),
// "language None" is silence; `languages` the names it knows (empty: Qwen3-ASR's 30); a second of audio takes
// `audio_tokens_per_second` prompt tokens.
std::unique_ptr<Transcription> make_transcription(std::vector<std::string> languages,
                                                  double audio_tokens_per_second);

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options);

}  // namespace xs::qwen3_asr
