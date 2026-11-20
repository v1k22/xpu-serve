#pragma once

// Qwen3-ASR (model_type "qwen3_asr": Qwen/Qwen3-ASR-1.7B, -0.6B). Reference: transformers models/qwen3_asr/
// (Qwen3ASRForConditionalGeneration, Qwen3ASREncoder, Qwen3ASRFeatureExtractor) and vLLM's qwen3_asr.py.
//
// A Whisper-style audio encoder and a Qwen3 decoder. Audio becomes 128-bin log-mel frames (16 kHz, hop 160); every
// chunk of 2 * n_window frames goes through three 3 x 3 stride-2 convolutions to 13 tokens, sinusoidal positions that
// restart in each chunk, then transformer layers that attend within windows of n_window_infer frames (8 chunks, 104
// tokens) and two projections into the decoder's width. The decoder reads one token per encoder row where the chat
// template put <|audio_pad|>, and answers "language <Name><asr_text><transcript>". Checkpoints come as released
// (weights under thinker.*, a thinker_config) or converted for transformers (model.*, flat configs). The forced aligner
// (Qwen3-ForcedAligner) is refused by name.

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "audio/mel.h"
#include "models/model.h"
#include "models/qwen3/qwen3.h"

namespace xs::qwen3_asr {

struct AudioConfig {
    int64_t d_model = 0;
    int64_t layers = 0;
    int64_t heads = 0;
    int64_t ffn = 0;
    int64_t mel_bins = 0;
    int64_t conv_channels = 0;   // downsample_hidden_size
    int64_t output_dim = 0;      // the decoder's width
    int64_t n_window = 0;        // half a chunk, in mel frames
    int64_t n_window_infer = 0;  // an attention window, in mel frames
    int64_t max_positions = 0;   // rows of the sinusoidal table
};

struct Config {
    qwen3::Config text;
    AudioConfig audio;
    int32_t audio_token = 0;  // <|audio_pad|>
    bool released_layout = true;  // weights under thinker.* (else transformers': model.*)
    std::vector<std::string> languages;  // support_languages (empty: the 30 Qwen3-ASR names)
};

// config.json (released: thinker_config holds audio_config, text_config and the audio token; converted: flat).
Config parse_config(const nlohmann::json& config_json);

// Mel frames of `samples` 16 kHz samples -> encoder rows: chunks of 2 * n_window frames give 13 rows when whole, fewer
// for the last (three halvings of its length).
int64_t encoder_rows(const AudioConfig& a, int64_t mel_frames);

// Qwen3-ASR's transcription format: "language <Name><asr_text>" heads a reply (and forces a language in the prompt),
// "language None" is silence; `languages` the names it knows (empty: Qwen3-ASR's 30); a second of audio takes
// `audio_tokens_per_second` prompt tokens.
std::unique_ptr<Transcription> make_transcription(std::vector<std::string> languages,
                                                  double audio_tokens_per_second);

// The log-mel settings of the checkpoint's preprocessor_config.json in `dir` (the extractor's defaults without one).
audio::MelParams mel_spec(const std::filesystem::path& dir);

// The audio encoder over `weights`, its front end set by `mel`.
std::unique_ptr<AudioEncoder> make_audio_tower(const Config& cfg, const WeightStore& weights,
                                               const audio::MelParams& mel);

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options);

}  // namespace xs::qwen3_asr
