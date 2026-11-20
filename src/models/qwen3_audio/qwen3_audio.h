#pragma once

// Qwen's audio encoder, Qwen3-ASR's and Qwen3-Omni's (transformers' Qwen3ASRAudioEncoder and Qwen3OmniMoeAudioEncoder,
// the same code at two sizes). Audio becomes 128-bin log-mel frames (16 kHz, hop 160); every chunk of 2 * n_window
// frames goes through three 3 x 3 stride-2 convolutions to 13 rows, sinusoidal positions that restart in each chunk,
// then transformer layers that attend within windows of n_window_infer frames (8 chunks, 104 rows), and two
// projections into the decoder's width.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "audio/mel.h"
#include "models/model.h"

namespace xs::qwen3_audio {

struct Config {
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

// An audio_config (`where` names it in errors); a non-GELU activation or scaled embeddings are refused by name.
Config parse_config(const nlohmann::json& audio_config, const std::string& where);

// Mel frames -> encoder rows: chunks of 2 * n_window frames give 13 rows when whole, fewer for the last (three halvings
// of its length).
int64_t encoder_rows(const Config& c, int64_t mel_frames);
// The encoder's rows per second of audio, over a minute.
double rows_per_second(const Config& c, const audio::MelParams& mel);

// The log-mel settings of the checkpoint's preprocessor_config.json in `dir` (the extractor's defaults without one).
audio::MelParams mel_spec(const std::filesystem::path& dir);

// Where the encoder's weights are: `tower` the prefix of its convolutions and layers ("thinker.audio_tower."), `proj1`
// and `proj2` the stems of its two projections (inside the tower in a released checkpoint, a projector beside it in
// transformers' Qwen3-ASR layout).
struct WeightNames {
    std::string tower;
    std::string proj1;
    std::string proj2;
};

// The encoder over `weights`, its front end set by `mel`; `audio_token` is what the chat template renders once per
// clip and what the prompt holds once per encoder row.
std::unique_ptr<AudioEncoder> make_encoder(const Config& c, const WeightStore& weights, const audio::MelParams& mel,
                                           WeightNames names, int32_t audio_token);

}  // namespace xs::qwen3_audio
