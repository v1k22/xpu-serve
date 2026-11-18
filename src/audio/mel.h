#pragma once

#include <cstdint>
#include <span>

#include "tensor/tensor.h"

namespace xs::audio {

// The settings of a Whisper-style log-mel front end; the defaults are Qwen3-ASR's (its preprocessor_config.json
// gives sampling_rate, n_fft, hop_length and feature_size, the number of mels).
struct MelParams {
    int sample_rate = 16000;
    int n_fft = 400;
    int hop_length = 160;
    int n_mels = 128;
    double f_min = 0.0;
    double f_max = 8000.0;
    int64_t min_length = 8000;  // clips shorter than this many samples are zero-padded to it first
};

// The log-mel features transformers' Qwen3ASRFeatureExtractor computes for one clip (WhisperFeatureExtractor's math)
// as Qwen3ASRProcessor calls it: padding to the longest clip of the batch, which for one clip is none (never to 30 s),
// and no truncation. `samples` are mono at params.sample_rate. Steps: zero-pad to min_length; torch.stft with a
// periodic Hann window of n_fft, centred frames (reflect padding of n_fft / 2), the last frame dropped; the power
// spectrum through mel_filter_bank(norm="slaney", mel_scale="slaney") over [f_min, f_max]; log10 of at least 1e-10,
// floored at the clip's maximum minus 8; then (x + 4) / 4. Returns F32 [n_mels, frames] with
// frames = (length + 2 * (n_fft / 2) - n_fft) / hop_length, length counted after the min_length padding (length /
// hop_length for an even n_fft). The extractor goes on to pad the frame axis with zeros to a multiple of the
// encoder's chunk and masks the padding; that layout is the model's: these are the frames the mask marks valid.
// Computed in double precision until the log, so it differs from the extractor only by the extractor's float32
// rounding (largest in bins far below their frame's peak).
HostTensor log_mel_spectrogram(std::span<const float> samples, const MelParams& params = {});

}  // namespace xs::audio
