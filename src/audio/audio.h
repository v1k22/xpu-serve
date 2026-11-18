#pragma once

// Audio files decoded into the mono samples speech models read. Resampling is resample.h, log-mel features mel.h;
// what a model does beyond that lives with the model code.

#include <cstdint>
#include <span>
#include <vector>

namespace xs::audio {

// The sample rates decode and resample accept: every rate audio files use (352.8 kHz is the highest common one).
inline constexpr int kMaxSampleRate = 768000;

// Mono audio at sample_rate Hz, samples in [-1, 1] (a float file's values as stored, which may lie outside).
struct Audio {
    int sample_rate = 0;
    std::vector<float> samples;
};

// Decodes a WAV, MP3 or FLAC file held in memory, telling them apart by their bytes, as librosa.load(file, sr=None,
// mono=True) reads it: integers divided by 2^(bits - 1) (8-bit WAV is unsigned around 128), float as stored,
// channels averaged into one; an MP3's encoder delay and padding (its LAME tag) left out.
// WAV: RIFF, RF64 or BW64; integer PCM of 1 to 32 bits (read by the bytes each sample takes), IEEE float of 32 or 64
// bits, plain or WAVE_FORMAT_EXTENSIBLE; any number of channels; unknown chunks skipped.
// Throws xs::Error naming the format of anything else (Ogg, MP4 / M4A, WebM, AAC, AIFF, ...) with an ffmpeg command
// that converts it, and for truncated or malformed files.
Audio decode(std::span<const uint8_t> bytes);

// 64-bit content hash of the sample rate and samples. Two clips with the same hash are treated as the same clip.
uint64_t content_hash(const Audio& audio);

enum class WavSample { Pcm16, Float32 };

// A mono WAV file of the clip. Pcm16 scales by 32768, rounds to nearest and clips to [-32768, 32767], so a clip
// decoded from 16-bit PCM is written back bit for bit; Float32 keeps the samples as they are. Samples must be finite.
std::vector<uint8_t> encode_wav(const Audio& audio, WavSample sample = WavSample::Pcm16);

}  // namespace xs::audio
