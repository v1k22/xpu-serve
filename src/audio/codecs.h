#pragma once

// The decoders behind audio::decode, one per container; decode sniffs the bytes and checks the sample rate.

#include <span>

#include "audio/audio.h"

namespace xs::audio::codecs {

Audio decode_wav(std::span<const uint8_t> bytes);
Audio decode_mp3(std::span<const uint8_t> bytes);
Audio decode_flac(std::span<const uint8_t> bytes);

// Frames of `channels` interleaved samples averaged into one, as librosa.to_mono does (a float32 sum in channel
// order, divided by the count). One channel is returned as it is.
std::vector<float> average_channels(std::vector<float> interleaved, int channels);

}  // namespace xs::audio::codecs
