#pragma once

#include <span>
#include <vector>

namespace xs::audio {

// Band-limited resampling as torchaudio.functional.resample(x, from_rate, to_rate) with its defaults
// (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99), the resampler vLLM runs: a windowed-sinc filter at
// 0.99 of the lower Nyquist, `ceil(n * to_rate / from_rate)` samples out, zeros beyond both ends. Within float32
// rounding of torchaudio on float32 input. The same rate returns the input unchanged. Rates up to kMaxSampleRate.
std::vector<float> resample(std::span<const float> x, int from_rate, int to_rate);

}  // namespace xs::audio
