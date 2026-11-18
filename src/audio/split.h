#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "audio/audio.h"

namespace xs::audio {

// [first, last) sample ranges that cover `clip` in order, none longer than about `max_seconds`: each cut goes to the
// quietest 100 ms (by the sum of |sample|) within 5 s of the limit, at that window's quietest sample, as the Qwen3-ASR
// package's split_audio_into_chunks does. A clip that fits is one range.
std::vector<std::pair<int64_t, int64_t>> split_at_quiet(const Audio& clip, double max_seconds);

}  // namespace xs::audio
