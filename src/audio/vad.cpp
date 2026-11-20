#include "audio/vad.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "util/error.h"

namespace xs::audio {

namespace {

constexpr int kFrameMs = 20;
constexpr int kStartMs = 60;            // speech frames in a row that start a turn: a click does not
constexpr double kNoiseMarginDb = 10.0;
constexpr double kNoiseDown = 0.3;      // a frame below the noise floor pulls it this fraction of the way down
constexpr double kNoiseUp = 0.004;      // one above pulls it up this fraction: about 5 s to follow a louder room
constexpr int kLearnMs = 200;           // the stream's start only sets the noise floor, quickly either way

}  // namespace

EnergyVad::EnergyVad(int sample_rate, VadSettings settings)
    : sample_rate_(sample_rate), frame_samples_(int64_t{sample_rate} * kFrameMs / 1000), settings_(settings) {
    XS_CHECK(sample_rate >= 1000, "vad: a sample rate of ", sample_rate, " Hz");
    XS_CHECK(settings.threshold >= 0.0 && settings.threshold <= 1.0 && settings.prefix_ms >= 0 &&
                 settings.silence_ms >= 0,
             "vad: threshold ", settings.threshold, " (0 to 1), prefix ", settings.prefix_ms, " ms, silence ",
             settings.silence_ms, " ms");
}

std::vector<VadEvent> EnergyVad::push(std::span<const float> samples) {
    for (const float x : samples) {
        partial_sum_ += static_cast<double>(x) * x;
        ++pushed_;
        if (++partial_count_ == frame_samples_) {
            frame(partial_sum_ / static_cast<double>(frame_samples_), pushed_);
            partial_sum_ = 0.0;
            partial_count_ = 0;
        }
    }
    return std::exchange(events_, {});
}

void EnergyVad::end_turn() {
    in_speech_ = false;
    speech_run_ = silence_run_ = 0;
    floor_sample_ = pushed_;
}

void EnergyVad::frame(double mean_square, int64_t end) {
    const double level = 10.0 * std::log10(mean_square + 1e-12);
    const double absolute = -55.0 + 35.0 * settings_.threshold;
    const bool learning = ++frames_ * kFrameMs <= kLearnMs;
    const bool speech = !learning && level >= absolute && level >= noise_db_ + kNoiseMarginDb;
    noise_db_ += (level < noise_db_ || learning ? kNoiseDown : kNoiseUp) * (level - noise_db_);
    if (!in_speech_) {
        speech_run_ = speech ? speech_run_ + 1 : 0;
        if (speech_run_ * kFrameMs < kStartMs) return;
        const int64_t first = end - speech_run_ * frame_samples_;
        const int64_t prefix = int64_t{settings_.prefix_ms} * sample_rate_ / 1000;
        events_.push_back({VadEvent::Kind::SpeechStarted, std::max(floor_sample_, first - prefix)});
        in_speech_ = true;
        speech_run_ = silence_run_ = 0;
        return;
    }
    silence_run_ = speech ? 0 : silence_run_ + 1;
    if (silence_run_ == 0 || silence_run_ * kFrameMs < settings_.silence_ms) return;
    events_.push_back({VadEvent::Kind::SpeechStopped, end});
    in_speech_ = false;
    silence_run_ = 0;
    floor_sample_ = end;
}

}  // namespace xs::audio
