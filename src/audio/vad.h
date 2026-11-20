#pragma once

// Voice activity from frame energy: where a stream's turns of speech start and stop (a realtime session's server
// VAD). Not a model: a 20 ms frame is speech when its level clears both an absolute level, set by `threshold`, and the
// noise floor the detector tracks (down quickly, up slowly; the stream's first 200 ms only set it) by 10 dB; a turn
// starts after 60 ms of speech frames in a row and stops after `silence_ms` of frames that are not.

#include <cstdint>
#include <span>
#include <vector>

namespace xs::audio {

struct VadSettings {
    double threshold = 0.5;  // 0 to 1: the absolute level speech must reach, -55 dBFS at 0 to -20 dBFS at 1
    int prefix_ms = 300;     // audio kept before the first speech frame
    int silence_ms = 500;    // silence that ends a turn
};

struct VadEvent {
    enum class Kind { SpeechStarted, SpeechStopped };
    Kind kind;
    // Started: the turn's first sample (prefix included, never before the stream's start or the previous turn's end).
    // Stopped: the sample after the turn's closing silence.
    int64_t sample = 0;
};

class EnergyVad {
public:
    EnergyVad(int sample_rate, VadSettings settings);

    // The stream's next samples; the events they complete, in order.
    std::vector<VadEvent> push(std::span<const float> samples);
    // Ends the turn in progress without an event (the turn was committed or cleared another way); later turns start
    // no earlier than the samples pushed so far.
    void end_turn();
    bool in_speech() const { return in_speech_; }

private:
    void frame(double mean_square, int64_t end);

    int sample_rate_;
    int64_t frame_samples_;
    VadSettings settings_;
    int64_t pushed_ = 0;          // samples pushed so far
    double partial_sum_ = 0.0;    // squares of the current frame's samples so far
    int64_t partial_count_ = 0;
    double noise_db_ = -70.0;
    int64_t frames_ = 0;
    bool in_speech_ = false;
    int64_t speech_run_ = 0;      // speech frames in a row while not in a turn
    int64_t silence_run_ = 0;     // frames that are not speech in a row while in a turn
    int64_t floor_sample_ = 0;    // a turn starts at or after this sample
    std::vector<VadEvent> events_;
};

}  // namespace xs::audio
