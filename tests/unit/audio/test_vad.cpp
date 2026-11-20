// The energy VAD (audio/vad.h): where turns start and stop on synthetic streams at 24 kHz (frames of 480 samples):
// the prefix and the closing silence, clicks ignored, short pauses kept in a turn, the threshold, a loud room, and the
// same events however the stream is cut into pushes.
#include <doctest/doctest.h>

#include <cmath>
#include <random>
#include <vector>

#include "audio/vad.h"

using xs::audio::EnergyVad;
using xs::audio::VadEvent;
using xs::audio::VadSettings;

namespace {

constexpr int kRate = 24000;

// Gaussian noise at `db` dBFS (RMS) for `ms` milliseconds.
void add(std::vector<float>& s, int ms, double db, uint32_t seed = 1) {
    std::mt19937 rng(seed + static_cast<uint32_t>(s.size()));
    std::normal_distribution<float> noise(0.0f, static_cast<float>(std::pow(10.0, db / 20.0)));
    for (int i = 0; i < ms * kRate / 1000; ++i) s.push_back(noise(rng));
}

std::vector<VadEvent> run(const std::vector<float>& s, VadSettings settings = {}, size_t piece = 2400) {
    EnergyVad vad(kRate, settings);
    std::vector<VadEvent> all;
    for (size_t at = 0; at < s.size(); at += piece) {
        const size_t n = std::min(piece, s.size() - at);
        for (const VadEvent& e : vad.push(std::span(s).subspan(at, n))) all.push_back(e);
    }
    return all;
}

int64_t at_ms(int ms) { return int64_t{ms} * kRate / 1000; }

}  // namespace

TEST_SUITE("audio vad") {
    TEST_CASE("audio vad: a turn starts its prefix before the first speech and stops after the closing silence") {
        std::vector<float> s;
        add(s, 1000, -60.0);
        add(s, 2000, -20.0);
        add(s, 1000, -60.0);
        const auto events = run(s);
        REQUIRE(events.size() == 2);
        CHECK(events[0].kind == VadEvent::Kind::SpeechStarted);
        CHECK(events[0].sample == at_ms(1000 - 300));
        CHECK(events[1].kind == VadEvent::Kind::SpeechStopped);
        CHECK(events[1].sample == at_ms(3000 + 500));

        VadSettings quick;
        quick.prefix_ms = 0;
        quick.silence_ms = 200;
        const auto q = run(s, quick);
        REQUIRE(q.size() == 2);
        CHECK(q[0].sample == at_ms(1000));
        CHECK(q[1].sample == at_ms(3000 + 200));
    }

    TEST_CASE("audio vad: the same events however the stream is cut into pushes") {
        std::vector<float> s;
        add(s, 700, -55.0);
        add(s, 1300, -18.0);
        add(s, 900, -55.0);
        add(s, 800, -25.0);
        add(s, 900, -55.0);
        const auto whole = run(s, {}, s.size());
        REQUIRE(whole.size() == 4);
        for (size_t piece : {1, 7, 480, 1000, 4801}) {
            const auto cut = run(s, {}, piece);
            REQUIRE(cut.size() == whole.size());
            for (size_t i = 0; i < cut.size(); ++i) {
                CHECK(cut[i].kind == whole[i].kind);
                CHECK(cut[i].sample == whole[i].sample);
            }
        }
        CHECK(whole[2].sample >= whole[1].sample);  // a turn starts no earlier than the one before ended
    }

    TEST_CASE("audio vad: a click is not a turn; 60 ms of speech is") {
        std::vector<float> click;
        add(click, 1000, -60.0);
        add(click, 40, -10.0);
        add(click, 1000, -60.0);
        CHECK(run(click).empty());
        std::vector<float> word;
        add(word, 1000, -60.0);
        add(word, 60, -10.0);
        add(word, 1000, -60.0);
        CHECK(run(word).size() == 2);
    }

    TEST_CASE("audio vad: a pause shorter than the silence keeps one turn; a longer one ends it") {
        std::vector<float> s;
        add(s, 500, -60.0);
        add(s, 800, -20.0);
        add(s, 400, -60.0);  // < 500 ms
        add(s, 800, -20.0);
        add(s, 700, -60.0);  // > 500 ms
        add(s, 800, -20.0);
        add(s, 1000, -60.0);
        const auto events = run(s);
        REQUIRE(events.size() == 4);
        CHECK(events[1].sample == at_ms(500 + 800 + 400 + 800 + 500));
        CHECK(events[2].kind == VadEvent::Kind::SpeechStarted);
        CHECK(events[2].sample == at_ms(500 + 800 + 400 + 800 + 700 - 200));  // the prefix stops at the last turn
    }

    TEST_CASE("audio vad: the threshold sets how loud speech must be") {
        std::vector<float> quiet;
        add(quiet, 1000, -70.0);
        add(quiet, 1000, -42.0);
        add(quiet, 1000, -70.0);
        CHECK(run(quiet).empty());  // 0.5: -37.5 dBFS
        VadSettings sensitive;
        sensitive.threshold = 0.2;  // -48 dBFS
        CHECK(run(quiet, sensitive).size() == 2);
        std::vector<float> loud;
        add(loud, 1000, -70.0);
        add(loud, 1000, -25.0);
        add(loud, 1000, -70.0);
        VadSettings strict;
        strict.threshold = 1.0;  // -20 dBFS
        CHECK(run(loud, strict).empty());
    }

    TEST_CASE("audio vad: in a loud room speech must clear the noise floor, which the stream's start sets") {
        std::vector<float> s;
        add(s, 1500, -30.0);  // a room above the absolute level
        add(s, 1000, -8.0);
        add(s, 1500, -30.0);
        const auto events = run(s);
        REQUIRE(events.size() == 2);
        CHECK(events[0].sample == at_ms(1500 - 300));
        CHECK(events[1].sample == at_ms(2500 + 500));
    }
}
