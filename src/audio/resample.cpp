// torchaudio's sinc_interp_hann resampling (torchaudio/functional/functional.py, _get_sinc_resample_kernel and
// _apply_sinc_resample_kernel): with g = gcd(from, to), output sample i * (to / g) + p is a filter of phase p over
// input samples from i * (from / g) - width on.

#include "audio/resample.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

#include "audio/audio.h"
#include "util/error.h"

namespace xs::audio {

namespace {

constexpr int kLowpassFilterWidth = 6;
constexpr double kRolloff = 0.99;

// One phase's filter: weights for input offsets first .. first + weights.size() - 1 of its frame.
struct Phase {
    int64_t first = 0;
    std::vector<float> weights;
};

// torchaudio computes the kernel in the waveform's dtype, so these are its float32 steps in its order (scalars
// rounded to float as torch rounds them). Taps whose argument torch clamps to the filter's edge weigh about 1e-23
// (cos(pi / 2)^2 in float32 times a sinc near zero) and are left out.
std::vector<Phase> phases(int64_t orig, int64_t neu, int64_t width) {
    const double base = static_cast<double>(std::min(orig, neu)) * kRolloff;
    const float base_f = static_cast<float>(base);
    const float pi_f = static_cast<float>(std::numbers::pi);
    const float scale_f = static_cast<float>(base / static_cast<double>(orig));
    const float lw = kLowpassFilterWidth;
    const int64_t taps = 2 * width + orig;
    std::vector<Phase> out(static_cast<size_t>(neu));
    for (int64_t p = 0; p < neu; ++p) {
        // |t| < lw for offsets k with |(k - width) / orig - p / neu| < lw / base, and a tap of margin for rounding.
        const double centre = static_cast<double>(width) + static_cast<double>(orig * p) / static_cast<double>(neu);
        const double reach = static_cast<double>(orig) * kLowpassFilterWidth / base;
        const int64_t lo = std::max<int64_t>(0, static_cast<int64_t>(std::floor(centre - reach)) - 1);
        const int64_t hi = std::min<int64_t>(taps - 1, static_cast<int64_t>(std::ceil(centre + reach)) + 1);
        const float shift = static_cast<float>(-p) / static_cast<float>(neu);
        Phase& phase = out[static_cast<size_t>(p)];
        phase.first = lo;
        for (int64_t k = lo; k <= hi; ++k) {
            const float idx = static_cast<float>(k - width) / static_cast<float>(orig);
            float t = (shift + idx) * base_f;
            t = std::clamp(t, -lw, lw);
            const float c = std::cos(t * pi_f / lw / 2.0f);
            const float window = c * c;
            t *= pi_f;
            const float sinc = t == 0.0f ? 1.0f : std::sin(t) / t;
            phase.weights.push_back(sinc * (window * scale_f));
        }
    }
    return out;
}

}  // namespace

std::vector<float> resample(std::span<const float> x, int from_rate, int to_rate) {
    for (const int rate : {from_rate, to_rate}) {
        if (rate < 1 || rate > kMaxSampleRate) {
            fail("audio: resampling at ", rate, " Hz is not supported (1 to ", kMaxSampleRate, " Hz are)");
        }
    }
    if (from_rate == to_rate) return {x.begin(), x.end()};
    const int64_t g = std::gcd(from_rate, to_rate);
    const int64_t orig = from_rate / g;
    const int64_t neu = to_rate / g;
    const double base = static_cast<double>(std::min(orig, neu)) * kRolloff;
    const auto width =
        static_cast<int64_t>(std::ceil(static_cast<double>(kLowpassFilterWidth * orig) / base));
    const std::vector<Phase> filters = phases(orig, neu, width);

    const auto n = static_cast<int64_t>(x.size());
    const int64_t out_len = (neu * n + orig - 1) / orig;
    std::vector<float> out(static_cast<size_t>(out_len));
    for (int64_t j = 0; j < out_len; ++j) {
        const int64_t frame = j / neu;
        const Phase& phase = filters[static_cast<size_t>(j % neu)];
        const int64_t start = frame * orig - width + phase.first;  // input index of the first weight
        const auto count = static_cast<int64_t>(phase.weights.size());
        const int64_t k0 = std::max<int64_t>(0, -start);
        const int64_t k1 = std::min<int64_t>(count, n - start);
        double sum = 0.0;
        for (int64_t k = k0; k < k1; ++k) {
            sum += static_cast<double>(phase.weights[static_cast<size_t>(k)]) * x[static_cast<size_t>(start + k)];
        }
        out[static_cast<size_t>(j)] = static_cast<float>(sum);
    }
    return out;
}

}  // namespace xs::audio
