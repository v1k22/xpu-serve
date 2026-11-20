// transformers' log-mel: Qwen3ASRFeatureExtractor._torch_extract_fbank_features and audio_utils.mel_filter_bank
// (models/qwen3_asr/feature_extraction_qwen3_asr.py, audio_utils.py), in double precision until the log.

#include "audio/mel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <thread>
#include <vector>

#include "audio/fft.h"
#include "util/error.h"

namespace xs::audio {

namespace {

constexpr int64_t kFramesPerThread = 500;  // a part's least work: 5 s at a 10 ms hop, about 7 ms on one core
constexpr int64_t kMaxThreads = 16;

// numpy.linspace(start, stop, num): start + i * step, the last point exactly stop.
std::vector<double> linspace(double start, double stop, int num) {
    std::vector<double> y(static_cast<size_t>(num));
    const double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) y[static_cast<size_t>(i)] = i * step + start;
    y.back() = stop;
    return y;
}

// The slaney mel scale (audio_utils.hertz_to_mel / mel_to_hertz with mel_scale="slaney"): linear below 1 kHz,
// logarithmic above.
constexpr double kMinLogHertz = 1000.0;
constexpr double kMinLogMel = 15.0;

double hertz_to_mel(double hertz) {
    if (hertz >= kMinLogHertz) return kMinLogMel + std::log(hertz / kMinLogHertz) * (27.0 / std::log(6.4));
    return 3.0 * hertz / 200.0;
}

double mel_to_hertz(double mel) {
    if (mel >= kMinLogMel) return kMinLogHertz * std::exp(std::log(6.4) / 27.0 * (mel - kMinLogMel));
    return 200.0 * mel / 3.0;
}

// A mel filter's weights over FFT bins first .. first + weights.size() - 1 (the bins where its triangle is not 0).
struct Filter {
    int first = 0;
    std::vector<double> weights;
};

// mel_filter_bank(n_fft / 2 + 1, n_mels, f_min, f_max, sample_rate, norm="slaney", mel_scale="slaney"), each weight
// rounded to float32 as the extractor hands the bank to torch.
std::vector<Filter> mel_filter_bank(const MelParams& p) {
    const int bins = p.n_fft / 2 + 1;
    std::vector<double> edges = linspace(hertz_to_mel(p.f_min), hertz_to_mel(p.f_max), p.n_mels + 2);
    for (double& e : edges) e = mel_to_hertz(e);
    const std::vector<double> bin_hertz = linspace(0.0, static_cast<double>(p.sample_rate / 2), bins);
    std::vector<Filter> filters(static_cast<size_t>(p.n_mels));
    for (size_t m = 0; m < filters.size(); ++m) {
        const double lower = edges[m], centre = edges[m + 1], upper = edges[m + 2];
        const double enorm = 2.0 / (upper - lower);
        Filter& f = filters[m];
        for (int k = 0; k < bins; ++k) {
            const double hz = bin_hertz[static_cast<size_t>(k)];
            const double down = -(lower - hz) / (centre - lower);
            const double up = (upper - hz) / (upper - centre);
            const auto w = static_cast<float>(std::max(0.0, std::min(down, up)) * enorm);
            if (w == 0.0f) continue;
            if (f.weights.empty()) f.first = k;
            f.weights.resize(static_cast<size_t>(k - f.first + 1));
            f.weights.back() = w;
        }
    }
    return filters;
}

}  // namespace

HostTensor log_mel_spectrogram(std::span<const float> samples, const MelParams& p) {
    XS_CHECK(p.sample_rate >= 1 && p.n_fft >= 2 && p.hop_length >= 1 && p.n_mels >= 1 && p.min_length >= 0,
             "audio: mel settings sample_rate ", p.sample_rate, ", n_fft ", p.n_fft, ", hop_length ", p.hop_length,
             ", n_mels ", p.n_mels, ", min_length ", p.min_length);
    XS_CHECK(p.f_min >= 0.0 && p.f_min < p.f_max, "audio: mel frequencies ", p.f_min, " to ", p.f_max, " Hz");
    const auto given = static_cast<int64_t>(samples.size());
    const int64_t length = std::max(given, p.min_length);
    const int64_t pad = p.n_fft / 2;
    if (length <= pad) {
        fail("audio: a clip of ", length, " samples is too short for log-mel features (the STFT's reflect padding "
             "needs more than ", pad, ")");
    }
    const int64_t frames = (length + 2 * pad - p.n_fft) / p.hop_length;  // torch.stft's, less the last
    const int bins = p.n_fft / 2 + 1;

    // torch.hann_window(n_fft), periodic.
    std::vector<double> window(static_cast<size_t>(p.n_fft));
    for (int j = 0; j < p.n_fft; ++j) {
        window[static_cast<size_t>(j)] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * j / p.n_fft);
    }
    const std::vector<Filter> filters = mel_filter_bank(p);

    // The clip zero-padded to `length`, reflected (without repeating the edge sample) over pad samples at each end.
    const auto sample_at = [&](int64_t i) -> double {
        i = i < 0 ? -i : i >= length ? 2 * (length - 1) - i : i;
        return i < given ? samples[static_cast<size_t>(i)] : 0.0;
    };

    HostTensor out(DType::F32, {p.n_mels, frames});
    float* values = out.data<float>();
    // Frames [begin, end) into `values`; returns their largest value.
    const auto compute = [&](int64_t begin, int64_t end) {
        RealFft fft(p.n_fft);
        std::vector<double> frame(static_cast<size_t>(p.n_fft));
        std::vector<std::complex<double>> spectrum(static_cast<size_t>(bins));
        std::vector<double> power(static_cast<size_t>(bins));
        float peak = -std::numeric_limits<float>::infinity();
        for (int64_t t = begin; t < end; ++t) {
            const int64_t start = t * p.hop_length - pad;
            if (start >= 0 && start + p.n_fft <= given) {
                for (int j = 0; j < p.n_fft; ++j) frame[static_cast<size_t>(j)] = samples[start + j] * window[j];
            } else {
                for (int j = 0; j < p.n_fft; ++j) frame[static_cast<size_t>(j)] = sample_at(start + j) * window[j];
            }
            fft.forward(frame.data(), spectrum.data());
            for (int k = 0; k < bins; ++k) power[static_cast<size_t>(k)] = std::norm(spectrum[static_cast<size_t>(k)]);
            for (size_t m = 0; m < filters.size(); ++m) {
                const Filter& f = filters[m];
                double mel = 0.0;
                for (size_t i = 0; i < f.weights.size(); ++i) mel += f.weights[i] * power[f.first + i];
                const auto v = static_cast<float>(std::log10(std::max(mel, 1e-10)));
                values[static_cast<int64_t>(m) * frames + t] = v;
                peak = std::max(peak, v);
            }
        }
        return peak;
    };
    // Frames are independent: parts of them on threads of their own, the first on the caller's.
    const int64_t cores = std::max<int64_t>(1, std::thread::hardware_concurrency());
    const int64_t parts = std::clamp<int64_t>(frames / kFramesPerThread, 1, std::min(cores, kMaxThreads));
    std::vector<float> peaks(static_cast<size_t>(parts));
    {
        std::vector<std::jthread> threads;
        for (int64_t i = 1; i < parts; ++i) {
            threads.emplace_back([&, i] { peaks[i] = compute(frames * i / parts, frames * (i + 1) / parts); });
        }
        peaks[0] = compute(0, frames / parts);
    }
    // As the extractor does it, in float32.
    const float floor = *std::max_element(peaks.begin(), peaks.end()) - 8.0f;
    for (float& v : out.span<float>()) v = (std::max(v, floor) + 4.0f) / 4.0f;
    return out;
}

}  // namespace xs::audio
