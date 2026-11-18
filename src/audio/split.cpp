#include "audio/split.h"

#include <algorithm>
#include <cmath>

namespace xs::audio {

std::vector<std::pair<int64_t, int64_t>> split_at_quiet(const Audio& clip, double max_seconds) {
    const auto total = static_cast<int64_t>(clip.samples.size());
    const double rate = clip.sample_rate;
    std::vector<std::pair<int64_t, int64_t>> pieces;
    if (static_cast<double>(total) / rate <= max_seconds) {
        pieces.emplace_back(0, total);
        return pieces;
    }
    const auto max_len = static_cast<int64_t>(max_seconds * rate);
    const auto expand = static_cast<int64_t>(5.0 * rate);
    const int64_t window = std::max<int64_t>(4, static_cast<int64_t>(0.1 * rate));
    const float* x = clip.samples.data();
    int64_t start = 0;
    while (total - start > max_len) {
        const int64_t cut = start + max_len;
        const int64_t left = std::max(start, cut - expand), right = std::min(total, cut + expand);
        int64_t boundary = cut;
        if (right - left > window) {
            // the window of `window` samples with the least sum of |x|, the first on a tie (np.argmin)
            double sum = 0;
            for (int64_t i = left; i < left + window; ++i) sum += std::fabs(x[i]);
            double best = sum;
            int64_t best_at = left;
            for (int64_t i = left + 1; i + window <= right; ++i) {
                sum += std::fabs(x[i + window - 1]) - std::fabs(x[i - 1]);
                if (sum < best) {
                    best = sum;
                    best_at = i;
                }
            }
            boundary = best_at;
            for (int64_t i = best_at + 1; i < best_at + window; ++i) {
                if (std::fabs(x[i]) < std::fabs(x[boundary])) boundary = i;
            }
        }
        boundary = std::min(std::max(boundary, start + 1), total);
        pieces.emplace_back(start, boundary);
        start = boundary;
    }
    pieces.emplace_back(start, total);
    return pieces;
}

}  // namespace xs::audio
