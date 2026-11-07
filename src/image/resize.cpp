// Antialiased bicubic resize, bit-exact with PyTorch's uint8 CPU kernel (aten/src/ATen/native/cpu/UpSampleKernel.cpp,
// _compute_index_ranges_int16_weights + UpSampleKernelAVXAntialias.h), itself a port of Pillow's ImagingResample.
// Hugging Face image processors resize through torchvision, which calls that kernel for uint8 images.

#include <algorithm>
#include <cmath>

#include "image/image.h"
#include "util/error.h"

namespace xs::image {

namespace {

// Keys cubic convolution with a = -0.5 (torch's HelperInterpCubic::aa_filter), written as torch writes it so the
// double weights round the same way.
double cubic_filter(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2.0) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
    return 0.0;
}

// Per output position: the first input index, how many inputs contribute, and their int16 weights with `precision`
// fractional bits.
struct AxisWeights {
    std::vector<int64_t> first;
    std::vector<int64_t> count;
    std::vector<int16_t> weights;  // `stride` entries per output position
    int64_t stride = 0;
    int precision = 0;
};

AxisWeights axis_weights(int64_t in_size, int64_t out_size) {
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    constexpr double kHalfWidth = 2.0;  // bicubic support at scale 1
    const double support = scale >= 1.0 ? kHalfWidth * scale : kHalfWidth;
    const double inv_scale = scale >= 1.0 ? 1.0 / scale : 1.0;

    AxisWeights w;
    w.stride = static_cast<int64_t>(std::ceil(support)) * 2 + 1;
    w.first.resize(static_cast<size_t>(out_size));
    w.count.resize(static_cast<size_t>(out_size));
    std::vector<double> real(static_cast<size_t>(out_size * w.stride), 0.0);
    double max_weight = 0.0;
    for (int64_t i = 0; i < out_size; ++i) {
        const double center = scale * (static_cast<double>(i) + 0.5);
        const int64_t first = std::max<int64_t>(static_cast<int64_t>(center - support + 0.5), 0);
        const int64_t count = std::min<int64_t>(static_cast<int64_t>(center + support + 0.5), in_size) - first;
        double* row = real.data() + i * w.stride;
        double total = 0.0;
        for (int64_t j = 0; j < count; ++j) {
            row[j] = cubic_filter((static_cast<double>(j + first) - center + 0.5) * inv_scale);
            total += row[j];
        }
        if (total != 0.0) {
            for (int64_t j = 0; j < count; ++j) {
                row[j] /= total;
                max_weight = std::max(max_weight, row[j]);
            }
        }
        w.first[static_cast<size_t>(i)] = first;
        w.count[static_cast<size_t>(i)] = count;
    }
    // The most fractional bits for which the largest weight still fits int16.
    for (w.precision = 0; w.precision < 22; ++w.precision) {
        if (static_cast<int>(0.5 + max_weight * (1 << (w.precision + 1))) >= (1 << 15)) break;
    }
    XS_CHECK(w.precision > 0, "image resize: weights without precision");
    w.weights.resize(real.size());
    for (size_t k = 0; k < real.size(); ++k) {
        const double v = real[k] * (1 << w.precision);
        w.weights[k] = static_cast<int16_t>(v < 0 ? static_cast<int>(-0.5 + v) : static_cast<int>(0.5 + v));
    }
    return w;
}

uint8_t to_u8(int32_t acc, int precision) {
    return static_cast<uint8_t>(std::clamp(acc >> precision, 0, 255));
}

// Resizes along x (each row independently).
Image horizontal(const Image& src, int64_t out_width) {
    const AxisWeights w = axis_weights(src.width, out_width);
    Image out;
    out.width = out_width;
    out.height = src.height;
    out.rgb.resize(static_cast<size_t>(out.width * out.height * 3));
    const int32_t rounding = 1 << (w.precision - 1);
    for (int64_t y = 0; y < src.height; ++y) {
        for (int64_t x = 0; x < out_width; ++x) {
            const int16_t* k = w.weights.data() + x * w.stride;
            const int64_t first = w.first[static_cast<size_t>(x)];
            const int64_t count = w.count[static_cast<size_t>(x)];
            for (int c = 0; c < 3; ++c) {
                int32_t acc = rounding;
                for (int64_t j = 0; j < count; ++j)
                    acc += static_cast<int32_t>(src.at(y, first + j, c)) * k[j];
                out.rgb[static_cast<size_t>((y * out_width + x) * 3 + c)] = to_u8(acc, w.precision);
            }
        }
    }
    return out;
}

// Resizes along y (each column independently).
Image vertical(const Image& src, int64_t out_height) {
    const AxisWeights w = axis_weights(src.height, out_height);
    Image out;
    out.width = src.width;
    out.height = out_height;
    out.rgb.resize(static_cast<size_t>(out.width * out.height * 3));
    const int32_t rounding = 1 << (w.precision - 1);
    for (int64_t y = 0; y < out_height; ++y) {
        const int16_t* k = w.weights.data() + y * w.stride;
        const int64_t first = w.first[static_cast<size_t>(y)];
        const int64_t count = w.count[static_cast<size_t>(y)];
        for (int64_t x = 0; x < src.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                int32_t acc = rounding;
                for (int64_t j = 0; j < count; ++j)
                    acc += static_cast<int32_t>(src.at(first + j, x, c)) * k[j];
                out.rgb[static_cast<size_t>((y * src.width + x) * 3 + c)] = to_u8(acc, w.precision);
            }
        }
    }
    return out;
}

}  // namespace

Image resize_bicubic(const Image& src, int64_t width, int64_t height) {
    XS_CHECK(src.width > 0 && src.height > 0 && width > 0 && height > 0, "image resize from ", src.width, "x",
             src.height, " to ", width, "x", height);
    if (width == src.width && height == src.height) return src;
    if (width == src.width) return vertical(src, height);
    const Image wide = horizontal(src, width);
    return height == src.height ? wide : vertical(wide, height);
}

}  // namespace xs::image
