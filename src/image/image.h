#pragma once

// Decoded images and the pixel operations model preprocessors share. Model-specific preprocessing (sizes, patches,
// normalization) lives with the model code.

#include <cstdint>
#include <span>
#include <vector>

namespace xs::image {

// 8-bit RGB, row-major, interleaved (r, g, b per pixel).
struct Image {
    int64_t width = 0;
    int64_t height = 0;
    std::vector<uint8_t> rgb;

    uint8_t at(int64_t y, int64_t x, int channel) const {
        return rgb[static_cast<size_t>((y * width + x) * 3 + channel)];
    }
};

// Decodes a PNG, JPEG, BMP or GIF (first frame) held in memory. Grayscale is replicated to RGB and alpha is dropped,
// as PIL's convert("RGB") does. Throws xs::Error naming the problem for anything else.
Image decode(std::span<const uint8_t> encoded);

// Bicubic resize with antialiasing, bit-exact with torchvision's resize(uint8 tensor, BICUBIC, antialias=True) on
// CPU, which is what Hugging Face image processors run: separable passes (horizontal first), Keys cubic with
// a = -0.5 stretched by the scale when shrinking, int16 fixed-point weights, uint8 rounding after each pass.
Image resize_bicubic(const Image& src, int64_t width, int64_t height);

// 64-bit content hash of the size and pixels. Two images with the same hash are treated as the same image.
uint64_t content_hash(const Image& img);

}  // namespace xs::image
