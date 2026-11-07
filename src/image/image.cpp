#include "image/image.h"

#include <climits>
#include <cstring>
#include <memory>

#include <stb_image.h>

#include "util/error.h"

namespace xs::image {

Image decode(std::span<const uint8_t> encoded) {
    XS_CHECK(!encoded.empty(), "the image is empty");
    XS_CHECK(encoded.size() <= static_cast<size_t>(INT_MAX), "the image is larger than 2 GiB");
    int width = 0, height = 0, channels = 0;
    const std::unique_ptr<stbi_uc, void (*)(void*)> pixels(
        stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width, &height, &channels, 3),
        stbi_image_free);
    if (!pixels) {
        fail("the image could not be decoded (", stbi_failure_reason(), "); supported formats: PNG, JPEG, BMP, GIF");
    }
    Image img;
    img.width = width;
    img.height = height;
    img.rgb.assign(pixels.get(), pixels.get() + static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    return img;
}

uint64_t content_hash(const Image& img) {
    // FNV-1a over 64-bit words, then the murmur3 finalizer so nearby inputs spread over all bits.
    uint64_t h = 0xcbf29ce484222325ULL;
    const auto mix = [&h](uint64_t word) {
        h ^= word;
        h *= 0x100000001b3ULL;
    };
    mix(static_cast<uint64_t>(img.width));
    mix(static_cast<uint64_t>(img.height));
    const size_t words = img.rgb.size() / 8;
    for (size_t i = 0; i < words; ++i) {
        uint64_t w;
        std::memcpy(&w, img.rgb.data() + i * 8, 8);
        mix(w);
    }
    uint64_t tail = 0;
    std::memcpy(&tail, img.rgb.data() + words * 8, img.rgb.size() - words * 8);
    mix(tail);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

}  // namespace xs::image
