#include "image/image.h"

#include <climits>
#include <memory>

#include <stb_image.h>

#include "util/error.h"
#include "util/hash.h"

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
    ContentHash h;
    h.add(static_cast<uint64_t>(img.width));
    h.add(static_cast<uint64_t>(img.height));
    h.add_bytes(img.rgb);
    return h.finish();
}

}  // namespace xs::image
