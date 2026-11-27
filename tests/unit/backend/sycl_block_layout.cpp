// The SYCL backend's host-side rewrite of the GGUF block formats a pack's kernels read (Q4_0, Q4_1, Q8_0): what goes
// to the device is the file's bytes, no more, and holds the values ggml decodes.
#ifdef XS_HAVE_SYCL

#include <doctest/doctest.h>

#include <cstring>
#include <vector>

#include "backend/sycl/sycl_block_layout.h"
#include "backend_test_util.h"
#include "quant/block_decode.h"
#include "util/half.h"

using namespace xs;

TEST_CASE("sycl block layout: Q4_0, Q4_1 and Q8_0 as a pack's streams take their file's bytes and hold its values") {
    // 37 rows of 96 columns: three blocks a row
    const int64_t rows = 37, cols = 96, groups = cols / 32;
    for (const BlockFormat f : {BlockFormat::Q4_0, BlockFormat::Q4_1, BlockFormat::Q8_0}) {
        CAPTURE(block_format_name(f));
        REQUIRE(sycl_be::block_as_pack(f));
        const std::vector<uint8_t> blocks = test::random_blocks(f, rows * groups, 61);
        const sycl_be::PackImage image = sycl_be::block_pack_image(f, blocks.data(), rows, cols);
        const size_t device_bytes = image.codes.size() + sizeof(uint16_t) * (image.scales.size() + image.biases.size());
        CHECK(device_bytes == blocks.size());
        CHECK(device_bytes == block_storage_bytes(f, rows * cols));
        CHECK(image.q4 == (f != BlockFormat::Q8_0));
        CHECK(image.biases.empty() == (f != BlockFormat::Q4_1));

        std::vector<float> want(static_cast<size_t>(rows * cols));
        quant::decode_blocks(f, blocks.data(), rows * groups, want.data());
        size_t wrong = 0;
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                const auto at = static_cast<size_t>(r * cols + c);
                const auto g = static_cast<size_t>(r * groups + c / 32);
                const float scale = f16_to_f32(image.scales[g]);
                float got = 0;
                if (!image.q4) {
                    got = scale * static_cast<float>(static_cast<int8_t>(image.codes[at]));
                } else {
                    const uint8_t byte = image.codes[at / 2];
                    const int code = (c & 1) == 0 ? (byte & 0x0F) : (byte >> 4);
                    got = image.biases.empty() ? scale * static_cast<float>(code - 8)
                                               : scale * static_cast<float>(code) + f16_to_f32(image.biases[g]);
                }
                if (std::memcmp(&got, &want[at], sizeof(float)) != 0) ++wrong;
            }
        }
        CHECK(wrong == 0);
    }
}

#endif  // XS_HAVE_SYCL
