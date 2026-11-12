#include "backend/sycl/sycl_block_layout.h"

#include <algorithm>
#include <cstring>
#include <thread>

#include "util/error.h"
#include "util/half.h"

namespace xs::sycl_be {

namespace {

constexpr size_t kAlign = 64;

// Rows are independent: a few threads each take a run of them.
template <typename Fn>
void for_rows(int64_t rows, int64_t cols, Fn&& fn) {
    const int64_t work = rows * cols;
    const auto threads = static_cast<int64_t>(
        std::clamp<int64_t>(std::min<int64_t>(std::thread::hardware_concurrency(), work >> 22), 1, 32));
    if (threads == 1) {
        for (int64_t r = 0; r < rows; ++r)
            fn(r);
        return;
    }
    std::vector<std::thread> pool;
    for (int64_t t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            for (int64_t r = rows * t / threads; r < rows * (t + 1) / threads; ++r)
                fn(r);
        });
    }
    for (auto& th : pool)
        th.join();
}

// Writers of a row's code streams in column order.
inline void put_nibble(uint8_t* row, int64_t c, uint32_t v) {
    row[c >> 1] = static_cast<uint8_t>(row[c >> 1] | (v << (4 * (c & 1))));
}
inline void put_pair(uint8_t* row, int64_t c, uint32_t v) {
    row[c >> 2] = static_cast<uint8_t>(row[c >> 2] | (v << (2 * (c & 3))));
}
inline void put_bit(uint8_t* row, int64_t c, uint32_t v) {
    row[c >> 3] = static_cast<uint8_t>(row[c >> 3] | (v << (c & 7)));
}

// The 32 nibbles of a Q4_0 / Q4_1 / IQ4 run of 16 bytes: byte j holds columns j (low) and j + 16 (high).
inline void put_split_nibbles(uint8_t* row, int64_t c0, const uint8_t* qs) {
    for (int j = 0; j < 16; ++j) {
        put_nibble(row, c0 + j, qs[j] & 0x0Fu);
        put_nibble(row, c0 + 16 + j, qs[j] >> 4);
    }
}

// Per-row stream sizes in bytes of a format, by stream.
struct Streams {
    int count = 0;
    size_t row[kBlockStreams] = {};
};

Streams streams_of(BlockFormat f, int64_t k) {
    const auto K = static_cast<size_t>(k);
    const size_t B = K / 256;
    switch (f) {
        case BlockFormat::Q4_K:
            return {2, {K / 2, 16 * B}};
        case BlockFormat::Q5_K:
            return {3, {K / 2, K / 8, 16 * B}};
        case BlockFormat::Q6_K:
            return {4, {K / 2, K / 4, K / 16, 2 * B}};
        case BlockFormat::Q3_K:
            return {4, {K / 4, K / 8, 12 * B, 2 * B}};
        case BlockFormat::Q2_K:
            return {3, {K / 4, K / 16, 4 * B}};
        case BlockFormat::IQ4_NL:
            return {2, {K / 2, K / 16}};
        case BlockFormat::IQ4_XS:
            return {2, {K / 2, 8 * B}};
        case BlockFormat::IQ3_XXS:
            return {3, {K / 4, K / 8, 2 * B}};
        case BlockFormat::IQ3_S:
            return {5, {K / 4, K / 8, K / 32, K / 64, 2 * B}};
        case BlockFormat::IQ2_XXS:
            return {2, {K / 4, 2 * B}};
        case BlockFormat::IQ2_S:
            return {5, {K / 8, K / 8, K / 32, K / 32, 2 * B}};
        default:
            return {};
    }
}

}  // namespace

bool block_on_device(BlockFormat f) {
    return streams_of(f, 256).count > 0;
}

void block_stream_rows(BlockFormat f, int64_t cols, int64_t out[kBlockStreams]) {
    const Streams st = streams_of(f, cols);
    for (int s = 0; s < kBlockStreams; ++s)
        out[s] = s < st.count ? static_cast<int64_t>(st.row[s]) : 0;
}

BlockImage block_image(BlockFormat f, const uint8_t* blocks, int64_t rows, int64_t cols) {
    const BlockLayout layout = block_layout(f);
    XS_CHECK(block_on_device(f), "sycl: no kernel reads the block format ", block_format_name(f));
    XS_CHECK(cols % 256 == 0, "sycl: a ", block_format_name(f), " weight needs a multiple of 256 columns, got ", cols);
    const Streams st = streams_of(f, cols);
    BlockImage image;
    image.streams = st.count;
    size_t total = 0;
    for (int s = 0; s < st.count; ++s) {
        image.offset[s] = total;
        total += (st.row[s] * static_cast<size_t>(rows) + kAlign - 1) / kAlign * kAlign;
    }
    image.bytes.assign(total, 0);
    const size_t row_bytes = block_row_bytes(f, cols);
    const int64_t per_row = cols / layout.weights;
    uint8_t* const base = image.bytes.data();
    for_rows(rows, cols, [&](int64_t r) {
        uint8_t* out[kBlockStreams];
        for (int s = 0; s < st.count; ++s)
            out[s] = base + image.offset[s] + static_cast<size_t>(r) * st.row[s];
        const uint8_t* in = blocks + static_cast<size_t>(r) * row_bytes;
        for (int64_t b = 0; b < per_row; ++b, in += layout.bytes) {
            const int64_t c0 = b * layout.weights;
            switch (f) {
                case BlockFormat::Q4_K:
                case BlockFormat::Q5_K: {  // d, dmin, scales[12], (Q5_K) qh[32], qs[128]
                    const bool five = f == BlockFormat::Q5_K;
                    const uint8_t* qh = in + 16;
                    const uint8_t* qs = in + (five ? 48 : 16);
                    std::memcpy(out[five ? 2 : 1] + 16 * b, in, 16);
                    for (int p = 0; p < 4; ++p) {  // 64 columns: the low nibbles of 32 bytes, then the high ones
                        for (int h = 0; h < 2; ++h) {
                            for (int l = 0; l < 32; ++l) {
                                const int64_t c = c0 + 64 * p + 32 * h + l;
                                put_nibble(out[0], c, (qs[32 * p + l] >> (4 * h)) & 0x0Fu);
                                if (five) put_bit(out[1], c, (qh[l] >> (2 * p + h)) & 1u);
                            }
                        }
                    }
                    break;
                }
                case BlockFormat::Q6_K: {  // ql[128], qh[64], scales[16], d
                    const uint8_t* ql = in;
                    const uint8_t* qh = in + 128;
                    std::memcpy(out[2] + 16 * b, in + 192, 16);
                    std::memcpy(out[3] + 2 * b, in + 208, 2);
                    for (int n = 0; n < 2; ++n) {  // 128 columns: four quarters of 32 share 64 + 32 bytes
                        for (int t = 0; t < 4; ++t) {
                            for (int l = 0; l < 32; ++l) {
                                const int64_t c = c0 + 128 * n + 32 * t + l;
                                put_nibble(out[0], c, (ql[64 * n + l + 32 * (t & 1)] >> (4 * (t >> 1))) & 0x0Fu);
                                put_pair(out[1], c, (qh[32 * n + l] >> (2 * t)) & 3u);
                            }
                        }
                    }
                    break;
                }
                case BlockFormat::Q3_K:
                case BlockFormat::Q2_K: {  // Q3_K: hmask[32], qs[64], scales[12], d; Q2_K: scales[16], qs[64], d, dmin
                    const bool three = f == BlockFormat::Q3_K;
                    const uint8_t* qs = in + (three ? 32 : 16);
                    if (three) {
                        std::memcpy(out[2] + 12 * b, in + 96, 12);
                        std::memcpy(out[3] + 2 * b, in + 108, 2);
                    } else {
                        std::memcpy(out[1] + 16 * b, in, 16);
                        std::memcpy(out[2] + 4 * b, in + 80, 4);
                    }
                    for (int n = 0; n < 2; ++n) {  // 128 columns: four quarters of 32 share 32 bytes
                        for (int j = 0; j < 4; ++j) {
                            for (int l = 0; l < 32; ++l) {
                                const int64_t c = c0 + 128 * n + 32 * j + l;
                                put_pair(out[0], c, (qs[32 * n + l] >> (2 * j)) & 3u);
                                if (three) put_bit(out[1], c, (in[l] >> (4 * n + j)) & 1u);
                            }
                        }
                    }
                    break;
                }
                case BlockFormat::IQ4_NL:  // d, qs[16]
                    std::memcpy(out[1] + 2 * b, in, 2);
                    put_split_nibbles(out[0], c0, in + 2);
                    break;
                case BlockFormat::IQ4_XS:  // d, scales_h, scales_l[4], qs[128]
                    std::memcpy(out[1] + 8 * b, in, 8);
                    for (int ib = 0; ib < 8; ++ib)
                        put_split_nibbles(out[0], c0 + 32 * ib, in + 8 + 16 * ib);
                    break;
                case BlockFormat::IQ3_XXS:  // d, qs[64], signs and scales u32[8]
                    std::memcpy(out[2] + 2 * b, in, 2);
                    std::memcpy(out[0] + 64 * b, in + 2, 64);
                    std::memcpy(out[1] + 32 * b, in + 66, 32);
                    break;
                case BlockFormat::IQ3_S:  // d, qs[64], qh[8], signs[32], scales[4]
                    std::memcpy(out[4] + 2 * b, in, 2);
                    std::memcpy(out[0] + 64 * b, in + 2, 64);
                    std::memcpy(out[2] + 8 * b, in + 66, 8);
                    std::memcpy(out[1] + 32 * b, in + 74, 32);
                    std::memcpy(out[3] + 4 * b, in + 106, 4);
                    break;
                case BlockFormat::IQ2_XXS:  // d, qs u16[32]
                    std::memcpy(out[1] + 2 * b, in, 2);
                    std::memcpy(out[0] + 64 * b, in + 2, 64);
                    break;
                case BlockFormat::IQ2_S:  // d, qs[32], signs[32], qh[8], scales[8]
                    std::memcpy(out[4] + 2 * b, in, 2);
                    std::memcpy(out[0] + 32 * b, in + 2, 32);
                    std::memcpy(out[1] + 32 * b, in + 34, 32);
                    std::memcpy(out[2] + 8 * b, in + 66, 8);
                    std::memcpy(out[3] + 8 * b, in + 74, 8);
                    break;
                default:
                    break;
            }
        }
    });
    return image;
}

bool block_as_pack(BlockFormat f) {
    return f == BlockFormat::Q8_0 || f == BlockFormat::Q4_1 || f == BlockFormat::Q4_0;
}

PackImage block_pack_image(BlockFormat f, const uint8_t* blocks, int64_t rows, int64_t cols) {
    XS_CHECK(block_as_pack(f), "sycl: ", block_format_name(f), " is not one of a pack's formats");
    const BlockLayout layout = block_layout(f);
    XS_CHECK(cols % 32 == 0, "sycl: a ", block_format_name(f), " weight needs a multiple of 32 columns, got ", cols);
    const int64_t groups = cols / 32;
    PackImage image;
    image.q4 = f != BlockFormat::Q8_0;
    const auto row_codes = static_cast<size_t>(image.q4 ? cols / 2 : cols);
    image.codes.assign(static_cast<size_t>(rows) * row_codes, 0);
    image.scales.resize(static_cast<size_t>(rows * groups));
    if (image.q4) image.biases.resize(image.scales.size());
    for_rows(rows, cols, [&](int64_t r) {
        const uint8_t* in = blocks + static_cast<size_t>(r * groups) * static_cast<size_t>(layout.bytes);
        uint8_t* codes = image.codes.data() + static_cast<size_t>(r) * row_codes;
        for (int64_t g = 0; g < groups; ++g, in += layout.bytes) {
            const auto at = static_cast<size_t>(r * groups + g);
            std::memcpy(&image.scales[at], in, 2);
            if (f == BlockFormat::Q8_0) {  // d, qs i8[32]
                std::memcpy(codes + 32 * g, in + 2, 32);
            } else if (f == BlockFormat::Q4_1) {  // d, m, qs[16]: w = d q + m
                std::memcpy(&image.biases[at], in + 2, 2);
                put_split_nibbles(codes, 32 * g, in + 4);
            } else {  // d, qs[16]: w = d (q - 8), and -8 d is exact in f16 short of overflow
                image.biases[at] = f32_to_f16(-8.0f * f16_to_f32(image.scales[at]));
                put_split_nibbles(codes, 32 * g, in + 2);
            }
        }
    });
    return image;
}

}  // namespace xs::sycl_be
