#include "quant/block_decode.h"

#include "quant/quant.h"
#include "util/error.h"
#include "util/half.h"

namespace xs::quant {

namespace {

// Block fields are little-endian, as GGUF stores them.
uint16_t u16_at(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
uint32_t u32_at(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 | static_cast<uint32_t>(p[2]) << 16 |
           static_cast<uint32_t>(p[3]) << 24;
}
float f16_at(const uint8_t* p) {
    return f16_to_f32(u16_at(p));
}

// One decoder per format writes the block_layout(f).weights values of one block. The float operations and their
// order are those of ggml's dequantize_row_* (ggml-quants.c). A product is rounded before anything is added to it
// (ggml's values are not fused multiply-adds), which is why products are statements of their own: a compiler may
// fuse a multiply and an add within one expression.

// Q4_0: f16 d, 16 bytes of 4-bit codes (column j in the low nibble of byte j, column j + 16 in its high nibble).
// w = (q - 8) * d.
void q4_0(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) {
        y[j] = static_cast<float>((qs[j] & 0x0F) - 8) * d;
        y[j + 16] = static_cast<float>((qs[j] >> 4) - 8) * d;
    }
}

// Q4_1: f16 d, f16 m, codes as Q4_0. w = q * d + m.
void q4_1(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const float m = f16_at(b + 2);
    const uint8_t* qs = b + 4;
    for (int j = 0; j < 16; ++j) {
        const float lo = static_cast<float>(qs[j] & 0x0F) * d;
        const float hi = static_cast<float>(qs[j] >> 4) * d;
        y[j] = lo + m;
        y[j + 16] = hi + m;
    }
}

// Q5_0: f16 d, 32 high bits (bit j: column j), low nibbles as Q4_0. w = (q - 16) * d.
void q5_0(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const uint32_t qh = u32_at(b + 2);
    const uint8_t* qs = b + 6;
    for (int j = 0; j < 16; ++j) {
        const int lo = (qs[j] & 0x0F) | static_cast<int>(((qh >> j) << 4) & 0x10);
        const int hi = (qs[j] >> 4) | static_cast<int>((qh >> (j + 12)) & 0x10);
        y[j] = static_cast<float>(lo - 16) * d;
        y[j + 16] = static_cast<float>(hi - 16) * d;
    }
}

// Q5_1: f16 d, f16 m, high bits and low nibbles as Q5_0. w = q * d + m.
void q5_1(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const float m = f16_at(b + 2);
    const uint32_t qh = u32_at(b + 4);
    const uint8_t* qs = b + 8;
    for (int j = 0; j < 16; ++j) {
        const int lo = (qs[j] & 0x0F) | static_cast<int>(((qh >> j) << 4) & 0x10);
        const int hi = (qs[j] >> 4) | static_cast<int>((qh >> (j + 12)) & 0x10);
        const float lo_d = static_cast<float>(lo) * d;
        const float hi_d = static_cast<float>(hi) * d;
        y[j] = lo_d + m;
        y[j + 16] = hi_d + m;
    }
}

// Q8_0: f16 d, 32 int8 codes. w = q * d.
void q8_0(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    for (int j = 0; j < 32; ++j)
        y[j] = static_cast<float>(static_cast<int8_t>(b[2 + j])) * d;
}

using BlockDecoder = void (*)(const uint8_t* block, float* out);

BlockDecoder decoder(BlockFormat f) {
    switch (f) {
        case BlockFormat::Q4_0:
            return q4_0;
        case BlockFormat::Q4_1:
            return q4_1;
        case BlockFormat::Q5_0:
            return q5_0;
        case BlockFormat::Q5_1:
            return q5_1;
        case BlockFormat::Q8_0:
            return q8_0;
        default:
            fail("block format ", block_format_name(f), ": decoding is not implemented yet");
    }
}

}  // namespace

void decode_blocks(BlockFormat format, const uint8_t* blocks, int64_t count, float* out) {
    const BlockDecoder fn = decoder(format);
    const BlockLayout l = block_layout(format);
    for (int64_t i = 0; i < count; ++i)
        fn(blocks + i * l.bytes, out + i * l.weights);
}

void decode_block_rows(const TensorView& v, int64_t first, int64_t count, float* out) {
    XS_CHECK(v.dtype == DType::Block && !v.shape.empty() && v.data != nullptr,
             "decode_block_rows needs a block format weight, got ", dtype_name(v.dtype), " ", dims_to_string(v.shape));
    const BlockLayout l = block_layout(v.block);
    const int64_t cols = v.shape.back();
    XS_CHECK(cols > 0 && cols % l.weights == 0, block_format_name(v.block), " weight ", dims_to_string(v.shape), ": ",
             cols, " columns are not whole blocks of ", l.weights);
    XS_CHECK(first >= 0 && count >= 0 && first + count <= matrix_rows(v.shape), block_format_name(v.block), ": rows ",
             first, " to ", first + count, " of a ", dims_to_string(v.shape), " weight");
    const size_t row_bytes = block_row_bytes(v.block, cols);
    decode_blocks(v.block, static_cast<const uint8_t*>(v.data) + static_cast<size_t>(first) * row_bytes,
                  count * (cols / l.weights), out);
}

}  // namespace xs::quant
