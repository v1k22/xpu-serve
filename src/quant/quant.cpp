#include "quant/quant.h"

#include <algorithm>
#include <cmath>
#include <thread>

#include "util/error.h"
#include "util/half.h"

namespace xs::quant {

namespace {

void row_to_f32(const TensorView& w, int64_t row, float* out) {
    const int64_t cols = w.shape[1];
    const size_t offset = static_cast<size_t>(row * cols);
    switch (w.dtype) {
        case DType::F32: {
            const float* src = static_cast<const float*>(w.data) + offset;
            std::copy(src, src + cols, out);
            return;
        }
        case DType::F16:
            convert_f16_to_f32(static_cast<const uint16_t*>(w.data) + offset, out, static_cast<size_t>(cols));
            return;
        case DType::BF16:
            convert_bf16_to_f32(static_cast<const uint16_t*>(w.data) + offset, out, static_cast<size_t>(cols));
            return;
        default:
            fail("quantize: weights must be f32, f16 or bf16, got ", dtype_name(w.dtype));
    }
}

// Runs fn(first_row, end_row) over `rows` split across the hardware threads.
template <typename Fn>
void parallel_rows(int64_t rows, Fn fn) {
    const int64_t threads = std::clamp<int64_t>(std::thread::hardware_concurrency(), 1, 64);
    const int64_t per = (rows + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (int64_t first = 0; first < rows; first += per) {
        pool.emplace_back(fn, first, std::min(rows, first + per));
    }
    for (std::thread& t : pool)
        t.join();
}

}  // namespace

TensorView Quantized::view() const {
    TensorView v;
    v.dtype = format;
    v.shape = {rows, cols};
    v.data = codes.data();
    v.quant.group = group;
    v.quant.scales = scales.data();
    v.quant.biases = format == DType::Q4 ? biases.data() : nullptr;
    return v;
}

std::string unsupported_reason(DType format, int64_t rows, int64_t cols, int64_t group) {
    if (format != DType::Q8 && format != DType::Q4) return str_cat("format ", dtype_name(format), " is not q8 or q4");
    if (rows <= 0 || cols <= 0) return "the matrix is empty";
    if (group != 32 && group != 64 && group != 128) return str_cat("group size ", group, " is not 32, 64 or 128");
    if (cols % group != 0) return str_cat(cols, " columns are not a multiple of the group size ", group);
    return {};
}

Quantized quantize(const TensorView& w, DType format, int64_t group) {
    XS_CHECK(w.shape.size() == 2, "quantize needs a 2-D matrix, got ", dims_to_string(w.shape));
    const int64_t rows = w.shape[0];
    const int64_t cols = w.shape[1];
    if (const std::string why = unsupported_reason(format, rows, cols, group); !why.empty()) fail("quantize: ", why);
    Quantized q;
    q.format = format;
    q.rows = rows;
    q.cols = cols;
    q.group = group;
    const int64_t groups = cols / group;
    q.codes.resize(storage_bytes(format, rows * cols));
    q.scales.resize(static_cast<size_t>(rows * groups));
    if (format == DType::Q4) q.biases.resize(static_cast<size_t>(rows * groups));

    parallel_rows(rows, [&](int64_t first, int64_t end) {
        std::vector<float> row(static_cast<size_t>(cols));
        for (int64_t r = first; r < end; ++r) {
            row_to_f32(w, r, row.data());
            for (int64_t g = 0; g < groups; ++g) {
                const float* x = row.data() + g * group;
                const size_t sg = static_cast<size_t>(r * groups + g);
                if (format == DType::Q8) {
                    float amax = 0.0f;
                    for (int64_t i = 0; i < group; ++i)
                        amax = std::max(amax, std::abs(x[i]));
                    const uint16_t s16 = f32_to_f16(amax / 127.0f);
                    const float s = f16_to_f32(s16);
                    XS_CHECK(std::isfinite(s), "quantize: row ", r, " has values beyond the f16 range");
                    q.scales[sg] = s16;
                    auto* codes = reinterpret_cast<int8_t*>(q.codes.data()) + r * cols + g * group;
                    for (int64_t i = 0; i < group; ++i) {
                        const float v = s > 0.0f ? std::round(x[i] / s) : 0.0f;
                        codes[i] = static_cast<int8_t>(std::clamp(v, -127.0f, 127.0f));
                    }
                } else {
                    float lo = x[0];
                    float hi = x[0];
                    for (int64_t i = 1; i < group; ++i) {
                        lo = std::min(lo, x[i]);
                        hi = std::max(hi, x[i]);
                    }
                    const uint16_t b16 = f32_to_f16(lo);
                    const uint16_t s16 = f32_to_f16((hi - lo) / 15.0f);
                    const float b = f16_to_f32(b16);
                    const float s = f16_to_f32(s16);
                    XS_CHECK(std::isfinite(s) && std::isfinite(b), "quantize: row ", r,
                             " has values beyond the f16 range");
                    q.scales[sg] = s16;
                    q.biases[sg] = b16;
                    uint8_t* bytes = q.codes.data() + (r * cols + g * group) / 2;
                    for (int64_t i = 0; i < group; i += 2) {
                        const auto code = [&](float v) {
                            const float c = s > 0.0f ? std::round((v - b) / s) : 0.0f;
                            return static_cast<uint8_t>(std::clamp(c, 0.0f, 15.0f));
                        };
                        bytes[i / 2] = static_cast<uint8_t>(code(x[i]) | (code(x[i + 1]) << 4));
                    }
                }
            }
        }
    });
    return q;
}

void dequantize_rows(const TensorView& v, int64_t first, int64_t count, float* out) {
    XS_CHECK(is_quantized(v.dtype) && v.shape.size() == 2, "dequantize needs a q8 / q4 matrix, got ",
             dtype_name(v.dtype), " ", dims_to_string(v.shape));
    const int64_t cols = v.shape[1];
    const int64_t group = v.quant.group;
    const int64_t groups = cols / group;
    const auto* scales = static_cast<const uint16_t*>(v.quant.scales);
    const auto* biases = static_cast<const uint16_t*>(v.quant.biases);
    for (int64_t r = first; r < first + count; ++r) {
        float* o = out + (r - first) * cols;
        for (int64_t c = 0; c < cols; ++c) {
            const size_t sg = static_cast<size_t>(r * groups + c / group);
            const float s = f16_to_f32(scales[sg]);
            if (v.dtype == DType::Q8) {
                o[c] = s * static_cast<float>(static_cast<const int8_t*>(v.data)[r * cols + c]);
            } else {
                const uint8_t byte = static_cast<const uint8_t*>(v.data)[(r * cols + c) / 2];
                const int code = (c % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                o[c] = s * static_cast<float>(code) + f16_to_f32(biases[sg]);
            }
        }
    }
}

}  // namespace xs::quant
