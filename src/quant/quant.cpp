#include "quant/quant.h"

#include <algorithm>
#include <cmath>
#include <thread>

#include "util/error.h"
#include "util/half.h"

namespace xs::quant {

namespace {

void row_to_f32(const TensorView& w, int64_t row, float* out) {
    const int64_t cols = w.shape.back();
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
    v.shape = shape.empty() ? Dims{rows, cols} : shape;
    v.data = codes.data();
    v.quant.group = group;
    v.quant.scales = scales.data();
    v.quant.biases = format != DType::Q8 ? biases.data() : nullptr;
    v.quant.input_scales = input_scales.empty() ? nullptr : input_scales.data();
    return v;
}

std::string unsupported_reason(DType format, int64_t rows, int64_t cols, int64_t group) {
    if (!is_quantized(format)) return str_cat("format ", dtype_name(format), " is not q8, q4, q3 or q2");
    if (rows <= 0 || cols <= 0) return "the matrix is empty";
    if (group != 32 && group != 64 && group != 128) return str_cat("group size ", group, " is not 32, 64 or 128");
    if (cols % group != 0) return str_cat(cols, " columns are not a multiple of the group size ", group);
    if ((format == DType::Q3 || format == DType::Q2) && cols % 32 != 0) {
        return str_cat(dtype_name(format), " packs 32 columns at a time; ", cols, " is not a multiple");
    }
    return {};
}

Quantized quantize(const TensorView& w, DType format, int64_t group) {
    XS_CHECK(w.shape.size() >= 2, "quantize needs a tensor of at least 2 dims, got ", dims_to_string(w.shape));
    const int64_t rows = matrix_rows(w.shape);
    const int64_t cols = w.shape.back();
    if (const std::string why = unsupported_reason(format, rows, cols, group); !why.empty()) fail("quantize: ", why);
    Quantized q;
    q.format = format;
    q.shape = w.shape;
    q.rows = rows;
    q.cols = cols;
    q.group = group;
    const int64_t groups = cols / group;
    q.codes.resize(storage_bytes(format, rows * cols));
    q.scales.resize(static_cast<size_t>(rows * groups));
    if (format != DType::Q8) q.biases.resize(static_cast<size_t>(rows * groups));
    const int bits = quant_bits(format);
    const float top = static_cast<float>((1 << bits) - 1);
    const int64_t row_bytes = static_cast<int64_t>(storage_bytes(format, cols));

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
                    const uint16_t s16 = f32_to_f16((hi - lo) / top);
                    const float b = f16_to_f32(b16);
                    const float s = f16_to_f32(s16);
                    XS_CHECK(std::isfinite(s) && std::isfinite(b), "quantize: row ", r,
                             " has values beyond the f16 range");
                    q.scales[sg] = s16;
                    q.biases[sg] = b16;
                    uint8_t* row_codes = q.codes.data() + r * row_bytes;
                    for (int64_t i = 0; i < group; ++i) {
                        const float c = s > 0.0f ? std::round((x[i] - b) / s) : 0.0f;
                        set_code(row_codes, bits, g * group + i, static_cast<int>(std::clamp(c, 0.0f, top)));
                    }
                }
            }
        }
    });
    return q;
}

namespace {

constexpr float kClips[] = {1.0f, 0.95f, 0.9f, 0.85f, 0.8f, 0.75f};

// One q4 group of `scaled` values (w * s): the clip (range shrunk around its middle) with the least weighted error
// sum imp[i] * (dequant / s - w)^2. Writes codes / scale / bias when `out_codes` is set; returns the error.
double q4_group(const float* scaled, const float* w, const float* s, const double* imp, int64_t n, bool clip,
                uint8_t* out_codes, uint16_t* out_scale, uint16_t* out_bias) {
    float lo = scaled[0];
    float hi = scaled[0];
    for (int64_t i = 1; i < n; ++i) {
        lo = std::min(lo, scaled[i]);
        hi = std::max(hi, scaled[i]);
    }
    const float mid = 0.5f * (lo + hi);
    double best = -1;
    uint16_t best_s = 0;
    uint16_t best_b = 0;
    for (const float f : kClips) {
        if (!clip && f != 1.0f) break;
        const uint16_t b16 = f32_to_f16(mid + (lo - mid) * f);
        const uint16_t s16 = f32_to_f16((hi - lo) * f / 15.0f);
        const float b = f16_to_f32(b16);
        const float sc = f16_to_f32(s16);
        double err = 0;
        for (int64_t i = 0; i < n; ++i) {
            const float c = sc > 0.0f ? std::clamp(std::round((scaled[i] - b) / sc), 0.0f, 15.0f) : 0.0f;
            const double d = (sc * c + b) / s[i] - w[i];
            err += imp[i] * d * d;
        }
        if (best < 0 || err < best) {
            best = err;
            best_s = s16;
            best_b = b16;
        }
    }
    if (out_codes != nullptr) {
        const float b = f16_to_f32(best_b);
        const float sc = f16_to_f32(best_s);
        for (int64_t i = 0; i < n; i += 2) {
            const auto code = [&](float v) {
                return static_cast<uint8_t>(sc > 0.0f ? std::clamp(std::round((v - b) / sc), 0.0f, 15.0f) : 0.0f);
            };
            out_codes[i / 2] = static_cast<uint8_t>(code(scaled[i]) | (code(scaled[i + 1]) << 4));
        }
        *out_scale = best_s;
        *out_bias = best_b;
    }
    return best;
}

// Weighted q4 error of `w` with column scales `s` (and codes into `q` when set).
double q4_scaled(const TensorView& w, int64_t group, const std::vector<float>& s, const std::vector<double>& imp,
                 bool clip, Quantized* q) {
    const int64_t rows = w.shape[0];
    const int64_t cols = w.shape[1];
    std::vector<double> row_err(static_cast<size_t>(rows), 0.0);
    parallel_rows(rows, [&](int64_t first, int64_t end) {
        std::vector<float> row(static_cast<size_t>(cols));
        std::vector<float> scaled(static_cast<size_t>(cols));
        for (int64_t r = first; r < end; ++r) {
            row_to_f32(w, r, row.data());
            for (int64_t c = 0; c < cols; ++c)
                scaled[static_cast<size_t>(c)] = row[static_cast<size_t>(c)] * s[static_cast<size_t>(c)];
            double err = 0;
            for (int64_t g = 0; g < cols / group; ++g) {
                const int64_t c0 = g * group;
                const size_t sg = static_cast<size_t>(r * (cols / group) + g);
                err += q4_group(scaled.data() + c0, row.data() + c0, s.data() + c0, imp.data() + c0, group, clip,
                                q != nullptr ? q->codes.data() + (r * cols + c0) / 2 : nullptr,
                                q != nullptr ? &q->scales[sg] : nullptr, q != nullptr ? &q->biases[sg] : nullptr);
            }
            row_err[static_cast<size_t>(r)] = err;
        }
    });
    double total = 0;
    for (const double e : row_err)
        total += e;
    return total;
}

}  // namespace

Quantized quantize_awq(const TensorView& w, int64_t group, const std::vector<double>& act_ms, AwqReport* report) {
    XS_CHECK(w.shape.size() == 2, "quantize_awq needs a 2-D matrix, got ", dims_to_string(w.shape));
    const int64_t rows = w.shape[0];
    const int64_t cols = w.shape[1];
    if (const std::string why = unsupported_reason(DType::Q4, rows, cols, group); !why.empty()) fail("quantize: ", why);
    XS_CHECK(static_cast<int64_t>(act_ms.size()) == cols, "quantize_awq: ", act_ms.size(), " activation statistics for ",
             cols, " columns");
    double peak = 0;
    for (const double m : act_ms)
        peak = std::max(peak, m);
    XS_CHECK(peak > 0, "quantize_awq: the calibration activations are all zero");
    std::vector<double> imp(act_ms.size());
    for (size_t c = 0; c < imp.size(); ++c)
        imp[c] = std::max(act_ms[c], peak * 1e-8);  // a channel calibration never lit still counts a little

    const auto scales_for = [&](double alpha) {
        std::vector<float> s(imp.size());
        double log_sum = 0;
        for (size_t c = 0; c < s.size(); ++c) {
            s[c] = static_cast<float>(std::pow(imp[c], alpha / 2));  // rms^alpha
            log_sum += std::log(static_cast<double>(s[c]));
        }
        const double norm = std::exp(log_sum / static_cast<double>(s.size()));
        for (float& v : s)
            v = std::clamp(static_cast<float>(v / norm), 1e-3f, 1e3f);
        return s;
    };
    const std::vector<float> ones(imp.size(), 1.0f);
    AwqReport r;
    r.error_rtn = q4_scaled(w, group, ones, imp, /*clip=*/false, nullptr);
    r.error = -1;
    for (int i = 0; i <= 10; ++i) {
        const double alpha = 0.1 * i;
        const double err = q4_scaled(w, group, scales_for(alpha), imp, /*clip=*/true, nullptr);
        if (r.error < 0 || err < r.error) {
            r.error = err;
            r.alpha = alpha;
        }
    }
    Quantized q;
    q.format = DType::Q4;
    q.shape = w.shape;
    q.rows = rows;
    q.cols = cols;
    q.group = group;
    q.codes.resize(storage_bytes(DType::Q4, rows * cols));
    q.scales.resize(static_cast<size_t>(rows * (cols / group)));
    q.biases.resize(q.scales.size());
    const std::vector<float> s = scales_for(r.alpha);
    q4_scaled(w, group, s, imp, /*clip=*/true, &q);
    q.input_scales.resize(s.size());
    for (size_t c = 0; c < s.size(); ++c)
        q.input_scales[c] = 1.0f / s[c];
    if (report != nullptr) *report = r;
    return q;
}

float quantize_kv_row_i8(const float* x, int64_t width, int8_t* codes) {
    float amax = 0.0f;
    for (int64_t i = 0; i < width; ++i)
        amax = std::max(amax, std::fabs(x[i]));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (int64_t i = 0; i < width; ++i)
        codes[i] = static_cast<int8_t>(std::clamp(std::nearbyint(x[i] * inv), -127.0f, 127.0f));
    return amax / 127.0f;
}

void round_trip_kv_i8(float* x, int64_t rows, int64_t width) {
    std::vector<int8_t> codes(static_cast<size_t>(width));
    for (int64_t r = 0; r < rows; ++r) {
        float* row = x + r * width;
        const float scale = quantize_kv_row_i8(row, width, codes.data());
        for (int64_t i = 0; i < width; ++i)
            row[i] = static_cast<float>(codes[static_cast<size_t>(i)]) * scale;
    }
}

void dequantize_rows(const TensorView& v, int64_t first, int64_t count, float* out) {
    XS_CHECK(is_quantized(v.dtype) && v.shape.size() >= 2, "dequantize needs a quantized tensor, got ",
             dtype_name(v.dtype), " ", dims_to_string(v.shape));
    const int64_t cols = v.shape.back();
    const int64_t group = v.quant.group;
    const int64_t groups = cols / group;
    const int bits = quant_bits(v.dtype);
    const auto row_bytes = static_cast<int64_t>(storage_bytes(v.dtype, cols));
    const auto* scales = static_cast<const uint16_t*>(v.quant.scales);
    const auto* biases = static_cast<const uint16_t*>(v.quant.biases);
    for (int64_t r = first; r < first + count; ++r) {
        float* o = out + (r - first) * cols;
        const auto* row = static_cast<const uint8_t*>(v.data) + r * row_bytes;
        for (int64_t c = 0; c < cols; ++c) {
            const size_t sg = static_cast<size_t>(r * groups + c / group);
            const float s = f16_to_f32(scales[sg]);
            if (v.dtype == DType::Q8) {
                o[c] = s * static_cast<float>(static_cast<int8_t>(row[c]));
            } else {
                o[c] = s * static_cast<float>(code_at(row, bits, c)) + f16_to_f32(biases[sg]);
            }
            if (v.quant.input_scales != nullptr) o[c] *= v.quant.input_scales[c];
        }
    }
}

int code_at(const uint8_t* row, int bits, int64_t c) {
    if (bits == 4) return (c % 2 == 0) ? (row[c / 2] & 0x0F) : (row[c / 2] >> 4);
    const uint8_t* chunk = row + (c / 32) * (bits == 3 ? 12 : 8);
    const int j = static_cast<int>(c % 32);
    int code = (chunk[j / 4] >> (2 * (j % 4))) & 3;
    if (bits == 3) code |= ((chunk[8 + j / 8] >> (j % 8)) & 1) << 2;
    return code;
}

void set_code(uint8_t* row, int bits, int64_t c, int code) {
    if (bits == 4) {
        uint8_t& byte = row[c / 2];
        byte = (c % 2 == 0) ? static_cast<uint8_t>((byte & 0xF0) | code) : static_cast<uint8_t>((byte & 0x0F) | (code << 4));
        return;
    }
    uint8_t* chunk = row + (c / 32) * (bits == 3 ? 12 : 8);
    const int j = static_cast<int>(c % 32);
    uint8_t& lo = chunk[j / 4];
    lo = static_cast<uint8_t>((lo & ~(3 << (2 * (j % 4)))) | ((code & 3) << (2 * (j % 4))));
    if (bits == 3) {
        uint8_t& hi = chunk[8 + j / 8];
        hi = static_cast<uint8_t>((hi & ~(1 << (j % 8))) | (((code >> 2) & 1) << (j % 8)));
    }
}

}  // namespace xs::quant
