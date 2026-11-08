#include "tensor/dtype.h"

#include "util/error.h"

namespace xs {

size_t dtype_size(DType t) {
    switch (t) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::I64: return 8;
        case DType::I32: return 4;
        case DType::I8: return 1;
        case DType::U8: return 1;
        case DType::Bool: return 1;
        case DType::Q8: return 1;
        case DType::Q4:
        case DType::Q3:
        case DType::Q2: fail(dtype_name(t), " packs several values per byte; use storage_bytes");
    }
    fail("unknown dtype");
}

size_t storage_bytes(DType t, int64_t numel) {
    if (t == DType::Q4 || t == DType::Q3 || t == DType::Q2) {
        return static_cast<size_t>((numel * quant_bits(t) + 7) / 8);  // Q3 / Q2: 12 / 8 bytes per 32 values
    }
    return static_cast<size_t>(numel) * dtype_size(t);
}

std::string_view dtype_name(DType t) {
    switch (t) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::I64: return "i64";
        case DType::I32: return "i32";
        case DType::I8: return "i8";
        case DType::U8: return "u8";
        case DType::Bool: return "bool";
        case DType::Q8: return "q8";
        case DType::Q4: return "q4";
        case DType::Q3: return "q3";
        case DType::Q2: return "q2";
    }
    return "?";
}

DType dtype_from_safetensors(std::string_view s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I64") return DType::I64;
    if (s == "I32") return DType::I32;
    if (s == "I8") return DType::I8;
    if (s == "U8") return DType::U8;
    if (s == "BOOL") return DType::Bool;
    fail("unsupported safetensors dtype '", s, "'");
}

bool is_float(DType t) { return t == DType::F32 || t == DType::F16 || t == DType::BF16; }

bool is_quantized(DType t) { return t == DType::Q8 || t == DType::Q4 || t == DType::Q3 || t == DType::Q2; }

int quant_bits(DType t) {
    switch (t) {
        case DType::Q8: return 8;
        case DType::Q4: return 4;
        case DType::Q3: return 3;
        case DType::Q2: return 2;
        default: fail(dtype_name(t), " is not a quantized dtype");
    }
}

}  // namespace xs
