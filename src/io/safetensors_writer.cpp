#include "io/safetensors_writer.h"

#include <fstream>
#include <nlohmann/json.hpp>

#include "util/error.h"

namespace xs {

namespace {

std::string safetensors_dtype(DType t) {
    switch (t) {
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::I64: return "I64";
        case DType::I32: return "I32";
        case DType::I8: return "I8";
        case DType::U8: return "U8";
        case DType::Bool: return "BOOL";
        case DType::F8: return "F8_E4M3";
        default: fail("safetensors has no dtype ", dtype_name(t));
    }
}

}  // namespace

size_t unit_bytes(const OutUnit& unit) {
    size_t total = 0;
    for (const OutTensor& t : unit.tensors)
        total += storage_bytes(t.dtype, numel(t.shape));
    return total;
}

void write_safetensors(const std::filesystem::path& path, const std::vector<OutUnit>& units,
                       const std::map<std::string, std::string>& metadata) {
    nlohmann::ordered_json header;
    if (!metadata.empty()) header["__metadata__"] = metadata;
    size_t offset = 0;
    for (const OutUnit& u : units) {
        for (const OutTensor& t : u.tensors) {
            const size_t bytes = storage_bytes(t.dtype, numel(t.shape));
            header[t.name] = {{"dtype", safetensors_dtype(t.dtype)}, {"shape", t.shape},
                              {"data_offsets", {offset, offset + bytes}}};
            offset += bytes;
        }
    }
    std::string text = header.dump();
    text.append((8 - text.size() % 8) % 8, ' ');  // the format pads the header to a multiple of 8 bytes
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) fail("cannot write '", path.string(), "'");
    const uint64_t len = text.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));  // little-endian on all supported hosts
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    const auto data_start = out.tellp();
    size_t expected = 0;
    for (const OutUnit& u : units) {
        u.write(out);
        expected += unit_bytes(u);
        XS_CHECK(static_cast<size_t>(out.tellp() - data_start) == expected, "safetensors writer: a unit of '",
                 u.tensors.front().name, "' wrote the wrong number of bytes");
    }
    out.close();
    if (!out) fail("writing '", path.string(), "' failed");
}

}  // namespace xs
