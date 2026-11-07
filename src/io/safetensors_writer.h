#pragma once

// Writes safetensors files. Data is streamed: each unit writes its tensors' bytes when its turn comes, so a converter
// never holds more than one tensor's worth of output in memory.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "tensor/tensor.h"

namespace xs {

struct OutTensor {
    std::string name;
    DType dtype = DType::F32;  // a safetensors dtype (not q8 / q4: their codes are stored as i8 / u8)
    Dims shape;
};

// Tensors written together: `write` must emit exactly their bytes, in order.
struct OutUnit {
    std::vector<OutTensor> tensors;
    std::function<void(std::ostream&)> write;
};

size_t unit_bytes(const OutUnit& unit);

void write_safetensors(const std::filesystem::path& path, const std::vector<OutUnit>& units,
                       const std::map<std::string, std::string>& metadata = {});

}  // namespace xs
