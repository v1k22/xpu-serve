#include "io/safetensors.h"

#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <algorithm>

#include "util/error.h"

namespace xs {

const TensorView& WeightStore::get(const std::string& name) const {
    const TensorView* v = find(name);
    if (v == nullptr) fail("weight '", name, "' not found");
    return *v;
}

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : file_(path) {
    const size_t size = file_.size();
    if (size < 8) fail("'", path.string(), "' is not a safetensors file (", size, " bytes)");
    uint64_t header_len = 0;
    std::memcpy(&header_len, file_.data(), sizeof(header_len));  // little-endian on all supported hosts
    if (header_len == 0 || header_len > size - 8) {
        fail("'", path.string(), "': header length ", header_len, " does not fit in a ", size, "-byte file");
    }
    const char* header_begin = reinterpret_cast<const char*>(file_.data()) + 8;
    nlohmann::json header;
    try {
        header = nlohmann::json::parse(header_begin, header_begin + header_len);
    } catch (const nlohmann::json::exception& e) {
        fail("'", path.string(), "': invalid safetensors header: ", e.what());
    }
    const std::byte* data_begin = file_.data() + 8 + header_len;
    const size_t data_size = size - 8 - header_len;

    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") {
            for (auto m = it.value().begin(); m != it.value().end(); ++m) {
                if (m.value().is_string()) metadata_[m.key()] = m.value().get<std::string>();
            }
            continue;
        }
        const auto& entry = it.value();
        TensorView v;
        v.dtype = dtype_from_safetensors(entry.at("dtype").get<std::string>());
        v.shape = entry.at("shape").get<Dims>();
        const auto offsets = entry.at("data_offsets").get<std::vector<uint64_t>>();
        if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > data_size) {
            fail("'", path.string(), "': tensor '", it.key(), "' has invalid data_offsets");
        }
        if (offsets[1] - offsets[0] != v.nbytes()) {
            fail("'", path.string(), "': tensor '", it.key(), "' is ", offsets[1] - offsets[0], " bytes but ",
                 dims_to_string(v.shape), " ", dtype_name(v.dtype), " needs ", v.nbytes());
        }
        v.data = data_begin + offsets[0];
        tensors_.emplace(it.key(), std::move(v));
    }
}

std::unique_ptr<SafetensorsWeights> SafetensorsWeights::open_dir(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    std::vector<fs::path> paths;
    const fs::path index_path = dir / "model.safetensors.index.json";
    if (fs::exists(index_path)) {
        std::ifstream in(index_path);
        const auto index = nlohmann::json::parse(in);
        std::set<std::string> shards;
        for (const auto& [name, shard] : index.at("weight_map").items()) shards.insert(shard.get<std::string>());
        for (const auto& s : shards) paths.push_back(dir / s);
    } else {
        for (const auto& e : fs::directory_iterator(dir)) {
            if (e.is_regular_file() && e.path().extension() == ".safetensors") paths.push_back(e.path());
        }
        std::sort(paths.begin(), paths.end());
    }
    if (paths.empty()) fail("no safetensors files in '", dir.string(), "'");

    auto weights = std::unique_ptr<SafetensorsWeights>(new SafetensorsWeights());
    for (const auto& p : paths) {
        auto file = std::make_unique<SafetensorsFile>(p);
        for (const auto& [name, view] : file->tensors()) {
            if (!weights->index_.emplace(name, &view).second) {
                fail("tensor '", name, "' appears in more than one safetensors file under '", dir.string(), "'");
            }
        }
        weights->files_.push_back(std::move(file));
    }
    return weights;
}

const TensorView* SafetensorsWeights::find(const std::string& name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : it->second;
}

std::vector<std::string> SafetensorsWeights::names() const {
    std::vector<std::string> out;
    out.reserve(index_.size());
    for (const auto& [name, view] : index_) out.push_back(name);
    return out;
}

}  // namespace xs
