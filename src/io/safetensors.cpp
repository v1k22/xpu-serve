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
    for (const auto& p : paths)
        weights->add_file(p);
    weights->pair_fp8();
    if (fs::exists(dir / kPackManifest)) weights->apply_pack_manifest(dir / kPackManifest);
    return weights;
}

std::unique_ptr<SafetensorsWeights> SafetensorsWeights::open_file(const std::filesystem::path& path) {
    auto weights = std::unique_ptr<SafetensorsWeights>(new SafetensorsWeights());
    weights->add_file(path);
    weights->pair_fp8();
    return weights;
}

void SafetensorsWeights::add_file(const std::filesystem::path& path) {
    auto file = std::make_unique<SafetensorsFile>(path);
    for (const auto& [name, view] : file->tensors()) {
        if (!index_.emplace(name, &view).second) {
            fail("tensor '", name, "' appears in more than one safetensors file under '", path.parent_path().string(),
                 "'");
        }
    }
    files_.push_back(std::move(file));
}

void SafetensorsWeights::pair_fp8() {
    std::vector<std::string> names;
    for (const auto& [name, view] : index_) {
        if (view->dtype == DType::F8) names.push_back(name);
    }
    for (const std::string& name : names) {
        const TensorView& codes = *index_.at(name);
        const std::string scale_name = name + "_scale_inv";
        const auto it = index_.find(scale_name);
        if (codes.shape.size() != 2 || it == index_.end()) {
            fail("fp8 tensor '", name, "' ", dims_to_string(codes.shape), ": only a matrix beside its scales '",
                 scale_name, "' is read");
        }
        const TensorView& grid = *it->second;
        const Dims blocks{(codes.shape[0] + kFp8Block - 1) / kFp8Block, (codes.shape[1] + kFp8Block - 1) / kFp8Block};
        if (!is_float(grid.dtype) || grid.shape != blocks) {
            fail("fp8 tensor '", name, "' ", dims_to_string(codes.shape), ": its scales '", scale_name, "' are ",
                 dtype_name(grid.dtype), " ", dims_to_string(grid.shape), ", expected one per block of ", kFp8Block,
                 " rows and columns, ", dims_to_string(blocks));
        }
        const std::vector<float> scales =
            HostTensor::borrow(grid.dtype, grid.shape, const_cast<void*>(grid.data)).to_f32();
        fp8_scales_.push_back(std::make_unique<float[]>(scales.size()));
        std::copy(scales.begin(), scales.end(), fp8_scales_.back().get());
        const std::vector<Fp8Run>& runs = fp8_runs_[name] = {
            Fp8Run{static_cast<const uint8_t*>(codes.data), fp8_scales_.back().get(), codes.shape[0]}};
        TensorView v;
        v.dtype = DType::F8;
        v.shape = codes.shape;
        v.fp8 = runs;
        index_.erase(it);
        index_[name] = &(fp8_views_[name] = v);
    }
}

void SafetensorsWeights::join_rows(const std::string& name, const std::vector<std::string>& parts, Dims shape) {
    XS_CHECK(index_.count(name) == 0, "join_rows: '", name, "' exists already");
    XS_CHECK(shape.size() >= 2 && !parts.empty(), "join_rows: '", name, "' needs parts and at least 2 dims");
    std::vector<Fp8Run> runs;
    int64_t rows = 0;
    for (const std::string& part : parts) {
        const auto it = fp8_views_.find(part);
        if (it == fp8_views_.end()) {
            const TensorView* dense = find(part);
            if (dense == nullptr) fail("join_rows: '", part, "', a part of '", name, "', is missing");
            fail("join_rows: '", part, "', a part of '", name, "', is ", dtype_name(dense->dtype),
                 "; only fp8 tensors are read in pieces (a dense one would have to be copied)");
        }
        XS_CHECK(it->second.shape.back() == shape.back(), "join_rows: '", part, "' ", dims_to_string(it->second.shape),
                 " does not have the ", shape.back(), " columns of '", name, "'");
        runs.insert(runs.end(), it->second.fp8.begin(), it->second.fp8.end());
        rows += it->second.shape[0];
    }
    XS_CHECK(rows * shape.back() == numel(shape), "join_rows: the parts of '", name, "' have ", rows, " rows of ",
             shape.back(), ", not the shape ", dims_to_string(shape));
    for (const std::string& part : parts) {
        index_.erase(part);
        fp8_views_.erase(part);
        fp8_runs_.erase(part);
    }
    TensorView v;
    v.dtype = DType::F8;
    v.shape = std::move(shape);
    v.fp8 = fp8_runs_[name] = std::move(runs);
    index_[name] = &(fp8_views_[name] = v);
}

void SafetensorsWeights::apply_pack_manifest(const std::filesystem::path& path) {
    std::ifstream in(path);
    nlohmann::json manifest;
    try {
        manifest = nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        fail("'", path.string(), "' is not valid JSON: ", e.what());
    }
    const int version = manifest.value("pack_version", 0);
    if (version != kPackVersion) {
        fail("'", path.string(), "' is pack version ", version, "; this build reads version ", kPackVersion);
    }
    const auto component = [&](const std::string& name, DType dtype, const Dims& shape) -> const TensorView& {
        const auto it = index_.find(name);
        if (it == index_.end()) fail("pack: tensor '", name, "' is missing");
        if (it->second->dtype != dtype || it->second->shape != shape) {
            fail("pack: tensor '", name, "' is ", dtype_name(it->second->dtype), " ", dims_to_string(it->second->shape),
                 ", expected ", dtype_name(dtype), " ", dims_to_string(shape));
        }
        return *it->second;
    };
    for (const auto& [name, entry] : manifest.at("tensors").items()) {
        const std::string format = entry.at("format").get<std::string>();
        const DType dtype = format == "q4" ? DType::Q4 : format == "q8" ? DType::Q8 : DType::F32;
        if (dtype == DType::F32) fail("pack: tensor '", name, "' has unknown format '", format, "'");
        const int64_t group = entry.at("group").get<int64_t>();
        const Dims shape = entry.at("shape").get<Dims>();
        XS_CHECK(shape.size() >= 2 && group > 0 && shape.back() % group == 0 &&
                     (dtype == DType::Q8 || shape.back() % 2 == 0),
                 "pack: tensor '", name, "' has shape ", dims_to_string(shape), " and group ", group);
        // Components keep the leading dims; the last is columns, column pairs (q4 codes) or groups.
        const auto with_last = [&](int64_t last) {
            Dims d = shape;
            d.back() = last;
            return d;
        };
        const Dims groups = with_last(shape.back() / group);
        TensorView v;
        v.dtype = dtype;
        v.shape = shape;
        v.data = component(name, dtype == DType::Q4 ? DType::U8 : DType::I8,
                           dtype == DType::Q4 ? with_last(shape.back() / 2) : shape).data;
        v.quant.group = group;
        v.quant.scales = component(name + ".scales", DType::F16, groups).data;
        index_.erase(name + ".scales");
        if (dtype == DType::Q4) {
            v.quant.biases = component(name + ".biases", DType::F16, groups).data;
            index_.erase(name + ".biases");
        }
        if (entry.value("input_scales", false)) {
            v.quant.input_scales =
                static_cast<const float*>(component(name + ".input_scales", DType::F32, {shape.back()}).data);
            index_.erase(name + ".input_scales");
        }
        index_[name] = &(quant_views_[name] = v);
    }
    is_pack_ = true;
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
