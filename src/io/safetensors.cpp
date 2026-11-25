#include "io/safetensors.h"

#include <cmath>
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

std::unique_ptr<SafetensorsWeights> SafetensorsWeights::open_dir(const std::filesystem::path& dir, Fp8 fp8) {
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
    if (fp8 == Fp8::Paired) weights->pair_fp8();
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
        // One scale per block of kFp8Block rows and columns, or one per row.
        const std::string grid_name = name + "_scale_inv", row_name = name + "_scale";
        const auto grid_it = index_.find(grid_name), row_it = index_.find(row_name);
        if (codes.shape.size() != 2 || (grid_it == index_.end()) == (row_it == index_.end())) {
            fail("fp8 tensor '", name, "' ", dims_to_string(codes.shape),
                 ": only a matrix beside its scales is read, '", grid_name, "' (one per block of ", kFp8Block,
                 " rows and columns) or '", row_name, "' (one per row)");
        }
        const bool per_row = row_it != index_.end();
        const auto it = per_row ? row_it : grid_it;
        const TensorView& scales_view = *it->second;
        Fp8Run run{static_cast<const uint8_t*>(codes.data), nullptr, codes.shape[0]};
        if (per_row) {
            run.block_rows = 1;
            run.block_cols = codes.shape[1];
            if (!is_float(scales_view.dtype) ||
                (scales_view.shape != Dims{codes.shape[0], 1} && scales_view.shape != Dims{codes.shape[0]})) {
                fail("fp8 tensor '", name, "' ", dims_to_string(codes.shape), ": its scales '", row_name, "' are ",
                     dtype_name(scales_view.dtype), " ", dims_to_string(scales_view.shape), ", expected one per row, ",
                     dims_to_string(Dims{codes.shape[0], 1}));
            }
        } else {
            const Dims blocks{(codes.shape[0] + kFp8Block - 1) / kFp8Block,
                              (codes.shape[1] + kFp8Block - 1) / kFp8Block};
            if (!is_float(scales_view.dtype) || scales_view.shape != blocks) {
                fail("fp8 tensor '", name, "' ", dims_to_string(codes.shape), ": its scales '", grid_name, "' are ",
                     dtype_name(scales_view.dtype), " ", dims_to_string(scales_view.shape),
                     ", expected one per block of ", kFp8Block, " rows and columns, ", dims_to_string(blocks));
            }
        }
        const std::vector<float> scales =
            HostTensor::borrow(scales_view.dtype, scales_view.shape, const_cast<void*>(scales_view.data)).to_f32();
        fp8_scales_.push_back(std::make_unique<float[]>(scales.size()));
        std::copy(scales.begin(), scales.end(), fp8_scales_.back().get());
        run.scales = fp8_scales_.back().get();
        const std::vector<Fp8Run>& runs = fp8_runs_[name] = {run};
        TensorView v;
        v.dtype = DType::F8;
        v.shape = codes.shape;
        v.fp8 = runs;
        index_.erase(it);
        index_[name] = &(fp8_views_[name] = v);
    }
}

void SafetensorsWeights::pair_mxfp8(const std::string& name) {
    const TensorView codes = get(name);
    const std::string scale_name = name + "_scale";
    const TensorView exponents = get(scale_name);
    Dims blocks = codes.shape;
    if (!blocks.empty()) blocks.back() = (blocks.back() + kMxfp8Block - 1) / kMxfp8Block;
    XS_CHECK(codes.dtype == DType::F8 && codes.fp8.empty() && codes.shape.size() >= 2 &&
                 exponents.dtype == DType::U8 && exponents.shape == blocks,
             "mxfp8 tensor '", name, "' ", dtype_name(codes.dtype), " ", dims_to_string(codes.shape), " beside '",
             scale_name, "' ", dtype_name(exponents.dtype), " ", dims_to_string(exponents.shape),
             ": expected f8 codes and one u8 exponent per ", kMxfp8Block, " columns of a row, ",
             dims_to_string(blocks));
    const auto count = static_cast<size_t>(exponents.numel());
    fp8_scales_.push_back(std::make_unique<float[]>(count));
    const auto* stored = static_cast<const uint8_t*>(exponents.data);
    for (size_t i = 0; i < count; ++i) fp8_scales_.back()[i] = std::ldexp(1.0f, static_cast<int>(stored[i]) - 127);
    Fp8Run run{static_cast<const uint8_t*>(codes.data), fp8_scales_.back().get(), codes.numel() / codes.shape.back()};
    run.block_rows = 1;
    run.block_cols = kMxfp8Block;
    TensorView v;
    v.dtype = DType::F8;
    v.shape = codes.shape;
    v.fp8 = fp8_runs_[name] = {run};
    index_.erase(scale_name);
    index_[name] = &(fp8_views_[name] = v);
}

void SafetensorsWeights::pair_nvfp4(const std::string& name) {
    const TensorView codes = get(name);
    const std::string scale_name = name + "_scale", amax_name = name + "_amax";
    const std::string denominator_name = name + "_outer_scale_denominator";
    XS_CHECK(codes.dtype == DType::U8 && codes.shape.size() == 3, "nvfp4 tensor '", name, "' is ",
             dtype_name(codes.dtype), " ", dims_to_string(codes.shape), ", expected u8 [runs, rows, cols / 2]");
    const TensorView scales = get(scale_name), amax = get(amax_name), denominator = get(denominator_name);
    const int64_t runs = codes.shape[0], rows = codes.shape[1], cols = 2 * codes.shape[2];
    XS_CHECK(cols % kFp4Block == 0 && scales.dtype == DType::F8 && scales.fp8.empty() &&
                 scales.shape == (Dims{runs, rows, cols / kFp4Block}),
             "nvfp4 tensor '", name, "' ", dims_to_string(codes.shape), ": its scales '", scale_name, "' are ",
             dtype_name(scales.dtype), " ", dims_to_string(scales.shape), ", expected f8 ",
             dims_to_string(Dims{runs, rows, cols / kFp4Block}));
    XS_CHECK(amax.dtype == DType::F32 && amax.shape == Dims{runs} && denominator.dtype == DType::F32 &&
                 denominator.numel() == 1,
             "nvfp4 tensor '", name, "': '", amax_name, "' ", dims_to_string(amax.shape), " and '", denominator_name,
             "' ", dims_to_string(denominator.shape), " must be f32 [", runs, "] and one f32");
    const float divisor = *static_cast<const float*>(denominator.data);
    std::vector<Fp4Run> parts;
    for (int64_t r = 0; r < runs; ++r) {
        parts.push_back(Fp4Run{static_cast<const uint8_t*>(codes.data) + r * rows * (cols / 2),
                               static_cast<const uint8_t*>(scales.data) + r * rows * (cols / kFp4Block),
                               static_cast<const float*>(amax.data)[r] / divisor, rows});
    }
    TensorView v;
    v.dtype = DType::F4;
    v.shape = {runs, rows, cols};
    v.fp4 = fp4_runs_[name] = std::move(parts);
    for (const std::string& gone : {scale_name, amax_name, denominator_name}) index_.erase(gone);
    index_[name] = &(fp4_views_[name] = v);
}

void SafetensorsWeights::join_rows(const std::string& name, const std::vector<std::string>& parts, Dims shape) {
    XS_CHECK(index_.count(name) == 0, "join_rows: '", name, "' exists already");
    XS_CHECK(shape.size() >= 2 && !parts.empty(), "join_rows: '", name, "' needs parts and at least 2 dims");
    const int64_t cols = shape.back();
    DType dtype = DType::F8;
    BlockFormat block = BlockFormat::Q4_0;
    int64_t rows = 0;
    // A part's format: its dtype, and a block format's name.
    const auto format = [](DType d, BlockFormat b) {
        return d == DType::Block ? std::string(block_format_name(b)) : std::string(dtype_name(d));
    };
    for (const std::string& part : parts) {
        const TensorView* v = find(part);
        if (v == nullptr) fail("join_rows: '", part, "', a part of '", name, "', is missing");
        if (&part == &parts.front()) {
            dtype = v->dtype;
            block = v->block;
        }
        const bool same = v->dtype == dtype && (dtype != DType::Block || v->block == block);
        if (!same || (dtype != DType::F8 && dtype != DType::Block && !is_float(dtype))) {
            fail("join_rows: '", part, "', a part of '", name, "', is ", format(v->dtype, v->block),
                 "; the parts of a weight are all fp8, all of one block format, or all floats of one dtype (the ",
                 "first is ", format(dtype, block), ")");
        }
        XS_CHECK(v->shape.size() >= 2 && v->shape.back() == cols, "join_rows: '", part, "' ",
                 dims_to_string(v->shape), " does not have the ", cols, " columns of '", name, "'");
        rows += v->numel() / cols;
    }
    XS_CHECK(rows * cols == numel(shape), "join_rows: the parts of '", name, "' have ", rows, " rows of ", cols,
             ", not the shape ", dims_to_string(shape));
    TensorView joined;
    joined.dtype = dtype;
    joined.block = block;
    joined.shape = std::move(shape);
    if (dtype == DType::F8) {
        std::vector<Fp8Run> runs;
        for (const std::string& part : parts) {
            const TensorView& v = *index_.at(part);
            runs.insert(runs.end(), v.fp8.begin(), v.fp8.end());
        }
        joined.fp8 = fp8_runs_[name] = std::move(runs);
    } else {
        joined_data_.push_back(std::make_unique<char[]>(joined.nbytes()));
        char* out = joined_data_.back().get();
        joined.data = out;
        for (const std::string& part : parts) {
            const TensorView& v = *index_.at(part);
            std::memcpy(out, v.data, v.nbytes());
            out += v.nbytes();
        }
    }
    for (const std::string& part : parts) {
        index_.erase(part);
        fp8_views_.erase(part);
        fp8_runs_.erase(part);
        joined_views_.erase(part);
    }
    index_[name] = &(joined_views_[name] = std::move(joined));
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
        if (const std::optional<BlockFormat> block = block_format_from_name(format)) {
            // A GGUF block format: the weight's rows of blocks as the file stores them, U8 [rows, row bytes].
            for (const auto& [key, value] : entry.items()) {
                if (key != "format" && key != "shape") {
                    fail("pack: tensor '", name, "' (", format, ") has key '", key,
                         "'; a block format entry has only format and shape");
                }
            }
            const Dims shape = entry.at("shape").get<Dims>();
            const int64_t weights = block_layout(*block).weights;
            XS_CHECK(shape.size() >= 2 && std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d > 0; }),
                     "pack: tensor '", name, "' (", format, ") has shape ", dims_to_string(shape),
                     "; a block format weight needs at least 2 dims, none empty");
            XS_CHECK(shape.back() % weights == 0, "pack: tensor '", name, "' (", format, ") has ", shape.back(),
                     " columns, not a whole number of its blocks of ", weights);
            const Dims stored{numel(shape) / shape.back(), static_cast<int64_t>(block_row_bytes(*block, shape.back()))};
            TensorView v;
            v.dtype = DType::Block;
            v.shape = shape;
            v.block = *block;
            v.data = component(name, DType::U8, stored).data;
            index_[name] = &(quant_views_[name] = v);
            continue;
        }
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
