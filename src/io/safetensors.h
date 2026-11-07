#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensor/tensor.h"
#include "util/mmap_file.h"

namespace xs {

// Named, read-only weights. Model code and backends see weights only through this interface.
class WeightStore {
public:
    virtual ~WeightStore() = default;
    // nullptr when absent.
    virtual const TensorView* find(const std::string& name) const = 0;
    virtual std::vector<std::string> names() const = 0;

    // Throws a named error when absent.
    const TensorView& get(const std::string& name) const;
    bool contains(const std::string& name) const { return find(name) != nullptr; }
};

// One safetensors file, memory-mapped. Tensor data is never copied.
class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::filesystem::path& path);

    const std::map<std::string, TensorView>& tensors() const { return tensors_; }
    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    const std::filesystem::path& path() const { return file_.path(); }

private:
    MappedFile file_;
    std::map<std::string, TensorView> tensors_;
    std::map<std::string, std::string> metadata_;
};

// The manifest of an xpu-serve pack (DESIGN.md section 6.4): which tensors are quantized, in what format.
inline constexpr const char* kPackManifest = "xpu_pack.json";
inline constexpr int kPackVersion = 1;

// All safetensors files of a model directory: either model.safetensors.index.json + shards, or every *.safetensors
// file in the directory. Tensor names must be unique across files. In a pack (a directory with xpu_pack.json) each
// quantized weight appears under its own name as one Q8 / Q4 view over its codes, `.scales` and `.biases` tensors.
class SafetensorsWeights final : public WeightStore {
public:
    static std::unique_ptr<SafetensorsWeights> open_dir(const std::filesystem::path& dir);

    const TensorView* find(const std::string& name) const override;
    std::vector<std::string> names() const override;
    size_t file_count() const { return files_.size(); }
    bool is_pack() const { return is_pack_; }

private:
    void apply_pack_manifest(const std::filesystem::path& path);

    std::vector<std::unique_ptr<SafetensorsFile>> files_;
    std::map<std::string, const TensorView*> index_;
    std::map<std::string, TensorView> quant_views_;
    bool is_pack_ = false;
};

}  // namespace xs
