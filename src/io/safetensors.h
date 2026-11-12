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
// quantized weight appears under its own name as one Q8 / Q4 view over its codes, `.scales` and `.biases` tensors,
// and each weight in a GGUF block format (a manifest entry {"format": "Q4_K", "shape": [..., cols]}, the format named
// as block_format_name does) as one Block view of that shape over its U8 tensor [rows, block_row_bytes(cols)].
// In an FP8 checkpoint (quant/fp8.h) each F8_E4M3 matrix appears as one F8 view over its codes and its
// `<name>_scale_inv` tensor.
class SafetensorsWeights final : public WeightStore {
public:
    static std::unique_ptr<SafetensorsWeights> open_dir(const std::filesystem::path& dir);
    // One file on its own (a head kept beside a model's shards, which open_dir does not read when an index lists them).
    static std::unique_ptr<SafetensorsWeights> open_file(const std::filesystem::path& path);

    const TensorView* find(const std::string& name) const override;
    std::vector<std::string> names() const override;
    size_t file_count() const { return files_.size(); }
    bool is_pack() const { return is_pack_; }

    // Shows the F8 matrices `parts` (the same number of columns each) as one weight `name` of `shape`, whose matrix
    // is their rows one after another, and drops their own names: for a checkpoint that stores in pieces what the
    // model reads as one tensor (an expert bank, expert by expert). Nothing is copied. Dense tensors are refused.
    void join_rows(const std::string& name, const std::vector<std::string>& parts, Dims shape);

private:
    void add_file(const std::filesystem::path& path);
    void apply_pack_manifest(const std::filesystem::path& path);
    void pair_fp8();

    std::vector<std::unique_ptr<SafetensorsFile>> files_;
    std::map<std::string, const TensorView*> index_;
    std::map<std::string, TensorView> quant_views_;
    // F8 weights: the views, the runs they point at and the scales read as f32.
    std::map<std::string, TensorView> fp8_views_;
    std::map<std::string, std::vector<Fp8Run>> fp8_runs_;
    std::vector<std::unique_ptr<float[]>> fp8_scales_;
    bool is_pack_ = false;
};

}  // namespace xs
