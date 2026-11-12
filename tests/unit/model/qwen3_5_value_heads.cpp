// Linear attention's value heads in the tiled order (config.json linear_value_head_order "tiled", the order of GGUF
// files): the tiny Qwen3.5 rewritten with every value-head tensor in that order is the same model, step after step,
// and read as grouped it is not.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend/device_test_support.h"
#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "models/model.h"
#include "models/qwen3_5/qwen3_5.h"
#include "tiny_checkpoints.h"
#include "util/error.h"

using namespace xs;
namespace fs = std::filesystem;
namespace dev = xs::test::device;
using xs::test::TempDir;
using xs::test::TinyQwen;

namespace {

bool ends_with(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// `bytes` of a tensor whose dim `axis` holds, from index `first`, hv value heads of `unit` entries each, rewritten so
// that tiled head s * hk + k holds grouped head k * r + s.
std::vector<char> tiled(const TensorView& v, int axis, int64_t first, int64_t unit, int64_t hk, int64_t hv) {
    const int64_t r = hv / hk;
    const size_t esize = dtype_size(v.dtype);
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= v.shape[static_cast<size_t>(d)];
    for (size_t d = static_cast<size_t>(axis) + 1; d < v.shape.size(); ++d)
        inner *= v.shape[d];
    const int64_t n = v.shape[static_cast<size_t>(axis)];
    const auto* in = static_cast<const char*>(v.data);
    std::vector<char> out(in, in + v.nbytes());
    const size_t run = static_cast<size_t>(unit * inner) * esize;  // one head's entries along the axis, contiguous
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t s = 0; s < r; ++s) {
            for (int64_t k = 0; k < hk; ++k) {
                const int64_t to = first + (s * hk + k) * unit, from = first + (k * r + s) * unit;
                std::memcpy(out.data() + static_cast<size_t>((o * n + to) * inner) * esize,
                            in + static_cast<size_t>((o * n + from) * inner) * esize, run);
            }
        }
    }
    return out;
}

// The checkpoint in `from` with every value-head tensor in the tiled order; `flag`: config.json says so.
void write_tiled(const fs::path& from, const fs::path& to, const TinyQwen& m, bool flag) {
    fs::create_directories(to);
    const auto source = SafetensorsWeights::open_dir(from);
    const int64_t hk = m.lin_k_heads, hv = m.lin_v_heads, d = m.lin_dim;
    std::vector<OutUnit> units;
    for (const std::string& name : source->names()) {
        const TensorView& v = source->get(name);
        std::vector<char> bytes(static_cast<const char*>(v.data), static_cast<const char*>(v.data) + v.nbytes());
        if (ends_with(name, "in_proj_qkv.weight") || ends_with(name, "conv1d.weight")) {
            bytes = tiled(v, 0, 2 * hk * d, d, hk, hv);
        } else if (ends_with(name, "in_proj_z.weight")) {
            bytes = tiled(v, 0, 0, d, hk, hv);
        } else if (ends_with(name, "in_proj_a.weight") || ends_with(name, "in_proj_b.weight") ||
                   ends_with(name, "A_log") || ends_with(name, "dt_bias")) {
            bytes = tiled(v, 0, 0, 1, hk, hv);
        } else if (ends_with(name, "out_proj.weight")) {
            bytes = tiled(v, 1, 0, d, hk, hv);
        }
        OutUnit u;
        u.tensors = {OutTensor{name, v.dtype, v.shape}};
        u.write = [held = std::make_shared<std::vector<char>>(std::move(bytes))](std::ostream& out) {
            out.write(held->data(), static_cast<std::streamsize>(held->size()));
        };
        units.push_back(std::move(u));
    }
    write_safetensors(to / "model.safetensors", units);
    nlohmann::json config = nlohmann::json::parse(std::ifstream(from / "config.json"));
    if (flag) config["linear_value_head_order"] = "tiled";
    std::ofstream(to / "config.json") << config.dump(1);
    fs::copy_file(from / "tokenizer_config.json", to / "tokenizer_config.json");
}

// The logits of a prefill and of three decode steps after it (the conv and recurrent state carried).
std::vector<std::vector<float>> step_logits(Backend& backend, const Model& model, const SessionOptions& options) {
    const auto session = backend.compile(model.graph(), model.weights(), options);
    std::vector<std::vector<float>> out;
    const std::vector<std::vector<int32_t>> steps{{3, 17, 42, 99, 5, 250, 7, 8, 64, 128}, {11}, {200}, {1}};
    int64_t pos = 0;
    for (const auto& ids : steps) {
        TensorMap inputs, outputs;
        model.prepare_inputs(ids, pos, {}, inputs);
        session->run(inputs, outputs);
        const HostTensor& l = outputs.at("logits");
        out.emplace_back(l.data<float>(), l.data<float>() + l.numel());
        pos += static_cast<int64_t>(ids.size());
    }
    return out;
}

// The largest difference between the two runs' logits, over the largest logit.
double relative_gap(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    double gap = 0.0, scale = 0.0;
    for (size_t s = 0; s < a.size(); ++s) {
        for (size_t i = 0; i < a[s].size(); ++i) {
            gap = std::max(gap, static_cast<double>(std::fabs(a[s][i] - b[s][i])));
            scale = std::max(scale, static_cast<double>(std::fabs(b[s][i])));
        }
    }
    return gap / scale;
}

}  // namespace

TEST_SUITE("qwen3_5 value head order") {
    TEST_CASE("a checkpoint with tiled value heads is the same model when its config says so") {
        const TempDir work("xs_qwen35_tiled_test");
        const TinyQwen tiny;  // 2 key heads, 4 value heads: the orders differ
        REQUIRE(tiny.lin_v_heads > tiny.lin_k_heads);
        xs::test::write_tiny_qwen(work.path / "grouped", tiny, 11);
        write_tiled(work.path / "grouped", work.path / "tiled", tiny, true);
        write_tiled(work.path / "grouped", work.path / "unflagged", tiny, false);
        const auto grouped = load_model(work.path / "grouped");
        const auto tiled = load_model(work.path / "tiled");
        const auto unflagged = load_model(work.path / "unflagged");
        // Four ops more per linear-attention layer, none in a grouped checkpoint.
        const auto linear_layers =
            static_cast<int64_t>(std::count(tiny.layers.begin(), tiny.layers.end(), std::string("linear_attention")));
        CHECK(static_cast<int64_t>(tiled->graph().nodes().size()) ==
              static_cast<int64_t>(grouped->graph().nodes().size()) + 4 * 3 * linear_layers);
        CHECK(unflagged->graph().nodes().size() == grouped->graph().nodes().size());

        const auto oracle = make_cpu_ref_backend();
        SessionOptions so;
        so.precision = DType::F32;
        const auto reference = step_logits(*oracle, *grouped, so);
        const double gap = relative_gap(step_logits(*oracle, *tiled, so), reference);
        MESSAGE("tiled against grouped: largest logit difference " << gap << " of the largest logit");
        CHECK(gap < 1e-6);  // only out_proj's sums run in another order
        CHECK(relative_gap(step_logits(*oracle, *unflagged, so), reference) > 1e-2);  // the order matters
        if (dev::enabled()) {
            SessionOptions gpu = dev::options();
            gpu.precision = DType::F32;
            CHECK(relative_gap(step_logits(dev::backend(), *tiled, gpu), step_logits(dev::backend(), *grouped, gpu)) <
                  1e-4);
        }
    }

    TEST_CASE("with one value head per key head the orders are the same, and nothing is added") {
        const TempDir work("xs_qwen35_tiled_square_test");
        TinyQwen tiny;
        tiny.lin_v_heads = tiny.lin_k_heads;
        xs::test::write_tiny_qwen(work.path / "grouped", tiny, 5);
        write_tiled(work.path / "grouped", work.path / "tiled", tiny, true);
        const auto grouped = load_model(work.path / "grouped");
        const auto tiled = load_model(work.path / "tiled");
        CHECK(tiled->graph().nodes().size() == grouped->graph().nodes().size());
        const auto oracle = make_cpu_ref_backend();
        SessionOptions so;
        so.precision = DType::F32;
        CHECK(step_logits(*oracle, *tiled, so) == step_logits(*oracle, *grouped, so));
    }
}
