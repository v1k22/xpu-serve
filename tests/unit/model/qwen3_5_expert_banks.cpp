// A Qwen3.5 mixture of experts whose gate and up banks are stored apart (experts.gate_proj and experts.up_proj, as a
// GGUF import keeps them) is the model whose bank is fused (experts.gate_up_proj, as Hugging Face stores it): the tiny
// checkpoint with every fused bank cut in two gives the same logits, step after step, and with the two halves under
// each other's name it does not. A checkpoint with neither layout, with both, with one half or with a bank of another
// shape is refused by name.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend/device_test_support.h"
#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "models/model.h"
#include "step_logits.h"
#include "tiny_checkpoints.h"
#include "util/error.h"

using namespace xs;
namespace fs = std::filesystem;
namespace dev = xs::test::device;
using xs::test::relative_gap;
using xs::test::TempDir;
using xs::test::TinyQwen;

namespace {

// How a checkpoint written by write_banks stores each layer's gate and up banks.
enum class Banks {
    Apart,      // gate_proj and up_proj, the fused bank's halves
    Exchanged,  // planted: the same two, each holding the other's rows
    Neither,
    Both,      // the fused bank and its halves
    GateOnly,  // gate_proj without up_proj
    Reshaped,  // the halves as [2 E, I / 2, hidden]
};

// The checkpoint in `from` with every fused bank [E, 2 I, hidden] (an expert's gate rows, then its up rows) stored as
// `layout` says.
void write_banks(const fs::path& from, const fs::path& to, Banks layout) {
    fs::create_directories(to);
    const auto source = SafetensorsWeights::open_dir(from);
    std::vector<OutUnit> units;
    const auto add = [&](const std::string& name, DType dtype, Dims shape, std::vector<char> bytes) {
        OutUnit u;
        u.tensors = {OutTensor{name, dtype, std::move(shape)}};
        u.write = [held = std::make_shared<std::vector<char>>(std::move(bytes))](std::ostream& out) {
            out.write(held->data(), static_cast<std::streamsize>(held->size()));
        };
        units.push_back(std::move(u));
    };
    for (const std::string& name : source->names()) {
        const TensorView& v = source->get(name);
        const auto* bytes = static_cast<const char*>(v.data);
        const bool fused = name.ends_with("experts.gate_up_proj");
        if (!fused || layout == Banks::Both) add(name, v.dtype, v.shape, {bytes, bytes + v.nbytes()});
        if (!fused || layout == Banks::Neither) continue;
        const int64_t experts = v.shape[0], inter = v.shape[1] / 2, hidden = v.shape[2];
        const size_t half = static_cast<size_t>(inter * hidden) * dtype_size(v.dtype);
        std::vector<char> gate, up;
        for (int64_t e = 0; e < experts; ++e) {
            const char* at = bytes + static_cast<size_t>(2 * e) * half;
            gate.insert(gate.end(), at, at + half);
            up.insert(up.end(), at + half, at + 2 * half);
        }
        if (layout == Banks::Exchanged) std::swap(gate, up);
        const std::string stem = name.substr(0, name.rfind('.') + 1);
        const Dims shape =
            layout == Banks::Reshaped ? Dims{2 * experts, inter / 2, hidden} : Dims{experts, inter, hidden};
        add(stem + "gate_proj", v.dtype, shape, std::move(gate));
        if (layout != Banks::GateOnly) add(stem + "up_proj", v.dtype, shape, std::move(up));
    }
    write_safetensors(to / "model.safetensors", units);
    for (const char* file : {"config.json", "tokenizer_config.json"})
        fs::copy_file(from / file, to / file);
}

std::vector<std::vector<float>> step_logits(Backend& backend, const Model& model, const SessionOptions& options) {
    return xs::test::step_logits(backend, model, options, {3, 17, 42, 99, 5, 250, 7, 8, 64, 128}, {11, 200, 1});
}

}  // namespace

TEST_SUITE("qwen3_5 expert banks") {
    TEST_CASE("gate and up banks stored apart are the model with the fused bank") {
        const TempDir work("xs_qwen35_banks_test");
        TinyQwen tiny;
        tiny.experts = 8;
        xs::test::write_tiny_qwen(work.path / "fused", tiny, 11);
        write_banks(work.path / "fused", work.path / "apart", Banks::Apart);
        write_banks(work.path / "fused", work.path / "exchanged", Banks::Exchanged);
        const auto fused = load_model(work.path / "fused");
        const auto apart = load_model(work.path / "apart");
        const auto exchanged = load_model(work.path / "exchanged");
        const std::string E = "model.layers.0.mlp.experts.";
        CHECK(fused->weights().get(E + "gate_up_proj").shape == Dims{8, 128, 128});
        CHECK(apart->weights().get(E + "gate_proj").shape == Dims{8, 64, 128});
        CHECK(apart->weights().get(E + "up_proj").shape == Dims{8, 64, 128});
        CHECK_FALSE(apart->weights().contains(E + "gate_up_proj"));

        const auto oracle = make_cpu_ref_backend();
        SessionOptions so;
        so.precision = DType::F32;
        const auto reference = step_logits(*oracle, *fused, so);
        CHECK(step_logits(*oracle, *apart, so) == reference);  // every product is one row's, wherever the row is stored
        CHECK(relative_gap(step_logits(*oracle, *exchanged, so), reference) > 1e-2);
        if (dev::enabled()) {
            SessionOptions gpu = dev::options();
            gpu.precision = DType::F32;
            CHECK(relative_gap(step_logits(dev::backend(), *apart, gpu), step_logits(dev::backend(), *fused, gpu)) <
                  1e-4);
        }
    }

    TEST_CASE("a checkpoint with neither layout of the banks, both, one half or another shape is refused by name") {
        const TempDir work("xs_qwen35_banks_refusal_test");
        TinyQwen tiny;
        tiny.experts = 8;
        xs::test::write_tiny_qwen(work.path / "fused", tiny, 11);
        const auto refusal = [&](const char* name, Banks layout) -> std::string {
            write_banks(work.path / "fused", work.path / name, layout);
            try {
                load_model(work.path / name);
            } catch (const Error& e) {
                return e.what();
            }
            return "";
        };
        const auto has = [](const std::string& error, const char* text) {
            const bool found = error.find(text) != std::string::npos;
            if (!found) MESSAGE("the error is: '" << error << "'");
            return found;
        };
        CHECK(refusal("apart", Banks::Apart).empty());
        CHECK(has(refusal("neither", Banks::Neither),
                  "qwen3_5: model.layers.0.mlp.experts has no gate and up banks: they come fused "
                  "(experts.gate_up_proj), apart (experts.gate_proj and experts.up_proj)"));
        CHECK(has(refusal("both", Banks::Both),
                  "qwen3_5: model.layers.0.mlp.experts holds the fused bank (gate_up_proj) and the banks apart "
                  "(gate_proj, up_proj): a checkpoint has one layout or the other"));
        CHECK(has(refusal("gate", Banks::GateOnly), "weight 'model.layers.0.mlp.experts.up_proj' not found"));
        CHECK(has(refusal("reshaped", Banks::Reshaped),
                  "qwen3_5: model.layers.0.mlp.experts.gate_proj has shape [16, 32, 128], expected [8, 64, 128]"));
    }
}
