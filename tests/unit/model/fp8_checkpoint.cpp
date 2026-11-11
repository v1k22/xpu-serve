// An FP8 checkpoint (quant/fp8.h) is the model it decodes to: a tiny Qwen3.5, dense and mixture of experts, written
// the way the FP8 releases store one and again with the decoded values in the usual layout. Same weights, same
// logits on cpu-ref (and with XS_DEVICE_TEST_DEVICE on the GPU), same packs out of `convert`.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend/device_test_support.h"
#include "convert/convert.h"
#include "io/safetensors.h"
#include "models/model.h"
#include "quant/quant.h"
#include "tiny_checkpoints.h"
#include "util/error.h"

using namespace xs;
namespace fs = std::filesystem;
namespace dev = xs::test::device;
using xs::test::TempDir;
using xs::test::TinyQwen;

namespace {

const std::vector<int32_t> kIds{3, 17, 42, 99, 5, 250, 7, 8, 64, 128};

std::vector<float> last_logits(Backend& backend, const Model& model, const SessionOptions& options) {
    const auto session = backend.compile(model.graph(), model.weights(), options);
    TensorMap inputs, outputs;
    model.prepare_inputs(kIds, 0, {}, inputs);
    session->run(inputs, outputs);
    const HostTensor& l = outputs.at("logits");
    return {l.data<float>(), l.data<float>() + l.numel()};
}

std::vector<float> values(const TensorView& v) {
    std::vector<float> out(static_cast<size_t>(v.numel()));
    quant::read_rows(v, 0, quant::matrix_rows(v.shape), out.data());
    return out;
}

// The tiny checkpoint and its two twins, loaded.
struct Twins {
    TempDir work;
    std::unique_ptr<Model> original, fp8, decoded;
    Twins(const std::string& name, const TinyQwen& tiny) : work(name) {
        xs::test::write_tiny_qwen(work.path / "tiny", tiny, 11);
        xs::test::write_fp8_twins(work.path / "tiny", work.path / "tiny-fp8", work.path / "tiny-decoded");
        original = load_model(work.path / "tiny");
        fp8 = load_model(work.path / "tiny-fp8");
        decoded = load_model(work.path / "tiny-decoded");
    }
};

// Every weight of the two models has the same shape and values; the FP8 one shows no scale tensors of its own.
void check_same_weights(const Model& fp8, const Model& decoded) {
    CHECK(fp8.weights().names() == decoded.weights().names());
    for (const std::string& name : decoded.weights().names()) {
        CAPTURE(name);
        const TensorView& a = fp8.weights().get(name);
        const TensorView& b = decoded.weights().get(name);
        REQUIRE(a.shape == b.shape);
        if (a.dtype == DType::F8) CHECK(values(a) == values(b));
    }
}

void check_same_logits(const Twins& t) {
    const auto oracle = make_cpu_ref_backend();
    SessionOptions so;
    so.precision = DType::F32;
    const std::vector<float> reference = last_logits(*oracle, *t.decoded, so);
    CHECK(last_logits(*oracle, *t.fp8, so) == reference);
    CHECK(last_logits(*oracle, *t.original, so) != reference);  // the twins are FP8's values, not the original's
    if (dev::enabled()) {
        for (const DType precision : {DType::F16, DType::F32}) {
            CAPTURE(dtype_name(precision));
            SessionOptions gpu = dev::options();
            gpu.precision = precision;
            CHECK(last_logits(dev::backend(), *t.fp8, gpu) == last_logits(dev::backend(), *t.decoded, gpu));
        }
    }
}

// `convert` writes the same pack from either twin: the quantized tensors byte for byte, and what it keeps of the FP8
// weights as their f32 values.
void check_same_packs(const Twins& t, DType format) {
    ConvertOptions o;
    o.format = format;
    o.group = 32;
    o.min_elements = 20000;  // the smallest projections (16,384 elements) are kept
    o.source = t.work.path / "tiny-fp8";
    o.output = t.work.path / "pack-fp8";
    const ConvertReport from_fp8 = convert_model(o);
    o.source = t.work.path / "tiny-decoded";
    o.output = t.work.path / "pack-decoded";
    const ConvertReport from_decoded = convert_model(o);
    CHECK(from_fp8.q4 == from_decoded.q4);
    CHECK(from_fp8.q8 == from_decoded.q8);
    CHECK(from_fp8.q4 + from_fp8.q8 > 0);
    CHECK(from_fp8.kept == from_decoded.kept);
    const auto a = SafetensorsWeights::open_dir(t.work.path / "pack-fp8");
    const auto b = SafetensorsWeights::open_dir(t.work.path / "pack-decoded");
    CHECK(a->names() == b->names());
    int kept_fp8 = 0;
    for (const std::string& name : b->names()) {
        CAPTURE(name);
        const TensorView& x = a->get(name);
        const TensorView& y = b->get(name);
        REQUIRE(x.dtype == y.dtype);  // no f8 in a pack
        REQUIRE(x.shape == y.shape);
        CHECK(std::memcmp(x.data, y.data, x.nbytes()) == 0);
        if (is_quantized(x.dtype)) CHECK(values(x) == values(y));
        if (t.fp8->weights().get(name).dtype == DType::F8 && !is_quantized(x.dtype)) ++kept_fp8;
    }
    CHECK(kept_fp8 > 0);
    const auto oracle = make_cpu_ref_backend();
    SessionOptions so;
    so.precision = DType::F32;
    CHECK(last_logits(*oracle, *load_model(t.work.path / "pack-fp8"), so) ==
          last_logits(*oracle, *load_model(t.work.path / "pack-decoded"), so));
    fs::remove_all(t.work.path / "pack-fp8");
    fs::remove_all(t.work.path / "pack-decoded");
}

}  // namespace

TEST_SUITE("fp8 checkpoint") {
    TEST_CASE("a dense Qwen3.5 stored as FP8: the weights, logits and packs of the checkpoint it decodes to") {
        const Twins t("xs_fp8_dense_test", TinyQwen{});
        const TensorView& up = t.fp8->weights().get("model.layers.0.mlp.up_proj.weight");
        CHECK(up.dtype == DType::F8);
        CHECK(up.shape == Dims{256, 128});
        CHECK_FALSE(t.fp8->weights().contains("model.layers.0.mlp.up_proj.weight_scale_inv"));
        CHECK(t.fp8->weights().get("model.embed_tokens.weight").dtype == DType::BF16);
        CHECK(t.fp8->weights().get("model.layers.0.linear_attn.in_proj_a.weight").dtype == DType::BF16);
        check_same_weights(*t.fp8, *t.decoded);
        check_same_logits(t);
        check_same_packs(t, DType::Q4);
        check_same_packs(t, DType::Q8);
    }

    TEST_CASE("a mixture of experts stored expert by expert: the fused banks, in expert order, gate rows first") {
        TinyQwen tiny;
        tiny.experts = 8;
        const Twins t("xs_fp8_moe_test", tiny);
        const TensorView& bank = t.fp8->weights().get("model.layers.0.mlp.experts.gate_up_proj");
        CHECK(bank.dtype == DType::F8);
        CHECK(bank.shape == Dims{8, 128, 128});
        CHECK(bank.fp8.size() == 16);  // a gate and an up run per expert, nothing copied
        CHECK(t.fp8->weights().get("model.layers.0.mlp.experts.down_proj").shape == Dims{8, 128, 64});
        CHECK_FALSE(t.fp8->weights().contains("model.layers.0.mlp.experts.0.gate_proj.weight"));
        CHECK(t.fp8->weights().get("model.layers.0.mlp.gate.weight").dtype == DType::BF16);  // the router
        check_same_weights(*t.fp8, *t.decoded);
        check_same_logits(t);
        check_same_packs(t, DType::Q4);
    }

    TEST_CASE("quantization_config: FP8 with 128 x 128 blocks is read, anything else is refused by name") {
        const TempDir work("xs_fp8_config_test");
        xs::test::write_tiny_qwen(work.path / "tiny", TinyQwen{}, 11);
        const auto refusal = [&](const nlohmann::json& quantization) -> std::string {
            nlohmann::json config = nlohmann::json::parse(std::ifstream(work.path / "tiny" / "config.json"));
            config["quantization_config"] = quantization;
            std::ofstream(work.path / "tiny" / "config.json") << config.dump(1);
            try {
                load_model(work.path / "tiny");
            } catch (const Error& e) {
                return e.what();
            }
            return "";
        };
        const nlohmann::json fp8 = {{"quant_method", "fp8"}, {"fmt", "e4m3"}, {"activation_scheme", "dynamic"},
                                    {"weight_block_size", {128, 128}}};
        CHECK(refusal(fp8).empty());
        CHECK(refusal(nullptr).empty());
        CHECK(refusal({{"quant_method", "bitsandbytes"}, {"load_in_4bit", true}}).find("'bitsandbytes'") !=
              std::string::npos);
        nlohmann::json other = fp8;
        other["weight_block_size"] = {64, 128};
        CHECK(refusal(other).find("weight_block_size must be [128, 128]") != std::string::npos);
        other = fp8;
        other["fmt"] = "e5m2";
        CHECK(refusal(other).find("'e5m2'") != std::string::npos);
        other = fp8;
        other["activation_scheme"] = "static";
        CHECK(refusal(other).find("'static'") != std::string::npos);
        other = fp8;
        other.erase("weight_block_size");
        CHECK(refusal(other).find("weight_block_size") != std::string::npos);
    }
}
