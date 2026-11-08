// `convert` end to end on a tiny random Qwen3.5 checkpoint written to disk: which tensors are quantized, the pack
// layout and manifest, that the loaded pack reproduces the quantizer's values exactly, logits close to the original
// model on cpu-ref, and (with XS_DEVICE_TEST_DEVICE) the same pack on the GPU backend matching cpu-ref.
#include <doctest/doctest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#include "backend/backend_test_util.h"
#include "backend/cpu_ref/cpu_backend.h"
#include "backend/device_test_support.h"
#include "convert/convert.h"
#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "models/model.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/half.h"

using namespace xs;
namespace fs = std::filesystem;
namespace dev = xs::test::device;

namespace {

struct TinyQwen {
    int64_t vocab = 256, hidden = 128, ffn = 256, heads = 4, kv_heads = 2, head_dim = 64;
    int64_t lin_k_heads = 2, lin_v_heads = 4, lin_dim = 32, conv = 4;
    std::vector<std::string> layers{"linear_attention", "linear_attention", "linear_attention", "full_attention"};
    int64_t experts = 0, top_k = 2, moe_ffn = 64, shared_ffn = 96;  // experts > 0: Qwen3.5 MoE (fused banks)
};

// Writes config.json and model.safetensors (bf16, random) for `m` into `dir`.
void write_tiny_qwen(const fs::path& dir, const TinyQwen& m, uint32_t seed) {
    fs::create_directories(dir);
    nlohmann::json config = {
        {"model_type", "qwen3_5_text"}, {"vocab_size", m.vocab}, {"hidden_size", m.hidden},
        {"intermediate_size", m.ffn}, {"num_hidden_layers", m.layers.size()}, {"rms_norm_eps", 1e-6},
        {"max_position_embeddings", 4096}, {"num_attention_heads", m.heads}, {"num_key_value_heads", m.kv_heads},
        {"head_dim", m.head_dim}, {"linear_num_key_heads", m.lin_k_heads}, {"linear_num_value_heads", m.lin_v_heads},
        {"linear_key_head_dim", m.lin_dim}, {"linear_value_head_dim", m.lin_dim}, {"linear_conv_kernel_dim", m.conv},
        {"hidden_act", "silu"}, {"tie_word_embeddings", true}, {"eos_token_id", 1}, {"layer_types", m.layers},
        {"rope_parameters", {{"rope_type", "default"}, {"rope_theta", 1e7}, {"partial_rotary_factor", 0.25}}}};
    if (m.experts > 0) {
        config["model_type"] = "qwen3_5_moe_text";
        config.erase("intermediate_size");
        config["num_experts"] = m.experts;
        config["num_experts_per_tok"] = m.top_k;
        config["moe_intermediate_size"] = m.moe_ffn;
        config["shared_expert_intermediate_size"] = m.shared_ffn;
    }
    std::ofstream(dir / "config.json") << config.dump(1);
    std::ofstream(dir / "tokenizer_config.json") << R"({"eos_token": "<|end|>"})";

    std::mt19937 rng(seed);
    std::vector<std::pair<std::string, Dims>> specs;
    const auto add = [&](const std::string& name, Dims shape) { specs.emplace_back("model." + name, std::move(shape)); };
    add("embed_tokens.weight", {m.vocab, m.hidden});
    add("norm.weight", {m.hidden});
    const int64_t conv_width = 2 * m.lin_k_heads * m.lin_dim + m.lin_v_heads * m.lin_dim;
    for (size_t i = 0; i < m.layers.size(); ++i) {
        const std::string L = "layers." + std::to_string(i) + ".";
        add(L + "input_layernorm.weight", {m.hidden});
        add(L + "post_attention_layernorm.weight", {m.hidden});
        if (m.experts > 0) {
            add(L + "mlp.gate.weight", {m.experts, m.hidden});
            add(L + "mlp.experts.gate_up_proj", {m.experts, 2 * m.moe_ffn, m.hidden});
            add(L + "mlp.experts.down_proj", {m.experts, m.hidden, m.moe_ffn});
            add(L + "mlp.shared_expert.gate_proj.weight", {m.shared_ffn, m.hidden});
            add(L + "mlp.shared_expert.up_proj.weight", {m.shared_ffn, m.hidden});
            add(L + "mlp.shared_expert.down_proj.weight", {m.hidden, m.shared_ffn});
            add(L + "mlp.shared_expert_gate.weight", {1, m.hidden});
        } else {
            add(L + "mlp.gate_proj.weight", {m.ffn, m.hidden});
            add(L + "mlp.up_proj.weight", {m.ffn, m.hidden});
            add(L + "mlp.down_proj.weight", {m.hidden, m.ffn});
        }
        if (m.layers[i] == "linear_attention") {
            const std::string A = L + "linear_attn.";
            add(A + "in_proj_qkv.weight", {conv_width, m.hidden});
            add(A + "conv1d.weight", {conv_width, 1, m.conv});
            add(A + "in_proj_b.weight", {m.lin_v_heads, m.hidden});
            add(A + "in_proj_a.weight", {m.lin_v_heads, m.hidden});
            add(A + "A_log", {m.lin_v_heads});
            add(A + "dt_bias", {m.lin_v_heads});
            add(A + "norm.weight", {m.lin_dim});
            add(A + "in_proj_z.weight", {m.lin_v_heads * m.lin_dim, m.hidden});
            add(A + "out_proj.weight", {m.hidden, m.lin_v_heads * m.lin_dim});
        } else {
            const std::string A = L + "self_attn.";
            add(A + "q_proj.weight", {2 * m.heads * m.head_dim, m.hidden});
            add(A + "k_proj.weight", {m.kv_heads * m.head_dim, m.hidden});
            add(A + "v_proj.weight", {m.kv_heads * m.head_dim, m.hidden});
            add(A + "o_proj.weight", {m.hidden, m.heads * m.head_dim});
            add(A + "q_norm.weight", {m.head_dim});
            add(A + "k_norm.weight", {m.head_dim});
        }
    }
    std::vector<std::vector<uint16_t>> data;
    data.reserve(specs.size());  // the units below keep references into it
    std::vector<OutUnit> units;
    for (const auto& [name, shape] : specs) {
        // 1/sqrt(fan-in) for matrices keeps activations O(1); small values for norms (zero-centered) and the rest. The
        // router is larger, so each token's top-k experts win clearly (a near tie may route differently at f16).
        const bool router = name.size() > 15 && name.compare(name.size() - 15, 15, "mlp.gate.weight") == 0;
        const float range = shape.size() >= 2 ? (router ? 8.0f : 1.0f) / std::sqrt(static_cast<float>(shape.back()))
                                              : 0.1f;
        std::uniform_real_distribution<float> dist(-range, range);
        std::vector<uint16_t> values(static_cast<size_t>(numel(shape)));
        for (uint16_t& v : values)
            v = f32_to_bf16(dist(rng));
        data.push_back(std::move(values));
        const std::vector<uint16_t>& bytes = data.back();
        OutUnit u;
        u.tensors = {OutTensor{name, DType::BF16, shape}};
        u.write = [&bytes](std::ostream& out) {
            out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size() * 2));
        };
        units.push_back(std::move(u));
    }
    write_safetensors(dir / "model.safetensors", units);
}

// Logits of the last prompt token, from a fresh state.
std::vector<float> last_logits(Session& session, const Model& model, const std::vector<int32_t>& ids) {
    session.reset();
    TensorMap inputs, outputs;
    model.prepare_inputs(ids, 0, {}, inputs);
    session.run(inputs, outputs);
    const HostTensor& l = outputs.at("logits");
    return {l.data<float>(), l.data<float>() + l.numel()};
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ab += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
    }
    return ab / std::sqrt(aa * bb);
}

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name) : path(fs::temp_directory_path() / name) { fs::remove_all(path); }
    ~TempDir() { fs::remove_all(path); }
};

}  // namespace

TEST_SUITE("convert") {
    TEST_CASE("q8 and q4 packs of a tiny Qwen3.5: plan, layout, exact values, logits close to the original") {
        const TempDir work("xs_convert_test");
        const fs::path src = work.path / "tiny-qwen";
        write_tiny_qwen(src, TinyQwen{}, 3);
        const auto original = load_model(src);
        const auto oracle = make_cpu_ref_backend();
        SessionOptions so;
        so.precision = DType::F32;
        const std::vector<int32_t> ids{3, 17, 42, 99, 5, 250, 7, 8, 64, 128};
        const std::vector<float> reference = last_logits(*oracle->compile(original->graph(), original->weights(), so),
                                                         *original, ids);
        double cos_q8 = 0;
        for (const DType format : {DType::Q8, DType::Q4}) {
            CAPTURE(dtype_name(format));
            const fs::path out = work.path / (format == DType::Q4 ? "tiny-q4" : "tiny-q8");
            ConvertOptions o;
            o.source = src;
            o.output = out;
            o.format = format;
            o.group = 32;
            o.min_elements = 0;  // the tiny matrices count
            const ConvertReport r = convert_model(o);
            // 3 linear layers x 8 matmuls + 1 full layer x 7; the tied embedding goes to q8.
            CHECK(r.q8 == (format == DType::Q8 ? 32 : 1));
            CHECK(r.q4 == (format == DType::Q4 ? 31 : 0));
            CHECK(r.output_bytes < r.source_bytes);
            CHECK(fs::exists(out / "config.json"));
            CHECK(fs::exists(out / "tokenizer_config.json"));
            CHECK(fs::exists(out / "model.safetensors.index.json"));
            const auto manifest = nlohmann::json::parse(std::ifstream(out / kPackManifest));
            CHECK(manifest.at("pack_version") == kPackVersion);
            CHECK(manifest.at("source").at("name") == "tiny-qwen");
            CHECK(manifest.at("tensors").size() == static_cast<size_t>(r.q4 + r.q8));

            const auto pack = load_model(out);
            const TensorView& up = pack->weights().get("model.layers.0.mlp.up_proj.weight");
            CHECK(up.dtype == format);
            CHECK(up.shape == Dims{256, 128});
            CHECK(pack->weights().get("model.embed_tokens.weight").dtype == DType::Q8);
            CHECK(pack->weights().get("model.layers.0.linear_attn.conv1d.weight").dtype == DType::BF16);
            CHECK_FALSE(pack->weights().contains("model.layers.0.mlp.up_proj.weight.scales"));
            // The pack holds exactly what the quantizer chose for the original weights.
            const quant::Quantized q =
                quant::quantize(original->weights().get("model.layers.0.mlp.up_proj.weight"), format, 32);
            std::vector<float> expected(256 * 128), got(256 * 128);
            quant::dequantize_rows(q.view(), 0, 256, expected.data());
            quant::dequantize_rows(up, 0, 256, got.data());
            CHECK(got == expected);

            const std::vector<float> logits =
                last_logits(*oracle->compile(pack->graph(), pack->weights(), so), *pack, ids);
            const double cos = cosine(logits, reference);
            MESSAGE(dtype_name(format), " pack vs bf16 original on cpu-ref: cosine ", cos);
            // Sanity bounds for random weights (a worst case: no structure); quality is measured on real models.
            CHECK(cos > (format == DType::Q8 ? 0.999 : 0.95));
            if (format == DType::Q8) cos_q8 = cos;
            if (format == DType::Q4) CHECK(cos < cos_q8);  // planted ordering: fewer bits, larger error

            if (dev::enabled()) {
                SessionOptions gpu = dev::options();
                gpu.precision = DType::F16;
                const std::vector<float> device_logits =
                    last_logits(*dev::backend().compile(pack->graph(), pack->weights(), gpu), *pack, ids);
                const double agree = cosine(device_logits, logits);
                MESSAGE(dtype_name(format), " pack on the GPU vs cpu-ref: cosine ", agree);
                CHECK(agree > 0.9999);
            }
        }
    }

    TEST_CASE("q4 / q8 packs of a tiny Qwen3.5 MoE: expert banks quantized whole, GPU agrees with cpu-ref") {
        const TempDir work("xs_convert_moe_test");
        const fs::path src = work.path / "tiny-qwen-moe";
        TinyQwen spec;
        spec.experts = 8;
        write_tiny_qwen(src, spec, 5);
        const auto original = load_model(src);
        const auto oracle = make_cpu_ref_backend();
        SessionOptions so;
        so.precision = DType::F32;
        // 10 tokens x top-2 = 20 (token, expert) pairs: the direct path; 40 tokens = 80 pairs: grouped by expert.
        std::vector<std::vector<int32_t>> prompts{{3, 17, 42, 99, 5, 250, 7, 8, 64, 128}, {}};
        for (int32_t i = 0; i < 40; ++i)
            prompts[1].push_back((i * 37 + 11) % 256);
        for (const DType format : {DType::Q8, DType::Q4}) {
            CAPTURE(dtype_name(format));
            const fs::path out = work.path / (format == DType::Q4 ? "moe-q4" : "moe-q8");
            ConvertOptions o;
            o.source = src;
            o.output = out;
            o.format = format;
            o.group = 32;
            o.min_elements = 0;
            convert_model(o);
            const auto pack = load_model(out);
            const std::string bank = "model.layers.0.mlp.experts.gate_up_proj";
            const TensorView& up = pack->weights().get(bank);
            CHECK(up.dtype == format);
            CHECK(up.shape == Dims{8, 128, 128});
            CHECK(pack->weights().get("model.layers.2.mlp.experts.down_proj").shape == Dims{8, 128, 64});
            // The bank holds what the quantizer chose for its E * N rows.
            const quant::Quantized q = quant::quantize(original->weights().get(bank), format, 32);
            std::vector<float> expected(8 * 128 * 128), got(8 * 128 * 128);
            quant::dequantize_rows(q.view(), 0, 8 * 128, expected.data());
            quant::dequantize_rows(up, 0, 8 * 128, got.data());
            CHECK(got == expected);

            // No closeness bound against the bf16 original here: in a random MoE any rounding upstream flips some
            // near-tied top-k choices, which moves the logits far more than the rounding itself. The GPU must agree
            // with cpu-ref on the same pack; at f32, so f16 noise cannot flip a tie either.
            const auto cpu = oracle->compile(pack->graph(), pack->weights(), so);
            for (const auto& ids : prompts) {
                CAPTURE(ids.size());
                const std::vector<float> logits = last_logits(*cpu, *pack, ids);
                if (dev::enabled()) {
                    SessionOptions gpu = dev::options();
                    gpu.precision = DType::F32;
                    const std::vector<float> device_logits =
                        last_logits(*dev::backend().compile(pack->graph(), pack->weights(), gpu), *pack, ids);
                    const double agree = cosine(device_logits, logits);
                    MESSAGE(dtype_name(format), " MoE pack on the GPU vs cpu-ref (", ids.size(), " tokens): ", agree);
                    CHECK(agree > 0.99999);
                }
            }
        }
    }

    TEST_CASE("awq q4 pack: input scales stored and used, logits at least as close as round to nearest") {
        const TempDir work("xs_convert_awq");
        const fs::path src = work.path / "tiny-qwen";
        write_tiny_qwen(src, TinyQwen{}, 5);
        const auto original = load_model(src);
        const auto oracle = make_cpu_ref_backend();
        SessionOptions so;
        so.precision = DType::F32;
        const std::vector<int32_t> ids{9, 31, 77, 140, 2, 201, 66, 13, 99, 180};
        const std::vector<float> reference = last_logits(*oracle->compile(original->graph(), original->weights(), so),
                                                         *original, ids);
        std::vector<int32_t> calibration(1024);
        for (size_t i = 0; i < calibration.size(); ++i)
            calibration[i] = static_cast<int32_t>((i * 7919 + 13) % 256);
        double cos_rtn = 0;
        for (const QuantMethod method : {QuantMethod::Rtn, QuantMethod::Awq}) {
            ConvertOptions o;
            o.source = src;
            o.output = work.path / (method == QuantMethod::Awq ? "awq" : "rtn");
            o.format = DType::Q4;
            o.group = 32;
            o.min_elements = 0;
            o.method = method;
            o.backend = "cpu-ref";
            o.calibration_ids = calibration;
            const ConvertReport r = convert_model(o);
            const auto pack = load_model(o.output);
            const TensorView& up = pack->weights().get("model.layers.0.mlp.up_proj.weight");
            CHECK((up.quant.input_scales != nullptr) == (method == QuantMethod::Awq));
            const std::vector<float> logits =
                last_logits(*oracle->compile(pack->graph(), pack->weights(), so), *pack, ids);
            const double cos = cosine(logits, reference);
            MESSAGE((method == QuantMethod::Awq ? "awq" : "rtn"), " q4 vs original: cosine ", cos);
            if (method == QuantMethod::Rtn) {
                cos_rtn = cos;
                CHECK(r.awq == 0);
            } else {
                CHECK(r.awq == r.q4);
                CHECK(r.awq_gain > 0);
                CHECK(cos >= cos_rtn - 0.002);
                if (dev::enabled()) {
                    SessionOptions gpu = dev::options();
                    gpu.precision = DType::F16;
                    const std::vector<float> device_logits =
                        last_logits(*dev::backend().compile(pack->graph(), pack->weights(), gpu), *pack, ids);
                    CHECK(cosine(device_logits, logits) > 0.9999);
                }
            }
        }
    }

    TEST_CASE("the plan quantizes the weights of bandwidth-bound matmuls; scope all takes every matmul weight") {
        // Rows per call decide, not names: 50 rows (an action expert) or a step-dependent count (decode) are
        // bandwidth-bound, 500 rows (an image encoder, a policy's prefix) compute-bound.
        test::MemoryWeights w;
        test::TensorRng rng(9);
        for (const char* name : {"few", "many", "shared", "decode", "table"}) w.add(name, rng.uniform(DType::F32, {2048, 512}));
        graph::Graph g;
        const graph::Value x50 = g.input("x50", DType::F32, {50, 512});
        const graph::Value x500 = g.input("x500", DType::F32, {500, 512});
        g.output("a", g.matmul(x50, g.weight(w, "few")));
        g.output("b", g.matmul(x500, g.weight(w, "many")));
        g.output("c", g.matmul(x50, g.weight(w, "shared")));
        g.output("d", g.matmul(x500, g.weight(w, "shared")));
        g.output("e", g.matmul(g.input("t", DType::F32, {graph::kT, 512}), g.weight(w, "decode")));
        g.output("f", g.gather(g.weight(w, "table"), g.input("ids", DType::I32, {4})));
        ConvertOptions o;
        o.group = 32;
        const auto plan = plan_graph_formats(g, w, DType::Q4, o);
        CHECK(plan.at("few") == DType::Q4);
        CHECK(plan.at("decode") == DType::Q4);
        CHECK(plan.at("table") == DType::Q8);
        CHECK(plan.count("many") == 0);
        CHECK(plan.count("shared") == 0);  // one compute-bound reader keeps it
        o.scope = QuantScope::All;
        const auto all = plan_graph_formats(g, w, DType::Q4, o);
        CHECK(all.at("many") == DType::Q4);
        CHECK(all.at("shared") == DType::Q4);
        CHECK(all.size() == 5);
    }

    TEST_CASE("refuses to overwrite, to convert a pack, and bad formats") {
        const TempDir work("xs_convert_refusals");
        const fs::path src = work.path / "tiny";
        write_tiny_qwen(src, TinyQwen{}, 4);
        ConvertOptions o;
        o.source = src;
        o.output = src;  // not empty
        CHECK_THROWS_WITH_AS(convert_model(o), doctest::Contains("not empty"), Error);
        o.output = work.path / "pack";
        o.min_elements = 0;
        o.group = 32;
        convert_model(o);
        ConvertOptions again = o;
        again.source = o.output;
        again.output = work.path / "pack2";
        CHECK_THROWS_WITH_AS(convert_model(again), doctest::Contains("already a pack"), Error);
        ConvertOptions bad = o;
        bad.output = work.path / "pack3";
        bad.format = DType::F16;
        CHECK_THROWS_WITH_AS(convert_model(bad), doctest::Contains("q4 or q8"), Error);
    }
}
