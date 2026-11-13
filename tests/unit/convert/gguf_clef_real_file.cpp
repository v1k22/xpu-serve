// A real clef GGUF (XS_GGUF_FILE, e.g. ggml-org/Clef-Flash-GGUF's Clef-Flash-Q4_K_M.gguf) against the Clef checkpoint
// it was converted from (XS_CLEF_DIR: its config.json and joint_head_config.json, and when there joint_head.safetensors
// and tokenizer.json; the decoder's weights are not read). config.json and joint_head_config.json give the model code
// what the checkpoint's do; the head's pack tensors are a converted checkpoint's, by name and shape, with its values
// (float tensors exactly, block formats within their rounding, the scales as computed from the three parameters); the
// tokenizer is the checkpoint's token by token; the imported pack loads as a decision model. Each case skips unless
// both variables are set and the file is a clef one.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "convert/gguf/architecture.h"
#include "convert/gguf/gguf_import.h"
#include "convert/real_gguf.h"
#include "io/gguf.h"
#include "io/safetensors.h"
#include "model/tiny_checkpoints.h"
#include "models/clef/clef.h"
#include "models/config_util.h"
#include "models/decision.h"
#include "models/policy.h"
#include "models/qwen3_5/qwen3_5.h"
#include "quant/quant.h"
#include "tokenizer/tokenizer.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::TempDir;

namespace {

std::optional<xs::test::RealGguf> real_file() {
    return xs::test::real_gguf("XS_CLEF_DIR", "clef");
}

std::vector<float> decoded(const TensorView& v) {
    if (v.shape.empty()) return HostTensor::borrow(v.dtype, v.shape, const_cast<void*>(v.data)).to_f32();  // a scalar
    std::vector<float> out(static_cast<size_t>(v.numel()));
    quant::read_rows(v, 0, quant::matrix_rows(v.shape), out.data());
    return out;
}

// |a - b| / |b| over the whole tensor (0 when both are zero).
double relative_error(const std::vector<float>& a, const std::vector<float>& b) {
    double diff = 0, norm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        diff += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
        norm += static_cast<double>(b[i]) * b[i];
    }
    return norm == 0 ? std::sqrt(diff) : std::sqrt(diff / norm);
}

}  // namespace

TEST_SUITE("gguf clef real file") {
    TEST_CASE("config: what the model code reads from a real clef GGUF equals the checkpoint's config") {
        const auto real = real_file();
        if (!real) return;
        const gguf::File file(real->gguf);
        const gguf_import::ImportPlan plan = gguf_import::find_architecture("clef").plan(file);
        const nlohmann::json ours = nlohmann::json::parse(plan.config.dump());
        const nlohmann::json theirs = cfg::read_json(real->hf / "config.json");
        const qwen3_5::Config a = qwen3_5::parse_config(ours);
        const qwen3_5::Config b = qwen3_5::parse_config(theirs);
        xs::test::check_decoder_config(a, b);
        CHECK(a.mtp_layers == 0);
        CHECK(ours["tie_word_embeddings"] == theirs.value("text_config", theirs)["tie_word_embeddings"]);

        const nlohmann::json head = nlohmann::json::parse(plan.files.at("joint_head_config.json").dump());
        CHECK(head == cfg::read_json(real->hf / "joint_head_config.json"));
        const clef::HeadConfig h = clef::parse_head_config(head);
        CHECK(h.hidden_size == a.hidden_size);
    }

    TEST_CASE("the head of a real clef GGUF is the checkpoint's: names, shapes and values") {
        const auto real = real_file();
        if (!real) return;
        const fs::path head_file = real->hf / "joint_head.safetensors";
        if (!fs::exists(head_file)) {
            MESSAGE("XS_CLEF_DIR has no joint_head.safetensors: the head's values are not compared");
            return;
        }
        const gguf::File file(real->gguf);
        const gguf_import::ImportPlan plan = gguf_import::find_architecture("clef").plan(file);
        std::map<std::string, const gguf_import::PackTensor*> pack;
        for (const gguf_import::PackTensor& p : plan.tensors) {
            if (p.name.starts_with("joint_head.")) pack[p.name] = &p;
        }

        // The checkpoint's head as a pack of it names it: each attention's in_proj cut in three by rows; the three
        // parameters apart.
        const auto checkpoint = SafetensorsWeights::open_file(head_file);
        std::map<std::string, std::pair<Dims, std::vector<float>>> expected;
        std::map<std::string, double> parameters;
        for (const std::string& name : checkpoint->names()) {
            const TensorView& v = checkpoint->get(name);
            const std::vector<float> values = decoded(v);
            if (name == "prior_logit_scale" || name == "joint_logit_scale" || name == "residual_gate") {
                parameters[name] = values.at(0);
                continue;
            }
            const size_t in_proj = name.find(".in_proj_");
            if (in_proj == std::string::npos) {
                expected["joint_head." + name] = {v.shape, values};
                continue;
            }
            const std::string stem = name.substr(0, in_proj + 1), part = name.substr(in_proj + 9);
            Dims shape = v.shape;
            shape[0] /= 3;
            const auto third = static_cast<std::ptrdiff_t>(values.size() / 3);
            for (int i = 0; i < 3; ++i) {
                expected["joint_head." + stem + "qkv"[i] + "_proj." + part] = {
                    shape, std::vector<float>(values.begin() + i * third, values.begin() + (i + 1) * third)};
            }
        }
        REQUIRE(parameters.size() == 3);
        std::vector<std::string> names, expected_names;
        for (const auto& [name, p] : pack)
            names.push_back(name);
        for (const auto& [name, e] : expected)
            expected_names.push_back(name);
        expected_names.push_back("joint_head.scales");
        std::sort(expected_names.begin(), expected_names.end());
        CHECK(names == expected_names);

        double worst = 0;
        std::string worst_name;
        for (const auto& [name, e] : expected) {
            CAPTURE(name);
            const auto it = pack.find(name);
            if (it == pack.end()) continue;
            const gguf_import::PackTensor& p = *it->second;
            REQUIRE(p.shape == e.first);
            const std::vector<float> ours = decoded(p.source->view(p.shape));
            if (p.source->dtype == DType::Block) {
                const double error = relative_error(ours, e.second);
                CHECK(error < 0.05);  // a block format's rounding; a wrong tensor is off by about 1.4
                if (error > worst) {
                    worst = error;
                    worst_name = name;
                }
            } else {
                CHECK(ours == e.second);
            }
        }
        MESSAGE("the head's block-format tensors against the checkpoint: largest relative error " << worst << " ("
                                                                                                  << worst_name << ")");

        // The values the head multiplies by, from the parameters, as the converter computes them (in double).
        const gguf_import::PackTensor& s = *pack.at("joint_head.scales");
        const std::vector<float> scales = decoded(s.source->view(s.shape));
        REQUIRE(scales.size() == 3);
        const auto scale = [](double x) { return std::exp(std::min(x, std::log(100.0))); };
        CHECK(scales[0] == doctest::Approx(scale(parameters["prior_logit_scale"])).epsilon(1e-6));
        CHECK(scales[1] == doctest::Approx(scale(parameters["joint_logit_scale"])).epsilon(1e-6));
        CHECK(scales[2] == doctest::Approx(1.0 / (1.0 + std::exp(-parameters["residual_gate"]))).epsilon(1e-6));
        MESSAGE("scales " << scales[0] << " " << scales[1] << " " << scales[2]);
    }

    TEST_CASE("the tokenizer of a real clef GGUF is the checkpoint's") {
        const auto real = real_file();
        if (!real) return;
        if (!fs::exists(real->hf / "tokenizer.json")) {
            MESSAGE("XS_CLEF_DIR has no tokenizer.json: the tokenizer is not compared");
            return;
        }
        const TempDir work("xs_gguf_clef_real_tokenizer_test");
        xs::test::write_gguf_tokenizer(gguf::File(real->gguf), work.path);
        const auto imported = tok::Tokenizer::load(work.path);
        const auto original = tok::Tokenizer::load(real->hf);
        CHECK(xs::test::differing_tokens(*imported, *original) == 0);
        for (const char* text :
             {"<|im_start|>system\nRead the complete state and schema.<|im_end|>\n",
              "{\"option_id\":\"true\",\"description\":\"Ünïcödé — 中文 😀\"}", "\nFIELD 1\nID: x\n"}) {
            CAPTURE(text);
            CHECK(imported->encode(text, true) == original->encode(text, true));
        }
    }

    TEST_CASE("a real clef GGUF imports into a pack that loads as a decision model") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_clef_real_import_test");
        gguf_import::ImportOptions o;
        o.source = real->gguf;
        o.output = work.path / "pack";
        const gguf_import::ImportReport report = gguf_import::import_gguf(o);
        CHECK(report.architecture == "clef");
        CHECK(model_kind(o.output) == ModelKind::Decision);
        const auto model = load_decision_model(o.output);
        CHECK(model->spec().arch == "clef");
        CHECK(model->programs().size() == 4);
        // Block format tensors byte for byte: the head's query projection of the first evidence layer, and the
        // output layer the head reads on the host.
        const gguf::File file(real->gguf);
        for (const auto& [gguf_name, name] :
             {std::pair{"dec.blk.0.cross_attn_q.weight", "joint_head.evidence_layers.0.attention.q_proj.weight"},
              std::pair{"output.weight", "lm_head.weight"}}) {
            CAPTURE(name);
            const gguf::TensorInfo* t = file.find_tensor(gguf_name);
            REQUIRE(t != nullptr);
            const TensorView& v = model->weights().get(name);
            CHECK(v.dtype == t->dtype);
            if (t->dtype == DType::Block) CHECK(v.block == t->block);
            CHECK(std::memcmp(v.data, t->data, t->nbytes) == 0);
        }
    }
}
