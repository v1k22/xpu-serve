#pragma once

// A tiny clef GGUF written the way llama.cpp's converter writes one (conversion/clef.py, ClefModel), from a tiny Clef
// checkpoint: tiny_gguf.h's decoder with its keys under clef.* and no MTP head, and a joint schema head beside it
// (decision_test_util.h). The head's blocks are dec.blk.<i>, the evidence layers first; each attention's in_proj is
// three tensors (q, k, v); the three scalars are decision.scales, the values the head multiplies by, computed in double
// as the converter computes them; matrices are Q8_0, Q4_0, BF16 and F16, norms, biases and the type embedding F32, and
// decision.scorer_out.weight is stored 1-D, as llama-quantize writes it.
//
// And its twin: the Clef checkpoint those bytes stand for, in the original layout (joint_head.safetensors with each
// attention's in_proj whole and the three parameters).

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "convert/tiny_gguf.h"
#include "model/decision_test_util.h"

namespace xs::test {

// How the tiny file departs from what the converter writes: a layout the converter also writes, and planted bugs.
struct TinyClefVariant {
    bool scorer_out_2d = false;   // decision.scorer_out.weight as [1, width], as the converter itself writes it
    bool swap_query_key = false;  // planted: every attention's query and key projections exchanged
    bool raw_scales = false;      // planted: decision.scales holding the parameters, not the values made of them
};

// Writes, under `dir`: hf/ (the tiny Clef checkpoint, its tokenizer and chat template), twin/ (the checkpoint the GGUF
// stands for). Returns the GGUF's writer.
inline GgufWriter build_tiny_clef_gguf(const std::filesystem::path& dir, uint32_t seed,
                                       const TinyClefVariant& variant = {}) {
    namespace fs = std::filesystem;
    const fs::path hf = dir / "hf";
    write_tiny_hf(hf, seed);
    write_tiny_clef_head(hf, tiny_gguf_model().hidden, seed + 1);
    const nlohmann::json config = nlohmann::json::parse(std::ifstream(hf / "joint_head_config.json"));
    const int64_t width = config["width"], routing = config["routing_layers"], layers = config["layers"];

    GgufWriter w;
    w.string("general.architecture", "clef")
        .string("general.type", "model")
        .string("general.name", "Tiny Clef")
        .u32("general.file_type", 7)
        .u32("general.quantization_version", 2);
    TinyGgufBuilder decoder(w), head(w);
    put_tiny_decoder(w, decoder, hf, "clef", false, seed);
    w.string("clef.decision.type", "clef")
        .u32("clef.decision.routing_block_count", static_cast<uint32_t>(routing))
        .u32("clef.decision.block_count", static_cast<uint32_t>(layers))
        .u32("clef.decision.head_count", config["heads"].get<uint32_t>())
        .f32("clef.attention.layer_norm_epsilon", 1e-5f)
        .string("tokenizer.chat_template.systemone", "{{ sep }}{{ state }}{{ sep }}")
        .strings("tokenizer.chat_templates", {"systemone"});

    const auto source = SafetensorsWeights::open_file(hf / "joint_head.safetensors");
    const auto values = [&](const std::string& name) {
        const TensorView& v = source->get(name);
        std::vector<float> out(static_cast<size_t>(v.numel()));
        quant::read_rows(v, 0, quant::matrix_rows(v.shape), out.data());
        return out;
    };
    const auto same = [](const std::vector<float>& v) { return v; };
    int matrices = 0;
    const auto next_format = [&] {
        const Storage formats[] = {Storage::Q8_0, Storage::Q4_0, Storage::BF16, Storage::F16};
        return formats[matrices++ % 4];
    };
    const auto put = [&](const std::string& gguf, const std::string& name, Storage s) {
        const Dims shape = source->get(name).shape;
        head.put(gguf, shape, values(name), s, same, {{name, shape}});
    };
    const auto norm = [&](const std::string& gguf, const std::string& stem) {
        put(gguf + ".weight", stem + ".weight", Storage::F32);
        put(gguf + ".bias", stem + ".bias", Storage::F32);
    };
    const auto linear = [&](const std::string& gguf, const std::string& stem) {
        put(gguf + ".weight", stem + ".weight", next_format());
        put(gguf + ".bias", stem + ".bias", Storage::F32);
    };
    // torch.nn.MultiheadAttention `stem` as the `kind` attention of block `b`: in_proj cut in three by rows (query,
    // key, value), as the converter's chunk(3) cuts it; the twin keeps it whole.
    const auto attention = [&](const std::string& b, const std::string& kind, const std::string& stem) {
        for (const std::string part : {"weight", "bias"}) {
            const std::string name = stem + ".in_proj_" + part;
            const std::vector<float> all = values(name);
            Dims shape = source->get(name).shape;
            shape[0] /= 3;
            const auto third = static_cast<std::ptrdiff_t>(all.size() / 3);
            const Storage s = part == "weight" ? next_format() : Storage::F32;
            std::vector<float> whole;
            for (int i = 0; i < 3; ++i) {
                const int as = variant.swap_query_key && i < 2 ? 1 - i : i;
                const std::vector<float> piece(all.begin() + i * third, all.begin() + (i + 1) * third);
                const std::vector<float> held = head.store(b + kind + "_" + "qkv"[as] + "." + part, shape, piece, s);
                whole.insert(whole.end(), held.begin(), held.end());
            }
            head.twin(name, source->get(name).shape, std::move(whole));
        }
        linear(b + kind + "_o", stem + ".out_proj");
    };

    for (int64_t i = 0; i < routing; ++i) {
        const std::string b = "dec.blk." + std::to_string(i) + ".", l = "evidence_layers." + std::to_string(i);
        norm(b + "cross_attn_norm", l + ".query_norm");
        norm(b + "cross_attn_norm_kv", l + ".memory_norm");
        attention(b, "cross_attn", l + ".attention");
        norm(b + "ffn_norm", l + ".feedforward_norm");
        linear(b + "ffn_up", l + ".feedforward.0");
        linear(b + "ffn_down", l + ".feedforward.3");
    }
    for (int64_t j = 0; j < layers; ++j) {
        const std::string b = "dec.blk." + std::to_string(routing + j) + ".", l = "layers." + std::to_string(j);
        norm(b + "attn_norm", l + ".norm1");
        attention(b, "attn", l + ".self_attn");
        norm(b + "cross_attn_norm", l + ".norm2");
        attention(b, "cross_attn", l + ".multihead_attn");
        norm(b + "ffn_norm", l + ".norm3");
        linear(b + "ffn_up", l + ".linear1");
        linear(b + "ffn_down", l + ".linear2");
    }
    for (const char* n : {"hidden_norm", "option_summary_norm", "field_norm", "option_norm"})
        norm(std::string("decision.") + n, n);
    for (const char* p : {"memory", "question", "option_question", "global", "option_context", "option_lexical"}) {
        put(std::string("decision.proj_") + p + ".weight", std::string(p) + "_projection.weight", next_format());
    }
    linear("decision.scorer", "residual_scorer.0");
    head.put("decision.scorer_out.weight", variant.scorer_out_2d ? Dims{1, width} : Dims{width},
             values("residual_scorer.3.weight"), Storage::BF16, same, {{"residual_scorer.3.weight", {1, width}}});
    put("decision.scorer_out.bias", "residual_scorer.3.bias", Storage::F32);
    put("token_types.weight", "type_embedding.weight", Storage::F32);

    const char* params[] = {"prior_logit_scale", "joint_logit_scale", "residual_gate"};
    std::vector<float> scales;
    for (int i = 0; i < 3; ++i) {
        const double x = values(params[i])[0];
        const double value = i == 2 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(std::min(x, std::log(100.0)));
        scales.push_back(static_cast<float>(variant.raw_scales ? x : value));
        head.twin(params[i], {1}, {static_cast<float>(x)});
    }
    head.store("decision.scales", {3}, scales, Storage::F32);

    write_tiny_twin(dir, decoder, 0);
    write_safetensors(dir / "twin" / "joint_head.safetensors", head.twin_units());
    fs::copy_file(hf / "joint_head_config.json", dir / "twin" / "joint_head_config.json",
                  fs::copy_options::overwrite_existing);
    return w;
}

}  // namespace xs::test
