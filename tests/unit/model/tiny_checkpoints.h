#pragma once

// Tiny random checkpoints written to disk for tests that need a model directory without downloading one: a Qwen3.5
// target (dense or mixture of experts) and block drafters for it (DFlash, DFlash 2, DSpine), in the released
// checkpoints' layout (config keys and tensor names).

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#include "io/safetensors_writer.h"
#include "tensor/dtype.h"
#include "util/half.h"

namespace xs::test {

namespace fs = std::filesystem;

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name) : path(fs::temp_directory_path() / name) { fs::remove_all(path); }
    ~TempDir() { fs::remove_all(path); }
};

// A tensor of uniform values in [lo, hi], or an i64 scalar holding `lo` (shape {}).
struct TinyTensor {
    std::string name;
    Dims shape;
    float lo = 0.0f;
    float hi = 0.0f;
};

// Writes `tensors` (bf16; scalars i64) to `path`, drawing them in order from one generator.
inline void write_random(const fs::path& path, const std::vector<TinyTensor>& tensors, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::vector<char>> data;
    data.reserve(tensors.size());  // the units below keep references into it
    std::vector<OutUnit> units;
    for (const TinyTensor& t : tensors) {
        std::vector<char> bytes;
        if (t.shape.empty()) {
            const auto value = static_cast<int64_t>(t.lo);
            bytes.assign(reinterpret_cast<const char*>(&value), reinterpret_cast<const char*>(&value) + sizeof(value));
        } else {
            std::uniform_real_distribution<float> dist(t.lo, t.hi);
            bytes.resize(static_cast<size_t>(numel(t.shape)) * 2);
            for (size_t i = 0; i < bytes.size(); i += 2) {
                const uint16_t v = f32_to_bf16(dist(rng));
                std::memcpy(bytes.data() + i, &v, 2);
            }
        }
        data.push_back(std::move(bytes));
        const std::vector<char>& held = data.back();
        OutUnit u;
        u.tensors = {OutTensor{t.name, t.shape.empty() ? DType::I64 : DType::BF16, t.shape}};
        u.write = [&held](std::ostream& out) { out.write(held.data(), static_cast<std::streamsize>(held.size())); };
        units.push_back(std::move(u));
    }
    write_safetensors(path, units);
}

struct TinyQwen {
    int64_t vocab = 256, hidden = 128, ffn = 256, heads = 4, kv_heads = 2, head_dim = 64;
    int64_t lin_k_heads = 2, lin_v_heads = 4, lin_dim = 32, conv = 4;
    std::vector<std::string> layers{"linear_attention", "linear_attention", "linear_attention", "full_attention"};
    int64_t experts = 0, top_k = 2, moe_ffn = 64, shared_ffn = 96;  // experts > 0: Qwen3.5 MoE (fused banks)
};

// Writes config.json and model.safetensors (bf16, random) for `m` into `dir`.
inline void write_tiny_qwen(const fs::path& dir, const TinyQwen& m, uint32_t seed) {
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

    // 1/sqrt(fan-in) for matrices keeps activations O(1); small values for norms (zero-centered) and the rest. The
    // router is larger, so each token's top-k experts win clearly (a near tie may route differently at f16).
    std::vector<TinyTensor> specs;
    const auto add = [&](const std::string& name, Dims shape) {
        const bool router = name.size() > 15 && name.compare(name.size() - 15, 15, "mlp.gate.weight") == 0;
        const float range = shape.size() >= 2 ? (router ? 8.0f : 1.0f) / std::sqrt(static_cast<float>(shape.back()))
                                              : 0.1f;
        specs.push_back(TinyTensor{"model." + name, std::move(shape), -range, range});
    };
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
    write_random(dir / "model.safetensors", specs, seed);
}

// A block drafter for a TinyQwen target. Its window is shorter than a test's prompt and it reads the target's last
// layer (whose state is the normed one), so both are exercised.
struct TinyDrafter {
    std::string kind = "dspine";  // dflash | dflash2 | dspine
    int64_t layers = 2, ffn = 256, heads = 4, kv_heads = 2, head_dim = 32;
    int64_t block = 6, window = 12, top_k = 4;
    int64_t code = 32, rank = 16;                       // dspine: the code bridge
    int64_t conv_group = 16, taps = 2, selector = 32;   // dflash2
};

inline void write_tiny_drafter(const fs::path& dir, const TinyDrafter& d, const TinyQwen& target, uint32_t seed) {
    fs::create_directories(dir);
    const bool sliding = d.kind != "dflash";
    const auto layers = static_cast<int64_t>(target.layers.size());
    const std::vector<int64_t> ids{0, layers / 2, layers - 1};
    nlohmann::json config = {
        {"architectures", {d.kind == "dflash2" ? "DFlash2DraftModel" : "DFlashDraftModel"}},
        {"model_type", "qwen3"}, {"vocab_size", target.vocab}, {"hidden_size", target.hidden},
        {"intermediate_size", d.ffn}, {"num_hidden_layers", d.layers}, {"num_attention_heads", d.heads},
        {"num_key_value_heads", d.kv_heads}, {"head_dim", d.head_dim}, {"hidden_act", "silu"},
        {"attention_bias", false}, {"rms_norm_eps", 1e-6}, {"num_target_layers", layers},
        {"layer_types", std::vector<std::string>(static_cast<size_t>(d.layers),
                                                 sliding ? "sliding_attention" : "full_attention")},
        {"sliding_window", sliding ? nlohmann::json(d.window) : nlohmann::json(nullptr)},
        {"rope_parameters", {{"rope_type", "default"}, {"rope_theta", 1e7}}}};
    nlohmann::json dflash = {{"mask_token_id", target.vocab - 1}, {"target_layer_ids", ids}};
    if (d.kind == "dflash") {
        config["block_size"] = d.block;
    } else if (d.kind == "dflash2") {
        config["is_causal"] = false;
        dflash.update({{"block_size", d.block}, {"conv_group_size", d.conv_group}, {"conv_kernel_size", d.taps},
                       {"selector_rank", d.selector}, {"selector_top_k", d.top_k}});
    } else {
        std::vector<int64_t> every;
        for (int64_t l = 1; l <= d.layers; ++l) every.push_back(l);
        config["is_causal"] = true;
        config["block_size"] = d.block;
        config["code_bridge_second_pass"] = {{"first_exact", d.layers - 1}, {"head_topk", d.top_k}, {"mode", "serial_table"}};
        dflash.update({{"draft_block_causal", true},
                       {"selected_prefix_code_bridge", {{"code_size", d.code}, {"eps", 1e-6}, {"kappa", 10.0}, {"rank", d.rank}}},
                       {"selected_prefix_decode_mode", "serial"},
                       {"selected_prefix_feature_layers", every},
                       {"selected_prefix_feature_write_mode", "independent"},
                       {"selected_prefix_feature_writes", true},
                       {"selected_prefix_last_write_refine", {{"head_topk", d.top_k}, {"kind", "exact_sender"}, {"version", 1}, {"weight", 1.0}}},
                       {"selected_prefix_terminal", false}});
    }
    config["dflash_config"] = dflash;
    std::ofstream(dir / "config.json") << config.dump(1);

    // Matrices at 1/sqrt(fan-in), norms around 1 (a skipped norm shows), bridge and selector terms as large as the
    // states they change.
    const int64_t h = target.hidden;
    std::vector<TinyTensor> specs;
    const auto matrix = [&](const std::string& name, Dims shape, float gain = 1.0f) {
        const float range = gain / std::sqrt(static_cast<float>(shape.back()));
        specs.push_back(TinyTensor{name, std::move(shape), -range, range});
    };
    const auto norm = [&](const std::string& name, int64_t width) { specs.push_back(TinyTensor{name, {width}, 0.5f, 1.5f}); };
    matrix("fc.weight", {h, static_cast<int64_t>(ids.size()) * h});
    norm("hidden_norm.weight", h);
    norm("norm.weight", h);
    for (int64_t l = 0; l < d.layers; ++l) {
        const std::string L = "layers." + std::to_string(l) + ".";
        norm(L + "input_layernorm.weight", h);
        norm(L + "post_attention_layernorm.weight", h);
        matrix(L + "self_attn.q_proj.weight", {d.heads * d.head_dim, h});
        matrix(L + "self_attn.k_proj.weight", {d.kv_heads * d.head_dim, h});
        matrix(L + "self_attn.v_proj.weight", {d.kv_heads * d.head_dim, h});
        matrix(L + "self_attn.o_proj.weight", {h, d.heads * d.head_dim});
        norm(L + "self_attn.q_norm.weight", d.head_dim);
        norm(L + "self_attn.k_norm.weight", d.head_dim);
        matrix(L + "mlp.gate_proj.weight", {d.ffn, h});
        matrix(L + "mlp.up_proj.weight", {d.ffn, h});
        matrix(L + "mlp.down_proj.weight", {h, d.ffn});
        if (d.kind == "dflash2") {
            for (const char* conv : {"attention_conv.", "mlp_conv."}) {
                specs.push_back(TinyTensor{L + conv + "base_kernel", {2, d.taps, h}, -0.5f, 1.0f});
                matrix(L + conv + "kernel_projection.weight", {2 * d.taps * (h / d.conv_group), h});
            }
        }
    }
    if (d.kind == "dflash2") {
        matrix("candidate_selector.hidden_projection.weight", {d.selector, h}, 3.0f);
        specs.push_back(TinyTensor{"candidate_selector.predecessor_codebook", {target.vocab, d.selector}, -0.5f, 0.5f});
        specs.push_back(TinyTensor{"candidate_selector.successor_codebook", {target.vocab, d.selector}, -0.5f, 0.5f});
    } else if (d.kind == "dspine") {
        const std::string b = "selected_prefix_bridge.";
        for (const char* version : {"draft_block_causal_version", "last_write_refine_version"}) {
            specs.push_back(TinyTensor{version, {}, 1.0f});
        }
        specs.push_back(TinyTensor{b + "independent_write_version", {}, 1.0f});
        specs.push_back(TinyTensor{b + "terminal_disabled_version", {}, 1.0f});
        const float unit = 1.7f / std::sqrt(static_cast<float>(d.code));  // rows of about unit length
        specs.push_back(TinyTensor{b + "codebook", {target.vocab, d.code}, -unit, unit});
        matrix(b + "early_read_weights", {d.layers - 1, d.code, h}, 2.0f);
        matrix(b + "read_weight", {d.code, h}, 2.0f);
        matrix(b + "feature_write_weights", {d.layers, h, d.rank}, 1.0f);
        matrix(b + "gate.weight", {d.rank, d.code}, 1.5f);
        specs.push_back(TinyTensor{b + "gate.bias", {d.rank}, -0.5f, 0.5f});
        matrix(b + "predecessor_gate.weight", {d.rank, d.code}, 1.5f);
        matrix(b + "prev.weight", {d.rank, d.code}, 1.5f);
        specs.push_back(TinyTensor{b + "out.weight", {d.code, d.rank}, 0.0f, 0.0f});
    }
    write_random(dir / "model.safetensors", specs, seed);
}

}  // namespace xs::test
