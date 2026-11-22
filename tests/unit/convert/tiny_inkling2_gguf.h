#pragma once

// A tiny Inkling 2 written as either converter writes it (convert/gguf/inkling2.cpp): Unsloth's llama.cpp builds
// ("inkling2": layer visits counted, the loop's message queries as two tensors) and mainline llama.cpp
// ("inkling2_loop": layers counted, each looped layer's four message queries its own). The same model in both: every
// tensor's values come from its checkpoint name, so the two files differ only in how they lay the model out.
//
// Eight layers, the sixth over the whole context, the others within a window; layers 2 to 5 (two blocks of two, so a
// pass hears a block of the one before) looped twice; every tensor F32,
// stored as the converter stores it (a full layer's convolution taps transposed and flipped, the relative profiles
// transposed).

#include <cmath>
#include <string>
#include <vector>

#include "convert/tiny_gguf.h"
#include "io/gguf_test_util.h"

namespace xs::test {

struct TinyInkling2 {
    int64_t hidden = 16, heads = 2, kv_heads = 1, head_dim = 8, rel_width = 4, rel_extent = 6, window = 4, taps = 3;
    int64_t experts = 4, experts_per_token = 2, expert_width = 8, gate_rank = 2, block_layers = 2;
    int64_t layers = 8, loop_start = 2, loop_end = 6, passes = 2;
    std::vector<bool> sliding{true, true, true, true, true, false, true, true};
};

// Deterministic values of the tensor a checkpoint names `name`, in its own order: small, so the model stays finite.
inline std::vector<float> tiny_inkling2_values(const std::string& name, int64_t n, float center, float amplitude) {
    float seed = 0.0f;
    for (const char c : name) seed = std::fmod(seed * 1.37f + static_cast<float>(c), 97.0f);
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = center + amplitude * std::sin(seed + 0.71f * static_cast<float>(i));
    }
    return v;
}

// The tiny model's file in mainline llama.cpp's layout or Unsloth's.
inline GgufWriter tiny_inkling2_gguf(bool mainline, const TinyInkling2& m = {}) {
    const std::string arch = mainline ? "inkling2_loop" : "inkling2";
    const auto k = [&](const std::string& rest) { return arch + "." + rest; };
    const int64_t visits = m.layers + (m.passes - 1) * (m.loop_end - m.loop_start);
    const int64_t entries = mainline ? m.layers : visits;  // of a per-layer array
    const auto layer_of = [&](int64_t e) {                  // the layer an entry describes
        if (mainline || e < m.loop_start) return e;
        const int64_t span = m.loop_end - m.loop_start;
        if (e < m.loop_start + m.passes * span) return m.loop_start + (e - m.loop_start) % span;
        return e - (m.passes - 1) * span;
    };

    GgufWriter w;
    w.string("general.architecture", arch).string("general.name", "Tiny Inkling 2");
    const nlohmann::json tokenizer = tiny_tokenizer_json();
    const auto tokens = static_cast<int64_t>(tokenizer["model"]["vocab"].size() + tokenizer["added_tokens"].size());
    const int64_t vocab = tokens + 4;  // the output layer's rows past the tokens are padding
    add_tokenizer_metadata(w, tokenizer, tokens, 256 + 9 + 2);  // <|im_end|>
    w.drop_value("tokenizer.ggml.pre").string("tokenizer.ggml.pre", "inkling2");
    w.u32(k("block_count"), static_cast<uint32_t>(entries)).u32(k("context_length"), 64);
    w.u32(k("embedding_length"), 16).u32(k("feed_forward_length"), 8).u32(k("expert_feed_forward_length"), 8);
    w.u32(k("expert_count"), 4).u32(k("expert_used_count"), 2).f32(k("expert_weights_scale"), 1.5f);
    w.u32(k("vocab_size"), static_cast<uint32_t>(vocab)).u32(k("vocab_size_unpadded"), static_cast<uint32_t>(tokens));
    w.i32s(k("attention.head_count"), std::vector<int32_t>(static_cast<size_t>(entries), 2));
    w.i32s(k("attention.head_count_kv"), std::vector<int32_t>(static_cast<size_t>(entries), 1));
    for (const char* d : {"attention.key_length", "attention.value_length", "attention.key_length_swa",
                          "attention.value_length_swa"}) {
        w.u32(k(d), 8);
    }
    w.f32(k("attention.layer_norm_rms_epsilon"), 1e-6f).u32(k("attention.sliding_window"), 4);
    std::vector<bool> pattern;
    for (int64_t e = 0; e < entries; ++e) pattern.push_back(m.sliding[static_cast<size_t>(layer_of(e))]);
    w.bools(k("attention.sliding_window_pattern"), pattern);
    w.u32(k("attention.relative_length"), 4).u32(k("attention.relative_extent"), 6);
    w.f32(k("attention.qk_clip"), 3.0f).f32(k("attention.relative_clip"), 2.0f);
    w.u32(k("attention.log_scale_floor"), 8).f32(k("attention.log_scale_alpha"), 0.1f);
    w.u32(k("rope.dimension_count"), 0).u32(k("shortconv.l_cache"), 3).u32(k("attn_res.block_size"), 2);
    w.f32(k("activation.situ_beta"), 8.0f).f32(k("activation.situ_linear_beta"), 6.0f);
    w.f32(k("final_logit_softcapping"), 15.0f);
    const auto u = [](int64_t v) { return static_cast<uint32_t>(v); };
    w.u32(k(mainline ? "loop.count" : "num_loops"), u(m.passes)).u32(k("loop.gate_rank"), u(m.gate_rank));
    w.u32(k(mainline ? "loop.start" : "loop.core_start"), u(m.loop_start));
    w.u32(k(mainline ? "loop.end" : "loop.core_end"), u(m.loop_end));
    if (mainline) w.boolean(k("loop.varying"), false);

    // A tensor stored as `stored` (row-major), in the checkpoint's order, its values from its checkpoint name.
    const auto put = [&](const std::string& file, const Dims& stored, const std::string& name, float center = 0.0f,
                         float amplitude = 0.3f) {
        w.f32_tensor(file, std::vector<int64_t>(stored.rbegin(), stored.rend()),
                     tiny_inkling2_values(name, numel(stored), center, amplitude));
    };
    // The same, its values the checkpoint's [rows, cols] transposed: stored [cols, rows].
    const auto put_transposed = [&](const std::string& file, int64_t rows, int64_t cols, const std::string& name) {
        const std::vector<float> v = tiny_inkling2_values(name, rows * cols, 0.0f, 0.3f);
        std::vector<float> s(v.size());
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) s[static_cast<size_t>(c * rows + r)] = v[static_cast<size_t>(r * cols + c)];
        }
        w.f32_tensor(file, {rows, cols}, s);
    };
    // A full layer's dynamic convolution: the checkpoint's [taps, channels] with the current token's tap first, stored
    // [channels, taps] with it last.
    const auto put_taps = [&](const std::string& file, int64_t channels, const std::string& name) {
        const std::vector<float> v = tiny_inkling2_values(name, m.taps * channels, 0.0f, 0.3f);
        std::vector<float> s(v.size());
        for (int64_t t = 0; t < m.taps; ++t) {
            for (int64_t c = 0; c < channels; ++c) {
                s[static_cast<size_t>(c * m.taps + (m.taps - 1 - t))] = v[static_cast<size_t>(t * channels + c)];
            }
        }
        w.f32_tensor(file, {m.taps, channels}, s);
    };

    const std::string P = "model.llm.";
    const int64_t h = m.hidden, q = m.heads * m.head_dim, kv = m.kv_heads * m.head_dim;
    put("token_embd.weight", {vocab, h}, P + "embed.weight");
    put("token_embd_norm.weight", {h}, P + "embed_norm.weight", 1.0f, 0.1f);
    put("output.weight", {vocab, h}, P + "unembed.weight");
    put("output_norm.weight", {h}, P + "norm.weight", 1.0f, 0.1f);
    put("output_res_query.weight", {h}, P + "attn_res.output_query");
    // The loop's message queries: as the checkpoint keeps them, rows (query, mlp) and (key, value) per looped layer.
    const int64_t looped = m.loop_end - m.loop_start;
    const std::vector<float> qm = tiny_inkling2_values(P + "middle_cycle_cross_pass_queries", 2 * looped * h, 0, 0.3f);
    const std::vector<float> kvm =
        tiny_inkling2_values(P + "middle_cycle_cross_pass_kv_queries", 2 * looped * h, 0, 0.3f);
    if (!mainline) {
        w.f32_tensor("loop_res_query.weight", {h, 2 * looped}, qm);
        w.f32_tensor("loop_kv_res_query.weight", {h, 2 * looped}, kvm);
    }
    const std::vector<std::string> gates =
        mainline ? std::vector<std::string>{"q_ctrl_", "k_ctrl_", "v_ctrl_", "ffn_ctrl_"}
                 : std::vector<std::string>{"attn_", "k_", "v_", "mlp_"};
    const std::vector<std::string> checkpoint_gates = {"attn", "k", "v", "mlp"};
    for (int64_t i = 0; i < m.layers; ++i) {
        const bool sliding = m.sliding[static_cast<size_t>(i)];
        const std::string B = "blk." + std::to_string(i) + ".";
        const std::string L = P + "layers." + std::to_string(i) + ".";
        const std::string A = L + "attn.";
        put(B + "attn_q.weight", {q, h}, A + "wq_du.weight");
        put(B + "attn_k.weight", {kv, h}, A + "wk_dv.weight");
        put(B + "attn_v.weight", {kv, h}, A + "wv_dv.weight");
        put(B + "attn_output.weight", {h, q}, A + "wo_ud.weight");
        put(B + "attn_gate.weight", {q, h}, A + "gate_attn.weight");
        put(B + "attn_rel.weight", {m.heads * m.rel_width, h}, A + "wr_du.weight");
        put(B + "attn_sink_proj.weight", {m.heads, h}, A + "attn_sink_proj.weight");
        put(B + "attn_q_clip.weight", {m.heads, m.head_dim}, A + "q_l2_clip.scale", 1.0f, 0.1f);
        put(B + "attn_rel_clip.weight", {m.heads, m.rel_width}, A + "rel_l2_clip.scale", 1.0f, 0.1f);
        put_transposed(B + "attn_rel_bias.weight", m.rel_width, sliding ? m.window : m.rel_extent,
                       A + "rel_logits_proj.proj");
        for (const std::string which : {"k", "v"}) {
            if (sliding) {
                put(B + "attn_" + which + "_conv.weight", {kv, m.taps}, A + which + "_sconv.weight");
            } else {
                put_taps(B + "attn_" + which + "_conv.weight", kv, A + which + "_sconv.static_w");
                put(B + "attn_" + which + "_conv_proj.weight", {m.kv_heads * m.taps, h},
                    A + which + "_sconv.weight_proj");
            }
        }
        if (sliding) {
            put(B + "attn_norm.weight", {h}, L + "attn_norm.weight", 1.0f, 0.1f);
        } else {
            for (const std::string which : {"q", "k", "v"}) {
                put(B + "attn_" + which + "_inp_norm.weight", {h}, L + which + "_input_norm.weight", 1.0f, 0.1f);
            }
        }
        if (i > 0) {
            const std::string name = P + "attn_res.attn_queries." + std::to_string(i - 1);
            put(B + "attn_res_query.weight", sliding ? Dims{h} : Dims{3, h}, name);
        }
        put(B + "attn_post_conv.weight", {h, m.taps}, L + "attn_sconv.weight");
        put(B + "ffn_post_conv.weight", {h, m.taps}, L + "mlp_sconv.weight");
        put(B + "ffn_res_query.weight", {h}, P + "attn_res.mlp_queries." + std::to_string(i));
        put(B + "ffn_norm.weight", {h}, L + "mlp_norm.weight", 1.0f, 0.1f);
        put(B + "ffn_gate_inp.weight", {m.experts, h}, L + "mlp.gate.weight");
        put(B + "exp_probs_b.bias", {m.experts}, L + "mlp.gate.bias", 0.0f, 0.05f);
        put(B + "ffn_gate_up_exps.weight", {m.experts, 2 * m.expert_width, h}, L + "mlp.experts.w13_weight");
        put(B + "ffn_down_exps.weight", {m.experts, h, m.expert_width}, L + "mlp.experts.w2_weight");
        if (i < m.loop_start || i >= m.loop_end) continue;
        for (size_t g = 0; g < gates.size(); ++g) {
            const std::string C = L + "controllers." + checkpoint_gates[g] + ".";
            put(B + "loop_" + gates[g] + "down.weight", {m.gate_rank, 2 * h}, C + "down_weight");
            put(B + "loop_" + gates[g] + "forget.weight", {h, m.gate_rank}, C + "forget_weight");
            put(B + "loop_" + gates[g] + "gain.weight", {h, m.gate_rank}, C + "gain_weight");
        }
        if (mainline) {  // the layer's queries of its query, key, value and MLP messages
            const int64_t j = i - m.loop_start;
            std::vector<float> four;
            for (const auto& [rows, r] : {std::pair{&qm, 0}, {&kvm, 0}, {&kvm, 1}, {&qm, 1}}) {
                four.insert(four.end(), rows->begin() + (2 * j + r) * h, rows->begin() + (2 * j + r + 1) * h);
            }
            w.f32_tensor(B + "loop_msg_query.weight", {h, 4}, four);
        }
    }
    return w;
}

}  // namespace xs::test
