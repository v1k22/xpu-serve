#pragma once

// A tiny qwen35 GGUF written the way llama.cpp's converter writes one (convert_hf_to_gguf.py, Qwen3_5TextModel), from
// the tiny Qwen3.5 checkpoint (tiny_checkpoints.h) with an MTP head and a byte-level BPE tokenizer of its own: GGUF
// names, every norm weight but linear attention's stored as 1 + w, A_log as ssm_a = -exp(A_log), the conv squeezed,
// linear attention's value heads tiled, the vocabulary padded with [PAD<id>] placeholders and every added token that
// looks like <|...|> typed control. Matrices are Q8_0 and Q4_0 blocks, one F16 and one BF16; layer 0 stores its input
// projections merged (attn_qkv_gate, ssm_beta_alpha), as the converter can.
//
// And its twin: the Hugging Face checkpoint those bytes stand for, every weight as the f32 values the file's blocks
// decode to, in the grouped order, with the norms and A_log computed back the way the import computes them.
//
// The decoder's part (put_tiny_decoder) also writes the decoder of a clef file (tiny_clef_gguf.h).

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/gguf_test_util.h"
#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "model/tiny_checkpoints.h"
#include "quant/quant.h"
#include "tokenizer/byte_level.h"
#include "util/half.h"

namespace xs::test {

// The tiny model a GGUF test imports: 288 token ids (274 used, the rest the converter's placeholders).
inline TinyQwen tiny_gguf_model() {
    TinyQwen m;
    m.vocab = 288;
    return m;
}

// A byte-level BPE tokenizer.json with Qwen3.5's normalizer and pre-tokenizer, a few merges, and added tokens: special
// ones, non-special ones (<think>, <tool_call>), and <|fim_prefix|>, which Qwen does not mark special.
inline nlohmann::json tiny_tokenizer_json() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int b = 0; b < 256; ++b)
        vocab[tok::byte_level::byte_piece(static_cast<uint8_t>(b))] = b;
    const std::vector<std::pair<std::string, std::string>> merges = {
        {"Ġ", "t"}, {"h", "e"}, {"Ġt", "he"}, {"i", "n"}, {"Ġ", "a"}, {"e", "r"}, {"Ġ", "Ġ"}, {"o", "n"}, {"Ċ", "Ċ"}};
    nlohmann::json merge_list = nlohmann::json::array();
    int id = 256;
    for (const auto& [a, b] : merges) {
        vocab[a + b] = id++;
        merge_list.push_back(a + " " + b);
    }
    nlohmann::json added = nlohmann::json::array();
    for (const auto& [text, special] : std::vector<std::pair<std::string, bool>>{{"<|endoftext|>", true},
                                                                                 {"<|im_start|>", true},
                                                                                 {"<|im_end|>", true},
                                                                                 {"<think>", false},
                                                                                 {"</think>", false},
                                                                                 {"<tool_call>", false},
                                                                                 {"</tool_call>", false},
                                                                                 {"<|fim_prefix|>", false},
                                                                                 {"<|vision_pad|>", true}}) {
        added.push_back({{"id", id++},
                         {"content", text},
                         {"single_word", false},
                         {"lstrip", false},
                         {"rstrip", false},
                         {"normalized", false},
                         {"special", special}});
    }
    const nlohmann::json byte_level = {
        {"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", false}, {"use_regex", false}};
    return {{"version", "1.0"},
            {"truncation", nullptr},
            {"padding", nullptr},
            {"added_tokens", added},
            {"normalizer", {{"type", "NFC"}}},
            {"pre_tokenizer",
             {{"type", "Sequence"},
              {"pretokenizers",
               {{{"type", "Split"},
                 {"pattern",
                  {{"Regex",
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| "
                    "?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}}},
                 {"behavior", "Isolated"},
                 {"invert", false}},
                byte_level}}}},
            {"post_processor", byte_level},
            {"decoder", byte_level},
            {"model",
             {{"type", "BPE"},
              {"dropout", nullptr},
              {"unk_token", nullptr},
              {"continuing_subword_prefix", ""},
              {"end_of_word_suffix", ""},
              {"fuse_unk", false},
              {"byte_fallback", false},
              {"ignore_merges", false},
              {"vocab", vocab},
              {"merges", merge_list}}}};
}

inline const char* kTinyChatTemplate =
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n<think>\n{%- endif %}";

// The tokenizer metadata llama.cpp's converter writes for `tokenizer` (get_vocab_base, SpecialVocab): one entry per id
// up to the model's vocab, [PAD<id>] where the tokenizer has none, added tokens control when special or when they look
// like <|...|>, user-defined otherwise.
inline void add_tokenizer_metadata(GgufWriter& w, const nlohmann::json& tokenizer, int64_t vocab, int32_t eos) {
    std::vector<std::string> tokens(static_cast<size_t>(vocab));
    std::vector<int32_t> types(static_cast<size_t>(vocab), 5);
    for (int64_t i = 0; i < vocab; ++i)
        tokens[static_cast<size_t>(i)] = "[PAD" + std::to_string(i) + "]";
    for (const auto& [piece, id] : tokenizer["model"]["vocab"].items()) {
        tokens[id.get<size_t>()] = piece;
        types[id.get<size_t>()] = 1;
    }
    for (const auto& t : tokenizer["added_tokens"]) {
        const std::string text = t["content"];
        const bool looks_special = text.rfind("<|", 0) == 0 && text.size() > 3 && text.substr(text.size() - 2) == "|>";
        tokens[t["id"].get<size_t>()] = text;
        types[t["id"].get<size_t>()] = t["special"].get<bool>() || looks_special ? 3 : 4;
    }
    std::vector<std::string> merges;
    for (const auto& m : tokenizer["model"]["merges"])
        merges.push_back(m.get<std::string>());
    w.string("tokenizer.ggml.model", "gpt2")
        .string("tokenizer.ggml.pre", "qwen35")
        .strings("tokenizer.ggml.tokens", tokens)
        .i32s("tokenizer.ggml.token_type", types)
        .strings("tokenizer.ggml.merges", merges)
        .u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(eos))
        .u32("tokenizer.ggml.padding_token_id", static_cast<uint32_t>(eos - 2))
        .string("tokenizer.chat_template", kTinyChatTemplate);
}

// The values of `v` along `axis` from index `first`, value heads of `unit` entries, moved between the grouped order
// (head k * r + s) and the tiled one (head s * hk + k): to_tiled true writes the converter's order.
inline std::vector<float> retile(const std::vector<float>& v, const Dims& shape, int axis, int64_t first, int64_t unit,
                                 int64_t hk, int64_t hv, bool to_tiled) {
    const int64_t r = hv / hk;
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= shape[static_cast<size_t>(d)];
    for (size_t d = static_cast<size_t>(axis) + 1; d < shape.size(); ++d)
        inner *= shape[d];
    const int64_t n = shape[static_cast<size_t>(axis)];
    std::vector<float> out = v;
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t s = 0; s < r; ++s) {
            for (int64_t k = 0; k < hk; ++k) {
                int64_t tiled = first + (s * hk + k) * unit, grouped = first + (k * r + s) * unit;
                if (!to_tiled) std::swap(tiled, grouped);
                std::copy_n(v.begin() + (o * n + grouped) * inner, unit * inner, out.begin() + (o * n + tiled) * inner);
            }
        }
    }
    return out;
}

enum class Storage { Q8_0, Q4_0, F16, BF16, F32 };

// Writes the tiny GGUF's twin checkpoint while building the file.
class TinyGgufBuilder {
public:
    explicit TinyGgufBuilder(GgufWriter& w) : w_(w) {}

    // A tensor of `shape` (row-major, Hugging Face order) stored as `storage` under `gguf_name` with `gguf_shape`
    // (the stored row-major shape: the same numel), its values in the file's order `stored`; the twin gets `twin_name`
    // with the values the file holds, brought back by `back` (the inverse of what made `stored`).
    template <typename Back>
    void put(const std::string& gguf_name, const Dims& gguf_shape, const std::vector<float>& stored, Storage storage,
             Back back, const std::vector<std::pair<std::string, Dims>>& twins) {
        const std::vector<float> values = back(store(gguf_name, gguf_shape, stored, storage));
        size_t at = 0;
        for (const auto& [name, shape] : twins) {
            const auto n = static_cast<size_t>(numel(shape));
            twin(name, shape,
                 std::vector<float>(values.begin() + static_cast<std::ptrdiff_t>(at),
                                    values.begin() + static_cast<std::ptrdiff_t>(at + n)));
            at += n;
        }
    }

    // The tensor of put() alone: written to the file, and the values the file holds returned.
    std::vector<float> store(const std::string& gguf_name, const Dims& gguf_shape, const std::vector<float>& stored,
                             Storage storage) {
        std::vector<float> held(stored.size());
        std::string bytes;
        uint32_t type = kGgmlF32;
        switch (storage) {
            case Storage::Q8_0:
                bytes = encode_q8_0(stored.data(), static_cast<int64_t>(stored.size()), held.data());
                type = kGgmlQ8_0;
                break;
            case Storage::Q4_0:
                bytes = encode_q4_0(stored.data(), static_cast<int64_t>(stored.size()), held.data());
                type = kGgmlQ4_0;
                break;
            case Storage::F16:
                for (size_t i = 0; i < stored.size(); ++i) {
                    const uint16_t h = f32_to_f16(stored[i]);
                    bytes += pod_bytes(h);
                    held[i] = f16_to_f32(h);
                }
                type = kGgmlF16;
                break;
            case Storage::BF16:
                for (size_t i = 0; i < stored.size(); ++i) {
                    const uint16_t h = f32_to_bf16(stored[i]);
                    bytes += pod_bytes(h);
                    held[i] = bf16_to_f32(h);
                }
                type = kGgmlBF16;
                break;
            case Storage::F32:
                bytes = pod_bytes(stored);
                held = stored;
                break;
        }
        w_.tensor(gguf_name, Dims(gguf_shape.rbegin(), gguf_shape.rend()), type, std::move(bytes));
        return held;
    }

    // A tensor of the twin.
    void twin(std::string name, Dims shape, std::vector<float> values) {
        twin_.emplace_back(std::move(name), std::move(shape), std::move(values));
    }

    // The twin's tensors as f32 safetensors units.
    std::vector<OutUnit> twin_units() const {
        std::vector<OutUnit> units;
        for (const auto& [name, shape, values] : twin_) {
            OutUnit u;
            u.tensors = {OutTensor{name, DType::F32, shape}};
            const auto held = std::make_shared<std::vector<float>>(values);
            u.write = [held](std::ostream& out) {
                out.write(reinterpret_cast<const char*>(held->data()),
                          static_cast<std::streamsize>(held->size() * sizeof(float)));
            };
            units.push_back(std::move(u));
        }
        return units;
    }

private:
    GgufWriter& w_;
    std::vector<std::tuple<std::string, Dims, std::vector<float>>> twin_;
};

// The tiny tokenizer's end of turn, the file's eos_token_id.
inline int32_t tiny_eos() {
    return tiny_tokenizer_json()["added_tokens"][2]["id"];
}

// Writes `hf`: the tiny checkpoint (tiny_gguf_model()), its tokenizer and chat template.
inline void write_tiny_hf(const std::filesystem::path& hf, uint32_t seed) {
    write_tiny_qwen(hf, tiny_gguf_model(), seed);
    std::ofstream(hf / "tokenizer.json") << tiny_tokenizer_json().dump();
    std::ofstream(hf / "tokenizer_config.json") << R"({"eos_token": "<|im_end|>", "bos_token": null})";
    std::ofstream(hf / "chat_template.jinja") << kTinyChatTemplate;
}

// The tiny checkpoint in `checkpoint` as the converter writes a Qwen3.5 decoder into a GGUF of architecture `arch`:
// its keys under "<arch>.", the tokenizer's, and its tensors into `w`, with an MTP head when `mtp`; their twin's into
// `b`.
inline void put_tiny_decoder(GgufWriter& w, TinyGgufBuilder& b, const std::filesystem::path& checkpoint,
                             const std::string& arch, bool mtp, uint32_t seed) {
    const TinyQwen m = tiny_gguf_model();
    const int64_t h = m.hidden, hk = m.lin_k_heads, hv = m.lin_v_heads, dv = m.lin_dim;
    const int64_t key_width = hk * m.lin_dim, value_width = hv * dv, conv_width = 2 * key_width + value_width;
    const auto layers = static_cast<int64_t>(m.layers.size());
    const auto key = [&](const char* rest) { return arch + "." + rest; };
    w.u32(key("block_count"), static_cast<uint32_t>(layers + (mtp ? 1 : 0)))
        .u32(key("context_length"), 4096)
        .u32(key("embedding_length"), static_cast<uint32_t>(h))
        .u32(key("feed_forward_length"), static_cast<uint32_t>(m.ffn))
        .u32(key("attention.head_count"), static_cast<uint32_t>(m.heads))
        .u32(key("attention.head_count_kv"), static_cast<uint32_t>(m.kv_heads))
        .i32s(key("rope.dimension_sections"), {11, 11, 10, 0})
        .f32(key("rope.freq_base"), 1e7f)
        .f32(key("attention.layer_norm_rms_epsilon"), 1e-6f)
        .u32(key("attention.key_length"), static_cast<uint32_t>(m.head_dim))
        .u32(key("attention.value_length"), static_cast<uint32_t>(m.head_dim))
        .u32(key("ssm.conv_kernel"), static_cast<uint32_t>(m.conv))
        .u32(key("ssm.state_size"), static_cast<uint32_t>(m.lin_dim))
        .u32(key("ssm.group_count"), static_cast<uint32_t>(hk))
        .u32(key("ssm.time_step_rank"), static_cast<uint32_t>(hv))
        .u32(key("ssm.inner_size"), static_cast<uint32_t>(value_width))
        .u32(key("full_attention_interval"), 4)
        .u32(key("rope.dimension_count"), static_cast<uint32_t>(m.head_dim / 4));
    if (mtp) w.u32(key("nextn_predict_layers"), 1);
    add_tokenizer_metadata(w, tiny_tokenizer_json(), m.vocab, tiny_eos());

    const auto source = SafetensorsWeights::open_dir(checkpoint);
    const auto values = [&](const std::string& name) {
        const TensorView& v = source->get(name);
        std::vector<float> out(static_cast<size_t>(v.numel()));
        quant::read_rows(v, 0, quant::matrix_rows(v.shape), out.data());
        return out;
    };
    std::mt19937 rng(seed + 1);
    const auto random = [&](const Dims& shape, float range) {
        std::uniform_real_distribution<float> dist(-range, range);
        std::vector<float> out(static_cast<size_t>(numel(shape)));
        for (float& x : out)
            x = bf16_to_f32(f32_to_bf16(dist(rng)));
        return out;
    };
    const auto same = [](const std::vector<float>& v) { return v; };
    const auto unnorm = [](std::vector<float> v) {
        for (float& x : v)
            x -= 1.0f;
        return v;
    };
    const auto plus_one = [](std::vector<float> v) {
        for (float& x : v)
            x += 1.0f;
        return v;
    };
    int matrices = 0;
    const auto next_block = [&] { return matrices++ % 2 == 0 ? Storage::Q8_0 : Storage::Q4_0; };
    // A matrix in the Hugging Face layout, stored without reordering.
    const auto matrix = [&](const std::string& gguf, const std::string& hf, const Dims& shape,
                            const std::vector<float>& v, Storage s) { b.put(gguf, shape, v, s, same, {{hf, shape}}); };
    const auto norm = [&](const std::string& gguf, const std::string& hf, const std::vector<float>& v) {
        const Dims shape{static_cast<int64_t>(v.size())};
        b.put(gguf, shape, plus_one(v), Storage::F32, unnorm, {{hf, shape}});
    };

    matrix("token_embd.weight", "model.embed_tokens.weight", {m.vocab, h}, values("model.embed_tokens.weight"),
           Storage::Q8_0);
    norm("output_norm.weight", "model.norm.weight", values("model.norm.weight"));

    // Value heads: tiled in the file, grouped again in the twin.
    const auto tiled = [&](const std::vector<float>& v, const Dims& shape, int axis, int64_t first, int64_t unit) {
        return retile(v, shape, axis, first, unit, hk, hv, true);
    };
    const auto grouped = [&](const Dims& shape, int axis, int64_t first, int64_t unit) {
        return [=](const std::vector<float>& v) { return retile(v, shape, axis, first, unit, hk, hv, false); };
    };
    for (int64_t i = 0; i < layers; ++i) {
        const std::string B = "blk." + std::to_string(i) + ".", L = "model.layers." + std::to_string(i) + ".";
        norm(B + "attn_norm.weight", L + "input_layernorm.weight", values(L + "input_layernorm.weight"));
        norm(B + "post_attention_norm.weight", L + "post_attention_layernorm.weight",
             values(L + "post_attention_layernorm.weight"));
        matrix(B + "ffn_gate.weight", L + "mlp.gate_proj.weight", {m.ffn, h}, values(L + "mlp.gate_proj.weight"),
               next_block());
        matrix(B + "ffn_up.weight", L + "mlp.up_proj.weight", {m.ffn, h}, values(L + "mlp.up_proj.weight"),
               next_block());
        matrix(B + "ffn_down.weight", L + "mlp.down_proj.weight", {h, m.ffn}, values(L + "mlp.down_proj.weight"),
               i == 1 ? Storage::BF16 : next_block());
        if (m.layers[static_cast<size_t>(i)] == "full_attention") {
            const std::string A = L + "self_attn.";
            matrix(B + "attn_q.weight", A + "q_proj.weight", {2 * m.heads * m.head_dim, h}, values(A + "q_proj.weight"),
                   next_block());
            matrix(B + "attn_k.weight", A + "k_proj.weight", {m.kv_heads * m.head_dim, h}, values(A + "k_proj.weight"),
                   next_block());
            matrix(B + "attn_v.weight", A + "v_proj.weight", {m.kv_heads * m.head_dim, h}, values(A + "v_proj.weight"),
                   next_block());
            matrix(B + "attn_output.weight", A + "o_proj.weight", {h, m.heads * m.head_dim},
                   values(A + "o_proj.weight"), Storage::F16);
            norm(B + "attn_q_norm.weight", A + "q_norm.weight", values(A + "q_norm.weight"));
            norm(B + "attn_k_norm.weight", A + "k_norm.weight", values(A + "k_norm.weight"));
            continue;
        }
        const std::string A = L + "linear_attn.";
        const Dims qkv{conv_width, h}, z{value_width, h}, ab{hv, h};
        const std::vector<float> qkv_t = tiled(values(A + "in_proj_qkv.weight"), qkv, 0, 2 * key_width, dv);
        const std::vector<float> z_t = tiled(values(A + "in_proj_z.weight"), z, 0, 0, dv);
        const std::vector<float> b_t = tiled(values(A + "in_proj_b.weight"), ab, 0, 0, 1);
        const std::vector<float> a_t = tiled(values(A + "in_proj_a.weight"), ab, 0, 0, 1);
        if (i == 0) {  // one tensor each for [qkv; z] and [b; a]
            std::vector<float> qkvz = qkv_t;
            qkvz.insert(qkvz.end(), z_t.begin(), z_t.end());
            b.put(B + "attn_qkv_gate.weight", {conv_width + value_width, h}, qkvz, next_block(),
                  [&](const std::vector<float>& v) {
                      std::vector<float> a = grouped(qkv, 0, 2 * key_width, dv)(
                          std::vector<float>(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(numel(qkv))));
                      const std::vector<float> c = grouped(z, 0, 0, dv)(
                          std::vector<float>(v.begin() + static_cast<std::ptrdiff_t>(numel(qkv)), v.end()));
                      a.insert(a.end(), c.begin(), c.end());
                      return a;
                  },
                  {{A + "in_proj_qkv.weight", qkv}, {A + "in_proj_z.weight", z}});
            std::vector<float> ba = b_t;
            ba.insert(ba.end(), a_t.begin(), a_t.end());
            b.put(B + "ssm_beta_alpha.weight", {2 * hv, h}, ba, Storage::Q8_0, grouped({2, hv, h}, 1, 0, 1),
                  {{A + "in_proj_b.weight", ab}, {A + "in_proj_a.weight", ab}});
        } else {
            b.put(B + "attn_qkv.weight", qkv, qkv_t, next_block(), grouped(qkv, 0, 2 * key_width, dv),
                  {{A + "in_proj_qkv.weight", qkv}});
            b.put(B + "attn_gate.weight", z, z_t, next_block(), grouped(z, 0, 0, dv), {{A + "in_proj_z.weight", z}});
            b.put(B + "ssm_beta.weight", ab, b_t, Storage::Q8_0, grouped(ab, 0, 0, 1), {{A + "in_proj_b.weight", ab}});
            b.put(B + "ssm_alpha.weight", ab, a_t, Storage::Q8_0, grouped(ab, 0, 0, 1), {{A + "in_proj_a.weight", ab}});
        }
        std::vector<float> ssm_a = values(A + "A_log");
        for (float& x : ssm_a)
            x = -std::exp(x);
        b.put(B + "ssm_a", {hv}, tiled(ssm_a, {hv}, 0, 0, 1), Storage::F32,
              [&](const std::vector<float>& v) {
                  std::vector<float> a = grouped({hv}, 0, 0, 1)(v);
                  for (float& x : a)
                      x = std::log(-x);
                  return a;
              },
              {{A + "A_log", {hv}}});
        b.put(B + "ssm_dt.bias", {hv}, tiled(values(A + "dt_bias"), {hv}, 0, 0, 1), Storage::F32,
              grouped({hv}, 0, 0, 1), {{A + "dt_bias", {hv}}});
        b.put(B + "ssm_conv1d.weight", {conv_width, m.conv},
              tiled(values(A + "conv1d.weight"), {conv_width, m.conv}, 0, 2 * key_width, dv), Storage::F32,
              grouped({conv_width, m.conv}, 0, 2 * key_width, dv), {{A + "conv1d.weight", {conv_width, 1, m.conv}}});
        b.put(B + "ssm_norm.weight", {dv}, values(A + "norm.weight"), Storage::F32, same, {{A + "norm.weight", {dv}}});
        b.put(B + "ssm_out.weight", {h, value_width}, tiled(values(A + "out_proj.weight"), {h, value_width}, 1, 0, dv),
              next_block(), grouped({h, value_width}, 1, 0, dv), {{A + "out_proj.weight", {h, value_width}}});
    }

    if (!mtp) return;
    // The MTP head: block `layers`, a full-attention layer with nextn.* beside it.
    const std::string B = "blk." + std::to_string(layers) + ".", M = "mtp.layers.0.";
    const float in = 1.0f / std::sqrt(static_cast<float>(h));
    norm(B + "attn_norm.weight", M + "input_layernorm.weight", random({h}, 0.1f));
    norm(B + "post_attention_norm.weight", M + "post_attention_layernorm.weight", random({h}, 0.1f));
    matrix(B + "attn_q.weight", M + "self_attn.q_proj.weight", {2 * m.heads * m.head_dim, h},
           random({2 * m.heads * m.head_dim, h}, in), next_block());
    matrix(B + "attn_k.weight", M + "self_attn.k_proj.weight", {m.kv_heads * m.head_dim, h},
           random({m.kv_heads * m.head_dim, h}, in), next_block());
    matrix(B + "attn_v.weight", M + "self_attn.v_proj.weight", {m.kv_heads * m.head_dim, h},
           random({m.kv_heads * m.head_dim, h}, in), next_block());
    matrix(B + "attn_output.weight", M + "self_attn.o_proj.weight", {h, m.heads * m.head_dim},
           random({h, m.heads * m.head_dim}, 1.0f / std::sqrt(static_cast<float>(m.heads * m.head_dim))), next_block());
    norm(B + "attn_q_norm.weight", M + "self_attn.q_norm.weight", random({m.head_dim}, 0.1f));
    norm(B + "attn_k_norm.weight", M + "self_attn.k_norm.weight", random({m.head_dim}, 0.1f));
    matrix(B + "ffn_gate.weight", M + "mlp.gate_proj.weight", {m.ffn, h}, random({m.ffn, h}, in), next_block());
    matrix(B + "ffn_up.weight", M + "mlp.up_proj.weight", {m.ffn, h}, random({m.ffn, h}, in), next_block());
    matrix(B + "ffn_down.weight", M + "mlp.down_proj.weight", {h, m.ffn},
           random({h, m.ffn}, 1.0f / std::sqrt(static_cast<float>(m.ffn))), next_block());
    matrix(B + "nextn.eh_proj.weight", "mtp.fc.weight", {h, 2 * h},
           random({h, 2 * h}, 1.0f / std::sqrt(static_cast<float>(2 * h))), next_block());
    norm(B + "nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight", random({h}, 0.1f));
    norm(B + "nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight", random({h}, 0.1f));
    norm(B + "nextn.shared_head_norm.weight", "mtp.norm.weight", random({h}, 0.1f));
}

// Writes `dir`/twin: the twin's tensors in `b`, and hf/'s config (with `mtp_layers` and the file's stop token),
// tokenizer and chat template.
inline void write_tiny_twin(const std::filesystem::path& dir, const TinyGgufBuilder& b, int mtp_layers) {
    namespace fs = std::filesystem;
    fs::create_directories(dir / "twin");
    write_safetensors(dir / "twin" / "model.safetensors", b.twin_units());
    nlohmann::json config = nlohmann::json::parse(std::ifstream(dir / "hf" / "config.json"));
    config["mtp_num_hidden_layers"] = mtp_layers;
    config["eos_token_id"] = tiny_eos();
    std::ofstream(dir / "twin" / "config.json") << config.dump(1);
    for (const char* f : {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}) {
        fs::copy_file(dir / "hf" / f, dir / "twin" / f, fs::copy_options::overwrite_existing);
    }
}

// Writes, under `dir`: hf/ (the tiny checkpoint, its tokenizer and chat template), twin/ (the checkpoint the GGUF
// stands for). Returns the GGUF's writer: the test writes the file, as it is or changed.
inline GgufWriter build_tiny_gguf(const std::filesystem::path& dir, uint32_t seed) {
    write_tiny_hf(dir / "hf", seed);
    GgufWriter w;
    w.string("general.architecture", "qwen35")
        .string("general.type", "model")
        .string("general.name", "Tiny Qwen3.5")
        .u32("general.file_type", 7)
        .u32("general.quantization_version", 2);
    TinyGgufBuilder b(w);
    put_tiny_decoder(w, b, dir / "hf", "qwen35", true, seed);
    write_tiny_twin(dir, b, 1);
    return w;
}

// A pack tensor's values in the twin's order: the import keeps linear attention's value heads tiled.
inline std::vector<float> grouped_values(const std::string& name, const TensorView& v) {
    const TinyQwen m = tiny_gguf_model();
    const int64_t hk = m.lin_k_heads, hv = m.lin_v_heads, d = m.lin_dim;
    std::vector<float> values(static_cast<size_t>(v.numel()));
    quant::read_rows(v, 0, quant::matrix_rows(v.shape), values.data());
    const auto ends_with = [&](std::string_view tail) { return name.ends_with(tail); };
    const auto back = [&](int axis, int64_t first, int64_t unit) {
        return retile(values, v.shape, axis, first, unit, hk, hv, false);
    };
    if (ends_with("in_proj_qkv.weight") || ends_with("conv1d.weight")) return back(0, 2 * hk * d, d);
    if (ends_with("in_proj_z.weight")) return back(0, 0, d);
    if (ends_with("in_proj_a.weight") || ends_with("in_proj_b.weight") || ends_with("A_log") || ends_with("dt_bias")) {
        return back(0, 0, 1);
    }
    if (name.find("linear_attn.out_proj.weight") != std::string::npos) return back(1, 0, d);
    return values;
}

// The tiny GGUF of build_tiny_gguf as unsloth publishes the small files of a model: the model without its MTP head
// (block_count and nextn_predict_layers say so), and the head in a file of its own, llama.cpp's draft model of it.
// That file has the qwen35.* metadata of a file that holds the head and no tokenizer; its tensors are the head's block,
// with the same bytes, and a token_embd, output and output_norm of its own (Q4_0, Q4_0 and F16, where the model's are
// Q8_0, tied to the embedding and F32).
struct TinyMtpFiles {
    GgufWriter model, mtp;
};

inline TinyMtpFiles split_tiny_gguf(const GgufWriter& with_head) {
    const TinyQwen m = tiny_gguf_model();
    const auto layers = static_cast<uint32_t>(m.layers.size());
    const std::string head = "blk." + std::to_string(layers) + ".";
    const auto in_head = [&](const std::string& name) { return name.rfind(head, 0) == 0; };
    TinyMtpFiles out{with_head, with_head};
    out.model.drop_tensors_if(in_head)
        .drop_value("qwen35.block_count")
        .u32("qwen35.block_count", layers)
        .drop_value("qwen35.nextn_predict_layers")
        .u32("qwen35.nextn_predict_layers", 0);
    out.mtp.drop_tensors_if([&](const std::string& name) { return !in_head(name); })
        .drop_values_if([](const std::string& key) { return key.rfind("tokenizer.", 0) == 0 || key == "general.name"; })
        .string("general.name", "Tiny Qwen3.5 MTP");
    const int64_t n = m.vocab * m.hidden;
    std::vector<float> table(static_cast<size_t>(n)), decoded(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        table[static_cast<size_t>(i)] = 0.05f * std::sin(0.71f * static_cast<float>(i));
    std::string norm;
    for (int64_t i = 0; i < m.hidden; ++i)
        norm += pod_bytes(f32_to_f16(1.0f + 0.01f * static_cast<float>(i % 7)));
    out.mtp.tensor("token_embd.weight", {m.hidden, m.vocab}, kGgmlQ4_0, encode_q4_0(table.data(), n, decoded.data()))
        .tensor("output.weight", {m.hidden, m.vocab}, kGgmlQ4_0, encode_q4_0(table.data(), n, decoded.data()))
        .tensor("output_norm.weight", {m.hidden}, kGgmlF16, norm);
    return out;
}

}  // namespace xs::test
