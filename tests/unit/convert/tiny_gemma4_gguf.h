#pragma once

// A tiny gemma4 GGUF written the way llama.cpp's converter writes one (conversion/gemma.py, Gemma4Model), from the
// tiny Gemma 4 checkpoint (tiny_checkpoints.h: per-layer embeddings, KV-shared layers with a double-wide MLP, sliding
// and full-attention layers of their own head sizes and KV heads) and a Gemma-style tokenizer of its own: GGUF names,
// every value as the checkpoint has it, rope_freqs computed from the config, head_count_kv and feed_forward_length per
// layer. Matrices are Q8_0 and Q4_0 blocks, one F16 and one BF16, the per-layer gates F32; layer 0 stores q, k and v
// as one tensor (attn_qkv). Token types as the converter gives them: byte pieces byte, the markers a chat parser
// reads user-defined, <eos> normal (the tokenizer_config.json names <turn|> as eos_token), the other added tokens
// control; every score -1000.
//
// And its twin: the Hugging Face checkpoint those bytes stand for, every weight as the f32 values the file's blocks
// decode to. The same for a tiny assistant drafting for it (gemma4-assistant).

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "convert/tiny_gguf.h"
#include "io/gguf_test_util.h"
#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "model/tiny_checkpoints.h"
#include "quant/quant.h"

namespace xs::test {

// The added tokens, every one special, ids 0 to 14; the user-defined ones are the converter's "visible" markers.
inline const std::vector<std::string> kTinyGemmaAdded = {"<pad>",
                                                         "<eos>",
                                                         "<bos>",
                                                         "<unk>",
                                                         "<mask>",
                                                         "<|turn>",
                                                         "<turn|>",
                                                         "<|tool_call>",
                                                         "<tool_call|>",
                                                         "<|tool_response>",
                                                         "<tool_response|>",
                                                         "<|\"|>",
                                                         "<|channel>",
                                                         "<channel|>",
                                                         "<|think|>"};
inline const std::vector<std::string> kTinyGemmaVisible = {
    "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|\"|>", "<|channel>", "<channel|>"};
inline constexpr int32_t kTinyGemmaEos = 6;  // <turn|>

// Gemma 4's tokenizer.json, in miniature: the added tokens (also pieces of the vocabulary), byte fallback pieces,
// characters and merges, among them merges that spell the marker <|"|>.
inline nlohmann::json tiny_gemma4_tokenizer_json() {
    nlohmann::json vocab = nlohmann::json::object();
    nlohmann::json added = nlohmann::json::array();
    int id = 0;
    for (const std::string& text : kTinyGemmaAdded) {
        added.push_back({{"id", id},
                         {"content", text},
                         {"single_word", false},
                         {"lstrip", false},
                         {"rstrip", false},
                         {"normalized", false},
                         {"special", true}});
        vocab[text] = id++;
    }
    for (int b = 0; b < 256; ++b) {
        char piece[8];
        std::snprintf(piece, sizeof(piece), "<0x%02X>", b);
        vocab[piece] = id++;
    }
    for (const char* c : {"▁", "\n", "t", "h", "e", "a", "i", "n", "r", "o", "s", "<", "|", "\"", ">"})
        vocab[c] = id++;
    const std::vector<std::pair<std::string, std::string>> merges = {
        {"▁", "t"},   {"h", "e"}, {"▁t", "he"}, {"i", "n"},   {"▁", "a"},    {"e", "r"},    {"o", "n"},
        {"\n", "\n"}, {"▁", "▁"}, {"<", "|"},   {"<|", "\""}, {"<|\"", "|"}, {"<|\"|", ">"}};
    nlohmann::json merge_list = nlohmann::json::array();
    for (const auto& [a, b] : merges) {
        if (!vocab.contains(a + b)) vocab[a + b] = id++;
        merge_list.push_back({a, b});
    }
    return {
        {"version", "1.0"},
        {"truncation", nullptr},
        {"padding", nullptr},
        {"added_tokens", added},
        {"normalizer", {{"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "▁"}}},
        {"pre_tokenizer",
         {{"type", "Split"}, {"pattern", {{"String", " "}}}, {"behavior", "MergedWithPrevious"}, {"invert", false}}},
        {"post_processor",
         {{"type", "TemplateProcessing"},
          {"single", {{{"Sequence", {{"id", "A"}, {"type_id", 0}}}}}},
          {"pair", {{{"Sequence", {{"id", "A"}, {"type_id", 0}}}}, {{"Sequence", {{"id", "B"}, {"type_id", 1}}}}}},
          {"special_tokens", nlohmann::json::object()}}},
        {"decoder",
         {{"type", "Sequence"},
          {"decoders",
           {{{"type", "Replace"}, {"pattern", {{"String", "▁"}}}, {"content", " "}},
            {{"type", "ByteFallback"}},
            {{"type", "Fuse"}}}}}},
        {"model",
         {{"type", "BPE"},
          {"dropout", nullptr},
          {"unk_token", "<unk>"},
          {"continuing_subword_prefix", nullptr},
          {"end_of_word_suffix", nullptr},
          {"fuse_unk", true},
          {"byte_fallback", true},
          {"ignore_merges", false},
          {"vocab", vocab},
          {"merges", merge_list}}}};
}

inline const char* kTinyGemmaChatTemplate =
    "{{ bos_token }}{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n{% endif %}";

// The tokenizer metadata Gemma4Model.set_vocab writes for `tokenizer` (LlamaHfVocab, SpecialVocab).
inline void add_gemma4_tokenizer_metadata(GgufWriter& w, const nlohmann::json& tokenizer) {
    const auto n = tokenizer["model"]["vocab"].size();
    std::vector<std::string> tokens(n);
    std::vector<int32_t> types(n, 1);
    for (const auto& [piece, id] : tokenizer["model"]["vocab"].items()) {
        tokens[id.get<size_t>()] = piece;
        if (piece.size() == 6 && piece.rfind("<0x", 0) == 0) types[id.get<size_t>()] = 6;
    }
    for (const auto& t : tokenizer["added_tokens"]) {
        const std::string text = t["content"];
        const bool visible =
            std::find(kTinyGemmaVisible.begin(), kTinyGemmaVisible.end(), text) != kTinyGemmaVisible.end();
        types[t["id"].get<size_t>()] = visible ? 4 : text == "<eos>" ? 1 : 3;
    }
    std::vector<std::string> merges;
    for (const auto& m : tokenizer["model"]["merges"])
        merges.push_back(m[0].get<std::string>() + " " + m[1].get<std::string>());
    w.string("tokenizer.ggml.model", "gemma4")
        .strings("tokenizer.ggml.tokens", tokens)
        .f32s("tokenizer.ggml.scores", std::vector<float>(n, -1000.0f))
        .i32s("tokenizer.ggml.token_type", types)
        .strings("tokenizer.ggml.merges", merges)
        .u32("tokenizer.ggml.bos_token_id", 2)
        .u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(kTinyGemmaEos))
        .u32("tokenizer.ggml.unknown_token_id", 3)
        .u32("tokenizer.ggml.padding_token_id", 0)
        .u32("tokenizer.ggml.mask_token_id", 4)
        .boolean("tokenizer.ggml.add_space_prefix", false)
        .boolean("tokenizer.ggml.add_bos_token", true)
        .string("tokenizer.chat_template", kTinyGemmaChatTemplate);
}

inline TinyGemma4 tiny_gemma4_gguf_model() {
    TinyGemma4 m;
    m.vocab = static_cast<int64_t>(tiny_gemma4_tokenizer_json()["model"]["vocab"].size());
    return m;
}

// Writes, under `dir`: hf/ (the tiny checkpoint, its tokenizer and chat template), twin/ (the checkpoint the GGUF
// stands for). Returns the GGUF's writer: the test writes the file, as it is or changed.
inline GgufWriter build_tiny_gemma4_gguf(const std::filesystem::path& dir, uint32_t seed) {
    namespace fs = std::filesystem;
    const TinyGemma4 m = tiny_gemma4_gguf_model();
    write_tiny_gemma4(dir / "hf", m, seed);
    const nlohmann::json tokenizer = tiny_gemma4_tokenizer_json();
    std::ofstream(dir / "hf" / "tokenizer.json") << tokenizer.dump();
    std::ofstream(dir / "hf" / "tokenizer_config.json")
        << R"({"bos_token": "<bos>", "eos_token": "<turn|>", "unk_token": "<unk>", "pad_token": "<pad>",)"
        << R"( "mask_token": "<mask>"})";
    std::ofstream(dir / "hf" / "chat_template.jinja") << kTinyGemmaChatTemplate;

    const auto layers = static_cast<int64_t>(m.layers.size());
    const int64_t h = m.hidden;
    const auto full = [&](int64_t i) { return m.layers[static_cast<size_t>(i)] == "full_attention"; };
    std::vector<int32_t> ffn, kv;
    std::vector<bool> sliding;
    for (int64_t i = 0; i < layers; ++i) {
        ffn.push_back(static_cast<int32_t>(i >= m.first_shared() ? 2 * m.ffn : m.ffn));
        kv.push_back(static_cast<int32_t>(full(i) ? m.global_kv_heads : m.kv_heads));
        sliding.push_back(!full(i));
    }
    GgufWriter w;
    w.string("general.architecture", "gemma4")
        .string("general.type", "model")
        .string("general.name", "Tiny Gemma 4")
        .u32("general.file_type", 7)
        .u32("general.quantization_version", 2)
        .u32("gemma4.block_count", static_cast<uint32_t>(layers))
        .u32("gemma4.context_length", 4096)
        .u32("gemma4.embedding_length", static_cast<uint32_t>(h))
        .i32s("gemma4.feed_forward_length", ffn)
        .u32("gemma4.attention.head_count", static_cast<uint32_t>(m.heads))
        .i32s("gemma4.attention.head_count_kv", kv)
        .f32("gemma4.rope.freq_base", 1e6f)
        .f32("gemma4.rope.freq_base_swa", 1e4f)
        .f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6f)
        .u32("gemma4.attention.key_length", static_cast<uint32_t>(m.global_head_dim))
        .u32("gemma4.attention.value_length", static_cast<uint32_t>(m.global_head_dim))
        .f32("gemma4.final_logit_softcapping", 30.0f)
        .u32("gemma4.attention.sliding_window", static_cast<uint32_t>(m.window))
        .u32("gemma4.attention.shared_kv_layers", static_cast<uint32_t>(m.shared))
        .u32("gemma4.embedding_length_per_layer_input", static_cast<uint32_t>(m.ple))
        .bools("gemma4.attention.sliding_window_pattern", sliding)
        .u32("gemma4.attention.key_length_swa", static_cast<uint32_t>(m.head_dim))
        .u32("gemma4.attention.value_length_swa", static_cast<uint32_t>(m.head_dim))
        .u32("gemma4.rope.dimension_count", static_cast<uint32_t>(m.global_head_dim))
        .u32("gemma4.rope.dimension_count_swa", static_cast<uint32_t>(m.head_dim));
    add_gemma4_tokenizer_metadata(w, tokenizer);

    const auto source = SafetensorsWeights::open_dir(dir / "hf");
    const auto values = [&](const std::string& name) {
        const TensorView& v = source->get(name);
        std::vector<float> out(static_cast<size_t>(v.numel()));
        quant::read_rows(v, 0, quant::matrix_rows(v.shape), out.data());
        return out;
    };
    const auto same = [](const std::vector<float>& v) { return v; };
    TinyGgufBuilder b(w);
    int matrices = 0;
    const auto next_block = [&] { return matrices++ % 2 == 0 ? Storage::Q8_0 : Storage::Q4_0; };
    const auto tensor = [&](const std::string& gguf, const std::string& hf, const Dims& shape, Storage s) {
        b.put(gguf, shape, values(hf), s, same, {{hf, shape}});
    };
    const auto f32 = [&](const std::string& gguf, const std::string& hf, const Dims& shape) {
        tensor(gguf, hf, shape, Storage::F32);
    };

    const std::string P = "model.";
    tensor("token_embd.weight", P + "embed_tokens.weight", {m.vocab, h}, Storage::Q8_0);
    tensor("per_layer_token_embd.weight", P + "embed_tokens_per_layer.weight", {m.vocab, layers * m.ple},
           Storage::Q4_0);
    tensor("per_layer_model_proj.weight", P + "per_layer_model_projection.weight", {layers * m.ple, h}, Storage::BF16);
    f32("per_layer_proj_norm.weight", P + "per_layer_projection_norm.weight", {m.ple});
    f32("output_norm.weight", P + "norm.weight", {h});
    // Proportional RoPE: a factor of 1 for the rotated quarter of a full-attention head's pairs, 1e30 for the rest.
    std::vector<float> factors(static_cast<size_t>(m.global_head_dim / 2), 1e30f);
    std::fill_n(factors.begin(), m.global_head_dim / 8, 1.0f);
    w.f32_tensor("rope_freqs.weight", {m.global_head_dim / 2}, factors);

    for (int64_t i = 0; i < layers; ++i) {
        const std::string B = "blk." + std::to_string(i) + ".", L = P + "layers." + std::to_string(i) + ".";
        const int64_t hd = full(i) ? m.global_head_dim : m.head_dim, n_kv = kv[static_cast<size_t>(i)];
        const int64_t f = ffn[static_cast<size_t>(i)];
        f32(B + "attn_norm.weight", L + "input_layernorm.weight", {h});
        const Dims q{m.heads * hd, h}, k{n_kv * hd, h};
        if (i == 0) {
            std::vector<float> qkv = values(L + "self_attn.q_proj.weight");
            for (const char* part : {"self_attn.k_proj.weight", "self_attn.v_proj.weight"}) {
                const std::vector<float> rows = values(L + part);
                qkv.insert(qkv.end(), rows.begin(), rows.end());
            }
            b.put(B + "attn_qkv.weight", {(m.heads + 2 * n_kv) * hd, h}, qkv, Storage::Q8_0, same,
                  {{L + "self_attn.q_proj.weight", q},
                   {L + "self_attn.k_proj.weight", k},
                   {L + "self_attn.v_proj.weight", k}});
        } else {
            tensor(B + "attn_q.weight", L + "self_attn.q_proj.weight", q, next_block());
            tensor(B + "attn_k.weight", L + "self_attn.k_proj.weight", k, next_block());
            tensor(B + "attn_v.weight", L + "self_attn.v_proj.weight", k, next_block());
        }
        tensor(B + "attn_output.weight", L + "self_attn.o_proj.weight", {h, m.heads * hd},
               i == 1 ? Storage::F16 : next_block());
        f32(B + "attn_q_norm.weight", L + "self_attn.q_norm.weight", {hd});
        f32(B + "attn_k_norm.weight", L + "self_attn.k_norm.weight", {hd});
        f32(B + "post_attention_norm.weight", L + "post_attention_layernorm.weight", {h});
        f32(B + "ffn_norm.weight", L + "pre_feedforward_layernorm.weight", {h});
        tensor(B + "ffn_gate.weight", L + "mlp.gate_proj.weight", {f, h}, next_block());
        tensor(B + "ffn_up.weight", L + "mlp.up_proj.weight", {f, h}, next_block());
        tensor(B + "ffn_down.weight", L + "mlp.down_proj.weight", {h, f}, next_block());
        f32(B + "post_ffw_norm.weight", L + "post_feedforward_layernorm.weight", {h});
        f32(B + "inp_gate.weight", L + "per_layer_input_gate.weight", {m.ple, h});
        f32(B + "proj.weight", L + "per_layer_projection.weight", {h, m.ple});
        f32(B + "post_norm.weight", L + "post_per_layer_input_norm.weight", {h});
        f32(B + "layer_output_scale.weight", L + "layer_scalar", {1});
    }

    fs::create_directories(dir / "twin");
    write_safetensors(dir / "twin" / "model.safetensors", b.twin_units());
    nlohmann::json config = nlohmann::json::parse(std::ifstream(dir / "hf" / "config.json"));
    config["eos_token_id"] = {1, kTinyGemmaEos,
                              9};  // the original's generation_config.json: <eos>, <turn|>, <|tool_response>
    std::ofstream(dir / "twin" / "config.json") << config.dump(1);
    for (const char* name : {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}) {
        fs::copy_file(dir / "hf" / name, dir / "twin" / name, fs::copy_options::overwrite_existing);
    }
    return w;
}

// The tiny assistant's GGUF, as Gemma4AssistantModel writes one: gemma4-assistant.* metadata (every layer drafting
// over the target's KV), the tokenizer without a chat template, nextn.*_projection, no centroid head. Writes, under
// `dir`: hf-assistant/ (the checkpoint) and twin-assistant/ (the checkpoint the GGUF stands for).
inline GgufWriter build_tiny_gemma4_assistant_gguf(const std::filesystem::path& dir, uint32_t seed) {
    namespace fs = std::filesystem;
    const TinyGemma4 target = tiny_gemma4_gguf_model();
    const TinyGemma4Assistant a;
    write_tiny_gemma4_assistant(dir / "hf-assistant", a, target, seed);
    const auto layers = static_cast<int64_t>(a.layers.size());
    const int64_t h = a.hidden;
    const auto full = [&](int64_t i) { return a.layers[static_cast<size_t>(i)] == "full_attention"; };
    std::vector<int32_t> kv;
    std::vector<bool> sliding;
    for (int64_t i = 0; i < layers; ++i) {
        kv.push_back(static_cast<int32_t>(full(i) ? target.global_kv_heads : target.kv_heads));
        sliding.push_back(!full(i));
    }
    const std::string A = "gemma4-assistant.";
    GgufWriter w;
    w.string("general.architecture", "gemma4-assistant")
        .string("general.type", "model")
        .string("general.name", "Tiny Gemma 4 assistant")
        .u32("general.file_type", 7)
        .u32("general.quantization_version", 2)
        .u32(A + "block_count", static_cast<uint32_t>(layers))
        .u32(A + "context_length", 4096)
        .u32(A + "embedding_length", static_cast<uint32_t>(h))
        .u32(A + "feed_forward_length", static_cast<uint32_t>(a.ffn))
        .u32(A + "attention.head_count", static_cast<uint32_t>(a.heads))
        .i32s(A + "attention.head_count_kv", kv)
        .f32(A + "rope.freq_base", 1e6f)
        .f32(A + "rope.freq_base_swa", 1e4f)
        .f32(A + "attention.layer_norm_rms_epsilon", 1e-6f)
        .u32(A + "attention.key_length", static_cast<uint32_t>(target.global_head_dim))
        .u32(A + "attention.value_length", static_cast<uint32_t>(target.global_head_dim))
        .u32(A + "attention.sliding_window", static_cast<uint32_t>(target.window))
        .u32(A + "attention.shared_kv_layers", static_cast<uint32_t>(layers))
        .u32(A + "embedding_length_per_layer_input", 0)
        .bools(A + "attention.sliding_window_pattern", sliding)
        .u32(A + "attention.key_length_swa", static_cast<uint32_t>(target.head_dim))
        .u32(A + "attention.value_length_swa", static_cast<uint32_t>(target.head_dim))
        .u32(A + "rope.dimension_count", static_cast<uint32_t>(target.global_head_dim))
        .u32(A + "rope.dimension_count_swa", static_cast<uint32_t>(target.head_dim))
        .u32(A + "embedding_length_out", static_cast<uint32_t>(target.hidden))
        .u32(A + "nextn_predict_layers", static_cast<uint32_t>(layers));
    add_gemma4_tokenizer_metadata(w, tiny_gemma4_tokenizer_json());
    w.drop_value("tokenizer.chat_template")
        .drop_value("tokenizer.ggml.eos_token_id")
        .u32("tokenizer.ggml.eos_token_id", 1);

    const auto source = SafetensorsWeights::open_dir(dir / "hf-assistant");
    const auto values = [&](const std::string& name) {
        const TensorView& v = source->get(name);
        std::vector<float> out(static_cast<size_t>(v.numel()));
        quant::read_rows(v, 0, quant::matrix_rows(v.shape), out.data());
        return out;
    };
    const auto same = [](const std::vector<float>& v) { return v; };
    TinyGgufBuilder b(w);
    int matrices = 0;
    const auto next_block = [&] { return matrices++ % 2 == 0 ? Storage::Q8_0 : Storage::Q4_0; };
    const auto tensor = [&](const std::string& gguf, const std::string& hf, const Dims& shape, Storage s) {
        b.put(gguf, shape, values(hf), s, same, {{hf, shape}});
    };
    tensor("token_embd.weight", "model.embed_tokens.weight", {target.vocab, h}, Storage::Q8_0);
    tensor("output_norm.weight", "model.norm.weight", {h}, Storage::F32);
    tensor("nextn.pre_projection.weight", "pre_projection.weight", {h, 2 * target.hidden}, Storage::Q8_0);
    tensor("nextn.post_projection.weight", "post_projection.weight", {target.hidden, h}, Storage::Q4_0);
    std::vector<float> factors(static_cast<size_t>(target.global_head_dim / 2), 1e30f);
    std::fill_n(factors.begin(), target.global_head_dim / 8, 1.0f);
    w.f32_tensor("rope_freqs.weight", {target.global_head_dim / 2}, factors);
    for (int64_t i = 0; i < layers; ++i) {
        const std::string B = "blk." + std::to_string(i) + ".", L = "model.layers." + std::to_string(i) + ".";
        const int64_t hd = full(i) ? target.global_head_dim : target.head_dim;
        tensor(B + "attn_norm.weight", L + "input_layernorm.weight", {h}, Storage::F32);
        tensor(B + "attn_q.weight", L + "self_attn.q_proj.weight", {a.heads * hd, h}, next_block());
        tensor(B + "attn_q_norm.weight", L + "self_attn.q_norm.weight", {hd}, Storage::F32);
        tensor(B + "attn_output.weight", L + "self_attn.o_proj.weight", {h, a.heads * hd}, next_block());
        tensor(B + "post_attention_norm.weight", L + "post_attention_layernorm.weight", {h}, Storage::F32);
        tensor(B + "ffn_norm.weight", L + "pre_feedforward_layernorm.weight", {h}, Storage::F32);
        tensor(B + "ffn_gate.weight", L + "mlp.gate_proj.weight", {a.ffn, h}, next_block());
        tensor(B + "ffn_up.weight", L + "mlp.up_proj.weight", {a.ffn, h}, next_block());
        tensor(B + "ffn_down.weight", L + "mlp.down_proj.weight", {h, a.ffn}, i == 1 ? Storage::BF16 : next_block());
        tensor(B + "post_ffw_norm.weight", L + "post_feedforward_layernorm.weight", {h}, Storage::F32);
        tensor(B + "layer_output_scale.weight", L + "layer_scalar", {1}, Storage::F32);
    }
    fs::create_directories(dir / "twin-assistant");
    write_safetensors(dir / "twin-assistant" / "model.safetensors", b.twin_units());
    fs::copy_file(dir / "hf-assistant" / "config.json", dir / "twin-assistant" / "config.json",
                  fs::copy_options::overwrite_existing);
    return w;
}

}  // namespace xs::test
