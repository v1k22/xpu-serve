#pragma once

// Tiny embedding models written to disk for tests: a BERT or NomicBERT encoder (random bf16 weights under the released
// checkpoints' names) with the sentence-transformers files around it.

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tiny_checkpoints.h"

namespace xs::test {

struct TinyEncoder {
    std::string arch = "bert";  // or "nomic_bert"
    int64_t vocab = 320, hidden = 64, heads = 2, layers = 2, ffn = 128;  // head_dim 32, all-MiniLM-L6-v2's
    int64_t positions = 64;  // bert: rows of the position table
    std::string pooling = "mean";
    bool normalize = false;
    int64_t max_seq_length = 48;  // 0: no sentence_bert_config.json
};

inline nlohmann::json tiny_encoder_config(const TinyEncoder& m) {
    if (m.arch == "bert") {
        return {{"model_type", "bert"},        {"vocab_size", m.vocab},          {"hidden_size", m.hidden},
                {"num_hidden_layers", m.layers}, {"num_attention_heads", m.heads}, {"intermediate_size", m.ffn},
                {"hidden_act", "gelu"},        {"max_position_embeddings", m.positions},
                {"type_vocab_size", 2},        {"layer_norm_eps", 1e-12},        {"position_embedding_type", "absolute"}};
    }
    return {{"model_type", "nomic_bert"}, {"vocab_size", m.vocab},     {"n_embd", m.hidden},
            {"n_head", m.heads},          {"n_layer", m.layers},       {"n_inner", m.ffn},
            {"activation_function", "swiglu"}, {"rotary_emb_fraction", 1.0}, {"rotary_emb_base", 1000},
            {"qkv_proj_bias", false},     {"mlp_fc1_bias", false},     {"mlp_fc2_bias", false},
            {"prenorm", false},           {"layer_norm_epsilon", 1e-12}, {"n_positions", 8192},
            {"max_position_embeddings", 2048}, {"max_trained_positions", 2048}, {"rotary_scaling_factor", nullptr},
            {"type_vocab_size", 2}};
}

// modules.json, 1_Pooling/config.json and sentence_bert_config.json for `m`.
inline void write_sentence_transformers_files(const fs::path& dir, const TinyEncoder& m) {
    fs::create_directories(dir);
    nlohmann::json modules = {{{"idx", 0}, {"name", "0"}, {"path", ""}, {"type", "sentence_transformers.models.Transformer"}},
                              {{"idx", 1}, {"name", "1"}, {"path", "1_Pooling"}, {"type", "sentence_transformers.models.Pooling"}}};
    if (m.normalize) {
        modules.push_back(
            {{"idx", 2}, {"name", "2"}, {"path", "2_Normalize"}, {"type", "sentence_transformers.models.Normalize"}});
    }
    std::ofstream(dir / "modules.json") << modules.dump(1);
    fs::create_directories(dir / "1_Pooling");
    std::ofstream(dir / "1_Pooling" / "config.json")
        << nlohmann::json{{"word_embedding_dimension", m.hidden},
                          {"pooling_mode_cls_token", m.pooling == "cls"},
                          {"pooling_mode_mean_tokens", m.pooling == "mean"},
                          {"pooling_mode_max_tokens", false},
                          {"pooling_mode_mean_sqrt_len_tokens", false}}
               .dump(1);
    if (m.max_seq_length > 0) {
        std::ofstream(dir / "sentence_bert_config.json")
            << nlohmann::json{{"max_seq_length", m.max_seq_length}, {"do_lower_case", false}}.dump(1);
    }
}

// A whole checkpoint: config.json, model.safetensors and the sentence-transformers files.
inline void write_tiny_encoder(const fs::path& dir, const TinyEncoder& m, uint32_t seed) {
    fs::create_directories(dir);
    std::ofstream(dir / "config.json") << tiny_encoder_config(m).dump(1);
    write_sentence_transformers_files(dir, m);
    std::vector<TinyTensor> specs;
    const auto matrix = [&](const std::string& name, int64_t out, int64_t in) {
        const float range = 1.5f / std::sqrt(static_cast<float>(in));
        specs.push_back({name, {out, in}, -range, range});
    };
    const auto norm = [&](const std::string& stem) {
        specs.push_back({stem + ".weight", {m.hidden}, 0.5f, 1.5f});
        specs.push_back({stem + ".bias", {m.hidden}, -0.2f, 0.2f});
    };
    const auto linear = [&](const std::string& stem, int64_t out, int64_t in) {
        matrix(stem + ".weight", out, in);
        specs.push_back({stem + ".bias", {out}, -0.2f, 0.2f});
    };
    specs.push_back({"embeddings.word_embeddings.weight", {m.vocab, m.hidden}, -1.0f, 1.0f});
    specs.push_back({"embeddings.token_type_embeddings.weight", {2, m.hidden}, -0.5f, 0.5f});
    if (m.arch == "bert") {
        specs.push_back({"embeddings.position_embeddings.weight", {m.positions, m.hidden}, -1.0f, 1.0f});
        norm("embeddings.LayerNorm");
        for (int64_t l = 0; l < m.layers; ++l) {
            const std::string L = "encoder.layer." + std::to_string(l) + ".";
            for (const char* p : {"query", "key", "value"})
                linear(L + "attention.self." + p, m.hidden, m.hidden);
            linear(L + "attention.output.dense", m.hidden, m.hidden);
            norm(L + "attention.output.LayerNorm");
            linear(L + "intermediate.dense", m.ffn, m.hidden);
            linear(L + "output.dense", m.hidden, m.ffn);
            norm(L + "output.LayerNorm");
        }
    } else {
        norm("emb_ln");
        for (int64_t l = 0; l < m.layers; ++l) {
            const std::string L = "encoder.layers." + std::to_string(l) + ".";
            matrix(L + "attn.Wqkv.weight", 3 * m.hidden, m.hidden);
            matrix(L + "attn.out_proj.weight", m.hidden, m.hidden);
            norm(L + "norm1");
            matrix(L + "mlp.fc11.weight", m.ffn, m.hidden);
            matrix(L + "mlp.fc12.weight", m.ffn, m.hidden);
            matrix(L + "mlp.fc2.weight", m.hidden, m.ffn);
            norm(L + "norm2");
        }
    }
    write_random(dir / "model.safetensors", specs, seed);
}

}  // namespace xs::test
