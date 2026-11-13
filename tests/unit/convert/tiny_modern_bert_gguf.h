#pragma once

// A tiny modern-bert decision GGUF written the way llama.cpp's converter writes one (conversion/bert.py,
// ModernBertDecisionModel), from a tiny Laya or Julia-1 checkpoint (model/decision_test_util.h, write_tiny_laya) with a
// tokenizer of the kind each model has, in miniature: ModernBERT's byte-level BPE for Laya, mmBERT's Gemma-style
// vocabulary for Julia-1. GGUF names, the head's layers as the blocks after the encoder's; the encoder's matrices and
// the embedding Q8_0 (its down projections F16: 48 columns are no whole block), the head's attention BF16 and its MLP
// Q8_0, the scorer's last row BF16 and 1-D in Laya's file (as llama-quantize leaves it), 2-D in Julia's; norms, biases
// and token_types F32. Token types as the converter gives them. The metadata the converter writes: the head's blocks in
// block_count, a feed-forward width per block, the decision keys (Laya's temperatures, both files' question budget)
// and the prompt template (tokenizer.chat_template.systemone).
//
// And its twin: the checkpoint those bytes stand for, every weight as the f32 values the file holds.

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "convert/tiny_gguf.h"
#include "io/gguf_test_util.h"
#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "model/decision_test_util.h"
#include "quant/quant.h"
#include "tokenizer/byte_level.h"

namespace xs::test {

inline nlohmann::json added_token(int64_t id, const std::string& text, bool special, bool normalized, bool lstrip) {
    return {{"id", id},          {"content", text},           {"single_word", false}, {"lstrip", lstrip},
            {"rstrip", false},   {"normalized", normalized}, {"special", special}};
}

// A BERT-style post-processor: `cls` A `sep` (pair: `cls` A `sep` B `sep`) and `tokens` (text, id) as its specials.
inline nlohmann::json bert_post_processor(const std::string& cls, const std::string& sep,
                                          const std::vector<std::pair<std::string, int64_t>>& tokens) {
    const auto piece = [](const char* kind, const std::string& id) {
        return nlohmann::json{{kind, {{"id", id}, {"type_id", 0}}}};
    };
    nlohmann::json special = nlohmann::json::object();
    for (const auto& [text, id] : tokens)
        special[text] = {{"id", text}, {"ids", {id}}, {"tokens", {text}}};
    return {{"type", "TemplateProcessing"},
            {"single", {piece("SpecialToken", cls), piece("Sequence", "A"), piece("SpecialToken", sep)}},
            {"pair",
             {piece("SpecialToken", cls), piece("Sequence", "A"), piece("SpecialToken", sep), piece("Sequence", "B"),
              piece("SpecialToken", sep)}},
            {"special_tokens", special}};
}

// ModernBERT's tokenizer.json in miniature: NFC, ByteLevel with the GPT-2 regex, the 256 byte pieces and a few merges,
// then added tokens: a run of two spaces and |||IP_ADDRESS||| (not special, normalized), [UNK] [CLS] [SEP] [PAD] and
// [MASK] (special; the mask takes the spaces before it).
inline nlohmann::json tiny_modern_bert_tokenizer_json() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int b = 0; b < 256; ++b)
        vocab[tok::byte_level::byte_piece(static_cast<uint8_t>(b))] = b;
    nlohmann::json merges = nlohmann::json::array();
    int64_t id = 256;
    for (const auto& [a, b] : std::vector<std::pair<std::string, std::string>>{
             {"Ġ", "t"}, {"h", "e"}, {"Ġt", "he"}, {"Ġ", "a"}, {"i", "n"}, {"Ġ", "Ġ"}, {"e", "r"}, {"o", "n"}}) {
        vocab[a + b] = id++;
        merges.push_back({a, b});
    }
    nlohmann::json added = nlohmann::json::array();
    added.push_back(added_token(id++, "  ", false, true, false));
    added.push_back(added_token(id++, "|||IP_ADDRESS|||", false, true, false));
    std::vector<std::pair<std::string, int64_t>> specials;
    for (const char* text : {"[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"}) {
        specials.emplace_back(text, id);
        added.push_back(added_token(id++, text, true, false, std::string(text) == "[MASK]"));
    }
    const nlohmann::json byte_level = {
        {"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", true}};
    return {{"version", "1.0"},
            {"truncation", nullptr},
            {"padding", nullptr},
            {"added_tokens", added},
            {"normalizer", {{"type", "NFC"}}},
            {"pre_tokenizer", byte_level},
            {"post_processor", bert_post_processor("[CLS]", "[SEP]", specials)},
            {"decoder", byte_level},
            {"model",
             {{"type", "BPE"},
              {"dropout", nullptr},
              {"unk_token", nullptr},
              {"continuing_subword_prefix", nullptr},
              {"end_of_word_suffix", nullptr},
              {"fuse_unk", false},
              {"byte_fallback", false},
              {"ignore_merges", false},
              {"vocab", vocab},
              {"merges", merges}}}};
}

// mmBERT's in miniature: spaces as U+2581, a Metaspace split, BPE with byte fallback. <pad> <eos> <bos> <unk> <mask>
// (special; the mask takes the spaces before it), <unused0> and a newline (added, not special), byte pieces, a run of
// two U+2581 (added, not special), U+2581 and letters, merges; every added token also a piece.
inline nlohmann::json tiny_mmbert_tokenizer_json() {
    nlohmann::json vocab = nlohmann::json::object();
    nlohmann::json added = nlohmann::json::array();
    int64_t id = 0;
    for (const char* text : {"<pad>", "<eos>", "<bos>", "<unk>", "<mask>"}) {
        added.push_back(added_token(id, text, true, false, std::string(text) == "<mask>"));
        vocab[text] = id++;
    }
    for (const char* text : {"<unused0>", "\n"}) {
        added.push_back(added_token(id, text, false, false, false));
        vocab[text] = id++;
    }
    for (int b = 0; b < 256; ++b) {
        char piece[8];
        std::snprintf(piece, sizeof(piece), "<0x%02X>", b);
        vocab[piece] = id++;
    }
    added.push_back(added_token(id, "▁▁", false, false, false));
    vocab["▁▁"] = id++;
    for (const char* c : {"▁", "a", "c", "d", "e", "h", "i", "n", "o", "r", "s", "t"})
        vocab[c] = id++;
    nlohmann::json merges = nlohmann::json::array();
    for (const auto& [a, b] : std::vector<std::pair<std::string, std::string>>{
             {"▁", "t"}, {"h", "e"}, {"▁t", "he"}, {"i", "n"}, {"▁", "a"}, {"e", "r"}, {"o", "n"}, {"▁", "▁"}}) {
        if (!vocab.contains(a + b)) vocab[a + b] = id++;
        merges.push_back({a, b});
    }
    return {{"version", "1.0"},
            {"truncation", nullptr},
            {"padding", nullptr},
            {"added_tokens", added},
            {"normalizer", {{"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "▁"}}},
            {"pre_tokenizer",
             {{"type", "Metaspace"}, {"replacement", "▁"}, {"prepend_scheme", "always"}, {"split", true}}},
            {"post_processor", bert_post_processor("<bos>", "<eos>", {{"<bos>", 2}, {"<eos>", 1}})},
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
              {"merges", merges}}}};
}

// The vocabulary size of a tokenizer.json: one past its largest id.
inline int64_t vocabulary_size(const nlohmann::json& tokenizer) {
    int64_t n = 0;
    for (const auto& [piece, id] : tokenizer["model"]["vocab"].items())
        n = std::max(n, id.get<int64_t>() + 1);
    for (const auto& t : tokenizer["added_tokens"])
        n = std::max(n, t["id"].get<int64_t>() + 1);
    return n;
}

// The tokenizer metadata the converter writes for `tokenizer` (_set_vocab_gpt2, SpecialVocab; for mmBERT its runs of
// U+2581 normal): pieces normal, special added tokens and the ones that look like <...> control, other added tokens
// user-defined.
inline void add_decision_tokenizer_metadata(GgufWriter& w, const nlohmann::json& tokenizer, bool mmbert) {
    const auto n = static_cast<size_t>(vocabulary_size(tokenizer));
    std::vector<std::string> tokens(n);
    std::vector<int32_t> types(n, 1);
    for (const auto& [piece, id] : tokenizer["model"]["vocab"].items())
        tokens[id.get<size_t>()] = piece;
    for (const auto& t : tokenizer["added_tokens"]) {
        const std::string text = t["content"];
        const bool looks_special = text.size() > 2 && text.front() == '<' && text.back() == '>';
        const auto i = t["id"].get<size_t>();
        tokens[i] = text;
        if (mmbert && text == "▁▁") continue;
        types[i] = t["special"].get<bool>() || looks_special ? 3 : 4;
    }
    std::vector<std::string> merges;
    for (const auto& m : tokenizer["model"]["merges"])
        merges.push_back(m[0].get<std::string>() + " " + m[1].get<std::string>());
    const auto id_of = [&](const std::string& text) {
        return static_cast<uint32_t>(std::find(tokens.begin(), tokens.end(), text) - tokens.begin());
    };
    const std::string cls = mmbert ? "<bos>" : "[CLS]", sep = mmbert ? "<eos>" : "[SEP]";
    w.boolean("tokenizer.ggml.add_bos_token", true)
        .boolean("tokenizer.ggml.add_eos_token", true)
        .boolean("tokenizer.ggml.add_sep_token", false)
        .string("tokenizer.ggml.model", "gpt2")
        .string("tokenizer.ggml.pre", mmbert ? "mmbert" : "modern-bert")
        .strings("tokenizer.ggml.tokens", tokens)
        .i32s("tokenizer.ggml.token_type", types)
        .strings("tokenizer.ggml.merges", merges)
        .u32("tokenizer.ggml.bos_token_id", id_of(cls))
        .u32("tokenizer.ggml.eos_token_id", id_of(sep))
        .u32("tokenizer.ggml.unknown_token_id", id_of(mmbert ? "<unk>" : "[UNK]"))
        .u32("tokenizer.ggml.seperator_token_id", id_of(sep))
        .u32("tokenizer.ggml.padding_token_id", id_of(mmbert ? "<pad>" : "[PAD]"))
        .u32("tokenizer.ggml.mask_token_id", id_of(mmbert ? "<mask>" : "[MASK]"))
        .boolean("tokenizer.ggml.clean_spaces", !mmbert)
        .u32("tokenizer.ggml.token_type_count", 3);
    if (mmbert) {
        w.string("tokenizer.ggml.metaspace.prepend_scheme", "always");
    } else {
        w.boolean("tokenizer.ggml.normalizer.nfc", true);
    }
}

// The decision prompt template the converter writes (ModernBertDecisionModel._systemone_template).
inline std::string systemone_template(bool julia, const std::string& cls, const std::string& sep,
                                      const std::string& mask) {
    const auto text_or_json = [](const std::string& name) {
        return "{{ " + name + " if " + name + " is string else " + name + " | tojson }}";
    };
    const std::string d = text_or_json("o.description");
    const std::string option =
        julia ? "{% if o.description %}" + d + "{% else %}{{ o.key }}{% endif %}"
              : "{% if type == 'choice' %}{{ o.key }}{% if o.description %}: " + d +
                    "{% endif %}{% elif type == 'score' %}level {{ o.key }}: " + d +
                    "{% else %}{{ o.key }}: {% if o.description %}" + d +
                    "{% elif o.key == 'true' %}yes, the statement holds{% else %}no, the statement does not hold"
                    "{% endif %}{% endif %}";
    return cls + "{{ type }} question: " + text_or_json("instructions") + sep + "{% for o in options %}" + mask + " " +
           option + "{% endfor %}" + sep + text_or_json("state") + sep;
}

// Laya's tiny file as published: the root model's question budget (192, which goes with a state budget of 512) and
// temperatures that float32 holds exactly.
inline TinyLaya tiny_decision_model(bool julia, int64_t vocab) {
    TinyLaya m;
    m.julia = julia;
    m.vocab = vocab;
    m.max_positions = 8192;
    m.max_tokens = julia ? 8192 : 512;
    m.head_tokens = julia ? 512 : 192;
    if (!julia) {
        m.temperature = {1.5, 0.75, 2.0};
        m.temperature_by_options = {{"choice:3-5", 0.5}, {"noul:2", 1.25}, {"choice:11+", 0.125}};
    }
    return m;
}

// Writes, under `dir`: hf/ (the tiny checkpoint and its tokenizer), twin/ (the checkpoint the GGUF stands for).
// Returns the GGUF's writer: the test writes the file, as it is or changed.
inline GgufWriter build_tiny_modern_bert_gguf(const std::filesystem::path& dir, bool julia, uint32_t seed) {
    namespace fs = std::filesystem;
    const nlohmann::json tokenizer = julia ? tiny_mmbert_tokenizer_json() : tiny_modern_bert_tokenizer_json();
    const TinyLaya m = tiny_decision_model(julia, vocabulary_size(tokenizer));
    write_tiny_laya(dir / "hf", m, seed);
    std::ofstream(dir / "hf" / "tokenizer" / "tokenizer.json") << tokenizer.dump();
    std::ofstream(dir / "hf" / "tokenizer" / "tokenizer_config.json")
        << (julia ? nlohmann::json{{"bos_token", "<bos>"},  {"cls_token", "<bos>"}, {"eos_token", "<eos>"},
                                   {"sep_token", "<eos>"},  {"mask_token", "<mask>"}, {"pad_token", "<pad>"},
                                   {"unk_token", "<unk>"}}
                  : nlohmann::json{{"cls_token", "[CLS]"}, {"sep_token", "[SEP]"}, {"mask_token", "[MASK]"},
                                   {"pad_token", "[PAD]"}, {"unk_token", "[UNK]"}})
               .dump(1);
    const std::string cls = julia ? "<bos>" : "[CLS]", sep = julia ? "<eos>" : "[SEP]";
    const std::string mask = julia ? "<mask>" : "[MASK]";

    const int64_t h = 128, inter = 48, layers = 4, head_layers = 2;
    std::vector<int32_t> widths(layers, static_cast<int32_t>(inter));
    widths.insert(widths.end(), head_layers, static_cast<int32_t>(4 * h));
    GgufWriter w;
    w.string("general.architecture", "modern-bert")
        .string("general.type", "model")
        .string("general.name", julia ? "Tiny Julia" : "Tiny Laya")
        .u32("general.file_type", 7)
        .u32("general.quantization_version", 2)
        .u32("modern-bert.block_count", static_cast<uint32_t>(layers + head_layers))
        .u32("modern-bert.context_length", static_cast<uint32_t>(m.max_positions))
        .u32("modern-bert.embedding_length", static_cast<uint32_t>(h))
        .i32s("modern-bert.feed_forward_length", widths)
        .u32("modern-bert.attention.head_count", 4)
        .f32("modern-bert.rope.freq_base", 160000.0f)
        .f32("modern-bert.rope.freq_base_swa", 10000.0f)
        .f32("modern-bert.attention.layer_norm_rms_epsilon", 1e-5f)
        .f32("modern-bert.attention.layer_norm_epsilon", 1e-5f)
        .boolean("modern-bert.attention.causal", false)
        .u32("modern-bert.classifier.pooling_type", 1)
        .string("modern-bert.classifier.activation", "gelu")
        .u32("modern-bert.attention.sliding_window", 8)
        .u32("modern-bert.attention.sliding_window_pattern", 3)
        .string("modern-bert.rope.scaling.type", "none")
        .u32("modern-bert.vocab_size", static_cast<uint32_t>(m.vocab))
        .string("modern-bert.hidden_activation", "gelu")
        .string("modern-bert.decision.type", "laya")
        .u32("modern-bert.decision.block_count", static_cast<uint32_t>(head_layers))
        .u32("modern-bert.decision.max_head_tokens", julia ? 256 : static_cast<uint32_t>(m.head_tokens));
    if (!julia) {
        const char* types[] = {"choice", "score", "noul"};
        for (size_t t = 0; t < 3; ++t)
            w.f32(str_cat("modern-bert.decision.temperature.", types[t]), static_cast<float>(m.temperature[t]));
        for (const auto& [name, value] : m.temperature_by_options.items()) {
            std::string key = name;
            key.replace(key.find(':'), 1, ".");
            if (key.back() == '+') key.pop_back();
            if (const size_t dash = key.find('-'); dash != std::string::npos) key[dash] = '_';
            w.f32("modern-bert.decision.temperature." + key, value.get<float>());
        }
    }
    add_decision_tokenizer_metadata(w, tokenizer, julia);
    w.string("tokenizer.chat_template.systemone", systemone_template(julia, cls, sep, mask))
        .strings("tokenizer.chat_templates", {"systemone"});

    const auto source = SafetensorsWeights::open_dir(dir / "hf");
    const auto values = [&](const std::string& name) {
        const TensorView& v = source->get(name);
        std::vector<float> out(static_cast<size_t>(v.numel()));
        quant::read_rows(v, 0, quant::matrix_rows(v.shape), out.data());
        return out;
    };
    const auto same = [](const std::vector<float>& v) { return v; };
    TinyGgufBuilder b(w);
    const auto put = [&](const std::string& gguf, const std::string& torch, Storage s) {
        const Dims shape = source->get(torch).shape;
        b.put(gguf, shape, values(torch), s, same, {{torch, shape}});
    };
    put("token_embd.weight", "encoder.embeddings.tok_embeddings.weight", Storage::Q8_0);
    put("token_embd_norm.weight", "encoder.embeddings.norm.weight", Storage::F32);
    put("output_norm.weight", "encoder.final_norm.weight", Storage::F32);
    for (int64_t i = 0; i < layers; ++i) {
        const std::string B = str_cat("blk.", i, "."), L = str_cat("encoder.layers.", i, ".");
        if (i > 0) put(B + "attn_norm.weight", L + "attn_norm.weight", Storage::F32);
        put(B + "attn_qkv.weight", L + "attn.Wqkv.weight", Storage::Q8_0);
        put(B + "attn_output.weight", L + "attn.Wo.weight", Storage::Q8_0);
        put(B + "ffn_norm.weight", L + "mlp_norm.weight", Storage::F32);
        put(B + "ffn_up.weight", L + "mlp.Wi.weight", Storage::Q8_0);
        put(B + "ffn_down.weight", L + "mlp.Wo.weight", Storage::F16);
    }
    for (int64_t j = 0; j < head_layers; ++j) {
        const std::string B = str_cat("blk.", layers + j, "."), L = str_cat("head.layers.", j, ".");
        for (const char* p : {".weight", ".bias"}) {
            put(B + "attn_norm" + p, L + "norm1" + p, Storage::F32);
            put(B + "ffn_norm" + p, L + "norm2" + p, Storage::F32);
        }
        put(B + "attn_qkv.weight", L + "self_attn.in_proj_weight", Storage::BF16);
        put(B + "attn_qkv.bias", L + "self_attn.in_proj_bias", Storage::F32);
        put(B + "attn_output.weight", L + "self_attn.out_proj.weight", Storage::BF16);
        put(B + "attn_output.bias", L + "self_attn.out_proj.bias", Storage::F32);
        put(B + "ffn_up.weight", L + "linear1.weight", Storage::Q8_0);
        put(B + "ffn_up.bias", L + "linear1.bias", Storage::F32);
        put(B + "ffn_down.weight", L + "linear2.weight", Storage::Q8_0);
        put(B + "ffn_down.bias", L + "linear2.bias", Storage::F32);
    }
    put("token_types.weight", "type_emb.weight", Storage::F32);
    put("cls.norm.weight", "scorer.0.weight", Storage::F32);
    put("cls.norm.bias", "scorer.0.bias", Storage::F32);
    put("cls.weight", "scorer.1.weight", Storage::Q8_0);
    put("cls.bias", "scorer.1.bias", Storage::F32);
    // Laya's as llama-quantize writes it (one dimension), Julia's as the converter does ([h, 1] in the file's order).
    b.put("cls.output.weight", julia ? Dims{1, h} : Dims{h}, values("scorer.3.weight"), Storage::BF16, same,
          {{"scorer.3.weight", {1, h}}});
    put("cls.output.bias", "scorer.3.bias", Storage::F32);

    // The twin: the checkpoint's files, the weights as the file holds them.
    fs::remove_all(dir / "twin");
    fs::create_directories(dir / "twin");
    for (const auto& e : fs::directory_iterator(dir / "hf")) {
        if (e.path().filename() != "model.safetensors") {
            fs::copy(e.path(), dir / "twin" / e.path().filename(), fs::copy_options::recursive);
        }
    }
    write_safetensors(dir / "twin" / "model.safetensors", b.twin_units());
    return w;
}

}  // namespace xs::test
