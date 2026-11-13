// NomicBERT's encoder as modeling_hf_nomic_bert.py computes it (NomicBertModel with post-norm NomicBertBlocks,
// NomicBertAttention, NomciBertGatedMLP).

#include "models/common/encoder.h"
#include "models/common/torch_nn.h"
#include "models/nomic_bert/nomic_bert.h"
#include "util/error.h"

namespace xs::nomic_bert {
namespace {

using encoder::Graph;
using encoder::Value;

Value encode(Graph& g, const WeightStore& w, const Config& c, Value ids, Value positions) {
    const float eps = static_cast<float>(c.layer_norm_eps);
    const int64_t h = c.hidden_size;
    Value x;
    {
        Graph::Scope s(g, "embeddings");
        x = g.add(g.gather(g.weight(w, "embeddings.word_embeddings.weight"), ids),
                  encoder::table_row(g, w, "embeddings.token_type_embeddings.weight", 0));
        x = torch_nn::layer_norm(g, w, x, "emb_ln", eps);
    }
    const Value inv_freq = g.constant(blocks::rope_inv_freq(c.rope_theta, c.head_dim, c.head_dim / 2, c.head_dim / 2));
    const blocks::AttentionSpec spec = encoder::attention_spec(c.num_heads, c.head_dim);
    const auto weight = [&](const std::string& name) { return blocks::matmul_by(g, g.weight(w, name)); };
    for (int64_t i = 0; i < c.num_layers; ++i) {
        Graph::Scope s(g, "layers." + std::to_string(i));
        const std::string L = "encoder.layers." + std::to_string(i) + ".";
        encoder::Layer layer;
        layer.attention = [&](Value v) {
            const Value qkv = g.matmul(v, g.weight(w, L + "attn.Wqkv.weight"));  // [T, 3 h]: q, k, v
            const auto part = [&g, qkv, h](int64_t k) {
                return blocks::Linear([&g, qkv, h, k](Value) { return g.slice(qkv, 1, k * h, (k + 1) * h); });
            };
            blocks::AttentionParts parts;
            parts.q = part(0);
            parts.k = part(1);
            parts.v = part(2);
            parts.o = weight(L + "attn.out_proj.weight");
            return blocks::attention(g, spec, parts, v, positions, inv_freq).out;
        };
        layer.norm1 = L + "norm1";
        layer.mlp = [&](Value v) {
            return blocks::gated_mlp(g, v, weight(L + "mlp.fc12.weight"), weight(L + "mlp.fc11.weight"),
                                     weight(L + "mlp.fc2.weight"), blocks::Act::Silu);
        };
        layer.norm2 = L + "norm2";
        x = encoder::post_norm_layer(g, w, layer, x, eps);
    }
    return x;
}

}  // namespace

std::unique_ptr<EmbeddingModel> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                                     const st::Pipeline& pipeline) {
    const Config c = parse_config(config_json);
    std::unique_ptr<WeightStore> weights = SafetensorsWeights::open_dir(dir);
    const Dims& words = weights->get("embeddings.word_embeddings.weight").shape;
    XS_CHECK(words.size() == 2 && words[1] == c.hidden_size, "nomic_bert: the word table ", dims_to_string(words),
             " does not have the hidden size ", c.hidden_size);
    const int64_t max_tokens = pipeline.max_seq_length.value_or(c.max_positions);
    EmbeddingSpec spec = encoder::embedding_spec(dir, "nomic_bert", pipeline, c.hidden_size, words[0], max_tokens);
    return encoder::make_model(std::move(spec), std::move(weights),
                               [&c](Graph& g, const WeightStore& w, Value ids, Value positions) {
                                   return encode(g, w, c, ids, positions);
                               });
}

}  // namespace xs::nomic_bert
