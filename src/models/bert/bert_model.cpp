// BERT's encoder as transformers' BertModel computes it (modeling_bert.py: BertEmbeddings, BertLayer).

#include <algorithm>

#include "models/bert/bert.h"
#include "models/common/encoder.h"
#include "util/error.h"

namespace xs::bert {
namespace {

using encoder::Graph;
using encoder::Value;

Value linear(Graph& g, const WeightStore& w, Value x, const std::string& stem) {
    return g.add(g.matmul(x, g.weight(w, stem + ".weight")), g.weight(w, stem + ".bias"));
}

Value encode(Graph& g, const WeightStore& w, const Config& c, Value ids, Value positions) {
    const float eps = static_cast<float>(c.layer_norm_eps);
    Value x;
    {
        Graph::Scope s(g, "embeddings");
        x = g.add(g.gather(g.weight(w, "embeddings.word_embeddings.weight"), ids),
                  encoder::table_row(g, w, "embeddings.token_type_embeddings.weight", 0));
        x = g.add(x, g.gather(g.weight(w, "embeddings.position_embeddings.weight"), positions));
        x = encoder::layer_norm(g, w, x, "embeddings.LayerNorm", eps);
    }
    const blocks::AttentionSpec spec = encoder::attention_spec(c.num_heads, c.head_dim);
    for (int64_t i = 0; i < c.num_layers; ++i) {
        Graph::Scope s(g, "layers." + std::to_string(i));
        const std::string L = "encoder.layer." + std::to_string(i) + ".";
        const auto biased = [&g, &w](std::string stem) {
            return blocks::Linear([&g, &w, stem = std::move(stem)](Value v) { return linear(g, w, v, stem); });
        };
        blocks::AttentionParts parts;
        parts.q = biased(L + "attention.self.query");
        parts.k = biased(L + "attention.self.key");
        parts.v = biased(L + "attention.self.value");
        parts.o = biased(L + "attention.output.dense");
        encoder::Layer layer;
        layer.attention = [&](Value v) { return blocks::attention(g, spec, parts, v, Value{}, Value{}).out; };
        layer.norm1 = L + "attention.output.LayerNorm";
        layer.mlp = [&](Value v) {
            Graph::Scope m(g, "mlp");
            return linear(g, w, g.gelu(linear(g, w, v, L + "intermediate.dense")), L + "output.dense");
        };
        layer.norm2 = L + "output.LayerNorm";
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
    const Dims& table = weights->get("embeddings.position_embeddings.weight").shape;
    XS_CHECK(words.size() == 2 && words[1] == c.hidden_size && table.size() == 2 && table[1] == c.hidden_size,
             "bert: the word and position tables ", dims_to_string(words), ", ", dims_to_string(table),
             " do not have the hidden size ", c.hidden_size);
    XS_CHECK(table[0] == c.max_positions, "bert: the position table has ", table[0],
             " rows, config.json max_position_embeddings ", c.max_positions);
    const int64_t max_tokens = std::min(pipeline.max_seq_length.value_or(c.max_positions), c.max_positions);
    EmbeddingSpec spec = encoder::embedding_spec(dir, "bert", pipeline, c.hidden_size, words[0], max_tokens);
    return encoder::make_model(std::move(spec), std::move(weights),
                               [&c](Graph& g, const WeightStore& w, Value ids, Value positions) {
                                   return encode(g, w, c, ids, positions);
                               });
}

}  // namespace xs::bert
