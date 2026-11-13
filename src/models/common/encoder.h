#pragma once

// What the encoders of the embedding models share (models/bert, models/nomic_bert): BERT's post-norm layer around
// attention in which every token sees every other, LayerNorms with a bias, and the EmbeddingModel that holds one such
// graph. The graph reads "ids" and "positions" ([T] i32: the tokens, and 0, 1, ...) and writes "hidden".

#include <functional>
#include <memory>
#include <string>

#include "models/common/blocks.h"
#include "models/common/sentence_transformers.h"
#include "models/embedding.h"

namespace xs::encoder {

using graph::Graph;
using graph::Value;

// LayerNorm over the last dim with the weights <stem>.weight and <stem>.bias.
Value layer_norm(Graph& g, const WeightStore& weights, Value x, const std::string& stem, float eps);

// Row `row` of a float table [rows, width] as a constant [width]: a token type's embedding, which every token of a
// single sequence adds (its type is 0).
Value table_row(Graph& g, const WeightStore& weights, const std::string& name, int64_t row);

// Self-attention of `heads` heads of `head_dim` over the step's own tokens, each seeing every other: no KV slot, so the
// program keeps nothing from one run to the next. Scores are scaled by 1 / sqrt(head_dim).
blocks::AttentionSpec attention_spec(int64_t heads, int64_t head_dim);

// One post-norm layer, BERT's order: x = norm1(x + attention(x)); x = norm2(x + mlp(x)), norm1 and norm2 the stems of
// LayerNorms.
struct Layer {
    std::function<Value(Value)> attention;
    std::string norm1;
    std::function<Value(Value)> mlp;
    std::string norm2;
};
Value post_norm_layer(Graph& g, const WeightStore& weights, const Layer& layer, Value x, float eps);

// The spec of the checkpoint in `dir`: an encoder of `hidden` values a token over `vocab` token ids that reads up to
// `max_tokens`, pooled as `pipeline` says. Refuses a Pooling module whose dimension is not the hidden size.
EmbeddingSpec embedding_spec(const std::filesystem::path& dir, std::string arch, const st::Pipeline& pipeline,
                             int64_t hidden, int64_t vocab, int64_t max_tokens);

// The embedding model of the graph `build` makes from the inputs "ids" and "positions": it returns the final hidden
// states [T, spec.dimension], the output "hidden".
using Build = std::function<Value(Graph& g, const WeightStore& weights, Value ids, Value positions)>;
std::unique_ptr<EmbeddingModel> make_model(EmbeddingSpec spec, std::unique_ptr<WeightStore> weights,
                                           const Build& build);

}  // namespace xs::encoder
