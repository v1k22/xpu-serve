#include "models/common/encoder.h"

#include <cmath>

#include "models/config_util.h"
#include "quant/quant.h"
#include "util/error.h"

namespace xs::encoder {
namespace {

using graph::kT;

class EncoderModel final : public EmbeddingModel {
public:
    EncoderModel(EmbeddingSpec spec, std::unique_ptr<WeightStore> weights, const Build& build)
        : spec_(std::move(spec)), weights_(std::move(weights)) {
        const Value ids = graph_.input("ids", DType::I32, {kT});
        const Value positions = graph_.input("positions", DType::I32, {kT});
        const Value hidden = build(graph_, *weights_, ids, positions);
        XS_CHECK(graph_.type(hidden).shape == (Dims{kT, spec_.dimension}), spec_.arch, ": the encoder's output is ",
                 dims_to_string(graph_.type(hidden).shape), ", expected [T, ", spec_.dimension, "]");
        graph_.output("hidden", hidden);
        graph_.validate();
        XS_CHECK(!graph_.stateful(), spec_.arch, ": an encoder keeps no state between runs");
    }

    const EmbeddingSpec& spec() const override { return spec_; }
    const WeightStore& weights() const override { return *weights_; }
    const graph::Graph& graph() const override { return graph_; }

    TensorMap prepare_inputs(std::span<const int32_t> ids) const override {
        const auto tokens = static_cast<int64_t>(ids.size());
        XS_CHECK(tokens >= 1 && tokens <= spec_.max_tokens, spec_.id, " reads 1 to ", spec_.max_tokens,
                 " tokens, got ", tokens);
        HostTensor id_tensor(DType::I32, {tokens});
        HostTensor positions(DType::I32, {tokens});
        for (int64_t t = 0; t < tokens; ++t) {
            const int32_t id = ids[static_cast<size_t>(t)];
            XS_CHECK(id >= 0 && id < spec_.vocab_size, "token id ", id, " is outside ", spec_.id, "'s vocabulary of ",
                     spec_.vocab_size);
            id_tensor.data<int32_t>()[t] = id;
            positions.data<int32_t>()[t] = static_cast<int32_t>(t);
        }
        TensorMap inputs;
        inputs.emplace("ids", std::move(id_tensor));
        inputs.emplace("positions", std::move(positions));
        return inputs;
    }

private:
    EmbeddingSpec spec_;
    std::unique_ptr<WeightStore> weights_;
    graph::Graph graph_;
};

}  // namespace

Value layer_norm(Graph& g, const WeightStore& weights, Value x, const std::string& stem, float eps) {
    return g.layer_norm(x, g.weight(weights, stem + ".weight"), g.weight(weights, stem + ".bias"), eps);
}

Value table_row(Graph& g, const WeightStore& weights, const std::string& name, int64_t row) {
    const TensorView& table = weights.get(name);
    XS_CHECK(table.shape.size() == 2 && row >= 0 && row < table.shape[0], "'", name, "' ",
             dims_to_string(table.shape), " has no row ", row);
    HostTensor out(DType::F32, {table.shape[1]});
    quant::read_rows(table, row, 1, out.data<float>());
    return g.constant(std::move(out));
}

blocks::AttentionSpec attention_spec(int64_t heads, int64_t head_dim) {
    blocks::AttentionSpec s;
    s.heads = heads;
    s.kv_heads = heads;
    s.head_dim = head_dim;
    s.attrs.causal = false;
    s.attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    s.kv_cache = false;
    return s;
}

Value post_norm_layer(Graph& g, const WeightStore& weights, const Layer& layer, Value x, float eps) {
    x = layer_norm(g, weights, g.add(x, layer.attention(x)), layer.norm1, eps);
    return layer_norm(g, weights, g.add(x, layer.mlp(x)), layer.norm2, eps);
}

EmbeddingSpec embedding_spec(const std::filesystem::path& dir, std::string arch, const st::Pipeline& pipeline,
                             int64_t hidden, int64_t vocab, int64_t max_tokens) {
    if (pipeline.dimension != hidden) {
        fail("the Pooling module's embedding dimension ", pipeline.dimension, " is not the encoder's hidden size ",
             hidden, " (one pooling mode keeps the hidden size)");
    }
    EmbeddingSpec spec;
    spec.id = cfg::model_id(dir);
    spec.arch = std::move(arch);
    spec.dimension = hidden;
    spec.max_tokens = max_tokens;
    spec.vocab_size = vocab;
    spec.pooling = pipeline.pooling;
    spec.normalize = pipeline.normalize;
    return spec;
}

std::unique_ptr<EmbeddingModel> make_model(EmbeddingSpec spec, std::unique_ptr<WeightStore> weights,
                                           const Build& build) {
    return std::make_unique<EncoderModel>(std::move(spec), std::move(weights), build);
}

}  // namespace xs::encoder
