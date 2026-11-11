#include <algorithm>
#include <cmath>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/lfm3/lfm3.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::lfm3 {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

constexpr const char* kPrefix = "model.";

class Lfm3Model final : public Model {
public:
    Lfm3Model(Config cfg, ModelInfo info, std::unique_ptr<WeightStore> weights)
        : cfg_(std::move(cfg)), info_(std::move(info)), weights_(std::move(weights)) {
        build(step_);
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return step_; }
    const WeightStore& weights() const override { return *weights_; }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        XS_CHECK(embeddings.empty(), "lfm3: image input is not supported");
        const auto n = static_cast<int64_t>(tokens.size());
        XS_CHECK(n > 0, "lfm3: a step needs at least one token");
        HostTensor ids(DType::I32, {n});
        HostTensor pos(DType::I32, {n});
        for (int64_t i = 0; i < n; ++i) {
            const int32_t t = tokens[static_cast<size_t>(i)];
            if (t < 0 || t >= cfg_.vocab_size) {
                fail("lfm3: token id ", t, " outside the vocabulary (", cfg_.vocab_size, ")");
            }
            ids.data<int32_t>()[i] = t;
            pos.data<int32_t>()[i] = static_cast<int32_t>(start_pos + i);
        }
        inputs["token_ids"] = std::move(ids);
        inputs["positions"] = std::move(pos);
    }

private:
    Value weight(Graph& g, const std::string& name) const { return g.weight(*weights_, name); }
    Value norm(Graph& g, Value x, const std::string& name) const {
        return g.rms_norm(x, weight(g, name), static_cast<float>(cfg_.norm_eps));
    }
    // A projection whose weight node is made when the block uses it.
    blocks::Linear proj(Graph& g, const std::string& name) const {
        return [this, &g, name](Value x) { return g.matmul(x, weight(g, name)); };
    }
    // The frequencies of one attention kind; invalid when the model has no layer of it.
    Value inv_freq(Graph& g, const Rope& rope) const {
        if (rope.pairs == 0) return {};
        return g.constant(blocks::rope_inv_freq(rope.theta, cfg_.head_dim, rope.pairs, cfg_.head_dim / 2));
    }

    void build(Graph& g) const {
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        const std::string embedding = std::string(kPrefix) + "embed_tokens.weight";
        Value h = g.gather(weight(g, embedding), ids);
        const Value full_freq = inv_freq(g, cfg_.full_rope);
        const Value sliding_freq = inv_freq(g, cfg_.sliding_rope);
        for (int64_t i = 0; i < cfg_.num_layers; ++i) {
            const std::string label = "layers." + std::to_string(i);
            const LayerKind kind = cfg_.layer_kinds[static_cast<size_t>(i)];
            h = decoder_layer(g, kPrefix + label + ".", label, kind, h, pos,
                              kind == LayerKind::Sliding ? sliding_freq : full_freq);
        }
        Graph::Scope scope(g, "head");
        const Value last = norm(g, g.slice(h, 0, -1), std::string(kPrefix) + "embedding_norm.weight");
        const Value logits = g.matmul(last, weight(g, embedding));  // the head is the embedding table
        g.output("logits", logits);
        g.output("argmax", blocks::argmax_rows(g, logits));
        g.validate();
        log::debug("lfm3: graph has ", g.nodes().size(), " nodes, ", g.kv_slots().size(), " kv and ",
                   g.conv_slots().size(), " conv slots");
    }

    // Lfm3DecoderLayer: the operator and the MLP each read a normed stream and are normed again before joining it.
    Value decoder_layer(Graph& g, const std::string& L, const std::string& label, LayerKind kind, Value h, Value pos,
                        Value freq) const {
        Graph::Scope s(g, label);
        const Value x = norm(g, h, L + "operator_norm.weight");
        const Value mixed =
            kind == LayerKind::Conv ? short_conv(g, L, label, x) : attention(g, L, label, kind, x, pos, freq);
        h = g.add(h, norm(g, mixed, L + "post_operator_norm.weight"));
        const Value y = norm(g, h, L + "ffn_norm.weight");
        const std::string M = L + "feed_forward.";  // w2(silu(w1 x) * w3 x)
        const Value mlp = blocks::gated_mlp(g, y, proj(g, M + "w1.weight"), proj(g, M + "w3.weight"),
                                            proj(g, M + "w2.weight"), blocks::Act::Silu);
        return g.add(h, norm(g, mlp, L + "post_ffn_norm.weight"));
    }

    // Lfm3ShortConv: in_proj gives [B | C | x]; out_proj(C * conv(B * x)), a causal depthwise conv over conv_kernel
    // tokens.
    Value short_conv(Graph& g, const std::string& layer, const std::string& label, Value x) const {
        Graph::Scope s(g, "conv");
        const std::string L = layer + "conv.";
        const int64_t width = cfg_.hidden_size, kernel = cfg_.conv_kernel;
        const Value in_w = weight(g, L + "in_proj.weight");
        const Value conv_w = weight(g, L + "conv.weight");
        XS_CHECK(g.type(in_w).shape == (Dims{3 * width, width}) && g.type(conv_w).shape == (Dims{width, 1, kernel}),
                 "lfm3: ", L, "in_proj.weight and conv.weight have shapes ", dims_to_string(g.type(in_w).shape),
                 " and ", dims_to_string(g.type(conv_w).shape), ", expected [", 3 * width, ", ", width, "] and [",
                 width, ", 1, ", kernel, "]");
        const Value bcx = g.matmul(x, in_w);
        const Value b = g.slice(bcx, 1, 0, width);
        const Value c = g.slice(bcx, 1, width, 2 * width);
        const Value gated = g.mul(b, g.slice(bcx, 1, 2 * width, 3 * width));
        const int slot = g.add_conv_slot(width, kernel, label);
        const Value conv = g.causal_conv(slot, gated, g.reshape(conv_w, {width, kernel}));
        return g.matmul(g.mul(c, conv), weight(g, L + "out_proj.weight"));
    }

    // Lfm3Attention: q/k norms per head, RoPE, and each head's output scaled by sigmoid(gate_proj(x)).
    Value attention(Graph& g, const std::string& layer, const std::string& label, LayerKind kind, Value x, Value pos,
                    Value freq) const {
        Graph::Scope s(g, "self_attn");
        const std::string L = layer + "self_attn.";
        blocks::AttentionSpec spec;
        spec.heads = cfg_.num_heads;
        spec.kv_heads = cfg_.num_kv_heads;
        spec.head_dim = cfg_.head_dim;
        spec.attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg_.head_dim)));
        spec.attrs.sliding_window = kind == LayerKind::Sliding ? cfg_.sliding_window : 0;
        spec.kv_window = spec.attrs.sliding_window;
        spec.kv_label = label;
        blocks::AttentionParts parts;
        parts.q = proj(g, L + "q_proj.weight");
        parts.k = proj(g, L + "k_proj.weight");
        parts.v = proj(g, L + "v_proj.weight");
        parts.o = proj(g, L + "o_proj.weight");
        parts.q_norm = [this, &g, L](Value q) { return norm(g, q, L + "q_norm.weight"); };
        parts.k_norm = [this, &g, L](Value k) { return norm(g, k, L + "k_norm.weight"); };
        parts.head_gate = proj(g, L + "gate_proj.weight");
        return blocks::attention(g, spec, parts, x, pos, freq).out;
    }

    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<WeightStore> weights_;
    graph::Graph step_;
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& /*options*/) {
    Config cfg = parse_config(config_json);

    ModelInfo info;
    info.arch = "lfm3";
    info.id = cfg::model_id(dir);
    info.vocab_size = cfg.vocab_size;
    info.hidden_size = cfg.hidden_size;
    info.max_position = cfg.max_position;
    // The chat template writes a turn as Qwen 3.5's does: <think>...</think>, then content and <tool_call> blocks
    // of <function=NAME> with <parameter=KEY> values.
    info.dialect = "qwen3_5";

    std::vector<int32_t> stops;
    info.default_sampling = cfg::read_generation_defaults(dir, &stops);
    const auto config_eos = cfg::get_int_list(config_json, "eos_token_id", "config.json");
    stops.insert(stops.end(), config_eos.begin(), config_eos.end());
    if (const auto eos = cfg::tokenizer_eos_id(dir)) stops.push_back(*eos);  // <|im_end|>: the end of a turn
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    XS_CHECK(!stops.empty(), "lfm3: no end-of-sequence token in config.json or generation_config.json");
    info.stop_token_ids = std::move(stops);

    return std::make_unique<Lfm3Model>(std::move(cfg), std::move(info), SafetensorsWeights::open_dir(dir));
}

}  // namespace xs::lfm3
