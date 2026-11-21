#include <algorithm>
#include <cmath>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/inkling2/inkling2.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::inkling2 {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

// The released checkpoint's names: everything of the decoder sits under model.llm.
constexpr const char* kPrefix = "model.llm.";
// transformers' Inkling2LoopUnweightedRMSNorm as the block residuals build it (its default, not rms_norm_eps).
constexpr float kUnitEps = 1e-6f;

// The residual stream and the states at the start of each block so far, normed, each [T, 1, hidden].
struct Stream {
    Value h;
    std::vector<Value> blocks;
};

// What a pass after the first hears of the one before it, for one layer: a message per input, [T, hidden] each.
struct Messages {
    Value q, k, v, mlp;
};

// The decoder's program for one step, built over a checkpoint's weights.
class Builder {
public:
    Builder(Graph& g, const Config& cfg, const WeightStore& weights) : g_(g), c_(cfg), weights_(weights) {}

    // The embedded tokens through every layer application and the last block residual: [T, hidden], before the
    // final norm.
    Value decoder() {
        ids_ = g_.input("token_ids", DType::I32, {kT});
        log_scale_ = g_.input("log_scale", DType::F32, {kT, 1, 1});
        Stream s;
        s.h = g_.rms_norm(g_.gather(weight("embed.weight"), ids_), weight("embed_norm.weight"), eps());
        for (int64_t i = 0; i < c_.loop.start; ++i) s = layer(i, 0, s, nullptr);
        const Stream before = s;
        for (int64_t i = c_.loop.start; i < c_.loop.end; ++i) s = layer(i, 0, s, nullptr);
        for (int64_t pass = 1; pass < c_.loop.passes; ++pass) {
            // The pass before: the blocks it appended after its first (the loop's own input), and where it ended.
            const std::vector<Value> heard(s.blocks.begin() + static_cast<std::ptrdiff_t>(before.blocks.size()) + 1,
                                           s.blocks.end());
            const Value ended = s.h;
            s = before;
            for (int64_t i = c_.loop.start; i < c_.loop.end; ++i) {
                const Messages m = messages(i - c_.loop.start, pass, heard, ended);
                s = layer(i, pass, s, &m);
            }
        }
        for (int64_t i = c_.loop.end; i < c_.num_layers; ++i) s = layer(i, 0, s, nullptr);
        Graph::Scope scope(g_, "final_input");
        return mix(s.blocks, s.h, query("attn_res.output_query", 1), 1);
    }

    // rows [n, hidden] -> logits [n, output_size].
    Value head(Value rows) {
        Graph::Scope scope(g_, "head");
        const Value logits = g_.matmul(g_.rms_norm(rows, weight("norm.weight"), eps()), weight("unembed.weight"));
        return g_.softcap(g_.slice(logits, 1, 0, c_.output_size), static_cast<float>(c_.logit_softcap));
    }

private:
    float eps() const { return static_cast<float>(c_.norm_eps); }
    Value weight(const std::string& name) { return g_.weight(weights_, kPrefix + name); }
    Value norm(Value x, const std::string& name) { return g_.rms_norm(x, weight(name), eps()); }
    Value unit(Value x) { return g_.reshape(g_.rms_norm(x, kUnitEps), {kT, 1, c_.hidden_size}); }

    // A float weight's values, read on the host.
    HostTensor host(const std::string& name) const {
        const TensorView& v = weights_.get(kPrefix + name);
        XS_CHECK(is_float(v.dtype), "inkling2: ", kPrefix, name, " is ", dtype_name(v.dtype), ", expected floats");
        const std::vector<float> values = HostTensor::borrow(v.dtype, v.shape, const_cast<void*>(v.data)).to_f32();
        return HostTensor::from_vector(DType::F32, v.shape, values);
    }

    // A block residual's queries [count, hidden]: one is stored as a vector.
    Value query(const std::string& name, int64_t count) {
        const Value q = weight(name);
        const Dims expected = count == 1 ? Dims{c_.hidden_size} : Dims{count, c_.hidden_size};
        XS_CHECK(g_.type(q).shape == expected, "inkling2: ", kPrefix, name, " is ", dims_to_string(g_.type(q).shape),
                 ", expected ", dims_to_string(expected));
        return g_.reshape(q, {count, c_.hidden_size});
    }

    // Inkling2LoopBlockAttentionResidual: each query attends over the normed block states and the normed h, with no
    // projections; its output is their mix. [T, count, hidden], or [T, hidden] for one query.
    Value mix(const std::vector<Value>& blocks, Value h, Value queries, int64_t count) {
        std::vector<Value> states = blocks;
        states.push_back(unit(h));
        const Value stack = states.size() == 1 ? states[0] : g_.concat(states, 1);  // [T, S, hidden]
        const float scale = 1.0f / std::sqrt(static_cast<float>(c_.hidden_size));
        const Value scores = g_.transpose(g_.scale(g_.matmul(stack, queries), scale), {0, 2, 1});  // [T, count, S]
        const Value out = g_.stream_mix(g_.softmax(scores), stack);
        return count == 1 ? g_.reshape(out, {kT, c_.hidden_size}) : out;
    }

    Value part(Value mixed, int64_t i) { return g_.reshape(g_.slice(mixed, 1, i, i + 1), {kT, c_.hidden_size}); }

    // Inkling2LoopMessageAttentionResidual: loop layer `index`'s four queries over the pass before.
    Messages messages(int64_t index, int64_t pass, const std::vector<Value>& heard, Value ended) {
        Graph::Scope scope(g_, "messages." + std::to_string(index) + ".pass" + std::to_string(pass));
        // Stored as (query, mlp) and (key, value) pairs of rows per loop layer.
        const HostTensor a = host("middle_cycle_cross_pass_queries"), b = host("middle_cycle_cross_pass_kv_queries");
        const int64_t loop_layers = c_.loop.end - c_.loop.start, d = c_.hidden_size;
        XS_CHECK(a.shape() == (Dims{2 * loop_layers, d}) && b.shape() == a.shape(),
                 "inkling2: the message queries are ", dims_to_string(a.shape()), " and ", dims_to_string(b.shape()),
                 ", expected ", dims_to_string(Dims{2 * loop_layers, d}), " each");
        HostTensor queries(DType::F32, {4, d});
        const auto row = [&](int64_t to, const HostTensor& from, int64_t r) {
            std::copy_n(from.data<float>() + (2 * index + r) * d, d, queries.data<float>() + to * d);
        };
        row(0, a, 0);
        row(1, b, 0);
        row(2, b, 1);
        row(3, a, 1);
        const Value mixed = mix(heard, ended, g_.constant(std::move(queries)), 4);
        return {part(mixed, 0), part(mixed, 1), part(mixed, 2), part(mixed, 3)};
    }

    // Inkling2LoopMessageGate: (1 - forget) * h + gain * message, both gates from the normed pair.
    Value gate(const std::string& name, Value h, Value message) {
        Graph::Scope scope(g_, "gate");
        const Value both = g_.concat({g_.rms_norm(h, eps()), g_.rms_norm(message, eps())}, 1);
        const Value low = g_.matmul(both, weight(name + "down_weight"));
        const Value forget = g_.tanh(g_.matmul(low, weight(name + "forget_weight")));
        const Value gain = g_.tanh(g_.matmul(low, weight(name + "gain_weight")));
        return g_.add(g_.add(h, g_.scale(g_.mul(forget, h), -1.0f)), g_.mul(gain, message));
    }

    // Inkling2LoopShortConvolution: (conv(x) + x) / sqrt(taps + 1).
    Value short_conv(const std::string& name, const std::string& label, Value x, int64_t channels) {
        const Value w = weight(name);
        XS_CHECK(g_.type(w).shape == (Dims{channels, 1, c_.conv_kernel}), "inkling2: ", kPrefix, name, " is ",
                 dims_to_string(g_.type(w).shape), ", expected ", dims_to_string(Dims{channels, 1, c_.conv_kernel}));
        const int slot = g_.add_conv_slot(channels, c_.conv_kernel, label);
        const Value conv = g_.causal_conv(slot, x, g_.reshape(w, {channels, c_.conv_kernel}));
        return g_.scale(g_.add(conv, x), 1.0f / std::sqrt(static_cast<float>(c_.conv_kernel + 1)));
    }

    // Inkling2LoopDynamicShortConvolution: taps per token and KV head, projected from the projection's own input,
    // beside the stored ones; (conv(x) + x) / sqrt(2 taps + 1).
    Value dynamic_conv(const std::string& name, const std::string& label, Value x, Value input) {
        const int64_t channels = c_.num_kv_heads * c_.head_dim;
        const Value taps =
            g_.reshape(g_.matmul(input, weight(name + "weight_proj")), {kT, c_.num_kv_heads, c_.conv_kernel});
        const int slot = g_.add_conv_slot(channels, c_.conv_kernel, label);
        const Value conv = g_.dynamic_conv(slot, x, weight(name + "static_w"), taps);
        return g_.scale(g_.add(conv, x), 1.0f / std::sqrt(static_cast<float>(2 * c_.conv_kernel + 1)));
    }

    // A key or value projection with its convolution: [T, kv_heads, head_dim].
    Value key_or_value(const std::string& attn, const char* which, const std::string& label, Value input,
                       bool sliding) {
        const std::string conv = attn + which + "_sconv.";
        const std::string slot = label + "." + which + "_conv";
        const Value x = g_.matmul(input, weight(attn + "w" + which + "_dv.weight"));
        const Value y = sliding ? short_conv(conv + "weight", slot, x, c_.num_kv_heads * c_.head_dim)
                                : dynamic_conv(conv, slot, x, input);
        return g_.reshape(y, {kT, c_.num_kv_heads, c_.head_dim});
    }

    // Inkling2LoopSlidingAttention / FullAttention over the normed inputs of the query, key and value projections.
    Value attention(int64_t i, const std::string& label, Value xq, Value xk, Value xv) {
        Graph::Scope scope(g_, "self_attn");
        const bool sliding = c_.sliding[static_cast<size_t>(i)];
        const std::string a = "layers." + std::to_string(i) + ".attn.";
        // Full layers scale queries and relative states with the position, past a floor.
        const auto scaled = [&](Value x) { return sliding ? x : g_.mul(x, log_scale_); };

        Value q = g_.reshape(g_.matmul(xq, weight(a + "wq_du.weight")), {kT, c_.num_heads, c_.head_dim});
        q = scaled(g_.l2_clip(g_.mul(q, weight(a + "q_l2_clip.scale")), static_cast<float>(c_.qk_clip)));
        const Value k = g_.l2_clip(key_or_value(a, "k", label, xk, sliding), static_cast<float>(c_.qk_clip));
        const Value v = g_.rms_norm(key_or_value(a, "v", label, xv, sliding), eps());
        const int kv = g_.add_kv_slot(c_.num_kv_heads, c_.head_dim, sliding ? c_.sliding_window : 0, label);
        const auto [k_all, v_all] = g_.kv_cache(kv, k, v);

        graph::AttentionAttrs attrs;
        attrs.scale = 1.0f / std::sqrt(static_cast<float>(c_.head_dim));
        attrs.sliding_window = sliding ? c_.sliding_window : 0;
        attrs.rel_extent = sliding ? c_.sliding_window : c_.rel_extent;
        // Inkling2LoopRelativePositionBias: each head's relative state mixes the stored profiles of bias by distance.
        Value r = g_.reshape(g_.matmul(xq, weight(a + "wr_du.weight")), {kT, c_.num_heads, c_.rel_width});
        r = scaled(g_.l2_clip(g_.mul(r, weight(a + "rel_l2_clip.scale")), static_cast<float>(c_.rel_clip)));
        const Value rel = g_.matmul(r, profiles(a + "rel_logits_proj.proj", attrs.rel_extent));
        const Value sink = g_.matmul(xq, weight(a + "attn_sink_proj.weight"));

        Value o = g_.attention_relative(q, k_all, v_all, rel, sink, attrs);
        const Value gate = g_.sigmoid(g_.matmul(xq, weight(a + "gate_attn.weight")));
        o = g_.mul(g_.reshape(o, {kT, c_.num_heads * c_.head_dim}), gate);
        return g_.matmul(o, weight(a + "wo_ud.weight"));
    }

    // The profiles [rel_width, extent] as a matmul's weight, [extent, rel_width].
    Value profiles(const std::string& name, int64_t extent) {
        const HostTensor stored = host(name);
        XS_CHECK(stored.shape() == (Dims{c_.rel_width, extent}), "inkling2: ", kPrefix, name, " is ",
                 dims_to_string(stored.shape()), ", expected ", dims_to_string(Dims{c_.rel_width, extent}));
        HostTensor w(DType::F32, {extent, c_.rel_width});
        for (int64_t e = 0; e < extent; ++e) {
            for (int64_t d = 0; d < c_.rel_width; ++d)
                w.data<float>()[e * c_.rel_width + d] = stored.data<float>()[d * extent + e];
        }
        return g_.constant(std::move(w));
    }

    // Inkling2LoopSparseMoeBlock: a sigmoid router whose bias only chooses, experts of a softcapped SwiGLU.
    Value experts(int64_t i, Value x) {
        const std::string m = "layers." + std::to_string(i) + ".mlp.";
        blocks::MoeSpec spec;
        spec.experts = c_.experts;
        spec.top_k = c_.experts_per_token;
        spec.router = blocks::Router::Sigmoid;
        spec.routed_scale = static_cast<float>(c_.route_scale);
        blocks::MoeParts parts;
        parts.router = weight(m + "gate.weight");
        parts.select_bias = weight(m + "gate.bias");
        parts.gate_up = weight(m + "experts.w13_weight");
        parts.interleaved = true;
        parts.down = weight(m + "experts.w2_weight");
        const Dims bank{c_.experts, 2 * c_.expert_width, c_.hidden_size};
        const Dims down{c_.experts, c_.hidden_size, c_.expert_width};
        XS_CHECK(g_.type(parts.gate_up).shape == bank && g_.type(parts.down).shape == down, "inkling2: ", kPrefix, m,
                 "experts.w13_weight and w2_weight are ", dims_to_string(g_.type(parts.gate_up).shape), " and ",
                 dims_to_string(g_.type(parts.down).shape), ", expected ", dims_to_string(bank), " and ",
                 dims_to_string(down));
        parts.hidden = [this](Value gate, Value up) {
            const Value capped = g_.softcap(gate, static_cast<float>(c_.gate_softcap));
            return g_.mul(g_.mul(capped, g_.sigmoid(gate)), g_.softcap(up, static_cast<float>(c_.up_softcap)));
        };
        return blocks::moe(g_, x, spec, parts);
    }

    // One application of layer i: `pass` > 0 in the loop's later passes, which hear `heard` and keep state of
    // their own.
    Stream layer(int64_t i, int64_t pass, Stream s, const Messages* heard) {
        const std::string l = "layers." + std::to_string(i) + ".";
        const std::string label = "layers." + std::to_string(i) + (pass == 0 ? "" : ".pass" + std::to_string(pass));
        Graph::Scope scope(g_, label);
        const bool sliding = c_.sliding[static_cast<size_t>(i)];
        const std::string ctl = l + "controllers.";
        const Value h = s.h;

        Value xq, xk, xv;
        if (sliding) {
            // One mix for the three projections; a message moves each apart before the shared norm.
            const Value x =
                i == 0 ? h : mix(s.blocks, h, query("attn_res.attn_queries." + std::to_string(i - 1), 1), 1);
            if (heard == nullptr) {
                xq = xk = xv = norm(x, l + "attn_norm.weight");
            } else {
                xq = norm(gate(ctl + "attn.", x, heard->q), l + "attn_norm.weight");
                xk = norm(gate(ctl + "k.", x, heard->k), l + "attn_norm.weight");
                xv = norm(gate(ctl + "v.", x, heard->v), l + "attn_norm.weight");
            }
        } else {
            xq = xk = xv = h;
            if (i > 0) {
                const Value mixed = mix(s.blocks, h, query("attn_res.attn_queries." + std::to_string(i - 1), 3), 3);
                xq = part(mixed, 0);
                xk = part(mixed, 1);
                xv = part(mixed, 2);
            }
            if (heard != nullptr) {
                xq = gate(ctl + "attn.", xq, heard->q);
                xk = gate(ctl + "k.", xk, heard->k);
                xv = gate(ctl + "v.", xv, heard->v);
            }
            xq = norm(xq, l + "q_input_norm.weight");
            xk = norm(xk, l + "k_input_norm.weight");
            xv = norm(xv, l + "v_input_norm.weight");
        }
        Value o = attention(i, label, xq, xk, xv);
        o = short_conv(l + "attn_sconv.weight", label + ".attn_conv", o, c_.hidden_size);

        // A block's first layer hands its input to the block states instead of adding it back.
        Value residual;
        if (i % c_.block_layers == 0) {
            s.blocks.push_back(unit(h));
            residual = o;
        } else {
            residual = g_.add(h, o);
        }
        Value m = mix(s.blocks, residual, query("attn_res.mlp_queries." + std::to_string(i), 1), 1);
        if (heard != nullptr) m = gate(ctl + "mlp.", m, heard->mlp);
        Value y = experts(i, norm(m, l + "mlp_norm.weight"));
        y = short_conv(l + "mlp_sconv.weight", label + ".mlp_conv", y, c_.hidden_size);
        s.h = g_.add(residual, y);
        return s;
    }

    Graph& g_;
    const Config& c_;
    const WeightStore& weights_;
    Value ids_, log_scale_;
};

// The experts of a released checkpoint are NVFP4 (U8 codes) or MXFP8 (F8 codes) beside their scales; a pack's, or a
// float checkpoint's, are weights already.
void pair_experts(SafetensorsWeights& weights, const Config& cfg) {
    for (int64_t i = 0; i < cfg.num_layers; ++i) {
        for (const char* bank : {"w13_weight", "w2_weight"}) {
            const std::string name = std::string(kPrefix) + "layers." + std::to_string(i) + ".mlp.experts." + bank;
            const TensorView& v = weights.get(name);
            if (v.dtype == DType::U8) weights.pair_nvfp4(name);
            if (v.dtype == DType::F8) weights.pair_mxfp8(name);
        }
    }
}

class Inkling2Model final : public Model {
public:
    Inkling2Model(Config cfg, ModelInfo info, std::unique_ptr<WeightStore> weights)
        : cfg_(std::move(cfg)), info_(std::move(info)), weights_(std::move(weights)) {
        Builder b(step_, cfg_, *weights_);
        const Value h = b.decoder();
        const Value logits = b.head(step_.slice(h, 0, -1));
        step_.output("logits", logits);
        step_.output("argmax", blocks::argmax_rows(step_, logits));
        step_.validate();
        log::debug("inkling2: graph has ", step_.nodes().size(), " nodes, ", step_.kv_slots().size(), " kv and ",
                   step_.conv_slots().size(), " conv slots over ", cfg_.applications(), " layer applications");
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return step_; }
    const WeightStore& weights() const override { return *weights_; }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        XS_CHECK(embeddings.empty(), "inkling2: image input is not supported yet");
        const auto n = static_cast<int64_t>(tokens.size());
        XS_CHECK(n > 0, "inkling2: a step needs at least one token");
        HostTensor ids(DType::I32, {n});
        HostTensor scale(DType::F32, {n, 1, 1});
        for (int64_t i = 0; i < n; ++i) {
            const int32_t t = tokens[static_cast<size_t>(i)];
            if (t < 0 || t >= cfg_.vocab_size) {
                fail("inkling2: token id ", t, " outside the vocabulary (", cfg_.vocab_size, ")");
            }
            ids.data<int32_t>()[i] = t;
            const double past_floor = std::max(static_cast<double>(start_pos + i + 1) / cfg_.log_scale_floor, 1.0);
            scale.data<float>()[i] = static_cast<float>(1.0 + cfg_.log_scale_alpha * std::log(past_floor));
        }
        inputs["token_ids"] = std::move(ids);
        inputs["log_scale"] = std::move(scale);
    }

private:
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
    info.arch = "inkling2";
    info.id = cfg::model_id(dir);
    info.vocab_size = cfg.output_size;
    info.hidden_size = cfg.hidden_size;
    info.max_position = cfg.max_position;
    info.dialect = "plain";

    std::vector<int32_t> stops;
    info.default_sampling = cfg::read_generation_defaults(dir, &stops);
    const auto config_eos = cfg::get_int_list(config_json, "eos_token_id", "config.json");
    stops.insert(stops.end(), config_eos.begin(), config_eos.end());
    if (const auto eos = cfg::tokenizer_eos_id(dir)) stops.push_back(*eos);
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    XS_CHECK(!stops.empty(), "inkling2: no end-of-sequence token in config.json or generation_config.json");
    info.stop_token_ids = std::move(stops);

    auto weights = SafetensorsWeights::open_dir(dir, SafetensorsWeights::Fp8::AsStored);
    pair_experts(*weights, cfg);
    return std::make_unique<Inkling2Model>(std::move(cfg), std::move(info), std::move(weights));
}

}  // namespace xs::inkling2
