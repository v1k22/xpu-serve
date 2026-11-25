#include "models/drafters/dspark.h"

#include <algorithm>
#include <cmath>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "util/error.h"

namespace xs::dspark {

using graph::Graph;
using graph::kT;
using graph::Value;

Config parse_config(const nlohmann::json& m, int64_t hidden, int64_t visits) {
    const std::string where = "config.json mtp_config";
    if (cfg::get_bool_or(m, "q_bias", false, where)) fail(where, ": q_bias is not supported");
    if (cfg::get_bool_or(m, "o_bias", false, where)) fail(where, ": o_bias is not supported");
    const std::string head = cfg::get_string(m, "markov_head_type", where);
    if (head != "rnn_hidden") fail(where, ": markov_head_type '", head, "' is not supported (rnn_hidden is)");

    Config c;
    c.hidden_size = cfg::get_int(m, "hidden_size", where);
    c.intermediate_size = cfg::get_int(m, "intermediate_size", where);
    c.num_layers = cfg::get_int(m, "num_hidden_layers", where);
    c.num_heads = cfg::get_int(m, "num_attention_heads", where);
    c.num_kv_heads = cfg::get_int(m, "num_key_value_heads", where);
    c.head_dim = cfg::get_int(m, "head_dim", where);
    c.block_size = cfg::get_int(m, "block_size", where);
    c.markov_rank = cfg::get_int(m, "markov_rank", where);
    c.rms_norm_eps = cfg::get_float(m, "rms_norm_eps", where);
    c.rope_theta = cfg::get_float(m, "rope_theta", where);
    XS_CHECK(c.hidden_size == hidden, where, ": hidden_size ", c.hidden_size, " is not the target's ", hidden);
    XS_CHECK(c.num_layers > 0 && c.intermediate_size > 0 && c.num_kv_heads > 0 && c.head_dim > 0 &&
                 c.head_dim % 2 == 0 && c.num_heads % c.num_kv_heads == 0 && c.block_size >= 2 && c.markov_rank > 0 &&
                 c.rope_theta > 0,
             where, ": sizes do not fit (", c.num_layers, " layers, ", c.num_heads, " heads over ", c.num_kv_heads,
             " of ", c.head_dim, ", a block of ", c.block_size, ", rank ", c.markov_rank, ")");

    const int64_t window = cfg::get_int(m, "sliding_window_size", where);
    c.windows.assign(static_cast<size_t>(c.num_layers), 0);
    for (const int32_t l : cfg::get_int_list(m, "local_layer_ids", where)) {
        XS_CHECK(l >= 0 && l < c.num_layers && window > 0, where, ": local_layer_ids names layer ", l, " of ",
                 c.num_layers, " with a window of ", window);
        c.windows[static_cast<size_t>(l)] = window;
    }
    for (const int32_t id : cfg::get_int_list(m, "target_layer_ids", where)) c.target_ids.push_back(id);
    // The last tap is the target's final state, the others the input of a layer visit.
    XS_CHECK(c.target_ids.size() >= 2 && std::is_sorted(c.target_ids.begin(), c.target_ids.end()) &&
                 c.target_ids.front() >= 0 && c.target_ids.back() == visits &&
                 c.target_ids[c.target_ids.size() - 2] < visits,
             where, ": target_layer_ids must rise to the target's ", visits, " layer visits");
    return c;
}

Drafter::Drafter(Config config, const WeightStore& weights, std::string prefix, int64_t drafts)
    : cfg_(std::move(config)), weights_(weights), prefix_(std::move(prefix)) {
    drafts_ = drafts > 0 ? std::min(cfg_.block_size - 1, drafts) : cfg_.block_size - 1;
    alpha_ = mixing_weights();
    folded_ = weights_.contains(prefix_ + "markov_head.fold.weight");
    XS_CHECK(folded_ != weights_.contains(prefix_ + "markov_head.hidden_out.weight"),
             "dspark: the head needs one of markov_head.hidden_out (the checkpoint's) and markov_head.fold (its fold ",
             "through the output layer, llama.cpp's)");
}

Value Drafter::weight(Graph& g, const std::string& name) const {
    return g.weight(weights_, prefix_ + name);
}

std::vector<float> Drafter::values(const std::string& name, Dims& shape) const {
    const TensorView& t = weights_.get(prefix_ + name);
    XS_CHECK(is_float(t.dtype), "dspark: ", name, " is ", dtype_name(t.dtype), ", expected floats");
    shape = t.shape;
    return HostTensor::borrow(t.dtype, t.shape, const_cast<void*>(t.data)).to_f32();
}

// A bias, read on the host: [1, n].
Value Drafter::bias(Graph& g, const std::string& name) const {
    Dims shape;
    std::vector<float> v = values(name, shape);
    XS_CHECK(shape.size() == 1, "dspark: ", name, " is ", dims_to_string(shape), ", expected a vector");
    return g.constant(HostTensor::from_vector(DType::F32, {1, shape[0]}, std::move(v)));
}

std::vector<float> Drafter::mixing_weights() const {
    const auto taps = static_cast<int64_t>(cfg_.target_ids.size());
    std::vector<float> alpha;
    Dims shape;
    if (weights_.contains(prefix_ + "fuse_weight")) {
        alpha = values("fuse_weight", shape);
        const Dims want{cfg_.num_layers + 1, taps};
        XS_CHECK(shape == want, "dspark: fuse_weight is ", dims_to_string(shape), ", expected ", dims_to_string(want),
                 " (a row per layer and the head's, a column per target state)");
    } else {  // a row per reader
        for (int64_t r = 0; r <= cfg_.num_layers; ++r) {
            const std::string name =
                r < cfg_.num_layers ? "layers." + std::to_string(r) + ".fuse_weight" : "markov_head.fuse_weight";
            const std::vector<float> row = values(name, shape);
            XS_CHECK(shape == Dims{taps}, "dspark: ", name, " is ", dims_to_string(shape), ", expected [", taps,
                     "] (a column per target state)");
            alpha.insert(alpha.end(), row.begin(), row.end());
        }
    }
    for (int64_t r = 0; r <= cfg_.num_layers; ++r) {
        float* row = alpha.data() + r * taps;
        const float top = *std::max_element(row, row + taps);
        double sum = 0;
        for (int64_t j = 0; j < taps; ++j) sum += std::exp(static_cast<double>(row[j] - top));
        for (int64_t j = 0; j < taps; ++j) {
            row[j] = static_cast<float>(std::exp(static_cast<double>(row[j] - top)) / sum);
        }
    }
    return alpha;
}

std::vector<float> Drafter::mask_vector() const {
    Dims shape;
    std::vector<float> v = values("mask_embed", shape);
    const int64_t h = cfg_.hidden_size;
    XS_CHECK(shape == Dims{h} || shape == (Dims{cfg_.block_size - 1, h}), "dspark: mask_embed is ",
             dims_to_string(shape), ", expected [", h, "] or a copy per draft, [", cfg_.block_size - 1, ", ", h, "]");
    for (size_t i = static_cast<size_t>(h); i < v.size(); ++i) {
        XS_CHECK(v[i] == v[i % static_cast<size_t>(h)], "dspark: mask_embed's rows differ (row ", i / h,
                 "): a mask vector per draft is not supported");
    }
    v.resize(static_cast<size_t>(h));
    return v;
}

Value Drafter::norm(Graph& g, Value x, const std::string& name) const {
    return g.rms_norm(x, weight(g, name), static_cast<float>(cfg_.rms_norm_eps));
}

Value Drafter::inv_freq(Graph& g) const {
    return g.constant(blocks::rope_inv_freq(cfg_.rope_theta, cfg_.head_dim, cfg_.head_dim / 2, cfg_.head_dim / 2));
}

std::string Drafter::slot_label(int64_t layer) const {
    return prefix_ + "layers." + std::to_string(layer);
}

std::vector<Value> Drafter::mixes(Graph& g, std::span<const Value> taps) const {
    const auto count = static_cast<int64_t>(taps.size());
    XS_CHECK(count == static_cast<int64_t>(cfg_.target_ids.size()), "dspark: ", count, " target states for ",
             cfg_.target_ids.size(), " ids");
    std::vector<Value> stack;
    for (const Value tap : taps) stack.push_back(g.reshape(tap, {kT, 1, cfg_.hidden_size}));
    const Value by_state = g.transpose(g.concat(stack, 1), {0, 2, 1});  // [T, hidden, states]
    const Value mixed =
        g.matmul(by_state, g.constant(HostTensor::from_vector(DType::F32, {cfg_.num_layers + 1, count}, alpha_)));
    std::vector<Value> out;
    for (int64_t r = 0; r <= cfg_.num_layers; ++r) {
        out.push_back(g.reshape(g.slice(mixed, 2, r, r + 1), {kT, cfg_.hidden_size}));
    }
    return out;
}

Value Drafter::append_context(Graph& g, std::span<const Value> taps, Value positions) const {
    Graph::Scope scope(g, "drafter.context");
    const std::vector<Value> mixed = mixes(g, taps);
    const Value freq = inv_freq(g);
    const Dims heads{kT, cfg_.num_kv_heads, cfg_.head_dim};
    for (int64_t l = 0; l < cfg_.num_layers; ++l) {
        const std::string L = "layers." + std::to_string(l) + ".";
        const std::string a = L + "self_attn.";
        const Value context = norm(g, mixed[static_cast<size_t>(l)], L + "dflare_norm.weight");
        const Value keys = g.reshape(g.matmul(context, weight(g, a + "k_proj_ctx.weight")), heads);
        const Value k = g.rope(norm(g, keys, a + "k_norm_ctx.weight"), positions, freq);
        const Value v = g.reshape(g.matmul(context, weight(g, a + "v_proj_ctx.weight")), heads);
        const int64_t window = cfg_.windows[static_cast<size_t>(l)];
        g.kv_cache(g.add_kv_slot(cfg_.num_kv_heads, cfg_.head_dim, window > 0 ? window + cfg_.block_size : 0,
                                 slot_label(l)),
                   k, v);
    }
    return norm(g, mixed.back(), "markov_head.dflare_norm.weight");
}

void Drafter::build_block(Graph& g, const Target& target) const {
    const int64_t rows = cfg_.block_size, hidden = cfg_.hidden_size, rank = cfg_.markov_rank;
    const Value ids = g.input("token_ids", DType::I32, {kT});
    const Value pos = g.input("positions", DType::I32, {kT});
    const Value context = g.input(kContextRow, DType::F32, {1, hidden});
    const Value freq = inv_freq(g);
    const auto one = [&] { return g.constant(HostTensor::from_vector(DType::F32, {1, 1}, std::vector<float>{1.0f})); };

    // The pending token's row is its embedding as the target norms it; every other row is the mask vector.
    const Value embedded =
        g.rms_norm(g.gather(target.embed, ids), target.embed_norm, static_cast<float>(cfg_.rms_norm_eps));
    const Value rest = g.prev_row(g.add(g.scale(g.slice(embedded, 1, 0, 1), 0.0f), one()));  // 0 on the first row
    const Value first = g.add(g.scale(rest, -1.0f), one());
    const Value mask = g.constant(HostTensor::from_vector(DType::F32, {1, hidden}, mask_vector()));
    Value h = g.add(g.mul(embedded, first), g.mul(mask, rest));

    for (int64_t l = 0; l < cfg_.num_layers; ++l) {
        const std::string name = "layers." + std::to_string(l);
        Graph::Scope scope(g, "drafter." + name);
        const std::string L = name + ".";
        const std::string a = L + "self_attn.";
        blocks::AttentionSpec s;
        s.heads = cfg_.num_heads;
        s.kv_heads = cfg_.num_kv_heads;
        s.head_dim = cfg_.head_dim;
        s.attrs.scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
        s.attrs.causal = false;
        // A local layer's block sees the window before the block's start, the same keys for every row.
        if (const int64_t window = cfg_.windows[static_cast<size_t>(l)]; window > 0) {
            s.attrs.sliding_window = window;
            s.attrs.anchored = true;
            s.kv_window = window + rows;
        }
        s.kv_label = slot_label(l);
        blocks::AttentionParts p;
        p.q = blocks::matmul_by(g, weight(g, a + "q_proj.weight"));
        p.k = blocks::matmul_by(g, weight(g, a + "k_proj.weight"));
        p.v = blocks::matmul_by(g, weight(g, a + "v_proj.weight"));
        p.o = blocks::matmul_by(g, weight(g, a + "o_proj.weight"));
        p.q_norm = [this, &g, a](Value v) { return norm(g, v, a + "q_norm.weight"); };
        p.k_norm = [this, &g, a](Value v) { return norm(g, v, a + "k_norm.weight"); };
        h = g.add(h, blocks::attention(g, s, p, norm(g, h, L + "input_layernorm.weight"), pos, freq).out);
        const Value y = norm(g, h, L + "post_attention_layernorm.weight");
        h = g.add(h, blocks::gated_mlp(g, y, blocks::matmul_by(g, weight(g, L + "mlp.gate_proj.weight")),
                                       blocks::matmul_by(g, weight(g, L + "mlp.up_proj.weight")),
                                       blocks::matmul_by(g, weight(g, L + "mlp.down_proj.weight")), blocks::Act::Silu));
    }

    // The head: a gated recurrent step per draft over (its state, the token before, the slot's row, the target's
    // context), whose latent corrects the row before the target's output layer reads it. The softcap the target
    // puts on its logits keeps their order, so the picks do not need it. Each pick reads the whole output layer: a
    // draft that follows a wrong one is lost, and a round is a few drafts beside a verify that costs far more. With
    // the correction folded through the output layer, the output layer reads the rows once and each pick adds its
    // latent's logits (logits of the row + fold x latent + fold's bias: the same sum).
    Graph::Scope scope(g, "drafter.head");
    const Value slots = g.slice(norm(g, h, "norm.weight"), 0, -(rows - 1));  // [rows - 1, hidden]: the masks' rows
    const Value table = weight(g, "markov_head.markov_w1.weight");
    const Value step_w = weight(g, "markov_head.rnn_proj.weight");
    const Value step_b = bias(g, "markov_head.rnn_proj.bias");
    const std::string out = folded_ ? "markov_head.fold." : "markov_head.hidden_out.";
    const Value out_w = weight(g, out + "weight");
    const Value out_b = bias(g, out + "bias");
    const Dims out_shape = folded_ ? Dims{target.tokens, rank} : Dims{hidden, rank};
    XS_CHECK(g.type(step_w).shape == (Dims{3 * rank, 2 * rank + 2 * hidden}) && g.type(out_w).shape == out_shape &&
                 g.type(table).shape.back() == rank,
             "dspark: the head's rnn_proj, ", out, "weight and markov_w1 are ", dims_to_string(g.type(step_w).shape),
             ", ", dims_to_string(g.type(out_w).shape), " and ", dims_to_string(g.type(table).shape), " for rank ",
             rank, ", hidden ", hidden, " and ", target.tokens, " tokens");
    const Value row_logits = folded_ ? g.slice(g.matmul(g.slice(slots, 0, 0, drafts_), target.head), 1, 0,
                                               target.tokens)
                                     : Value{};
    Value previous = g.slice(g.slice(ids, 0, -rows), 0, 0, 1);  // the pending token
    Value state = g.constant(HostTensor(DType::F32, {1, rank}));
    std::vector<Value> drafts;
    for (int64_t i = 0; i < drafts_; ++i) {
        const Value row = g.slice(slots, 0, i, i + 1);
        const Value z = g.add(g.matmul(g.concat({state, g.gather(table, previous), row, context}, 1), step_w), step_b);
        const Value gate = g.sigmoid(g.slice(z, 1, 0, rank));
        const Value candidate = g.tanh(g.slice(z, 1, rank, 2 * rank));
        const Value latent = g.tanh(g.slice(z, 1, 2 * rank, 3 * rank));
        state = g.add(candidate, g.mul(gate, g.add(state, g.scale(candidate, -1.0f))));  // gate s + (1 - gate) n
        const Value correction = g.add(g.matmul(latent, out_w), out_b);
        const Value logits = folded_ ? g.add(g.slice(row_logits, 0, i, i + 1), correction)
                                     : g.slice(g.matmul(g.add(row, correction), target.head), 1, 0, target.tokens);
        previous = blocks::argmax_rows(g, logits);
        drafts.push_back(previous);
    }
    g.output("drafts", drafts.size() == 1 ? drafts[0] : g.concat(drafts, 0));
    g.validate();
}

void Drafter::prepare_block(int32_t token, int64_t position, TensorMap& inputs) const {
    HostTensor ids(DType::I32, {cfg_.block_size});
    HostTensor pos(DType::I32, {cfg_.block_size});
    for (int64_t i = 0; i < cfg_.block_size; ++i) {
        ids.data<int32_t>()[i] = token;  // read on the first row only
        pos.data<int32_t>()[i] = static_cast<int32_t>(position + i);
    }
    inputs["token_ids"] = std::move(ids);
    inputs["positions"] = std::move(pos);
}

}  // namespace xs::dspark
