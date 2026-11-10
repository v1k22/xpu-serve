#include "models/drafters/dflash.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>

#include <nlohmann/json.hpp>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "util/error.h"
#include "util/half.h"

namespace xs::dflash {

namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

const char* const kWhere = "drafter config.json";

// The stacked matrices of a checkpoint that the graph reads one at a time.
const char* const kStacked[] = {"selected_prefix_bridge.early_read_weights", "selected_prefix_bridge.feature_write_weights"};

// A checkpoint's tensors, and each matrix of a stacked one ([count, rows, cols]) as a view under "<name>.<index>".
class StackedViews final : public WeightStore {
public:
    explicit StackedViews(std::unique_ptr<WeightStore> inner) : inner_(std::move(inner)) {
        for (const char* name : kStacked) {
            const TensorView* t = inner_->find(name);
            if (t == nullptr) continue;
            XS_CHECK(t->shape.size() == 3 && is_float(t->dtype), "drafter: '", name, "' is ", dtype_name(t->dtype),
                     dims_to_string(t->shape), ", expected a stack of float matrices");
            const size_t bytes = static_cast<size_t>(t->shape[1] * t->shape[2]) * dtype_size(t->dtype);
            for (int64_t i = 0; i < t->shape[0]; ++i) {
                views_[std::string(name) + "." + std::to_string(i)] =
                    TensorView{t->dtype, {t->shape[1], t->shape[2]},
                               static_cast<const char*>(t->data) + static_cast<size_t>(i) * bytes, {}};
            }
        }
    }

    const TensorView* find(const std::string& name) const override {
        const auto it = views_.find(name);
        return it != views_.end() ? &it->second : inner_->find(name);
    }
    std::vector<std::string> names() const override {
        std::vector<std::string> out = inner_->names();
        for (const auto& [name, view] : views_) out.push_back(name);
        return out;
    }

private:
    std::unique_ptr<WeightStore> inner_;
    std::map<std::string, TensorView> views_;
};

// z-lab/dflash build_target_layer_ids: the layers a checkpoint reads when its config does not list them.
std::vector<int64_t> default_target_layers(int64_t target_layers, int64_t draft_layers) {
    if (draft_layers == 1) return {target_layers / 2};
    std::vector<int64_t> ids;
    const double span = static_cast<double>(target_layers - 3 - 1);
    for (int64_t i = 0; i < draft_layers; ++i) {
        ids.push_back(static_cast<int64_t>(std::llround(1.0 + static_cast<double>(i) * span / static_cast<double>(draft_layers - 1))));
    }
    return ids;
}

void refuse_unknown(const nlohmann::json& obj, const std::set<std::string>& known, const char* what) {
    for (const auto& [key, value] : obj.items()) {
        if (known.count(key) == 0) fail("drafter: ", what, ".", key, " is not supported");
    }
}

}  // namespace

std::string_view variant_name(Variant v) {
    switch (v) {
        case Variant::DFlash: return "dflash";
        case Variant::DFlash2: return "dflash2";
        case Variant::DSpine: return "dspine";
    }
    return "?";
}

Config parse_config(const nlohmann::json& j) {
    XS_CHECK(j.contains("architectures") && j.at("architectures").is_array() && j.at("architectures").size() == 1,
             "drafter: config.json names no architecture");
    const std::string arch = j.at("architectures").at(0).get<std::string>();
    XS_CHECK(j.contains("dflash_config") && j.at("dflash_config").is_object(), "drafter: config.json has no dflash_config");
    const nlohmann::json& d = j.at("dflash_config");
    Config c;
    if (arch == "DFlash2DraftModel") {
        c.variant = Variant::DFlash2;
    } else if (arch == "DFlashDraftModel") {
        c.variant = d.contains("selected_prefix_code_bridge") ? Variant::DSpine : Variant::DFlash;
    } else {
        fail("drafter: architecture '", arch, "' is not supported (DFlashDraftModel, DFlash2DraftModel)");
    }
    c.hidden_size = cfg::get_int(j, "hidden_size", kWhere);
    c.intermediate_size = cfg::get_int(j, "intermediate_size", kWhere);
    c.num_layers = cfg::get_int(j, "num_hidden_layers", kWhere);
    c.num_heads = cfg::get_int(j, "num_attention_heads", kWhere);
    c.num_kv_heads = cfg::get_int(j, "num_key_value_heads", kWhere);
    c.head_dim = cfg::get_int(j, "head_dim", kWhere);
    c.vocab_size = cfg::get_int(j, "vocab_size", kWhere);
    c.num_target_layers = cfg::get_int(j, "num_target_layers", kWhere);
    c.rms_norm_eps = cfg::get_float(j, "rms_norm_eps", kWhere);
    if (j.contains("rope_parameters") && j.at("rope_parameters").is_object()) {
        const nlohmann::json& r = j.at("rope_parameters");
        XS_CHECK(r.value("rope_type", std::string("default")) == "default", "drafter: rope_type '",
                 r.value("rope_type", std::string()), "' is not supported");
        c.rope_theta = cfg::get_float(r, "rope_theta", kWhere);
    } else {
        c.rope_theta = cfg::get_float(j, "rope_theta", kWhere);
    }
    XS_CHECK(!j.contains("rope_scaling") || j.at("rope_scaling").is_null(), "drafter: rope_scaling is not supported");
    XS_CHECK(cfg::get_string(j, "hidden_act", kWhere) == "silu", "drafter: hidden_act '",
             cfg::get_string(j, "hidden_act", kWhere), "' is not supported");
    XS_CHECK(!cfg::get_bool_or(j, "attention_bias", false, kWhere), "drafter: attention_bias is not supported");
    XS_CHECK(c.num_layers >= 1 && c.num_heads % c.num_kv_heads == 0 && c.head_dim % 2 == 0,
             "drafter: ", c.num_layers, " layers, ", c.num_heads, " heads over ", c.num_kv_heads, " of ", c.head_dim);

    // Every layer is of one kind: sliding attention (a window, causal unless is_causal says otherwise) or full.
    bool sliding = false;
    if (j.contains("layer_types") && j.at("layer_types").is_array()) {
        XS_CHECK(static_cast<int64_t>(j.at("layer_types").size()) == c.num_layers, "drafter: layer_types names ",
                 j.at("layer_types").size(), " layers of ", c.num_layers);
        const std::string first = j.at("layer_types").at(0).get<std::string>();
        for (const auto& t : j.at("layer_types")) {
            XS_CHECK(t.get<std::string>() == first, "drafter: mixed layer_types are not supported");
        }
        XS_CHECK(first == "sliding_attention" || first == "full_attention", "drafter: layer type '", first,
                 "' is not supported");
        sliding = first == "sliding_attention";
    }
    c.causal = j.contains("is_causal") && !j.at("is_causal").is_null() ? j.at("is_causal").get<bool>() : sliding;
    c.sliding_window = sliding ? cfg::get_int(j, "sliding_window", kWhere) : 0;

    c.mask_token_id = cfg::get_int(d, "mask_token_id", "drafter dflash_config");
    c.block_size = d.contains("block_size") ? cfg::get_int(d, "block_size", "drafter dflash_config")
                                             : cfg::get_int_or(j, "block_size", 16, kWhere);
    XS_CHECK(c.block_size >= 2 && c.block_size <= 64, "drafter: block_size ", c.block_size, " is not supported");
    XS_CHECK(c.sliding_window == 0 || c.block_size <= c.sliding_window, "drafter: a block of ", c.block_size,
             " in a window of ", c.sliding_window);
    if (d.contains("target_layer_ids")) {
        for (const int32_t id : cfg::get_int_list(d, "target_layer_ids", "drafter dflash_config")) c.target_layer_ids.push_back(id);
    } else {
        c.target_layer_ids = default_target_layers(c.num_target_layers, c.num_layers);
    }
    for (const int64_t id : c.target_layer_ids) {
        XS_CHECK(id >= 0 && id < c.num_target_layers, "drafter: target layer ", id, " of ", c.num_target_layers);
    }
    // Gemma-style drafters scale embeddings and logits: none of the checkpoints served here does.
    for (const char* key : {"input_embedding_scale", "output_multiplier", "final_logit_softcapping"}) {
        XS_CHECK(!d.contains(key) && !j.contains(key), "drafter: ", key, " is not supported");
    }

    switch (c.variant) {
        case Variant::DFlash:
            refuse_unknown(d, {"mask_token_id", "target_layer_ids", "block_size"}, "dflash_config");
            break;
        case Variant::DFlash2:
            refuse_unknown(d, {"mask_token_id", "target_layer_ids", "block_size", "conv_group_size", "conv_kernel_size",
                               "selector_rank", "selector_top_k"},
                           "dflash_config");
            c.conv_taps = cfg::get_int(d, "conv_kernel_size", "drafter dflash_config");
            c.conv_group = cfg::get_int(d, "conv_group_size", "drafter dflash_config");
            c.selector_rank = cfg::get_int(d, "selector_rank", "drafter dflash_config");
            c.top_k = cfg::get_int(d, "selector_top_k", "drafter dflash_config");
            XS_CHECK(c.conv_taps >= 1 && c.conv_group >= 1 && c.hidden_size % c.conv_group == 0,
                     "drafter: convolutions of ", c.conv_taps, " taps over groups of ", c.conv_group);
            break;
        case Variant::DSpine: {
            // Training settings are named here so that an unknown key still stops the load.
            refuse_unknown(d, {"mask_token_id", "target_layer_ids", "block_size", "context_fusion_checkpointing",
                               "decoder_checkpointing", "draft_block_causal", "selected_prefix_code_bridge",
                               "selected_prefix_decode_mode", "selected_prefix_feature_gt_training",
                               "selected_prefix_feature_layers", "selected_prefix_feature_teacher_layers",
                               "selected_prefix_feature_teacher_schedule", "selected_prefix_feature_write_mode",
                               "selected_prefix_feature_writes", "selected_prefix_last_write_refine",
                               "selected_prefix_terminal"},
                           "dflash_config");
            const auto is = [&](const char* key, const nlohmann::json& want) { return d.contains(key) && d.at(key) == want; };
            XS_CHECK(is("draft_block_causal", true) && c.causal, "drafter: DSpine without a causal block is not supported");
            XS_CHECK(is("selected_prefix_decode_mode", "serial") && is("selected_prefix_feature_writes", true) &&
                         is("selected_prefix_feature_write_mode", "independent") && is("selected_prefix_terminal", false),
                     "drafter: this DSpine bridge is not supported (serial decode, independent feature writes and no ",
                     "terminal are)");
            std::vector<int32_t> every;  // a write after every layer
            for (int64_t l = 1; l <= c.num_layers; ++l) every.push_back(static_cast<int32_t>(l));
            XS_CHECK(cfg::get_int_list(d, "selected_prefix_feature_layers", "drafter dflash_config") == every,
                     "drafter: DSpine writes after every layer are supported, not a subset");
            const nlohmann::json& b = d.at("selected_prefix_code_bridge");
            c.code_size = cfg::get_int(b, "code_size", "drafter code bridge");
            c.bridge_rank = cfg::get_int(b, "rank", "drafter code bridge");
            c.bridge_eps = cfg::get_float(b, "eps", "drafter code bridge");
            XS_CHECK(d.contains("selected_prefix_last_write_refine"), "drafter: DSpine without a last-write refinement");
            const nlohmann::json& refine = d.at("selected_prefix_last_write_refine");
            XS_CHECK(refine.value("kind", std::string()) == "exact_sender" && refine.value("version", 0) == 1,
                     "drafter: last-write refinement '", refine.value("kind", std::string()), "' version ",
                     refine.value("version", 0), " is not supported");
            c.top_k = cfg::get_int(refine, "head_topk", "drafter last-write refinement");
            if (j.contains("code_bridge_second_pass")) {
                const nlohmann::json& second = j.at("code_bridge_second_pass");
                XS_CHECK(second.value("mode", std::string()) == "serial_table" &&
                             second.value("first_exact", int64_t{-1}) == c.num_layers - 1 &&
                             second.value("head_topk", int64_t{-1}) == c.top_k,
                         "drafter: code_bridge_second_pass must redo the last write over ", c.top_k, " candidates");
            }
            break;
        }
    }
    XS_CHECK(c.variant == Variant::DFlash || (c.top_k >= 1 && c.top_k <= 64), "drafter: ", c.top_k, " candidates");
    return c;
}

std::unique_ptr<WeightStore> open_weights(const std::filesystem::path& dir) {
    return std::make_unique<StackedViews>(SafetensorsWeights::open_dir(dir));
}

Drafter::Drafter(Config config, const WeightStore& weights, std::string prefix, int64_t drafts)
    : cfg_(std::move(config)), weights_(weights), prefix_(std::move(prefix)) {
    rows_ = cfg_.causal && drafts > 0 ? std::min(cfg_.block_size, drafts + 1) : cfg_.block_size;
    const TensorView& fc = weights_.get(prefix_ + "fc.weight");
    const Dims want{cfg_.hidden_size, static_cast<int64_t>(cfg_.target_layer_ids.size()) * cfg_.hidden_size};
    XS_CHECK(fc.shape == want, "drafter: fc.weight is ", dims_to_string(fc.shape), ", expected ", dims_to_string(want));
}

Value Drafter::weight(Graph& g, const std::string& name) const {
    return g.weight(weights_, prefix_ + name);
}

HostTensor Drafter::host_f32(const std::string& name) const {
    const TensorView& t = weights_.get(prefix_ + name);
    HostTensor out(DType::F32, t.shape);
    const auto n = static_cast<size_t>(t.numel());
    switch (t.dtype) {
        case DType::F32: std::copy_n(static_cast<const float*>(t.data), n, out.data<float>()); break;
        case DType::BF16: convert_bf16_to_f32(static_cast<const uint16_t*>(t.data), out.data<float>(), n); break;
        case DType::F16: convert_f16_to_f32(static_cast<const uint16_t*>(t.data), out.data<float>(), n); break;
        default: fail("drafter: '", name, "' has dtype ", dtype_name(t.dtype));
    }
    return out;
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

void Drafter::append_context(Graph& g, std::span<const Value> layers, Value positions) const {
    XS_CHECK(layers.size() == cfg_.target_layer_ids.size(), "drafter: ", layers.size(), " target states for ",
             cfg_.target_layer_ids.size(), " layers");
    Graph::Scope scope(g, "drafter.context");
    const Value features = g.concat(std::vector<Value>(layers.begin(), layers.end()), 1);
    const Value context = norm(g, g.matmul(features, weight(g, "fc.weight")), "hidden_norm.weight");
    const Value freq = inv_freq(g);
    const Dims heads{kT, cfg_.num_kv_heads, cfg_.head_dim};
    for (int64_t l = 0; l < cfg_.num_layers; ++l) {
        const std::string a = "layers." + std::to_string(l) + ".self_attn.";
        const Value k = g.rope(norm(g, g.reshape(g.matmul(context, weight(g, a + "k_proj.weight")), heads), a + "k_norm.weight"),
                               positions, freq);
        const Value v = g.reshape(g.matmul(context, weight(g, a + "v_proj.weight")), heads);
        g.kv_cache(g.add_kv_slot(cfg_.num_kv_heads, cfg_.head_dim, cfg_.sliding_window, slot_label(l)), k, v);
    }
}

Value Drafter::attention(Graph& g, int64_t layer, Value x, Value pos, Value freq) const {
    const std::string a = "layers." + std::to_string(layer) + ".self_attn.";
    blocks::AttentionSpec s;
    s.heads = cfg_.num_heads;
    s.kv_heads = cfg_.num_kv_heads;
    s.head_dim = cfg_.head_dim;
    s.attrs.scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
    s.attrs.causal = cfg_.causal;
    s.attrs.sliding_window = cfg_.sliding_window;
    s.kv_window = cfg_.sliding_window;
    s.kv_label = slot_label(layer);
    blocks::AttentionParts p;
    p.q = blocks::matmul_by(g, weight(g, a + "q_proj.weight"));
    p.k = blocks::matmul_by(g, weight(g, a + "k_proj.weight"));
    p.v = blocks::matmul_by(g, weight(g, a + "v_proj.weight"));
    p.o = blocks::matmul_by(g, weight(g, a + "o_proj.weight"));
    p.q_norm = [this, &g, a](Value v) { return norm(g, v, a + "q_norm.weight"); };
    p.k_norm = [this, &g, a](Value v) { return norm(g, v, a + "k_norm.weight"); };
    return blocks::attention(g, s, p, x, pos, freq).out;
}

// GroupedDynamicCausalConv: out[t] = sum over taps o of x[t - o] * (base[half][o] + kernel[t][half][o]), the kernel
// one gain per group of channels. `kernels` is the layer's kernel projection of x before the sublayer:
// [kT, 2 * taps * groups], half-major.
Value Drafter::convolve(Graph& g, const std::string& conv, Value x, Value kernels, int half) const {
    const int64_t groups = cfg_.hidden_size / cfg_.conv_group;
    const HostTensor base = host_f32(conv + "base_kernel");  // [2, taps, hidden]
    XS_CHECK(base.shape() == (Dims{2, cfg_.conv_taps, cfg_.hidden_size}), "drafter: ", conv, "base_kernel is ",
             dims_to_string(base.shape()));
    Value out;
    Value tap = x;
    for (int64_t o = 0; o < cfg_.conv_taps; ++o) {
        if (o > 0) tap = g.prev_row(tap);
        HostTensor fixed(DType::F32, {groups, cfg_.conv_group});
        std::copy_n(base.data<float>() + (half * cfg_.conv_taps + o) * cfg_.hidden_size, cfg_.hidden_size, fixed.data<float>());
        const int64_t at = (half * cfg_.conv_taps + o) * groups;
        const Value gain = g.add(g.reshape(g.slice(kernels, 1, at, at + groups), {kT, groups, 1}), g.constant(std::move(fixed)));
        const Value term = g.mul(g.reshape(tap, {kT, groups, cfg_.conv_group}), gain);
        out = o == 0 ? term : g.add(out, term);
    }
    return g.reshape(out, {kT, cfg_.hidden_size});
}

Value Drafter::bridge_write(Graph& g, int64_t layer, Value h, Value anchor_code, Value rest, LastWrite* last) const {
    const std::string b = "selected_prefix_bridge.";
    const auto eps = static_cast<float>(cfg_.bridge_eps);
    // h = rms * unit, and a code is what the layer's read weight sees of unit, on the sphere of radius sqrt(code_size)
    const Value unit = g.rms_norm(h, eps);
    const Value rms = g.reshape(g.scale(g.reduce_sum(g.mul(h, unit), 1), 1.0f / static_cast<float>(cfg_.hidden_size)), {kT, 1});
    const std::string read = layer + 1 < cfg_.num_layers ? b + "early_read_weights." + std::to_string(layer) : b + "read_weight";
    const Value read_codes = g.rms_norm(g.matmul(unit, weight(g, read)), eps);
    // the pending token's row sends the code of its token instead
    const Value first = g.add(g.scale(rest, -1.0f), g.constant(HostTensor::from_vector(DType::F32, {1, 1}, std::vector<float>{1.0f})));
    const Value codes = g.add(g.mul(read_codes, rest), g.mul(anchor_code, first));
    const Value query = g.matmul(codes, weight(g, b + "gate.weight"));
    const Value sent_gate = g.prev_row(g.matmul(codes, weight(g, b + "predecessor_gate.weight")));
    const Value sent = g.prev_row(g.matmul(codes, weight(g, b + "prev.weight")));
    const Value gate = g.sigmoid(g.add(g.add(query, sent_gate), g.constant(host_f32(b + "gate.bias"))));
    const Value delta = g.matmul(g.mul(gate, sent), weight(g, b + "feature_write_weights." + std::to_string(layer)));
    if (last != nullptr) *last = {h, rms, query};
    return g.add(h, g.mul(rms, delta));
}

// DFlash 2's candidate selector: score[i][j][k] = logit of candidate k of draft i + sum_r P[sender j][r] H[i][r]
// S[candidate k][r], the sender the pending token for the first draft and candidate j of the draft before after it.
void Drafter::selector_outputs(Graph& g, Value rows, Value logits, Value anchor) const {
    const int64_t n = block_drafts();
    const int64_t k = cfg_.top_k;
    const int64_t r = cfg_.selector_rank;
    const std::string s = "candidate_selector.";
    const auto [unary, cand] = g.topk(logits, k);
    const Value senders = n > 1 ? g.concat({anchor, g.slice(cand, 0, 0, n - 1)}, 0) : anchor;
    const Value hidden = g.reshape(g.matmul(rows, weight(g, s + "hidden_projection.weight")), {n, 1, r});
    const Value successor = g.mul(g.gather(weight(g, s + "successor_codebook"), cand), hidden);  // [n, k, r]
    const Value predecessor = g.gather(weight(g, s + "predecessor_codebook"), senders);         // [n, k, r]
    const Value pair = g.reduce_sum(g.mul(g.reshape(predecessor, {n, k, 1, r}), g.reshape(successor, {n, 1, k, r})), 3);
    g.output("candidates", cand);
    g.output("scores", g.add(pair, g.reshape(unary, {n, 1, k})));
}

// DSpine's last-write refinement (dspine_mlx/propose.py RefinementTail.greedy): the last layer's write redone per
// candidate sender. With w the final norm's weight and W the last write weight, candidate k of draft i scores
// (raw_i * w) . head_k + rms_i * message(sender j) . ((head_k * w) W), where message(j) is the gated code of the
// sender: the pending token's for the first draft, candidate j of the draft before after it.
void Drafter::refine_outputs(Graph& g, Value logits, Value lm_head, Value anchor, const LastWrite& last) const {
    const int64_t n = block_drafts();
    const int64_t k = cfg_.top_k;
    const int64_t r = cfg_.bridge_rank;
    const int64_t h = cfg_.hidden_size;
    const std::string b = "selected_prefix_bridge.";
    const Value cand = g.topk(logits, k).second;
    const Value head_rows = g.gather(lm_head, cand);  // [n, k, hidden]
    const Value norm_w = g.constant(host_f32("norm.weight"));
    const Value raw = g.mul(g.slice(last.raw, 0, -n), norm_w);
    const Value base = g.reduce_sum(g.mul(head_rows, g.reshape(raw, {n, 1, h})), 2);  // [n, k]
    const Value senders = n > 1 ? g.concat({anchor, g.slice(cand, 0, 0, n - 1)}, 0) : anchor;
    const Value codes = g.scale(g.gather(weight(g, b + "codebook"), senders), std::sqrt(static_cast<float>(cfg_.code_size)));
    const Value gate = g.sigmoid(g.add(g.add(g.matmul(codes, weight(g, b + "predecessor_gate.weight")),
                                             g.reshape(g.slice(last.query, 0, -n), {n, 1, r})),
                                       g.constant(host_f32(b + "gate.bias"))));
    const Value message = g.mul(gate, g.matmul(codes, weight(g, b + "prev.weight")));  // [n, k senders, r]
    const Value write = g.transpose(weight(g, b + "feature_write_weights." + std::to_string(cfg_.num_layers - 1)), {1, 0});
    const Value reach = g.matmul(g.mul(head_rows, norm_w), write);  // [n, k candidates, r]
    const Value pair = g.reduce_sum(g.mul(g.reshape(message, {n, k, 1, r}), g.reshape(reach, {n, 1, k, r})), 3);
    g.output("candidates", cand);
    g.output("scores", g.add(g.mul(pair, g.reshape(g.slice(last.rms, 0, -n), {n, 1, 1})), g.reshape(base, {n, 1, k})));
}

void Drafter::build_block(Graph& g, Value embed, Value lm_head) const {
    const int64_t n = block_drafts();
    const Value ids = g.input("token_ids", DType::I32, {kT});
    const Value pos = g.input("positions", DType::I32, {kT});
    const bool paths = cfg_.variant != Variant::DFlash;
    // the pending token once per candidate: the sender of the first draft
    const Value anchor = paths ? g.input("anchor", DType::I32, {1, cfg_.top_k}) : Value{};
    const Value freq = inv_freq(g);
    Value h = g.gather(embed, ids);

    Value anchor_code, rest;
    LastWrite last;
    if (cfg_.variant == Variant::DSpine) {
        Graph::Scope scope(g, "drafter.bridge");
        const Value token = g.reshape(g.slice(anchor, 1, 0, 1), {1});
        anchor_code = g.scale(g.gather(weight(g, "selected_prefix_bridge.codebook"), token),
                              std::sqrt(static_cast<float>(cfg_.code_size)));
        // 0 on the pending token's row, 1 on the others
        const Value ones = g.add(g.scale(g.slice(h, 1, 0, 1), 0.0f),
                                 g.constant(HostTensor::from_vector(DType::F32, {1, 1}, std::vector<float>{1.0f})));
        rest = g.prev_row(ones);
    }
    for (int64_t l = 0; l < cfg_.num_layers; ++l) {
        const std::string name = "layers." + std::to_string(l);
        Graph::Scope scope(g, "drafter." + name);
        const std::string L = name + ".";
        const bool conv = cfg_.variant == Variant::DFlash2;
        Value x = norm(g, h, L + "input_layernorm.weight");
        Value kernels;
        if (conv) {
            kernels = g.matmul(x, weight(g, L + "attention_conv.kernel_projection.weight"));
            x = convolve(g, L + "attention_conv.", x, kernels, 0);
        }
        Value a = attention(g, l, x, pos, freq);
        if (conv) a = convolve(g, L + "attention_conv.", a, kernels, 1);
        h = g.add(h, a);
        Value y = norm(g, h, L + "post_attention_layernorm.weight");
        if (conv) {
            kernels = g.matmul(y, weight(g, L + "mlp_conv.kernel_projection.weight"));
            y = convolve(g, L + "mlp_conv.", y, kernels, 0);
        }
        Value m = blocks::gated_mlp(g, y, blocks::matmul_by(g, weight(g, L + "mlp.gate_proj.weight")),
                                    blocks::matmul_by(g, weight(g, L + "mlp.up_proj.weight")),
                                    blocks::matmul_by(g, weight(g, L + "mlp.down_proj.weight")), blocks::Act::Silu);
        if (conv) m = convolve(g, L + "mlp_conv.", m, kernels, 1);
        h = g.add(h, m);
        if (cfg_.variant == Variant::DSpine) {
            h = bridge_write(g, l, h, anchor_code, rest, l + 1 == cfg_.num_layers ? &last : nullptr);
        }
    }
    Graph::Scope scope(g, "drafter.head");
    const Value rows = g.slice(norm(g, h, "norm.weight"), 0, -n);  // the drafts' rows: all but the pending token's
    const Value logits = g.matmul(rows, lm_head);
    switch (cfg_.variant) {
        case Variant::DFlash: g.output("drafts", blocks::argmax_rows(g, logits)); break;
        case Variant::DFlash2: selector_outputs(g, rows, logits, anchor); break;
        case Variant::DSpine: refine_outputs(g, logits, lm_head, anchor, last); break;
    }
    g.validate();
}

void Drafter::prepare_block(int32_t token, int64_t position, TensorMap& inputs) const {
    HostTensor ids(DType::I32, {rows_});
    HostTensor pos(DType::I32, {rows_});
    for (int64_t i = 0; i < rows_; ++i) {
        ids.data<int32_t>()[i] = i == 0 ? token : static_cast<int32_t>(cfg_.mask_token_id);
        pos.data<int32_t>()[i] = static_cast<int32_t>(position + i);
    }
    inputs["token_ids"] = std::move(ids);
    inputs["positions"] = std::move(pos);
    if (cfg_.variant != Variant::DFlash) {
        inputs["anchor"] = HostTensor::from_vector(DType::I32, {1, cfg_.top_k},
                                                   std::vector<int32_t>(static_cast<size_t>(cfg_.top_k), token));
    }
}

}  // namespace xs::dflash
