#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/pi05/pi05.h"
#include "util/error.h"
#include "util/half.h"
#include "util/log.h"

namespace xs::pi05 {
namespace {

using graph::AttentionAttrs;
using graph::Graph;
using graph::Value;

constexpr int64_t kPatch = 14;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr float kNormEps = 1e-6f;
constexpr int64_t kPromptBucket = 32;  // programs read the prompt padded to a multiple of this
const std::string kRoot = "paligemma_with_expert.";
const std::string kVision = kRoot + "paligemma.model.vision_tower.vision_model.";
const std::string kVlm = kRoot + "paligemma.model.language_model.";
const std::string kExpert = kRoot + "gemma_expert.model.";
const std::string kEmbed = kRoot + "paligemma.lm_head.weight";  // the token table, tied to the LM head

std::vector<float> to_f32(const WeightStore& w, const std::string& name) {
    const TensorView& t = w.get(name);
    const auto n = static_cast<size_t>(t.numel());
    std::vector<float> out(n);
    switch (t.dtype) {
        case DType::F32: std::memcpy(out.data(), t.data, n * sizeof(float)); break;
        case DType::BF16: convert_bf16_to_f32(static_cast<const uint16_t*>(t.data), out.data(), n); break;
        case DType::F16: convert_f16_to_f32(static_cast<const uint16_t*>(t.data), out.data(), n); break;
        default: fail("pi05: '", name, "' has dtype ", dtype_name(t.dtype));
    }
    return out;
}

// y = W x + b on the host, W [rows, cols] (a Linear), accumulated in f32 like the reference.
std::vector<float> linear(const std::vector<float>& w, const std::vector<float>& b, const std::vector<float>& x) {
    const size_t cols = x.size();
    const size_t rows = w.size() / cols;
    std::vector<float> y(rows);
    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c)
            acc += w[r * cols + c] * x[c];
        y[r] = acc + b[r];
    }
    return y;
}

std::vector<float> silu(std::vector<float> x) {
    for (float& v : x)
        v = v / (1.0f + std::exp(-v));
    return x;
}

HostTensor vec(const std::vector<float>& v, size_t first, size_t count) {
    HostTensor t(DType::F32, {static_cast<int64_t>(count)});
    std::copy(v.begin() + static_cast<std::ptrdiff_t>(first), v.begin() + static_cast<std::ptrdiff_t>(first + count),
              t.data<float>());
    return t;
}

class Pi05Policy final : public Policy {
public:
    Pi05Policy(Config cfg, PolicySpec spec, std::unique_ptr<SafetensorsWeights> weights)
        : cfg_(std::move(cfg)), spec_(std::move(spec)), weights_(std::move(weights)) {
        while (weights_->contains(kVision + "encoder.layers." + std::to_string(vision_layers_) + ".layer_norm1.weight"))
            ++vision_layers_;
        XS_CHECK(vision_layers_ > 0, "pi05: no vision tower weights under ", kVision);
        vision_width_ = weights_->get(kVision + "post_layernorm.weight").shape[0];
        vision_heads_ = vision_width_ / 72;  // SigLIP So400m/14: head dim 72
        const Dims& patch = weights_->get(kVision + "embeddings.patch_embedding.weight").shape;
        XS_CHECK(patch == (Dims{vision_width_, 3, kPatch, kPatch}), "pi05: patch embedding ", dims_to_string(patch));
        const Dims& pos = weights_->get(kVision + "embeddings.position_embedding.weight").shape;
        patches_ = (cfg_.image_size / kPatch) * (cfg_.image_size / kPatch);
        XS_CHECK(pos == (Dims{patches_, vision_width_}), "pi05: ", patches_, " patches but position embedding ",
                 dims_to_string(pos));
        const Dims& embed = weights_->get(kEmbed).shape;
        XS_CHECK(embed.size() == 2 && embed[1] == cfg_.vlm.width, "pi05: token table ", dims_to_string(embed));
        vocab_ = embed[0];
        embed_scale_ = static_cast<float>(std::sqrt(static_cast<double>(cfg_.vlm.width)));
        // The flow-timestep path on the host: sinusoid -> time MLP -> each adaptive norm's dense layer.
        time_in_w_ = to_f32(*weights_, "time_mlp_in.weight");
        time_in_b_ = to_f32(*weights_, "time_mlp_in.bias");
        time_out_w_ = to_f32(*weights_, "time_mlp_out.weight");
        time_out_b_ = to_f32(*weights_, "time_mlp_out.bias");
        for (int64_t l = 0; l < cfg_.expert.depth; ++l) {
            for (const char* norm : {"input_layernorm", "post_attention_layernorm"}) {
                const std::string base = kExpert + "layers." + std::to_string(l) + "." + norm + ".dense.";
                dense_w_.push_back(to_f32(*weights_, base + "weight"));
                dense_b_.push_back(to_f32(*weights_, base + "bias"));
            }
        }
        dense_w_.push_back(to_f32(*weights_, kExpert + "norm.dense.weight"));
        dense_b_.push_back(to_f32(*weights_, kExpert + "norm.dense.bias"));
    }

    const PolicySpec& spec() const override { return spec_; }
    const WeightStore& weights() const override { return *weights_; }
    int32_t prompt_bos() const override { return 2; }  // PaliGemma's <bos>

    // processor_pi05.py Pi05PrepareStateTokenizerProcessorStep: the state (normalized to [-1, 1]) in 256 bins.
    std::string prompt(const Observation& obs) const override {
        XS_CHECK(static_cast<int64_t>(obs.state.size()) == cfg_.state_dim, "pi05: the state has ", obs.state.size(),
                 " values; the policy reads ", cfg_.state_dim);
        std::string task = obs.instruction;
        const auto not_space = [](unsigned char c) { return !std::isspace(c); };
        task.erase(task.begin(), std::find_if(task.begin(), task.end(), not_space));
        task.erase(std::find_if(task.rbegin(), task.rend(), not_space).base(), task.end());
        std::replace(task.begin(), task.end(), '_', ' ');
        std::replace(task.begin(), task.end(), '\n', ' ');
        std::string state;
        for (const float s : obs.state) {
            // np.digitize(s, linspace(-1, 1, 257)[:-1]) - 1: the bin edges -1 + k / 128 are exact in double.
            int bins = 0;
            for (int k = 0; k < 256; ++k)
                bins += -1.0 + k / 128.0 <= static_cast<double>(s) ? 1 : 0;
            state += (state.empty() ? "" : " ") + std::to_string(bins - 1);
        }
        return "Task: " + task + ", State: " + state + ";\nAction: ";
    }

    // LeRobot pads every prompt to tokenizer_max_length. The pads are masked out as keys and nothing reads their
    // outputs, so a program reads the prompt padded only to the next multiple of kPromptBucket: the same actions.
    int64_t prompt_tokens(int64_t tokens) const override {
        return std::min(cfg_.prompt_tokens, (std::max<int64_t>(tokens, 1) + kPromptBucket - 1) / kPromptBucket * kPromptBucket);
    }

    graph::Graph build(const CallShape& shape) const override {
        const int steps = shape.steps;
        XS_CHECK(!shape.cameras.empty() && steps > 0, "pi05: a program needs at least one camera and one step");
        XS_CHECK(shape.prompt_tokens > 0 && shape.prompt_tokens <= cfg_.prompt_tokens, "pi05: ", shape.prompt_tokens,
                 " prompt tokens; the policy reads 1 to ", cfg_.prompt_tokens);
        Graph g;
        const auto n_cam = static_cast<int64_t>(shape.cameras.size());
        const int64_t image_tokens = n_cam * patches_;
        const int64_t prefix = image_tokens + shape.prompt_tokens;
        const Value pixels = g.input("pixels", DType::F32, {image_tokens, 3 * kPatch * kPatch});
        const Value text = g.input("text", DType::F32, {shape.prompt_tokens, cfg_.vlm.width});
        const Value prefix_bias = g.input("prefix_bias", DType::F32, {prefix});
        const Value positions = g.input("positions", DType::I32, {prefix});
        const Value suffix_positions = g.input("suffix_positions", DType::I32, {cfg_.horizon});
        const Value noise = g.input("noise", DType::F32, {cfg_.horizon, cfg_.max_action_dim});

        const Value image = vision(g, pixels, n_cam);
        const int64_t hd = cfg_.vlm.head_dim;
        const Value inv_freq = g.constant(blocks::rope_inv_freq(10000.0, hd, hd / 2, hd / 2));
        std::vector<std::pair<Value, Value>> kv;  // each prefix layer's keys and values (after RoPE)
        {
            Graph::Scope s(g, "prefix");
            Value h = g.concat({image, text}, 0);
            for (int64_t l = 0; l < cfg_.vlm.depth; ++l) h = prefix_layer(g, l, h, positions, inv_freq, prefix_bias, kv);
        }
        HostTensor zeros(DType::F32, {cfg_.horizon});
        std::fill(zeros.data<float>(), zeros.data<float>() + cfg_.horizon, 0.0f);
        const Value joint_bias = g.concat({prefix_bias, g.constant(std::move(zeros))}, 0);

        // Forward Euler from noise (t = 1) to actions (t = 0): flow_matching.py euler_integrate, NOISE_AT_ONE.
        const float dt = -1.0f / static_cast<float>(steps);
        Value x = noise;
        for (int k = 0; k < steps; ++k) {
            Graph::Scope s(g, "step" + std::to_string(k));
            const float t = 1.0f + static_cast<float>(k) * dt;
            const Value v = expert(g, x, condition(t), suffix_positions, inv_freq, joint_bias, kv);
            x = g.add(x, g.scale(v, dt));
        }
        g.output("actions", x);
        g.validate();
        log::debug("pi05: program for ", n_cam, " camera(s), ", steps, " steps, ", shape.prompt_tokens, " prompt tokens: ",
                   g.nodes().size(), " nodes");
        return g;
    }

    void prepare(const Observation& obs, const CallShape& shape, std::span<const int32_t> prompt_ids,
                 const HostTensor& noise, TensorMap& inputs) const override {
        const std::vector<int>& cameras = shape.cameras;
        const auto n_cam = static_cast<int64_t>(cameras.size());
        const int64_t side = cfg_.image_size / kPatch;
        const int64_t patch_dim = 3 * kPatch * kPatch;
        HostTensor pixels(DType::F32, {n_cam * patches_, patch_dim});
        float* px = pixels.data<float>();
        for (int64_t c = 0; c < n_cam; ++c) {
            const CameraSpec& spec = spec_.cameras[static_cast<size_t>(cameras[static_cast<size_t>(c)])];
            const image::Image* img = nullptr;
            for (const auto& [name, im] : obs.cameras) {
                if (name == spec.name) img = &im;
            }
            XS_CHECK(img != nullptr, "pi05: camera ", spec.name, " is missing from the observation");
            XS_CHECK(img->width == cfg_.image_size && img->height == cfg_.image_size, "pi05: camera ", spec.name, " is ",
                     img->width, "x", img->height, "; the policy reads ", cfg_.image_size, "x", cfg_.image_size,
                     " (resizing is not supported yet)");
            // Patch (py, px) row-major; within a patch (channel, row, column), the conv weight's layout. The pixel
            // value is u8 / 255 * 2 - 1 in f32, as LeRobot computes it.
            for (int64_t py = 0; py < side; ++py) {
                for (int64_t pxi = 0; pxi < side; ++pxi) {
                    float* out = px + ((c * side + py) * side + pxi) * patch_dim;
                    for (int ch = 0; ch < 3; ++ch) {
                        for (int64_t r = 0; r < kPatch; ++r) {
                            for (int64_t col = 0; col < kPatch; ++col) {
                                const float v = static_cast<float>(img->at(py * kPatch + r, pxi * kPatch + col, ch));
                                *out++ = v / 255.0f * 2.0f - 1.0f;
                            }
                        }
                    }
                }
            }
        }
        inputs["pixels"] = std::move(pixels);

        // The prompt, truncated to and padded up to the shape's prompt tokens; pads are masked out as keys and keep
        // the last valid position (big_vision make_att_2d_masks with cumsum positions).
        const int64_t slots = shape.prompt_tokens;
        const int64_t valid = std::min<int64_t>(static_cast<int64_t>(prompt_ids.size()), slots);
        const TensorView& table = weights_->get(kEmbed);
        HostTensor text(DType::F32, {slots, cfg_.vlm.width});
        float* tx = text.data<float>();
        const int64_t width = cfg_.vlm.width;
        for (int64_t i = 0; i < slots; ++i) {
            float* row = tx + i * width;
            if (i >= valid) {
                std::fill(row, row + width, 0.0f);
                continue;
            }
            const int32_t id = prompt_ids[static_cast<size_t>(i)];
            XS_CHECK(id >= 0 && id < vocab_, "pi05: token id ", id, " outside the vocabulary (", vocab_, ")");
            const size_t at = static_cast<size_t>(id) * static_cast<size_t>(width);
            if (table.dtype == DType::F32) {
                std::memcpy(row, static_cast<const float*>(table.data) + at, static_cast<size_t>(width) * sizeof(float));
            } else if (table.dtype == DType::BF16) {
                convert_bf16_to_f32(static_cast<const uint16_t*>(table.data) + at, row, static_cast<size_t>(width));
            } else if (table.dtype == DType::F16) {
                convert_f16_to_f32(static_cast<const uint16_t*>(table.data) + at, row, static_cast<size_t>(width));
            } else {
                fail("pi05: the token table has dtype ", dtype_name(table.dtype));
            }
            for (int64_t d = 0; d < width; ++d)
                row[d] *= embed_scale_;
        }
        inputs["text"] = std::move(text);

        const int64_t image_tokens = n_cam * patches_;
        const int64_t prefix = image_tokens + slots;
        HostTensor bias(DType::F32, {prefix});
        HostTensor pos(DType::I32, {prefix});
        for (int64_t i = 0; i < prefix; ++i) {
            const bool keep = i < image_tokens + valid;
            bias.data<float>()[i] = keep ? 0.0f : -std::numeric_limits<float>::infinity();
            pos.data<int32_t>()[i] = static_cast<int32_t>(std::min(i, image_tokens + valid - 1));
        }
        inputs["prefix_bias"] = std::move(bias);
        inputs["positions"] = std::move(pos);
        HostTensor suffix(DType::I32, {cfg_.horizon});
        for (int64_t i = 0; i < cfg_.horizon; ++i)
            suffix.data<int32_t>()[i] = static_cast<int32_t>(image_tokens + valid + i);
        inputs["suffix_positions"] = std::move(suffix);
        XS_CHECK(noise.dtype() == DType::F32 && noise.shape() == (Dims{cfg_.horizon, cfg_.max_action_dim}),
                 "pi05: noise must be f32 [", cfg_.horizon, ", ", cfg_.max_action_dim, "], got ",
                 dims_to_string(noise.shape()));
        inputs["noise"] = noise;
    }

    HostTensor finish(const TensorMap& outputs) const override {
        const HostTensor& all = outputs.at("actions");
        HostTensor out(DType::F32, {cfg_.horizon, cfg_.action_dim});
        for (int64_t t = 0; t < cfg_.horizon; ++t) {
            std::copy(all.data<float>() + t * cfg_.max_action_dim,
                      all.data<float>() + t * cfg_.max_action_dim + cfg_.action_dim, out.data<float>() + t * cfg_.action_dim);
        }
        return out;  // unnormalization is the identity until statistics are supported (parse_config refuses them)
    }

private:
    Value weight(Graph& g, const std::string& name) const { return g.weight(*weights_, name); }
    Value linear_bias(Graph& g, Value x, const std::string& base) const {
        return g.add(g.matmul(x, weight(g, base + "weight")), weight(g, base + "bias"));
    }

    // SigLIP So400m/14 (transformers SiglipVisionModel, last hidden state) and the PaliGemma projector. The cameras'
    // tokens share every matmul; attention stays within each camera's patches: with several cameras, one attention
    // call whose heads are (camera, head) pairs.
    Value vision(Graph& g, Value pixels, int64_t n_cam) const {
        Graph::Scope s(g, "vision");
        const int64_t tokens = n_cam * patches_;
        const int64_t w = vision_width_;
        const int64_t head_dim = w / vision_heads_;
        const Value patch_w = g.reshape(weight(g, kVision + "embeddings.patch_embedding.weight"), {w, 3 * kPatch * kPatch});
        Value x = g.add(g.matmul(pixels, patch_w), weight(g, kVision + "embeddings.patch_embedding.bias"));
        x = g.reshape(g.add(g.reshape(x, {n_cam, patches_, w}), weight(g, kVision + "embeddings.position_embedding.weight")),
                      {tokens, w});
        AttentionAttrs attrs;
        attrs.causal = false;
        attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
        for (int64_t l = 0; l < vision_layers_; ++l) {
            Graph::Scope ls(g, "layers." + std::to_string(l));
            const std::string L = kVision + "encoder.layers." + std::to_string(l) + ".";
            Value h = g.layer_norm(x, weight(g, L + "layer_norm1.weight"), weight(g, L + "layer_norm1.bias"), kNormEps);
            // Attention per camera: a camera's rows of q, k and v are a view, and the outputs concatenate along tokens
            // (cheaper than folding the cameras into heads, which takes four transposes per layer).
            const Value q = linear_bias(g, h, L + "self_attn.q_proj.");
            const Value k = linear_bias(g, h, L + "self_attn.k_proj.");
            const Value v = linear_bias(g, h, L + "self_attn.v_proj.");
            std::vector<Value> per_camera;
            for (int64_t c = 0; c < n_cam; ++c) {
                const auto rows = [&](Value t) {
                    return g.reshape(g.slice(t, 0, c * patches_, (c + 1) * patches_), {patches_, vision_heads_, head_dim});
                };
                per_camera.push_back(g.reshape(g.attention(rows(q), rows(k), rows(v), attrs), {patches_, w}));
            }
            const Value att = n_cam == 1 ? per_camera[0] : g.concat(per_camera, 0);
            x = g.add(x, linear_bias(g, att, L + "self_attn.out_proj."));
            h = g.layer_norm(x, weight(g, L + "layer_norm2.weight"), weight(g, L + "layer_norm2.bias"), kNormEps);
            h = linear_bias(g, g.gelu_tanh(linear_bias(g, h, L + "mlp.fc1.")), L + "mlp.fc2.");
            x = g.add(x, h);
        }
        x = g.layer_norm(x, weight(g, kVision + "post_layernorm.weight"), weight(g, kVision + "post_layernorm.bias"),
                         kNormEps);
        return linear_bias(g, x, kRoot + "paligemma.model.multi_modal_projector.linear.");
    }

    // One Gemma 2B layer over the prefix (bidirectional, padding masked). The last layer's MLP is skipped: only its
    // keys and values are read.
    Value prefix_layer(Graph& g, int64_t l, Value h, Value positions, Value inv_freq, Value bias,
                       std::vector<std::pair<Value, Value>>& kv) const {
        Graph::Scope s(g, "layers." + std::to_string(l));
        const std::string L = kVlm + "layers." + std::to_string(l) + ".";
        const GemmaDims& d = cfg_.vlm;
        const int64_t tokens = g.type(h).shape[0];
        const Value x = blocks::rms_norm_one_plus(g, h, *weights_, L + "input_layernorm.weight", kNormEps);
        const Value q = g.rope(g.reshape(g.matmul(x, weight(g, L + "self_attn.q_proj.weight")), {tokens, d.heads, d.head_dim}),
                               positions, inv_freq);
        const Value k = g.rope(g.reshape(g.matmul(x, weight(g, L + "self_attn.k_proj.weight")), {tokens, d.kv_heads, d.head_dim}),
                               positions, inv_freq);
        const Value v = g.reshape(g.matmul(x, weight(g, L + "self_attn.v_proj.weight")), {tokens, d.kv_heads, d.head_dim});
        kv.emplace_back(k, v);
        AttentionAttrs attrs;
        attrs.causal = false;
        attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.head_dim)));
        const Value att = g.reshape(g.attention(q, k, v, bias, attrs), {tokens, d.heads * d.head_dim});
        h = g.add(h, g.matmul(att, weight(g, L + "self_attn.o_proj.weight")));
        if (l + 1 == d.depth) return h;
        const Value y = blocks::rms_norm_one_plus(g, h, *weights_, L + "post_attention_layernorm.weight", kNormEps);
        return g.add(h, blocks::gated_mlp(g, y, blocks::matmul_by(g, weight(g, L + "mlp.gate_proj.weight")),
                                          blocks::matmul_by(g, weight(g, L + "mlp.up_proj.weight")),
                                          blocks::matmul_by(g, weight(g, L + "mlp.down_proj.weight")),
                                          blocks::Act::GeluTanh));
    }

    // Every adaptive norm's (scale, shift, gate) for flow time t: dense(silu(time_mlp_out(silu(time_mlp_in(sin/cos(t))))).
    std::vector<std::vector<float>> condition(float t) const {
        const int64_t width = cfg_.expert.width;
        std::vector<float> emb(static_cast<size_t>(width));
        const int64_t half = width / 2;
        for (int64_t i = 0; i < half; ++i) {  // vla_utils.py create_sinusoidal_pos_embedding, in float64
            const double fraction = half > 1 ? static_cast<double>(i) / static_cast<double>(half - 1) : 0.0;
            const double period = cfg_.min_period * std::pow(cfg_.max_period / cfg_.min_period, fraction);
            const double arg = static_cast<double>(t) * (1.0 / period * 2.0 * kPi);
            emb[static_cast<size_t>(i)] = static_cast<float>(std::sin(arg));
            emb[static_cast<size_t>(i + half)] = static_cast<float>(std::cos(arg));
        }
        const std::vector<float> cond =
            silu(linear(time_out_w_, time_out_b_, silu(linear(time_in_w_, time_in_b_, emb))));
        std::vector<std::vector<float>> out;
        for (size_t n = 0; n < dense_w_.size(); ++n)
            out.push_back(linear(dense_w_[n], dense_b_[n], cond));
        return out;
    }

    // rms_norm(x) * (1 + scale) + shift; returns the normed value, the gate goes to the residual.
    Value adaptive_norm(Graph& g, Value x, const std::vector<float>& modulation) const {
        const auto w = static_cast<size_t>(cfg_.expert.width);
        std::vector<float> one_plus(w);
        for (size_t i = 0; i < w; ++i)
            one_plus[i] = 1.0f + modulation[i];
        return g.add(g.rms_norm(x, g.constant(vec(one_plus, 0, w)), kNormEps), g.constant(vec(modulation, w, w)));
    }
    Value gate(Graph& g, const std::vector<float>& modulation) const {
        const auto w = static_cast<size_t>(cfg_.expert.width);
        return g.constant(vec(modulation, 2 * w, w));
    }

    // One flow step: the action expert over the noisy actions, attending to the prefix's keys and values and its own.
    Value expert(Graph& g, Value x_t, const std::vector<std::vector<float>>& mod, Value positions, Value inv_freq,
                 Value joint_bias, const std::vector<std::pair<Value, Value>>& kv) const {
        const GemmaDims& d = cfg_.expert;
        const int64_t tokens = cfg_.horizon;
        Value h = linear_bias(g, x_t, "action_in_proj.");
        AttentionAttrs attrs;
        attrs.causal = false;
        attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.head_dim)));
        for (int64_t l = 0; l < d.depth; ++l) {
            Graph::Scope s(g, "expert." + std::to_string(l));
            const std::string L = kExpert + "layers." + std::to_string(l) + ".";
            const auto& m_in = mod[static_cast<size_t>(2 * l)];
            const auto& m_post = mod[static_cast<size_t>(2 * l + 1)];
            Value x = adaptive_norm(g, h, m_in);
            const Value q = g.rope(g.reshape(g.matmul(x, weight(g, L + "self_attn.q_proj.weight")), {tokens, d.heads, d.head_dim}),
                                   positions, inv_freq);
            const Value k = g.rope(g.reshape(g.matmul(x, weight(g, L + "self_attn.k_proj.weight")), {tokens, d.kv_heads, d.head_dim}),
                                   positions, inv_freq);
            const Value v = g.reshape(g.matmul(x, weight(g, L + "self_attn.v_proj.weight")), {tokens, d.kv_heads, d.head_dim});
            const auto& [pk, pv] = kv[static_cast<size_t>(l)];
            const Value att = g.attention(q, g.concat({pk, k}, 0), g.concat({pv, v}, 0), joint_bias, attrs);
            const Value o = g.matmul(g.reshape(att, {tokens, d.heads * d.head_dim}), weight(g, L + "self_attn.o_proj.weight"));
            h = g.add(h, g.mul(o, gate(g, m_in)));
            x = adaptive_norm(g, h, m_post);
            const Value mlp = blocks::gated_mlp(g, x, blocks::matmul_by(g, weight(g, L + "mlp.gate_proj.weight")),
                                                blocks::matmul_by(g, weight(g, L + "mlp.up_proj.weight")),
                                                blocks::matmul_by(g, weight(g, L + "mlp.down_proj.weight")),
                                                blocks::Act::GeluTanh);
            h = g.add(h, g.mul(mlp, gate(g, m_post)));
        }
        return linear_bias(g, adaptive_norm(g, h, mod.back()), "action_out_proj.");
    }

    Config cfg_;
    PolicySpec spec_;
    std::unique_ptr<SafetensorsWeights> weights_;
    int64_t vision_layers_ = 0;
    int64_t vision_width_ = 0;
    int64_t vision_heads_ = 0;
    int64_t patches_ = 0;
    int64_t vocab_ = 0;
    float embed_scale_ = 1.0f;
    std::vector<float> time_in_w_, time_in_b_, time_out_w_, time_out_b_;
    std::vector<std::vector<float>> dense_w_, dense_b_;  // per expert norm (input, post per layer, then final)
};

}  // namespace

std::unique_ptr<Policy> load(const std::filesystem::path& dir, const nlohmann::json& config) {
    const std::filesystem::path pre = dir / "policy_preprocessor.json";
    const nlohmann::json preprocessor = std::filesystem::exists(pre) ? cfg::read_json(pre) : nlohmann::json();
    Config cfg = parse_config(config, std::filesystem::exists(pre) ? &preprocessor : nullptr);
    if (!std::filesystem::exists(dir / "tokenizer.json")) {
        fail("pi05: ", (dir / "tokenizer.json").string(), " is missing; the policy reads PaliGemma's tokenizer "
             "(google/paligemma-3b-pt-224, or the SentencePiece model openpi uses, converted)");
    }
    PolicySpec spec;
    spec.arch = "pi05";
    spec.id = cfg::model_id(dir);
    spec.cameras = cfg.cameras;
    spec.state_dim = cfg.state_dim;
    spec.action_dim = cfg.action_dim;
    spec.horizon = cfg.horizon;
    spec.noise_dim = cfg.max_action_dim;
    spec.denoise_steps = cfg.steps;
    return std::make_unique<Pi05Policy>(std::move(cfg), std::move(spec), SafetensorsWeights::open_dir(dir));
}

}  // namespace xs::pi05
