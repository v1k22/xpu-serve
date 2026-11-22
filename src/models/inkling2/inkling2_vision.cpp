#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "image/image.h"
#include "models/common/torch_nn.h"
#include "models/inkling2/inkling2.h"
#include "util/error.h"

namespace xs::inkling2 {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

constexpr const char* kPrefix = "model.visual.";
constexpr int64_t kTaps = 4;  // bilinear: two rows of the position table by two columns

// One axis of transformers' get_vision_interpolation_indices_and_weights (bilinear over half-pixel centres, taps
// clamped to the table): the table's two rows around index `i` of `size` and their weights, in float32 as torch
// computes them.
void axis_taps(int64_t i, int64_t size, int64_t side, int64_t taps[2], float weights[2]) {
    const float src = (static_cast<float>(i) + 0.5f) * static_cast<float>(side) / static_cast<float>(size) - 0.5f;
    const float floor = std::floor(src);
    for (int k = 0; k < 2; ++k) {
        taps[k] = std::clamp<int64_t>(static_cast<int64_t>(floor) + k, 0, side - 1);
        weights[k] = std::max(0.0f, 1.0f - std::fabs(src - floor - static_cast<float>(k)));
    }
}

// The tower's program runs over T = an image's tokens and one more, each with its merge x merge patches as rows: the
// last holds the class token in its first row, and its other rows are padding no query sees (key_bias). The rows of
// the output are the image tokens' in the prompt's order; the last, the class token's, is nobody's.
class VisionTower final : public VisionEncoder {
public:
    VisionTower(VisionConfig cfg, const WeightStore& weights) : cfg_(std::move(cfg)), weights_(weights) {
        const int64_t values = cfg_.channels * cfg_.patch * cfg_.patch;
        expect("backbone.embeddings.patch_embedding.weight", {cfg_.hidden, values});
        expect("backbone.embeddings.position_embedding", {cfg_.grid * cfg_.grid, cfg_.hidden});
        expect("proj_in.weight", {cfg_.projection, cfg_.merge * cfg_.merge * cfg_.hidden});
        expect("proj_out.weight", {cfg_.out_hidden, cfg_.projection});
        build();
    }

    std::vector<const Graph*> programs() const override { return {&graph_}; }
    int32_t placeholder_token() const override { return cfg_.image_token; }
    int32_t soft_token() const override { return cfg_.image_token; }
    std::vector<int32_t> token_run(int64_t n) const override {
        return std::vector<int32_t>(static_cast<size_t>(n), cfg_.image_token);
    }

    // Inkling2ImageProcessor's: the image as it is, padded down and to the right with -1 (of 0 to 255) to whole
    // squares of patch * merge pixels, one image token each; scaled to [-1, 1]; a token's patches together, each as
    // (y, x, channel). Then each patch's position-table taps and its rotary (row, column), which count from 1: the
    // class token sits at (0, 0).
    MediaInput prepare(const image::Image& img) const override {
        const int64_t m = cfg_.merge, rows = m * m, side = cfg_.patch * m;
        const int64_t th = (img.height + side - 1) / side, tw = (img.width + side - 1) / side;
        const int64_t tokens = th * tw, gh = th * m, gw = tw * m;
        const int64_t values = cfg_.channels * cfg_.patch * cfg_.patch;
        HostTensor pixels(DType::F32, {tokens + 1, rows, values});
        HostTensor pos_idx(DType::I32, {tokens + 1, rows, kTaps});
        HostTensor pos_w(DType::F32, {tokens + 1, rows, kTaps, 1});
        HostTensor rot(DType::I32, {tokens + 1, rows, 2});
        HostTensor is_class(DType::F32, {tokens + 1, rows, 1});
        HostTensor key_bias(DType::F32, {tokens + 1, rows});
        const auto zero = [](HostTensor& t) { std::fill_n(static_cast<char*>(t.raw()), t.nbytes(), 0); };
        zero(pixels);
        zero(pos_idx);
        zero(pos_w);
        zero(rot);
        zero(is_class);
        zero(key_bias);
        // The processor's rescaling folded into its mean and deviation, as transformers computes it.
        const auto scaled = [](float level) { return (level - 127.5f) / 127.5f; };
        const float padding = scaled(-1.0f);
        int64_t t = 0;
        for (int64_t br = 0; br < th; ++br) {
            for (int64_t bc = 0; bc < tw; ++bc, ++t) {
                for (int64_t r = 0; r < rows; ++r) {
                    const int64_t row = br * m + r / m, col = bc * m + r % m, at = t * rows + r;
                    float* px = pixels.data<float>() + at * values;
                    for (int64_t y = 0; y < cfg_.patch; ++y) {
                        for (int64_t x = 0; x < cfg_.patch; ++x) {
                            const int64_t py = row * cfg_.patch + y, px_x = col * cfg_.patch + x;
                            const bool inside = py < img.height && px_x < img.width;
                            for (int64_t c = 0; c < cfg_.channels; ++c) {
                                *px++ = inside ? scaled(img.at(py, px_x, static_cast<int>(c))) : padding;
                            }
                        }
                    }
                    int64_t ht[2], wt[2];
                    float hw[2], ww[2];
                    axis_taps(row, gh, cfg_.grid, ht, hw);
                    axis_taps(col, gw, cfg_.grid, wt, ww);
                    for (int a = 0; a < 2; ++a) {
                        for (int b = 0; b < 2; ++b) {
                            const int64_t tap = at * kTaps + a * 2 + b;
                            pos_idx.data<int32_t>()[tap] = static_cast<int32_t>(ht[a] * cfg_.grid + wt[b]);
                            pos_w.data<float>()[tap] = hw[a] * ww[b];
                        }
                    }
                    rot.data<int32_t>()[at * 2] = static_cast<int32_t>(row + 1);
                    rot.data<int32_t>()[at * 2 + 1] = static_cast<int32_t>(col + 1);
                }
            }
        }
        is_class.data<float>()[tokens * rows] = 1.0f;
        std::fill_n(key_bias.data<float>() + tokens * rows + 1, rows - 1, -std::numeric_limits<float>::infinity());

        MediaInput media;
        media.modality = Modality::Image;
        EncoderRun& run = media.runs.emplace_back();
        run.rows = tokens;
        run.inputs.emplace("pixels", std::move(pixels));
        run.inputs.emplace("pos_idx", std::move(pos_idx));
        run.inputs.emplace("pos_w", std::move(pos_w));
        run.inputs.emplace("rot_pos", std::move(rot));
        run.inputs.emplace("is_class", std::move(is_class));
        run.inputs.emplace("key_bias", std::move(key_bias));
        media.num_tokens = tokens;
        return media;
    }

private:
    void expect(const std::string& name, const Dims& shape) const {
        const TensorView& v = weights_.get(kPrefix + name);
        XS_CHECK(v.shape == shape, "inkling2: ", kPrefix, name, " is ", dims_to_string(v.shape), ", expected ",
                 dims_to_string(shape));
    }
    Value weight(const std::string& name) { return graph_.weight(weights_, kPrefix + name); }
    Value linear(Value x, const std::string& stem) { return torch_nn::linear(graph_, weights_, x, kPrefix + stem); }
    Value layer_norm(Value x, const std::string& stem) {
        return torch_nn::layer_norm(graph_, weights_, x, kPrefix + stem, static_cast<float>(cfg_.norm_eps));
    }

    void build() {
        Graph& g = graph_;
        const int64_t rows = cfg_.merge * cfg_.merge, d = cfg_.hidden, head_dim = d / cfg_.heads;
        const int64_t values = cfg_.channels * cfg_.patch * cfg_.patch;
        const Value pixels = g.input("pixels", DType::F32, {kT, rows, values});
        const Value pos_idx = g.input("pos_idx", DType::I32, {kT, rows, kTaps});
        const Value pos_w = g.input("pos_w", DType::F32, {kT, rows, kTaps, 1});
        const Value rot_pos = g.input("rot_pos", DType::I32, {kT, rows, 2});
        const Value is_class = g.input("is_class", DType::F32, {kT, rows, 1});
        const Value key_bias = g.input("key_bias", DType::F32, {kT, rows});

        // A patch's embedding and its place in the interpolated position table; the class token's two vectors where
        // is_class says (its pixels and taps are zeros).
        Value x;
        {
            Graph::Scope s(g, "embeddings");
            const std::string e = "backbone.embeddings.";
            x = g.matmul(pixels, weight(e + "patch_embedding.weight"));
            x = g.add(x, g.reduce_sum(g.mul(g.gather(weight(e + "position_embedding"), pos_idx), pos_w), 2));
            x = g.add(x, g.mul(is_class, g.add(weight(e + "class_embedding"), weight(e + "cls_position_embedding"))));
            x = layer_norm(x, "backbone.ln_pre");
        }

        // 2-D rotary over the whole head: its channels pair up as neighbours (2 i, 2 i + 1), the first half of the
        // pairs turning with the column and the second with the row, by the same frequencies. The graph's Rope pairs
        // channel c with c + D / 2, so queries and keys go through it with their channels in that order; their
        // products do not care which.
        const int64_t pairs = head_dim / 2, quarter = head_dim / 4;
        std::vector<float> inv_freq(static_cast<size_t>(pairs));
        std::vector<int32_t> axes(static_cast<size_t>(pairs));
        for (int64_t i = 0; i < pairs; ++i) {
            inv_freq[static_cast<size_t>(i)] = static_cast<float>(
                1.0 / std::pow(cfg_.rope_theta, static_cast<double>(2 * (i % quarter)) / static_cast<double>(pairs)));
            axes[static_cast<size_t>(i)] = i < quarter ? 1 : 0;  // rot_pos is (row, column)
        }
        const Value freq = g.constant(HostTensor::from_vector(DType::F32, {pairs}, inv_freq));
        const Value rope_axes = g.constant(HostTensor::from_vector(DType::I32, {pairs}, axes));
        const auto rotated = [&](Value projected) {
            const Value by_pair = g.reshape(projected, {kT, rows, cfg_.heads, pairs, 2});
            const Value halves = g.reshape(g.transpose(by_pair, {0, 1, 2, 4, 3}), {kT, rows, cfg_.heads, head_dim});
            return g.rope(halves, rot_pos, freq, 0, rope_axes);
        };

        graph::AttentionAttrs attrs;
        attrs.causal = false;
        attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
        for (int64_t i = 0; i < cfg_.layers; ++i) {
            const std::string l = "backbone.layers." + std::to_string(i) + ".";
            Graph::Scope s(g, "layers." + std::to_string(i));
            const Value normed = layer_norm(x, l + "ln_1");
            const Value q = rotated(linear(normed, l + "self_attn.q_proj"));
            const Value k = rotated(linear(normed, l + "self_attn.k_proj"));
            const Value v = g.reshape(linear(normed, l + "self_attn.v_proj"), {kT, rows, cfg_.heads, head_dim});
            const Value attn = g.reshape(g.attention(q, k, v, key_bias, attrs), {kT, rows, d});
            x = g.add(x, g.mul(linear(attn, l + "self_attn.out_proj"), weight(l + "ls_1.gamma")));
            const Value mlp = linear(g.gelu(linear(layer_norm(x, l + "ln_2"), l + "mlp.c_fc")), l + "mlp.c_proj");
            x = g.add(x, g.mul(mlp, weight(l + "ls_2.gamma")));
        }
        // A token's patches side by side through the projection.
        Graph::Scope s(g, "projection");
        const Value joined = g.reshape(x, {kT, rows * d});
        const Value out = g.matmul(g.gelu(g.matmul(joined, weight("proj_in.weight"))), weight("proj_out.weight"));
        g.output("embeddings", out);
        g.validate();
    }

    VisionConfig cfg_;
    const WeightStore& weights_;
    Graph graph_;
};

}  // namespace

std::unique_ptr<VisionEncoder> make_vision(const VisionConfig& cfg, const WeightStore& weights) {
    return std::make_unique<VisionTower>(cfg, weights);
}

}  // namespace xs::inkling2
