#include <algorithm>
#include <cmath>
#include <utility>

#include "image/image.h"
#include "models/common/torch_nn.h"
#include "models/qwen3_vision/qwen3_vision.h"
#include "util/error.h"

namespace xs::qwen3_vision {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

constexpr float kLayerNormEps = 1e-6f;
constexpr int64_t kTaps = 4;  // bilinear: two rows of the position table by two columns

// One axis of transformers' get_vision_interpolation_indices_and_weights (bilinear, align_corners): the table's two
// rows around index `i` of `size` and their weights, in float32 as torch computes them.
void axis_taps(int64_t i, int64_t size, int64_t side, int64_t taps[2], float weights[2]) {
    const float span = static_cast<float>(std::max<int64_t>(size - 1, 1));
    const float src = static_cast<float>(i) * static_cast<float>(side - 1) / span;
    const float floor = std::floor(src);
    for (int k = 0; k < 2; ++k) {
        taps[k] = std::clamp<int64_t>(static_cast<int64_t>(floor) + k, 0, side - 1);
        weights[k] = std::max(0.0f, 1.0f - std::fabs(src - floor - static_cast<float>(k)));
    }
}

class VisionTower final : public VisionEncoder {
public:
    VisionTower(Config cfg, Processor proc, const WeightStore& weights, std::string prefix, int32_t image_token)
        : cfg_(std::move(cfg)), proc_(proc), weights_(weights), prefix_(std::move(prefix)), image_token_(image_token) {
        // Qwen3-Omni names a merger's layers ln_q, mlp.0, mlp.2; Qwen3-VL and Qwen3.5 norm, linear_fc1, linear_fc2.
        if (!weights_.contains(prefix_ + "merger.ln_q.weight")) merger_names_ = {".norm", ".linear_fc1", ".linear_fc2"};
        const TensorView& table = weights_.get(prefix_ + "pos_embed.weight");
        side_ = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(table.shape[0]))));
        XS_CHECK(side_ * side_ == table.shape[0] && table.shape[1] == cfg_.hidden, "qwen3_vision: ", prefix_,
                 "pos_embed.weight is ", dims_to_string(table.shape), ", not a square grid of rows of ", cfg_.hidden);
        build();
    }

    std::vector<const Graph*> programs() const override { return {&graph_}; }
    int32_t placeholder_token() const override { return image_token_; }
    int32_t soft_token() const override { return image_token_; }
    std::vector<int32_t> token_run(int64_t n) const override {
        return std::vector<int32_t>(static_cast<size_t>(n), image_token_);
    }

    // Qwen2VLImageProcessor's: resized (bicubic) to a multiple of patch * merge a side within the pixel bounds,
    // scaled to [0, 1], normalized, cut into patches of `temporal_patch` copies of the frame, in merge-block order
    // (a token's patches together); then each patch's position-table taps and its rotary (row, column).
    MediaInput prepare(const image::Image& img) const override {
        const int64_t factor = cfg_.patch * cfg_.merge;
        const auto [h, w] = smart_resize(img.height, img.width, factor, proc_.min_pixels, proc_.max_pixels);
        const image::Image resized = h == img.height && w == img.width ? img : image::resize_bicubic(img, w, h);
        const int64_t gh = h / cfg_.patch, gw = w / cfg_.patch, m = cfg_.merge, rows = m * m;
        const int64_t tokens = (gh / m) * (gw / m);
        const int64_t patch_values = cfg_.channels * cfg_.temporal_patch * cfg_.patch * cfg_.patch;
        HostTensor pixels(DType::F32, {tokens, rows, patch_values});
        HostTensor pos_idx(DType::I32, {tokens, rows, kTaps});
        HostTensor pos_w(DType::F32, {tokens, rows, kTaps, 1});
        HostTensor rot(DType::I32, {tokens, rows, 2});
        MediaInput media;
        media.modality = Modality::Image;
        int64_t t = 0;
        for (int64_t br = 0; br < gh / m; ++br) {
            for (int64_t bc = 0; bc < gw / m; ++bc, ++t) {
                media.positions.push_back({0, static_cast<int32_t>(br), static_cast<int32_t>(bc)});
                for (int64_t r = 0; r < rows; ++r) {
                    const int64_t row = br * m + r / m, col = bc * m + r % m, at = t * rows + r;
                    float* px = pixels.data<float>() + at * patch_values;
                    for (int64_t c = 0; c < cfg_.channels; ++c) {
                        for (int64_t f = 0; f < cfg_.temporal_patch; ++f) {
                            for (int64_t y = 0; y < cfg_.patch; ++y) {
                                for (int64_t x = 0; x < cfg_.patch; ++x) {
                                    const int64_t py = row * cfg_.patch + y, px_x = col * cfg_.patch + x;
                                    const float v = resized.at(py, px_x, static_cast<int>(c));
                                    *px++ = (v / 255.0f - proc_.mean[c]) / proc_.std[c];
                                }
                            }
                        }
                    }
                    int64_t ht[2], wt[2];
                    float hw[2], ww[2];
                    axis_taps(row, gh, side_, ht, hw);
                    axis_taps(col, gw, side_, wt, ww);
                    for (int a = 0; a < 2; ++a) {
                        for (int b = 0; b < 2; ++b) {
                            const int64_t tap = at * kTaps + a * 2 + b;
                            pos_idx.data<int32_t>()[tap] = static_cast<int32_t>(ht[a] * side_ + wt[b]);
                            pos_w.data<float>()[tap] = hw[a] * ww[b];
                        }
                    }
                    rot.data<int32_t>()[at * 2] = static_cast<int32_t>(row);
                    rot.data<int32_t>()[at * 2 + 1] = static_cast<int32_t>(col);
                }
            }
        }
        EncoderRun& run = media.runs.emplace_back();
        run.rows = tokens;
        run.inputs.emplace("pixels", std::move(pixels));
        run.inputs.emplace("pos_idx", std::move(pos_idx));
        run.inputs.emplace("pos_w", std::move(pos_w));
        run.inputs.emplace("rot_pos", std::move(rot));
        media.num_tokens = tokens;
        return media;
    }

private:
    Value linear(Value x, const std::string& stem) { return torch_nn::linear(graph_, weights_, x, prefix_ + stem); }
    Value layer_norm(Value x, const std::string& stem) {
        return torch_nn::layer_norm(graph_, weights_, x, prefix_ + stem, kLayerNormEps);
    }

    // A merger over each token's patch rows x [T, 4, hidden]: LayerNorm per patch (the last merger) or over the four
    // patches joined (deepstack's), then Linear, exact GELU, Linear to the decoder's width.
    Value merger(Value x, const std::string& stem, bool after_join) {
        Graph& g = graph_;
        const int64_t joined = cfg_.merge * cfg_.merge * cfg_.hidden;
        const MergerNames& n = merger_names_;
        x = after_join ? layer_norm(g.reshape(x, {kT, joined}), stem + n.norm)
                       : g.reshape(layer_norm(x, stem + n.norm), {kT, joined});
        return linear(g.gelu(linear(x, stem + n.fc1)), stem + n.fc2);
    }

    void build() {
        Graph& g = graph_;
        const int64_t rows = cfg_.merge * cfg_.merge, d = cfg_.hidden, head_dim = d / cfg_.heads;
        const int64_t patch_values = cfg_.channels * cfg_.temporal_patch * cfg_.patch * cfg_.patch;
        const Value pixels = g.input("pixels", DType::F32, {kT, rows, patch_values});
        const Value pos_idx = g.input("pos_idx", DType::I32, {kT, rows, kTaps});
        const Value pos_w = g.input("pos_w", DType::F32, {kT, rows, kTaps, 1});
        const Value rot_pos = g.input("rot_pos", DType::I32, {kT, rows, 2});

        // The Conv3d of stride = kernel is a Linear over a patch's values (channel, frame, y, x); a GGUF import keeps
        // it as one kernel a frame (llama.cpp's converter cuts it so): the sum of each frame's Linear.
        Value x;
        {
            Graph::Scope s(g, "patch_embed");
            const std::string whole = prefix_ + "patch_embed.proj.weight";
            if (weights_.contains(whole)) {
                x = g.matmul(pixels, g.reshape(g.weight(weights_, whole), {d, patch_values}));
            } else {
                const int64_t frame_values = patch_values / cfg_.temporal_patch;
                const Value frames = g.reshape(pixels, {kT, rows, cfg_.channels, cfg_.temporal_patch,
                                                        frame_values / cfg_.channels});
                for (int64_t t = 0; t < cfg_.temporal_patch; ++t) {
                    const std::string name = prefix_ + "patch_embed.proj.frames." + std::to_string(t) + ".weight";
                    const Value frame = g.reshape(g.slice(frames, 3, t, t + 1), {kT, rows, frame_values});
                    const Value y = g.matmul(frame, g.reshape(g.weight(weights_, name), {d, frame_values}));
                    x = t == 0 ? y : g.add(x, y);
                }
            }
            x = g.add(x, g.weight(weights_, prefix_ + "patch_embed.proj.bias"));
            const Value taps = g.gather(g.weight(weights_, prefix_ + "pos_embed.weight"), pos_idx);
            x = g.add(x, g.reduce_sum(g.mul(taps, pos_w), 2));
        }

        // 2-D rotary: the head's first half of frequencies reads the row, the second the column, the same frequencies.
        const int64_t pairs = head_dim / 2, quarter = head_dim / 4;
        std::vector<float> inv_freq(static_cast<size_t>(pairs));
        std::vector<int32_t> axes(static_cast<size_t>(pairs));
        for (int64_t i = 0; i < pairs; ++i) {
            const int64_t j = i % quarter;
            inv_freq[static_cast<size_t>(i)] = static_cast<float>(
                1.0 / std::pow(cfg_.rope_theta, static_cast<double>(2 * j) / static_cast<double>(pairs)));
            axes[static_cast<size_t>(i)] = i < quarter ? 0 : 1;
        }
        const Value freq = g.constant(HostTensor::from_vector(DType::F32, {pairs}, inv_freq));
        const Value rope_axes = g.constant(HostTensor::from_vector(DType::I32, {pairs}, axes));

        graph::AttentionAttrs attrs;
        attrs.causal = false;
        attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
        std::vector<Value> rows_out;  // the merged row, then each deepstack row
        for (int64_t i = 0; i < cfg_.depth; ++i) {
            const std::string B = "blocks." + std::to_string(i) + ".";
            Graph::Scope s(g, "blocks." + std::to_string(i));
            const Value qkv = linear(layer_norm(x, B + "norm1"), B + "attn.qkv");
            const auto part = [&](int64_t k) {
                return g.reshape(g.slice(qkv, 2, k * d, (k + 1) * d), {kT, rows, cfg_.heads, head_dim});
            };
            const Value q = g.rope(part(0), rot_pos, freq, 0, rope_axes);
            const Value k = g.rope(part(1), rot_pos, freq, 0, rope_axes);
            const Value attn = g.attention(q, k, part(2), attrs);
            x = g.add(x, linear(g.reshape(attn, {kT, rows, d}), B + "attn.proj"));
            x = g.add(x, linear(g.gelu_tanh(linear(layer_norm(x, B + "norm2"), B + "mlp.linear_fc1")),
                                B + "mlp.linear_fc2"));
            for (size_t j = 0; j < cfg_.deepstack.size(); ++j) {
                if (cfg_.deepstack[j] == i) rows_out.push_back(merger(x, "merger_list." + std::to_string(j), true));
            }
        }
        rows_out.insert(rows_out.begin(), merger(x, "merger", false));
        g.output("embeddings", rows_out.size() == 1 ? rows_out[0] : g.concat(rows_out, 1));
        g.validate();
    }

    Config cfg_;
    Processor proc_;
    const WeightStore& weights_;
    std::string prefix_;
    int32_t image_token_;
    int64_t side_ = 0;  // the position table's grid side
    struct MergerNames {
        std::string norm, fc1, fc2;
    };
    MergerNames merger_names_{".ln_q", ".mlp.0", ".mlp.2"};
    Graph graph_;
};

}  // namespace

std::unique_ptr<VisionEncoder> make_encoder(const Config& c, const Processor& p, const WeightStore& weights,
                                            std::string prefix, int32_t image_token) {
    return std::make_unique<VisionTower>(c, p, weights, std::move(prefix), image_token);
}

}  // namespace xs::qwen3_vision
