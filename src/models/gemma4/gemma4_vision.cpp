// Gemma 4 vision: the image processor (Gemma4ImageProcessor) and the vision tower (Gemma4VisionModel followed by
// Gemma4MultimodalEmbedder, the projection into the text model's hidden size).
//
// The tower runs one image per graph run with T = the image's patch count, so no padding to the 2520-patch budget is
// needed: Hugging Face masks its padding patches out as attention keys and out of the pooling, which leaves the real
// patches' results exactly as an unpadded run computes them.

#include <cmath>
#include <fstream>

#include "models/config_util.h"
#include "models/gemma4/gemma4.h"
#include "util/error.h"
#include "util/half.h"
#include "util/log.h"

namespace xs::gemma4 {

namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

// Values that change the pixels the tower sees. Anything else in the processor config is irrelevant here.
void read_image_processor(const std::filesystem::path& model_dir, VisionConfig& vc) {
    const std::filesystem::path path = model_dir / "processor_config.json";
    if (!std::filesystem::exists(path)) return;
    const nlohmann::json root = cfg::read_json(path);
    if (!root.contains("image_processor")) return;
    const nlohmann::json& p = root.at("image_processor");
    const std::string where = "processor_config.json image_processor";
    if (p.contains("patch_size") && cfg::get_int(p, "patch_size", where) != vc.patch_size) {
        fail(where, ": patch_size differs from the vision tower's ", vc.patch_size);
    }
    if (p.contains("pooling_kernel_size") && cfg::get_int(p, "pooling_kernel_size", where) != vc.pooling_kernel) {
        fail(where, ": pooling_kernel_size differs from the vision tower's ", vc.pooling_kernel);
    }
    vc.max_soft_tokens = cfg::get_int_or(p, "max_soft_tokens", vc.max_soft_tokens, where);
    if (!cfg::get_bool_or(p, "do_resize", true, where)) fail(where, ": do_resize=false is not supported");
    if (cfg::get_int_or(p, "resample", 3, where) != 3) fail(where, ": only bicubic resampling (3) is supported");
    if (!cfg::get_bool_or(p, "do_rescale", true, where)) {
        vc.rescale_factor = 1.0f;
    } else if (p.contains("rescale_factor")) {
        vc.rescale_factor = static_cast<float>(cfg::get_float(p, "rescale_factor", where));
    }
    if (cfg::get_bool_or(p, "do_normalize", false, where)) {
        const auto identity = [&](const char* key, double value) {
            if (!p.contains(key)) return true;
            for (const auto& v : p.at(key)) {
                if (!v.is_number() || v.get<double>() != value) return false;
            }
            return true;
        };
        if (!identity("image_mean", 0.0) || !identity("image_std", 1.0)) {
            fail(where, ": do_normalize with a mean or std other than 0 and 1 is not supported");
        }
    }
}

int32_t token_id(const nlohmann::json& root, const char* key) {
    return static_cast<int32_t>(cfg::get_int(root, key, "config.json"));
}

float scalar_weight(const WeightStore& weights, const std::string& name) {
    const TensorView& t = weights.get(name);
    XS_CHECK(t.numel() == 1, "gemma4 vision: '", name, "' should hold one value, has shape ", dims_to_string(t.shape));
    switch (t.dtype) {
        case DType::F32: return *static_cast<const float*>(t.data);
        case DType::BF16: return bf16_to_f32(*static_cast<const uint16_t*>(t.data));
        case DType::F16: return f16_to_f32(*static_cast<const uint16_t*>(t.data));
        default: fail("gemma4 vision: '", name, "' has dtype ", dtype_name(t.dtype));
    }
}

class Gemma4Vision final : public VisionEncoder {
public:
    Gemma4Vision(VisionConfig vc, const WeightStore& weights, std::string prefix)
        : vc_(std::move(vc)), weights_(weights), prefix_(std::move(prefix)) {
        build_graph();
    }

    const graph::Graph& graph() const override { return graph_; }
    int32_t placeholder_token() const override { return vc_.image_token_id; }
    int32_t soft_token() const override { return vc_.image_token_id; }

    std::vector<int32_t> token_run(int64_t num_tokens) const override {
        std::vector<int32_t> run;
        run.reserve(static_cast<size_t>(num_tokens) + 2);
        run.push_back(vc_.boi_token_id);
        run.insert(run.end(), static_cast<size_t>(num_tokens), vc_.image_token_id);
        run.push_back(vc_.eoi_token_id);
        return run;
    }

    // Gemma4ImageProcessor: aspect-preserving resize, rescale, 16x16 patches of (row, column, channel) values in
    // row-major patch order, (x, y) patch positions; then the tower's 2 * (x - 0.5) and the k x k pooling map.
    MediaInput prepare(const image::Image& img) const override {
        const auto [height, width] = image_target_size(img.height, img.width, vc_);
        const image::Image resized = image::resize_bicubic(img, width, height);
        const int64_t p = vc_.patch_size;
        const int64_t grid_h = height / p;
        const int64_t grid_w = width / p;
        const int64_t patches = grid_h * grid_w;
        const int64_t patch_dim = p * p * 3;

        HostTensor pixels(DType::F32, {patches, patch_dim});
        HostTensor positions(DType::I32, {patches, 2});
        float* px = pixels.data<float>();
        for (int64_t gy = 0; gy < grid_h; ++gy) {
            for (int64_t gx = 0; gx < grid_w; ++gx) {
                const int64_t patch = gy * grid_w + gx;
                positions.data<int32_t>()[2 * patch] = static_cast<int32_t>(gx);
                positions.data<int32_t>()[2 * patch + 1] = static_cast<int32_t>(gy);
                float* out = px + patch * patch_dim;
                for (int64_t dy = 0; dy < p; ++dy) {
                    for (int64_t dx = 0; dx < p; ++dx) {
                        for (int c = 0; c < 3; ++c) {
                            // f32 arithmetic in the reference's order: rescale, then 2 * (x - 0.5).
                            const float v = static_cast<float>(resized.at(gy * p + dy, gx * p + dx, c)) * vc_.rescale_factor;
                            *out++ = 2.0f * (v - 0.5f);
                        }
                    }
                }
            }
        }

        // Soft token j averages the k x k patches with floor(x / k) + (grid_w / k) * floor(y / k) == j.
        const int64_t k = vc_.pooling_kernel;
        HostTensor pool(DType::F32, {vc_.max_soft_tokens, patches});
        const float weight = 1.0f / static_cast<float>(k * k);
        for (int64_t patch = 0; patch < patches; ++patch) {
            const int64_t x = patch % grid_w;
            const int64_t y = patch / grid_w;
            const int64_t token = x / k + (grid_w / k) * (y / k);
            pool.data<float>()[token * patches + patch] = weight;
        }

        MediaInput media;
        media.num_tokens = patches / (k * k);
        media.encoder_inputs.emplace("pixels", std::move(pixels));
        media.encoder_inputs.emplace("patch_pos", std::move(positions));
        media.encoder_inputs.emplace("pool", std::move(pool));
        return media;
    }

private:
    Value weight(const std::string& name) { return graph_.weight(weights_, prefix_ + name); }
    float eps() const { return static_cast<float>(vc_.rms_norm_eps); }

    // Gemma4ClippableLinear: clamp the input, matmul, clamp the output (bounds are checkpoint scalars).
    Value linear(Value x, const std::string& name) {
        if (!vc_.clipped_linears) return graph_.matmul(x, weight(name + ".linear.weight"));
        const auto bound = [&](const char* which) { return scalar_weight(weights_, prefix_ + name + "." + which); };
        x = graph_.clamp(x, bound("input_min"), bound("input_max"));
        x = graph_.matmul(x, weight(name + ".linear.weight"));
        return graph_.clamp(x, bound("output_min"), bound("output_max"));
    }

    void build_graph() {
        Graph& g = graph_;
        const int64_t hidden = vc_.hidden_size;
        const int64_t patch_dim = vc_.patch_size * vc_.patch_size * 3;
        const Value pixels = g.input("pixels", DType::F32, {kT, patch_dim});
        const Value patch_pos = g.input("patch_pos", DType::I32, {kT, 2});
        const Value pool = g.input("pool", DType::F32, {vc_.max_soft_tokens, kT});

        // Patch embedding plus the learned x and y position rows.
        Value h;
        {
            Graph::Scope s(g, "patch_embedder");
            h = g.matmul(pixels, weight("vision_tower.patch_embedder.input_proj.weight"));
            const Value table = weight("vision_tower.patch_embedder.position_embedding_table");
            const Dims& ts = g.type(table).shape;
            XS_CHECK(ts == (Dims{2, vc_.position_embedding_size, hidden}), "gemma4 vision: position table has shape ",
                     dims_to_string(ts));
            for (int axis = 0; axis < 2; ++axis) {
                const Value rows = g.reshape(g.slice(table, 0, axis, axis + 1), {vc_.position_embedding_size, hidden});
                const Value ids = g.reshape(g.slice(patch_pos, 1, axis, axis + 1), {kT});
                h = g.add(h, g.gather(rows, ids));
            }
        }

        // Axial RoPE: the head dim splits into an x half and a y half, each rotated with head_dim / 4 frequencies.
        const int64_t pairs = vc_.head_dim / 4;
        HostTensor inv_freq(DType::F32, {pairs});
        for (int64_t i = 0; i < pairs; ++i) {
            inv_freq.data<float>()[i] = static_cast<float>(
                1.0 / std::pow(vc_.rope_theta, static_cast<double>(2 * i) / static_cast<double>(vc_.head_dim / 2)));
        }
        const Value freq = g.constant(std::move(inv_freq));

        for (int64_t i = 0; i < vc_.num_layers; ++i) h = encoder_layer(i, h, patch_pos, freq);

        // Gemma4VisionPooler, then Gemma4MultimodalEmbedder. The pooler scales by sqrt(hidden) right before a
        // weightless RMSNorm, and rms_norm(c * x, eps) == rms_norm(x, eps / c^2): folding the scale into eps keeps
        // the f16 range (the scaled values can pass 65504) with the same math.
        Graph::Scope s(g, "embed_vision");
        const Value pooled = g.matmul(pool, g.transpose(h, {1, 0}));  // [max_soft_tokens, hidden]
        const Value normed = g.rms_norm(pooled, eps() / static_cast<float>(hidden));
        g.output("embeddings", g.matmul(normed, weight("embed_vision.embedding_projection.weight")));
        g.validate();
    }

    Value encoder_layer(int64_t i, Value h, Value patch_pos, Value freq) {
        Graph& g = graph_;
        const std::string L = "vision_tower.encoder.layers." + std::to_string(i) + ".";
        Graph::Scope s(g, "vision.layers." + std::to_string(i));
        const int64_t hd = vc_.head_dim;

        Value x = g.rms_norm(h, weight(L + "input_layernorm.weight"), eps());
        Value q = g.reshape(linear(x, L + "self_attn.q_proj"), {kT, vc_.num_heads, hd});
        q = g.rope(g.rms_norm(q, weight(L + "self_attn.q_norm.weight"), eps()), patch_pos, freq);
        Value k = g.reshape(linear(x, L + "self_attn.k_proj"), {kT, vc_.num_kv_heads, hd});
        k = g.rope(g.rms_norm(k, weight(L + "self_attn.k_norm.weight"), eps()), patch_pos, freq);
        const Value v = g.rms_norm(g.reshape(linear(x, L + "self_attn.v_proj"), {kT, vc_.num_kv_heads, hd}), eps());
        graph::AttentionAttrs attrs;
        attrs.causal = false;
        attrs.scale = 1.0f;  // q_norm / k_norm carry the scale
        const Value o = g.attention(q, k, v, attrs);
        x = linear(g.reshape(o, {kT, vc_.num_heads * hd}), L + "self_attn.o_proj");
        h = g.add(h, g.rms_norm(x, weight(L + "post_attention_layernorm.weight"), eps()));

        x = g.rms_norm(h, weight(L + "pre_feedforward_layernorm.weight"), eps());
        x = g.mul(g.gelu_tanh(linear(x, L + "mlp.gate_proj")), linear(x, L + "mlp.up_proj"));
        x = linear(x, L + "mlp.down_proj");
        return g.add(h, g.rms_norm(x, weight(L + "post_feedforward_layernorm.weight"), eps()));
    }

    const VisionConfig vc_;
    const WeightStore& weights_;
    const std::string prefix_;
    graph::Graph graph_;
};

}  // namespace

std::optional<VisionConfig> parse_vision_config(const nlohmann::json& config_json,
                                                const std::filesystem::path& model_dir, int64_t text_hidden_size) {
    if (!config_json.contains("vision_config") || config_json.at("vision_config").is_null()) return std::nullopt;
    const nlohmann::json& v = config_json.at("vision_config");
    const std::string where = "config.json vision_config";
    const std::string type = cfg::get_string(v, "model_type", where);
    if (type != "gemma4_vision") fail(where, ": model_type '", type, "' is not supported (supported: gemma4_vision)");
    if (cfg::get_bool_or(v, "standardize", false, where)) {
        fail(where, ": standardize (the 26B / 31B vision towers) is not supported yet");
    }
    if (cfg::get_bool_or(v, "attention_bias", false, where)) fail(where, ": attention_bias is not supported");
    const std::string act = cfg::get_string(v, "hidden_activation", where);
    if (act != "gelu_pytorch_tanh") fail(where, ": hidden_activation '", act, "' is not supported");

    VisionConfig vc;
    vc.hidden_size = cfg::get_int(v, "hidden_size", where);
    vc.intermediate_size = cfg::get_int(v, "intermediate_size", where);
    vc.num_layers = cfg::get_int(v, "num_hidden_layers", where);
    vc.num_heads = cfg::get_int(v, "num_attention_heads", where);
    vc.num_kv_heads = cfg::get_int_or(v, "num_key_value_heads", vc.num_heads, where);
    vc.head_dim = cfg::get_int_or(v, "head_dim", vc.hidden_size / vc.num_heads, where);
    vc.patch_size = cfg::get_int(v, "patch_size", where);
    vc.pooling_kernel = cfg::get_int(v, "pooling_kernel_size", where);
    vc.position_embedding_size = cfg::get_int(v, "position_embedding_size", where);
    vc.max_soft_tokens = cfg::get_int_or(v, "default_output_length", 280, where);
    vc.rms_norm_eps = cfg::get_float(v, "rms_norm_eps", where);
    vc.clipped_linears = cfg::get_bool_or(v, "use_clipped_linears", false, where);
    if (v.contains("rope_parameters")) {
        const nlohmann::json& r = v.at("rope_parameters");
        const std::string rope_type = cfg::get_string(r, "rope_type", where + " rope_parameters");
        if (rope_type != "default" && rope_type != "axial") {
            fail(where, ": rope_type '", rope_type, "' is not supported (supported: default, axial)");
        }
        vc.rope_theta = cfg::get_float(r, "rope_theta", where + " rope_parameters");
    }
    vc.text_hidden_size = text_hidden_size;
    vc.image_token_id = token_id(config_json, "image_token_id");
    vc.boi_token_id = token_id(config_json, "boi_token_id");
    vc.eoi_token_id = token_id(config_json, "eoi_token_id");
    read_image_processor(model_dir, vc);

    XS_CHECK(vc.num_heads % vc.num_kv_heads == 0 && vc.head_dim % 4 == 0, where, ": ", vc.num_heads, " heads, ",
             vc.num_kv_heads, " kv heads and head_dim ", vc.head_dim, " do not fit 2-D rope with grouped heads");
    XS_CHECK(vc.patch_size > 0 && vc.pooling_kernel > 0 && vc.max_soft_tokens > 0, where,
             ": patch_size, pooling_kernel_size and the soft-token budget must be positive");
    return vc;
}

std::pair<int64_t, int64_t> image_target_size(int64_t height, int64_t width, const VisionConfig& vc) {
    XS_CHECK(height > 0 && width > 0, "image of ", width, "x", height, " pixels");
    const int64_t k2 = vc.pooling_kernel * vc.pooling_kernel;
    const int64_t max_patches = vc.max_soft_tokens * k2;
    const double total_px = static_cast<double>(height) * static_cast<double>(width);
    const auto target_px = static_cast<double>(max_patches * vc.patch_size * vc.patch_size);
    const double factor = std::sqrt(target_px / total_px);
    const int64_t side = vc.pooling_kernel * vc.patch_size;
    int64_t th = static_cast<int64_t>(std::floor(factor * static_cast<double>(height) / static_cast<double>(side))) * side;
    int64_t tw = static_cast<int64_t>(std::floor(factor * static_cast<double>(width) / static_cast<double>(side))) * side;
    XS_CHECK(th > 0 || tw > 0, "image of ", width, "x", height, " pixels cannot be resized for the vision tower");
    const int64_t max_side = (max_patches / k2) * side;
    if (th == 0) {
        th = side;
        tw = std::min(width / height * side, max_side);
    } else if (tw == 0) {
        tw = side;
        th = std::min(height / width * side, max_side);
    }
    XS_CHECK(static_cast<double>(th) * static_cast<double>(tw) <= target_px, "image of ", width, "x", height,
             " pixels does not fit the vision tower's budget of ", max_patches, " patches");
    return {th, tw};
}

std::unique_ptr<VisionEncoder> make_vision_encoder(const VisionConfig& vc, const WeightStore& weights,
                                                   const std::string& prefix) {
    return std::make_unique<Gemma4Vision>(vc, weights, prefix);
}

}  // namespace xs::gemma4
