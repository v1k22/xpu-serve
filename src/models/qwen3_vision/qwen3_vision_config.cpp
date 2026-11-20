#include <algorithm>
#include <cmath>

#include "models/config_util.h"
#include "models/qwen3_vision/qwen3_vision.h"
#include "util/error.h"

namespace xs::qwen3_vision {

Config parse_config(const nlohmann::json& v, const std::string& where) {
    const std::string act = cfg::get_string(v, "hidden_act", where);
    if (act != "gelu_pytorch_tanh") fail(where, ": hidden_act '", act, "' is not supported (gelu_pytorch_tanh is)");
    if (!cfg::get_bool_or(v, "apply_vit_abs_pos_embed", true, where)) {
        fail(where, ": a tower without its learned position table (apply_vit_abs_pos_embed false) is not supported");
    }
    Config c;
    c.depth = cfg::get_int(v, "depth", where);
    c.hidden = cfg::get_int(v, "hidden_size", where);
    c.intermediate = cfg::get_int(v, "intermediate_size", where);
    c.heads = cfg::get_int(v, "num_heads", where);
    c.out_hidden = cfg::get_int(v, "out_hidden_size", where);
    c.patch = cfg::get_int(v, "patch_size", where);
    c.merge = cfg::get_int(v, "spatial_merge_size", where);
    c.temporal_patch = cfg::get_int(v, "temporal_patch_size", where);
    c.channels = v.contains("in_channels") ? cfg::get_int(v, "in_channels", where)
                                           : cfg::get_int_or(v, "in_chans", 3, where);
    if (v.contains("deepstack_visual_indexes")) {
        for (const int32_t i : cfg::get_int_list(v, "deepstack_visual_indexes", where)) c.deepstack.push_back(i);
    }
    if (v.contains("rope_parameters") && v.at("rope_parameters").contains("rope_theta")) {
        c.rope_theta = cfg::get_float(v.at("rope_parameters"), "rope_theta", where);
    }
    XS_CHECK(c.hidden % c.heads == 0 && (c.hidden / c.heads) % 4 == 0, where, ": hidden_size ", c.hidden, " over ",
             c.heads, " heads is not a head width of a multiple of 4 (two axes of rotary pairs)");
    XS_CHECK(c.merge == 2 && c.channels == 3 && c.temporal_patch >= 1, where, ": spatial_merge_size ", c.merge,
             ", in_channels ", c.channels, " (2 and 3 are supported)");
    for (const int64_t i : c.deepstack) {
        XS_CHECK(i >= 0 && i < c.depth, where, ": deepstack_visual_indexes holds block ", i, " of ", c.depth);
    }
    return c;
}

Processor processor_spec(const std::filesystem::path& dir, const Config& c, int64_t max_pixels) {
    Processor p;
    const std::filesystem::path path = dir / "preprocessor_config.json";
    if (!std::filesystem::exists(path)) {
        if (max_pixels > 0) p.max_pixels = std::min(p.max_pixels, max_pixels);
        return p;
    }
    const nlohmann::json j = cfg::read_json(path);
    const std::string where = "preprocessor_config.json";
    if (j.contains("min_pixels")) p.min_pixels = cfg::get_int(j, "min_pixels", where);
    if (j.contains("max_pixels")) p.max_pixels = cfg::get_int(j, "max_pixels", where);
    if (j.contains("size") && j.at("size").is_object()) {
        const nlohmann::json& size = j.at("size");
        if (size.contains("shortest_edge")) p.min_pixels = cfg::get_int(size, "shortest_edge", where);
        if (size.contains("longest_edge")) p.max_pixels = cfg::get_int(size, "longest_edge", where);
    }
    for (const auto& [key, out] : {std::pair{"image_mean", p.mean}, std::pair{"image_std", p.std}}) {
        if (!j.contains(key)) continue;
        const nlohmann::json& values = j.at(key);
        XS_CHECK(values.is_array() && values.size() == 3, where, ": ", key, " must be three numbers");
        for (size_t i = 0; i < 3; ++i) out[i] = values[i].get<float>();
    }
    for (const auto& [key, expected] : {std::pair{"patch_size", c.patch}, std::pair{"merge_size", c.merge},
                                        std::pair{"temporal_patch_size", c.temporal_patch}}) {
        if (j.contains(key)) {
            XS_CHECK(cfg::get_int(j, key, where) == expected, where, ": ", key, " ", cfg::get_int(j, key, where),
                     " is not the vision tower's ", expected);
        }
    }
    if (max_pixels > 0) {
        XS_CHECK(max_pixels >= p.min_pixels, "--image-max-pixels ", max_pixels, " is below the processor's least ",
                 p.min_pixels, " pixels");
        p.max_pixels = std::min(p.max_pixels, max_pixels);
    }
    XS_CHECK(p.min_pixels > 0 && p.min_pixels <= p.max_pixels, where, ": pixel bounds ", p.min_pixels, " to ",
             p.max_pixels);
    return p;
}

std::pair<int64_t, int64_t> smart_resize(int64_t height, int64_t width, int64_t factor, int64_t min_pixels,
                                         int64_t max_pixels) {
    XS_CHECK(height > 0 && width > 0, "qwen3_vision: an empty image");
    const double h = static_cast<double>(height), w = static_cast<double>(width), f = static_cast<double>(factor);
    XS_CHECK(std::max(h, w) / std::min(h, w) <= 200.0, "qwen3_vision: an image of ", width, " x ", height,
             " pixels: its aspect ratio is over 200");
    // Python's round: halves to even, as nearbyint in the default rounding mode.
    auto h_bar = static_cast<int64_t>(std::nearbyint(h / f)) * factor;
    auto w_bar = static_cast<int64_t>(std::nearbyint(w / f)) * factor;
    if (h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt(h * w / static_cast<double>(max_pixels));
        h_bar = std::max(factor, static_cast<int64_t>(std::floor(h / beta / f)) * factor);
        w_bar = std::max(factor, static_cast<int64_t>(std::floor(w / beta / f)) * factor);
    } else if (h_bar * w_bar < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / (h * w));
        h_bar = static_cast<int64_t>(std::ceil(h * beta / f)) * factor;
        w_bar = static_cast<int64_t>(std::ceil(w * beta / f)) * factor;
    }
    return {h_bar, w_bar};
}

}  // namespace xs::qwen3_vision
