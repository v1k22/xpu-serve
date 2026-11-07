#include "convert/convert.h"

#include <set>

#include "convert/calibrate.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>

#include "graph/graph.h"
#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "models/config_util.h"
#include "models/model.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/log.h"

namespace xs {

namespace {

namespace fs = std::filesystem;
using graph::OpKind;

constexpr uint64_t kShardBytes = uint64_t{4} << 30;

// Q8 / Q4 when `format` fits the tensor (a float matrix of at least min_elements); explains a refusal in the log.
std::optional<DType> fitting(const std::string& name, const TensorView& v, DType format, const ConvertOptions& o) {
    if (!is_float(v.dtype) || v.shape.size() != 2 || v.numel() < o.min_elements) return std::nullopt;
    if (const std::string why = quant::unsupported_reason(format, v.shape[0], v.shape[1], o.group); !why.empty()) {
        log::info("convert: keeping '", name, "' as ", dtype_name(v.dtype), ": ", why);
        return std::nullopt;
    }
    return format;
}

// The format each weight of `g` gets: what the graph does with it decides. A weight read only as a matmul weight takes
// `matmul_format`; one that is also (or only) a gather table takes q8; anything else is copied.
void plan_graph(const graph::Graph& g, const WeightStore& weights, DType matmul_format, const ConvertOptions& o,
                std::map<std::string, DType>& plan) {
    const auto& nodes = g.nodes();
    std::vector<std::vector<std::pair<OpKind, size_t>>> readers(nodes.size());  // per node: (reader op, input index)
    for (const auto& n : nodes) {
        for (size_t k = 0; k < n.inputs.size(); ++k)
            readers[static_cast<size_t>(n.inputs[k].node)].emplace_back(n.kind, k);
    }
    for (const auto& out : g.outputs())
        readers[static_cast<size_t>(out.value.node)].emplace_back(OpKind::Input, 0);  // a graph output: keep
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].kind != OpKind::Weight || readers[i].empty()) continue;
        const std::string& name = std::get<graph::WeightAttrs>(nodes[i].attrs).name;
        bool matmul = false;
        bool table = false;
        bool other = false;
        for (const auto& [kind, k] : readers[i]) {
            if (kind == OpKind::MatMul && k == 1) {
                matmul = true;
            } else if (kind == OpKind::Gather && k == 0) {
                table = true;
            } else {
                other = true;
            }
        }
        if (other || (!matmul && !table)) continue;
        const bool q8_by_name = std::any_of(o.q8_names.begin(), o.q8_names.end(), [&](const std::string& part) {
            return !part.empty() && name.find(part) != std::string::npos;
        });
        if (const auto f = fitting(name, weights.get(name), table || q8_by_name ? DType::Q8 : matmul_format, o)) {
            plan[name] = *f;
        }
    }
}

// The text decoder's graph at the requested format, the vision tower's at q8 (it runs once per image; its quality
// is not measured at q4 yet), and the tables the model reads on the host at q8.
std::map<std::string, DType> plan_formats(const Model& model, const ConvertOptions& o) {
    std::map<std::string, DType> plan;
    plan_graph(model.graph(), model.weights(), o.format, o, plan);
    if (model.vision() != nullptr) plan_graph(model.vision()->graph(), model.weights(), DType::Q8, o, plan);
    for (const std::string& name : model.host_tables()) {
        if (const auto f = fitting(name, model.weights().get(name), DType::Q8, o)) plan[name] = *f;
    }
    return plan;
}

OutUnit copy_unit(const std::string& name, const TensorView& v) {
    OutUnit u;
    u.tensors.push_back(OutTensor{name, v.dtype, v.shape});
    u.write = [v](std::ostream& out) {
        out.write(static_cast<const char*>(v.data), static_cast<std::streamsize>(v.nbytes()));
    };
    return u;
}

// `act_ms` (AWQ calibration statistics) set: the q4 weight is quantized with quantize_awq and gets input_scales.
OutUnit quant_unit(const std::string& name, const TensorView& v, DType format, int64_t group,
                   const std::vector<double>* act_ms, ConvertReport* report) {
    const Dims groups{v.shape[0], v.shape[1] / group};
    OutUnit u;
    u.tensors.push_back(OutTensor{name, format == DType::Q4 ? DType::U8 : DType::I8,
                                  format == DType::Q4 ? Dims{v.shape[0], v.shape[1] / 2} : v.shape});
    u.tensors.push_back(OutTensor{name + ".scales", DType::F16, groups});
    if (format == DType::Q4) u.tensors.push_back(OutTensor{name + ".biases", DType::F16, groups});
    if (act_ms != nullptr) u.tensors.push_back(OutTensor{name + ".input_scales", DType::F32, {v.shape[1]}});
    u.write = [v, format, group, act_ms, report, name](std::ostream& out) {
        quant::Quantized q;
        if (act_ms != nullptr) {
            quant::AwqReport r;
            q = quant::quantize_awq(v, group, *act_ms, &r);
            const double gain = r.error_rtn > 0 ? 1.0 - r.error / r.error_rtn : 0.0;
            report->awq_gain += gain;
            log::debug("convert: awq '", name, "' alpha ", r.alpha, ", error ", r.error, " vs ", r.error_rtn, " (rtn)");
        } else {
            q = quant::quantize(v, format, group);
        }
        out.write(reinterpret_cast<const char*>(q.codes.data()), static_cast<std::streamsize>(q.codes.size()));
        out.write(reinterpret_cast<const char*>(q.scales.data()), static_cast<std::streamsize>(q.scales.size() * 2));
        if (format == DType::Q4) {
            out.write(reinterpret_cast<const char*>(q.biases.data()),
                      static_cast<std::streamsize>(q.biases.size() * 2));
        }
        if (act_ms != nullptr) {
            out.write(reinterpret_cast<const char*>(q.input_scales.data()),
                      static_cast<std::streamsize>(q.input_scales.size() * sizeof(float)));
        }
    };
    return u;
}

void copy_side_files(const fs::path& from, const fs::path& to) {
    for (const auto& e : fs::directory_iterator(from)) {
        const std::string name = e.path().filename().string();
        if (!e.is_regular_file() || name.empty() || name[0] == '.') continue;
        if (e.path().extension() == ".safetensors" || name == "model.safetensors.index.json" || name == kPackManifest) {
            continue;
        }
        fs::copy_file(e.path(), to / name, fs::copy_options::overwrite_existing);
    }
}

}  // namespace

ConvertReport convert_model(const ConvertOptions& o) {
    XS_CHECK(o.format == DType::Q4 || o.format == DType::Q8, "convert: the format must be q4 or q8, got ",
             dtype_name(o.format));
    if (!fs::exists(o.source / "config.json")) fail("convert: '", o.source.string(), "' has no config.json");
    if (fs::exists(o.source / kPackManifest)) fail("convert: '", o.source.string(), "' is already a pack");
    if (fs::exists(o.output) && !fs::is_empty(o.output)) {
        fail("convert: '", o.output.string(), "' exists and is not empty");
    }
    if (fs::exists(o.output) && fs::equivalent(o.source, o.output)) fail("convert: the output is the source");

    const auto model = load_model(o.source);
    const WeightStore& weights = model->weights();
    const std::map<std::string, DType> plan = plan_formats(*model, o);

    ConvertReport report;
    // AWQ: activation statistics for the text decoder's q4 weights.
    std::map<std::string, std::vector<double>> act_ms;
    if (o.method == QuantMethod::Awq && o.format == DType::Q4) {
        std::set<std::string> q4;
        for (const auto& [name, format] : plan) {
            if (format == DType::Q4) q4.insert(name);
        }
        std::vector<int32_t> tokens = o.calibration_ids;
        if (tokens.empty()) {
            const auto tokenizer = tok::Tokenizer::load(o.source);
            tokens = tokenizer->encode(o.calibration_text.empty() ? builtin_calibration_text() : o.calibration_text,
                                       false);
        }
        CalibrationOptions co;
        co.backend = o.backend;
        co.device = o.device;
        co.max_tokens = o.calibration_tokens;
        act_ms = input_channel_mean_squares(*model, tokens, q4, co);
    }
    std::vector<OutUnit> units;
    std::vector<std::string> names = weights.names();
    std::sort(names.begin(), names.end());
    nlohmann::ordered_json tensors = nlohmann::ordered_json::object();
    for (const std::string& name : names) {
        const TensorView& v = weights.get(name);
        report.source_bytes += v.nbytes();
        const auto it = plan.find(name);
        if (it == plan.end()) {
            units.push_back(copy_unit(name, v));
            ++report.kept;
            continue;
        }
        const auto stats = act_ms.find(name);
        const std::vector<double>* ms = stats == act_ms.end() ? nullptr : &stats->second;
        units.push_back(quant_unit(name, v, it->second, o.group, ms, &report));
        tensors[name] = {{"format", it->second == DType::Q4 ? "q4" : "q8"}, {"group", o.group}, {"shape", v.shape}};
        if (ms != nullptr) {
            tensors[name]["input_scales"] = true;
            ++report.awq;
        }
        ++(it->second == DType::Q4 ? report.q4 : report.q8);
    }

    // Shards of at most kShardBytes (a larger unit gets one of its own), in name order.
    std::vector<std::vector<size_t>> shards(1);
    uint64_t in_shard = 0;
    for (size_t i = 0; i < units.size(); ++i) {
        const uint64_t bytes = unit_bytes(units[i]);
        if (in_shard > 0 && in_shard + bytes > kShardBytes) {
            shards.emplace_back();
            in_shard = 0;
        }
        shards.back().push_back(i);
        in_shard += bytes;
        report.output_bytes += bytes;
    }
    report.shards = static_cast<int>(shards.size());

    fs::create_directories(o.output);
    nlohmann::ordered_json weight_map = nlohmann::ordered_json::object();
    for (size_t s = 0; s < shards.size(); ++s) {
        char file[64];
        std::snprintf(file, sizeof(file), "model-%05zu-of-%05zu.safetensors", s + 1, shards.size());
        std::vector<OutUnit> shard_units;
        for (const size_t i : shards[s]) {
            shard_units.push_back(units[i]);
            for (const OutTensor& t : units[i].tensors)
                weight_map[t.name] = file;
        }
        log::info("convert: writing ", file, " (", shard_units.size(), " tensors)");
        write_safetensors(o.output / file, shard_units, {{"format", "pt"}});
    }
    {
        nlohmann::ordered_json index;
        index["metadata"] = {{"total_size", report.output_bytes}};
        index["weight_map"] = weight_map;
        std::ofstream(o.output / "model.safetensors.index.json") << index.dump(2) << "\n";
    }
    copy_side_files(o.source, o.output);
    if (report.awq > 0) report.awq_gain /= static_cast<double>(report.awq);

    const auto config = cfg::read_json(o.source / "config.json");
    nlohmann::ordered_json manifest;
    manifest["pack_version"] = kPackVersion;
    manifest["converter"] = o.converter;
    fs::path source_dir = fs::absolute(o.source).lexically_normal();
    if (source_dir.filename().empty()) source_dir = source_dir.parent_path();
    manifest["source"] = {{"name", source_dir.filename().string()},
                          {"model_type", config.value("model_type", "")}};
    manifest["quant"] = {{"format", o.format == DType::Q4 ? "q4" : "q8"}, {"group", o.group},
                         {"method", report.awq > 0 ? "awq" : "rtn"}, {"min_elements", o.min_elements},
                         {"q8_names", o.q8_names}};
    manifest["tensors"] = tensors;
    std::ofstream(o.output / kPackManifest) << manifest.dump(1) << "\n";
    return report;
}

}  // namespace xs
