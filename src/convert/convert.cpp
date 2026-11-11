#include "convert/convert.h"

#include <set>

#include "convert/calibrate.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>

#include "graph/graph.h"
#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "models/config_util.h"
#include "models/model.h"
#include "models/policy.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/log.h"

namespace xs {

namespace {

namespace fs = std::filesystem;
using graph::OpKind;

constexpr uint64_t kShardBytes = uint64_t{4} << 30;

// Q8 / Q4 when `format` fits the tensor (a float tensor, as stored or as an FP8 checkpoint's f8, of at least 2 dims
// and min_elements, quant.h's matrix view); explains a refusal in the log.
std::optional<DType> fitting(const std::string& name, const TensorView& v, DType format, const ConvertOptions& o) {
    if (!(is_float(v.dtype) || v.dtype == DType::F8) || v.shape.size() < 2 || v.numel() < o.min_elements) {
        return std::nullopt;
    }
    if (const std::string why = quant::unsupported_reason(format, quant::matrix_rows(v.shape), v.shape.back(), o.group);
        !why.empty()) {
        log::info("convert: keeping '", name, "' as ", dtype_name(v.dtype), ": ", why);
        return std::nullopt;
    }
    return format;
}

// Weights of the matmuls whose output decides a TopK (through Softmax / Sigmoid) that picks what the graph reads next
// (experts, a drafter's centroids): mixture-of-experts routers. They stay unquantized: a rounded router picks
// different experts for some tokens, a much larger error than rounding a weight. A TopK whose indices only leave the
// graph (a head's argmax) picks nothing.
std::set<std::string> router_weights(const graph::Graph& g) {
    const auto& nodes = g.nodes();
    const auto selects = [&](size_t topk) {  // its indices reach an ExpertMatMul or a Gather, through reshapes
        std::vector<graph::Value> open{graph::Value{static_cast<int32_t>(topk), 1}};
        while (!open.empty()) {
            const graph::Value v = open.back();
            open.pop_back();
            for (size_t c = 0; c < nodes.size(); ++c) {
                if (std::find(nodes[c].inputs.begin(), nodes[c].inputs.end(), v) == nodes[c].inputs.end()) continue;
                if (nodes[c].kind == OpKind::ExpertMatMul || nodes[c].kind == OpKind::Gather) return true;
                if (nodes[c].kind == OpKind::Reshape) open.push_back(graph::Value{static_cast<int32_t>(c), 0});
            }
        }
        return false;
    };
    std::set<std::string> out;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        if (n.kind != OpKind::TopK || !selects(i)) continue;
        graph::Value v = n.inputs[0];
        while (g.node(v).kind == OpKind::Softmax || g.node(v).kind == OpKind::Sigmoid) v = g.node(v).inputs[0];
        const auto& mm = g.node(v);
        if (mm.kind == OpKind::MatMul && g.node(mm.inputs[1]).kind == OpKind::Weight) {
            out.insert(std::get<graph::WeightAttrs>(g.node(mm.inputs[1]).attrs).name);
        }
    }
    return out;
}

// A matmul that reads its weight more than it computes: a step-dependent row count (decode) or at most kBandwidthRows
// rows per call. Expert banks count as bandwidth-bound (few tokens per expert).
bool bandwidth_bound(const graph::Graph& g, const graph::Node& reader) {
    if (reader.kind != OpKind::MatMul) return true;
    const Dims& x = g.type(reader.inputs[0]).shape;
    if (std::find(x.begin(), x.end(), graph::kT) != x.end()) return true;
    return numel(x) / x.back() <= kBandwidthRows;
}

// The format each weight of `g` gets: what the graph does with it decides. A weight read only as a matmul weight (or
// an expert bank) takes `matmul_format` when the scope admits its matmuls; one that is also (or only) a gather table
// takes q8; routers and anything else are copied.
void plan_graph(const graph::Graph& g, const WeightStore& weights, DType matmul_format, QuantScope scope,
                const ConvertOptions& o, std::map<std::string, DType>& plan) {
    const auto& nodes = g.nodes();
    // per node: (reader op, input index, the reader is bandwidth-bound)
    struct Reader {
        OpKind kind;
        size_t index;
        bool bandwidth;
    };
    std::vector<std::vector<Reader>> readers(nodes.size());
    for (const auto& n : nodes) {
        for (size_t k = 0; k < n.inputs.size(); ++k)
            readers[static_cast<size_t>(n.inputs[k].node)].push_back(Reader{n.kind, k, bandwidth_bound(g, n)});
    }
    for (const auto& out : g.outputs())
        readers[static_cast<size_t>(out.value.node)].push_back(Reader{OpKind::Input, 0, false});  // an output: keep
    const std::set<std::string> routers = router_weights(g);
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].kind != OpKind::Weight || readers[i].empty()) continue;
        const std::string& name = std::get<graph::WeightAttrs>(nodes[i].attrs).name;
        if (routers.count(name) != 0) {
            log::info("convert: keeping router '", name, "' unquantized");
            continue;
        }
        bool matmul = false;
        bool table = false;
        bool other = false;
        bool compute = false;  // read by a compute-bound matmul
        for (const Reader& r : readers[i]) {
            if ((r.kind == OpKind::MatMul || r.kind == OpKind::ExpertMatMul) && r.index == 1) {
                matmul = true;
                compute = compute || !r.bandwidth;
            } else if (r.kind == OpKind::Gather && r.index == 0) {
                table = true;
            } else {
                other = true;
            }
        }
        if (other || (!matmul && !table)) continue;
        if (!table && compute && scope == QuantScope::Bandwidth) continue;
        const bool q8_by_name = std::any_of(o.q8_names.begin(), o.q8_names.end(), [&](const std::string& part) {
            return !part.empty() && name.find(part) != std::string::npos;
        });
        // A weight planned from an earlier graph keeps that format (a drafter's shortlist gathers rows of the head
        // the target multiplies by).
        if (const auto f = fitting(name, weights.get(name), table || q8_by_name ? DType::Q8 : matmul_format, o)) {
            plan.try_emplace(name, *f);
        }
    }
}

// A language model: the text decoder's graph and its drafter's (speculative decoding) at the requested format, the
// vision tower's at q8 (it runs once per image and is compute-bound: only with scope All; its quality is not measured
// at q4 yet), and the tables the model reads on the host at q8.
std::map<std::string, DType> plan_formats(const Model& model, const ConvertOptions& o) {
    std::map<std::string, DType> plan;
    plan_graph(model.graph(), model.weights(), o.format, o.scope, o, plan);
    if (const Speculation* spec = model.speculation()) {
        plan_graph(spec->draft_graph(), model.weights(), o.format, o.scope, o, plan);
    }
    if (model.vision() != nullptr) plan_graph(model.vision()->graph(), model.weights(), DType::Q8, o.scope, o, plan);
    for (const std::string& name : model.host_tables()) {
        if (const auto f = fitting(name, model.weights().get(name), DType::Q8, o)) plan[name] = *f;
    }
    return plan;
}

// A drafter checkpoint beside `model`: the drafter's program and the target's verify program (what a block drafter
// reads of the target is computed there) at the requested format.
std::map<std::string, DType> plan_drafter_formats(const Model& model, const ConvertOptions& o) {
    std::map<std::string, DType> plan;
    plan_graph(model.speculation()->draft_graph(), model.weights(), o.format, o.scope, o, plan);
    plan_graph(model.speculation()->verify_graph(), model.weights(), o.format, o.scope, o, plan);
    return plan;
}

// A policy: the program for every camera, the checkpoint's flow steps and its longest prompt (the call shape every
// other one is a part of).
std::map<std::string, DType> plan_formats(const Policy& policy, const ConvertOptions& o) {
    CallShape shape;
    for (int c = 0; c < static_cast<int>(policy.spec().cameras.size()); ++c) shape.cameras.push_back(c);
    shape.steps = policy.spec().denoise_steps;
    shape.prompt_tokens = policy.prompt_tokens(std::numeric_limits<int32_t>::max());
    std::map<std::string, DType> plan;
    plan_graph(policy.build(shape), policy.weights(), o.format, o.scope, o, plan);
    return plan;
}

// A tensor the pack keeps: its bytes as they are; an FP8 checkpoint's weight as its f32 values (a pack holds no f8,
// and the weights that matter for size are quantized).
OutUnit copy_unit(const std::string& name, const TensorView& v) {
    OutUnit u;
    if (v.dtype == DType::F8) {
        u.tensors.push_back(OutTensor{name, DType::F32, v.shape});
        u.write = [v](std::ostream& out) {
            const int64_t rows = quant::matrix_rows(v.shape);
            const int64_t cols = v.shape.back();
            const int64_t piece = std::max<int64_t>(1, (int64_t{16} << 20) / cols);
            std::vector<float> values(static_cast<size_t>(std::min(rows, piece) * cols));
            for (int64_t first = 0; first < rows; first += piece) {
                const int64_t count = std::min(piece, rows - first);
                quant::read_rows(v, first, count, values.data());
                out.write(reinterpret_cast<const char*>(values.data()),
                          static_cast<std::streamsize>(static_cast<size_t>(count * cols) * sizeof(float)));
            }
        };
        return u;
    }
    u.tensors.push_back(OutTensor{name, v.dtype, v.shape});
    u.write = [v](std::ostream& out) {
        out.write(static_cast<const char*>(v.data), static_cast<std::streamsize>(v.nbytes()));
    };
    return u;
}

// `act_ms` (AWQ calibration statistics) set: the q4 weight is quantized with quantize_awq and gets input_scales.
OutUnit quant_unit(const std::string& name, const TensorView& v, DType format, int64_t group,
                   const std::vector<double>* act_ms, ConvertReport* report) {
    const int64_t cols = v.shape.back();
    const Dims groups = quant::component_shape(v.shape, cols / group);
    OutUnit u;
    u.tensors.push_back(OutTensor{name, format == DType::Q4 ? DType::U8 : DType::I8,
                                  format == DType::Q4 ? quant::component_shape(v.shape, cols / 2) : v.shape});
    u.tensors.push_back(OutTensor{name + ".scales", DType::F16, groups});
    if (format == DType::Q4) u.tensors.push_back(OutTensor{name + ".biases", DType::F16, groups});
    if (act_ms != nullptr) u.tensors.push_back(OutTensor{name + ".input_scales", DType::F32, {cols}});
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

std::map<std::string, DType> plan_graph_formats(const graph::Graph& g, const WeightStore& weights, DType matmul_format,
                                                const ConvertOptions& o) {
    std::map<std::string, DType> plan;
    plan_graph(g, weights, matmul_format, o.scope, o, plan);
    return plan;
}

ConvertReport convert_model(const ConvertOptions& o) {
    XS_CHECK(o.format == DType::Q4 || o.format == DType::Q8, "convert: the format must be q4 or q8, got ",
             dtype_name(o.format));
    if (!fs::exists(o.source / "config.json")) fail("convert: '", o.source.string(), "' has no config.json");
    if (fs::exists(o.source / kPackManifest)) fail("convert: '", o.source.string(), "' is already a pack");
    if (fs::exists(o.output) && !fs::is_empty(o.output)) {
        fail("convert: '", o.output.string(), "' exists and is not empty");
    }
    if (fs::exists(o.output) && fs::equivalent(o.source, o.output)) fail("convert: the output is the source");

    const bool drafter = !o.target.empty();
    const bool policy = !drafter && model_kind(o.source) == ModelKind::Policy;
    XS_CHECK(!policy || o.method == QuantMethod::Rtn, "convert: AWQ calibrates on text; policies take round to nearest");
    XS_CHECK(!drafter || o.method == QuantMethod::Rtn, "convert: AWQ calibrates a model; a drafter takes round to nearest");
    std::unique_ptr<Model> model;
    std::unique_ptr<Policy> policy_model;
    // What is written: the model's weights, or a drafter's own tensors, which the model holds under `prefix`.
    std::unique_ptr<WeightStore> drafter_weights;
    std::string prefix;
    if (policy) {
        policy_model = load_policy(o.source);
    } else if (drafter) {
        model = load_model(o.target, LoadOptions{.vision = false, .speculation = true, .drafter = o.source});
        const Speculation* spec = model->speculation();
        XS_CHECK(spec != nullptr && !spec->drafter_weights().empty(), "convert: '", o.target.string(),
                 "' does not draft with '", o.source.string(), "'");
        prefix = spec->drafter_weights();
        drafter_weights = SafetensorsWeights::open_dir(o.source);
    } else {
        model = load_model(o.source, LoadOptions{.vision = true, .speculation = true});
    }
    const WeightStore& weights = drafter ? *drafter_weights : policy ? policy_model->weights() : model->weights();
    const std::map<std::string, DType> plan =
        policy ? plan_formats(*policy_model, o) : drafter ? plan_drafter_formats(*model, o) : plan_formats(*model, o);

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
        const auto it = plan.find(prefix + name);
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
                          {"model_type", config.value("model_type", config.value("type", ""))}};
    if (drafter) manifest["source"]["drafter"] = model->speculation()->drafter_name();
    manifest["quant"] = {{"format", o.format == DType::Q4 ? "q4" : "q8"}, {"group", o.group},
                         {"method", report.awq > 0 ? "awq" : "rtn"}, {"min_elements", o.min_elements},
                         {"q8_names", o.q8_names}, {"scope", o.scope == QuantScope::All ? "all" : "bandwidth"}};
    manifest["tensors"] = tensors;
    std::ofstream(o.output / kPackManifest) << manifest.dump(1) << "\n";
    return report;
}

}  // namespace xs
