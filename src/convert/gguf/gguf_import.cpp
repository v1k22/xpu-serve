#include "convert/gguf/gguf_import.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string_view>

#include <nlohmann/json.hpp>

#include "convert/gguf/architecture.h"
#include "convert/gguf/tokenizer.h"
#include "io/gguf.h"
#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"
#include "util/half.h"
#include "util/log.h"

namespace xs::gguf_import {
namespace {

namespace fs = std::filesystem;

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.substr(0, prefix.size()) == prefix;
}

// What no architecture's importer reads but would change what the file means.
// `type`: the file's general.type, when it says one: a model's, or a vision tower's ("mmproj").
void check_general(const gguf::File& f, std::string_view type = "model") {
    for (const gguf::Value& v : f.metadata()) {
        if (starts_with(v.key(), "prism.")) {
            fail("GGUF metadata '", v.key(), "': weights with transforms folded into them are not supported");
        }
    }
    if (const gguf::Value* t = f.find("general.type"); t != nullptr && t->as_string() != type) {
        fail("GGUF '", f.path().string(), "' is general.type '", t->as_string(), "', not a ", type);
    }
    // The block layouts read here are quantization version 2's (version 1 laid out Q4_0, Q4_1 and Q8_0 otherwise).
    if (const gguf::Value* q = f.find("general.quantization_version"); q != nullptr && q->as_int() != 2) {
        fail("GGUF general.quantization_version ", q->as_int(), " is not supported (supported: 2)");
    }
}

// Every row of every tensor of the files goes into the pack exactly once, but for the derived ones, the copies and
// the ones left out, which go nowhere.
void check_coverage(const std::vector<const gguf::File*>& files, const ImportPlan& plan) {
    std::map<const gguf::TensorInfo*, std::vector<std::pair<int64_t, int64_t>>> runs;
    for (const auto* list : {&plan.derived, &plan.copies, &plan.left_out}) {
        for (const gguf::TensorInfo* t : *list)
            runs[t].emplace_back(0, t->rows());
    }
    for (const PackTensor& p : plan.tensors) {
        if (p.reorder != nullptr) {  // the whole tensor, in another order
            XS_CHECK(p.first_row == 0 && numel(p.shape) == p.source->numel(), "gguf import: '", p.name, "' ",
                     dims_to_string(p.shape), " is not '", p.source->name, "' whole, reordered");
            runs[p.source].emplace_back(0, p.source->rows());
            continue;
        }
        XS_CHECK(!p.shape.empty() && p.shape.back() == p.source->cols(), "gguf import: '", p.name, "' ",
                 dims_to_string(p.shape), " does not keep the ", p.source->cols(), " columns of '", p.source->name,
                 "'");
        const int64_t rows = numel(p.shape) / p.shape.back();
        XS_CHECK(p.first_row >= 0 && p.first_row + rows <= p.source->rows(), "gguf import: '", p.name, "' takes rows ",
                 p.first_row, " to ", p.first_row + rows, " of '", p.source->name, "', which has ", p.source->rows());
        runs[p.source].emplace_back(p.first_row, rows);
    }
    for (const gguf::File* f : files) {
        for (const gguf::TensorInfo& t : f->tensors()) {
            auto it = runs.find(&t);
            if (it == runs.end()) {
                fail("gguf import: tensor '", t.name, "' of ", f->path().filename().string(), " is not imported");
            }
            std::sort(it->second.begin(), it->second.end());
            int64_t next = 0;
            for (const auto& [first, count] : it->second) {
                XS_CHECK(first == next, "gguf import: rows of tensor '", t.name, "' are imported twice or not at all");
                next = first + count;
            }
            XS_CHECK(next == t.rows(), "gguf import: rows ", next, " to ", t.rows(), " of tensor '", t.name,
                     "' are not imported");
        }
    }
}

OutUnit write_bytes(OutTensor tensor, const std::byte* begin, size_t bytes) {
    OutUnit u;
    u.tensors = {std::move(tensor)};
    u.write = [begin, bytes](std::ostream& out) {
        out.write(reinterpret_cast<const char*>(begin), static_cast<std::streamsize>(bytes));
    };
    return u;
}

// The pack tensor `p`: a block format weight byte for byte as U8 [rows, row bytes]; a float tensor as stored, or as
// F32 through its transform (computed here, so that a value the transform cannot take is refused before anything is
// written).
OutUnit unit(const PackTensor& p) {
    const gguf::TensorInfo& s = *p.source;
    if (p.reorder != nullptr) {
        if (s.dtype == DType::Block) {
            fail("gguf import: '", s.name, "' is stored as ", s.type_name(), ", but ", p.reorder->what,
                 " needs a float tensor");
        }
        auto values = std::make_shared<std::vector<float>>(p.reorder->apply(float_rows(s, 0, s.rows()), s.shape()));
        XS_CHECK(static_cast<int64_t>(values->size()) == numel(p.shape), "gguf import: '", s.name, "' through ",
                 p.reorder->what, " has ", values->size(), " values for ", dims_to_string(p.shape));
        OutUnit u;
        u.tensors = {OutTensor{p.name, DType::F32, p.shape}};
        u.write = [values](std::ostream& out) {
            out.write(reinterpret_cast<const char*>(values->data()),
                      static_cast<std::streamsize>(values->size() * sizeof(float)));
        };
        return u;
    }
    const int64_t rows = numel(p.shape) / p.shape.back();
    const size_t row_bytes = s.nbytes / static_cast<size_t>(s.rows());
    const std::byte* begin = s.data + static_cast<size_t>(p.first_row) * row_bytes;
    const size_t bytes = static_cast<size_t>(rows) * row_bytes;
    if (s.dtype == DType::Block) {
        if (p.map != nullptr) {
            fail("gguf import: '", s.name, "' is stored as ", s.type_name(), ", but ", p.map->what,
                 " needs a float tensor");
        }
        return write_bytes(OutTensor{p.name, DType::U8, {rows, static_cast<int64_t>(row_bytes)}}, begin, bytes);
    }
    if (p.map == nullptr) return write_bytes(OutTensor{p.name, s.dtype, p.shape}, begin, bytes);
    auto values = std::make_shared<std::vector<float>>(float_rows(s, p.first_row, rows));
    for (size_t i = 0; i < values->size(); ++i) {
        const float x = (*values)[i];
        (*values)[i] = p.map->apply(x);
        if (std::isfinite(x) && !std::isfinite((*values)[i])) {
            fail("gguf import: '", s.name, "' holds ", x, " at ", i, ", for which ", p.map->what, " is not finite");
        }
    }
    OutUnit u;
    u.tensors = {OutTensor{p.name, DType::F32, p.shape}};
    u.write = [values](std::ostream& out) {
        out.write(reinterpret_cast<const char*>(values->data()),
                  static_cast<std::streamsize>(values->size() * sizeof(float)));
    };
    return u;
}

template <typename Json>
void write_json(const fs::path& path, const Json& j, int indent) {
    std::string text;
    try {
        text = j.dump(indent);
    } catch (const nlohmann::json::exception& e) {
        fail("gguf import: cannot write ", path.filename().string(), ": ", e.what());
    }
    std::ofstream out(path, std::ios::binary);
    out << text << "\n";
    if (!out) fail("gguf import: writing '", path.string(), "' failed");
}

// generation_config.json from general.sampling.*: the sampling defaults the engine reads, or nullopt when the file
// has none (the model's defaults then apply, as for a checkpoint without the file).
std::optional<nlohmann::ordered_json> generation_config(const gguf::File& f) {
    nlohmann::ordered_json g;
    for (const auto& [key, name] :
         {std::pair{"temp", "temperature"}, std::pair{"top_k", "top_k"}, std::pair{"top_p", "top_p"},
          std::pair{"min_p", "min_p"}, std::pair{"penalty_repeat", "repetition_penalty"}}) {
        const gguf::Value* v = f.find(str_cat("general.sampling.", key));
        if (v == nullptr) continue;
        if (std::string_view(key) == "top_k") {
            g[name] = v->as_int();
        } else {
            g[name] = config_number(*v);
        }
    }
    for (const gguf::Value& v : f.metadata()) {
        const std::string_view key = v.key();
        if (starts_with(key, "general.sampling.") && !g.contains(key.substr(17))) {
            const std::string_view rest = key.substr(17);
            if (rest != "temp" && rest != "top_k" && rest != "top_p" && rest != "min_p" && rest != "penalty_repeat") {
                log::warn("gguf: ", key, " is not imported; the engine has no such sampling setting");
            }
        }
    }
    if (g.empty()) return std::nullopt;
    nlohmann::ordered_json out;
    out["do_sample"] = !g.contains("temperature") || g["temperature"].get<double>() > 0.0;
    out.update(g);
    return out;
}

// A source file's name in the manifest: a split model's is the name of the file its parts were split from, so that
// the parts and the whole make the same pack.
std::string source_name(const gguf::File& f) {
    return f.parts().size() > 1 ? gguf::model_name(f.path()) + ".gguf" : f.path().filename().string();
}

}  // namespace

double config_number(const gguf::Value& v) {
    if (v.type() == gguf::ValueType::F32) {
        char text[32];
        const auto end = std::to_chars(text, text + sizeof(text), static_cast<float>(v.as_float())).ptr;
        double d = 0.0;
        std::from_chars(text, end, d);
        return d;
    }
    if (v.type() == gguf::ValueType::F64) return v.as_float();
    return static_cast<double>(v.as_int());
}

std::vector<float> float_rows(const gguf::TensorInfo& t, int64_t first, int64_t rows) {
    const auto n = static_cast<size_t>(rows * t.cols());
    const size_t start = static_cast<size_t>(first * t.cols());
    std::vector<float> out(n);
    switch (t.dtype) {
        case DType::F32:
            std::memcpy(out.data(), reinterpret_cast<const float*>(t.data) + start, n * sizeof(float));
            break;
        case DType::F16:
            convert_f16_to_f32(reinterpret_cast<const uint16_t*>(t.data) + start, out.data(), n);
            break;
        case DType::BF16:
            convert_bf16_to_f32(reinterpret_cast<const uint16_t*>(t.data) + start, out.data(), n);
            break;
        default:
            fail("gguf import: '", t.name, "' is not a float tensor");
    }
    return out;
}

ImportPlan Architecture::plan_with_mtp(const gguf::File&, const gguf::File&) const {
    fail(name(), ": an MTP head in a file of its own (--mtp) is not supported for this architecture");
}

namespace {

const std::vector<const Architecture*>& all_architectures() {
    static const std::vector<const Architecture*> all{&qwen35(), &qwen35moe(),   &gemma4(),   &gemma4_assistant(),
                                                      &clef(),   &modern_bert(), &inkling2(), &inkling2_loop()};
    return all;
}

}  // namespace

const Architecture* architecture_named(std::string_view name) {
    for (const Architecture* a : all_architectures()) {
        if (a->name() == name) return a;
    }
    return nullptr;
}

const Architecture& find_architecture(std::string_view name) {
    if (const Architecture* a = architecture_named(name)) return *a;
    std::string list;
    for (const Architecture* a : all_architectures()) list += str_cat(list.empty() ? "" : ", ", a->name());
    fail("GGUF architecture '", name, "' is not supported (supported: ", list, ")");
}

void Architecture::add_projector(ImportPlan&, const gguf::File&, const gguf::File&) const {
    fail(name(), ": a vision tower from a file of its own (--mmproj) is not supported for this architecture");
}

std::optional<std::string> Architecture::adapter_target(std::string_view) const { return std::nullopt; }

ImportReport import_gguf(const ImportOptions& o) {
    if (fs::exists(o.output) && !fs::is_empty(o.output)) {
        fail("convert: '", o.output.string(), "' exists and is not empty");
    }
    const gguf::File file(o.source);
    check_general(file);
    std::vector<const gguf::File*> files{&file};  // the files the plan's tensors come from
    std::optional<gguf::File> mtp;
    if (!o.mtp.empty()) {
        mtp.emplace(o.mtp);
        check_general(*mtp);
        files.push_back(&*mtp);
    }
    const std::string architecture(file.get_string("general.architecture"));
    const Architecture& arch = find_architecture(architecture);
    ImportPlan plan = mtp ? arch.plan_with_mtp(file, *mtp) : arch.plan(file);
    std::optional<gguf::File> mmproj;
    if (!o.mmproj.empty()) {
        mmproj.emplace(o.mmproj);
        check_general(*mmproj, "mmproj");
        files.push_back(&*mmproj);
        arch.add_projector(plan, file, *mmproj);
    }
    // A language model's tokenizer, chat template and sampling defaults (a drafter has the target's); a decision
    // model's tokenizer.
    const bool language = plan.kind == ImportPlan::Kind::Language;
    std::optional<TokenizerFiles> tokenizer;
    std::string chat;
    std::optional<nlohmann::ordered_json> generation;
    if (plan.kind != ImportPlan::Kind::Drafter) tokenizer = tokenizer_files(file);
    if (language) {
        chat = chat_template(file, plan.unused_templates);
        generation = generation_config(file);
        const std::vector<int32_t>& stops = tokenizer->stop_ids;
        plan.config["eos_token_id"] =
            stops.size() == 1 ? nlohmann::ordered_json(stops[0]) : nlohmann::ordered_json(stops);
    }
    check_coverage(files, plan);
    for (const gguf::TensorInfo* t : plan.copies) {
        log::info("gguf: ", t->name, " of ", o.mtp.filename().string(), " is left out: the head reads the model's");
    }
    if (!plan.left_out.empty()) {
        log::info("gguf: ", plan.left_out.size(), " tensors left out (", plan.left_out_why, "), from ",
                  plan.left_out.front()->name, " to ", plan.left_out.back()->name);
    }

    std::sort(plan.tensors.begin(), plan.tensors.end(),
              [](const PackTensor& a, const PackTensor& b) { return a.name < b.name; });
    std::vector<OutUnit> units;
    nlohmann::ordered_json manifest_tensors = nlohmann::ordered_json::object();
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
        const PackTensor& p = plan.tensors[i];
        if (i > 0 && plan.tensors[i - 1].name == p.name) fail("gguf import: two tensors become '", p.name, "'");
        units.push_back(unit(p));
        if (p.source->dtype == DType::Block) {
            manifest_tensors[p.name] = {{"format", block_format_name(p.source->block)}, {"shape", p.shape}};
        }
    }

    fs::create_directories(o.output);
    log::info("gguf: writing ", units.size(), " tensors to ", (o.output / "model.safetensors").string());
    write_safetensors(o.output / "model.safetensors", units, {{"format", "pt"}});
    write_json(o.output / "config.json", plan.config, 2);
    for (const auto& [path, json] : plan.files) {
        fs::create_directories((o.output / path).parent_path());
        write_json(o.output / path, json, 2);
    }
    if (tokenizer) {
        const fs::path dir = o.output / plan.tokenizer_dir;
        fs::create_directories(dir);
        write_json(dir / "tokenizer.json", tokenizer->tokenizer, -1);
        write_json(dir / "tokenizer_config.json", tokenizer->config, 2);
        if (generation) write_json(o.output / "generation_config.json", *generation, 2);
        if (language) {
            std::ofstream out(o.output / "chat_template.jinja", std::ios::binary);
            out << chat;
            if (!out) fail("gguf import: writing chat_template.jinja failed");
        }
        // The tokenizer as the engine will load it: one whose merges do not fit its vocabulary is refused now rather
        // than when the pack is served.
        tok::Tokenizer::load(dir);
    }

    nlohmann::ordered_json manifest;
    manifest["pack_version"] = kPackVersion;
    manifest["converter"] = o.converter;
    nlohmann::ordered_json source;
    source["name"] = source_name(file);
    source["format"] = "gguf";
    source["model_type"] = plan.config.value("model_type", "");
    source["general.architecture"] = architecture;
    for (const char* key : {"general.name", "general.quantized_by"}) {
        if (const gguf::Value* v = file.find(key)) source[key] = std::string(v->as_string());
    }
    if (const gguf::Value* v = file.find("general.file_type")) source["general.file_type"] = v->as_int();
    if (mtp) source["mtp"] = source_name(*mtp);
    if (mmproj) source["mmproj"] = source_name(*mmproj);
    manifest["source"] = std::move(source);
    manifest["tensors"] = std::move(manifest_tensors);
    write_json(o.output / kPackManifest, manifest, 1);

    ImportReport report;
    report.architecture = architecture;
    if (const gguf::Value* v = file.find("general.name")) report.model_name = std::string(v->as_string());
    std::map<std::string, FormatShare> formats;
    for (const gguf::File* f : files) {
        for (const gguf::TensorInfo& t : f->tensors()) {
            if (std::find(plan.copies.begin(), plan.copies.end(), &t) != plan.copies.end()) continue;
            if (std::find(plan.left_out.begin(), plan.left_out.end(), &t) != plan.left_out.end()) continue;
            FormatShare& s = formats[t.type_name()];
            s.format = t.type_name();
            ++s.tensors;
            s.weights += t.numel();
            s.bytes += t.nbytes;
            ++report.tensors;
            report.weights += t.numel();
        }
    }
    for (auto& [name, share] : formats)
        report.formats.push_back(share);
    std::sort(report.formats.begin(), report.formats.end(),
              [](const FormatShare& a, const FormatShare& b) { return a.weights > b.weights; });
    for (const auto& e : fs::directory_iterator(o.output)) {
        if (e.is_regular_file()) report.output_bytes += e.file_size();
    }
    return report;
}

}  // namespace xs::gguf_import
