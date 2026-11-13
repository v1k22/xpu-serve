#include "convert/gguf/layout.h"

#include "util/error.h"

namespace xs::gguf_import {

Entry one(Dims stored, std::string name, const FloatMap* map) {
    Dims shape = stored;
    return Entry{std::move(stored), {Piece{std::move(name), std::move(shape), map}}};
}

int64_t positive(const gguf::File& file, std::string_view architecture, const std::string& key) {
    const int64_t v = file.get_int(key);
    if (v <= 0) fail(architecture, ": ", key, " is ", v);
    return v;
}

void refuse_unknown_keys(const gguf::File& file, std::string_view architecture,
                         const std::set<std::string_view>& known) {
    const std::string prefix = str_cat(architecture, ".");
    for (const gguf::Value& v : file.metadata()) {
        const std::string_view key = v.key();
        if (key.substr(0, prefix.size()) == prefix && !known.contains(key.substr(prefix.size()))) {
            fail(architecture, ": metadata '", key, "' is not read by this importer; a key it does not know may ",
                 "change the model, so the file is refused");
        }
    }
}

void map_tensors(const gguf::File& file, std::string_view architecture, const Layout& layout, ImportPlan& plan) {
    std::vector<const gguf::TensorInfo*> tensors;
    for (const gguf::TensorInfo& t : file.tensors())
        tensors.push_back(&t);
    map_tensors(tensors, architecture, layout, plan);
}

void map_tensors(const std::vector<const gguf::TensorInfo*>& tensors, std::string_view architecture,
                 const Layout& layout, ImportPlan& plan) {
    const auto blocks = static_cast<int64_t>(layout.blocks.size());
    std::map<std::string, int, std::less<>> made;  // pack tensor name -> how many tensors of the file made it
    for (const gguf::TensorInfo* tensor : tensors) {
        const gguf::TensorInfo& t = *tensor;
        const Entry* e = nullptr;
        if (t.name.rfind("blk.", 0) == 0) {
            const size_t dot = t.name.find('.', 4);
            const std::string index = t.name.substr(4, dot == std::string::npos ? std::string::npos : dot - 4);
            if (dot == std::string::npos || index.empty() || index.size() > 9 ||
                index.find_first_not_of("0123456789") != std::string::npos || std::stoll(index) >= blocks) {
                fail(architecture, ": tensor '", t.name, "' is not in one of the file's ", blocks, " blocks");
            }
            const int64_t i = std::stoll(index);
            const Entries& entries = layout.blocks[static_cast<size_t>(i)];
            const auto it = entries.find(std::string_view(t.name).substr(dot + 1));
            if (it == entries.end()) {
                fail(architecture, ": tensor '", t.name, "' has no counterpart in block ", i, " (",
                     layout.block_kind(i), ")");
            }
            e = &it->second;
        } else {
            const auto it = layout.top.find(t.name);
            if (it == layout.top.end()) fail(architecture, ": tensor '", t.name, "' has no counterpart in the model");
            e = &it->second;
        }
        if (t.shape() != e->stored) {
            fail(architecture, ": tensor '", t.name, "' has shape ", dims_to_string(t.shape()), "; the metadata gives ",
                 dims_to_string(e->stored));
        }
        if (e->pieces.empty()) plan.derived.push_back(&t);
        int64_t row = 0;
        for (const Piece& p : e->pieces) {
            plan.tensors.push_back(PackTensor{p.name, p.shape, &t, row, p.map});
            row += numel(p.shape) / p.shape.back();
            ++made[p.name];
        }
    }

    // Every pack tensor the model reads, made once (not both from one tensor and from its parts).
    std::map<std::string, bool, std::less<>> required;  // pack tensor name -> some entry naming it is not optional
    const auto collect = [&](const Entries& entries) {
        for (const auto& [suffix, e] : entries) {
            for (const Piece& p : e.pieces)
                required[p.name] = required[p.name] || !e.optional;
        }
    };
    const auto check = [&](const Entries& entries, const std::string& where) {
        for (const auto& [suffix, e] : entries) {
            for (const Piece& p : e.pieces) {
                const auto it = made.find(p.name);
                const int n = it == made.end() ? 0 : it->second;
                if (n == 0 && required.at(p.name)) {
                    fail(architecture, ": the file has no tensor for '", p.name, "' (", where, ")");
                }
                if (n > 1)
                    fail(architecture, ": '", p.name, "' comes from more than one tensor of the file (", where, ")");
            }
        }
    };
    collect(layout.top);
    for (const Entries& b : layout.blocks)
        collect(b);
    check(layout.top, "the model's");
    for (int64_t i = 0; i < blocks; ++i)
        check(layout.blocks[static_cast<size_t>(i)], str_cat("block ", i, ", ", layout.block_kind(i)));
}

}  // namespace xs::gguf_import
