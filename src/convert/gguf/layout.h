#pragma once

// The tensors of a GGUF file as an architecture's importer lays them out: for each name llama.cpp's converter writes
// (a whole name, or the name after "blk.<i>." in block i), the shape the metadata implies and the pack tensors it
// holds, in row order. map_tensors() checks a file against its layout and gives the import plan's tensors.

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "convert/gguf/architecture.h"

namespace xs::gguf_import {

// A pack tensor that a tensor of the file holds.
struct Piece {
    std::string name;
    Dims shape;
    const FloatMap* map = nullptr;
};

// What one tensor of the file becomes: its stored shape (row-major) and the pack tensors it holds, in row order. An
// entry without pieces is a tensor the converter computes from the config, which the architecture checks against the
// config it writes; it is not imported.
struct Entry {
    Dims stored;
    std::vector<Piece> pieces;
    // The file may lack it; its pieces are then required only by the entries that also name them.
    bool optional = false;
};

// A tensor of the file that is one pack tensor of the same shape.
Entry one(Dims stored, std::string name, const FloatMap* map = nullptr);

using Entries = std::map<std::string, Entry, std::less<>>;

struct Layout {
    Entries top;                                     // by whole name
    std::vector<Entries> blocks;                     // block i: by the name after "blk.<i>."
    std::function<std::string(int64_t)> block_kind;  // what block i is, for errors ("linear attention")
};

// Every tensor of `file` through `layout`, into plan.tensors (and the computed ones into plan.derived). Refuses by
// name, as "<architecture>: ...": a tensor the layout has no entry for, a shape other than its entry's, a pack tensor
// no tensor of the file holds (unless every entry naming it is optional) or that more than one does.
void map_tensors(const gguf::File& file, std::string_view architecture, const Layout& layout, ImportPlan& plan);

// Metadata `key`, an integer that must be positive.
int64_t positive(const gguf::File& file, std::string_view architecture, const std::string& key);

// Refuses a key "<architecture>.<rest>" whose rest `known` does not list: a key the importer does not know may change
// the model.
void refuse_unknown_keys(const gguf::File& file, std::string_view architecture,
                         const std::set<std::string_view>& known);

}  // namespace xs::gguf_import
