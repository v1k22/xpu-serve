#pragma once

// A block drafter's runner (BlockDrafting: DFlash, DSpine): one run of the draft program proposes the whole block.
// The drafter reads the target through state the target's own programs write, so nothing waits here.

#include <cstdint>
#include <span>
#include <vector>

#include "engine/spec/drafter.h"

namespace xs::spec {

class BlockDrafter final : public Drafter {
public:
    BlockDrafter(const Speculation& spec, Session& session);

    std::span<const std::string> target_outputs() const override { return {}; }
    int max_drafts() const override;
    void add(const TensorMap&, int64_t, int64_t, std::span<const int32_t>) override {}
    void flush() override {}
    std::vector<int32_t> propose(int32_t pending, int k) override;
    void truncate(int64_t) override {}
    void reset() override {}

private:
    const BlockDrafting& block_;
    Session& session_;
    TensorMap inputs_;
};

// The drafts of a block program's outputs (BlockDrafting): "drafts" as they are, or the path through "candidates"
// that takes the best score of each row in turn, starting from row 0.
std::vector<int32_t> block_drafts(const TensorMap& outputs);

}  // namespace xs::spec
