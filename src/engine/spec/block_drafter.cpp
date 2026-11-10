#include "engine/spec/block_drafter.h"

#include <algorithm>

#include "util/error.h"

namespace xs::spec {

namespace {

const BlockDrafting& block_of(const Speculation& spec) {
    XS_CHECK(spec.block() != nullptr, "spec: ", spec.drafter_name(), " is not a block drafter");
    return *spec.block();
}

}  // namespace

BlockDrafter::BlockDrafter(const Speculation& spec, Session& session) : block_(block_of(spec)), session_(session) {}

int BlockDrafter::max_drafts() const {
    return static_cast<int>(block_.block_drafts());
}

std::vector<int32_t> BlockDrafter::propose(int32_t pending, int k) {
    XS_CHECK(k >= 1 && k <= max_drafts(), "spec: ", k, " drafts of a block of ", max_drafts());
    // The draft program's slots hold what the target's programs appended: the positions before the pending token.
    const int64_t position = session_.num_cached(kDraft);
    inputs_.clear();
    block_.prepare_block(pending, position, inputs_);
    TensorMap out;
    session_.run(kDraft, inputs_, out);
    session_.truncate(kDraft, position);  // the block's rows were drafts, not the sequence
    std::vector<int32_t> drafts = block_drafts(out);
    XS_CHECK(static_cast<int>(drafts.size()) >= k, "spec: the block program gave ", drafts.size(), " drafts, ", k,
             " asked");
    drafts.resize(static_cast<size_t>(k));
    return drafts;
}

std::vector<int32_t> block_drafts(const TensorMap& outputs) {
    if (const auto it = outputs.find("drafts"); it != outputs.end()) {
        const HostTensor& d = it->second;
        XS_CHECK(d.dtype() == DType::I32, "spec: drafts output is ", dtype_name(d.dtype()));
        return {d.data<int32_t>(), d.data<int32_t>() + d.numel()};
    }
    const HostTensor& cand = outputs.at("candidates");
    const HostTensor& scores = outputs.at("scores");
    XS_CHECK(cand.dtype() == DType::I32 && cand.rank() == 2 && scores.dtype() == DType::F32 &&
                 scores.shape() == (Dims{cand.dim(0), cand.dim(1), cand.dim(1)}),
             "spec: block outputs are candidates ", dtype_name(cand.dtype()), dims_to_string(cand.shape()),
             " and scores ", dtype_name(scores.dtype()), dims_to_string(scores.shape()));
    const int64_t n = cand.dim(0);
    const int64_t width = cand.dim(1);
    std::vector<int32_t> drafts;
    int64_t from = 0;  // the candidate the draft before took
    for (int64_t i = 0; i < n; ++i) {
        const float* row = scores.data<float>() + (i * width + from) * width;
        from = std::max_element(row, row + width) - row;
        drafts.push_back(cand.data<int32_t>()[i * width + from]);
    }
    return drafts;
}

}  // namespace xs::spec
