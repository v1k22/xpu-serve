#include "engine/spec/chain_drafter.h"

#include <algorithm>
#include <limits>

#include "util/error.h"

namespace xs::spec {

namespace {

const ChainDrafting& chain_of(const Speculation& spec) {
    XS_CHECK(spec.chain() != nullptr, "spec: ", spec.drafter_name(), " is not a chain drafter");
    return *spec.chain();
}

}  // namespace

ChainDrafter::ChainDrafter(const Speculation& spec, Session& session) : chain_(chain_of(spec)), session_(session) {}

std::span<const std::string> ChainDrafter::target_outputs() const {
    static const std::string kHidden[] = {"hidden"};
    return kHidden;
}

int ChainDrafter::max_drafts() const {
    return std::numeric_limits<int>::max();
}

int64_t ChainDrafter::rows() const {
    return width_ == 0 ? 0 : static_cast<int64_t>(hidden_.size()) / width_;
}

void ChainDrafter::add(const TensorMap& outputs, int64_t keep, int64_t start, std::span<const int32_t> tokens) {
    const HostTensor& hidden = outputs.at("hidden");
    XS_CHECK(hidden.dtype() == DType::F32 && hidden.rank() == 2 && keep >= 1 && keep <= hidden.dim(0) &&
                 static_cast<int64_t>(tokens.size()) >= keep,
             "spec: hidden output is ", dtype_name(hidden.dtype()), dims_to_string(hidden.shape()), " for ", keep,
             " kept rows of ", tokens.size(), " tokens");
    if (width_ == 0) width_ = hidden.dim(1);
    XS_CHECK(hidden.dim(1) == width_, "spec: hidden rows of width ", hidden.dim(1), " after ", width_);
    if (start == start_ + rows()) {
        if (rows() > static_cast<int64_t>(next_.size())) next_.push_back(tokens[0]);
    } else {
        // Not where the waiting rows end (the target restarted elsewhere): the drafter never sees their pairs.
        hidden_.clear();
        next_.clear();
        start_ = start;
    }
    hidden_.insert(hidden_.end(), hidden.data<float>(), hidden.data<float>() + keep * width_);
    next_.insert(next_.end(), tokens.begin() + 1, tokens.begin() + keep);
    if (!chain_.drafter_keeps_state() && rows() > 1) {  // a round starts from the last pair alone
        const int64_t drop = rows() - 1;
        hidden_.erase(hidden_.begin(), hidden_.begin() + drop * width_);
        next_.erase(next_.begin(), next_.begin() + std::min(drop, static_cast<int64_t>(next_.size())));
        start_ += drop;
    }
}

void ChainDrafter::flush() {
    const auto complete = static_cast<int64_t>(next_.size());
    if (!chain_.drafter_keeps_state() || complete == 0) return;
    TensorMap out;
    run_pairs(complete, out);
}

std::vector<int32_t> ChainDrafter::propose(int32_t pending, int k) {
    XS_CHECK(k >= 1, "spec: a round drafts at least one token");
    XS_CHECK(rows() > 0 && static_cast<int64_t>(next_.size()) == rows() - 1, "spec: no target row to draft after");
    next_.push_back(pending);
    const int64_t position = start_ + rows() - 1;  // of the last pair: chained drafts follow it
    TensorMap out;
    run_pairs(rows(), out);
    std::vector<int32_t> drafts{pick(out)};
    const int64_t kept = chain_.drafter_keeps_state() ? session_.num_cached(kDraft) : 0;
    for (int s = 1; s < k; ++s) {
        const HostTensor& h = out.at("hidden");
        const int32_t token[] = {drafts.back()};
        inputs_.clear();
        chain_.prepare_draft(token, std::span<const float>(h.data<float>(), static_cast<size_t>(h.numel())), position,
                            s, inputs_);
        session_.run(kDraft, inputs_, out);
        drafts.push_back(pick(out));
    }
    if (chain_.drafter_keeps_state()) session_.truncate(kDraft, kept);  // chained pairs were drafts, not the sequence
    return drafts;
}

void ChainDrafter::truncate(int64_t n) {
    if (n <= start_) {
        // The drafter took the pair of position n - 1 with the token the target no longer holds at n: it restarts.
        if (chain_.drafter_keeps_state()) session_.truncate(kDraft, 0);
        hidden_.clear();
        next_.clear();
        start_ = n;
        return;
    }
    // The row at n - 1 waits for the token the target is fed next at n.
    const int64_t keep = std::min(n - start_, rows());
    hidden_.resize(static_cast<size_t>(keep * width_));
    next_.resize(std::min(next_.size(), static_cast<size_t>(keep - 1)));
}

void ChainDrafter::reset() {
    hidden_.clear();
    next_.clear();
    start_ = 0;
}

void ChainDrafter::run_pairs(int64_t n, TensorMap& out) {
    inputs_.clear();
    chain_.prepare_draft(std::span<const int32_t>(next_).first(static_cast<size_t>(n)),
                        std::span<const float>(hidden_).first(static_cast<size_t>(n * width_)), start_, 0, inputs_);
    session_.run(kDraft, inputs_, out);
    hidden_.erase(hidden_.begin(), hidden_.begin() + n * width_);
    next_.erase(next_.begin(), next_.begin() + n);
    start_ += n;
}

int32_t ChainDrafter::pick(const TensorMap& out) const {
    const HostTensor& logits = out.at("logits");
    const float* l = logits.data<float>();
    const auto best = std::max_element(l, l + logits.numel()) - l;
    const auto it = out.find("candidates");
    return static_cast<int32_t>(it == out.end() ? best : it->second.data<int32_t>()[best]);
}

}  // namespace xs::spec
