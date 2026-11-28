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
    TensorMap out;
    begin(pending, inputs_);
    session_.run(kDraft, inputs_, out);
    std::vector<int32_t> drafts{first(out, 0)};
    for (int s = 1; s < k; ++s) {
        chained(s, inputs_);
        session_.run(kDraft, inputs_, out);
        drafts.push_back(next(out, 0));
    }
    end();
    return drafts;
}

int64_t ChainDrafter::begin(int32_t pending, TensorMap& inputs) {
    XS_CHECK(rows() > 0 && static_cast<int64_t>(next_.size()) == rows() - 1, "spec: no target row to draft after");
    next_.push_back(pending);
    position_ = start_ + rows() - 1;  // of the last pair: chained drafts follow it
    const int64_t n = rows();
    // The pairs are the sequence's: the drafter keeps them; what the chained runs append is cut back.
    kept_ = chain_.drafter_keeps_state() ? session_.num_cached(kDraft) + n : 0;
    inputs.clear();
    chain_.prepare_draft(std::span<const int32_t>(next_).first(static_cast<size_t>(n)),
                        std::span<const float>(hidden_).first(static_cast<size_t>(n * width_)), start_, 0, inputs);
    return n;
}

int32_t ChainDrafter::first(const TensorMap& out, int64_t row) {
    drop(rows());
    return take(out, row);
}

void ChainDrafter::chained(int depth, TensorMap& inputs) const {
    const int32_t token[] = {last_};
    inputs.clear();
    chain_.prepare_draft(token, last_hidden_, position_, depth, inputs);
}

int32_t ChainDrafter::next(const TensorMap& out, int64_t row) { return take(out, row); }

void ChainDrafter::end() {
    if (chain_.drafter_keeps_state()) session_.truncate(kDraft, kept_);  // chained pairs were drafts, not the sequence
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
    drop(n);
}

void ChainDrafter::drop(int64_t n) {
    hidden_.erase(hidden_.begin(), hidden_.begin() + n * width_);
    next_.erase(next_.begin(), next_.begin() + std::min(n, static_cast<int64_t>(next_.size())));
    start_ += n;
}

int32_t ChainDrafter::take(const TensorMap& out, int64_t row) {
    const HostTensor& logits = out.at("logits");
    const HostTensor& hidden = out.at("hidden");
    XS_CHECK(logits.rank() == 2 && row < logits.dim(0) && hidden.rank() == 2 && row < hidden.dim(0),
             "spec: the drafter's outputs hold no row ", row);
    const int64_t n = logits.dim(1);
    const float* l = logits.data<float>() + row * n;
    const auto best = std::max_element(l, l + n) - l;
    const auto it = out.find("candidates");
    if (it == out.end()) {
        last_ = static_cast<int32_t>(best);
    } else {  // [n] for the last pair's run, [rows, n] for every row's
        const HostTensor& c = it->second;
        last_ = c.data<int32_t>()[(c.rank() == 2 ? row * c.dim(1) : 0) + best];
    }
    const int64_t width = hidden.dim(1);
    last_hidden_.assign(hidden.data<float>() + row * width, hidden.data<float>() + (row + 1) * width);
    return last_;
}

}  // namespace xs::spec
