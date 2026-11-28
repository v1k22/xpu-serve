#include "engine/spec/draft_length.h"

#include <algorithm>

#include "util/error.h"

namespace xs::spec {

Acceptance::Acceptance(double prior) : prior_(prior) {
    XS_CHECK(prior > 0.0 && prior < 1.0, "spec: an acceptance prior of ", prior);
}

void Acceptance::round(int drafted, int accepted) {
    XS_CHECK(accepted >= 0 && accepted <= drafted, "spec: ", accepted, " of ", drafted, " drafts accepted");
    if (drafted == 0) return;
    // Every accepted draft was reached; so was the first rejected one, when there is one.
    const auto reached = static_cast<size_t>(std::min(drafted, accepted + 1));
    if (accepted_.size() < reached) {
        accepted_.resize(reached, 0.0);
        reached_.resize(reached, 0.0);
    }
    for (size_t i = 0; i < accepted_.size(); ++i) {
        accepted_[i] *= kDecay;
        reached_[i] *= kDecay;
        if (i < reached) {
            reached_[i] += 1.0;
            accepted_[i] += static_cast<int>(i) < accepted ? 1.0 : 0.0;
        }
    }
}

double Acceptance::rate(int i) const {
    double r = prior_;
    for (size_t at = 0; at < static_cast<size_t>(std::max(i, 1)); ++at) {
        const double reached = at < reached_.size() ? reached_[at] : 0.0;
        const double accepted = at < accepted_.size() ? accepted_[at] : 0.0;
        r = (accepted + kPriorDrafts * r) / (reached + kPriorDrafts);
    }
    // Never certain either way: a request that rejected (or accepted) everything so far still tries rounds.
    return std::clamp(r, 0.02, 0.98);
}

double Acceptance::tokens(int k) const {
    double total = 1.0, reach = 1.0;
    for (int i = 1; i <= k; ++i) {
        reach *= rate(i);
        total += reach;
    }
    return total;
}

DraftLength::DraftLength(int most) : ms_(static_cast<size_t>(most) + 1, 0.0), rounds_(static_cast<size_t>(most) + 1, 0) {
    XS_CHECK(most >= 1, "spec: rounds of up to ", most, " drafts");
}

void DraftLength::record(int drafts, double draft_ms, double verify_ms) {
    if (drafts < 1 || drafts >= static_cast<int>(ms_.size())) return;
    const auto k = static_cast<size_t>(drafts);
    if (rounds_[k]++ == 0) return;
    const double ms = std::max(draft_ms + verify_ms, 1e-3);
    ms_[k] = ms_[k] == 0.0 ? ms : kKeep * ms_[k] + (1.0 - kKeep) * ms;
}

double DraftLength::cost(int drafts) const {
    const auto k = static_cast<size_t>(drafts);
    if (k == 0 || k >= ms_.size()) return 0.0;
    if (ms_[k] > 0.0) return ms_[k];
    size_t below = 0, below2 = 0, above = 0;  // the nearest measured counts: below k (and the one below it), above
    for (size_t j = k; j-- > 1;) {
        if (ms_[j] == 0.0) continue;
        if (below == 0) {
            below = j;
        } else {
            below2 = j;
            break;
        }
    }
    for (size_t j = k + 1; j < ms_.size(); ++j) {
        if (ms_[j] > 0.0) {
            above = j;
            break;
        }
    }
    const auto line = [&](size_t a, size_t b) {
        return ms_[a] + (ms_[b] - ms_[a]) * (static_cast<double>(k) - a) / (static_cast<double>(b) - a);
    };
    if (below > 0 && above > 0) return line(below, above);
    if (above > 0) return ms_[above];
    if (below2 > 0) return std::max(ms_[below], line(below2, below));
    return below > 0 ? ms_[below] : 0.0;
}

int DraftLength::best(const Acceptance& acceptance, int limit) const {
    const int top = std::min(limit, static_cast<int>(ms_.size()) - 1);
    if (top <= 1) return std::max(top, 1);
    int pick = 1;
    double best_rate = -1.0;
    for (int k = 1; k <= top; ++k) {
        const double ms = cost(k);
        if (ms == 0.0) return k;  // nothing measured yet: one draft, then (taken to cost as much) the most
        const double per_ms = acceptance.tokens(k) / ms;
        if (per_ms > best_rate) {
            best_rate = per_ms;
            pick = k;
        }
    }
    return pick;
}

}  // namespace xs::spec
