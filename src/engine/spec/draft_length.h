#pragma once

// How many drafts a speculative round asks for when that is left to the engine (EngineOptions::adaptive_k): the count
// that yields the most tokens a millisecond, from the acceptance a request's rounds have shown and from what rounds
// of each count cost on this device. A round of k drafts yields 1 + a1 + a1 a2 + ... + a1 ... ak tokens, where ai is
// the chance the i-th draft is accepted once the drafts before it were (a chain drafter's later drafts are accepted
// less often than its first); it costs the drafter's time for k drafts and a verify of k + 1 rows, both measured as
// rounds run (a verify of a few rows costs little more than one on a discrete GPU; a mixture of experts' rows each
// read experts of their own; a chain drafter's every draft is a run of its own).

#include <cstdint>
#include <vector>

namespace xs::spec {

// What a request's drafts have shown, draft by draft: each one the target reached (the drafts before it accepted) is
// accepted or not. Recent rounds count most: the counts decay by kDecay a round, so a request whose text starts or
// stops repeating itself (prompt lookup) moves within some ten rounds. A draft no recent round reached is taken to be
// accepted as often as the one before it.
class Acceptance {
public:
    static constexpr double kDecay = 0.9;
    // Before any round: the first draft accepted with probability `prior`.
    explicit Acceptance(double prior = 0.6);
    // A round proposed `drafted` drafts and the target accepted the first `accepted` of them.
    void round(int drafted, int accepted);
    // The probability the i-th draft (from 1) is accepted, the drafts before it accepted.
    double rate(int i) const;
    // The tokens a round of `k` drafts yields on average: the target's own and the accepted drafts.
    double tokens(int k) const;

private:
    static constexpr double kPriorDrafts = 2.0;  // the weight of what a draft is taken to be before its own rounds
    double prior_;
    std::vector<double> accepted_, reached_;  // by draft, from the first
};

class DraftLength {
public:
    // Rounds of up to `most` drafts.
    explicit DraftLength(int most);
    // A round of `drafts` drafts (the pending token and them verified) took `draft_ms` in the drafter and `verify_ms`
    // in the target's verify. The first round of each count is not counted: it pays for a new step size's setup.
    void record(int drafts, double draft_ms, double verify_ms);
    // The count of drafts, 1 to min(limit, most), with the most tokens a millisecond for `acceptance`. Nothing
    // measured yet: 1; then the counts past it, taken to cost as much, are tried, and measured.
    int best(const Acceptance& acceptance, int limit) const;
    // What a round of `drafts` drafts costs: as measured, or between the counts measured on either side (in a line),
    // past the largest one on the line of the two largest (never falling), below the smallest as the smallest. 0:
    // nothing measured yet.
    double cost(int drafts) const;

private:
    static constexpr double kKeep = 0.8;  // an average's weight on what it held: costs move over some five rounds
    std::vector<double> ms_;   // by drafts in the round, 0: not measured
    std::vector<int> rounds_;  // by drafts in the round, the first not counted
};

}  // namespace xs::spec
