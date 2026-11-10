#pragma once

// The drafter side of speculative decoding (DESIGN.md 10): what proposes the tokens a verify step checks. The engine
// owns the loop (verify, acceptance, rollback) and drives the model's drafter through this interface, whatever its
// way of drafting (models/model.h: ChainDrafting, BlockDrafting).

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "models/model.h"

namespace xs::spec {

// The programs of a speculating session, in the order the engine compiles them.
enum Program : size_t { kStep = 0, kPrefill = 1, kVerify = 2, kDraft = 3 };

class Drafter {
public:
    virtual ~Drafter() = default;
    // The outputs of the target's prefill and verify programs the drafter reads, beside the engine's own.
    virtual std::span<const std::string> target_outputs() const = 0;
    // The most drafts a round can ask for.
    virtual int max_drafts() const = 0;
    // The target ran `tokens` at positions start, start + 1, ... and its first `keep` rows count (a verify keeps the
    // accepted ones); `outputs` holds target_outputs() of that run.
    virtual void add(const TensorMap& outputs, int64_t keep, int64_t start, std::span<const int32_t> tokens) = 0;
    // The end of a prefill chunk: bounds what waits in the drafter.
    virtual void flush() = 0;
    // Drafts `k` tokens after `pending`, the token after the last row added.
    virtual std::vector<int32_t> propose(int32_t pending, int k) = 0;
    // The target kept its first `n` positions.
    virtual void truncate(int64_t n) = 0;
    virtual void reset() = 0;
};

// The runner of the model's way of drafting: over `session`, whose programs are in Program order.
std::unique_ptr<Drafter> make_drafter(const Speculation& spec, Session& session);

}  // namespace xs::spec
