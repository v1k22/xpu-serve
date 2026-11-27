#pragma once

// A rule a request's output follows token by token (response_format's JSON, held by a grammar), which the engine drives
// without knowing what it is. A constrained request reads its rows' logits, not their argmax: each row masked by the
// constraint as the tokens before it leave it (in a speculative round, the drafts before that row), the stop tokens
// kept only where the output may end, then sampled or its masked argmax taken.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace xs {

// One generation's constraint; its state is the tokens accepted so far. Used on the engine thread only.
class TokenConstraint {
public:
    virtual ~TokenConstraint() = default;
    // Sets the logits of the tokens that may not come next to -infinity, and returns whether it allows any. The stop
    // tokens are the engine's: it keeps them where complete() and masks them elsewhere, whatever this does with them.
    virtual bool mask(std::span<float> logits) = 0;
    // `token`, which mask() allowed, comes next.
    virtual void accept(int32_t token) = 0;
    // The output may end here.
    virtual bool complete() const = 0;
    // How many of `tokens`, one after another from here, it allows; nothing is accepted.
    virtual size_t allowed_prefix(std::span<const int32_t> tokens) const = 0;
};

// What a request carries (GenerateRequest::constraint): it starts each generation's constraint.
class OutputConstraint {
public:
    virtual ~OutputConstraint() = default;
    // On the engine thread, as the generation starts.
    virtual std::unique_ptr<TokenConstraint> start() const = 0;
};

}  // namespace xs
