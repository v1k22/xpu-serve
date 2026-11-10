#include "engine/spec/drafter.h"

#include "engine/spec/block_drafter.h"
#include "engine/spec/chain_drafter.h"
#include "util/error.h"

namespace xs::spec {

std::unique_ptr<Drafter> make_drafter(const Speculation& spec, Session& session) {
    XS_CHECK((spec.chain() != nullptr) != (spec.block() != nullptr), "spec: drafter ", spec.drafter_name(),
             " must draft by chain or by block");
    if (spec.block() != nullptr) return std::make_unique<BlockDrafter>(spec, session);
    return std::make_unique<ChainDrafter>(spec, session);
}

}  // namespace xs::spec
