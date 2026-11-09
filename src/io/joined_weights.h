#pragma once

// Two weight stores read as one: the second's tensors appear under a name prefix (a drafter checkpoint beside its
// target, so one session can upload both and share what the drafter reads of the target).

#include <memory>
#include <string>
#include <vector>

#include "io/safetensors.h"

namespace xs {

class JoinedWeights final : public WeightStore {
public:
    JoinedWeights(std::unique_ptr<WeightStore> first, std::unique_ptr<WeightStore> second, std::string prefix);

    const TensorView* find(const std::string& name) const override;
    std::vector<std::string> names() const override;

private:
    std::unique_ptr<WeightStore> first_;
    std::unique_ptr<WeightStore> second_;
    std::string prefix_;
};

}  // namespace xs
