#include "models/common/expert_banks.h"

#include <vector>

namespace xs {

void join_expert_banks(SafetensorsWeights& weights, std::span<const std::string> experts, int64_t count, int64_t inter,
                       int64_t hidden) {
    for (const std::string& E : experts) {
        if (weights.contains(E + "gate_up_proj") || !weights.contains(E + "0.gate_proj.weight")) continue;
        std::vector<std::string> gate_up, down;
        for (int64_t e = 0; e < count; ++e) {
            const std::string X = E + std::to_string(e) + ".";
            gate_up.push_back(X + "gate_proj.weight");
            gate_up.push_back(X + "up_proj.weight");
            down.push_back(X + "down_proj.weight");
        }
        weights.join_rows(E + "gate_up_proj", gate_up, {count, 2 * inter, hidden});
        weights.join_rows(E + "down_proj", down, {count, hidden, inter});
    }
}

}  // namespace xs
