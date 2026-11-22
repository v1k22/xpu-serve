#pragma once

// Weights held in host memory by name: tensors a program reads that no checkpoint file holds (an adapter's, read and
// converted at load).

#include <map>
#include <string>
#include <vector>

#include "io/safetensors.h"
#include "tensor/tensor.h"

namespace xs {

class HostWeights final : public WeightStore {
public:
    void add(const std::string& name, HostTensor t) {
        tensors_[name] = std::move(t);
        views_[name] = tensors_[name].view();
    }
    const TensorView* find(const std::string& name) const override {
        const auto it = views_.find(name);
        return it == views_.end() ? nullptr : &it->second;
    }
    std::vector<std::string> names() const override {
        std::vector<std::string> out;
        for (const auto& [name, view] : views_) out.push_back(name);
        return out;
    }

private:
    std::map<std::string, HostTensor> tensors_;
    std::map<std::string, TensorView> views_;
};

}  // namespace xs
