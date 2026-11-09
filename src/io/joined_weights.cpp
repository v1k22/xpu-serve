#include "io/joined_weights.h"

#include "util/error.h"

namespace xs {

JoinedWeights::JoinedWeights(std::unique_ptr<WeightStore> first, std::unique_ptr<WeightStore> second,
                             std::string prefix)
    : first_(std::move(first)), second_(std::move(second)), prefix_(std::move(prefix)) {
    XS_CHECK(!prefix_.empty(), "joined weights need a prefix for the second store");
    for (const std::string& name : first_->names()) {
        XS_CHECK(name.rfind(prefix_, 0) != 0, "weight '", name, "' already starts with the prefix '", prefix_, "'");
    }
}

const TensorView* JoinedWeights::find(const std::string& name) const {
    if (name.rfind(prefix_, 0) == 0) return second_->find(name.substr(prefix_.size()));
    return first_->find(name);
}

std::vector<std::string> JoinedWeights::names() const {
    std::vector<std::string> out = first_->names();
    for (const std::string& name : second_->names()) out.push_back(prefix_ + name);
    return out;
}

}  // namespace xs
