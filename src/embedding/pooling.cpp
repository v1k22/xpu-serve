#include "embedding/pooling.h"

#include <algorithm>
#include <cmath>

#include "util/error.h"

namespace xs {

std::vector<float> pool(const HostTensor& hidden, Pooling pooling, bool normalize) {
    XS_CHECK(hidden.dtype() == DType::F32 && hidden.rank() == 2 && hidden.dim(0) >= 1, "pool: hidden states ",
             dims_to_string(hidden.shape()), " ", dtype_name(hidden.dtype()), ", expected [T, D] f32 with T >= 1");
    const int64_t rows = pooling == Pooling::Cls ? 1 : hidden.dim(0);
    const int64_t width = hidden.dim(1);
    const float* h = hidden.data<float>();
    std::vector<double> sum(static_cast<size_t>(width), 0.0);
    for (int64_t t = 0; t < rows; ++t) {
        for (int64_t d = 0; d < width; ++d)
            sum[static_cast<size_t>(d)] += h[t * width + d];
    }
    double norm = 0;
    for (double& v : sum) {
        v /= static_cast<double>(rows);
        norm += v * v;
    }
    const double scale = normalize ? 1.0 / std::max(std::sqrt(norm), 1e-12) : 1.0;
    std::vector<float> out(sum.size());
    std::transform(sum.begin(), sum.end(), out.begin(), [scale](double v) { return static_cast<float>(v * scale); });
    return out;
}

}  // namespace xs
