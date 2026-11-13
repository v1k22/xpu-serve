// Embedding models on tiny checkpoints the test writes: what must hold whatever the weights are. The numbers
// themselves are checked against sentence-transformers in embedding_parity.cpp.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>

#include "../backend/device_test_support.h"
#include "embedding_test_util.h"
#include "models/embedding.h"

namespace {

using xs::HostTensor;
using xs::TensorMap;
using xs::test::TempDir;
using xs::test::TinyEncoder;

HostTensor run(xs::Session& session, const TensorMap& inputs) {
    TensorMap out;
    session.run(inputs, out);
    return std::move(out.at("hidden"));
}

double cosine(const HostTensor& a, const HostTensor& b) {
    REQUIRE(a.shape() == b.shape());
    double dot = 0, aa = 0, bb = 0;
    for (int64_t i = 0; i < a.numel(); ++i) {
        const double x = a.data<float>()[i], y = b.data<float>()[i];
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    return dot / std::sqrt(aa * bb);
}

double max_diff(const HostTensor& a, const HostTensor& b) {
    REQUIRE(a.shape() == b.shape());
    double d = 0;
    for (int64_t i = 0; i < a.numel(); ++i)
        d = std::max(d, static_cast<double>(std::fabs(a.data<float>()[i] - b.data<float>()[i])));
    return d;
}

// Every position moved by `by`.
TensorMap shifted(TensorMap inputs, int32_t by) {
    for (int32_t& p : inputs.at("positions").span<int32_t>())
        p += by;
    return inputs;
}

const std::vector<int32_t> kIds = {1, 17, 230, 5, 99, 42, 7, 2};

}  // namespace

TEST_SUITE("embedding models") {
    TEST_CASE("an encoder reads a sequence in one run that keeps nothing for the next") {
        for (const char* arch : {"bert", "nomic_bert"}) {
            CAPTURE(arch);
            const TempDir dir(std::string("xs_tiny_encoder_") + arch);
            TinyEncoder m;
            m.arch = arch;
            m.normalize = true;
            xs::test::write_tiny_encoder(dir.path, m, 11);
            const auto model = xs::load_embedding_model(dir.path);
            const xs::EmbeddingSpec& spec = model->spec();
            CHECK(spec.arch == arch);
            CHECK(spec.id == dir.path.filename().string());
            CHECK(spec.dimension == 64);
            CHECK(spec.vocab_size == 320);
            CHECK(spec.max_tokens == 48);
            CHECK(spec.pooling == xs::Pooling::Mean);
            CHECK(spec.normalize);
            CHECK(!model->graph().stateful());

            const auto session = xs::make_backend("cpu-ref")->compile(model->graph(), model->weights(),
                                                                      xs::SessionOptions{.precision = xs::DType::F32});
            const HostTensor first = run(*session, model->prepare_inputs(kIds));
            CHECK(first.shape() == (xs::Dims{8, 64}));
            run(*session, model->prepare_inputs(std::vector<int32_t>(40, 3)));
            CHECK(session->num_cached() == 0);
            CHECK(max_diff(first, run(*session, model->prepare_inputs(kIds))) == 0);  // nothing carried over
            // Every token sees every other: changing the last token changes the first token's row.
            std::vector<int32_t> other = kIds;
            other.back() = 300;
            const HostTensor changed = run(*session, model->prepare_inputs(other));
            double row0 = 0;
            for (int64_t d = 0; d < 64; ++d)
                row0 = std::max(row0, static_cast<double>(std::fabs(changed.data<float>()[d] - first.data<float>()[d])));
            CHECK(row0 > 1e-3);

            // BERT's positions are rows of a table: a shift changes every row. NomicBERT's are rotations of queries and
            // keys only, which attention reads as differences: a shift changes nothing.
            const HostTensor moved = run(*session, shifted(model->prepare_inputs(kIds), 5));
            if (std::string(arch) == "bert") {
                CHECK(max_diff(first, moved) > 1e-2);
            } else {
                CHECK(max_diff(first, moved) < 1e-4);
                std::vector<int32_t> reversed(kIds.rbegin(), kIds.rend());  // but the order of tokens matters
                TensorMap in = model->prepare_inputs(reversed);
                const HostTensor back = run(*session, in);
                double d = 0;
                for (int64_t t = 0; t < 8; ++t) {
                    for (int64_t c = 0; c < 64; ++c)
                        d = std::max(d, static_cast<double>(std::fabs(back.data<float>()[(7 - t) * 64 + c] -
                                                                       first.data<float>()[t * 64 + c])));
                }
                CHECK(d > 1e-2);
            }

            CHECK_THROWS_WITH_AS(model->prepare_inputs(std::vector<int32_t>(49, 1)),
                                 doctest::Contains("reads 1 to 48 tokens, got 49"), xs::Error);
            CHECK_THROWS_WITH_AS(model->prepare_inputs(std::vector<int32_t>{1, 320}),
                                 doctest::Contains("token id 320 is outside"), xs::Error);

            SUBCASE("the GPU agrees with the reference backend") {
                namespace dev = xs::test::device;
                if (!dev::enabled()) return;
                const auto gpu = dev::backend().compile(model->graph(), model->weights(), dev::options());
                for (const auto& ids : {kIds, std::vector<int32_t>(48, 9)}) {
                    const HostTensor ref = run(*session, model->prepare_inputs(ids));
                    const HostTensor out = run(*gpu, model->prepare_inputs(ids));
                    if (dev::is_f16()) {
                        CHECK(cosine(ref, out) >= 0.999);
                    } else {
                        CHECK(max_diff(ref, out) < 1e-4);
                    }
                }
            }
        }
    }
}
