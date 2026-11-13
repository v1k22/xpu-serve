// The embedding runtime: pooling as sentence-transformers pools one sequence, the runner over a tiny checkpoint (one
// run a sequence, the limits it refuses), and the service (text through the checkpoint's tokenizer, requests from
// several threads).

#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <thread>

#include "convert/tiny_gguf.h"
#include "embedding/pooling.h"
#include "embedding/runner.h"
#include "embedding/service.h"
#include "model/embedding_test_util.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"

namespace {

using xs::HostTensor;
using xs::Pooling;
using xs::test::TempDir;
using xs::test::TinyEncoder;

double norm(const std::vector<float>& v) {
    double s = 0;
    for (const float x : v)
        s += static_cast<double>(x) * x;
    return std::sqrt(s);
}

xs::EmbeddingRunnerOptions cpu_options(int64_t max_context = 4096) {
    xs::EmbeddingRunnerOptions o;
    o.backend = "cpu-ref";
    o.session.precision = xs::DType::F32;
    o.session.max_context = max_context;
    return o;
}

}  // namespace

TEST_SUITE("embedding runtime") {
    TEST_CASE("pooling: the mean of every row or the first row, then the L2 norm when asked") {
        const HostTensor hidden = HostTensor::from_vector<float>(xs::DType::F32, {3, 2}, {1, 2, 3, 4, 5, 9});
        CHECK(xs::pool(hidden, Pooling::Mean, false) == std::vector<float>{3, 5});
        CHECK(xs::pool(hidden, Pooling::Cls, false) == std::vector<float>{1, 2});
        const std::vector<float> mean = xs::pool(hidden, Pooling::Mean, true);
        CHECK(mean[0] == doctest::Approx(3 / std::sqrt(34.0)));
        CHECK(mean[1] == doctest::Approx(5 / std::sqrt(34.0)));
        const std::vector<float> cls = xs::pool(hidden, Pooling::Cls, true);
        CHECK(cls[0] == doctest::Approx(1 / std::sqrt(5.0)));
        CHECK(norm(cls) == doctest::Approx(1.0));
        const HostTensor zero(xs::DType::F32, {2, 3});
        CHECK(xs::pool(zero, Pooling::Mean, true) == std::vector<float>{0, 0, 0});  // torch's F.normalize: no NaN
        CHECK_THROWS_AS(xs::pool(HostTensor(xs::DType::F32, {0, 3}), Pooling::Mean, false), xs::Error);
        CHECK_THROWS_AS(xs::pool(HostTensor(xs::DType::F32, {4}), Pooling::Mean, false), xs::Error);
    }

    TEST_CASE("the runner embeds each sequence in a run of its own and refuses what the model cannot read") {
        const TempDir dir("xs_embedding_runner");
        TinyEncoder m;
        m.arch = "nomic_bert";
        m.normalize = true;
        xs::test::write_tiny_encoder(dir.path, m, 21);
        const auto runner = xs::EmbeddingRunner::create(dir.path, cpu_options());
        CHECK(runner->spec().max_tokens == 48);
        const std::vector<int32_t> a = {1, 50, 60, 70, 2}, b = {1, 300, 2};
        const xs::EmbeddingResult r = runner->embed({a, b, a});
        REQUIRE(r.embeddings.size() == 3);
        CHECK(r.tokens == 13);
        CHECK(r.timing.device_ms > 0);
        CHECK(r.timing.total_ms >= r.timing.device_ms);
        CHECK(r.embeddings[0] == r.embeddings[2]);  // a request's other inputs change nothing
        CHECK(r.embeddings[1] != r.embeddings[0]);
        CHECK(r.embeddings[0] == runner->embed({a}).embeddings[0]);
        CHECK(r.embeddings[0] == xs::pool(runner->hidden_states(a), Pooling::Mean, true));
        CHECK(norm(r.embeddings[1]) == doctest::Approx(1.0));
        CHECK(r.embeddings[1].size() == 64);

        const xs::EmbeddingSpec& spec = runner->spec();
        CHECK(*xs::sequence_problem(spec, {}) == "no token");
        CHECK(xs::sequence_problem(spec, std::vector<int32_t>(49, 1))->find("49 tokens, more than the 48 the model reads") !=
              std::string::npos);
        CHECK(*xs::sequence_problem(spec, std::vector<int32_t>{1, 320}) == "token id 320, outside the vocabulary (0 to 319)");
        CHECK(xs::sequence_problem(spec, std::vector<int32_t>{-1})->find("token id -1") != std::string::npos);
        CHECK(!xs::sequence_problem(spec, std::vector<int32_t>(48, 319)).has_value());
        CHECK_THROWS_WITH_AS(runner->embed({a, std::vector<int32_t>(49, 1)}), doctest::Contains("49 tokens"), xs::Error);

        // A context below the model's limit lowers it.
        const auto short_runner = xs::EmbeddingRunner::create(dir.path, cpu_options(16));
        CHECK(short_runner->spec().max_tokens == 16);
        CHECK(xs::sequence_problem(short_runner->spec(), std::vector<int32_t>(17, 1)).has_value());
        CHECK(short_runner->embed({a}).embeddings[0] == r.embeddings[0]);
    }

    TEST_CASE("the service reads text through the checkpoint's tokenizer and embeds on its own thread") {
        const TempDir dir("xs_embedding_service");
        xs::test::write_tiny_encoder(dir.path, TinyEncoder{}, 22);
        std::ofstream(dir.path / "tokenizer.json") << xs::test::tiny_tokenizer_json().dump();
        auto tokenizer = xs::tok::Tokenizer::load(dir.path);
        const std::vector<int32_t> expected = tokenizer->with_special_tokens(tokenizer->encode("the cat sat"));
        const auto service = xs::EmbeddingService::create(std::move(tokenizer),
                                                          xs::EmbeddingRunner::create(dir.path, cpu_options()));
        CHECK(service->spec().dimension == 64);
        const std::vector<int32_t> ids = service->tokenize("the cat sat");
        CHECK(ids == expected);
        const auto runner = xs::EmbeddingRunner::create(dir.path, cpu_options());
        const std::vector<float> direct = runner->embed({ids}).embeddings[0];
        std::vector<std::thread> clients;
        std::vector<int> same(6, 0);
        for (int t = 0; t < 6; ++t) {
            clients.emplace_back([&, t] {
                std::vector<std::vector<int32_t>> sequences(static_cast<size_t>(t + 1), ids);
                const xs::EmbeddingCall call = service->embed(sequences);
                same[static_cast<size_t>(t)] =
                    call.result.embeddings.size() == sequences.size() && call.result.embeddings.back() == direct;
            });
        }
        for (std::thread& c : clients)
            c.join();
        CHECK(same == std::vector<int>(6, 1));
        CHECK_THROWS_AS(service->embed({{}}), xs::Error);  // the device thread's error reaches the caller
        CHECK(service->embed({ids}).result.embeddings[0] == direct);
    }
}
