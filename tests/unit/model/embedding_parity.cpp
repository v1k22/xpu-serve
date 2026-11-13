// An embedding model against sentence-transformers (tests/parity/embedding/reference.py): each case's token ids give
// the same final hidden states, pooled vector and sentence embedding. Opt-in:
//
//   XS_TEST_EMBEDDING_DIR   the checkpoint (a released one, or make_tiny.py's --arch bert | nomic_bert)
//   XS_EMBEDDING_FIXTURES   reference.py's output directory for it
//   XS_PARITY_BACKEND       cpu-ref (default) | sycl
//   XS_PARITY_DEVICE, XS_PARITY_PRECISION (f32 by default)
//   XS_EMBEDDING_TEXTS=1    also tokenize each text the reference did not cut and expect its ids (the checkpoint's
//                           tokenizer: WordPiece for the released models)

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "embedding/pooling.h"
#include "embedding/runner.h"
#include "embedding/service.h"
#include "parity_util.h"
#include "tokenizer/tokenizer.h"
#include "util/timer.h"

namespace {

using xs::test::env_or;

bool enabled() {
    return !env_or("XS_TEST_EMBEDDING_DIR", "").empty() && !env_or("XS_EMBEDDING_FIXTURES", "").empty();
}

// 1 - cosine: what a cosine near 1 differs by.
std::string gap(double cosine) {
    char text[32];
    std::snprintf(text, sizeof text, "%.2e", 1.0 - cosine);
    return text;
}

double max_magnitude(std::span<const float> v) {
    double m = 0;
    for (const float x : v)
        m = std::max(m, static_cast<double>(std::fabs(x)));
    return m;
}

}  // namespace

TEST_CASE("embedding model matches sentence-transformers" * doctest::skip(!enabled())) {
    const std::filesystem::path fixtures_dir = env_or("XS_EMBEDDING_FIXTURES", "");
    std::ifstream in(fixtures_dir / "fixtures.json");
    REQUIRE_MESSAGE(in.good(), "cannot read XS_EMBEDDING_FIXTURES/fixtures.json");
    const auto fixtures = nlohmann::json::parse(in);
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    const double min_cosine = f32 ? 0.99999 : 0.999;
    // Largest difference against the largest value of the reference vector.
    const double max_relative = f32 ? 1e-3 : 5e-2;

    xs::EmbeddingRunnerOptions options;
    options.backend = env_or("XS_PARITY_BACKEND", "cpu-ref");
    options.session = xs::test::parity_options();
    options.session.max_context = 1 << 20;
    const std::string dir = env_or("XS_TEST_EMBEDDING_DIR", "");
    const auto runner = xs::EmbeddingRunner::create(dir, options);
    const xs::EmbeddingSpec& spec = runner->spec();
    CHECK(xs::pooling_name(spec.pooling) == fixtures["pooling"].get<std::string>());
    CHECK(spec.normalize == fixtures["normalize"].get<bool>());
    CHECK(spec.max_tokens == fixtures["max_seq_length"].get<int64_t>());
    std::unique_ptr<xs::tok::Tokenizer> tokenizer;
    if (env_or("XS_EMBEDDING_TEXTS", "") == "1") tokenizer = xs::tok::Tokenizer::load(dir);

    double worst_hidden = 1, worst_embedding = 1;
    for (size_t i = 0; i < fixtures["cases"].size(); ++i) {
        const auto& c = fixtures["cases"][i];
        CAPTURE(i);
        const auto ids = c["ids"].get<std::vector<int32_t>>();
        if (tokenizer && !c["cut"].get<bool>()) CHECK(xs::embedding_tokens(*tokenizer, c["text"].get<std::string>()) == ids);
        xs::Timer timer;
        const xs::HostTensor hidden = runner->hidden_states(ids);
        const double ms = timer.elapsed_ms();
        const std::vector<float> ref_hidden = xs::test::read_f32(fixtures_dir / c["hidden"].get<std::string>());
        REQUIRE(hidden.numel() == static_cast<int64_t>(ref_hidden.size()));
        const xs::test::Compare h = xs::test::compare(hidden.span<float>(), ref_hidden);
        const std::vector<float> pooled = xs::pool(hidden, spec.pooling, false);
        const xs::test::Compare p = xs::test::compare(pooled, c["pooled"].get<std::vector<float>>());
        const std::vector<float> ref_embedding = c["embedding"].get<std::vector<float>>();
        const std::vector<float> embedding = xs::pool(hidden, spec.pooling, spec.normalize);
        const xs::test::Compare e = xs::test::compare(embedding, ref_embedding);
        MESSAGE("case ", i, ": ", ids.size(), " tokens in ", static_cast<int>(ms), " ms; hidden 1 - cosine ", gap(h.cosine),
                " max |diff| ", h.max_abs, " (largest value ", max_magnitude(ref_hidden), "); embedding 1 - cosine ",
                gap(e.cosine), " max |diff| ", e.max_abs);
        CHECK(h.cosine >= min_cosine);
        CHECK(h.max_abs <= max_relative * max_magnitude(ref_hidden));
        CHECK(p.cosine >= min_cosine);
        CHECK(e.cosine >= min_cosine);
        CHECK(e.max_abs <= max_relative * max_magnitude(ref_embedding));
        worst_hidden = std::min(worst_hidden, h.cosine);
        worst_embedding = std::min(worst_embedding, e.cosine);
        if (i == 0) {  // the runtime's own path to an embedding: the same vector
            const xs::EmbeddingResult r = runner->embed({ids});
            REQUIRE(r.embeddings.size() == 1);
            CHECK(xs::test::compare(r.embeddings[0], embedding).cosine >= 0.999999);
            CHECK(r.tokens == static_cast<int64_t>(ids.size()));
        }
    }
    MESSAGE(fixtures["model"].get<std::string>(), ": ", fixtures["cases"].size(), " cases, largest 1 - cosine: hidden ",
            gap(worst_hidden), ", embedding ", gap(worst_embedding));
}
