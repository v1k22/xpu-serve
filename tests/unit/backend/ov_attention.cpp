// Attention through the stateful KV cache against the fp64 reference: head dims 64/256/512, GQA, sliding window,
// softcap, both lowerings (ScaledDotProductAttention and MatMul/Softmax), over several steps of different T.
#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "ov_test_support.h"
#include "reference_ops.h"
#include "util/error.h"

using namespace xs;
using namespace xs::graph;
using xs::test::Tensor64;
using xs::test::to_f64;
namespace ovt = xs::test::ov;

namespace {

const ovt::Tolerance kAttention = ovt::tolerance({1e-5, 1e-4}, {4e-3, 4e-3});

struct AttentionCase {
    int64_t heads = 4;
    int64_t kv_heads = 4;
    int64_t head_dim = 64;
    int64_t window = 0;
    float softcap = 0.0f;
    float scale = 0.0f;  // 0 = 1/sqrt(head_dim)
    std::string impl = "auto";
    std::vector<int64_t> steps{5, 1, 3};
};

std::string describe(const AttentionCase& c) {
    return str_cat("D", c.head_dim, " H", c.heads, "/", c.kv_heads, " w", c.window, " cap", c.softcap, " ", c.impl);
}

float scale_of(const AttentionCase& c) {
    return c.scale > 0 ? c.scale : 1.0f / std::sqrt(static_cast<float>(c.head_dim));
}

Graph attention_graph(const AttentionCase& c) {
    Graph g;
    const int slot = g.add_kv_slot(c.kv_heads, c.head_dim, c.window, "layer0");
    const Value q = g.input("q", DType::F32, {kT, c.heads, c.head_dim});
    const auto [k_all, v_all] = g.kv_cache(slot, g.input("k", DType::F32, {kT, c.kv_heads, c.head_dim}),
                                           g.input("v", DType::F32, {kT, c.kv_heads, c.head_dim}));
    g.output("out", g.attention(q, k_all, v_all, AttentionAttrs{scale_of(c), c.window, c.softcap}));
    return g;
}

void append_rows(Tensor64& all, const HostTensor& rows) {
    const Tensor64 r = to_f64(rows);
    if (all.shape.empty()) all.shape = r.shape, all.shape[0] = 0;
    all.data.insert(all.data.end(), r.data.begin(), r.data.end());
    all.shape[0] += r.shape[0];
}

// Runs every step of `c` through one session. Returns false at the first step whose output is not within `tol` of
// the reference computed with `reference_spec`.
bool attention_matches(const AttentionCase& c, const test::ref::AttentionSpec& reference_spec, ovt::Tolerance tol,
                       uint32_t seed) {
    test::TensorRng rng(seed);
    SessionOptions options = ovt::options();
    options.extra["attention"] = c.impl;
    const auto session = ovt::backend().compile(attention_graph(c), test::MemoryWeights{}, options);
    Tensor64 k_all, v_all;
    bool ok = true;
    for (const int64_t t : c.steps) {
        const HostTensor q = rng.uniform(DType::F32, {t, c.heads, c.head_dim});
        const HostTensor k = rng.uniform(DType::F32, {t, c.kv_heads, c.head_dim});
        const HostTensor v = rng.uniform(DType::F32, {t, c.kv_heads, c.head_dim});
        append_rows(k_all, k);
        append_rows(v_all, v);
        TensorMap out;
        session->run({{"q", q}, {"k", k}, {"v", v}}, out);
        const Tensor64 expected = test::ref::attention(to_f64(q), k_all, v_all, reference_spec);
        ok = ovt::close(str_cat("attention ", describe(c)), out.at("out"), expected, tol) && ok;
    }
    return ok;
}

bool attention_matches(const AttentionCase& c, uint32_t seed) {
    return attention_matches(c, {scale_of(c), c.window, c.softcap, 0}, kAttention, seed);
}

}  // namespace

TEST_SUITE("openvino attention") {
    TEST_CASE("causal attention with GQA for head dims 64, 256 and 512, both lowerings") {
        for (const int64_t head_dim : {64, 256, 512}) {
            for (const auto& [heads, kv_heads] : {std::pair<int64_t, int64_t>{4, 4}, {8, 2}, {8, 1}}) {
                for (const char* impl : {"sdpa", "matmul"}) {
                    AttentionCase c;
                    c.head_dim = head_dim;
                    c.heads = heads;
                    c.kv_heads = kv_heads;
                    c.impl = impl;
                    CAPTURE(describe(c));
                    CHECK(attention_matches(c, static_cast<uint32_t>(head_dim + heads * 10 + kv_heads)));
                }
            }
        }
    }

    TEST_CASE("sliding window over a prefill longer than the window, then decode steps") {
        for (const int64_t head_dim : {64, 256, 512}) {
            for (const char* impl : {"sdpa", "matmul"}) {
                AttentionCase c;
                c.head_dim = head_dim;
                c.heads = 8;
                c.kv_heads = 2;
                c.window = 16;
                c.impl = impl;
                c.steps = {37, 1, 1, 4};
                CAPTURE(describe(c));
                CHECK(attention_matches(c, static_cast<uint32_t>(100 + head_dim)));
            }
        }
    }

    TEST_CASE("scale 1 (q/k-normed models) and logit softcap with and without a window") {
        for (const int64_t window : {0, 4}) {
            AttentionCase c;
            c.head_dim = 256;
            c.heads = 4;
            c.kv_heads = 2;
            c.window = window;
            c.softcap = 20.0f;
            c.scale = 1.0f;
            CAPTURE(describe(c));
            CHECK(attention_matches(c, static_cast<uint32_t>(200 + window)));
        }
        AttentionCase no_cap;
        no_cap.head_dim = 512;
        no_cap.scale = 1.0f;
        no_cap.window = 3;
        CHECK(attention_matches(no_cap, 210));
    }

    TEST_CASE("planted bug: a reference whose window is one key wider must not match") {
        for (const char* impl : {"sdpa", "matmul"}) {
            AttentionCase c;
            c.window = 3;
            c.impl = impl;
            c.steps = {8, 1};
            CAPTURE(impl);
            CHECK(attention_matches(c, {scale_of(c), c.window, 0.0, 0}, kAttention, 300));
            CHECK_FALSE(attention_matches(c, {scale_of(c), c.window, 0.0, 1}, kAttention, 300));
        }
    }

    TEST_CASE("sdpa cannot express a softcap and says so") {
        AttentionCase c;
        c.softcap = 30.0f;
        SessionOptions options = ovt::options();
        options.extra["attention"] = "sdpa";
        CHECK_THROWS_WITH_AS(ovt::backend().compile(attention_graph(c), test::MemoryWeights{}, options),
                             doctest::Contains("softcap"), Error);
    }
}
