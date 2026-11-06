// Attention + KvCache contract for the cpu-ref backend: multi-step runs against an fp64 reference over the history.
#include <doctest/doctest.h>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend_test_util.h"
#include "reference_ops.h"
#include "util/error.h"

using namespace xs;
using namespace xs::graph;
using xs::test::Tensor64;
using xs::test::to_f64;

namespace {

constexpr double kAtol = 1e-5;
constexpr double kRtol = 1e-5;

struct AttentionCase {
    int64_t q_heads;
    int64_t kv_heads;
    int64_t head_dim;
    AttentionAttrs attrs;
};

Graph attention_graph(const AttentionCase& c) {
    Graph g;
    const Value q = g.input("q", DType::F32, {kT, c.q_heads, c.head_dim});
    const Value k = g.input("k", DType::F32, {kT, c.kv_heads, c.head_dim});
    const Value v = g.input("v", DType::F32, {kT, c.kv_heads, c.head_dim});
    const int slot = g.add_kv_slot(c.kv_heads, c.head_dim, c.attrs.sliding_window, "layer");
    const auto [k_all, v_all] = g.kv_cache(slot, k, v);
    g.output("out", g.attention(q, k_all, v_all, c.attrs));
    g.output("k_all", k_all);
    return g;
}

void append_rows(Tensor64& history, const Tensor64& step) {
    if (history.shape.empty()) {
        history = step;
        return;
    }
    history.shape[0] += step.shape[0];
    history.data.insert(history.data.end(), step.data.begin(), step.data.end());
}

test::ref::AttentionSpec spec_of(const AttentionAttrs& a) {
    return {a.scale, a.sliding_window, a.softcap, 0};
}

// Runs `steps` token counts through one session; checks every step against the reference over the full history.
// Returns the last step's output and reference inputs for planted-bug checks.
struct LastStep {
    HostTensor out;
    Tensor64 q, k_all, v_all;
};

LastStep check_steps(const AttentionCase& c, const std::vector<int64_t>& steps, uint32_t seed) {
    const auto backend = make_cpu_ref_backend();
    const auto session = backend->compile(attention_graph(c), test::MemoryWeights{}, {});
    test::TensorRng rng(seed);
    Tensor64 k_hist, v_hist;
    LastStep last;
    for (const int64_t t : steps) {
        CAPTURE(t);
        const HostTensor q = rng.uniform(DType::F32, {t, c.q_heads, c.head_dim}, 2.0f);
        const HostTensor k = rng.uniform(DType::F32, {t, c.kv_heads, c.head_dim}, 2.0f);
        const HostTensor v = rng.uniform(DType::F32, {t, c.kv_heads, c.head_dim});
        append_rows(k_hist, to_f64(k));
        append_rows(v_hist, to_f64(v));
        TensorMap out;
        session->run({{"q", q}, {"k", k}, {"v", v}}, out);
        CHECK(session->num_cached() == k_hist.shape[0]);
        CHECK(test::all_close(out.at("k_all"), k_hist, 0, 0));
        CHECK(test::all_close(out.at("out"), test::ref::attention(to_f64(q), k_hist, v_hist, spec_of(c.attrs)), kAtol,
                              kRtol));
        last = {out.at("out"), to_f64(q), k_hist, v_hist};
    }
    return last;
}

}  // namespace

TEST_SUITE("cpu-ref attention") {
    TEST_CASE("causal attention over multiple runs, GQA ratios 1, 4 and 8") {
        for (const auto& [hq, hkv] : {std::pair{2, 2}, std::pair{8, 2}, std::pair{8, 1}}) {
            CAPTURE(hq);
            CAPTURE(hkv);
            check_steps({hq, hkv, 16, {0.25f, 0, 0.0f}}, {5, 1, 3, 1}, 11);
        }
    }

    TEST_CASE("sliding window, T = 1 and T > 1") {
        check_steps({4, 1, 8, {0.3f, 3, 0.0f}}, {1, 1, 6, 1, 1, 4}, 12);
    }

    TEST_CASE("logit softcap with and without a window") {
        check_steps({4, 2, 8, {0.5f, 0, 2.0f}}, {4, 1, 2}, 13);
        check_steps({4, 2, 8, {0.5f, 4, 2.0f}}, {6, 1, 3}, 14);
    }

    TEST_CASE("head_dim 256 and 512") {
        check_steps({2, 1, 256, {1.0f / 16, 0, 0.0f}}, {3, 1}, 15);
        check_steps({2, 2, 512, {1.0f / 22.6f, 2, 0.0f}}, {3, 1}, 16);
    }

    TEST_CASE("planted bugs: off-by-one window and missing scale must not match") {
        const AttentionCase c{4, 2, 8, {0.4f, 3, 0.0f}};
        const LastStep last = check_steps(c, {5, 3}, 17);
        test::ref::AttentionSpec off_by_one = spec_of(c.attrs);
        off_by_one.window_slack = 1;
        CHECK_FALSE(
            test::all_close(last.out, test::ref::attention(last.q, last.k_all, last.v_all, off_by_one), kAtol, kRtol));
        test::ref::AttentionSpec unscaled = spec_of(c.attrs);
        unscaled.scale = 1.0;
        CHECK_FALSE(
            test::all_close(last.out, test::ref::attention(last.q, last.k_all, last.v_all, unscaled), kAtol, kRtol));
    }
}
