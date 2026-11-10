// Attention through the stateful KV cache against the fp64 reference: head dims 64/128/256/512, GQA, sliding window,
// softcap, every attention path (the per-row kernel, GEMMs, the fused matrix-engine kernel), over several steps of
// different T.
#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "device_test_support.h"
#include "quant/quant.h"
#include "reference_ops.h"
#include "util/error.h"

using namespace xs;
using namespace xs::graph;
using xs::test::Tensor64;
using xs::test::to_f64;
namespace dev = xs::test::device;

namespace {

const dev::Tolerance kAttention = dev::tolerance({1e-5, 1e-4}, {4e-3, 4e-3});
// int8 KV against the int8-rounded reference: a code the device rounds the other way at a .5 tie moves one key.
const dev::Tolerance kAttentionI8 = dev::tolerance({1e-4, 1e-4}, {4e-3, 4e-3});

struct AttentionCase {
    int64_t heads = 4;
    int64_t kv_heads = 4;
    int64_t head_dim = 64;
    int64_t window = 0;
    float softcap = 0.0f;
    float scale = 0.0f;  // 0 = 1/sqrt(head_dim)
    std::string impl = "auto";
    std::vector<int64_t> steps{5, 1, 3};
    bool kv_i8 = false;  // --kv-dtype i8
    bool causal = true;
};

std::string describe(const AttentionCase& c) {
    return str_cat("D", c.head_dim, " H", c.heads, "/", c.kv_heads, " w", c.window, " cap", c.softcap, " ", c.impl,
                   c.kv_i8 ? " kv-i8" : "", c.causal ? "" : " not causal");
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
    g.output("out", g.attention(q, k_all, v_all, AttentionAttrs{scale_of(c), c.window, c.softcap, c.causal}));
    return g;
}

void append_rows(Tensor64& all, const HostTensor& rows) {
    const Tensor64 r = to_f64(rows);
    if (all.shape.empty()) all.shape = r.shape, all.shape[0] = 0;
    all.data.insert(all.data.end(), r.data.begin(), r.data.end());
    all.shape[0] += r.shape[0];
}

HostTensor kv_i8_values(const HostTensor& x) {
    HostTensor r = x.clone();
    quant::round_trip_kv_i8(r.data<float>(), r.numel() / r.shape().back(), r.shape().back());
    return r;
}

// Runs every step of `c` through one session. Returns false at the first step whose output is not within `tol` of
// the reference computed with `reference_spec`, over int8 KV values when `c.kv_i8` and `rounded_reference`.
bool attention_matches(const AttentionCase& c, const test::ref::AttentionSpec& reference_spec, dev::Tolerance tol,
                       uint32_t seed, bool rounded_reference = true) {
    test::TensorRng rng(seed);
    SessionOptions options = dev::options();
    options.extra["attention"] = c.impl;
    if (c.kv_i8) options.kv_dtype = DType::I8;
    const bool rounded = c.kv_i8 && rounded_reference;
    const auto session = dev::backend().compile(attention_graph(c), test::MemoryWeights{}, options);
    Tensor64 k_all, v_all;
    bool ok = true;
    for (const int64_t t : c.steps) {
        const HostTensor q = rng.uniform(DType::F32, {t, c.heads, c.head_dim});
        const HostTensor k = rng.uniform(DType::F32, {t, c.kv_heads, c.head_dim});
        const HostTensor v = rng.uniform(DType::F32, {t, c.kv_heads, c.head_dim});
        append_rows(k_all, rounded ? kv_i8_values(k) : k);
        append_rows(v_all, rounded ? kv_i8_values(v) : v);
        TensorMap out;
        session->run({{"q", q}, {"k", k}, {"v", v}}, out);
        const Tensor64 expected = test::ref::attention(to_f64(q), k_all, v_all, reference_spec);
        ok = dev::close(str_cat("attention ", describe(c)), out.at("out"), expected, tol) && ok;
    }
    return ok;
}

bool attention_matches(const AttentionCase& c, uint32_t seed) {
    return attention_matches(c, {scale_of(c), c.window, c.softcap, 0, c.causal}, c.kv_i8 ? kAttentionI8 : kAttention,
                             seed);
}

}  // namespace

TEST_SUITE("device attention" * doctest::skip(!dev::enabled())) {
    TEST_CASE("causal attention with GQA for head dims 64, 128, 256 and 512, every lowering") {
        for (const int64_t head_dim : {64, 128, 256, 512}) {
            for (const auto& [heads, kv_heads] : {std::pair<int64_t, int64_t>{4, 4}, {8, 2}, {8, 1}}) {
                for (const std::string& impl : dev::attention_impls()) {
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
            for (const std::string& impl : dev::attention_impls()) {
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

    TEST_CASE("not causal with a window (a drafter's block): steps of up to the window, every lowering, int8 KV") {
        for (const int64_t head_dim : {64, 128}) {
            for (const std::string& impl : dev::attention_impls()) {
                for (const bool kv_i8 : {false, true}) {
                    AttentionCase c;
                    c.head_dim = head_dim;
                    c.heads = 8;
                    c.kv_heads = 2;
                    c.window = 16;
                    c.causal = false;
                    c.impl = impl;
                    c.kv_i8 = kv_i8;
                    c.steps = {16, 9, 16, 1, 8, 16};
                    CAPTURE(describe(c));
                    CHECK(attention_matches(c, static_cast<uint32_t>(900 + head_dim)));
                }
            }
        }
        AttentionCase c;
        c.window = 6;
        c.causal = false;
        c.steps = {6, 4, 5};
        CHECK(attention_matches(c, {scale_of(c), c.window, 0.0, 0, false}, kAttention, 950));
        CHECK_FALSE(attention_matches(c, {scale_of(c), c.window, 0.0, 1, false}, kAttention, 950));
        CHECK_FALSE(attention_matches(c, {scale_of(c), c.window, 0.0, 0, true}, kAttention, 950));
    }

    TEST_CASE("decode over a long cache, whole and windowed (split across work-groups on SYCL)") {
        for (const int64_t window : {0, 1100}) {
            for (const std::string& impl : dev::attention_impls()) {
                AttentionCase c;
                c.head_dim = 128;
                c.heads = 8;
                c.kv_heads = 2;
                c.window = window;
                c.impl = impl;
                c.steps = {2100, 1, 2};
                CAPTURE(describe(c));
                CHECK(attention_matches(c, static_cast<uint32_t>(400 + window)));
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
        for (const std::string& impl : dev::attention_impls()) {
            AttentionCase c;
            c.window = 3;
            c.impl = impl;
            c.steps = {8, 1};
            CAPTURE(impl);
            CHECK(attention_matches(c, {scale_of(c), c.window, 0.0, 0}, kAttention, 300));
            CHECK_FALSE(attention_matches(c, {scale_of(c), c.window, 0.0, 1}, kAttention, 300));
        }
    }

    TEST_CASE("int8 KV cache: head dims 64, 256 and 512, GQA, windows, long decode, both lowerings") {
        for (const std::string& impl : dev::attention_impls()) {
            for (const int64_t head_dim : {64, 256, 512}) {
                AttentionCase c;
                c.head_dim = head_dim;
                c.heads = 8;
                c.kv_heads = 2;
                c.impl = impl;
                c.kv_i8 = true;
                CAPTURE(describe(c));
                CHECK(attention_matches(c, static_cast<uint32_t>(500 + head_dim)));
                c.window = 16;
                c.steps = {37, 1, 1, 4};
                CAPTURE(describe(c));
                CHECK(attention_matches(c, static_cast<uint32_t>(600 + head_dim)));
            }
            for (const int64_t window : {0, 1100}) {
                AttentionCase c;
                c.head_dim = 128;
                c.heads = 8;
                c.kv_heads = 1;
                c.window = window;
                c.softcap = 30.0f;
                c.impl = impl;
                c.kv_i8 = true;
                c.steps = {2100, 1, 2};
                CAPTURE(describe(c));
                CHECK(attention_matches(c, static_cast<uint32_t>(700 + window)));
            }
        }
    }

    TEST_CASE("planted bug: int8 KV output must not match the reference over the unrounded keys and values") {
        if (dev::is_f16()) return;  // f16 rounding is as large as the int8 error: only f32 can tell them apart
        for (const std::string& impl : dev::attention_impls()) {
            AttentionCase c;
            c.impl = impl;
            c.kv_i8 = true;
            CAPTURE(impl);
            CHECK(attention_matches(c, {scale_of(c), c.window, 0.0, 0}, kAttentionI8, 800));
            CHECK_FALSE(attention_matches(c, {scale_of(c), c.window, 0.0, 0}, kAttentionI8, 800, false));
        }
    }
}
