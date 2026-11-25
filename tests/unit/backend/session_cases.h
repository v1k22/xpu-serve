#pragma once

// Backend-agnostic cases for sessions of several programs (Backend::compile over a list of graphs): state slots shared
// by label, KvRead, truncate inside a short step (SessionOptions::rewind_tokens) for KV, conv and recurrent state, and
// runs asked for some of their outputs.
// Each backend's test file runs them with its own tolerance.

#include <doctest/doctest.h>

#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "backend_test_util.h"
#include "recurrent_ops_cases.h"
#include "reference_ops.h"
#include "util/error.h"

namespace xs::test::session {

using graph::Graph;
using graph::kT;
using graph::Value;

struct Env {
    std::function<std::unique_ptr<Session>(std::span<const Graph* const>, const WeightStore&, int64_t rewind_tokens,
                                           int64_t sequences)>
        compile;
    double atol = 1e-5;
    double rtol = 1e-5;
};

// One layer of each kind of state: a causal conv, a gated delta rule and attention over a KV cache (within a sliding
// window, when given one), each fed from its own inputs.
struct HybridLayer {
    static constexpr int64_t kChannels = 24, kKernel = 4, kHk = 2, kHv = 4, kDk = 32, kDv = 16, kHkv = 2, kD = 32;
    MemoryWeights weights;
    Graph graph;

    // `segments`: the graph allows them (Graph::allow_segments). `dk`: the gated delta rule's key width (64 and 128 take
    // the GPU's lanes kernel); `d`: attention's head width (a TurboQuant cache takes multiples of 128).
    explicit HybridLayer(int64_t window = 0, bool segments = false, int64_t dk = kDk, int64_t d = kD) {
        TensorRng rng(41);
        if (segments) graph.allow_segments();
        weights.add("conv_w", rng.uniform(DType::BF16, {kChannels, kKernel}));
        const int conv = graph.add_conv_slot(kChannels, kKernel, "conv");
        graph.output("conv", graph.causal_conv(conv, graph.input("x", DType::F32, {kT, kChannels}),
                                               graph.weight(weights, "conv_w")));
        const int gdn = graph.add_recurrent_slot(kHv, dk, kDv, "gdn");
        graph.output("gdn", graph.gated_delta(gdn, graph.input("q", DType::F32, {kT, kHk, dk}),
                                              graph.input("k", DType::F32, {kT, kHk, dk}),
                                              graph.input("v", DType::F32, {kT, kHv, kDv}),
                                              graph.input("g", DType::F32, {kT, kHv}),
                                              graph.input("beta", DType::F32, {kT, kHv}), 1e-6f));
        const int kv = graph.add_kv_slot(kHkv, d, window, "attn");
        const auto [k_all, v_all] =
            graph.kv_cache(kv, graph.input("ak", DType::F32, {kT, kHkv, d}), graph.input("av", DType::F32, {kT, kHkv, d}));
        graph::AttentionAttrs attrs{0.2f};
        attrs.sliding_window = window;
        graph.output("attn", graph.attention(graph.input("aq", DType::F32, {kT, 2 * kHkv, d}), k_all, v_all, attrs));
        graph.validate();
    }

    // `tokens` rows of every input.
    static TensorMap inputs(TensorRng& rng, int64_t tokens, int64_t dk = kDk, int64_t d = kD) {
        TensorMap in;
        in["x"] = rng.uniform(DType::F32, {tokens, kChannels});
        in["q"] = rng.uniform(DType::F32, {tokens, kHk, dk});
        in["k"] = rng.uniform(DType::F32, {tokens, kHk, dk});
        in["v"] = rng.uniform(DType::F32, {tokens, kHv, kDv});
        HostTensor g = rng.uniform(DType::F32, {tokens, kHv}, 0.5f), beta = rng.uniform(DType::F32, {tokens, kHv}, 0.5f);
        for (int64_t i = 0; i < g.numel(); ++i) {
            g.data<float>()[i] -= 0.6f;     // log decay in [-1.1, -0.1]
            beta.data<float>()[i] += 0.5f;  // beta in [0, 1]
        }
        in["g"] = std::move(g);
        in["beta"] = std::move(beta);
        in["ak"] = rng.uniform(DType::F32, {tokens, kHkv, d});
        in["av"] = rng.uniform(DType::F32, {tokens, kHkv, d});
        in["aq"] = rng.uniform(DType::F32, {tokens, 2 * kHkv, d});
        return in;
    }

    // The first `n` rows of every input.
    static TensorMap head(const TensorMap& in, int64_t n) {
        TensorMap out;
        for (const auto& [name, t] : in) {
            const Tensor64 all = to_f64(t);
            Dims shape = t.shape();
            shape[0] = n;
            const auto row = static_cast<size_t>(all.numel() / t.dim(0));
            out[name] = HostTensor::from_vector(
                DType::F32, shape, std::vector<float>(all.data.begin(), all.data.begin() + static_cast<long>(row * n)));
        }
        return out;
    }
};

inline bool outputs_close(const TensorMap& a, const TensorMap& b, const Env& env) {
    bool ok = true;
    for (const char* name : {"conv", "gdn", "attn"}) {
        CAPTURE(name);
        const bool close = all_close(a.at(name), b.at(name), env.atol, env.rtol);
        CHECK(close);
        ok = ok && close;
    }
    return ok;
}

// A run asked for some outputs returns just those, with a full run's values, and advances every state slot all the
// same (what follows agrees with a session that asked for everything); a name the program lacks is refused.
inline void requested_outputs(const Env& env) {
    const HybridLayer layer;
    const Graph* programs[] = {&layer.graph};
    TensorRng rng(43);
    const TensorMap first = HybridLayer::inputs(rng, 3), second = HybridLayer::inputs(rng, 2);
    const auto full = env.compile(programs, layer.weights, 0, 1);
    const auto some = env.compile(programs, layer.weights, 0, 1);
    TensorMap want, got;
    full->run(0, first, want);
    const std::string only[] = {"attn"};
    some->run(0, first, got, only);
    CHECK(got.size() == 1);
    CHECK(all_close(got.at("attn"), want.at("attn"), env.atol, env.rtol));
    full->run(0, second, want);
    some->run(0, second, got);
    CHECK(got.size() == 3);
    outputs_close(got, want, env);
    const std::string missing[] = {"attn", "nope"};
    CHECK_THROWS_AS(some->run(0, second, got, missing), Error);
    CHECK(some->num_cached() == 5);  // the refusal ran nothing

    // An output nothing else reads (a row's argmax beside its scores) is not computed unless asked for, and equals
    // the full run's when it is.
    MemoryWeights w;
    w.add("w", rng.uniform(DType::F32, {40, 64}));
    Graph g;
    const Value scores = g.matmul(g.input("x", DType::F32, {kT, 64}), g.weight(w, "w"));
    g.output("scores", scores);
    g.output("argmax", g.reshape(g.topk(scores, 1).second, {kT}));
    const Graph* heads[] = {&g};
    const auto session = env.compile(heads, w, 0, 1);
    const TensorMap in{{"x", rng.uniform(DType::F32, {4, 64})}};
    TensorMap all, one;
    session->run(0, in, all);
    const std::string top[] = {"argmax"};
    session->run(0, in, one, top);
    CHECK(one.size() == 1);
    CHECK(to_ints(one.at("argmax")) == to_ints(all.at("argmax")));
    const std::string values[] = {"scores"};
    one.clear();
    session->run(0, in, one, values);
    CHECK(one.size() == 1);
    CHECK(all_close(one.at("scores"), all.at("scores"), 0, 0));
}

// After a step of up to rewind_tokens tokens, truncating to any count inside it and continuing gives what a session
// that never ran the dropped tokens gives; counts it cannot rebuild are refused and change nothing.
inline void truncate_inside_step(const Env& env) {
    const HybridLayer layer;
    const Graph* programs[] = {&layer.graph};
    TensorRng rng(42);
    const TensorMap prefix = HybridLayer::inputs(rng, 5), step = HybridLayer::inputs(rng, 3),
                    next = HybridLayer::inputs(rng, 2);
    for (const int64_t keep : {0, 1, 2, 3}) {
        CAPTURE(keep);
        const auto spec = env.compile(programs, layer.weights, 4, 1);
        const auto plain = env.compile(programs, layer.weights, 0, 1);
        TensorMap out, want;
        spec->run(prefix, out);
        plain->run(prefix, want);
        CHECK_THROWS_AS(spec->truncate(4), xs::Error);  // the prefix (5 tokens) is longer than rewind_tokens
        spec->run(step, out);
        spec->truncate(5 + keep);
        CHECK(spec->num_cached() == 5 + keep);
        if (keep > 0) plain->run(HybridLayer::head(step, keep), want);
        spec->run(next, out);
        plain->run(next, want);
        outputs_close(out, want, env);
        // The last step started at 5 + keep: below that is refused, and the refusal leaves every slot as it was.
        CHECK_THROWS_AS(spec->truncate(4 + keep), xs::Error);
        CHECK(spec->num_cached() == 7 + keep);
        const TensorMap after = HybridLayer::inputs(rng, 1);
        spec->run(after, out);
        plain->run(after, want);
        outputs_close(out, want, env);
    }
    // Without rewind_tokens a step is kept or dropped whole: no count inside it.
    const auto plain = env.compile(programs, layer.weights, 0, 1);
    TensorMap out;
    plain->run(prefix, out);
    plain->run(step, out);
    CHECK_THROWS_AS(plain->truncate(6), xs::Error);
    CHECK(plain->num_cached() == 8);
    plain->truncate(0);
    CHECK(plain->num_cached() == 0);
}

// The last step can be undone whole, whatever its size and without rewind_tokens: what follows is what a session
// that never ran it gives. Once undone, the step before it cannot be (its state is gone).
inline void undo_last_step(const Env& env) {
    const HybridLayer layer;
    const Graph* programs[] = {&layer.graph};
    TensorRng rng(43);
    const TensorMap prefix = HybridLayer::inputs(rng, 5), first = HybridLayer::inputs(rng, 6),
                    second = HybridLayer::inputs(rng, 3), third = HybridLayer::inputs(rng, 4);
    const auto session = env.compile(programs, layer.weights, 0, 1);
    const auto plain = env.compile(programs, layer.weights, 0, 1);
    TensorMap out, want;
    session->run(prefix, out);
    plain->run(prefix, want);
    // Three tails after one prefix, each read from the prefix's state.
    for (const TensorMap* tail : {&first, &second, &third}) {
        session->run(*tail, out);
        const auto fresh = env.compile(programs, layer.weights, 0, 1);
        TensorMap prefix_out;
        fresh->run(prefix, prefix_out);
        fresh->run(*tail, want);
        outputs_close(out, want, env);
        session->truncate(5);
        CHECK(session->num_cached() == 5);
        session->truncate(5);  // already there: nothing to do
    }
    CHECK_THROWS_AS(session->truncate(3), xs::Error);  // inside the prefix
    CHECK(session->num_cached() == 5);
    // Two steps on, only the last can go.
    session->run(first, out);
    session->run(second, out);
    CHECK_THROWS_AS(session->truncate(5), xs::Error);
    session->truncate(11);
    session->run(third, out);
    plain->run(first, want);
    plain->run(third, want);
    outputs_close(out, want, env);
}

// A KV slot with a sliding window beside conv and recurrent state, which rule out cutting the sequence back before
// its last step: a backend may then keep only the window's keys. Over many steps, far past a few windows, with
// steps undone whole and cut inside, attention gives what it gives over every token kept.
inline void windowed_slot(const Env& env) {
    const int64_t window = 6;
    const HybridLayer layer(window);
    const Graph* programs[] = {&layer.graph};
    TensorRng rng(45);
    const auto session = env.compile(programs, layer.weights, 4, 1);
    ref::AttentionSpec spec;
    spec.scale = 0.2;
    spec.sliding_window = window;
    std::vector<HostTensor> ks, vs;
    struct Step {
        int64_t tokens;
        int64_t keep;  // of the step, after it ran: all, none (undone whole) or a count inside it
    };
    int64_t cached = 0;
    for (const Step& s : {Step{5, 5}, Step{1, 1},  Step{1, 1},  Step{9, 9}, Step{3, 0}, Step{3, 3}, Step{1, 1},
                          Step{4, 2}, Step{1, 1},  Step{20, 20}, Step{1, 1}, Step{2, 1}, Step{7, 7}, Step{1, 0},
                          Step{1, 1}, Step{30, 30}, Step{4, 3},  Step{1, 1}, Step{1, 1}, Step{12, 0}, Step{2, 2}}) {
        CAPTURE(cached);
        CAPTURE(s.tokens);
        const TensorMap in = HybridLayer::inputs(rng, s.tokens);
        TensorMap out;
        session->run(in, out);
        ks.push_back(in.at("ak").clone());
        vs.push_back(in.at("av").clone());
        CHECK(all_close(out.at("attn"),
                        ref::attention(to_f64(in.at("aq")), recurrent::concat_rows(ks), recurrent::concat_rows(vs), spec),
                        env.atol, env.rtol));
        if (s.keep < s.tokens) {
            session->truncate(cached + s.keep);
            ks.pop_back();
            vs.pop_back();
            if (s.keep > 0) {
                const TensorMap kept = HybridLayer::head(in, s.keep);
                ks.push_back(kept.at("ak").clone());
                vs.push_back(kept.at("av").clone());
            }
        }
        cached += s.keep;
        CHECK(session->num_cached() == cached);
    }
    CHECK_THROWS_AS(session->truncate(cached - 3), xs::Error);  // before the last step
    session->truncate(0);
    const TensorMap in = HybridLayer::inputs(rng, 9);
    TensorMap out;
    session->run(in, out);
    CHECK(all_close(out.at("attn"), ref::attention(to_f64(in.at("aq")), to_f64(in.at("ak")), to_f64(in.at("av")), spec),
                    env.atol, env.rtol));
}

// Programs that name the same KV slot share it: two writers extend one sequence, a KvRead reader attends to all of it
// without appending, and a slot declared with two shapes is refused.
inline void shared_kv_slot(const Env& env) {
    constexpr int64_t kHkv = 2, kD = 32, kHq = 4;
    const auto writer = [&](int64_t head_dim) {
        auto g = std::make_unique<Graph>();
        const int slot = g->add_kv_slot(kHkv, head_dim, 0, "layer");
        const auto [k_all, v_all] = g->kv_cache(slot, g->input("k", DType::F32, {kT, kHkv, head_dim}),
                                                g->input("v", DType::F32, {kT, kHkv, head_dim}));
        g->output("out", g->attention(g->input("q", DType::F32, {kT, kHq, head_dim}), k_all, v_all,
                                      graph::AttentionAttrs{0.25f}));
        return g;
    };
    const auto reader = [&](int64_t head_dim) {
        auto g = std::make_unique<Graph>();
        const int slot = g->add_kv_slot(kHkv, head_dim, 0, "layer");
        const auto [k_all, v_all] = g->kv_read(slot);
        g->output("out", g->attention(g->input("q", DType::F32, {kT, kHq, head_dim}), k_all, v_all,
                                      graph::AttentionAttrs{0.25f}));
        return g;
    };
    const auto first = writer(kD), second = writer(kD), read = reader(kD);
    const Graph* programs[] = {first.get(), second.get(), read.get()};
    const auto session = env.compile(programs, MemoryWeights{}, 0, 1);
    TensorRng rng(43);
    std::vector<HostTensor> ks, vs;
    const auto write = [&](size_t program, int64_t tokens) {
        ks.push_back(rng.uniform(DType::F32, {tokens, kHkv, kD}));
        vs.push_back(rng.uniform(DType::F32, {tokens, kHkv, kD}));
        TensorMap out;
        session->run(program, {{"k", ks.back()}, {"v", vs.back()}, {"q", rng.uniform(DType::F32, {tokens, kHq, kD})}},
                     out);
    };
    const auto rows = [](const std::vector<HostTensor>& parts, int64_t n) {
        Tensor64 all;
        for (const HostTensor& p : parts) {
            const Tensor64 t = to_f64(p);
            if (all.shape.empty()) {
                all.shape = t.shape;
                all.shape[0] = 0;
            }
            all.shape[0] += t.shape[0];
            all.data.insert(all.data.end(), t.data.begin(), t.data.end());
        }
        all.data.resize(static_cast<size_t>(all.numel() / all.shape[0] * n));
        all.shape[0] = n;
        return all;
    };
    const auto check_reader = [&](int64_t len) {
        CAPTURE(len);
        const HostTensor q = rng.uniform(DType::F32, {1, kHq, kD});
        TensorMap out;
        session->run(2, {{"q", q}}, out);
        ref::AttentionSpec spec;
        spec.scale = 0.25;
        CHECK(all_close(out.at("out"), ref::attention(to_f64(q), rows(ks, len), rows(vs, len), spec), env.atol,
                        env.rtol));
        CHECK(session->num_cached(2) == 0);  // a reader writes nothing
    };
    write(0, 3);
    write(1, 2);
    CHECK(session->num_cached(0) == 5);
    CHECK(session->num_cached(1) == 5);
    check_reader(5);
    session->truncate(1, 4);
    CHECK(session->num_cached(0) == 4);
    check_reader(4);
    session->reset();
    ks.clear();
    vs.clear();
    CHECK(session->num_cached(0) == 0);
    write(1, 2);
    check_reader(2);

    const auto wide = reader(2 * kD);
    const Graph* mismatched[] = {first.get(), wide.get()};
    CHECK_THROWS_AS(env.compile(mismatched, MemoryWeights{}, 0, 1), xs::Error);
}

// Three sequences of one session, each with its conv, recurrent and KV state, run in turns: each gives what a session
// of its own gives over the same steps; reset and truncate touch the current one alone.
inline void sequences(const Env& env) {
    const HybridLayer layer(6);
    const Graph* programs[] = {&layer.graph};
    const auto shared = env.compile(programs, layer.weights, 0, 3);
    TensorRng rng(9);
    std::vector<std::unique_ptr<Session>> alone;
    for (int s = 0; s < 3; ++s) alone.push_back(env.compile(programs, layer.weights, 0, 1));
    const std::vector<std::pair<int, int64_t>> turns = {{0, 5}, {1, 3}, {0, 1}, {2, 7}, {1, 1}, {0, 2}, {2, 1}, {1, 4}};
    for (const auto& [s, tokens] : turns) {
        CAPTURE(s);
        const TensorMap in = HybridLayer::inputs(rng, tokens);
        TensorMap got, want;
        shared->use(s);
        shared->run(in, got);
        alone[static_cast<size_t>(s)]->run(in, want);
        CHECK(outputs_close(got, want, env));
        CHECK(shared->num_cached() == alone[static_cast<size_t>(s)]->num_cached());
    }
    // A reset of sequence 1 leaves 0 and 2 as they were.
    shared->use(1);
    shared->reset();
    CHECK(shared->num_cached() == 0);
    shared->use(2);
    CHECK(shared->num_cached() == 8);
    const TensorMap in = HybridLayer::inputs(rng, 2);
    TensorMap got, want;
    shared->run(in, got);
    alone[2]->run(in, want);
    CHECK(outputs_close(got, want, env));
    CHECK_THROWS_AS(shared->use(3), xs::Error);
}

// Steps of several sequences at once (Session::run_segments), between steps of one: each segment gives what a session
// of its own gives over the same steps, its rows in the segments' order; the current sequence stays current; a step
// that fails leaves every sequence as it was; a program that does not allow segments, and a sequence twice, are
// refused. `d`: attention's head width (128 for a TurboQuant cache).
inline void segments(const Env& env, int64_t d = HybridLayer::kD) {
    const HybridLayer layer(6, true, 64, d);
    const Graph* programs[] = {&layer.graph};
    const auto shared = env.compile(programs, layer.weights, 0, 3);
    TensorRng rng(13);
    std::vector<std::unique_ptr<Session>> alone;
    for (int s = 0; s < 3; ++s) alone.push_back(env.compile(programs, layer.weights, 0, 1));
    // One sequence's step; `segments` empty: several, in that order.
    const auto step = [&](std::vector<Segment> segs) {
        CAPTURE(segs.size());
        std::vector<TensorMap> parts;
        TensorMap want_all;
        for (const Segment& s : segs) {
            parts.push_back(HybridLayer::inputs(rng, s.tokens, 64, d));
            TensorMap want;
            alone[static_cast<size_t>(s.sequence)]->run(parts.back(), want);
            for (const char* name : {"conv", "gdn", "attn"}) {
                want_all[name] = want_all.count(name) == 0 ? want.at(name)
                                                           : concat_rows(std::vector{want_all.at(name), want.at(name)});
            }
        }
        TensorMap in, got;
        for (const auto& [name, t] : parts[0]) {
            std::vector<HostTensor> rows;
            for (const TensorMap& p : parts) rows.push_back(p.at(name));
            in[name] = concat_rows(rows);
        }
        if (segs.size() == 1) {
            shared->use(segs[0].sequence);
            shared->run(in, got);
        } else {
            const int64_t current = shared->current_sequence();
            shared->run_segments(0, segs, in, got, {});
            CHECK(shared->current_sequence() == current);
        }
        CHECK(outputs_close(got, want_all, env));
        for (const Segment& s : segs) {
            shared->use(s.sequence);
            CHECK(shared->num_cached() == alone[static_cast<size_t>(s.sequence)]->num_cached());
        }
    };
    step({{0, 5}});
    step({{1, 3}});
    step({{2, 9}});
    shared->use(1);
    step({{0, 1}, {2, 1}, {1, 1}});
    step({{2, 1}, {0, 2}});
    step({{1, 4}});
    for (int k = 0; k < 4; ++k) step({{1, 1}, {0, 1}, {2, 1}});

    // A step whose second segment fails: the first one's sequence is as it was, the next step agrees.
    shared->use(0);
    const TensorMap in = HybridLayer::inputs(rng, 2, 64, d);
    TensorMap got;
    const Segment bad[] = {{2, 1}, {7, 1}};
    CHECK_THROWS_AS(shared->run_segments(0, bad, in, got, {}), xs::Error);
    CHECK(shared->current_sequence() == 0);
    step({{2, 1}, {1, 1}});
    const Segment twice[] = {{1, 1}, {1, 1}};
    CHECK_THROWS_AS(shared->run_segments(0, twice, in, got, {}), xs::Error);

    const HybridLayer plain(6, false, 64, d);
    const Graph* plain_programs[] = {&plain.graph};
    const auto refused = env.compile(plain_programs, plain.weights, 0, 2);
    const Segment two[] = {{0, 1}, {1, 1}};
    CHECK_THROWS_AS(refused->run_segments(0, two, in, got, {}), xs::Error);
}

// Speculative rounds of several sequences at once: a step of segments of up to rewind_tokens tokens each, then each
// sequence cut back inside its own segment (truncate with it selected), gives each sequence what a session of its own
// gives over the same steps and cuts; a cut that keeps nothing of the step and one that keeps all of it included.
inline void segments_truncate(const Env& env, int64_t d = HybridLayer::kD) {
    const HybridLayer layer(6, true, 64, d);
    const Graph* programs[] = {&layer.graph};
    const auto shared = env.compile(programs, layer.weights, 4, 3);
    std::vector<std::unique_ptr<Session>> alone;
    for (int s = 0; s < 3; ++s) alone.push_back(env.compile(programs, layer.weights, 4, 1));
    TensorRng rng(29);
    for (int s = 0; s < 3; ++s) {
        const TensorMap in = HybridLayer::inputs(rng, 2 + 3 * s, 64, d);
        TensorMap got, want;
        shared->use(s);
        shared->run(in, got);
        alone[static_cast<size_t>(s)]->run(in, want);
    }
    // One step of `segs`, then each sequence keeps the first `keep` tokens of its segment.
    const auto round = [&](std::vector<Segment> segs, std::vector<int64_t> keep) {
        CAPTURE(segs.size());
        std::vector<TensorMap> parts;
        for (const Segment& s : segs) parts.push_back(HybridLayer::inputs(rng, s.tokens, 64, d));
        TensorMap in, got;
        for (const auto& [name, t] : parts[0]) {
            std::vector<HostTensor> rows;
            for (const TensorMap& p : parts) rows.push_back(p.at(name));
            in[name] = concat_rows(rows);
        }
        shared->run_segments(0, segs, in, got, {});
        int64_t row = 0;
        for (size_t i = 0; i < segs.size(); ++i) {
            CAPTURE(i);
            Session& own = *alone[static_cast<size_t>(segs[i].sequence)];
            TensorMap want, mine;
            own.run(parts[i], want);
            for (const char* name : {"conv", "gdn", "attn"}) mine[name] = rows_of(got.at(name), row, segs[i].tokens);
            CHECK(outputs_close(mine, want, env));
            row += segs[i].tokens;
            shared->use(segs[i].sequence);
            const int64_t start = shared->num_cached() - segs[i].tokens;
            shared->truncate(start + keep[i]);
            own.truncate(start + keep[i]);
            CHECK(shared->num_cached() == own.num_cached());
        }
    };
    round({{0, 3}, {1, 1}, {2, 4}}, {1, 1, 2});
    round({{2, 2}, {0, 4}, {1, 3}}, {2, 0, 3});
    round({{1, 4}, {2, 1}}, {4, 1});
    for (int k = 0; k < 3; ++k) round({{1, 1}, {0, 1}, {2, 1}}, {1, 1, 1});
}

}  // namespace xs::test::session
