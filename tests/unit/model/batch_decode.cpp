// Each language model's batch graph (Model::batch_graph) on tiny checkpoints: decode steps of three sequences at once,
// as segments of one row each, give every sequence the logits of its own step graph run alone, on cpu-ref (whose
// segmented steps are the definition, Session::run_segments) and on the GPU when one is given (whose are its own).
// With a drafter, speculative rounds of three sequences verified as one step give each sequence its own rounds.
#include <doctest/doctest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend/device_test_support.h"
#include "convert/import_test_util.h"
#include "convert/tiny_gguf.h"
#include "convert/tiny_inkling2_gguf.h"
#include "engine/spec/drafter.h"
#include "models/model.h"
#include "step_logits.h"
#include "tiny_checkpoints.h"

using namespace xs;
namespace fs = std::filesystem;
namespace dev = xs::test::device;

namespace {

int32_t argmax(const HostTensor& logits, int64_t row) {
    const int64_t vocab = logits.dim(logits.rank() - 1);
    const float* r = logits.data<float>() + row * vocab;
    return static_cast<int32_t>(std::max_element(r, r + vocab) - r);
}

// Three prompts, each prefilled on a sequence of its own with the step graph, then decoded together with the batch
// graph, in a segment order that turns every step; against a session of one sequence a prompt, running the step graph
// alone. `tolerance`: the largest gap allowed, over the largest logit.
void check_batch(Backend& backend, const Model& model, SessionOptions options, double tolerance) {
    REQUIRE(model.batch_graph() != nullptr);
    CHECK(model.batch_graph()->segmented());
    const graph::Graph* programs[] = {&model.graph(), model.batch_graph()};
    options.sequences = 3;
    const auto session = backend.compile(programs, model.weights(), options);
    options.sequences = 1;
    const std::vector<std::vector<int32_t>> prompts = {{5, 77, 26, 3, 40}, {12, 8}, {20, 1, 2, 3, 4, 5, 6, 7, 9, 11}};
    std::vector<std::unique_ptr<Session>> alone;
    std::vector<int32_t> next;
    std::vector<int64_t> pos;
    for (size_t s = 0; s < prompts.size(); ++s) {
        alone.push_back(backend.compile(model.graph(), model.weights(), options));
        TensorMap in, out, want;
        model.prepare_inputs(prompts[s], 0, {}, in);
        session->use(static_cast<int64_t>(s));
        session->run(in, out);
        alone.back()->run(in, want);
        CHECK(xs::test::relative_gap({out.at("logits").to_f32()}, {want.at("logits").to_f32()}) <= tolerance);
        next.push_back(argmax(want.at("logits"), 0));
        pos.push_back(static_cast<int64_t>(prompts[s].size()));
    }
    session->use(0);
    double worst = 0;
    for (int step = 0; step < 5; ++step) {
        std::vector<Segment> segments;
        std::map<std::string, std::vector<HostTensor>> parts;
        for (int k = 0; k < 3; ++k) {
            const auto s = static_cast<size_t>((step + k) % 3);
            segments.push_back({static_cast<int64_t>(s), 1});
            TensorMap in;
            const int32_t token[] = {next[s]};
            model.prepare_inputs(token, pos[s], {}, in);
            for (auto& [name, t] : in) parts[name].push_back(t);
        }
        TensorMap in, out;
        for (auto& [name, list] : parts) in[name] = concat_rows(list);
        session->run_segments(1, segments, in, out, {});
        for (size_t r = 0; r < segments.size(); ++r) {
            const auto s = static_cast<size_t>(segments[r].sequence);
            TensorMap one, want;
            const int32_t token[] = {next[s]};
            model.prepare_inputs(token, pos[s], {}, one);
            alone[s]->run(one, want);
            const double gap = xs::test::relative_gap({rows_of(out.at("logits"), static_cast<int64_t>(r), 1).to_f32()},
                                                      {want.at("logits").to_f32()});
            worst = std::max(worst, gap);
            CHECK(out.at("argmax").data<int32_t>()[r] == argmax(out.at("logits"), static_cast<int64_t>(r)));
            next[s] = argmax(want.at("logits"), 0);
            ++pos[s];
        }
    }
    MESSAGE(model.info().arch, ": batched against alone, largest gap ", worst, " of the largest logit");
    CHECK(worst <= tolerance);
}

void check_both(const Model& model) {
    SessionOptions f32;
    f32.precision = DType::F32;
    check_batch(*make_cpu_ref_backend(), model, f32, 1e-6);
    if (dev::enabled()) check_batch(dev::backend(), model, dev::options(), dev::is_f16() ? 2e-2 : 1e-4);
}

LoadOptions batched() {
    LoadOptions o;
    o.batch = true;
    return o;
}

// The rows [row, row + n) of every output.
TensorMap rows_of_all(const TensorMap& outputs, int64_t row, int64_t n) {
    TensorMap out;
    for (const auto& [name, t] : outputs) out[name] = rows_of(t, row, n);
    return out;
}

// A sequence's session and drafter: one of three in a shared session, or a session of its own.
struct Speculating {
    Session* session;
    int64_t sequence;
    std::unique_ptr<spec::Drafter> drafter;
    int32_t next = 0;
    int64_t pos = 0;
};

// Three prompts prefilled with the prefill program, each on a sequence of its own, then speculative rounds of all
// three verified as one step (segments of 1 to 4 rows: the pending token and its drafts, as many as `drafts` turns
// to); each sequence keeps some of its segment, every count from 1 to all in turn (the session and the drafter cut
// back inside the segment, as a round that accepted that many), and goes on from the target's token after the last
// row kept. Against a session of one sequence a prompt, running the same rounds alone: the same drafts and verify
// outputs. (A tiny drafter's drafts are rarely right, so the counts kept are the test's, not the drafts'.)
void check_rounds(Backend& backend, const Model& model, SessionOptions options, double tolerance) {
    const Speculation* spec = model.speculation();
    REQUIRE(spec != nullptr);
    CHECK(spec->verify_graph().segmented());
    const graph::Graph* programs[] = {&model.graph(), &spec->prefill_graph(), &spec->verify_graph(),
                                      &spec->draft_graph()};
    options.rewind_tokens = 4;
    options.sequences = 3;
    const auto shared = backend.compile(programs, model.weights(), options);
    options.sequences = 1;
    const std::vector<std::vector<int32_t>> prompts = {{5, 77, 26, 3, 40}, {12, 8}, {20, 1, 2, 3, 4, 5, 6, 7, 9, 11}};
    std::vector<std::unique_ptr<Session>> own;
    std::vector<Speculating> mine, alone;
    for (size_t s = 0; s < prompts.size(); ++s) {
        own.push_back(backend.compile(programs, model.weights(), options));
        shared->use(static_cast<int64_t>(s));
        mine.push_back({shared.get(), static_cast<int64_t>(s), spec::make_drafter(*spec, *shared)});
        alone.push_back({own.back().get(), 0, spec::make_drafter(*spec, *own.back())});
        const auto n = static_cast<int64_t>(prompts[s].size());
        TensorMap in, out, want;
        model.prepare_inputs(prompts[s], 0, {}, in);
        shared->run(spec::kPrefill, in, out);
        own.back()->run(spec::kPrefill, in, want);
        CHECK(xs::test::relative_gap({out.at("logits").to_f32()}, {want.at("logits").to_f32()}) <= tolerance);
        for (Speculating* q : {&mine.back(), &alone.back()}) {
            q->drafter->add(q == &mine.back() ? out : want, n, 0, prompts[s]);
            q->drafter->flush();
            q->next = argmax(want.at("logits"), 0);
            q->pos = n;
        }
    }
    const int max_drafts = std::min(3, mine[0].drafter->max_drafts());
    double worst = 0;
    bool kept_all = false;
    for (int round = 0; round < 6; ++round) {
        std::vector<Segment> segments;
        std::vector<std::vector<int32_t>> fed;
        std::map<std::string, std::vector<HostTensor>> parts;
        for (int k = 0; k < 3; ++k) {
            const auto s = static_cast<size_t>((round + k) % 3);
            const int drafts = (round + k) % (max_drafts + 1);
            shared->use(static_cast<int64_t>(s));
            std::vector<int32_t> f{mine[s].next};
            if (drafts > 0) {
                const std::vector<int32_t> d = mine[s].drafter->propose(mine[s].next, drafts);
                CHECK(d == alone[s].drafter->propose(alone[s].next, drafts));
                f.insert(f.end(), d.begin(), d.end());
            }
            TensorMap in;
            model.prepare_inputs(f, mine[s].pos, {}, in);
            for (auto& [name, t] : in) parts[name].push_back(t);
            segments.push_back({static_cast<int64_t>(s), static_cast<int64_t>(f.size())});
            fed.push_back(std::move(f));
        }
        TensorMap in, out;
        for (auto& [name, list] : parts) in[name] = concat_rows(list);
        shared->run_segments(spec::kVerify, segments, in, out, {});
        int64_t row = 0;
        for (size_t i = 0; i < segments.size(); ++i) {
            const auto s = static_cast<size_t>(segments[i].sequence);
            const auto n = static_cast<int64_t>(fed[i].size());
            TensorMap one, want;
            model.prepare_inputs(fed[i], alone[s].pos, {}, one);
            own[s]->run(spec::kVerify, one, want);
            const TensorMap got = rows_of_all(out, row, n);
            for (const auto& [name, t] : want) {
                if (t.dtype() != DType::F32) continue;
                CAPTURE(name);
                worst = std::max(worst, xs::test::relative_gap({got.at(name).to_f32()}, {t.to_f32()}));
            }
            const int64_t keep = 1 + (round + static_cast<int64_t>(i)) % n;
            kept_all = kept_all || (n > 1 && keep == n);
            const int32_t next = argmax(want.at("logits"), keep - 1);
            shared->use(static_cast<int64_t>(s));
            for (Speculating* q : {&mine[s], &alone[s]}) {
                q->session->truncate(spec::kVerify, q->pos + keep);
                q->drafter->add(q == &mine[s] ? got : want, keep, q->pos, fed[i]);
                q->next = next;
                q->pos += keep;
            }
            CHECK(shared->num_cached(spec::kVerify) == own[s]->num_cached(spec::kVerify));
            row += n;
        }
    }
    MESSAGE(model.info().arch, " (", spec->drafter_name(), "): rounds verified together against alone, largest gap ",
            worst);
    CHECK(worst <= tolerance);
    CHECK(kept_all);
}

void check_rounds_both(const Model& model) {
    SessionOptions f32;
    f32.precision = DType::F32;
    check_rounds(*make_cpu_ref_backend(), model, f32, 1e-6);
    if (dev::enabled()) check_rounds(dev::backend(), model, dev::options(), dev::is_f16() ? 2e-2 : 1e-4);
}

LoadOptions speculating() {
    LoadOptions o = batched();
    o.vision = false;
    o.speculation = true;
    return o;
}

}  // namespace

TEST_SUITE("batch decode") {
    TEST_CASE("qwen3_5: dense, a mixture of experts, and looped layers in hyper-connections") {
        const xs::test::TempDir work("xs_batch_qwen");
        xs::test::TinyQwen moe;
        moe.experts = 4;
        xs::test::TinyQwen loop;
        loop.layers = {"linear_attention", "linear_attention", "linear_attention", "full_attention",
                       "linear_attention", "linear_attention", "linear_attention", "full_attention"};
        loop.loop_start = 2;
        loop.loop_end = 6;
        loop.loop_count = 2;
        for (const auto& [name, m] : {std::pair{"dense", xs::test::TinyQwen{}}, std::pair{"moe", moe},
                                      std::pair{"loop", loop}}) {
            CAPTURE(name);
            xs::test::write_tiny_qwen(work.path / name, m, 3);
            check_both(*load_model(work.path / name, batched()));
        }
        // Built only when asked for.
        CHECK(load_model(work.path / "dense")->batch_graph() == nullptr);
    }
    TEST_CASE("gemma4: sliding and full layers, per-layer embeddings, KV-shared layers") {
        const xs::test::TempDir work("xs_batch_gemma4");
        xs::test::write_tiny_gemma4(work.path, xs::test::TinyGemma4{}, 5);
        check_both(*load_model(work.path, batched()));
    }
    TEST_CASE("inkling2: looped layers, convolutions with taps per token, attention with a relative bias") {
        const xs::test::TempDir work("xs_batch_inkling2");
        fs::create_directories(work.path);
        xs::test::tiny_inkling2_gguf(true).write(work.path / "tiny.gguf");
        xs::test::import_into(work.path / "tiny.gguf", work.path / "pack");
        check_both(*load_model(work.path / "pack", batched()));
    }
    TEST_CASE("speculative rounds verified together: qwen3_5 with its MTP head, and a block drafter of each kind") {
        const xs::test::TempDir work("xs_batch_rounds_qwen");
        xs::test::build_tiny_gguf(work.path / "mtp", 11);  // twin/: a checkpoint with an MTP head
        check_rounds_both(*load_model(work.path / "mtp" / "twin", speculating()));
        xs::test::write_tiny_qwen(work.path / "target", xs::test::TinyQwen{}, 3);
        for (const char* kind : {"dflash", "dflash2", "dspine"}) {
            CAPTURE(kind);
            xs::test::TinyDrafter d;
            d.kind = kind;
            xs::test::write_tiny_drafter(work.path / kind, d, xs::test::TinyQwen{}, 11);
            LoadOptions o = speculating();
            o.drafter = work.path / kind;
            check_rounds_both(*load_model(work.path / "target", o));
        }
    }
    TEST_CASE("speculative rounds verified together: inkling2 with DSpark") {
        const xs::test::TempDir work("xs_batch_rounds_inkling2");
        fs::create_directories(work.path);
        xs::test::tiny_inkling2_gguf(true).write(work.path / "tiny.gguf");
        xs::test::tiny_inkling2_drafter_gguf(true).write(work.path / "dspark.gguf");
        xs::test::import_into(work.path / "tiny.gguf", work.path / "pack", work.path / "dspark.gguf");
        check_rounds_both(*load_model(work.path / "pack", speculating()));
    }
}
