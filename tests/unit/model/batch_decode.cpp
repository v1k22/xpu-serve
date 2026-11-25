// Each language model's batch graph (Model::batch_graph) on tiny checkpoints: decode steps of three sequences at once,
// as segments of one row each, give every sequence the logits of its own step graph run alone, on cpu-ref (whose
// segmented steps are the definition, Session::run_segments) and on the GPU when one is given (whose are its own).
#include <doctest/doctest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend/device_test_support.h"
#include "convert/import_test_util.h"
#include "convert/tiny_inkling2_gguf.h"
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
}
