#pragma once

// What the GGUF import tests share: importing a file, a model's logits, greedy tokens and drafts on cpu-ref, a weight's
// values, and the error a changed file gives.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "backend/cpu_ref/cpu_backend.h"
#include "convert/gguf/gguf_import.h"
#include "engine/spec/chain_drafter.h"
#include "io/gguf_test_util.h"
#include "model/step_logits.h"
#include "models/model.h"
#include "quant/quant.h"
#include "util/error.h"

namespace xs::test {

// `gguf` imported into `out`, with its MTP head from the file `mtp` when one is given.
inline gguf_import::ImportReport import_into(const std::filesystem::path& gguf, const std::filesystem::path& out,
                                             const std::filesystem::path& mtp = {}) {
    gguf_import::ImportOptions o;
    o.source = gguf;
    o.output = out;
    o.mtp = mtp;
    return gguf_import::import_gguf(o);
}

inline SessionOptions f32_session() {
    SessionOptions so;
    so.precision = DType::F32;
    return so;
}

// The logits of a prefill of `prompt` and of one decode step per token of `steps` after it, on cpu-ref.
inline std::vector<std::vector<float>> step_logits(const Model& model, const std::vector<int32_t>& prompt,
                                                   const std::vector<int32_t>& steps) {
    return step_logits(*make_cpu_ref_backend(), model, f32_session(), prompt, steps);
}

// `tokens` greedy tokens after `prompt`.
inline std::vector<int32_t> greedy(const Model& model, const std::vector<int32_t>& prompt, int tokens) {
    const auto backend = make_cpu_ref_backend();
    const auto session = backend->compile(model.graph(), model.weights(), f32_session());
    std::vector<int32_t> out, feed = prompt;
    int64_t pos = 0;
    for (int i = 0; i < tokens; ++i) {
        TensorMap inputs, outputs;
        model.prepare_inputs(feed, pos, {}, inputs);
        session->run(inputs, outputs);
        pos += static_cast<int64_t>(feed.size());
        const HostTensor& l = outputs.at("logits");
        const float* row = l.data<float>();
        const auto best = static_cast<int32_t>(std::max_element(row, row + l.numel()) - row);
        out.push_back(best);
        feed = {best};
    }
    return out;
}

// The MTP head's logits for three pairs (hidden row, next token), from its own program.
inline std::vector<float> draft_logits(const Model& model) {
    const Speculation* spec = model.speculation();
    REQUIRE(spec != nullptr);
    REQUIRE(spec->chain() != nullptr);
    const auto backend = make_cpu_ref_backend();
    const auto session = backend->compile(spec->draft_graph(), model.weights(), f32_session());
    const int64_t hidden = model.info().hidden_size;
    std::vector<float> rows(static_cast<size_t>(3 * hidden));
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = std::sin(0.37f * static_cast<float>(i));
    TensorMap inputs, outputs;
    spec->chain()->prepare_draft(std::vector<int32_t>{5, 77, 266}, rows, 4, 0, inputs);
    session->run(inputs, outputs);
    const HostTensor& l = outputs.at("logits");
    return {l.data<float>(), l.data<float>() + l.numel()};
}

// A model's drafter after a prefill of `prompt`: its logits for (the first greedy token, the prompt's last hidden row),
// and the three drafts it chains from them.
struct Drafts {
    std::vector<float> logits;
    std::vector<int32_t> tokens;
};
inline Drafts drafts(const Model& model, const std::vector<int32_t>& prompt) {
    const Speculation* spec = model.speculation();
    REQUIRE(spec != nullptr);
    REQUIRE(spec->chain() != nullptr);
    SessionOptions options = f32_session();
    options.rewind_tokens = 4;
    const graph::Graph* programs[] = {&model.graph(), &spec->prefill_graph(), &spec->verify_graph(),
                                      &spec->draft_graph()};
    const auto session = make_cpu_ref_backend()->compile(programs, model.weights(), options);
    TensorMap inputs, outputs;
    model.prepare_inputs(prompt, 0, {}, inputs);
    session->run(spec::kPrefill, inputs, outputs);
    const HostTensor hidden = std::move(outputs.at("hidden"));
    const HostTensor& l = outputs.at("logits");
    const auto first =
        static_cast<int32_t>(std::max_element(l.data<float>(), l.data<float>() + l.numel()) - l.data<float>());
    const std::span<const float> rows(hidden.data<float>(), static_cast<size_t>(hidden.numel()));
    inputs.clear();
    spec->chain()->prepare_draft(std::vector<int32_t>{first}, rows.last(static_cast<size_t>(hidden.dim(1))),
                                 static_cast<int64_t>(prompt.size()) - 1, 0, inputs);
    session->run(spec::kDraft, inputs, outputs);
    Drafts out;
    out.logits.assign(outputs.at("logits").data<float>(),
                      outputs.at("logits").data<float>() + outputs.at("logits").numel());
    spec::ChainDrafter drafter(*spec, *session);
    drafter.add(TensorMap{{"hidden", hidden}}, hidden.dim(0), 0, prompt);
    out.tokens = drafter.propose(first, 3);
    return out;
}

inline std::vector<float> values(const TensorView& v) {
    std::vector<float> out(static_cast<size_t>(v.numel()));
    quant::read_rows(v, 0, quant::matrix_rows(v.shape), out.data());
    return out;
}

// `error` names `text`; else it is shown.
inline bool has(const std::string& error, const std::string& text) {
    const bool found = error.find(text) != std::string::npos;
    if (!found) MESSAGE("the error is: '" << error << "'");
    return found;
}

// The GGUF `build(dir / "src")` writes, with `change` applied, imported into a fresh directory: the error it gives,
// or "" when it imports.
template <typename Build, typename Change>
std::string refusal(const std::filesystem::path& dir, Build build, Change change) {
    GgufWriter w = build(dir / "src");
    change(w);
    w.write(dir / "variant.gguf");
    std::filesystem::remove_all(dir / "variant");
    try {
        import_into(dir / "variant.gguf", dir / "variant");
    } catch (const Error& e) {
        return e.what();
    }
    return "";
}

}  // namespace xs::test
