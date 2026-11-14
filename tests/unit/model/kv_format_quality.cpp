// What a quantized KV cache changes in a model's predictions. Opt-in: XS_TEST_MODEL_DIR and XS_KV_QUALITY_TEXT. The
// model reads the same text over the float cache and over each format, and its next-token distributions are compared
// position by position: every position attends to keys and values the format has rounded.
//
//   XS_KV_QUALITY_TEXT     a UTF-8 text file (wikitext-2's test split, say), read with the model's tokenizer
//   XS_KV_QUALITY_FORMATS  --kv-dtype values, comma separated (default i8,turbo4,turbo3,turbo2)
//   XS_KV_QUALITY_WINDOW   tokens per window (default 512); each window starts from an empty cache
//   XS_KV_QUALITY_WINDOWS  windows read (default 16)
//   XS_KV_QUALITY_SCORED   positions compared at the end of each window (default half of it); the tokens before
//                          them are prefilled, these are fed one at a time
//   XS_PARITY_BACKEND, XS_PARITY_DEVICE, XS_PARITY_PRECISION as model_parity.cpp
//
// Reports per format the perplexity and its change, the mean and 99th percentile KL divergence from the float cache's
// distribution, and how often the most likely token stays the same. The float cache's log-probabilities are kept in
// host memory: 4 bytes times the vocabulary per compared position.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "models/model.h"
#include "parity_util.h"
#include "tokenizer/tokenizer.h"
#include "util/timer.h"

namespace {

using xs::test::env_or;

bool enabled() { return xs::test::model_dir().has_value() && !env_or("XS_KV_QUALITY_TEXT", "").empty(); }

struct Plan {
    std::vector<int32_t> tokens;  // windows * window of them
    int64_t window = 0;
    int64_t windows = 0;
    int64_t scored = 0;
};

// log_softmax of a row of logits.
std::vector<float> log_probabilities(const xs::HostTensor& logits) {
    const float* x = logits.data<float>();
    const auto n = static_cast<size_t>(logits.numel());
    const float top = *std::max_element(x, x + n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += std::exp(static_cast<double>(x[i] - top));
    const float shift = top + static_cast<float>(std::log(sum));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = x[i] - shift;
    return out;
}

// Reads every window over a cache in `kv`; `each(position, log-probabilities)` for each compared position in order.
// Returns the decode steps per second.
template <typename Each>
double read_windows(const xs::Model& model, const Plan& plan, const xs::quant::KvFormats& kv, Each each) {
    xs::SessionOptions options = xs::test::parity_options();
    options.kv = kv;
    options.max_context = plan.window;
    const auto session =
        xs::make_backend(env_or("XS_PARITY_BACKEND", "cpu-ref"))->compile(model.graph(), model.weights(), options);
    const std::string only[] = {"logits"};
    const int64_t prefill = plan.window - plan.scored;
    constexpr int64_t kChunk = 1024;
    double decode_seconds = 0.0;
    int64_t position = 0;
    for (int64_t w = 0; w < plan.windows; ++w) {
        const std::span<const int32_t> tokens(plan.tokens.data() + w * plan.window, static_cast<size_t>(plan.window));
        session->reset();
        xs::TensorMap inputs, outputs;
        for (int64_t at = 0; at < prefill; at += kChunk) {
            const int64_t n = std::min(kChunk, prefill - at);
            model.prepare_inputs(tokens.subspan(static_cast<size_t>(at), static_cast<size_t>(n)), at, {}, inputs);
            session->run(0, inputs, outputs, only);
        }
        each(position++, log_probabilities(outputs.at("logits")));  // what follows the prefilled tokens
        const xs::Timer timer;
        for (int64_t at = prefill; at + 1 < plan.window; ++at) {
            model.prepare_inputs(tokens.subspan(static_cast<size_t>(at), 1), at, {}, inputs);
            session->run(0, inputs, outputs, only);
            each(position++, log_probabilities(outputs.at("logits")));
        }
        decode_seconds += timer.elapsed_ms() / 1000.0;
    }
    return static_cast<double>(plan.windows * (plan.scored - 1)) / decode_seconds;
}

// The token each compared position predicts.
int32_t target(const Plan& plan, int64_t position) {
    const int64_t w = position / plan.scored;
    return plan.tokens[static_cast<size_t>(w * plan.window + plan.window - plan.scored + position % plan.scored)];
}

// The largest rise in perplexity a format may cost: the bar this measurement holds a model to.
double perplexity_bar(const xs::quant::KvFormats& kv) {
    const auto bar = [](xs::quant::KvFormat f) {
        switch (f) {
            case xs::quant::KvFormat::Float: return 0.0;
            case xs::quant::KvFormat::I8: return 0.01;
            case xs::quant::KvFormat::Turbo4: return 0.02;
            case xs::quant::KvFormat::Turbo3: return 0.05;
            case xs::quant::KvFormat::Turbo2: return 0.20;
        }
        return 0.0;
    };
    return std::max(bar(kv.k), bar(kv.v));
}

}  // namespace

TEST_CASE("a quantized KV cache keeps the model's predictions" * doctest::skip(!enabled())) {
    const auto dir = *xs::test::model_dir();
    const auto model = xs::load_model(dir);
    Plan plan;
    plan.window = std::stoll(env_or("XS_KV_QUALITY_WINDOW", "512"));
    plan.windows = std::stoll(env_or("XS_KV_QUALITY_WINDOWS", "16"));
    plan.scored = std::stoll(env_or("XS_KV_QUALITY_SCORED", std::to_string(plan.window / 2)));
    REQUIRE((plan.scored >= 2 && plan.scored < plan.window));
    {
        std::ifstream in(env_or("XS_KV_QUALITY_TEXT", ""), std::ios::binary);
        REQUIRE_MESSAGE(in.good(), "cannot read XS_KV_QUALITY_TEXT");
        std::stringstream text;
        text << in.rdbuf();
        plan.tokens = xs::tok::Tokenizer::load(dir)->encode(text.str(), false);
    }
    REQUIRE_MESSAGE(static_cast<int64_t>(plan.tokens.size()) >= plan.window, "the text has ", plan.tokens.size(),
                    " tokens, fewer than one window");
    plan.windows = std::min<int64_t>(plan.windows, static_cast<int64_t>(plan.tokens.size()) / plan.window);
    const int64_t positions = plan.windows * plan.scored;

    std::vector<std::vector<float>> reference(static_cast<size_t>(positions));
    double reference_nll = 0.0;
    const double reference_speed = read_windows(*model, plan, {}, [&](int64_t p, std::vector<float> logp) {
        reference_nll -= logp[static_cast<size_t>(target(plan, p))];
        reference[static_cast<size_t>(p)] = std::move(logp);
    });
    const double reference_ppl = std::exp(reference_nll / static_cast<double>(positions));
    MESSAGE("float cache: perplexity ", reference_ppl, " over ", positions, " positions (", plan.windows,
            " windows of ", plan.window, " tokens), ", reference_speed, " tok/s");

    std::stringstream formats(env_or("XS_KV_QUALITY_FORMATS", "i8,turbo4,turbo3,turbo2"));
    for (std::string name; std::getline(formats, name, ',');) {
        CAPTURE(name);
        const xs::quant::KvFormats kv = xs::quant::parse_kv_formats(name);
        double nll = 0.0, kl_sum = 0.0;
        int64_t same_top = 0;
        std::vector<double> kls(static_cast<size_t>(positions));
        const double speed = read_windows(*model, plan, kv, [&](int64_t p, std::vector<float> logp) {
            const std::vector<float>& ref = reference[static_cast<size_t>(p)];
            nll -= logp[static_cast<size_t>(target(plan, p))];
            double kl = 0.0;
            for (size_t i = 0; i < ref.size(); ++i)
                kl += std::exp(static_cast<double>(ref[i])) * (static_cast<double>(ref[i]) - logp[i]);
            kls[static_cast<size_t>(p)] = kl;
            kl_sum += kl;
            same_top += std::max_element(ref.begin(), ref.end()) - ref.begin() ==
                                std::max_element(logp.begin(), logp.end()) - logp.begin()
                            ? 1
                            : 0;
        });
        std::sort(kls.begin(), kls.end());
        const double ppl = std::exp(nll / static_cast<double>(positions));
        const double change = ppl / reference_ppl - 1.0;
        const double agree = static_cast<double>(same_top) / static_cast<double>(positions);
        MESSAGE(name, ": perplexity ", ppl, " (", change * 100.0, " %), KL mean ", kl_sum / static_cast<double>(positions),
                ", p99 ", kls[static_cast<size_t>(positions * 99 / 100)], ", same top token ", agree * 100.0, " %, ",
                speed, " tok/s");
        CHECK(change <= perplexity_bar(kv));
    }
}
