#include "engine/engine.h"

#include <algorithm>

#include "engine/sampler.h"
#include "util/error.h"
#include "util/log.h"
#include "util/timer.h"

namespace xs {

Engine::Engine(std::unique_ptr<Model> model, std::unique_ptr<Session> session, EngineOptions options)
    : model_(std::move(model)), session_(std::move(session)), options_(options) {
    XS_CHECK(options_.max_context > 1 && options_.prefill_chunk > 0, "invalid engine options");
}

std::vector<float> Engine::forward(std::span<const int32_t> tokens) {
    XS_CHECK(!tokens.empty(), "forward without tokens");
    const auto start = static_cast<int64_t>(cached_.size());
    XS_CHECK(start + static_cast<int64_t>(tokens.size()) <= options_.max_context, "sequence of ",
             start + static_cast<int64_t>(tokens.size()), " tokens exceeds the context limit of ", options_.max_context);
    inputs_.clear();
    model_->prepare_inputs(tokens, start, inputs_);
    session_->run(inputs_, outputs_);
    cached_.insert(cached_.end(), tokens.begin(), tokens.end());
    XS_CHECK(session_->num_cached() == static_cast<int64_t>(cached_.size()), "backend holds ", session_->num_cached(),
             " cached tokens, engine expected ", cached_.size());
    const HostTensor& logits = outputs_.at("logits");
    XS_CHECK(logits.dtype() == DType::F32 && logits.numel() == model_->info().vocab_size, "logits output is ",
             dtype_name(logits.dtype()), dims_to_string(logits.shape()));
    return {logits.data<float>(), logits.data<float>() + logits.numel()};
}

int64_t Engine::reuse_prefix(const std::vector<int32_t>& prompt) {
    const auto common = static_cast<int64_t>(
        std::mismatch(cached_.begin(), cached_.end(), prompt.begin(), prompt.end()).first - cached_.begin());
    // At least the last prompt token must run, because its logits pick the first new token.
    const int64_t keep = std::min<int64_t>(common, static_cast<int64_t>(prompt.size()) - 1);
    if (keep <= 0) {
        session_->reset();
        cached_.clear();
        return 0;
    }
    if (keep < static_cast<int64_t>(cached_.size())) {
        session_->truncate(keep);
        cached_.resize(static_cast<size_t>(keep));
    }
    return keep;
}

GenerateResult Engine::generate(const GenerateRequest& request, const TokenCallback& on_token) {
    GenerateResult result;
    const auto& prompt = request.prompt;
    if (prompt.empty()) {
        result.finish = FinishReason::Error;
        result.error = "empty prompt";
        return result;
    }
    if (static_cast<int64_t>(prompt.size()) >= options_.max_context) {
        result.finish = FinishReason::Error;
        result.error = str_cat("prompt of ", prompt.size(), " tokens does not fit the context limit of ",
                               options_.max_context, " tokens");
        return result;
    }
    result.stats.prompt_tokens = static_cast<int>(prompt.size());

    try {
        Timer prefill_timer;
        const int64_t cached = reuse_prefix(prompt);
        result.stats.cached_tokens = static_cast<int>(cached);
        std::vector<float> logits;
        for (auto pos = static_cast<size_t>(cached); pos < prompt.size();) {
            const size_t n = std::min(prompt.size() - pos, static_cast<size_t>(options_.prefill_chunk));
            logits = forward(std::span<const int32_t>(prompt).subspan(pos, n));
            pos += n;
        }
        result.stats.prefill_ms = prefill_timer.elapsed_ms();

        Sampler sampler(request.sampling);
        Timer decode_timer;
        const int64_t room = options_.max_context - static_cast<int64_t>(prompt.size());
        const int64_t max_new = std::min<int64_t>(request.max_new_tokens, room);
        result.finish = FinishReason::Length;
        while (static_cast<int64_t>(result.tokens.size()) < max_new) {
            const int32_t token = sampler.sample(logits, prompt, result.tokens);
            if (std::find(request.stop_token_ids.begin(), request.stop_token_ids.end(), token) !=
                request.stop_token_ids.end()) {
                result.finish = FinishReason::Stop;
                break;
            }
            // Decode speed is measured between the first and the last emitted token.
            if (result.tokens.empty()) decode_timer.reset();
            result.tokens.push_back(token);
            result.stats.decode_ms = decode_timer.elapsed_ms();
            if (on_token && !on_token(token)) {
                result.finish = FinishReason::Cancelled;
                break;
            }
            if (static_cast<int64_t>(result.tokens.size()) >= max_new) break;
            const int32_t next[] = {token};
            logits = forward(next);
        }
        result.stats.generated_tokens = static_cast<int>(result.tokens.size());
    } catch (const std::exception& e) {
        // The KV state may be partially written; start the next request from scratch.
        session_->reset();
        cached_.clear();
        result.finish = FinishReason::Error;
        result.error = e.what();
        log::error("generation failed: ", e.what());
    }
    return result;
}

}  // namespace xs
