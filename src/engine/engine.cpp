#include "engine/engine.h"

#include <algorithm>
#include <numeric>

#include "engine/sampler.h"
#include "util/error.h"
#include "util/log.h"
#include "util/timer.h"

namespace xs {

namespace {

constexpr uint64_t kMediaKeyBit = 1ULL << 63;

uint64_t text_key(int32_t token) { return static_cast<uint32_t>(token); }

// Distinct from every text key (top bit) and from the rows of other images (the image hash).
uint64_t media_key(uint64_t image, int64_t row) {
    uint64_t h = image ^ (static_cast<uint64_t>(row) * 0x9e3779b97f4a7c15ULL);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h | kMediaKeyBit;
}

}  // namespace

Engine::Engine(std::unique_ptr<Model> model, std::unique_ptr<Session> session, std::unique_ptr<Session> vision_session,
               EngineOptions options)
    : model_(std::move(model)),
      session_(std::move(session)),
      vision_session_(std::move(vision_session)),
      options_(options) {
    XS_CHECK(options_.max_context > 1 && options_.prefill_chunk > 0, "invalid engine options");
    XS_CHECK(vision_session_ == nullptr || model_->vision() != nullptr, "a vision session for a model without vision");
}

void Engine::set_prefill_chunk(int64_t tokens) {
    XS_CHECK(tokens > 0, "prefill chunk must be positive, got ", tokens);
    options_.prefill_chunk = tokens;
}

std::vector<float> Engine::forward(std::span<const int32_t> tokens, std::span<const PositionKey> keys,
                                   std::span<const float* const> embeddings) {
    XS_CHECK(!tokens.empty() && keys.size() == tokens.size(), "forward without tokens");
    const auto start = static_cast<int64_t>(cached_.size());
    XS_CHECK(start + static_cast<int64_t>(tokens.size()) <= options_.max_context, "sequence of ",
             start + static_cast<int64_t>(tokens.size()), " tokens exceeds the context limit of ", options_.max_context);
    inputs_.clear();
    model_->prepare_inputs(tokens, start, embeddings, inputs_);
    session_->run(inputs_, outputs_);
    cached_.insert(cached_.end(), keys.begin(), keys.end());
    XS_CHECK(session_->num_cached() == static_cast<int64_t>(cached_.size()), "backend holds ", session_->num_cached(),
             " cached tokens, engine expected ", cached_.size());
    const HostTensor& logits = outputs_.at("logits");
    XS_CHECK(logits.dtype() == DType::F32 && logits.numel() == model_->info().vocab_size, "logits output is ",
             dtype_name(logits.dtype()), dims_to_string(logits.shape()));
    return {logits.data<float>(), logits.data<float>() + logits.numel()};
}

int64_t Engine::reuse_prefix(const std::vector<PositionKey>& prompt_keys) {
    const auto common = static_cast<int64_t>(
        std::mismatch(cached_.begin(), cached_.end(), prompt_keys.begin(), prompt_keys.end()).first - cached_.begin());
    // At least the last prompt token must run, because its logits pick the first new token.
    const int64_t keep = std::min<int64_t>(common, static_cast<int64_t>(prompt_keys.size()) - 1);
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

std::shared_ptr<const HostTensor> Engine::encode(const MediaInput& media, GenerateStats& stats) {
    for (auto it = encoded_.begin(); it != encoded_.end(); ++it) {
        if (it->first != media.key) continue;
        encoded_.splice(encoded_.begin(), encoded_, it);
        return encoded_.front().second;
    }
    Timer timer;
    TensorMap out;
    vision_session_->run(media.encoder_inputs, out);
    auto rows = std::make_shared<const HostTensor>(std::move(out.at("embeddings")));
    stats.encode_ms += timer.elapsed_ms();
    ++stats.images_encoded;
    XS_CHECK(rows->dtype() == DType::F32 && rows->rank() == 2 && rows->dim(1) == model_->info().hidden_size &&
                 rows->dim(0) >= media.num_tokens,
             "vision encoder output is ", dtype_name(rows->dtype()), dims_to_string(rows->shape()), "; expected ",
             media.num_tokens, " rows of ", model_->info().hidden_size);
    encoded_.emplace_front(media.key, rows);
    if (encoded_.size() > options_.encoded_media_cache) encoded_.pop_back();
    return rows;
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
        // Soft-token positions take the encoder rows of the request's images, in order.
        const auto n = prompt.size();
        std::vector<PositionKey> keys(n);
        std::vector<int32_t> media_item(n, -1);
        std::vector<int64_t> media_row(n, 0);
        const VisionEncoder* vision = vision_session_ != nullptr ? model_->vision() : nullptr;
        if (!request.media.empty() && vision == nullptr) fail("this model does not take images");
        const int64_t needed = std::accumulate(request.media.begin(), request.media.end(), int64_t{0},
                                               [](int64_t sum, const MediaInput& m) { return sum + m.num_tokens; });
        int64_t found = 0;
        for (size_t i = 0; i < n; ++i) found += vision != nullptr && prompt[i] == vision->soft_token();
        XS_CHECK(found == needed, "the prompt holds ", found, " image tokens but its ", request.media.size(),
                 " images need ", needed);
        size_t item = 0;
        int64_t row = 0;
        for (size_t i = 0; i < n; ++i) {
            if (vision == nullptr || prompt[i] != vision->soft_token()) {
                keys[i] = text_key(prompt[i]);
                continue;
            }
            while (row == request.media[item].num_tokens) {
                ++item;
                row = 0;
            }
            media_item[i] = static_cast<int32_t>(item);
            media_row[i] = row;
            keys[i] = media_key(request.media[item].key, row++);
        }

        Timer prefill_timer;
        const int64_t cached = reuse_prefix(keys);
        result.stats.cached_tokens = static_cast<int>(cached);
        std::vector<std::shared_ptr<const HostTensor>> encoded(request.media.size());
        std::vector<const float*> rows;
        std::vector<float> logits;
        const int64_t width = model_->info().hidden_size;
        for (auto pos = static_cast<size_t>(cached); pos < n;) {
            const size_t len = std::min(n - pos, static_cast<size_t>(options_.prefill_chunk));
            rows.assign(len, nullptr);
            bool any_rows = false;
            for (size_t j = 0; j < len; ++j) {
                const int32_t m = media_item[pos + j];
                if (m < 0) continue;
                auto& e = encoded[static_cast<size_t>(m)];
                if (!e) e = encode(request.media[static_cast<size_t>(m)], result.stats);
                rows[j] = e->data<float>() + media_row[pos + j] * width;
                any_rows = true;
            }
            logits = forward(std::span<const int32_t>(prompt).subspan(pos, len),
                             std::span<const PositionKey>(keys).subspan(pos, len),
                             any_rows ? std::span<const float* const>(rows) : std::span<const float* const>());
            pos += len;
        }
        result.stats.prefill_ms = prefill_timer.elapsed_ms() - result.stats.encode_ms;

        Sampler sampler(request.sampling);
        Timer decode_timer;
        const int64_t room = options_.max_context - static_cast<int64_t>(n);
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
            const PositionKey next_key[] = {text_key(token)};
            logits = forward(next, next_key, {});
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
