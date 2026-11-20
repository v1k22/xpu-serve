#include "engine/engine.h"

#include <algorithm>
#include <deque>
#include <numeric>

#include "engine/sampler.h"
#include "util/error.h"
#include "util/log.h"
#include "util/timer.h"

namespace xs {

namespace {

constexpr uint64_t kMediaKeyBit = 1ULL << 63;

uint64_t text_key(int32_t token) { return static_cast<uint32_t>(token); }

// Distinct from every text key (top bit) and from the rows of other media items (the item's hash).
uint64_t media_key(uint64_t item, int64_t row) {
    uint64_t h = item ^ (static_cast<uint64_t>(row) * 0x9e3779b97f4a7c15ULL);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h | kMediaKeyBit;
}

}  // namespace

Engine::Engine(std::unique_ptr<Model> model, std::unique_ptr<Session> session, EncoderSessions encoders,
               EngineOptions options)
    : model_(std::move(model)), session_(std::move(session)), encoders_(std::move(encoders)), options_(options) {
    XS_CHECK(options_.max_context > 1 && options_.prefill_chunk > 0, "invalid engine options");
    for (size_t k = 0; k < kModalities; ++k) {
        const auto m = static_cast<Modality>(k);
        XS_CHECK(encoders_[k] == nullptr || model_->encoder(m) != nullptr, "an encoder session for ", modality_name(m),
                 " input, which model ", model_->info().id, " does not take");
    }
    XS_CHECK(options_.spec_k >= 0, "spec_k must not be negative, got ", options_.spec_k);
    const auto offers_argmax = [](const graph::Graph& g) {
        return std::any_of(g.outputs().begin(), g.outputs().end(), [](const auto& o) { return o.name == "argmax"; });
    };
    has_argmax_ = offers_argmax(model_->graph());
    if (options_.spec_k > 0) {
        const Speculation* spec = model_->speculation();
        XS_CHECK(spec != nullptr, "model ", model_->info().id, " has no drafter for speculative decoding");
        drafter_ = spec::make_drafter(*spec, *session_);
        XS_CHECK(options_.spec_k <= drafter_->max_drafts(), "spec_k ", options_.spec_k, " is more than the ",
                 drafter_->max_drafts(), " drafts a round of ", spec->drafter_name(), " proposes");
        has_argmax_ = has_argmax_ && offers_argmax(spec->prefill_graph()) && offers_argmax(spec->verify_graph());
    }
}

size_t Engine::state_program() const {
    return drafter_ ? spec::kVerify : spec::kStep;
}

void Engine::reset_sequence() {
    session_->reset();
    cached_.clear();
    if (drafter_) drafter_->reset();
}

void Engine::set_prefill_chunk(int64_t tokens) {
    XS_CHECK(tokens > 0, "prefill chunk must be positive, got ", tokens);
    options_.prefill_chunk = tokens;
}

void Engine::run_target(size_t program, std::span<const int32_t> tokens, std::span<const PositionKey> keys,
                        std::span<const RotaryPosition> positions, std::span<const float* const> embeddings) {
    XS_CHECK(!tokens.empty() && keys.size() == tokens.size() && positions.size() == tokens.size(),
             "forward without tokens");
    const auto start = static_cast<int64_t>(cached_.size());
    XS_CHECK(start + static_cast<int64_t>(tokens.size()) <= options_.max_context, "sequence of ",
             start + static_cast<int64_t>(tokens.size()), " tokens exceeds the context limit of ", options_.max_context);
    inputs_.clear();
    model_->prepare_positioned_inputs(tokens, positions, embeddings, inputs_);
    std::vector<std::string> wanted{argmax_only_ ? "argmax" : "logits"};
    if (drafter_) wanted.insert(wanted.end(), drafter_->target_outputs().begin(), drafter_->target_outputs().end());
    outputs_.clear();
    session_->run(program, inputs_, outputs_, wanted);
    cached_.insert(cached_.end(), keys.begin(), keys.end());
    XS_CHECK(session_->num_cached() == static_cast<int64_t>(cached_.size()), "backend holds ", session_->num_cached(),
             " cached tokens, engine expected ", cached_.size());
}

void Engine::forward(std::span<const int32_t> tokens, std::span<const PositionKey> keys,
                     std::span<const RotaryPosition> positions, std::span<const float* const> embeddings) {
    const auto start = static_cast<int64_t>(cached_.size());
    run_target(drafter_ ? spec::kPrefill : spec::kStep, tokens, keys, positions, embeddings);
    if (drafter_) {
        drafter_->add(outputs_, static_cast<int64_t>(tokens.size()), start, tokens);
        drafter_->flush();
    }
}

int32_t Engine::target_token(Sampler& sampler, int64_t row, std::span<const int32_t> prompt,
                             std::span<const int32_t> generated, const int32_t* draft) {
    const int64_t vocab = model_->info().vocab_size;
    if (argmax_only_) {
        const HostTensor& top = outputs_.at("argmax");
        XS_CHECK(top.dtype() == DType::I32 && row < top.numel(), "argmax output is ", dtype_name(top.dtype()),
                 dims_to_string(top.shape()), ", row ", row, " asked");
        const int32_t token = top.data<int32_t>()[row];
        XS_CHECK(token >= 0 && token < vocab, "argmax output holds token ", token, " of ", vocab);
        return token;
    }
    HostTensor& logits = outputs_.at("logits");
    XS_CHECK(logits.dtype() == DType::F32 && logits.numel() % vocab == 0 && row < logits.numel() / vocab,
             "logits output is ", dtype_name(logits.dtype()), dims_to_string(logits.shape()), ", row ", row, " asked");
    const std::span<float> values(logits.data<float>() + row * vocab, static_cast<size_t>(vocab));
    return draft != nullptr ? sampler.verify(values, prompt, generated, *draft) : sampler.sample(values, prompt, generated);
}

std::vector<int32_t> Engine::spec_round(int32_t token, int k, Sampler& sampler, std::span<const int32_t> prompt,
                                        std::span<const int32_t> generated, GenerateStats& stats) {
    std::vector<int32_t> fed{token};
    Timer timer;
    if (k > 0) {
        const std::vector<int32_t> drafts = drafter_->propose(token, k);
        fed.insert(fed.end(), drafts.begin(), drafts.end());
    }
    const double draft_ms = timer.elapsed_ms();
    std::vector<PositionKey> keys(fed.size());
    std::transform(fed.begin(), fed.end(), keys.begin(), text_key);
    const auto start = static_cast<int64_t>(cached_.size());
    run_target(spec::kVerify, fed, keys, text_positions(start, fed.size()), {});
    stats.spec_draft_ms += draft_ms;
    stats.spec_verify_ms += timer.elapsed_ms() - draft_ms;

    // Row i of the verify is the target's distribution after fed[i]: it accepts the draft fed[i + 1] or replaces it.
    std::vector<int32_t> history(generated.begin(), generated.end());
    std::vector<int32_t> out;
    for (size_t i = 0; i < fed.size(); ++i) {
        const bool last = i + 1 == fed.size();
        const int32_t t = target_token(sampler, static_cast<int64_t>(i), prompt, history, last ? nullptr : &fed[i + 1]);
        out.push_back(t);
        history.push_back(t);
        if (!last && t != fed[i + 1]) break;
    }
    const auto accepted = static_cast<int64_t>(out.size()) - 1;
    const int64_t keep = start + accepted + 1;  // the token and the accepted drafts
    if (keep < static_cast<int64_t>(cached_.size())) {
        session_->truncate(state_program(), keep);
        cached_.resize(static_cast<size_t>(keep));
    }
    drafter_->add(outputs_, accepted + 1, start, fed);
    ++stats.spec_rounds;
    stats.spec_drafted += k;
    stats.spec_accepted += static_cast<int>(accepted);
    return out;
}

int64_t Engine::reuse_prefix(const std::vector<PositionKey>& prompt_keys) {
    const auto common = static_cast<int64_t>(
        std::mismatch(cached_.begin(), cached_.end(), prompt_keys.begin(), prompt_keys.end()).first - cached_.begin());
    // At least the last prompt token must run, because its logits pick the first new token.
    const int64_t keep = std::min<int64_t>(common, static_cast<int64_t>(prompt_keys.size()) - 1);
    if (keep <= 0) {
        reset_sequence();
        return 0;
    }
    if (keep < static_cast<int64_t>(cached_.size())) {
        if (model_->graph().has_fixed_state()) {
            // Conv and recurrent state cannot be cut back into the sequence; start over.
            reset_sequence();
            return 0;
        }
        session_->truncate(state_program(), keep);
        cached_.resize(static_cast<size_t>(keep));
        if (drafter_) drafter_->truncate(keep);
    }
    return keep;
}

std::vector<RotaryPosition> Engine::text_positions(int64_t index, size_t count) const {
    std::vector<RotaryPosition> out(count);
    for (size_t i = 0; i < count; ++i) {
        const auto p = static_cast<int32_t>(index + static_cast<int64_t>(i) + rotary_delta_);
        out[i] = {p, p, p};
    }
    return out;
}

int64_t Engine::media_width() const {
    return model_->info().media_width > 0 ? model_->info().media_width : model_->info().hidden_size;
}

std::shared_ptr<const HostTensor> Engine::encode(const MediaInput& media, GenerateStats& stats) {
    for (auto it = encoded_.begin(); it != encoded_.end(); ++it) {
        if (it->first != media.key) continue;
        encoded_.splice(encoded_.begin(), encoded_, it);
        return encoded_.front().second;
    }
    Timer timer;
    const int64_t width = media_width();
    auto rows = std::make_shared<HostTensor>(DType::F32, Dims{media.num_tokens, width});  // zeros past an encoder's
    int64_t at = 0;
    for (const EncoderRun& run : media.runs) {
        TensorMap out;
        encoders_[static_cast<size_t>(media.modality)]->run(run.program, run.inputs, out);
        const HostTensor& e = out.at("embeddings");
        XS_CHECK(e.dtype() == DType::F32 && e.rank() == 2 && e.dim(1) <= width && e.dim(0) >= run.rows &&
                     at + run.rows <= media.num_tokens,
                 modality_name(media.modality), " encoder output is ", dtype_name(e.dtype()), dims_to_string(e.shape()),
                 "; expected ", run.rows, " rows of up to ", width, " (", at, " of ", media.num_tokens, " so far)");
        const int64_t cols = e.dim(1);
        for (int64_t r = 0; r < run.rows; ++r) {
            std::copy_n(e.data<float>() + r * cols, cols, rows->data<float>() + (at + r) * width);
        }
        at += run.rows;
    }
    XS_CHECK(at == media.num_tokens, "a ", modality_name(media.modality), " item's runs give ", at, " rows for ",
             media.num_tokens, " soft tokens");
    stats.encode_ms += timer.elapsed_ms();
    ++stats.media_encoded;
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
        // Soft-token positions take the encoder rows of the request's media items, in order.
        const auto n = prompt.size();
        std::vector<PositionKey> keys(n);
        std::vector<int32_t> media_item(n, -1);
        std::vector<int64_t> media_row(n, 0);
        for (const MediaInput& m : request.media) {
            if (!accepts(m.modality)) fail("this model does not take ", modality_name(m.modality), " input");
        }
        // The modality whose soft token `token` is, among the loaded encoders; -1 for none.
        const auto soft_modality = [&](int32_t token) {
            for (size_t k = 0; k < kModalities; ++k) {
                if (encoders_[k] != nullptr && model_->encoder(static_cast<Modality>(k))->soft_token() == token) {
                    return static_cast<int>(k);
                }
            }
            return -1;
        };
        const int64_t needed = std::accumulate(request.media.begin(), request.media.end(), int64_t{0},
                                               [](int64_t sum, const MediaInput& m) { return sum + m.num_tokens; });
        int64_t found = 0;
        for (size_t i = 0; i < n; ++i) found += soft_modality(prompt[i]) >= 0;
        XS_CHECK(found == needed, "the prompt holds ", found, " media tokens but its ", request.media.size(),
                 " media items need ", needed);
        size_t item = 0;
        int64_t row = 0;
        for (size_t i = 0; i < n; ++i) {
            const int k = soft_modality(prompt[i]);
            if (k < 0) {
                keys[i] = text_key(prompt[i]);
                continue;
            }
            while (row == request.media[item].num_tokens) {
                ++item;
                row = 0;
            }
            XS_CHECK(static_cast<int>(request.media[item].modality) == k, "the prompt's ",
                     modality_name(static_cast<Modality>(k)), " tokens at position ", i, " fall to media item ", item,
                     ", which is ", modality_name(request.media[item].modality));
            media_item[i] = static_cast<int32_t>(item);
            media_row[i] = row;
            keys[i] = media_key(request.media[item].key, row++);
        }

        // Rotary positions: text one past what came before; a media item's tokens at their offsets from the item's
        // first position, the token after it one past its largest (transformers' get_rope_index).
        std::vector<RotaryPosition> positions(n);
        int64_t next = 0, base = 0;
        for (size_t i = 0; i < n; ++i) {
            const int32_t m = media_item[i];
            if (m < 0) {
                const auto p = static_cast<int32_t>(next++);
                positions[i] = {p, p, p};
                continue;
            }
            const MediaInput& media = request.media[static_cast<size_t>(m)];
            const int64_t r = media_row[i];
            XS_CHECK(media.positions.empty() || static_cast<int64_t>(media.positions.size()) == media.num_tokens,
                     "a ", modality_name(media.modality), " item with ", media.positions.size(), " positions for ",
                     media.num_tokens, " tokens");
            if (r == 0) base = next;
            const auto r32 = static_cast<int32_t>(r);
            const RotaryPosition offset =
                media.positions.empty() ? RotaryPosition{r32, r32, r32} : media.positions[static_cast<size_t>(r)];
            for (size_t a = 0; a < 3; ++a) positions[i][a] = static_cast<int32_t>(base + offset[a]);
            if (r + 1 == media.num_tokens) {
                int64_t largest = media.num_tokens - 1;
                if (!media.positions.empty()) {
                    largest = 0;
                    for (const RotaryPosition& p : media.positions) {
                        largest = std::max<int64_t>({largest, p[0], p[1], p[2]});
                    }
                }
                next = base + largest + 1;
            }
        }
        rotary_delta_ = next - static_cast<int64_t>(n);

        Sampler sampler(request.sampling);
        argmax_only_ = has_argmax_ && !sampler.needs_logits();
        Timer prefill_timer;
        const int64_t cached = reuse_prefix(keys);
        result.stats.cached_tokens = static_cast<int>(cached);
        std::vector<std::shared_ptr<const HostTensor>> encoded(request.media.size());
        std::vector<const float*> rows;
        const int64_t width = media_width();
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
            forward(std::span<const int32_t>(prompt).subspan(pos, len),
                    std::span<const PositionKey>(keys).subspan(pos, len),
                    std::span<const RotaryPosition>(positions).subspan(pos, len),
                    any_rows ? std::span<const float* const>(rows) : std::span<const float* const>());
            pos += len;
        }
        result.stats.prefill_ms = prefill_timer.elapsed_ms() - result.stats.encode_ms;

        Timer decode_timer;
        const int64_t room = options_.max_context - static_cast<int64_t>(n);
        const int64_t max_new = std::min<int64_t>(request.max_new_tokens, room);
        result.finish = FinishReason::Length;
        std::deque<int32_t> ready;  // tokens a speculative round produced, not emitted yet
        int32_t token = max_new > 0 ? target_token(sampler, 0, prompt, result.tokens, nullptr) : 0;
        while (static_cast<int64_t>(result.tokens.size()) < max_new) {
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
            const auto emitted = static_cast<int64_t>(result.tokens.size());
            if (emitted >= max_new) break;
            if (ready.empty() && drafter_) {
                // Drafts past the last token wanted, or past the context, are never verified.
                const int64_t fits = options_.max_context - static_cast<int64_t>(cached_.size()) - 1;
                const auto k = static_cast<int>(std::min<int64_t>({options_.spec_k, max_new - emitted - 1, fits}));
                const std::vector<int32_t> round = spec_round(token, k, sampler, prompt, result.tokens, result.stats);
                ready.assign(round.begin(), round.end());
            } else if (ready.empty()) {
                const int32_t fed[] = {token};
                const PositionKey fed_key[] = {text_key(token)};
                forward(fed, fed_key, text_positions(static_cast<int64_t>(cached_.size()), 1), {});
                ready.push_back(target_token(sampler, 0, prompt, result.tokens, nullptr));
            }
            token = ready.front();
            ready.pop_front();
        }
        result.stats.generated_tokens = static_cast<int>(result.tokens.size());
    } catch (const std::exception& e) {
        // The KV state may be partially written; start the next request from scratch.
        reset_sequence();
        result.finish = FinishReason::Error;
        result.error = e.what();
        log::error("generation failed: ", e.what());
    }
    return result;
}

}  // namespace xs
