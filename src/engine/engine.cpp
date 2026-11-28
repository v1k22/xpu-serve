#include "engine/engine.h"

#include <algorithm>
#include <deque>
#include <iterator>
#include <numeric>

#include "engine/media.h"
#include "engine/sampler.h"
#include "engine/spec/chain_drafter.h"
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
    if (options_.spec_k > 0) draft_length_ = spec::DraftLength(options_.spec_k);
    XS_CHECK(options_.verify_rows >= 1, "verify_rows must be positive, got ", options_.verify_rows);
    XS_CHECK(options_.sequences >= 1, "an engine of ", options_.sequences, " sequences");
    sequences_.resize(static_cast<size_t>(options_.sequences));
    fixed_state_ = model_->graph().has_fixed_state();
    const auto offers_argmax = [](const graph::Graph& g) {
        return std::any_of(g.outputs().begin(), g.outputs().end(), [](const auto& o) { return o.name == "argmax"; });
    };
    has_argmax_ = offers_argmax(model_->graph());
    if (options_.spec_k > 0 && options_.ngram) {
        const graph::Graph* batch = model_->batch_graph();
        XS_CHECK(batch != nullptr, "model ", model_->info().id, " was loaded without a batch graph, which verifies ",
                 "prompt lookup's drafts (LoadOptions::batch)");
        XS_CHECK(!options_.batch || batch->segmented(), "model ", model_->info().id, "'s batch graph takes no segments");
        for (Sequence& q : sequences_) q.drafter = std::make_unique<spec::NgramDrafter>(*options_.ngram, options_.spec_k);
        prefill_program_ = spec::kStep;
        verify_program_ = kBatchProgram;
        has_argmax_ = has_argmax_ && offers_argmax(*batch);
    } else if (options_.spec_k > 0) {
        const Speculation* spec = model_->speculation();
        XS_CHECK(spec != nullptr, "model ", model_->info().id, " has no drafter for speculative decoding");
        for (Sequence& q : sequences_) q.drafter = spec::make_drafter(*spec, *session_);
        const int most = sequences_[0].drafter->max_drafts();
        XS_CHECK(options_.spec_k <= most, "spec_k ", options_.spec_k, " is more than the ", most, " drafts a round of ",
                 spec->drafter_name(), " proposes");
        has_argmax_ = has_argmax_ && offers_argmax(spec->prefill_graph()) && offers_argmax(spec->verify_graph());
    }
    XS_CHECK(!options_.batch_draft || (options_.batch && options_.spec_k > 0 && !options_.ngram &&
                                       model_->speculation()->chain() != nullptr &&
                                       model_->speculation()->batch_draft_graph() != nullptr),
             "batched drafts need batched rounds of a chain drafter whose model offers its batch draft program");
    if (options_.batch && options_.spec_k > 0 && !options_.ngram) {
        XS_CHECK(model_->speculation()->verify_graph().segmented(), "model ", model_->info().id,
                 " was loaded without a verify graph of several requests (LoadOptions::batch)");
    } else if (options_.batch && options_.spec_k == 0) {
        const graph::Graph* batch = model_->batch_graph();
        XS_CHECK(batch != nullptr, "model ", model_->info().id, " was loaded without a batch graph");
        has_argmax_ = has_argmax_ && offers_argmax(*batch);
    }
}

Engine::~Engine() = default;

// A request in flight: its prompt as positions, where its prefill stands, and its decode state.
struct Engine::Generation {
    GenerateRequest request;
    TokenCallback on_token;
    GenerateResult result;
    bool done = false;
    bool holds_sequence = false;
    size_t sequence = 0;
    std::vector<PositionKey> keys;      // the prompt's positions
    std::vector<int32_t> media_item;    // a soft token's media item, -1 for text
    std::vector<int64_t> media_row;     // and its row of that item's encoder output
    std::vector<RotaryPosition> positions;
    std::vector<std::shared_ptr<const HostTensor>> encoded;  // by media item, once encoded
    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<TokenConstraint> constraint;  // the request's, when it has one
    bool argmax_only = false;
    size_t fed = 0;  // prompt positions the sequence holds
    bool decoding = false;
    int64_t max_new = 0;
    int32_t token = 0;           // the next token to emit
    std::deque<int32_t> ready;   // tokens a speculative round produced, not emitted yet
    // GenerateRequest::logprobs: those of the tokens produced and not emitted yet (the pending token and `ready`, and
    // a stop token or the last round's surplus never emitted), in the order they were produced and are emitted.
    std::deque<TokenLogprobs> logprobs;
    spec::Acceptance acceptance;  // of its drafts (EngineOptions::adaptive_k)
    Timer decode_timer;          // from the first emitted token
};

void Engine::select(size_t s) {
    if (s == current_) return;
    session_->use(static_cast<int64_t>(s));
    current_ = s;
}

bool Engine::has_free_sequence() const {
    return std::any_of(sequences_.begin(), sequences_.end(), [](const Sequence& q) { return !q.busy; });
}

size_t Engine::pick_sequence(const std::vector<PositionKey>& keys) const {
    size_t best = sequences_.size();
    int64_t best_common = -1;
    for (size_t q = 0; q < sequences_.size(); ++q) {
        if (sequences_[q].busy) continue;
        const int64_t common = usable_prefix(sequences_[q], keys);
        if (common > best_common || (common == best_common && sequences_[q].used < sequences_[best].used)) {
            best = q;
            best_common = common;
        }
    }
    XS_CHECK(best < sequences_.size(), "every sequence of the engine is busy");
    return best;
}

size_t Engine::state_program() const {
    return seq().drafter ? verify_program_ : spec::kStep;
}

void Engine::reset_sequence() {
    session_->reset();
    seq().cached.clear();
    seq().snapshot.reset();
    if (seq().drafter) seq().drafter->reset();
}

void Engine::set_prefill_chunk(int64_t tokens) {
    XS_CHECK(tokens > 0, "prefill chunk must be positive, got ", tokens);
    options_.prefill_chunk = tokens;
}

void Engine::run_target(size_t program, std::span<const int32_t> tokens, std::span<const PositionKey> keys,
                        std::span<const RotaryPosition> positions, std::span<const float* const> embeddings) {
    XS_CHECK(!tokens.empty() && keys.size() == tokens.size() && positions.size() == tokens.size(),
             "forward without tokens");
    std::vector<PositionKey>& cached = seq().cached;
    const auto start = static_cast<int64_t>(cached.size());
    XS_CHECK(start + static_cast<int64_t>(tokens.size()) <= options_.max_context, "sequence of ",
             start + static_cast<int64_t>(tokens.size()), " tokens exceeds the context limit of ", options_.max_context);
    inputs_.clear();
    model_->prepare_positioned_inputs(tokens, positions, embeddings, inputs_);
    std::vector<std::string> wanted{argmax_only_ ? "argmax" : "logits"};
    const std::unique_ptr<spec::Drafter>& drafter = seq().drafter;
    if (drafter) wanted.insert(wanted.end(), drafter->target_outputs().begin(), drafter->target_outputs().end());
    outputs_.clear();
    session_->run(program, inputs_, outputs_, wanted);
    cached.insert(cached.end(), keys.begin(), keys.end());
    XS_CHECK(session_->num_cached() == static_cast<int64_t>(cached.size()), "backend holds ", session_->num_cached(),
             " cached tokens, engine expected ", cached.size());
}

void Engine::forward(std::span<const int32_t> tokens, std::span<const PositionKey> keys,
                     std::span<const RotaryPosition> positions, std::span<const float* const> embeddings) {
    const auto start = static_cast<int64_t>(seq().cached.size());
    const std::unique_ptr<spec::Drafter>& drafter = seq().drafter;
    run_target(drafter ? prefill_program_ : spec::kStep, tokens, keys, positions, embeddings);
    if (drafter) {
        drafter->add(outputs_, static_cast<int64_t>(tokens.size()), start, tokens);
        drafter->flush();
    }
}

int32_t Engine::target_token(Generation& g, int64_t row, std::span<const int32_t> generated, const int32_t* draft) {
    const int64_t vocab = model_->info().vocab_size;
    if (argmax_only_) {
        XS_CHECK(g.constraint == nullptr, "a constrained request's step read its rows' argmax, not their logits");
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
    if (g.constraint) constrain(g, values);
    Sampler& sampler = *g.sampler;
    // Log-probabilities read the row as the constraint left it, before the sampler's penalties change it.
    const bool logprobs = g.request.logprobs >= 0;
    std::span<const float> row_seen = values;
    if (logprobs && sampler.penalizes()) {
        row_copy_.assign(values.begin(), values.end());
        row_seen = row_copy_;
    }
    const std::vector<int32_t>& prompt = g.request.prompt;
    const int32_t t = draft != nullptr ? sampler.verify(values, prompt, generated, *draft)
                                       : sampler.sample(values, prompt, generated);
    if (logprobs) g.logprobs.push_back(token_logprobs(row_seen, t, g.request.logprobs));
    const std::vector<int32_t>& stops = g.request.stop_token_ids;
    if (g.constraint && std::find(stops.begin(), stops.end(), t) == stops.end()) {
        Timer timer;
        g.constraint->accept(t);
        g.result.stats.constraint_ms += timer.elapsed_ms();
    }
    return t;
}

void Engine::constrain(Generation& g, std::span<float> logits) {
    Timer timer;
    const std::vector<int32_t>& stops = g.request.stop_token_ids;
    const auto in_vocab = [&](int32_t t) { return t >= 0 && static_cast<size_t>(t) < logits.size(); };
    stop_logits_.clear();
    for (const int32_t s : stops) stop_logits_.push_back(in_vocab(s) ? logits[static_cast<size_t>(s)] : 0.0f);
    const bool allowed = g.constraint->mask(logits);
    const bool end = g.constraint->complete();
    for (size_t i = 0; i < stops.size(); ++i) {
        if (in_vocab(stops[i])) {
            logits[static_cast<size_t>(stops[i])] = end ? stop_logits_[i] : -std::numeric_limits<float>::infinity();
        }
    }
    GenerateStats& stats = g.result.stats;
    stats.constraint_ms += timer.elapsed_ms();
    ++stats.constrained_rows;
    XS_CHECK(allowed || end, "the request's output constraint allows no token after its ", g.result.tokens.size(),
             " generated");
}

std::vector<int32_t> Engine::propose(Generation& g, int k, double& ms) {
    Timer timer;
    std::vector<int32_t> drafts = seq().drafter->propose(g.token, k);
    ms = timer.elapsed_ms();
    if (g.constraint) {
        Timer checking;
        drafts.resize(g.constraint->allowed_prefix(drafts));
        g.result.stats.constraint_ms += checking.elapsed_ms();
    }
    return drafts;
}

std::vector<std::vector<int32_t>> Engine::propose_together(std::span<Generation* const> gs, std::span<const int> ks,
                                                           std::vector<double>& ms) {
    Timer timer;
    std::vector<std::vector<int32_t>> drafts(gs.size());
    std::vector<spec::ChainDrafter*> drafter(gs.size(), nullptr);
    std::vector<size_t> active;  // the requests that draft at this depth
    for (size_t i = 0; i < gs.size(); ++i) {
        if (ks[i] <= 0) continue;
        drafter[i] = sequences_[gs[i]->sequence].drafter->as_chain();
        XS_CHECK(drafter[i] != nullptr, "batched drafts need chain drafters");
        active.push_back(i);
    }
    // One run of the batch draft program over the inputs `in` of `active`, `rows` each; each one's draft from the last
    // of its rows.
    const auto run = [&](std::vector<TensorMap>& in, const std::vector<int64_t>& rows, bool first) {
        std::map<std::string, std::vector<HostTensor>> parts;
        std::vector<Segment> segments;
        for (size_t j = 0; j < active.size(); ++j) {
            for (auto& [name, t] : in[j]) parts[name].push_back(std::move(t));
            segments.push_back({static_cast<int64_t>(gs[active[j]]->sequence), rows[j]});
        }
        TensorMap inputs, outputs;
        for (auto& [name, list] : parts) inputs[name] = concat_rows(list);
        session_->run_segments(spec::kBatchDraft, segments, inputs, outputs, {});
        int64_t row = 0;
        for (size_t j = 0; j < active.size(); ++j) {
            TensorMap own;
            for (const auto& [name, t] : outputs) own[name] = rows_of(t, row, rows[j]);
            spec::ChainDrafter& d = *drafter[active[j]];
            drafts[active[j]].push_back(first ? d.first(own, rows[j] - 1) : d.next(own, rows[j] - 1));
            row += rows[j];
        }
    };
    std::vector<TensorMap> in(active.size());
    std::vector<int64_t> rows(active.size());
    for (size_t j = 0; j < active.size(); ++j) {
        select(gs[active[j]]->sequence);
        rows[j] = drafter[active[j]]->begin(gs[active[j]]->token, in[j]);
    }
    run(in, rows, true);
    for (int depth = 1;; ++depth) {
        std::erase_if(active, [&](size_t i) { return ks[i] <= depth; });
        if (active.empty()) break;
        in.assign(active.size(), TensorMap{});
        rows.assign(active.size(), 1);
        for (size_t j = 0; j < active.size(); ++j) drafter[active[j]]->chained(depth, in[j]);
        run(in, rows, false);
    }
    size_t drafting = 0;
    for (size_t i = 0; i < gs.size(); ++i) {
        if (drafter[i] == nullptr) continue;
        select(gs[i]->sequence);
        drafter[i]->end();
        ++drafting;
    }
    const double each = drafting > 0 ? timer.elapsed_ms() / static_cast<double>(drafting) : 0.0;
    for (size_t i = 0; i < gs.size(); ++i) {
        if (drafter[i] == nullptr) continue;
        ms[i] = each;
        Generation& g = *gs[i];
        if (g.constraint) {
            Timer checking;
            drafts[i].resize(g.constraint->allowed_prefix(drafts[i]));
            g.result.stats.constraint_ms += checking.elapsed_ms();
        }
    }
    return drafts;
}

std::vector<int32_t> Engine::spec_round(Generation& g, int k) {
    std::vector<int32_t> fed{g.token};
    double draft_ms = 0.0;
    if (k > 0) {
        const std::vector<int32_t> drafts = propose(g, k, draft_ms);
        fed.insert(fed.end(), drafts.begin(), drafts.end());
    }
    std::vector<PositionKey> keys(fed.size());
    std::transform(fed.begin(), fed.end(), keys.begin(), text_key);
    const auto start = static_cast<int64_t>(seq().cached.size());
    Timer timer;
    run_target(verify_program_, fed, keys, text_positions(seq(), start, fed.size()), {});
    const double verify_ms = timer.elapsed_ms();
    GenerateStats& stats = g.result.stats;
    stats.spec_draft_ms += draft_ms;
    stats.spec_verify_ms += verify_ms;
    draft_length_.record(static_cast<int>(fed.size()) - 1, draft_ms, verify_ms);
    return accept_round(g, fed, 0, start, outputs_);
}

std::vector<int32_t> Engine::accept_round(Generation& g, std::span<const int32_t> fed, int64_t row, int64_t start,
                                          const TensorMap& target) {
    // Row i of the round is the target's distribution after fed[i]: it accepts the draft fed[i + 1] or replaces it.
    std::vector<int32_t> history(g.result.tokens.begin(), g.result.tokens.end());
    std::vector<int32_t> out;
    for (size_t i = 0; i < fed.size(); ++i) {
        const bool last = i + 1 == fed.size();
        const int32_t t = target_token(g, row + static_cast<int64_t>(i), history, last ? nullptr : &fed[i + 1]);
        out.push_back(t);
        history.push_back(t);
        if (!last && t != fed[i + 1]) break;
    }
    const auto accepted = static_cast<int64_t>(out.size()) - 1;
    const int64_t keep = start + accepted + 1;  // the token and the accepted drafts
    std::vector<PositionKey>& cached = seq().cached;
    if (keep < static_cast<int64_t>(cached.size())) {
        session_->truncate(state_program(), keep);
        cached.resize(static_cast<size_t>(keep));
    }
    seq().drafter->add(target, accepted + 1, start, fed);
    GenerateStats& stats = g.result.stats;
    ++stats.spec_rounds;
    stats.spec_drafted += static_cast<int>(fed.size()) - 1;
    stats.spec_accepted += static_cast<int>(accepted);
    g.acceptance.round(static_cast<int>(fed.size()) - 1, static_cast<int>(accepted));
    acceptance_.round(static_cast<int>(fed.size()) - 1, static_cast<int>(accepted));
    return out;
}

int Engine::round_drafts(const Generation& g) const {
    const Sequence& q = sequences_[g.sequence];
    const auto emitted = static_cast<int64_t>(g.result.tokens.size());
    // Drafts past the last token wanted, or past the context, are never verified.
    const int64_t fits = options_.max_context - static_cast<int64_t>(q.cached.size()) - 1;
    const auto limit = static_cast<int>(std::min<int64_t>({options_.spec_k, g.max_new - emitted - 1, fits}));
    return options_.adaptive_k && limit > 1 ? draft_length_.best(g.acceptance, limit) : limit;
}

int64_t Engine::usable_prefix(const Sequence& q, const std::vector<PositionKey>& keys) const {
    const std::vector<PositionKey>& cached = q.cached;
    const auto common = static_cast<int64_t>(
        std::mismatch(cached.begin(), cached.end(), keys.begin(), keys.end()).first - cached.begin());
    // At least the last prompt token must run, because its logits pick the first new token.
    const int64_t keep = std::min<int64_t>(common, static_cast<int64_t>(keys.size()) - 1);
    if (keep <= 0) return 0;
    if (keep == static_cast<int64_t>(cached.size()) || !fixed_state_) return keep;
    // Conv and recurrent state cannot be cut back into the sequence: only to its snapshot.
    return q.snapshot && q.snapshot->tokens <= keep ? q.snapshot->tokens : 0;
}

namespace {

// A copy costs the prefix's KV copied on the device and, with conv or recurrent state, a snapshot restored from host
// memory (tens of ms): worth it from this many tokens more than the request's own sequence keeps.
constexpr int64_t kCopyGain = 64;

}  // namespace

int64_t Engine::copyable_prefix(const Sequence& q, const std::vector<PositionKey>& keys) const {
    const std::vector<PositionKey>& cached = q.cached;
    const auto common = static_cast<int64_t>(
        std::mismatch(cached.begin(), cached.end(), keys.begin(), keys.end()).first - cached.begin());
    const int64_t keep = std::min<int64_t>(common, static_cast<int64_t>(keys.size()) - 1);
    if (!fixed_state_) return std::max<int64_t>(keep, 0);
    return q.snapshot && q.snapshot->tokens <= keep ? q.snapshot->tokens : 0;
}

void Engine::take_prefix(const std::vector<PositionKey>& keys) {
    size_t donor = current_;
    int64_t tokens = usable_prefix(seq(), keys) + kCopyGain - 1;
    for (size_t q = 0; q < sequences_.size(); ++q) {
        if (q == current_) continue;
        const int64_t n = copyable_prefix(sequences_[q], keys);
        if (n > tokens) {
            donor = q;
            tokens = n;
        }
    }
    if (donor == current_) return;
    Timer timer;
    reset_sequence();
    session_->copy_prefix(state_program(), static_cast<int64_t>(donor), tokens);
    if (fixed_state_) session_->restore_state(state_program(), *sequences_[donor].snapshot);
    seq().cached.assign(keys.begin(), keys.begin() + tokens);
    seq().snapshot = fixed_state_ ? sequences_[donor].snapshot : nullptr;
    if (seq().drafter) {
        std::vector<int32_t> copied(static_cast<size_t>(tokens));
        std::transform(keys.begin(), keys.begin() + tokens, copied.begin(),
                       [](PositionKey k) { return (k & kMediaKeyBit) != 0 ? -1 : static_cast<int32_t>(k); });
        seq().drafter->adopt(copied);
    }
    log::debug("engine: sequence ", current_, " took ", tokens, " tokens of sequence ", donor, "'s prefix in ",
               static_cast<int>(timer.elapsed_ms()), " ms");
}

int64_t Engine::reuse_prefix(const std::vector<PositionKey>& prompt_keys) {
    std::vector<PositionKey>& cached = seq().cached;
    const int64_t keep = usable_prefix(seq(), prompt_keys);
    if (keep == 0) {
        reset_sequence();
        return 0;
    }
    if (keep < static_cast<int64_t>(cached.size())) {
        if (fixed_state_) {
            Timer restoring;
            session_->restore_state(state_program(), *seq().snapshot);
            log::debug("engine: restored sequence ", current_, "'s state at ", keep, " tokens in ",
                       static_cast<int>(restoring.elapsed_ms()), " ms");
        } else {
            session_->truncate(state_program(), keep);
        }
        cached.resize(static_cast<size_t>(keep));
        if (seq().drafter) seq().drafter->truncate(keep);
    }
    return keep;
}

std::vector<RotaryPosition> Engine::text_positions(const Sequence& q, int64_t index, size_t count) {
    std::vector<RotaryPosition> out(count);
    for (size_t i = 0; i < count; ++i) {
        const auto p = static_cast<int32_t>(index + static_cast<int64_t>(i) + q.rotary_delta);
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

std::optional<Engine::Handle> Engine::start(const GenerateRequest& request, TokenCallback on_token) {
    if (!has_free_sequence()) return std::nullopt;
    auto owned = std::make_unique<Generation>();
    Generation& g = *owned;
    g.request = request;
    g.on_token = std::move(on_token);
    const Handle h = next_handle_++;
    active_.emplace(h, std::move(owned));
    GenerateResult& result = g.result;
    const auto& prompt = g.request.prompt;
    if (prompt.empty()) {
        result.finish = FinishReason::Error;
        result.error = "empty prompt";
        g.done = true;
        return h;
    }
    if (static_cast<int64_t>(prompt.size()) >= options_.max_context) {
        result.finish = FinishReason::Error;
        result.error = str_cat("prompt of ", prompt.size(), " tokens does not fit the context limit of ",
                               options_.max_context, " tokens");
        g.done = true;
        return h;
    }
    if (g.request.logprobs > model_->info().vocab_size) {
        result.finish = FinishReason::Error;
        result.error = str_cat("log-probabilities of ", g.request.logprobs, " tokens a position, of a vocabulary of ",
                               model_->info().vocab_size);
        g.done = true;
        return h;
    }
    if (g.request.constraint && g.request.stop_token_ids.empty()) {
        result.finish = FinishReason::Error;
        result.error = "a constrained request needs a stop token to end on";
        g.done = true;
        return h;
    }
    result.stats.prompt_tokens = static_cast<int>(prompt.size());
    try {
        // Soft-token positions take the encoder rows of the request's media items, in order.
        const auto n = prompt.size();
        g.keys.assign(n, 0);
        g.media_item.assign(n, -1);
        g.media_row.assign(n, 0);
        for (const MediaInput& m : g.request.media) {
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
        const int64_t needed = std::accumulate(g.request.media.begin(), g.request.media.end(), int64_t{0},
                                               [](int64_t sum, const MediaInput& m) { return sum + m.num_tokens; });
        int64_t found = 0;
        for (size_t i = 0; i < n; ++i) found += soft_modality(prompt[i]) >= 0;
        XS_CHECK(found == needed, "the prompt holds ", found, " media tokens but its ", g.request.media.size(),
                 " media items need ", needed);
        size_t item = 0;
        int64_t row = 0;
        for (size_t i = 0; i < n; ++i) {
            const int k = soft_modality(prompt[i]);
            if (k < 0) {
                g.keys[i] = text_key(prompt[i]);
                continue;
            }
            while (row == g.request.media[item].num_tokens) {
                ++item;
                row = 0;
            }
            XS_CHECK(static_cast<int>(g.request.media[item].modality) == k, "the prompt's ",
                     modality_name(static_cast<Modality>(k)), " tokens at position ", i, " fall to media item ", item,
                     ", which is ", modality_name(g.request.media[item].modality));
            g.media_item[i] = static_cast<int32_t>(item);
            g.media_row[i] = row;
            g.keys[i] = media_key(g.request.media[item].key, row++);
        }
        PromptPositions rotary = rotary_positions(g.media_item, g.media_row, g.request.media);
        g.positions = std::move(rotary.positions);

        g.sequence = pick_sequence(g.keys);
        sequences_[g.sequence].busy = true;
        sequences_[g.sequence].used = ++clock_;
        g.holds_sequence = true;
        select(g.sequence);
        seq().rotary_delta = rotary.next - static_cast<int64_t>(n);
        g.sampler = std::make_unique<Sampler>(g.request.sampling);
        g.acceptance = acceptance_;  // what the requests before it showed, until its own rounds show otherwise
        if (g.request.constraint) g.constraint = g.request.constraint->start();
        g.argmax_only = has_argmax_ && !g.sampler->needs_logits() && !g.constraint && g.request.logprobs < 0;
        argmax_only_ = g.argmax_only;
        Timer timer;
        take_prefix(g.keys);
        const int64_t cached = reuse_prefix(g.keys);
        result.stats.cached_tokens = static_cast<int>(cached);
        result.stats.prefill_ms += timer.elapsed_ms();
        g.fed = static_cast<size_t>(cached);
        g.encoded.resize(g.request.media.size());
    } catch (const std::exception& e) {
        fail_generation(g, e);
    }
    return h;
}

bool Engine::advance(Handle h) {
    const auto it = active_.find(h);
    XS_CHECK(it != active_.end(), "no request ", h, " in flight");
    Generation& g = *it->second;
    if (g.done) return true;
    try {
        select(g.sequence);
        argmax_only_ = g.argmax_only;
        if (!g.decoding) {
            prefill_chunk(g);
        } else {
            decode_step(g);
        }
    } catch (const std::exception& e) {
        fail_generation(g, e);
    }
    release(g);
    return g.done;
}

bool Engine::decoding(Handle h) const {
    const auto it = active_.find(h);
    XS_CHECK(it != active_.end(), "no request ", h, " in flight");
    return it->second->decoding;
}

GenerateResult Engine::finish(Handle h) {
    const auto it = active_.find(h);
    XS_CHECK(it != active_.end() && it->second->done, "request ", h, " is not done");
    GenerateResult result = std::move(it->second->result);
    result.stats.generated_tokens = static_cast<int>(result.tokens.size());
    if (options_.adaptive_k && result.stats.spec_rounds > 0) {
        const spec::Acceptance& a = it->second->acceptance;
        std::string rates, costs;
        for (int k = 1; k <= options_.spec_k; ++k) {
            rates += str_cat(" ", static_cast<int>(a.rate(k) * 100));
            costs += str_cat(" ", static_cast<int>(draft_length_.cost(k) * 10) / 10.0);
        }
        log::debug("engine: adaptive drafts, ", result.stats.spec_drafted, " in ", result.stats.spec_rounds,
                   " rounds; acceptance % by draft", rates, "; round ms by drafts", costs);
    }
    active_.erase(it);
    return result;
}

GenerateResult Engine::generate(const GenerateRequest& request, const TokenCallback& on_token) {
    const std::optional<Handle> h = start(request, on_token);
    XS_CHECK(h.has_value(), "every sequence of the engine is busy");
    while (!advance(*h)) {
    }
    return finish(*h);
}

void Engine::fail_generation(Generation& g, const std::exception& e) {
    // The KV state may be partially written; the sequence's next request starts from scratch.
    if (g.holds_sequence) {
        select(g.sequence);
        reset_sequence();
    }
    g.result.finish = FinishReason::Error;
    g.result.error = e.what();
    g.done = true;
    release(g);  // advance() returns at once for a request that is done: one that failed in start() too
    log::error("generation failed: ", e.what());
}

void Engine::prefill_chunk(Generation& g) {
    GenerateStats& stats = g.result.stats;
    Timer timer;
    const double encode_before = stats.encode_ms;
    const std::vector<int32_t>& prompt = g.request.prompt;
    const size_t n = prompt.size();
    const size_t pos = g.fed;
    size_t len = std::min(n - pos, static_cast<size_t>(options_.prefill_chunk));
    const auto snapshot_at = static_cast<size_t>(std::max<int64_t>(0, g.request.snapshot_at));
    if (fixed_state_ && snapshot_at > pos && snapshot_at < pos + len) len = snapshot_at - pos;  // a chunk ends there
    if (model_->info().media_whole_steps) len = whole_media_chunk(g.media_item, pos, len);
    const int64_t width = media_width();
    std::vector<const float*> rows(len, nullptr);
    bool any_rows = false;
    for (size_t j = 0; j < len; ++j) {
        const int32_t m = g.media_item[pos + j];
        if (m < 0) continue;
        auto& e = g.encoded[static_cast<size_t>(m)];
        if (!e) e = encode(g.request.media[static_cast<size_t>(m)], stats);
        rows[j] = e->data<float>() + g.media_row[pos + j] * width;
        any_rows = true;
    }
    forward(std::span<const int32_t>(prompt).subspan(pos, len), std::span<const PositionKey>(g.keys).subspan(pos, len),
            std::span<const RotaryPosition>(g.positions).subspan(pos, len),
            any_rows ? std::span<const float* const>(rows) : std::span<const float* const>());
    g.fed += len;
    if (fixed_state_ && g.fed == snapshot_at) {
        Timer saving;
        seq().snapshot = std::make_shared<const StateSnapshot>(session_->save_state(state_program()));
        log::debug("engine: saved sequence ", g.sequence, "'s state at ", snapshot_at, " tokens (",
                   seq().snapshot->bytes() >> 20, " MiB) in ", static_cast<int>(saving.elapsed_ms()), " ms");
    }
    stats.prefill_ms += timer.elapsed_ms() - (stats.encode_ms - encode_before);
    if (g.fed == n) begin_decode(g);  // the last row's logits give the first token: before another request runs
}

void Engine::begin_decode(Generation& g) {
    const int64_t room = options_.max_context - static_cast<int64_t>(g.request.prompt.size());
    g.max_new = std::min<int64_t>(g.request.max_new_tokens, room);
    g.result.finish = FinishReason::Length;
    g.decoding = true;
    if (g.max_new <= 0) {
        g.done = true;
        return;
    }
    g.token = target_token(g, 0, g.result.tokens, nullptr);
}

bool Engine::emit(Generation& g) {
    GenerateResult& result = g.result;
    const std::vector<int32_t>& stops = g.request.stop_token_ids;
    if (std::find(stops.begin(), stops.end(), g.token) != stops.end()) {
        result.finish = FinishReason::Stop;
        g.done = true;
        return false;
    }
    // Decode speed is measured between the first and the last emitted token.
    if (result.tokens.empty()) g.decode_timer.reset();
    result.tokens.push_back(g.token);
    result.stats.decode_ms = g.decode_timer.elapsed_ms();
    std::optional<TokenLogprobs> logprobs;
    if (g.request.logprobs >= 0) {
        XS_CHECK(!g.logprobs.empty(), "no log-probabilities for token ", g.token, " of a request that asked for them");
        logprobs = std::move(g.logprobs.front());
        g.logprobs.pop_front();
    }
    if (g.on_token && !g.on_token(g.token, logprobs ? &*logprobs : nullptr)) {
        result.finish = FinishReason::Cancelled;
        g.done = true;
        return false;
    }
    if (static_cast<int64_t>(result.tokens.size()) >= g.max_new) {
        g.done = true;
        return false;
    }
    return true;
}

void Engine::decode_step(Generation& g) {
    if (!emit(g)) return;
    if (g.ready.empty() && seq().drafter) {
        const std::vector<int32_t> round = spec_round(g, round_drafts(g));
        g.ready.assign(round.begin(), round.end());
    } else if (g.ready.empty()) {
        const int32_t fed[] = {g.token};
        const PositionKey fed_key[] = {text_key(g.token)};
        forward(fed, fed_key, text_positions(seq(), static_cast<int64_t>(seq().cached.size()), 1), {});
        g.ready.push_back(target_token(g, 0, g.result.tokens, nullptr));
    }
    g.token = g.ready.front();
    g.ready.pop_front();
}

std::vector<bool> Engine::advance_decoding(std::span<const Handle> hs) {
    std::vector<bool> done;
    if (!options_.batch) {
        for (const Handle h : hs) done.push_back(advance(h));
        return done;
    }
    std::vector<Generation*> gs, step;
    for (const Handle h : hs) {
        const auto it = active_.find(h);
        XS_CHECK(it != active_.end() && it->second->decoding, "request ", h, " is not decoding");
        gs.push_back(it->second.get());
    }
    for (Generation* g : gs) {
        if (g->done) continue;
        try {
            // The pending token, and every token the request's last round gave, so that all requests run their
            // next rounds in this step: the last emitted is the round's pending token.
            bool goes_on = emit(*g);
            while (goes_on && !g->ready.empty()) {
                g->token = g->ready.front();
                g->ready.pop_front();
                goes_on = emit(*g);
            }
            if (goes_on) step.push_back(g);
        } catch (const std::exception& e) {
            fail_generation(*g, e);
        }
    }
    if (!step.empty()) {
        try {
            decode_together(step);
        } catch (const std::exception& e) {
            for (Generation* g : step) fail_generation(*g, e);
        }
    }
    for (Generation* g : gs) {
        release(*g);
        done.push_back(g->done);
    }
    return done;
}

void Engine::decode_together(std::span<Generation* const> gs) {
    const bool speculating = options_.spec_k > 0;
    // Each request's segment: its pending token, and speculating the drafts its drafter proposes after it, within
    // its share of the step's rows.
    const auto share = static_cast<int>(
        gs.size() == 1 ? options_.spec_k : options_.verify_rows / static_cast<int64_t>(gs.size()) - 1);
    std::vector<std::vector<int32_t>> fed(gs.size());
    std::vector<double> draft_ms(gs.size(), 0.0);
    std::vector<Segment> segments;
    std::map<std::string, std::vector<HostTensor>> rows;
    bool logits = false;
    std::vector<int> ks(gs.size(), 0);
    for (size_t i = 0; i < gs.size() && speculating; ++i) ks[i] = std::min(round_drafts(*gs[i]), share);
    std::vector<std::vector<int32_t>> together;  // the drafts of all, one run a depth
    if (options_.batch_draft && std::count_if(ks.begin(), ks.end(), [](int k) { return k > 0; }) > 1) {
        together = propose_together(gs, ks, draft_ms);
    }
    for (size_t i = 0; i < gs.size(); ++i) {
        Generation& g = *gs[i];
        select(g.sequence);
        fed[i] = {g.token};
        if (!together.empty()) {
            fed[i].insert(fed[i].end(), together[i].begin(), together[i].end());
        } else if (ks[i] > 0) {
            const std::vector<int32_t> drafts = propose(g, ks[i], draft_ms[i]);
            fed[i].insert(fed[i].end(), drafts.begin(), drafts.end());
        }
        const auto at = static_cast<int64_t>(seq().cached.size());
        const auto n = static_cast<int64_t>(fed[i].size());
        XS_CHECK(at + n <= options_.max_context, "sequence of ", at + n, " tokens exceeds the context limit of ",
                 options_.max_context);
        TensorMap in;
        model_->prepare_positioned_inputs(fed[i], text_positions(seq(), at, fed[i].size()), {}, in);
        for (auto& [name, t] : in) rows[name].push_back(std::move(t));
        segments.push_back({static_cast<int64_t>(g.sequence), n});
        logits = logits || !g.argmax_only;
    }
    inputs_.clear();
    for (auto& [name, list] : rows) inputs_[name] = concat_rows(list);
    outputs_.clear();
    std::vector<std::string> wanted{logits ? "logits" : "argmax"};
    std::span<const std::string> drafter_outputs;
    if (speculating) {
        drafter_outputs = sequences_[gs[0]->sequence].drafter->target_outputs();
        wanted.insert(wanted.end(), drafter_outputs.begin(), drafter_outputs.end());
    }
    Timer timer;
    session_->run_segments(speculating ? verify_program_ : kBatchProgram, segments, inputs_, outputs_, wanted);
    const double verify_ms = timer.elapsed_ms();
    argmax_only_ = !logits;
    int64_t row = 0;
    for (size_t i = 0; i < gs.size(); ++i) {
        Generation& g = *gs[i];
        select(g.sequence);
        std::vector<PositionKey>& cached = seq().cached;
        const auto start = static_cast<int64_t>(cached.size());
        std::transform(fed[i].begin(), fed[i].end(), std::back_inserter(cached), text_key);
        XS_CHECK(session_->num_cached() == static_cast<int64_t>(cached.size()), "backend holds ",
                 session_->num_cached(), " cached tokens of sequence ", g.sequence, ", engine expected ",
                 cached.size());
        const auto n = static_cast<int64_t>(fed[i].size());
        if (!speculating) {
            g.token = target_token(g, row, g.result.tokens, nullptr);
        } else {
            TensorMap target;  // the drafter's outputs of this request's rows
            for (const std::string& name : drafter_outputs) target[name] = rows_of(outputs_.at(name), row, n);
            GenerateStats& stats = g.result.stats;
            stats.spec_draft_ms += draft_ms[i];
            stats.spec_verify_ms += verify_ms;
            const std::vector<int32_t> round = accept_round(g, fed[i], row, start, target);
            g.ready.assign(round.begin() + 1, round.end());
            g.token = round.front();
        }
        row += n;
    }
}

void Engine::release(Generation& g) {
    if (g.done && g.holds_sequence) {
        sequences_[g.sequence].busy = false;
        g.holds_sequence = false;
    }
}

}  // namespace xs
