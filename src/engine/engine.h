#pragma once

#include <array>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "backend/backend.h"
#include "engine/spec/draft_length.h"
#include "engine/spec/drafter.h"
#include "engine/spec/ngram_drafter.h"
#include "engine/types.h"
#include "models/model.h"

namespace xs {

class Sampler;

struct EngineOptions {
    int64_t max_context = 8192;  // tokens per sequence (prompt + generated)
    int64_t prefill_chunk = 1024;
    size_t encoded_media_cache = 16;  // media items whose encoder output stays cached (LRU)
    // Speculative decoding with the model's drafter: drafts per round, 0 = off. The session must hold the programs
    // of spec::Program (the model's Speculation) and rewind steps of spec_k + 1 tokens.
    int spec_k = 0;
    // Speculating by prompt lookup (engine/spec/ngram_drafter.h) instead of with the model's drafter, which the model
    // then needs not have: the session's programs are the step graph, which also prefills, and the batch graph
    // (kBatchProgram, Model::batch_graph), which verifies.
    std::optional<spec::NgramOptions> ngram;
    // The drafts of each round chosen between 1 and spec_k by what rounds yield and cost (spec::DraftLength): the
    // request's acceptance so far, and the time rounds of each count took on this device. Off: spec_k every round.
    bool adaptive_k = false;
    // Requests in flight at once: the session's sequences (SessionOptions::sequences), each its own cached prompt.
    int64_t sequences = 1;
    // The decode steps of the requests in flight run as one (advance_decoding): the session's program kBatchProgram
    // is the model's batch_graph(); speculating, each request drafts and the verifies of all run as one step of the
    // verify graph, which then allows segments (LoadOptions::batch).
    bool batch = false;
    // Speculating in batched steps: the rows a verify of several requests aims at, which they share (each drafts up to
    // verify_rows / requests - 1, none when more requests decode than that). A step of up to 8 rows costs about one
    // row's on a discrete GPU (each weight read once); past it the rows add up, a mixture of experts' most.
    int64_t verify_rows = 8;
};

// The session's program that runs Model::batch_graph() (EngineOptions::batch), after the step graph.
inline constexpr size_t kBatchProgram = 1;

// The stateless sessions of a model's media encoders, by modality: null where the model has none (or it is not
// loaded).
using EncoderSessions = std::array<std::unique_ptr<Session>, kModalities>;

// Runs generations on one backend session of EngineOptions::sequences sequences. Not thread-safe: one caller (the
// engine thread) owns it. A request runs on a free sequence, the one whose cached prompt it shares the longest prefix
// with (a media item's soft tokens count as shared only when the item's content is the same), and reuses that KV;
// the steps of the requests in flight interleave as the caller advances them, so a long prefill gives way to the
// others at the end of each of its chunks; with EngineOptions::batch the decode steps of all that decode run as one.
class Engine {
public:
    using Handle = int64_t;  // a request in flight

    Engine(std::unique_ptr<Model> model, std::unique_ptr<Session> session, EncoderSessions encoders,
           EngineOptions options);
    ~Engine();

    // The request on a free sequence; nothing when every sequence is busy. `on_token` runs on the caller's thread,
    // in advance(), for every token generated (stop tokens are not passed); false stops the request (Cancelled).
    std::optional<Handle> start(const GenerateRequest& request, TokenCallback on_token);
    // One step of the request: a prefill chunk (with its first token once the prompt is in), or its next token (or a
    // speculative round's). True when it is done: finish() then gives its result, and its sequence is free.
    bool advance(Handle h);
    // A step of each of `hs`, requests past their prompts (decoding): with EngineOptions::batch, the next tokens of all
    // that go on run as one step (speculating, each first emits every token its last round gave, then all run their
    // next rounds together); otherwise one request after another. Whether each is done, in order.
    std::vector<bool> advance_decoding(std::span<const Handle> hs);
    GenerateResult finish(Handle h);
    bool has_free_sequence() const;
    // Whether the request is past its prompt (its steps are decode tokens).
    bool decoding(Handle h) const;
    // One request alone: start it, advance it until it is done.
    GenerateResult generate(const GenerateRequest& request, const TokenCallback& on_token);

    const Model& model() const { return *model_; }
    const EngineOptions& options() const { return options_; }
    void set_prefill_chunk(int64_t tokens);
    bool accepts(Modality m) const { return encoders_[static_cast<size_t>(m)] != nullptr; }

private:
    // What a cached position holds: the token id, or for a media item's soft token the item's hash and the row.
    using PositionKey = uint64_t;
    struct Generation;
    // What a sequence of the session holds between requests.
    struct Sequence {
        std::vector<PositionKey> cached;          // positions whose KV the session holds, in order
        // The state at the first snapshot->tokens of `cached` (GenerateRequest::snapshot_at), for a model whose conv
        // or recurrent state cannot be cut back; shared by the sequences that copied that prefix (take_prefix).
        std::shared_ptr<const StateSnapshot> snapshot;
        std::unique_ptr<spec::Drafter> drafter;   // speculating
        int64_t rotary_delta = 0;                 // a token past the prompt at cached position i is at i + delta
        bool busy = false;
        uint64_t used = 0;                        // when it last ran a request (the oldest free one is taken first)
    };
    // Makes sequence `s` the one the session's runs act on and seq() is.
    void select(size_t s);
    Sequence& seq() { return sequences_[current_]; }
    const Sequence& seq() const { return sequences_[current_]; }
    // The sequence a request with these prompt positions runs on: the free one with the longest usable prefix.
    size_t pick_sequence(const std::vector<PositionKey>& keys) const;
    // The prompt positions a request with `keys` can keep of sequence `q`'s cache (reuse_prefix).
    int64_t usable_prefix(const Sequence& q, const std::vector<PositionKey>& keys) const;
    // The prompt positions a request with `keys` could copy of another sequence `q`'s cache: its common prefix, or for
    // a model with conv or recurrent state, the prefix up to its snapshot.
    int64_t copyable_prefix(const Sequence& q, const std::vector<PositionKey>& keys) const;
    // Makes the selected sequence hold another sequence's prefix of the prompt when that one offers kCopyGain tokens
    // more than its own does (Session::copy_prefix, and that sequence's snapshot for conv or recurrent state).
    void take_prefix(const std::vector<PositionKey>& keys);
    // The steps of advance(): the rest of the prompt's next chunk, and the next token.
    void prefill_chunk(Generation& g);
    void begin_decode(Generation& g);
    void decode_step(Generation& g);
    // Emits the request's pending token: false when that ends it (a stop token, a cancel, its last token).
    bool emit(Generation& g);
    // The next tokens of each of `gs` (decoding, nothing ready), their steps as one (EngineOptions::batch): a token
    // each through the batch graph, or speculating, a round each (its drafts, one verify of all, its acceptance).
    void decode_together(std::span<Generation* const> gs);
    // The drafts a round of `g` asks for: up to spec_k, none past its last token or past the context.
    int round_drafts(const Generation& g) const;
    // Frees the sequence of a request that is done.
    void release(Generation& g);
    void fail_generation(Generation& g, const std::exception& e);

    // Runs a target program over `tokens` (appended after the cached ones); the outputs the round reads are in
    // outputs_: each row's logits, or with argmax_only_ its argmax alone, and the hidden rows a drafter takes.
    void run_target(size_t program, std::span<const int32_t> tokens, std::span<const PositionKey> keys,
                    std::span<const RotaryPosition> positions, std::span<const float* const> embeddings);
    // Feeds `tokens` (appended after the cached ones); target_token(sampler, 0, ...) is then the token after them.
    void forward(std::span<const int32_t> tokens, std::span<const PositionKey> keys,
                 std::span<const RotaryPosition> positions, std::span<const float* const> embeddings);
    // The rotary positions of `count` text tokens from cached position `index` of sequence `q` on, past the prompt.
    static std::vector<RotaryPosition> text_positions(const Sequence& q, int64_t index, size_t count);
    // Width of the rows a media item's encoder runs give (ModelInfo::media_width, or the hidden size).
    int64_t media_width() const;
    // The token of `g` after row `row` of the last run, its output so far `generated` (a round's tokens included):
    // sampled from the row's logits as its constraint allows them (checked against `draft` when there is one,
    // Sampler::verify), or the row's argmax when that is all its sampler reads (argmax_only_). The constraint accepts
    // the token.
    int32_t target_token(Generation& g, int64_t row, std::span<const int32_t> generated, const int32_t* draft);
    // The row's logits as `g`'s constraint allows them: its mask, and the stop tokens only where the output may end.
    void constrain(Generation& g, std::span<float> logits);
    // The drafts of a round of `g` after its pending token: up to `k`, none from the first its constraint refuses (a
    // target row masked by it never takes that draft), and the time drafting took.
    std::vector<int32_t> propose(Generation& g, int k, double& ms);
    // One speculative round of `g` after its pending token (emitted, not yet cached): `k` drafts, one verify of them
    // all, acceptance and rollback. Returns the accepted drafts and the target's next token.
    std::vector<int32_t> spec_round(Generation& g, int k);
    // The end of a round of `g`'s sequence (selected), which ran `fed` (the pending token and its drafts) from cached
    // position `start` as rows `row`, `row` + 1, ... of the last run: the target's token after each row while the
    // drafts agree, the sequence cut back past the last draft kept, and the kept rows handed to the drafter (`target`:
    // its outputs of the round's rows, from row 0). Returns the accepted drafts and the target's next token.
    std::vector<int32_t> accept_round(Generation& g, std::span<const int32_t> fed, int64_t row, int64_t start,
                                      const TensorMap& target);
    // The program whose state slots are the sequence's: the step graph's, and speculating the verify graph's, which
    // also writes what a block drafter reads of the target.
    size_t state_program() const;
    // Drops the sequence: every slot of the session, and what waits for the drafter.
    void reset_sequence();
    // Makes the session hold a prefix of the prompt and returns how many prompt positions are already cached: the
    // longest one it shares, or with conv or recurrent state the whole cache when the prompt extends it, else the
    // sequence's snapshot when the prompt shares its tokens, else none.
    int64_t reuse_prefix(const std::vector<PositionKey>& prompt_keys);
    // The encoder rows of one media item ([num_tokens, hidden] f32), from the cache or its encoder's runs.
    std::shared_ptr<const HostTensor> encode(const MediaInput& media, GenerateStats& stats);

    std::unique_ptr<Model> model_;
    std::unique_ptr<Session> session_;
    EncoderSessions encoders_;
    EngineOptions options_;
    std::vector<Sequence> sequences_;
    size_t current_ = 0;  // the selected sequence (select)
    std::map<Handle, std::unique_ptr<Generation>> active_;
    Handle next_handle_ = 0;
    uint64_t clock_ = 0;
    std::list<std::pair<uint64_t, std::shared_ptr<const HostTensor>>> encoded_;  // most recent first
    TensorMap inputs_;
    TensorMap outputs_;
    std::vector<float> stop_logits_;  // constrain()'s scratch
    // The model has conv or recurrent state: a prefix is kept whole or from the sequence's snapshot, taken at
    // GenerateRequest::snapshot_at.
    bool fixed_state_ = false;
    // The session's programs that prefill and verify when speculating: the model's drafter's (spec::Program), or the
    // step and batch graphs for prompt lookup.
    size_t prefill_program_ = spec::kPrefill;
    size_t verify_program_ = spec::kVerify;
    // EngineOptions::adaptive_k: what rounds of each count of drafts cost, and the acceptance every request's rounds
    // showed, which a new request starts from.
    spec::DraftLength draft_length_{1};
    spec::Acceptance acceptance_;
    bool has_argmax_ = false;   // every target program outputs "argmax" beside "logits"
    bool argmax_only_ = false;  // the running request's sampler reads no more than a row's argmax, and programs give it
};

}  // namespace xs
