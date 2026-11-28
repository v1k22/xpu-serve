#include "server/grammar_constraint.h"

#include <algorithm>
#include <limits>
#include <optional>

#include "grammar/matcher.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"

namespace xs::server {
namespace {

constexpr float kMasked = -std::numeric_limits<float>::infinity();

// Every logit but those of `allowed` set to -infinity; past the vocabulary too (a head padded beyond it).
void keep_only(const grammar::TokenSet& allowed, std::span<float> logits) {
    const size_t n = std::min(logits.size(), static_cast<size_t>(allowed.size()));
    const std::vector<uint64_t>& words = allowed.words();
    for (size_t w = 0; w * 64 < n; ++w) {
        const uint64_t bits = words[w];
        const size_t first = w * 64, end = std::min(n, first + 64);
        if (bits == ~uint64_t{0} && end == first + 64) continue;
        if (bits == 0) {
            std::fill(logits.data() + first, logits.data() + end, kMasked);
            continue;
        }
        for (size_t i = first; i < end; ++i) {
            if (!((bits >> (i - first)) & 1)) logits[i] = kMasked;
        }
    }
    std::fill(logits.data() + n, logits.data() + logits.size(), kMasked);
}

// Where a generation's output stands: still reasoning (the last bytes it wrote, where the marker may be starting),
// writing the opening, or inside the grammar.
struct Position {
    enum class Stage : uint8_t { reasoning, opening, grammar };
    Stage stage = Stage::grammar;
    std::string tail;
    size_t opened = 0;
    grammar::Matcher matcher;
};

class Answer final : public TokenConstraint {
public:
    Answer(std::shared_ptr<const GrammarConstraint::Spec> spec, Position at)
        : spec_(std::move(spec)), at_(std::move(at)) {}

    bool mask(std::span<float> logits) override {
        switch (at_.stage) {
            case Position::Stage::reasoning:
                return true;
            case Position::Stage::opening: {
                const auto next = static_cast<size_t>(spec_->opening[at_.opened]);
                const float kept = next < logits.size() ? logits[next] : kMasked;
                std::fill(logits.begin(), logits.end(), kMasked);
                if (next < logits.size()) logits[next] = kept;
                return next < logits.size();
            }
            case Position::Stage::grammar: {
                const grammar::TokenSet& allowed = at_.matcher.allowed();
                keep_only(allowed, logits);
                return allowed.count() > 0;
            }
        }
        return false;
    }

    void accept(int32_t token) override {
        XS_CHECK(allows(at_, token), "response_format: token ", token, " is not what the format allows next");
        advance(at_, token);
    }

    bool complete() const override { return at_.stage == Position::Stage::grammar && at_.matcher.complete(); }

    size_t allowed_prefix(std::span<const int32_t> tokens) const override {
        Position p = at_;
        size_t n = 0;
        for (; n < tokens.size() && allows(p, tokens[n]); ++n) advance(p, tokens[n]);
        return n;
    }

private:
    bool allows(const Position& p, int32_t token) const {
        switch (p.stage) {
            case Position::Stage::reasoning:
                return true;
            case Position::Stage::opening:
                return token == spec_->opening[p.opened];
            case Position::Stage::grammar:
                return p.matcher.allows(token);
        }
        return false;
    }

    void advance(Position& p, int32_t token) const {
        const grammar::Vocabulary& v = *spec_->vocabulary;
        switch (p.stage) {
            case Position::Stage::reasoning: {
                if (token >= 0 && token < v.size()) p.tail += v.bytes(token);
                const std::string& marker = spec_->reasoning_end;
                const size_t at = p.tail.find(marker);
                if (at == std::string::npos) {
                    if (p.tail.size() >= marker.size()) p.tail.erase(0, p.tail.size() - marker.size() + 1);
                    return;
                }
                const std::string rest = p.tail.substr(at + marker.size());  // what the token wrote after the marker
                p.tail.clear();
                p.stage = spec_->opening.empty() ? Position::Stage::grammar : Position::Stage::opening;
                if (p.stage == Position::Stage::grammar) p.matcher.accept_bytes(rest);
                return;
            }
            case Position::Stage::opening:
                if (++p.opened == spec_->opening.size()) p.stage = Position::Stage::grammar;
                return;
            case Position::Stage::grammar:
                p.matcher.accept_token(token);
                return;
        }
    }

    std::shared_ptr<const GrammarConstraint::Spec> spec_;
    Position at_;
};

// Free, or inside a call since its opener (the matcher of the call's grammar). One automaton for all of a generation's
// calls: each starts from its start state.
class AfterOpener final : public TokenConstraint {
public:
    AfterOpener(int32_t opener, std::shared_ptr<grammar::Automaton> automaton)
        : opener_(opener), automaton_(std::move(automaton)) {}

    bool mask(std::span<float> logits) override {
        if (!call_) return true;
        const grammar::TokenSet& allowed = call_->allowed();
        keep_only(allowed, logits);
        return allowed.count() > 0;
    }
    void accept(int32_t token) override {
        XS_CHECK(allows(call_, token), "tool call: token ", token, " is not what the call's grammar allows next");
        advance(call_, token);
    }
    bool complete() const override { return !call_; }
    size_t allowed_prefix(std::span<const int32_t> tokens) const override {
        std::optional<grammar::Matcher> at = call_;
        size_t n = 0;
        for (; n < tokens.size() && allows(at, tokens[n]); ++n) advance(at, tokens[n]);
        return n;
    }

private:
    static bool allows(const std::optional<grammar::Matcher>& at, int32_t token) {
        return !at || at->allows(token);
    }
    void advance(std::optional<grammar::Matcher>& at, int32_t token) const {
        if (!at) {
            if (token == opener_) at.emplace(automaton_);
            return;
        }
        at->accept_token(token);
        if (at->complete()) at.reset();  // the call's closer: free again
    }

    int32_t opener_;
    std::shared_ptr<grammar::Automaton> automaton_;
    std::optional<grammar::Matcher> call_;  // inside a call
};

}  // namespace

OpenerConstraint::OpenerConstraint(int32_t opener, std::shared_ptr<const grammar::Grammar> grammar,
                                   std::shared_ptr<const grammar::Vocabulary> vocabulary)
    : opener_(opener), grammar_(std::move(grammar)), vocabulary_(std::move(vocabulary)) {
    XS_CHECK(grammar_ && vocabulary_, "tool call: a constraint without its grammar or vocabulary");
}

std::unique_ptr<TokenConstraint> OpenerConstraint::start() const {
    return std::make_unique<AfterOpener>(opener_, std::make_shared<grammar::Automaton>(grammar_, vocabulary_));
}

std::shared_ptr<const grammar::Vocabulary> grammar_vocabulary(const tok::Tokenizer& tokenizer) {
    std::vector<grammar::Vocabulary::Token> tokens(static_cast<size_t>(tokenizer.vocab_size()));
    for (int32_t id = 0; id < tokenizer.vocab_size(); ++id) {
        try {
            tokens[static_cast<size_t>(id)] = {tokenizer.token_bytes(id), tokenizer.is_added(id)};
        } catch (const Error&) {
            tokens[static_cast<size_t>(id)] = {"", true};  // an id the vocabulary leaves out
        }
    }
    return std::make_shared<const grammar::Vocabulary>(std::move(tokens));
}

GrammarConstraint::GrammarConstraint(std::shared_ptr<const grammar::Grammar> grammar,
                                     std::shared_ptr<const grammar::Vocabulary> vocabulary, std::string reasoning_end,
                                     std::vector<int32_t> opening)
    : spec_(std::make_shared<const Spec>(
          Spec{std::move(grammar), std::move(vocabulary), std::move(reasoning_end), std::move(opening)})) {
    XS_CHECK(spec_->grammar && spec_->vocabulary, "response_format: a constraint without its grammar or vocabulary");
}

std::unique_ptr<TokenConstraint> GrammarConstraint::start() const {
    Position at{.matcher = grammar::Matcher(std::make_shared<grammar::Automaton>(spec_->grammar, spec_->vocabulary))};
    if (!spec_->reasoning_end.empty()) {
        at.stage = Position::Stage::reasoning;
    } else if (!spec_->opening.empty()) {
        at.stage = Position::Stage::opening;
    }
    return std::make_unique<Answer>(spec_, std::move(at));
}

}  // namespace xs::server
