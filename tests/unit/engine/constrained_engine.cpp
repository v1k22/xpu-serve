// Constrained decoding in the engine, on cpu-ref over the tiny model, its 16 tokens read as pieces of JSON: the output
// always parses and is of the schema, greedy or sampled; where the free output already conforms, the same tokens;
// stop tokens only where the output may end; speculating with a chain or a block drafter, the tokens of the constrained
// run without one; batched beside a free request, each request as it runs alone.
#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/constraint.h"
#include "grammar/json_schema.h"
#include "grammar/matcher.h"
#include "tiny_engine.h"

namespace xs {

using test::greedy;
using test::make_engine;

namespace {

constexpr int32_t kStop = 15;

// The tiny model's tokens as JSON pieces; kStop is a marker.
const std::vector<std::string> kPieces = {"{", "}", "\"", ":", ",", " ", "a", "b",
                                          "1", "2", "true", "null", "[", "]", "\"a\"", "<|end|>"};

std::shared_ptr<const grammar::Vocabulary> vocabulary_of(const std::vector<std::string>& pieces, int32_t marker) {
    std::vector<grammar::Vocabulary::Token> tokens;
    for (size_t id = 0; id < pieces.size(); ++id) tokens.push_back({pieces[id], static_cast<int32_t>(id) == marker});
    return std::make_shared<const grammar::Vocabulary>(std::move(tokens));
}

// A grammar as the engine's constraint: what its matcher allows (the server's adds where an answer starts).
class GrammarRule final : public TokenConstraint {
public:
    explicit GrammarRule(grammar::Matcher m) : m_(std::move(m)) {}
    bool mask(std::span<float> logits) override {
        const grammar::TokenSet& allowed = m_.allowed();
        for (size_t t = 0; t < logits.size(); ++t) {
            if (!allowed.has(static_cast<int32_t>(t))) logits[t] = -std::numeric_limits<float>::infinity();
        }
        return allowed.count() > 0;
    }
    void accept(int32_t token) override { REQUIRE(m_.accept_token(token)); }
    bool complete() const override { return m_.complete(); }
    size_t allowed_prefix(std::span<const int32_t> tokens) const override {
        grammar::Matcher m = m_;
        size_t n = 0;
        while (n < tokens.size() && m.accept_token(tokens[n])) ++n;
        return n;
    }

private:
    grammar::Matcher m_;
};

class GrammarOutput final : public OutputConstraint {
public:
    GrammarOutput(std::shared_ptr<const grammar::Grammar> g, std::shared_ptr<const grammar::Vocabulary> v)
        : g_(std::move(g)), v_(std::move(v)) {}
    std::unique_ptr<TokenConstraint> start() const override {
        return std::make_unique<GrammarRule>(grammar::Matcher(std::make_shared<grammar::Automaton>(g_, v_)));
    }

private:
    std::shared_ptr<const grammar::Grammar> g_;
    std::shared_ptr<const grammar::Vocabulary> v_;
};

std::shared_ptr<const OutputConstraint> json_output(const char* schema,
                                                    const std::vector<std::string>& pieces = kPieces,
                                                    int32_t marker = kStop) {
    return std::make_shared<GrammarOutput>(grammar::json_schema_grammar(nlohmann::ordered_json::parse(schema)),
                                           vocabulary_of(pieces, marker));
}

std::string text_of(const std::vector<int32_t>& tokens, const std::vector<std::string>& pieces = kPieces) {
    std::string out;
    for (const int32_t t : tokens) out += pieces.at(static_cast<size_t>(t));
    return out;
}

// No string or array of any length: every way through it ends within a few dozen tokens, each byte one the
// vocabulary spells (a real byte-level one spells every byte).
const char* kSchema = R"({"type": "object", "properties": {
    "a": {"enum": [1, 2, "a"]},
    "b": {"anyOf": [{"type": "boolean"}, {"type": "null"}, {"$ref": "#/$defs/pair"}]}},
  "required": ["a", "b"], "additionalProperties": false,
  "$defs": {"pair": {"type": "object", "properties": {"a": {"const": [1, 2]}}, "required": ["a"],
                     "additionalProperties": false}}})";

void check_of_schema(const GenerateResult& r) {
    REQUIRE(r.error == "");
    REQUIRE(r.finish == FinishReason::Stop);
    const std::string text = text_of(r.tokens);
    CAPTURE(text);
    const nlohmann::json doc = nlohmann::json::parse(text, nullptr, false);
    REQUIRE(doc.is_object());
    CHECK(doc.size() == 2);
    CHECK((doc["a"] == 1 || doc["a"] == 2 || doc["a"] == "a"));
    const nlohmann::json& b = doc["b"];
    CHECK((b == true || b.is_null() || b == nlohmann::json{{"a", {1, 2}}}));
    CHECK(r.stats.constrained_rows >= static_cast<int>(r.tokens.size()));
}

GenerateRequest constrained(std::vector<int32_t> prompt, int max_new, std::shared_ptr<const OutputConstraint> c,
                            std::vector<int32_t> stops = {kStop}) {
    GenerateRequest r = greedy(std::move(prompt), max_new, std::move(stops));
    r.constraint = std::move(c);
    return r;
}

// The tiny model's tokens read so that `quote` is a JSON string's quote, `stop` a marker and every other a letter.
std::vector<std::string> string_pieces(int32_t quote, int32_t stop) {
    std::vector<std::string> pieces;
    for (int32_t t = 0; t < test::TinySpecModel::kVocab; ++t) {
        pieces.push_back(t == quote ? "\"" : t == stop ? "<|end|>" : std::string(1, static_cast<char>('c' + t)));
    }
    return pieces;
}

}  // namespace

TEST_CASE("constrained: the output always parses and is of the schema, greedy or sampled, speculating or not") {
    const auto c = json_output(kSchema);
    struct Drafting {
        int spec_k;
        bool block;
    };
    for (const std::vector<int32_t>& prompt : {std::vector<int32_t>{1, 7, 3, 12}, std::vector<int32_t>{6, 2, 13}}) {
        GenerateResult plain;
        for (const Drafting d : {Drafting{0, false}, Drafting{2, false}, Drafting{4, true}}) {
            CAPTURE(d.spec_k);
            CAPTURE(d.block);
            const auto engine = make_engine(d.spec_k, 3, false, d.block);
            const GenerateResult got = engine->generate(constrained(prompt, 64, c), {});
            check_of_schema(got);
            if (d.spec_k == 0) plain = got;
            CHECK(got.tokens == plain.tokens);  // speculating gives the constrained greedy tokens
            // The drafter writes the free model's tokens: those the grammar refuses, and the drafts after them, are
            // never verified (every round has room for spec_k drafts: the answer ends far before 64 tokens).
            if (d.spec_k > 0) CHECK(got.stats.spec_drafted < d.spec_k * got.stats.spec_rounds);
            for (uint64_t seed = 0; seed < 12; ++seed) {
                GenerateRequest sampled = constrained(prompt, 64, c);
                sampled.sampling.temperature = 1.5f;
                sampled.sampling.seed = seed;
                check_of_schema(engine->generate(sampled, {}));
            }
        }
    }
}

TEST_CASE("constrained: where the free output already conforms, the same tokens; with drafters too") {
    const std::vector<int32_t> prompt = {1, 7, 3, 12, 5, 9};
    const GenerateResult free = make_engine(0, 3)->generate(greedy(prompt, 14), {});
    // The free output's first token is the string's opening quote, every other token a letter, and the stop token one
    // the free output never writes: what it writes before its first token comes again is a JSON string's beginning.
    const int32_t quote = free.tokens.at(0);
    size_t n = 1;
    while (n < free.tokens.size() && free.tokens[n] != quote) ++n;
    REQUIRE(n >= 6);
    int32_t stop = 0;
    while (std::find(free.tokens.begin(), free.tokens.end(), stop) != free.tokens.end()) ++stop;
    const auto c = json_output(R"({"type": "string"})", string_pieces(quote, stop), stop);
    for (const auto& [k, block] : {std::pair{0, false}, std::pair{2, false}, std::pair{3, true}}) {
        CAPTURE(k);
        const GenerateResult got =
            make_engine(k, 3, false, block)->generate(constrained(prompt, static_cast<int>(n), c, {stop}), {});
        CHECK(got.error == "");
        CHECK(got.tokens == std::vector<int32_t>(free.tokens.begin(), free.tokens.begin() + static_cast<long>(n)));
        CHECK(got.finish == FinishReason::Length);
    }
}

TEST_CASE("constrained: a stop token counts only where the output may end") {
    const std::vector<int32_t> prompt = {1, 7, 3, 12, 5, 9};
    const GenerateResult free = make_engine(0, 3)->generate(greedy(prompt, 14), {});
    // The free output opens a string with its first token and would stop at its third, inside the string.
    const int32_t stop = free.tokens.at(2);
    REQUIRE(free.tokens.at(0) != stop);
    const auto c = json_output(R"({"type": "string"})", string_pieces(free.tokens.at(0), stop), stop);
    CHECK(make_engine(0, 3)->generate(greedy(prompt, 14, {stop}), {}).tokens.size() == 2);
    for (const int k : {0, 3}) {
        CAPTURE(k);
        const GenerateResult held = make_engine(k, 3)->generate(constrained(prompt, 14, c, {stop}), {});
        REQUIRE(held.error == "");
        REQUIRE(held.tokens.size() > 2);
        CHECK(std::vector<int32_t>(held.tokens.begin(), held.tokens.begin() + 2) ==
              std::vector<int32_t>(free.tokens.begin(), free.tokens.begin() + 2));
        CHECK(held.tokens[2] != stop);
    }
}

TEST_CASE("constrained: batched beside a free request, each as it runs alone") {
    const auto c = json_output(kSchema);
    const GenerateRequest json = constrained({1, 7, 3, 12}, 64, c);
    GenerateRequest free = greedy({3, 1, 4, 1, 5, 9, 2, 6}, 20);
    for (const auto& [k, block] : {std::pair{0, false}, std::pair{2, false}, std::pair{3, true}}) {
        for (const bool argmax : {false, true}) {
            CAPTURE(k);
            CAPTURE(argmax);
            const GenerateResult json_alone = make_engine(k, 3, argmax, block)->generate(json, {});
            const GenerateResult free_alone = make_engine(k, 3, argmax, block)->generate(free, {});
            check_of_schema(json_alone);
            const auto engine = make_engine(k, 3, argmax, block, 2, true);
            const std::vector<Engine::Handle> hs{*engine->start(json, {}), *engine->start(free, {})};
            for (const Engine::Handle h : hs) {
                while (!engine->decoding(h)) REQUIRE_FALSE(engine->advance(h));
            }
            std::vector<Engine::Handle> step = hs;
            std::vector<GenerateResult> got(2);
            while (!step.empty()) {
                const std::vector<bool> done = engine->advance_decoding(step);
                std::vector<Engine::Handle> next;
                for (size_t i = 0; i < step.size(); ++i) {
                    if (!done[i]) {
                        next.push_back(step[i]);
                        continue;
                    }
                    got[step[i] == hs[0] ? 0 : 1] = engine->finish(step[i]);
                }
                step = next;
            }
            CHECK(got[0].tokens == json_alone.tokens);
            CHECK(got[1].tokens == free_alone.tokens);
            CHECK(got[1].finish == free_alone.finish);
        }
    }
}

TEST_CASE("constrained: without a stop token it cannot end, and a constraint that allows nothing fails the request") {
    const auto engine = make_engine(0, 3);
    const GenerateResult no_stop = engine->generate(constrained({1, 2, 3}, 8, json_output(kSchema), {}), {});
    CHECK(no_stop.finish == FinishReason::Error);
    CHECK(no_stop.error.find("stop token") != std::string::npos);
    // A grammar no token can begin: nothing is allowed and the output cannot end.
    const auto none = json_output(R"({"const": "z"})");
    const GenerateResult stuck = engine->generate(constrained({1, 2, 3}, 8, none), {});
    CHECK(stuck.finish == FinishReason::Error);
    CHECK(stuck.error.find("allows no token") != std::string::npos);
    CHECK(engine->generate(greedy({1, 2, 3}, 4), {}).finish == FinishReason::Length);  // the sequence still serves
}

TEST_CASE("constrained with log-probabilities: the row as the constraint left it, masked tokens at -infinity") {
    GenerateRequest r = constrained({1, 7, 3, 12}, 64, json_output(kSchema));
    r.logprobs = static_cast<int>(test::TinySpecModel::kVocab);  // the whole row
    std::vector<TokenLogprobs> got;
    const GenerateResult result = make_engine(0, 8)->generate(r, [&](int32_t, const TokenLogprobs* lp) {
        got.push_back(*lp);
        return true;
    });
    check_of_schema(result);
    REQUIRE(got.size() == result.tokens.size());
    int masked = 0;
    for (const TokenLogprobs& lp : got) {
        CHECK(std::isfinite(lp.logprob));
        double total = 0.0;
        for (const auto& [id, value] : lp.top) {
            if (std::isinf(value)) {
                ++masked;
            } else {
                total += std::exp(static_cast<double>(value));
            }
        }
        CHECK(total == doctest::Approx(1.0));  // renormalized over the tokens the grammar allows
    }
    CHECK(masked > 0);
}

}  // namespace xs
