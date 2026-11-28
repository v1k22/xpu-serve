// response_format's constraint (server/grammar_constraint.h): the vocabulary a tokenizer gives a grammar, rows masked
// to what the grammar allows, complete only once it is, drafts checked without committing; and where the answer
// starts: free while the prompt's reasoning goes on, the dialect's opening, then the grammar.
#include <doctest/doctest.h>

#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "grammar/json_schema.h"
#include "server/grammar_constraint.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"

namespace xs {

using server::GrammarConstraint;

namespace {

constexpr float kMasked = -std::numeric_limits<float>::infinity();
constexpr int32_t kThinkEnd = 256, kEnd = 257, kOpening = 258, kTail = 259, kThink = 260;

// Token b is the byte b; then markers (</think>, <|im_end|>, <|msg|>), one piece that ends a thought and opens an
// object, and the marker <think>.
std::shared_ptr<const grammar::Vocabulary> vocabulary() {
    std::vector<grammar::Vocabulary::Token> tokens;
    for (int b = 0; b < 256; ++b) tokens.push_back({std::string(1, static_cast<char>(b)), false});
    tokens.push_back({"</think>", true});
    tokens.push_back({"<|im_end|>", true});
    tokens.push_back({"<|msg|>", true});
    tokens.push_back({"nk>\n{", false});
    tokens.push_back({"<think>", true});
    return std::make_shared<const grammar::Vocabulary>(std::move(tokens));
}

std::vector<int32_t> bytes_of(std::string_view text) {
    std::vector<int32_t> out;
    for (const char c : text) out.push_back(static_cast<unsigned char>(c));
    return out;
}

// The tokens a mask leaves, of a row as wide as `width`.
std::vector<int32_t> left(TokenConstraint& c, size_t width = 261) {
    std::vector<float> logits(width, 1.0f);
    c.mask(logits);
    std::vector<int32_t> out;
    for (size_t t = 0; t < width; ++t) {
        if (logits[t] != kMasked) out.push_back(static_cast<int32_t>(t));
    }
    return out;
}

}  // namespace

TEST_SUITE("server.grammar_constraint") {
    TEST_CASE("a tokenizer's tokens as a grammar reads them: each one's bytes, added tokens markers") {
        const auto tok =
            tok::Tokenizer::load(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/tiny");
        const auto v = server::grammar_vocabulary(*tok);
        CHECK(v->size() == tok->vocab_size());
        CHECK(v->bytes(29) == " hello");
        CHECK(v->bytes(13) == "\xC3");
        CHECK(v->writable(29));
        CHECK_FALSE(v->writable(1));  // <s>, special
        CHECK_FALSE(v->writable(3));  // <mark>, added but not special
        CHECK(v->bytes(3) == "<mark>");
    }

    TEST_CASE("rows masked to the grammar, complete only where it is, drafts checked without committing") {
        const GrammarConstraint format(grammar::json_object_grammar(), vocabulary(), {});
        const auto c = format.start();
        CHECK(left(*c) == std::vector<int32_t>{'\n', ' ', '{'});
        CHECK_FALSE(c->complete());
        const std::vector<int32_t> drafts = bytes_of(R"({"a":1}x)");
        CHECK(c->allowed_prefix(drafts) == 7);
        CHECK(left(*c) == std::vector<int32_t>{'\n', ' ', '{'});  // nothing was accepted
        for (size_t i = 0; i < 7; ++i) c->accept(drafts[i]);
        CHECK(c->complete());
        std::vector<float> logits(300, 1.0f);
        CHECK_FALSE(c->mask(logits));  // nothing may follow but a stop token, which is the engine's
        CHECK(logits[280] == kMasked);  // nor anything past the vocabulary
        CHECK_THROWS(c->accept('x'));
        // A second generation of the same format starts from the start.
        CHECK(left(*format.start()) == std::vector<int32_t>{'\n', ' ', '{'});
    }

    TEST_CASE("reasoning first: free until the thought closes, then the dialect's opening, then the grammar") {
        const auto g = grammar::json_schema_grammar(nlohmann::ordered_json::parse(R"({"type": "integer"})"));
        const GrammarConstraint format(g, vocabulary(), {.reasoning_end = "</think>", .opening = {kOpening}});
        const auto c = format.start();
        CHECK(left(*c).size() == 261);  // the thought is free (the engine keeps stop tokens out)
        CHECK_FALSE(c->complete());
        for (const int32_t t : bytes_of("Seven, I think. ")) c->accept(t);
        CHECK(left(*c).size() == 261);
        CHECK(c->allowed_prefix(std::vector<int32_t>{'x', kThinkEnd, kOpening, '7', '}'}) == 4);
        c->accept(kThinkEnd);
        CHECK(left(*c) == std::vector<int32_t>{kOpening});
        CHECK_FALSE(c->complete());
        c->accept(kOpening);
        CHECK(left(*c) == std::vector<int32_t>{'\n', ' ', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'});
        c->accept('7');
        CHECK(c->complete());

        // The marker written in pieces, and a piece that also writes what comes after it.
        const GrammarConstraint object(grammar::json_object_grammar(), vocabulary(), {.reasoning_end = "</think>"});
        const auto d = object.start();
        for (const int32_t t : bytes_of("hmm </thi")) d->accept(t);
        CHECK(left(*d).size() == 261);
        d->accept(kTail);  // "nk>\n{": the thought ends, and the object opens
        CHECK_FALSE(d->complete());
        for (const int32_t t : bytes_of("}")) d->accept(t);
        CHECK(d->complete());
        // A marker that is not the dialect's closes nothing.
        const auto e = object.start();
        e->accept(kEnd);
        CHECK(left(*e).size() == 261);
    }

    TEST_CASE("a thought the model may open itself (Qwen 3): its opener beside the answer's first tokens") {
        const auto g = grammar::json_schema_grammar(nlohmann::ordered_json::parse(R"({"type": "integer"})"));
        const server::AnswerGate gate{.thought_open = kThink, .thought_close = "</think>", .opening = {kOpening}};
        const GrammarConstraint format(g, vocabulary(), gate);
        const auto c = format.start();
        CHECK(left(*c) == std::vector<int32_t>{kOpening, kThink});
        CHECK(c->allowed_prefix(std::vector<int32_t>{kThink, 'h', 'm', kThinkEnd, kOpening, '4', 'x'}) == 6);
        c->accept(kThink);
        CHECK(left(*c).size() == 261);  // the thought is free
        CHECK_THROWS(format.start()->accept('h'));  // not without its opener
        for (const int32_t t : bytes_of("Four.\n")) c->accept(t);
        c->accept(kThinkEnd);
        CHECK(left(*c) == std::vector<int32_t>{kOpening});
        c->accept(kOpening);
        c->accept('4');
        CHECK(c->complete());
        // Or the answer at once; then the opener is a token like any other the grammar refuses.
        const auto d = format.start();
        d->accept(kOpening);
        CHECK(left(*d) == std::vector<int32_t>{'\n', ' ', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'});
        CHECK_THROWS(d->accept(kThink));
        // Without an opening the grammar's first tokens stand beside the opener.
        const GrammarConstraint plain(grammar::json_object_grammar(), vocabulary(),
                                      {.thought_open = kThink, .thought_close = "</think>"});
        CHECK(left(*plain.start()) == std::vector<int32_t>{'\n', ' ', '{', kThink});
        CHECK_THROWS_AS(GrammarConstraint(grammar::json_object_grammar(), vocabulary(), {.thought_open = kThink}),
                        Error);
    }

    TEST_CASE("the automata of grammars kept by name: requests held to one grammar share its states and masks") {
        server::AutomatonCache cache(2);
        const auto object = grammar::json_object_grammar();
        const auto v = vocabulary();
        const auto a = cache.get("json_object", object, v);
        CHECK(cache.get("json_object", object, v) == a);
        CHECK(cache.get("", object, v) != a);  // unnamed: not kept
        const auto b = cache.get("other", grammar::json_object_grammar(), v);
        CHECK(b != a);
        CHECK(cache.get("json_object", object, v) == a);
        cache.get("third", object, v);  // past the bound the least recently used goes: "other"
        CHECK(cache.get("other", object, v) != b);
        // Two generations of one constraint walk one automaton: the second finds the states the first made.
        GrammarConstraint shared(object, v, {}, std::make_shared<server::AutomatonCache>(), "json_object");
        const auto first = shared.start();
        std::vector<float> logits(static_cast<size_t>(v->size()), 0.0f);
        const std::vector<int32_t> doc = bytes_of("{\"a\": [1, {\"b\": null}]}");
        for (const int32_t t : doc) {
            first->mask(logits);
            first->accept(t);
        }
        CHECK(first->complete());
        const auto second = shared.start();
        CHECK(second->allowed_prefix(doc) == doc.size());
    }
}

}  // namespace xs
