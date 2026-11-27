// The tokens a grammar allows next, from the walk of a vocabulary's trie, against a brute-force check of every token's
// bytes from the same state: on a hand-written vocabulary (pieces of JSON, parts of UTF-8 characters, markers, tokens
// with the same bytes, one with none) and on a random one large enough that the masks of long walks are kept. With
// XS_QWEN35_DIR (a checkpoint's tokenizer), what a mask costs a step on that vocabulary.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "grammar/json_schema.h"
#include "grammar_test_util.h"
#include "tokenizer/tokenizer.h"

namespace xs {

using namespace grammar;

namespace {

std::vector<int32_t> brute_force(Automaton& a, Automaton::State s) {
    std::vector<int32_t> out;
    for (int32_t id = 0; id < a.vocabulary().size(); ++id) {
        if (a.step_token(s, id) != Automaton::kDead) out.push_back(id);
    }
    return out;
}

std::vector<int32_t> ids_of(const TokenSet& set) {
    std::vector<int32_t> out;
    for (int32_t id = 0; id < set.size(); ++id) {
        if (set.has(id)) out.push_back(id);
    }
    return out;
}

// Along each document, byte by byte, the walk's tokens equal the brute force's, asked twice (the second time a long
// walk's tokens are the kept ones).
void check_masks(const std::shared_ptr<const Grammar>& g, const std::shared_ptr<const Vocabulary>& v,
                 const std::vector<std::string>& documents) {
    auto a = std::make_shared<Automaton>(g, v);
    for (const std::string& doc : documents) {
        CAPTURE(doc);
        Automaton::State s = a->start();
        for (size_t i = 0; i <= doc.size(); ++i) {
            CAPTURE(i);
            const std::vector<int32_t> want = brute_force(*a, s);
            CHECK(ids_of(a->allowed(s)) == want);
            CHECK(ids_of(a->allowed(s)) == want);
            if (i == doc.size()) break;
            s = a->step(s, static_cast<uint8_t>(doc[i]));
            REQUIRE(s != Automaton::kDead);
        }
        CHECK(a->accepting(s));
    }
}

const char* kSchema = R"({"type": "object", "properties": {
    "name": {"type": "string"}, "n": {"type": ["integer", "null"]}, "ok": {"type": "boolean"},
    "tags": {"type": "array", "items": {"enum": ["red", "rød"]}}},
  "required": ["name", "n", "ok", "tags"], "additionalProperties": false})";

const std::vector<std::string> kDocuments = {
    R"({"name": "Zoë \"Z\" 😀", "n": -12, "ok": true, "tags": ["red", "rød"]})",
    "{\n  \"name\": \"\",\n  \"n\": null,\n  \"ok\": false,\n  \"tags\": []\n}",
};

}  // namespace

TEST_SUITE("grammar: token masks") {
    TEST_CASE("the trie's walk allows what each token's bytes allow, on a hand-written vocabulary") {
        std::vector<Vocabulary::Token> tokens;
        for (const std::string t :
             {"{", "}", "\"", ":", ",", " ", "\n", "  ", "\n  ", "{\"", "\":", "\",", "\"}", "\":\"", " \"", "\", \"",
              "name", "nam", "e", "n", "ok", "tags", "true", "tr", "ue", "false", "null", "nul", "l", "-", "1", "12",
              "-1", "0", "[", "]", "[\"", "\"]", "red", "r", "ø", "\xC3", "\xB8", "d", "Z", "o", "\xF0\x9F", "\x98\x80",
              "\\", "\\\"", "a", "ab", "}\n", "\"}]"}) {
            tokens.push_back({t, false});
        }
        tokens.push_back({"<|end|>", true});  // a marker: never written by a grammar
        tokens.push_back({"", false});        // no bytes: never allowed
        tokens.push_back({"red", false});     // the same bytes as another token
        const auto v = std::make_shared<const Vocabulary>(std::move(tokens));
        check_masks(json_schema_grammar(nlohmann::ordered_json::parse(kSchema)), v,
                    {R"({"name": "ab\"Zo", "n": 12, "ok": true, "tags": ["red", "rød"]})",
                     R"({"name":"","n":null,"ok":false,"tags":[]})"});
        check_masks(json_object_grammar(), v, {R"({"name": {"ok": [true, null, -1]}})"});
    }

    TEST_CASE("on a vocabulary large enough that long walks' masks are kept") {
        std::mt19937 rng(5);
        const std::string alphabet = "{}[]\":, \n-0123456789.eEtrufalsnmogd\\\xC3\xB8\xF0\x9F\x98\x80";
        std::vector<Vocabulary::Token> tokens;
        for (int i = 0; i < 6000; ++i) {
            std::string t;
            for (size_t n = 1 + rng() % 6; n > 0; --n) t += alphabet[rng() % alphabet.size()];
            tokens.push_back({t, i % 997 == 0});
        }
        for (const char c : alphabet) tokens.push_back({std::string(1, c), false});
        const auto v = std::make_shared<const Vocabulary>(std::move(tokens));
        REQUIRE(v->trie().nodes() > 4096);
        check_masks(json_schema_grammar(nlohmann::ordered_json::parse(kSchema)), v, kDocuments);
        check_masks(json_object_grammar(), v, kDocuments);
    }

    TEST_CASE("the trie: nodes in depth-first order, subtrees, tokens where their bytes end") {
        const Vocabulary v({{"ab", false}, {"a", false}, {"b", false}, {"ab", false}, {"x", true}, {"abc", false}});
        const Vocabulary::Trie& t = v.trie();
        REQUIRE(t.nodes() == 4);  // a, ab, abc, b
        CHECK(std::string(t.byte.begin(), t.byte.end()) == "abcb");
        CHECK(t.depth == std::vector<uint16_t>{1, 2, 3, 1});
        CHECK(t.end == std::vector<uint32_t>{3, 3, 3, 4});
        CHECK(std::vector<int32_t>(t.tokens.begin() + t.first[1], t.tokens.begin() + t.first[2]) ==
              std::vector<int32_t>{0, 3});
        CHECK(v.bytes(5) == "abc");
        CHECK_FALSE(v.writable(4));
    }

    TEST_CASE("what a mask costs a step on a real vocabulary (XS_QWEN35_DIR)") {
        const char* dir = std::getenv("XS_QWEN35_DIR");
        if (dir == nullptr) return;
        const auto tok = tok::Tokenizer::load(dir);
        std::vector<Vocabulary::Token> tokens;
        for (int32_t id = 0; id < tok->vocab_size(); ++id) tokens.push_back({tok->token_bytes(id), tok->is_added(id)});
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        const auto v = std::make_shared<const Vocabulary>(std::move(tokens));
        const double build_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        const std::string doc = nlohmann::ordered_json::parse(R"({"name": "Ada Lovelace, née Byron", "n": 1815,
            "ok": true, "tags": ["red", "rød", "red", "red"]})").dump(2);
        const std::vector<int32_t> ids = tok->encode(doc, false);
        // One request's automaton, as a generation has it: every state is new to it once.
        const auto a = std::make_shared<Automaton>(json_schema_grammar(nlohmann::ordered_json::parse(kSchema)), v);
        Matcher m(a);
        std::vector<double> ms;
        for (const int32_t id : ids) {
            const auto start = clock::now();
            const TokenSet& allowed = m.allowed();
            ms.push_back(std::chrono::duration<double, std::milli>(clock::now() - start).count());
            REQUIRE(allowed.has(id));
            REQUIRE(m.accept_token(id));
        }
        CHECK(m.complete());
        std::vector<double> sorted = ms;
        std::sort(sorted.begin(), sorted.end());
        double total = 0;
        for (const double x : ms) total += x;
        std::printf("grammar: vocabulary of %d tokens, trie of %zu nodes, built in %.0f ms; %zu steps: mean %.3f ms, "
                    "median %.3f, max %.3f, total %.1f; %zu states\n",
                    v->size(), v->trie().nodes(), build_ms, ms.size(), total / static_cast<double>(ms.size()),
                    sorted[sorted.size() / 2], sorted.back(), total, a->states());
    }
}

}  // namespace xs
