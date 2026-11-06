#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "server/stop_strings.h"
#include "server/token_output.h"
#include "unit/server/fake_service.h"

using xs::server::StopStringMatcher;
using xs::server::TokenOutput;

namespace {

// Feeds `pieces` one by one and returns everything released, including the final flush.
std::string run(StopStringMatcher& m, const std::vector<std::string>& pieces) {
    std::string out;
    for (const auto& p : pieces) out += m.push(p);
    return out + m.flush();
}

}  // namespace

TEST_SUITE("server.stop_strings") {
    TEST_CASE("no stop strings passes text straight through") {
        StopStringMatcher m({});
        CHECK(m.push("abc") == "abc");
        CHECK(m.flush().empty());
        CHECK_FALSE(m.stopped());
    }

    TEST_CASE("holds back a possible prefix and releases it when it stops matching") {
        StopStringMatcher m({"END"});
        CHECK(m.push("abE") == "ab");
        CHECK(m.push("N") == "");
        CHECK(m.push("x") == "ENx");
        CHECK_FALSE(m.stopped());
    }

    TEST_CASE("match split across pieces cuts exactly before the stop string") {
        StopStringMatcher m({"END"});
        CHECK(run(m, {"hello E", "N", "D and more"}) == "hello ");
        CHECK(m.stopped());
        CHECK(m.push("later").empty());
    }

    TEST_CASE("match at the very start releases nothing") {
        StopStringMatcher m({"stop"});
        CHECK(m.push("stop here").empty());
        CHECK(m.stopped());
    }

    TEST_CASE("earliest match wins among several stop strings") {
        StopStringMatcher m({"world", "lo"});
        CHECK(m.push("hello world") == "hel");
    }

    TEST_CASE("held-back prefix is released at flush when the stream ends") {
        StopStringMatcher m({"</end>"});
        CHECK(m.push("text </e") == "text ");
        CHECK(m.flush() == "</e");
    }

    TEST_CASE("empty stop strings are ignored") {
        StopStringMatcher m({"", "x"});
        CHECK(m.push("abc") == "abc");
    }

    TEST_CASE("token output applies stop strings to decoded content") {
        xs::test::ByteTokenizer tokenizer;
        TokenOutput out(tokenizer, xs::chat::make_output_parser("plain"), {"\n\n"});
        std::string content;
        for (int32_t t : xs::test::ByteTokenizer::tokens("one\ntwo\n\nthree")) {
            content += out.push(t).content;
            if (out.stop_string_hit()) break;
        }
        content += out.finish().content;
        CHECK(content == "one\ntwo");
        CHECK(out.stop_string_hit());
    }
}
