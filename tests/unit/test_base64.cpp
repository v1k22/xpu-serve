#include <doctest/doctest.h>

#include <string>

#include "util/base64.h"

using xs::base64_decode;

namespace {

std::string decoded(std::string_view text) {
    const auto bytes = base64_decode(text);
    REQUIRE(bytes.has_value());
    return {bytes->begin(), bytes->end()};
}

}  // namespace

TEST_CASE("base64 decodes the RFC 4648 vectors, with or without padding") {
    CHECK(decoded("") == "");
    CHECK(decoded("Zg==") == "f");
    CHECK(decoded("Zm8=") == "fo");
    CHECK(decoded("Zm9v") == "foo");
    CHECK(decoded("Zm9vYg==") == "foob");
    CHECK(decoded("Zm9vYmE=") == "fooba");
    CHECK(decoded("Zm9vYmFy") == "foobar");
    CHECK(decoded("Zm9vYg") == "foob");
    CHECK(decoded("Zm9vYmE") == "fooba");
    CHECK(decoded("Zm9v\r\nYmFy") == "foobar");
    CHECK(decoded("+/+/") == std::string("\xfb\xff\xbf"));
}

TEST_CASE("base64 refuses characters outside the alphabet and malformed padding") {
    CHECK_FALSE(base64_decode("Zm9v!").has_value());
    CHECK_FALSE(base64_decode("Zm9v-_").has_value());  // URL-safe alphabet is not data-URL base64
    CHECK_FALSE(base64_decode("Z").has_value());       // one sextet cannot encode a byte
    CHECK_FALSE(base64_decode("Zm=9").has_value());    // data after padding
    CHECK_FALSE(base64_decode("Zg=").has_value());     // padding that does not complete the group
    CHECK_FALSE(base64_decode("Zg===").has_value());
}
