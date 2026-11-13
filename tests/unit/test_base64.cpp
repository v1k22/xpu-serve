#include <doctest/doctest.h>

#include <string>

#include "util/base64.h"

using xs::base64_decode;
using xs::base64_encode;

namespace {

std::string decoded(std::string_view text) {
    const auto bytes = base64_decode(text);
    REQUIRE(bytes.has_value());
    return {bytes->begin(), bytes->end()};
}

std::string encoded(std::string_view text) {
    return base64_encode({reinterpret_cast<const uint8_t*>(text.data()), text.size()});
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

TEST_CASE("base64 encodes the RFC 4648 vectors with padding, and decodes back every byte") {
    CHECK(encoded("") == "");
    CHECK(encoded("f") == "Zg==");
    CHECK(encoded("fo") == "Zm8=");
    CHECK(encoded("foo") == "Zm9v");
    CHECK(encoded("foob") == "Zm9vYg==");
    CHECK(encoded("fooba") == "Zm9vYmE=");
    CHECK(encoded("foobar") == "Zm9vYmFy");
    CHECK(encoded(std::string("\xfb\xff\xbf")) == "+/+/");
    std::string all;
    for (int b = 0; b < 256; ++b)
        all.push_back(static_cast<char>(b));
    CHECK(decoded(encoded(all)) == all);
}
