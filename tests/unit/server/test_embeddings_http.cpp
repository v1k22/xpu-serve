// The embeddings API against a fake model: every input form, each refusal a 400 naming its field, the float and
// base64 encodings, and the server around it (models, health, no chat routes, concurrent clients); then a tiny model
// with a WordPiece tokenizer of its own behind it.

#include <doctest/doctest.h>

#include <cstring>
#include <thread>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "embedding/service.h"
#include "model/embedding_test_util.h"
#include "server/embedding_handlers.h"
#include "server/server.h"
#include "util/base64.h"
#include "util/error.h"

namespace {

using Json = nlohmann::ordered_json;

xs::EmbeddingSpec fake_spec() {
    xs::EmbeddingSpec s;
    s.id = "fake-embedder";
    s.arch = "fake";
    s.dimension = 4;
    s.max_tokens = 8;
    s.vocab_size = 100;
    return s;
}

// [CLS] = 1, a token per byte (10 + byte % 90), [SEP] = 2. "BOOM" cannot be tokenized.
std::vector<int32_t> fake_tokenize(std::string_view text) {
    if (text == "BOOM") xs::fail("the tokenizer exploded");
    std::vector<int32_t> ids = {1};
    for (const char c : text)
        ids.push_back(10 + static_cast<unsigned char>(c) % 90);
    ids.push_back(2);
    return ids;
}

// Values whose float32 bits a decimal round trip could change; a sequence that starts with 99 breaks the device.
std::vector<float> fake_vector(const std::vector<int32_t>& ids) {
    return {static_cast<float>(ids.size()) + 0.1f, static_cast<float>(ids.front()) / 3.0f, -1e-7f,
            static_cast<float>(ids.back()) * 1.0001f};
}

xs::EmbeddingResult fake_embed(const std::vector<std::vector<int32_t>>& sequences) {
    xs::EmbeddingResult r;
    for (const auto& ids : sequences) {
        if (ids.front() == 99) xs::fail("the device fell over");
        r.embeddings.push_back(fake_vector(ids));
        r.tokens += static_cast<int64_t>(ids.size());
    }
    r.timing.device_ms = 1.0;
    r.timing.total_ms = 1.5;
    return r;
}

xs::server::EmbeddingsRequest parse(const std::string& body) {
    return xs::server::parse_embeddings_request(Json::parse(body), fake_spec(), fake_tokenize);
}

std::vector<float> decode_floats(const std::string& text) {
    const auto bytes = xs::base64_decode(text);
    REQUIRE(bytes.has_value());
    REQUIRE(bytes->size() % 4 == 0);
    std::vector<float> out(bytes->size() / 4);
    std::memcpy(out.data(), bytes->data(), bytes->size());
    return out;
}

class RunningServer {
public:
    explicit RunningServer(xs::EmbeddingService& service)
        : server_(service, options()), thread_([this] { server_.run(); }) {
        REQUIRE(server_.wait_until_listening(5000));
    }
    ~RunningServer() {
        server_.stop();
        thread_.join();
    }
    httplib::Client client() const {
        httplib::Client c("127.0.0.1", server_.bound_port());
        c.set_read_timeout(10, 0);
        return c;
    }

private:
    static xs::server::ServerOptions options() {
        xs::server::ServerOptions o;
        o.port = 0;
        o.http_threads = 4;
        return o;
    }
    xs::server::Server server_;
    std::thread thread_;
};

Json post(httplib::Client& c, const std::string& body, int status) {
    const auto res = c.Post("/v1/embeddings", body, "application/json");
    REQUIRE(res);
    CHECK(res->status == status);
    return Json::parse(res->body);
}

}  // namespace

TEST_SUITE("embeddings api") {
    TEST_CASE("every input form: text with its special tokens, token ids as given") {
        CHECK(parse(R"({"input": "hi"})").sequences == std::vector<std::vector<int32_t>>{{1, 24, 25, 2}});
        CHECK(parse(R"({"input": ["hi", "a"], "model": "x", "user": "u"})").sequences ==
              std::vector<std::vector<int32_t>>{{1, 24, 25, 2}, {1, 17, 2}});
        CHECK(parse(R"({"input": [5, 6, 99]})").sequences == std::vector<std::vector<int32_t>>{{5, 6, 99}});
        CHECK(parse(R"({"input": [[5], [6, 7]]})").sequences == std::vector<std::vector<int32_t>>{{5}, {6, 7}});
        CHECK(!parse(R"({"input": "hi"})").base64);
        CHECK(!parse(R"({"input": "hi", "encoding_format": "float"})").base64);
        CHECK(!parse(R"({"input": "hi", "encoding_format": null, "dimensions": null})").base64);
        CHECK(parse(R"({"input": "hi", "encoding_format": "base64", "dimensions": 4})").base64);
        CHECK(parse(R"({"input": "abcdef"})").sequences[0].size() == 8);  // the limit, special tokens included
    }

    TEST_CASE("what the model cannot read whole is a 400 naming the field") {
        struct Case {
            const char* body;
            const char* param;
            const char* message;
        };
        const Case cases[] = {
            {R"({})", "input", "'input' is required"},
            {R"({"input": null})", "input", "'input' is required"},
            {R"({"input": 3})", "input", "must be a string, an array of strings"},
            {R"({"input": {"text": "hi"}})", "input", "must be a string, an array of strings"},
            {R"({"input": []})", "input", "'input' must not be empty"},
            {R"({"input": ""})", "input", "'input' is an empty string"},
            {R"({"input": ["a", ""]})", "input[1]", "'input[1]' is an empty string"},
            {R"({"input": ["a", 3]})", "input[1]", "'input[1]' must be a string, as 'input[0]' is"},
            {R"({"input": [[1], "a"]})", "input[1]", "must be an array of token ids, as 'input[0]' is"},
            {R"({"input": [[1], []]})", "input[1]", "'input[1]' has no token"},
            {R"({"input": [1, "a"]})", "input", "'input' must be an array of token ids (integers)"},
            {R"({"input": [1, 2.5]})", "input", "'input' must be an array of token ids (integers)"},
            {R"({"input": [[1, 2.0]]})", "input[0]", "'input[0]' must be an array of token ids (integers)"},
            {R"({"input": [1, 100]})", "input", "'input' has token id 100, outside the vocabulary (0 to 99)"},
            {R"({"input": [-1]})", "input", "'input' has token id -1, outside the vocabulary"},
            {R"({"input": [[1], [99999999999]]})", "input[1]", "'input[1]' has token id 99999999999, outside"},
            {R"({"input": [[1], [18446744073709551615]]})", "input[1]", "token id 18446744073709551615, outside"},
            {R"({"input": [1, 2, 3, 4, 5, 6, 7, 8, 9]})", "input", "'input' has 9 tokens, more than the 8"},
            {R"({"input": "abcdefg"})", "input", "'input' has 9 tokens, more than the 8 the model reads"},
            {R"({"input": ["ok", "abcdefghij"]})", "input[1]", "'input[1]' has 12 tokens, more than the 8"},
            {R"({"input": [{"text": "a"}]})", "input[0]", "must be a string, a token id or an array of token ids"},
            {R"({"input": [true]})", "input[0]", "must be a string, a token id or an array of token ids"},
            {R"({"input": [null]})", "input[0]", "must be a string, a token id or an array of token ids"},
            {R"({"input": ["a", "BOOM"]})", "input[1]", "could not tokenize 'input[1]': the tokenizer exploded"},
            {R"({"input": "a", "encoding_format": "int8"})", "encoding_format", "must be 'float' or 'base64'"},
            {R"({"input": "a", "encoding_format": 1})", "encoding_format", "must be 'float' or 'base64'"},
            {R"({"input": "a", "dimensions": 3})", "dimensions", "'dimensions' must be 4, the model's"},
            {R"({"input": "a", "dimensions": -4})", "dimensions", "'dimensions' must be 4"},
            {R"({"input": "a", "dimensions": 4.0})", "dimensions", "'dimensions' must be 4"},
            {R"({"input": "a", "dimensions": "4"})", "dimensions", "'dimensions' must be 4"},
        };
        for (const Case& c : cases) {
            CAPTURE(c.body);
            try {
                parse(c.body);
                FAIL("expected a refusal");
            } catch (const xs::server::InvalidRequest& e) {
                CHECK(e.param() == c.param);
                CHECK(std::string(e.what()).find(c.message) != std::string::npos);
            }
        }
    }

    TEST_CASE("the response: one entry per input in order, floats or their float32 bytes in base64, usage") {
        const xs::EmbeddingResult result = fake_embed({{1, 5, 2}, {3}});
        const Json floats = xs::server::embeddings_response(result, false, "m");
        CHECK(floats.dump().rfind(R"({"object":"list","data":[{"object":"embedding","index":0,"embedding":[)", 0) == 0);
        CHECK(floats["model"] == "m");
        CHECK(floats["usage"] == Json{{"prompt_tokens", 4}, {"total_tokens", 4}});
        REQUIRE(floats["data"].size() == 2);
        CHECK(floats["data"][1]["index"] == 1);
        CHECK(floats["data"][1]["embedding"].get<std::vector<float>>() == fake_vector({3}));
        const Json base64 = xs::server::embeddings_response(result, true, "m");
        for (size_t i = 0; i < 2; ++i) {
            REQUIRE(base64["data"][i]["embedding"].is_string());
            CHECK(decode_floats(base64["data"][i]["embedding"].get<std::string>()) == result.embeddings[i]);
        }
        // The float values survive JSON's decimal text exactly, as float32.
        const Json parsed = Json::parse(floats.dump());
        CHECK(parsed["data"][0]["embedding"].get<std::vector<float>>() == result.embeddings[0]);
    }

    TEST_CASE("the server: embeddings, the model list, health, and no chat routes") {
        xs::EmbeddingService service(fake_spec(), fake_tokenize, fake_embed);
        RunningServer server(service);
        auto c = server.client();
        const auto res = c.Post("/v1/embeddings", R"({"input": ["hi", "a"], "model": "whatever", "user": "me"})",
                                "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        CHECK(res->get_header_value("Server-Timing").find("device;dur=") != std::string::npos);
        const Json out = Json::parse(res->body);
        CHECK(out["object"] == "list");
        CHECK(out["model"] == "fake-embedder");  // the served model, whatever the request named
        CHECK(out["usage"] == Json{{"prompt_tokens", 7}, {"total_tokens", 7}});
        REQUIRE(out["data"].size() == 2);
        CHECK(out["data"][0] == Json{{"object", "embedding"},
                                     {"index", 0},
                                     {"embedding", Json(fake_vector({1, 24, 25, 2}))}});
        const Json b64 = post(c, R"({"input": [[1, 14, 15, 2]], "encoding_format": "base64"})", 200);
        CHECK(decode_floats(b64["data"][0]["embedding"].get<std::string>()) == fake_vector({1, 24, 25, 2}));

        const Json bad = post(c, R"({"input": ["a", ""]})", 400);
        CHECK(bad["error"]["type"] == "invalid_request_error");
        CHECK(bad["error"]["param"] == "input[1]");
        CHECK(bad["error"]["code"].is_null());
        CHECK(post(c, "{not json", 400)["error"]["param"].is_null());
        CHECK(post(c, R"({"input": [99, 3]})", 500)["error"]["type"] == "server_error");
        post(c, R"({"input": "fine"})", 200);  // the service goes on after a failed request

        const Json models = Json::parse(c.Get("/v1/models")->body);
        CHECK(models["object"] == "list");
        CHECK(models["data"][0]["id"] == "fake-embedder");
        CHECK(models["data"][0]["context_length"] == 8);
        CHECK(c.Get("/health")->status == 200);
        for (const char* route : {"/v1/chat/completions", "/v1/completions", "/v1/systemone"})
            CHECK(c.Post(route, "{}", "application/json")->status == 404);
    }

    TEST_CASE("requests from several clients each get their own embeddings") {
        xs::EmbeddingService service(fake_spec(), fake_tokenize, fake_embed);
        RunningServer server(service);
        std::vector<std::thread> clients;
        std::vector<int> right(8, 0);
        for (int t = 0; t < 8; ++t) {
            clients.emplace_back([&, t] {
                auto c = server.client();
                Json ids = Json::array();
                for (int i = 0; i <= t % 5; ++i)
                    ids.push_back(10 + t);
                const auto res = c.Post("/v1/embeddings", Json{{"input", Json::array({ids, ids})}}.dump(),
                                        "application/json");
                if (!res || res->status != 200) return;
                const Json out = Json::parse(res->body);
                const auto expected = fake_vector(ids.get<std::vector<int32_t>>());
                right[static_cast<size_t>(t)] = out["data"].size() == 2 &&
                                                out["data"][1]["embedding"].get<std::vector<float>>() == expected;
            });
        }
        for (std::thread& t : clients)
            t.join();
        CHECK(right == std::vector<int>(8, 1));
    }

    TEST_CASE("a tiny model end to end: text through its tokenizer, ids as given, the runner's embeddings") {
        const xs::test::TempDir dir("xs_embedding_http_model");
        xs::test::TinyEncoder m;
        m.normalize = true;
        xs::test::write_tiny_encoder(dir.path, m, 31);
        xs::EmbeddingRunnerOptions o;
        o.backend = "cpu-ref";
        o.session.precision = xs::DType::F32;
        const auto tokenizer = xs::tok::Tokenizer::load(dir.path);
        const std::vector<int32_t> ids = xs::embedding_tokens(*tokenizer, "the cat");
        CHECK(ids == std::vector<int32_t>{2, 5, 6, 3});  // [CLS] the cat [SEP]
        const auto runner = xs::EmbeddingRunner::create(dir.path, o);
        const xs::EmbeddingResult expected = runner->embed({ids, {5, 6, 7}});
        const auto service = xs::EmbeddingService::create(xs::tok::Tokenizer::load(dir.path),
                                                          xs::EmbeddingRunner::create(dir.path, o));
        RunningServer server(*service);
        auto c = server.client();
        const Json out = post(c, R"({"input": "the cat"})", 200);
        CHECK(out["model"] == dir.path.filename().string());
        CHECK(out["usage"]["prompt_tokens"] == ids.size());
        CHECK(out["data"][0]["embedding"].get<std::vector<float>>() == expected.embeddings[0]);
        const Json b64 = post(c, R"({"input": [[5, 6, 7]], "encoding_format": "base64", "dimensions": 64})", 200);
        CHECK(decode_floats(b64["data"][0]["embedding"].get<std::string>()) == expected.embeddings[1]);
        CHECK(post(c, R"({"input": [5, 320]})", 400)["error"]["message"].get<std::string>().find("token id 320") !=
              std::string::npos);
        CHECK(post(c, Json{{"input", std::vector<int32_t>(49, 5)}}.dump(), 400)["error"]["message"]
                  .get<std::string>()
                  .find("49 tokens, more than the 48") != std::string::npos);
    }
}
