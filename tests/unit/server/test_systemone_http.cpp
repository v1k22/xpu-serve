// The System One API and the decision service against a fake model: the request schema refused field by field with
// the API's error bodies, each question type's answer, requests answered in order.
#include <doctest/doctest.h>

#include <thread>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "decision/service.h"
#include "server/decision_handlers.h"
#include "server/server.h"

namespace {

using Json = nlohmann::ordered_json;

xs::DecisionSpec fake_spec() {
    xs::DecisionSpec s;
    s.id = "fake-decider";
    s.arch = "fake";
    s.max_tokens = 100;
    return s;
}

// Option o of a question with n options gets probability (o + 1) / (1 + ... + n): the last one wins. A state that is
// the text "too long" is refused as the model would refuse one over its limit.
xs::DecisionResult fake_decide(const xs::DecisionRequest& request) {
    if (request.state.is_string() && request.state.get<std::string>() == "too long") {
        throw xs::DecisionRefused("the request is 101 tokens; the limit is 100", "max_tokens_exceeded");
    }
    if (request.state.is_string() && request.state.get<std::string>() == "broken") xs::fail("the device fell over");
    xs::DecisionResult r;
    for (const xs::DecisionQuestion& q : request.questions) {
        std::vector<double> p;
        for (size_t o = 0; o < q.options.size(); ++o)
            p.push_back(static_cast<double>(o + 1));
        r.answers.push_back(xs::make_answer(q.type, p));
    }
    r.input_tokens = 42;
    r.steps = 1;
    r.timing.total_ms = 1.5;
    return r;
}

class RunningServer {
public:
    explicit RunningServer(xs::DecisionService& service)
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

Json body() {
    return Json::parse(R"({
        "model": "anything",
        "state": {"b": 1, "a": "text"},
        "questions": {
            "zone": {"type": "choice", "instructions": "Where?", "criteria": {"north": "up", "east": null, "south": ["down"]}},
            "open": {"type": "noul", "criteria": {"true": "it is open"}},
            "level": {"type": "score", "instructions": {"rate": "it"}, "criteria": ["low", "mid", {"name": "high"}]}
        }
    })");
}

Json post(httplib::Client& c, const std::string& text, int status) {
    const auto res = c.Post("/v1/systemone", text, "application/json");
    REQUIRE(res);
    CHECK(res->status == status);
    return Json::parse(res->body);
}

// The loc of a 422's first error, joined with dots.
std::string where(const Json& error) {
    std::string out;
    for (const Json& part : error["detail"][0]["loc"])
        out += (out.empty() ? "" : ".") + part.get<std::string>();
    return out;
}

}  // namespace

TEST_SUITE("systemone api") {
    TEST_CASE("a request keeps its order and names what it leaves out") {
        const xs::DecisionRequest r = xs::server::parse_systemone_request(body());
        CHECK(r.state.begin().key() == "b");  // the client's key order, not the sorted one
        REQUIRE(r.questions.size() == 3);
        CHECK(r.questions[0].id == "zone");
        CHECK(r.questions[0].options.size() == 3);
        CHECK(r.questions[0].options[0].id == "north");
        CHECK(r.questions[0].options[1].description->is_null());  // named, not described
        CHECK(r.questions[1].type == xs::QuestionType::Noul);
        CHECK(r.questions[1].instructions.is_null());
        CHECK(r.questions[1].options[0].id == "true");
        CHECK(r.questions[1].options[0].description->get<std::string>() == "it is open");
        CHECK(!r.questions[1].options[1].description.has_value());  // the request says nothing of false
        CHECK(r.questions[2].options[2].id == "2");
        CHECK(r.questions[2].instructions.is_object());
    }

    TEST_CASE("answers of the three question types") {
        xs::DecisionService service(fake_spec(), 100, fake_decide);
        RunningServer server(service);
        auto c = server.client();
        const auto res = c.Post("/v1/systemone", body().dump(), "application/json");
        REQUIRE(res);
        REQUIRE(res->status == 200);
        CHECK(res->get_header_value("Server-Timing").find("device;dur=") != std::string::npos);
        const Json out = Json::parse(res->body);
        CHECK(out["model"] == "fake-decider");
        CHECK(out["usage"] == Json{{"input_tokens", 42}, {"output_tokens", 0}});
        CHECK(out["answers"].begin().key() == "zone");  // in the request's order
        const Json& zone = out["answers"]["zone"];
        CHECK(zone["type"] == "choice");
        CHECK(zone["choice"] == "south");
        CHECK(zone["probabilities"]["north"].get<double>() == doctest::Approx(1.0 / 6));
        CHECK(zone["confidence"].get<double>() == doctest::Approx((0.5 - 1.0 / 3) / (1 - 1.0 / 3)));
        const Json& open = out["answers"]["open"];
        CHECK(open == Json{{"type", "noul"}, {"noul", open["noul"]}});  // nothing but the probability of true
        CHECK(open["noul"].get<double>() == doctest::Approx(1.0 / 3));
        const Json& level = out["answers"]["level"];
        CHECK(level["score"].get<double>() == doctest::Approx((0 * 1 + 1 * 2 + 2 * 3) / 6.0));
        CHECK(level["legend"] == Json{{"0", "low"}, {"1", "mid"}, {"2", Json{{"name", "high"}}}});
        CHECK(level["probabilities"]["2"].get<double>() == doctest::Approx(0.5));
        CHECK(level["confidence"].get<double>() == doctest::Approx(1 - (1.0 / 6 * 2 + 2.0 / 6) / (2.0 / 3)));

        const auto models = c.Get("/v1/models");
        REQUIRE(models);
        const Json list = Json::parse(models->body);
        CHECK(list["models"][0]["name"] == "fake-decider");
        CHECK(list["data"][0]["id"] == "fake-decider");
        CHECK(c.Get("/health")->status == 200);
        // A decision model generates no text.
        CHECK(c.Post("/v1/chat/completions", "{}", "application/json")->status == 404);
    }

    TEST_CASE("a body that does not follow the schema is a 422 naming the field") {
        xs::DecisionService service(fake_spec(), 100, fake_decide);
        RunningServer server(service);
        auto c = server.client();
        struct Case {
            const char* edit;  // a JSON merge patch on the valid body
            const char* loc;
            const char* type;
        };
        const Case cases[] = {
            {R"({"model": null})", "body.model", "missing"},
            {R"({"model": 3})", "body.model", "string_type"},
            {R"({"state": null})", "body.state", "missing"},
            {R"({"state": 12})", "body.state", "content_type"},
            {R"({"questions": null})", "body.questions", "missing"},
            {R"({"questions": []})", "body.questions", "dict_type"},
            {R"({"questions": {"zone": null, "open": null, "level": null}})", "body.questions", "too_short"},
            {R"({"temperature": 0.5})", "body.temperature", "extra_forbidden"},
            {R"({"assets": {"IMG": {"type": "image"}}})", "body.assets", "unsupported"},
            {R"({"images": ["data:image/png;base64,AAAA"]})", "body.images", "unsupported"},
            {R"({"questions": {"zone": {"type": null}}})", "body.questions.zone.type", "missing"},
            {R"({"questions": {"zone": {"type": "rank"}}})", "body.questions.zone.type", "union_tag_invalid"},
            {R"({"questions": {"zone": {"type": "box"}}})", "body.questions.zone.type", "unsupported"},
            {R"({"questions": {"zone": {"weight": 2}}})", "body.questions.zone.weight", "extra_forbidden"},
            {R"({"questions": {"zone": {"instructions": 7}}})", "body.questions.zone.instructions", "content_type"},
            {R"({"questions": {"zone": {"criteria": null}}})", "body.questions.zone.criteria", "missing"},
            {R"({"questions": {"zone": {"criteria": {"north": null, "east": null, "south": null}}}})",
             "body.questions.zone.criteria", "too_short"},
            {R"({"questions": {"zone": {"criteria": {"north": 5}}}})", "body.questions.zone.criteria.north",
             "content_type"},
            {R"({"questions": {"open": {"criteria": {"maybe": "x"}}}})", "body.questions.open.criteria.maybe",
             "extra_forbidden"},
            {R"({"questions": {"open": {"criteria": ["yes", "no"]}}})", "body.questions.open.criteria", "dict_type"},
            {R"({"questions": {"level": {"criteria": {"0": "low"}}}})", "body.questions.level.criteria", "list_type"},
            {R"({"questions": {"level": {"criteria": ["low", null]}}})", "body.questions.level.criteria.1",
             "content_type"},
            {R"({"questions": {"level": {"criteria": ["0","1","2","3","4","5","6","7","8","9","10"]}}})",
             "body.questions.level.criteria", "too_long"},
        };
        for (const Case& t : cases) {
            CAPTURE(t.edit);
            Json b = body();
            b.merge_patch(Json::parse(t.edit));
            const Json error = post(c, b.dump(), 422);
            CHECK(where(error) == t.loc);
            CHECK(error["detail"][0]["type"] == t.type);
            CHECK(!error["detail"][0]["msg"].get<std::string>().empty());
        }
        Json many = body();
        for (int i = 0; i < 256; ++i)
            many["questions"]["zone"]["criteria"]["o" + std::to_string(i)] = nullptr;
        CHECK(post(c, many.dump(), 422)["detail"][0]["type"] == "too_long");
        CHECK(post(c, "[1, 2]", 422)["detail"][0]["type"] == "dict_type");
    }

    TEST_CASE("what is not JSON, too long for the model, or fails inside it") {
        xs::DecisionService service(fake_spec(), 100, fake_decide);
        RunningServer server(service);
        auto c = server.client();
        CHECK(post(c, "{not json", 400)["detail"].is_string());
        Json b = body();
        b["state"] = "too long";
        const Json refused = post(c, b.dump(), 400);
        CHECK(refused["detail"]["type"] == "max_tokens_exceeded");
        CHECK(refused["detail"]["msg"].get<std::string>().find("101 tokens") != std::string::npos);
        b["state"] = "broken";
        CHECK(post(c, b.dump(), 500)["detail"].get<std::string>().find("fell over") != std::string::npos);
        b["state"] = "fine";
        post(c, b.dump(), 200);  // the service goes on after a failed request
    }

    TEST_CASE("requests from several clients each get their own answer") {
        xs::DecisionService service(fake_spec(), 100, fake_decide);
        RunningServer server(service);
        std::vector<std::thread> clients;
        std::vector<int> options(8, 0);
        for (int t = 0; t < 8; ++t) {
            clients.emplace_back([&, t] {
                auto c = server.client();
                Json b = body();
                b["questions"] = Json{{"q", Json{{"type", "choice"}, {"criteria", Json::object()}}}};
                for (int o = 0; o <= t; ++o)
                    b["questions"]["q"]["criteria"]["o" + std::to_string(o)] = nullptr;
                const auto res = c.Post("/v1/systemone", b.dump(), "application/json");
                if (res && res->status == 200) {
                    options[static_cast<size_t>(t)] =
                        static_cast<int>(Json::parse(res->body)["answers"]["q"]["probabilities"].size());
                }
            });
        }
        for (std::thread& t : clients)
            t.join();
        for (int t = 0; t < 8; ++t)
            CHECK(options[static_cast<size_t>(t)] == t + 1);
    }
}
