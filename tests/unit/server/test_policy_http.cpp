// The robot-policy API and its service against a fake policy: routes, request validation by name, latest wins.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "policy/service.h"
#include "server/policy_handlers.h"
#include "server/server.h"

namespace {

using nlohmann::json;

xs::PolicySpec fake_spec() {
    xs::PolicySpec s;
    s.id = "fake-policy";
    s.arch = "fake";
    s.cameras = {{"cam_a", 2, 2}, {"cam_b", 2, 2}};
    s.state_dim = 3;
    s.action_dim = 2;
    s.horizon = 4;
    s.noise_dim = 2;
    s.denoise_steps = 5;
    return s;
}

// actions[r][c] = r + 10 * cameras + 100 * state[0] + steps / 1000; prompt_ids: one per instruction character.
xs::ActResult fake_act(const xs::Observation& obs, const xs::ActOptions& o) {
    xs::ActResult r;
    r.actions = xs::HostTensor(xs::DType::F32, {4, 2});
    for (int64_t i = 0; i < 8; ++i) {
        r.actions.data<float>()[i] = static_cast<float>(i / 2) + 10.0f * static_cast<float>(obs.cameras.size()) +
                                     100.0f * obs.state[0] + static_cast<float>(o.steps) / 1000.0f;
    }
    r.prompt_ids.assign(obs.instruction.size(), 1);
    r.timing.total_ms = 1.0;
    return r;
}

std::string raw_image(int64_t w, int64_t h) {
    static const char* b64 = "AAAAAAAAAAAAAAAA";  // 12 zero bytes = 2 x 2 RGB
    return w == 2 && h == 2 ? b64 : "AAAA";
}

json act_body() {
    return {{"cameras", {{"cam_a", {{"width", 2}, {"height", 2}, {"data", raw_image(2, 2)}}}}},
            {"state", {0.5, 0.0, 0.0}},
            {"instruction", "pick"}};
}

class RunningPolicyServer {
public:
    explicit RunningPolicyServer(xs::PolicyService& service)
        : server_(service, options()), thread_([this] { server_.run(); }) {
        REQUIRE(server_.wait_until_listening(5000));
    }
    ~RunningPolicyServer() {
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

json post(httplib::Client& c, const json& body, int status) {
    const auto res = c.Post("/v1/policy/act", body.dump(), "application/json");
    REQUIRE(res);
    CHECK(res->status == status);
    return json::parse(res->body);
}

}  // namespace

TEST_SUITE("policy api") {
    TEST_CASE("info, ready and an action chunk") {
        xs::PolicyService service(fake_spec(), fake_act, [] {});
        RunningPolicyServer server(service);
        auto c = server.client();
        const auto info = c.Get("/v1/policy/info");
        REQUIRE(info);
        const json spec = json::parse(info->body);
        CHECK(spec["id"] == "fake-policy");
        CHECK(spec["cameras"].size() == 2);
        CHECK(spec["cameras"][1]["name"] == "cam_b");
        CHECK(spec["state_dim"] == 3);
        CHECK(spec["horizon"] == 4);
        CHECK(spec["steps"] == 5);
        for (int i = 0; i < 100 && !service.ready(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const auto ready = c.Get("/ready");
        REQUIRE(ready);
        CHECK(ready->status == 200);

        json body = act_body();
        body["seed"] = 7;
        const json out = post(c, body, 200);
        REQUIRE(out["actions"].size() == 4);
        CHECK(out["actions"][3][1].get<float>() == doctest::Approx(3 + 10 + 50 + 0.005));
        CHECK(out["seed"] == 7);
        CHECK(out["steps"] == 5);
        CHECK(out["prompt_tokens"] == 4);
        CHECK(out["timing"].contains("queue_ms"));

        body["steps"] = 3;  // the request's flow steps reach the policy
        CHECK(post(c, body, 200)["actions"][0][0].get<float>() == doctest::Approx(0 + 10 + 50 + 0.003));
        // a PNG (1 x 1) decodes and is refused for its size, by camera name
        json png = act_body();
        png["cameras"]["cam_b"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
                                  "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
        const json err = post(c, png, 400);
        CHECK(err["error"]["param"] == "cameras.cam_b");
        CHECK(err["error"]["message"].get<std::string>().find("1x1") != std::string::npos);
    }

    TEST_CASE("requests fail by name") {
        xs::PolicyService service(fake_spec(), fake_act, [] {});
        RunningPolicyServer server(service);
        auto c = server.client();
        struct Case {
            const char* param;
            json body;
        };
        json unknown = act_body();
        unknown["cameras"]["cam_z"] = raw_image(2, 2);
        json short_state = act_body();
        short_state["state"] = {1.0, 2.0};
        json no_instruction = act_body();
        no_instruction.erase("instruction");
        json bad_bytes = act_body();
        bad_bytes["cameras"]["cam_a"]["data"] = raw_image(1, 1);
        json bad_b64 = act_body();
        bad_b64["cameras"]["cam_a"] = "not base64 !";
        json rtc = act_body();
        rtc["previous_actions"] = json::array();
        json noise = act_body();
        noise["noise"] = {{1.0, 2.0}};
        json no_cameras = act_body();
        no_cameras["cameras"] = json::object();
        for (const Case& k : {Case{"cameras.cam_z", unknown}, Case{"state", short_state},
                              Case{"instruction", no_instruction}, Case{"cameras.cam_a.data", bad_bytes},
                              Case{"cameras.cam_a", bad_b64}, Case{"previous_actions", rtc}, Case{"noise", noise},
                              Case{"cameras", no_cameras}}) {
            CAPTURE(k.param);
            CHECK(post(c, k.body, 400)["error"]["param"] == k.param);
        }
    }

    TEST_CASE("noise given reproduces a call; the seed is then null") {
        const xs::PolicySpec spec = fake_spec();
        json body = act_body();
        body["noise"] = json::array();
        for (int r = 0; r < 4; ++r) body["noise"].push_back({0.1 * r, -0.1 * r});
        const xs::server::PolicyRequest req = xs::server::parse_policy_request(body, spec);
        REQUIRE(req.options.noise != nullptr);
        CHECK(req.options.noise->data<float>()[6] == doctest::Approx(0.3f));
        xs::PolicyCall call;
        call.result = fake_act(req.obs, req.options);
        CHECK(xs::server::policy_act_json(call, req.options)["seed"].is_null());
    }

    TEST_CASE("the latest observation wins: a waiting call is superseded, the running one finishes") {
        std::promise<void> release;
        std::shared_future<void> released = release.get_future().share();
        std::atomic<int> started = 0;
        xs::PolicyService service(
            fake_spec(),
            [&](const xs::Observation& obs, const xs::ActOptions& o) {
                if (started.fetch_add(1) == 0) released.wait();  // the first call holds the device
                return fake_act(obs, o);
            },
            [] {});
        const auto call = [&](float s0) {
            xs::Observation obs;
            obs.cameras.emplace_back("cam_a", xs::image::Image{});
            obs.state = {s0, 0, 0};
            obs.instruction = "x";
            xs::ActOptions o;
            o.steps = 1;
            return service.act(obs, o);
        };
        auto first = std::async(std::launch::async, call, 1.0f);
        while (started.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto second = std::async(std::launch::async, call, 2.0f);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // second is waiting
        auto third = std::async(std::launch::async, call, 3.0f);
        CHECK_THROWS_AS(second.get(), xs::Superseded);
        release.set_value();
        CHECK(first.get().result.actions.data<float>()[0] == doctest::Approx(10 + 100 + 0.001));
        CHECK(third.get().result.actions.data<float>()[0] == doctest::Approx(10 + 300 + 0.001));
        CHECK(started.load() == 2);  // the superseded call never ran
    }
}
