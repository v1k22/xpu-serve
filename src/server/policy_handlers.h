#pragma once

// The robot-policy API (docs/design-vla.md section 4.6), JSON form:
//   GET  /v1/policy/info  the PolicySpec: cameras (names, sizes), state and action dims, horizon, flow steps
//   POST /v1/policy/act   {"cameras": {name: image}, "state": [...], "instruction": "...", "seed"?, "steps"?, "noise"?}
//                         -> {"actions": [[...]], "seed", "steps", "prompt_tokens", "timing": {...}}
//   GET  /ready           200 once warmed up, 503 before
// An image is base64 PNG / JPEG (a data URL is fine) or {"width", "height", "data": base64 of RGB bytes}. Requests
// fail by name; a call that a newer observation replaced while it waited gets 409 (latest wins).

#include <cstdint>
#include <optional>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "policy/service.h"

namespace xs::server {

struct PolicyRequest {
    Observation obs;
    ActOptions options;
    std::optional<HostTensor> noise;  // what options.noise points at, when given
};

// Throws InvalidRequest naming the field.
PolicyRequest parse_policy_request(const nlohmann::json& body, const PolicySpec& spec);
nlohmann::json policy_info_json(const PolicySpec& spec);
nlohmann::json policy_act_json(const PolicyCall& call, const ActOptions& options);

class PolicyHandlers {
public:
    explicit PolicyHandlers(PolicyService& service) : service_(service) {}
    void info(const httplib::Request& req, httplib::Response& res) const;
    void act(const httplib::Request& req, httplib::Response& res) const;
    void ready(const httplib::Request& req, httplib::Response& res) const;

private:
    PolicyService& service_;
};

}  // namespace xs::server
