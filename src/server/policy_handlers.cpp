#include "server/policy_handlers.h"

#include <atomic>
#include <random>
#include <set>

#include "image/image.h"
#include "server/openai_format.h"
#include "server/openai_request.h"
#include "util/base64.h"

namespace xs::server {
namespace {

using nlohmann::json;

std::vector<uint8_t> decode_base64_field(std::string_view text, const std::string& param) {
    if (text.rfind("data:", 0) == 0) {  // data URL: keep what follows ";base64,"
        const size_t at = text.find(";base64,");
        if (at == std::string_view::npos) throw InvalidRequest(str_cat("'", param, "' is a data URL without ;base64,"), param);
        text = text.substr(at + 8);
    }
    std::optional<std::vector<uint8_t>> bytes = base64_decode(text);
    if (!bytes) throw InvalidRequest(str_cat("'", param, "' is not valid base64"), param);
    return std::move(*bytes);
}

image::Image parse_image(const json& v, const CameraSpec& cam, const std::string& param) {
    image::Image img;
    if (v.is_string()) {
        const std::vector<uint8_t> bytes = decode_base64_field(v.get<std::string>(), param);
        try {
            img = image::decode(bytes);
        } catch (const std::exception& e) {
            throw InvalidRequest(str_cat("'", param, "': ", e.what()), param);
        }
    } else if (v.is_object()) {
        if (!v.contains("width") || !v["width"].is_number_integer() || !v.contains("height") ||
            !v["height"].is_number_integer() || !v.contains("data") || !v["data"].is_string()) {
            throw InvalidRequest(str_cat("'", param, "' must be a base64 PNG / JPEG or {width, height, data}"), param);
        }
        img.width = v["width"].get<int64_t>();
        img.height = v["height"].get<int64_t>();
        img.rgb = decode_base64_field(v["data"].get<std::string>(), param + ".data");
        if (img.width <= 0 || img.height <= 0 ||
            static_cast<int64_t>(img.rgb.size()) != img.width * img.height * 3) {
            throw InvalidRequest(str_cat("'", param, ".data' holds ", img.rgb.size(), " bytes, want width * height * 3 = ",
                                         std::max<int64_t>(img.width * img.height * 3, 0)),
                                 param + ".data");
        }
    } else {
        throw InvalidRequest(str_cat("'", param, "' must be a base64 PNG / JPEG or {width, height, data}"), param);
    }
    if (img.width != cam.width || img.height != cam.height) {
        throw InvalidRequest(str_cat("'", param, "' is ", img.width, "x", img.height, "; the policy reads ", cam.width,
                                     "x", cam.height),
                             param);
    }
    return img;
}

uint64_t fresh_seed() {
    static std::atomic<uint64_t> counter{std::random_device{}()};
    return counter.fetch_add(0x9E3779B97F4A7C15ull);
}

}  // namespace

PolicyRequest parse_policy_request(const json& body, const PolicySpec& spec) {
    if (!body.is_object()) throw InvalidRequest("the body must be a JSON object", "");
    for (const char* unsupported : {"previous_actions", "delay"}) {
        if (body.contains(unsupported)) {
            throw InvalidRequest(str_cat("'", unsupported, "' (real-time chunking) is not supported yet"), unsupported);
        }
    }
    PolicyRequest r;
    if (!body.contains("cameras") || !body["cameras"].is_object() || body["cameras"].empty()) {
        throw InvalidRequest("'cameras' must be an object of camera name -> image, with at least one camera", "cameras");
    }
    const json& cams = body["cameras"];
    std::set<std::string> known;
    for (const CameraSpec& cam : spec.cameras) {  // in the order the network reads them
        known.insert(cam.name);
        if (cams.contains(cam.name)) r.obs.cameras.emplace_back(cam.name, parse_image(cams[cam.name], cam, "cameras." + cam.name));
    }
    for (const auto& [name, v] : cams.items()) {
        if (!known.count(name)) {
            std::string names;
            for (const CameraSpec& cam : spec.cameras) names += (names.empty() ? "" : ", ") + cam.name;
            throw InvalidRequest(str_cat("unknown camera '", name, "' (the policy's: ", names, ")"), "cameras." + name);
        }
    }
    if (!body.contains("state") || !body["state"].is_array()) throw InvalidRequest("'state' must be an array of numbers", "state");
    for (const json& v : body["state"]) {
        if (!v.is_number()) throw InvalidRequest("'state' must be an array of numbers", "state");
        r.obs.state.push_back(v.get<float>());
    }
    if (static_cast<int64_t>(r.obs.state.size()) != spec.state_dim) {
        throw InvalidRequest(str_cat("'state' has ", r.obs.state.size(), " values; the policy reads ", spec.state_dim), "state");
    }
    if (!body.contains("instruction") || !body["instruction"].is_string()) {
        throw InvalidRequest("'instruction' must be a string", "instruction");
    }
    r.obs.instruction = body["instruction"].get<std::string>();
    if (body.contains("steps")) {
        if (!body["steps"].is_number_integer() || body["steps"].get<int64_t>() < 1 || body["steps"].get<int64_t>() > 1000) {
            throw InvalidRequest("'steps' must be an integer from 1 to 1000", "steps");
        }
        r.options.steps = body["steps"].get<int>();
    } else {
        r.options.steps = spec.denoise_steps;
    }
    if (body.contains("seed")) {
        if (!body["seed"].is_number_unsigned()) throw InvalidRequest("'seed' must be a non-negative integer", "seed");
        r.options.seed = body["seed"].get<uint64_t>();
    } else {
        r.options.seed = fresh_seed();
    }
    if (body.contains("noise")) {  // [horizon][noise_dim], to reproduce a reference
        const json& n = body["noise"];
        std::vector<float> values;
        bool ok = n.is_array() && static_cast<int64_t>(n.size()) == spec.horizon;
        for (size_t i = 0; ok && i < n.size(); ++i) {
            ok = n[i].is_array() && static_cast<int64_t>(n[i].size()) == spec.noise_dim;
            for (size_t j = 0; ok && j < n[i].size(); ++j) {
                ok = n[i][j].is_number();
                if (ok) values.push_back(n[i][j].get<float>());
            }
        }
        if (!ok) {
            throw InvalidRequest(str_cat("'noise' must be ", spec.horizon, " rows of ", spec.noise_dim, " numbers"), "noise");
        }
        r.noise = HostTensor::from_vector(DType::F32, {spec.horizon, spec.noise_dim}, values);
        r.options.noise = &*r.noise;
    }
    return r;
}

json policy_info_json(const PolicySpec& spec) {
    json cams = json::array();
    for (const CameraSpec& c : spec.cameras) cams.push_back({{"name", c.name}, {"width", c.width}, {"height", c.height}});
    return {{"id", spec.id},        {"arch", spec.arch},         {"cameras", cams},
            {"state_dim", spec.state_dim}, {"action_dim", spec.action_dim}, {"horizon", spec.horizon},
            {"noise_dim", spec.noise_dim}, {"steps", spec.denoise_steps}};
}

json policy_act_json(const PolicyCall& call, const ActOptions& options) {
    const HostTensor& a = call.result.actions;
    const int64_t rows = a.dim(0);
    const int64_t cols = a.dim(1);
    json actions = json::array();
    const float* p = a.data<float>();
    for (int64_t r = 0; r < rows; ++r) actions.push_back(std::vector<float>(p + r * cols, p + (r + 1) * cols));
    const ActTiming& t = call.result.timing;
    return {{"actions", actions},
            {"seed", options.noise != nullptr ? json(nullptr) : json(options.seed)},
            {"steps", options.steps},
            {"prompt_tokens", call.result.prompt_ids.size()},
            {"timing",
             {{"queue_ms", call.queue_ms},
              {"prepare_ms", t.prepare_ms},
              {"device_ms", t.device_ms},
              {"finish_ms", t.finish_ms},
              {"total_ms", t.total_ms}}}};
}

void PolicyHandlers::info(const httplib::Request&, httplib::Response& res) const {
    res.set_content(policy_info_json(service_.spec()).dump(), "application/json");
}

void PolicyHandlers::act(const httplib::Request& req, httplib::Response& res) const {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const json::exception& e) {
        throw InvalidRequest(str_cat("the body is not valid JSON: ", e.what()), "");
    }
    PolicyRequest r = parse_policy_request(body, service_.spec());
    try {
        const PolicyCall call = service_.act(std::move(r.obs), r.options);
        res.set_content(policy_act_json(call, r.options).dump(), "application/json");
    } catch (const Superseded& e) {
        res.status = 409;
        res.set_content(error_body(e.what(), "superseded", {}, "superseded").dump(), "application/json");
    }
}

void PolicyHandlers::ready(const httplib::Request&, httplib::Response& res) const {
    const bool ok = service_.ready();
    res.status = ok ? 200 : 503;
    res.set_content(ok ? R"({"status":"ready"})" : R"({"status":"warming up"})", "application/json");
}

}  // namespace xs::server
