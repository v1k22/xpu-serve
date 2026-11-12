#include "server/decision_handlers.h"

#include <cstdio>

#include "util/log.h"

namespace xs::server {
namespace {

using Json = nlohmann::ordered_json;
using Loc = std::vector<std::string>;

constexpr size_t kMaxChoices = 255;
constexpr size_t kMaxLevels = 10;

Loc at(Loc loc, const std::string& key) {
    loc.push_back(key);
    return loc;
}

// States, instructions and descriptions are text or structured content.
bool is_content(const Json& v) {
    return v.is_string() || v.is_object() || v.is_array();
}

void check_keys(const Json& object, std::initializer_list<const char*> known, const Loc& loc) {
    for (auto it = object.begin(); it != object.end(); ++it) {
        bool found = false;
        for (const char* k : known)
            found |= it.key() == k;
        if (!found) throw SystemOneInvalid(at(loc, it.key()), "Extra inputs are not permitted", "extra_forbidden");
    }
}

DecisionQuestion parse_question(const std::string& name, const Json& q, const Loc& loc) {
    if (name.empty()) throw SystemOneInvalid(loc, "A question needs a name", "string_too_short");
    if (!q.is_object()) throw SystemOneInvalid(loc, "Input should be an object", "dict_type");
    if (!q.contains("type")) throw SystemOneInvalid(at(loc, "type"), "Field required", "missing");
    const std::string type = q["type"].is_string() ? q["type"].get<std::string>() : "";
    DecisionQuestion out;
    out.id = name;
    if (type == "noul") {
        out.type = QuestionType::Noul;
    } else if (type == "choice") {
        out.type = QuestionType::Choice;
    } else if (type == "score") {
        out.type = QuestionType::Score;
    } else if (type == "box") {
        throw SystemOneInvalid(at(loc, "type"),
                               "A box question locates something in an image; this server reads no images",
                               "unsupported");
    } else {
        throw SystemOneInvalid(at(loc, "type"), "Input should be 'noul', 'choice' or 'score'", "union_tag_invalid");
    }
    check_keys(q, {"type", "instructions", "criteria"}, loc);
    if (q.contains("instructions")) {
        const Json& i = q["instructions"];
        if (!i.is_null() && !is_content(i)) {
            throw SystemOneInvalid(at(loc, "instructions"), "Input should be a string, an object, an array or null",
                                   "content_type");
        }
        out.instructions = i;
    }
    const Loc criteria_loc = at(loc, "criteria");
    const Json* criteria = q.contains("criteria") ? &q["criteria"] : nullptr;
    if (out.type == QuestionType::Noul) {
        out.options = {{"true", std::nullopt}, {"false", std::nullopt}};
        if (criteria == nullptr || criteria->is_null()) return out;
        if (!criteria->is_object())
            throw SystemOneInvalid(criteria_loc, "Input should be an object with 'true' and 'false'", "dict_type");
        check_keys(*criteria, {"true", "false"}, criteria_loc);
        for (DecisionOption& option : out.options) {
            if (!criteria->contains(option.id)) continue;
            const Json& d = (*criteria)[option.id];
            if (!d.is_null() && !is_content(d)) {
                throw SystemOneInvalid(at(criteria_loc, option.id),
                                       "Input should be a string, an object, an array or null", "content_type");
            }
            option.description = d;
        }
        return out;
    }
    if (criteria == nullptr) throw SystemOneInvalid(criteria_loc, "Field required", "missing");
    if (out.type == QuestionType::Choice) {
        if (!criteria->is_object()) {
            throw SystemOneInvalid(criteria_loc, "Input should be an object of choice name -> description or null",
                                   "dict_type");
        }
        if (criteria->empty()) throw SystemOneInvalid(criteria_loc, "A choice needs at least 1 option", "too_short");
        if (criteria->size() > kMaxChoices) {
            throw SystemOneInvalid(criteria_loc,
                                   str_cat("A choice has at most ", kMaxChoices, " options, got ", criteria->size()),
                                   "too_long");
        }
        for (auto it = criteria->begin(); it != criteria->end(); ++it) {
            if (!it.value().is_null() && !is_content(it.value())) {
                throw SystemOneInvalid(at(criteria_loc, it.key()),
                                       "Input should be a string, an object, an array or null", "content_type");
            }
            out.options.push_back({it.key(), it.value()});
        }
        return out;
    }
    if (!criteria->is_array())
        throw SystemOneInvalid(criteria_loc, "Input should be an array of level descriptions, lowest first",
                               "list_type");
    if (criteria->empty()) throw SystemOneInvalid(criteria_loc, "A score needs at least 1 level", "too_short");
    if (criteria->size() > kMaxLevels) {
        throw SystemOneInvalid(
            criteria_loc, str_cat("A score has at most ", kMaxLevels, " levels, got ", criteria->size()), "too_long");
    }
    for (size_t i = 0; i < criteria->size(); ++i) {
        if (!is_content((*criteria)[i])) {
            throw SystemOneInvalid(at(criteria_loc, std::to_string(i)),
                                   "Input should be a string, an object or an array", "content_type");
        }
        out.options.push_back({std::to_string(i), (*criteria)[i]});
    }
    return out;
}

Json validation_body(const SystemOneInvalid& e) {
    Json loc = Json::array({"body"});
    for (const std::string& part : e.loc())
        loc.push_back(part);
    return Json{{"detail", Json::array({Json{{"loc", loc}, {"msg", e.what()}, {"type", e.type()}}})}};
}

void reply(httplib::Response& res, int status, const Json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

}  // namespace

DecisionRequest parse_systemone_request(const Json& body) {
    if (!body.is_object()) throw SystemOneInvalid({}, "Input should be an object", "dict_type");
    for (const char* media : {"assets", "images", "videos"}) {
        if (body.contains(media))
            throw SystemOneInvalid({media}, "This server reads no images or video", "unsupported");
    }
    check_keys(body, {"model", "state", "questions"}, {});
    if (!body.contains("model")) throw SystemOneInvalid({"model"}, "Field required", "missing");
    if (!body["model"].is_string()) throw SystemOneInvalid({"model"}, "Input should be a valid string", "string_type");
    if (!body.contains("state")) throw SystemOneInvalid({"state"}, "Field required", "missing");
    if (!is_content(body["state"])) {
        throw SystemOneInvalid({"state"}, "Input should be a string, an object or an array", "content_type");
    }
    if (!body.contains("questions")) throw SystemOneInvalid({"questions"}, "Field required", "missing");
    const Json& questions = body["questions"];
    if (!questions.is_object())
        throw SystemOneInvalid({"questions"}, "Input should be an object of question name -> question", "dict_type");
    if (questions.empty()) throw SystemOneInvalid({"questions"}, "A request needs at least 1 question", "too_short");
    DecisionRequest request;
    request.state = body["state"];
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        request.questions.push_back(parse_question(it.key(), it.value(), {"questions", it.key()}));
    }
    return request;
}

Json systemone_response(const DecisionRequest& request, const DecisionResult& result, const std::string& model) {
    XS_CHECK(result.answers.size() == request.questions.size(), "an answer per question");
    Json answers = Json::object();
    for (size_t i = 0; i < request.questions.size(); ++i) {
        const DecisionQuestion& q = request.questions[i];
        const DecisionAnswer& a = result.answers[i];
        Json out = Json{{"type", question_type_name(q.type)}};
        if (q.type == QuestionType::Noul) {
            out["noul"] = a.noul;
        } else {
            Json probabilities = Json::object();
            for (size_t o = 0; o < q.options.size(); ++o)
                probabilities[q.options[o].id] = a.probabilities[o];
            if (q.type == QuestionType::Choice) {
                out["choice"] = q.options[a.best].id;
                out["confidence"] = a.confidence;
            } else {
                Json legend = Json::object();
                for (const DecisionOption& option : q.options)
                    legend[option.id] = *option.description;
                out["score"] = a.score;
                out["confidence"] = a.confidence;
                out["legend"] = std::move(legend);
            }
            out["probabilities"] = std::move(probabilities);
        }
        answers[q.id] = std::move(out);
    }
    return Json{{"model", model},
                {"answers", std::move(answers)},
                {"usage", Json{{"input_tokens", result.input_tokens}, {"output_tokens", 0}}}};
}

void DecisionHandlers::decide(const httplib::Request& req, httplib::Response& res) const {
    Json body = Json::parse(req.body, nullptr, false);
    if (body.is_discarded()) {
        log::info(req.method, " ", req.path, " -> 400: the body is not JSON");
        return reply(res, 400, Json{{"detail", "the body is not valid JSON"}});
    }
    try {
        const DecisionRequest request = parse_systemone_request(body);
        const DecisionCall call = service_.decide(request);
        const DecisionTiming& t = call.result.timing;
        char timing[160];
        std::snprintf(timing, sizeof timing, "queue;dur=%.2f, prompt;dur=%.2f, device;dur=%.2f, total;dur=%.2f",
                      call.queue_ms, t.prompt_ms, t.device_ms, t.total_ms);
        res.set_header("Server-Timing", timing);
        log::info(req.method, " ", req.path, ": ", request.questions.size(), " questions, ", call.result.input_tokens,
                  " tokens, ", call.result.steps, " steps, ", static_cast<int>(t.total_ms * 10) / 10.0, " ms (device ",
                  static_cast<int>(t.device_ms * 10) / 10.0, ")");
        reply(res, 200, systemone_response(request, call.result, service_.spec().id));
    } catch (const SystemOneInvalid& e) {
        log::info(req.method, " ", req.path, " -> 422: ", e.what());
        reply(res, 422, validation_body(e));
    } catch (const DecisionRefused& e) {
        log::info(req.method, " ", req.path, " -> 400: ", e.what());
        reply(res, 400, Json{{"detail", Json{{"type", e.code()}, {"msg", e.what()}}}});
    } catch (const std::exception& e) {
        log::error(req.method, " ", req.path, " -> 500: ", e.what());
        reply(res, 500, Json{{"detail", e.what()}});
    }
}

void DecisionHandlers::models(const httplib::Request&, httplib::Response& res, int64_t created) const {
    const DecisionSpec& spec = service_.spec();
    const std::string description =
        str_cat("Decision model (", spec.arch, "): noul, choice and score questions over a text state of up to ",
                service_.max_tokens(), " tokens with its questions");
    reply(res, 200,
          Json{{"models", Json::array({Json{{"name", spec.id}, {"description", description}, {"release_date", ""}}})},
               {"object", "list"},
               {"data", Json::array({Json{
                            {"id", spec.id}, {"object", "model"}, {"created", created}, {"owned_by", "xpu-serve"}}})}});
}

}  // namespace xs::server
