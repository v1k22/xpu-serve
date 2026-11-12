#pragma once

// The System One API (TypeSafe's decision API, OpenAPI 0.2.0), for a decision model:
//   POST /v1/systemone  {"model", "state", "questions": {name: {"type": "noul" | "choice" | "score",
//                        "instructions"?, "criteria"?}}}
//                       -> {"model", "answers": {name: answer}, "usage": {"input_tokens", "output_tokens": 0}}
//   GET  /v1/models     {"models": [{"name", "description", "release_date"}]}, with OpenAI's "data" list beside it
// An answer: noul {"type", "noul": P(true)}; choice {"type", "choice", "confidence", "probabilities": {name: p}};
// score {"type", "score": the expected level, "confidence", "legend": {level: text}, "probabilities": {level: p}}.
// Errors have the API's bodies: 422 {"detail": [{"loc", "msg", "type"}]} for a body that does not follow the schema,
// 400 {"detail": ...} for one that is not JSON or is longer than the model reads. Images ("assets", box questions)
// are refused by name. Each response carries its timings in a Server-Timing header.

#include <string>
#include <vector>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "decision/service.h"
#include "util/error.h"

namespace xs::server {

// A body that does not follow the schema: where (the path from "body"), what, and the error's kind.
class SystemOneInvalid : public Error {
public:
    SystemOneInvalid(std::vector<std::string> loc, const std::string& msg, std::string type)
        : Error(msg), loc_(std::move(loc)), type_(std::move(type)) {}
    const std::vector<std::string>& loc() const { return loc_; }
    const std::string& type() const { return type_; }

private:
    std::vector<std::string> loc_;
    std::string type_;
};

// Throws SystemOneInvalid.
DecisionRequest parse_systemone_request(const nlohmann::ordered_json& body);
nlohmann::ordered_json systemone_response(const DecisionRequest& request, const DecisionResult& result,
                                          const std::string& model);

class DecisionHandlers {
public:
    explicit DecisionHandlers(DecisionService& service) : service_(service) {}
    void decide(const httplib::Request& req, httplib::Response& res) const;
    void models(const httplib::Request& req, httplib::Response& res, int64_t created) const;

private:
    DecisionService& service_;
};

}  // namespace xs::server
