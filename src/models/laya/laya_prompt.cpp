#include <algorithm>
#include <optional>

#include "models/common/py_text.h"
#include "models/laya/laya.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::laya {
namespace {

// Python's truth of a JSON value: what `not v` and `v or default` test.
bool truthy(const Json& v) {
    if (v.is_null()) return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v.get<double>() != 0;
    if (v.is_string()) return !v.get_ref<const std::string&>().empty();
    return !v.empty();
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size()))
        s.replace(at, from.size(), to);
    return s;
}

// Python's a // b.
int64_t floor_div(int64_t a, int64_t b) {
    return a / b - ((a % b != 0) && ((a < 0) != (b < 0)) ? 1 : 0);
}

[[noreturn]] void refuse(const DecisionQuestion& q, const std::string& family, const std::string& what,
                         const char* code = "unsupported_question") {
    throw DecisionRefused(str_cat("question '", q.id, "': ", family == "julia" ? "Julia-1" : "Laya", " ", what), code);
}

// The question's instructions and its options' texts in the prompt's order, as Laya's reference writes them
// (rl_agent_api.py _to_internal, rl_common.py render_options).
std::pair<std::string, std::vector<std::string>> laya_texts(const DecisionQuestion& q) {
    // A string as it is, anything else as json.dumps writes it by default (null: "null").
    std::string instructions = q.instructions.is_string() ? q.instructions.get<std::string>()
                                                          : py::json_dumps(q.instructions, {.ensure_ascii = true});
    std::vector<std::string> options;
    if (q.type == QuestionType::Noul) {  // false, then true; a description replaces the reference's wording
        for (const auto& [index, name, fallback] : {std::tuple{1, "false", "no, the statement does not hold"},
                                                    std::tuple{0, "true", "yes, the statement holds"}}) {
            const std::optional<Json>& d = q.options[static_cast<size_t>(index)].description;
            if (d && truthy(*d) && !d->is_string()) {
                refuse(q, "laya", str_cat("reads a noul's '", name, "' description as text; an object or array is not "
                                          "supported (its reference adds it to a string)"));
            }
            options.push_back(str_cat(name, ": ", d && truthy(*d) ? d->get<std::string>() : std::string(fallback)));
        }
        return {std::move(instructions), std::move(options)};
    }
    for (size_t i = 0; i < q.options.size(); ++i) {
        const DecisionOption& o = q.options[i];
        const Json& d = o.description ? *o.description : Json(nullptr);
        if (q.type == QuestionType::Score) {
            options.push_back(str_cat("level ", i, ": ", py::str(d)));
        } else {
            options.push_back(truthy(d) ? str_cat(o.id, ": ", py::str(d)) : o.id);
        }
    }
    return {std::move(instructions), std::move(options)};
}

// The same as Julia-1's typed API passes them (julia/typed.py predict_typed, checked by data.py validate_row): the
// instructions text and each option's description alone.
std::pair<std::string, std::vector<std::string>> julia_texts(const DecisionQuestion& q) {
    if (!q.instructions.is_string()) refuse(q, "julia", "takes its instructions as text");
    std::vector<std::string> options;
    if (q.type == QuestionType::Noul) {
        const std::optional<Json>& on = q.options[0].description;
        const std::optional<Json>& off = q.options[1].description;
        if (!on && !off) return {q.instructions.get<std::string>(), {"false", "true"}};
        if (!on || !off) refuse(q, "julia", "takes a noul's criteria for both 'false' and 'true', or neither");
        for (const Json* d : {&*off, &*on}) {
            if (!d->is_string() || d->get_ref<const std::string&>().empty()) {
                refuse(q, "julia", "reads a noul's criteria as non-empty text");
            }
            options.push_back(d->get<std::string>());
        }
        return {q.instructions.get<std::string>(), std::move(options)};
    }
    if (q.options.size() < 2 || q.options.size() > 20) {
        refuse(q, "julia", str_cat("takes 2 to 20 options; the question has ", q.options.size()));
    }
    for (const DecisionOption& o : q.options) {
        if (!o.description || !o.description->is_string() || o.description->get_ref<const std::string&>().empty()) {
            refuse(q, "julia",
                   str_cat("reads each option as its description, a non-empty text; '", o.id, "' has none"));
        }
        options.push_back(o.description->get<std::string>());
    }
    return {q.instructions.get<std::string>(), std::move(options)};
}

}  // namespace

Encoded encode_question(const DecisionRequest& request, const DecisionQuestion& q, const PromptTools& tools,
                        const Config& config, int64_t max_tokens) {
    const Policy& p = config.policy;
    const auto [instructions, texts] = p.descriptions_only ? julia_texts(q) : laya_texts(q);
    // json.dumps(state, ensure_ascii=False), the separators Python's.
    const std::string state =
        request.state.is_string() ? request.state.get<std::string>() : py::json_dumps(request.state);
    if (p.strict) {
        bool spelled = state.find(config.mask) != std::string::npos ||
                       instructions.find(config.mask) != std::string::npos;
        for (const std::string& t : texts)
            spelled = spelled || t.find(config.mask) != std::string::npos;
        if (spelled) refuse(q, p.family, str_cat("refuses a request that spells its marker '", config.mask, "'"));
    }
    const auto token = [&](const std::string& text) {
        const std::optional<int32_t> id = tools.token_id(text);
        XS_CHECK(id.has_value(), p.family, ": the tokenizer has no token '", text, "'");
        return *id;
    };
    const int32_t cls = token(config.cls), sep = token(config.sep), mask = token(config.mask);
    const auto clean = [&](const std::string& text) { return replace_all(text, config.mask, " "); };

    const std::vector<int32_t> head =
        tools.encode(str_cat(question_type_name(q.type), " question: ", clean(instructions)));
    const auto n = static_cast<int64_t>(texts.size());
    std::vector<std::vector<int32_t>> options;  // the mask token, then the text's first tokens
    std::vector<size_t> lengths;                // each text's tokens, uncut
    int64_t total = 0;
    for (const std::string& text : texts) {
        const std::vector<int32_t> ids = tools.encode(" " + clean(text));
        lengths.push_back(ids.size());
        if (p.strict && static_cast<int64_t>(ids.size()) > kOptionTokens) {
            refuse(q, p.family, str_cat("reads at most ", kOptionTokens, " tokens of an option; one has ", ids.size()),
                   "max_tokens_exceeded");
        }
        std::vector<int32_t> o{mask};
        o.insert(o.end(), ids.begin(),
                 ids.begin() + std::min<int64_t>(kOptionTokens, static_cast<int64_t>(ids.size())));
        total += static_cast<int64_t>(o.size());
        options.push_back(std::move(o));
    }
    int64_t budget = p.head_tokens - total;
    if (budget < 16) {  // too many or too long options: every one cut to the same length
        const int64_t per = std::max<int64_t>(4, floor_div(p.head_tokens - 16, std::max<int64_t>(1, n)));
        total = 0;
        for (std::vector<int32_t>& o : options) {
            if (static_cast<int64_t>(o.size()) > per) o.resize(static_cast<size_t>(per));
            total += static_cast<int64_t>(o.size());
        }
        budget = p.head_tokens - total;
    }
    if (p.strict) {
        bool cut = static_cast<int64_t>(head.size()) > budget;
        for (size_t i = 0; i < options.size(); ++i)
            cut = cut || options[i].size() != lengths[i] + 1;
        if (cut) {
            refuse(q, p.family,
                   str_cat("reads a question and its options in ", p.head_tokens, " tokens; this one needs ",
                           static_cast<int64_t>(head.size()) + total, " with its options cut"),
                   "max_tokens_exceeded");
        }
    }

    Encoded e;
    e.ids.push_back(cls);
    e.ids.insert(e.ids.end(), head.begin(),
                 head.begin() + std::min<int64_t>(std::max<int64_t>(8, budget), static_cast<int64_t>(head.size())));
    e.ids.push_back(sep);
    for (const std::vector<int32_t>& o : options) {
        e.markers.push_back(static_cast<int64_t>(e.ids.size()));
        e.ids.insert(e.ids.end(), o.begin(), o.end());
    }
    e.ids.push_back(sep);
    const std::vector<int32_t> state_ids = tools.encode(clean(state));
    e.state_tokens = static_cast<int64_t>(state_ids.size());
    int64_t room = max_tokens - static_cast<int64_t>(e.ids.size()) - 1;
    if (p.room_for_state) {
        if (room < 1) {
            refuse(q, p.family,
                   str_cat("needs ", e.ids.size() + 1, " tokens before the state; the limit is ", max_tokens),
                   "max_tokens_exceeded");
        }
        if (p.strict && e.state_tokens > room) {
            refuse(q, p.family, str_cat("needs ", static_cast<int64_t>(e.ids.size()) + 1 + e.state_tokens,
                                        " tokens with the state; the limit is ", max_tokens),
                   "max_tokens_exceeded");
        }
    }
    room = std::max<int64_t>(0, room);
    e.state_kept = std::min(e.state_tokens, room);
    e.ids.insert(e.ids.end(), state_ids.begin(), state_ids.begin() + e.state_kept);
    e.ids.push_back(sep);
    // Laya cuts the whole sequence to the limit; a question whose options the cut reaches is refused, as its reference
    // refuses one.
    if (static_cast<int64_t>(e.ids.size()) > max_tokens) e.ids.resize(static_cast<size_t>(max_tokens));
    if (std::any_of(e.markers.begin(), e.markers.end(), [&](int64_t m) { return m >= max_tokens; })) {
        refuse(q, p.family, str_cat("has ", n, " options, which do not fit in ", max_tokens, " tokens"),
               "max_tokens_exceeded");
    }
    if (e.state_kept < e.state_tokens) {
        log::info(p.family, ": question '", q.id, "': the state's ", e.state_tokens, " tokens are cut to ",
                  e.state_kept, " to fit ", max_tokens, ", as the model's reference cuts them");
    }
    return e;
}

}  // namespace xs::laya
