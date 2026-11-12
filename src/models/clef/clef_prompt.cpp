#include <algorithm>
#include <numeric>

#include "models/clef/clef.h"
#include "models/common/py_text.h"

namespace xs::clef {
namespace {

constexpr const char* kSystemPrompt =
    "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's "
    "allowed options.";
constexpr const char* kTrue = "The proposition is true or the answer is yes.";
constexpr const char* kFalse = "The proposition is false or the answer is no.";

// render(): a string is itself, anything else compact JSON with sorted keys.
std::string render(const Json& value) {
    return value.is_string() ? value.get<std::string>() : py::json_dumps(value, {.sort_keys = true, .compact = true});
}

void append(std::vector<int32_t>& ids, const PromptTools& tools, const std::string& text) {
    const std::vector<int32_t> piece = tools.encode(text);
    ids.insert(ids.end(), piece.begin(), piece.end());
}

// The options in the prompt's order: a choice's sorted by name, a score's by level, a noul's true then false.
std::vector<int> prompt_order(const DecisionQuestion& q) {
    std::vector<int> order(q.options.size());
    std::iota(order.begin(), order.end(), 0);
    if (q.type == QuestionType::Choice) {
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return q.options[static_cast<size_t>(a)].id < q.options[static_cast<size_t>(b)].id;
        });
    }
    return order;
}

// What an option's line says of it: {"description": ..., "option_id": ...}; a noul outcome the request does not
// describe has the model's own wording, and a description given as null is left out.
std::string semantics(const DecisionOption& option) {
    Json s = Json::object();
    s["option_id"] = option.id;
    if (!option.description) {
        s["description"] = option.id == "true" ? kTrue : kFalse;
    } else if (!option.description->is_null()) {
        s["description"] = *option.description;
    }
    return render(s);
}

}  // namespace

EncodedRecord encode_record(const DecisionRequest& request, const PromptTools& tools, int64_t max_tokens) {
    EncodedRecord record;
    std::vector<int32_t> schema;
    append(schema, tools, "\n\nSCHEMA FIELDS:\n");
    for (size_t i = 0; i < request.questions.size(); ++i) {
        const DecisionQuestion& q = request.questions[i];
        EncodedQuestion e;
        append(schema, tools,
               str_cat("\nFIELD ", i + 1, "\nID: ", q.id, "\nTYPE: ", question_type_name(q.type), "\nINSTRUCTION: "));
        e.instruction.begin = static_cast<int64_t>(schema.size());
        const bool none =
            q.instructions.is_null() || (q.instructions.is_string() && q.instructions.get<std::string>().empty());
        append(schema, tools, none ? q.id : render(q.instructions));
        e.instruction.end = static_cast<int64_t>(schema.size());
        append(schema, tools, "\nALLOWED OPTIONS:\n");
        e.option = prompt_order(q);
        for (size_t j = 0; j < e.option.size(); ++j) {
            append(schema, tools, str_cat("OPTION ", j + 1, ": "));
            Span span{static_cast<int64_t>(schema.size()), 0};
            append(schema, tools, semantics(q.options[static_cast<size_t>(e.option[j])]));
            span.end = static_cast<int64_t>(schema.size());
            e.options.push_back(span);
            append(schema, tools, "\n");
        }
        append(schema, tools, "END FIELD\n");
        record.questions.push_back(std::move(e));
    }

    const std::vector<int32_t> prefix =
        tools.encode(str_cat("<|im_start|>system\n", kSystemPrompt, "<|im_end|>\n<|im_start|>user\nSTATE:\n"));
    const std::vector<int32_t> suffix =
        tools.encode("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
    const std::vector<int32_t> state = tools.encode(render(request.state));
    const size_t total = prefix.size() + state.size() + schema.size() + suffix.size();
    if (static_cast<int64_t>(total) > max_tokens) {
        throw DecisionRefused(str_cat("the request is ", total, " tokens (", state.size(), " of state, ", schema.size(),
                                      " of questions); the limit is ", max_tokens),
                              "max_tokens_exceeded");
    }
    const auto offset = static_cast<int64_t>(prefix.size() + state.size());
    for (EncodedQuestion& e : record.questions) {
        e.instruction.begin += offset;
        e.instruction.end += offset;
        for (Span& s : e.options) {
            s.begin += offset;
            s.end += offset;
        }
    }
    record.ids = prefix;
    record.ids.insert(record.ids.end(), state.begin(), state.end());
    record.ids.insert(record.ids.end(), schema.begin(), schema.end());
    record.ids.insert(record.ids.end(), suffix.begin(), suffix.end());
    return record;
}

}  // namespace xs::clef
