#include "server/embedding_handlers.h"

#include <bit>
#include <cstdio>

#include "server/openai_format.h"
#include "util/base64.h"
#include "util/log.h"

namespace xs::server {
namespace {

using Json = RequestJson;

[[noreturn]] void reject(const std::string& param, const std::string& message) { throw InvalidRequest(message, param); }

// The field's value, or nullptr when it is absent or null.
const Json* find_field(const Json& object, const char* name) {
    const auto it = object.find(name);
    return it == object.end() || it->is_null() ? nullptr : &*it;
}

// Every element an integer token id of the vocabulary.
std::vector<int32_t> token_ids(const Json& list, const std::string& param, const EmbeddingSpec& spec) {
    std::vector<int32_t> ids;
    ids.reserve(list.size());
    for (const Json& v : list) {
        if (!v.is_number_integer()) reject(param, str_cat("'", param, "' must be an array of token ids (integers)"));
        const bool inside = v.is_number_unsigned() ? v.get<uint64_t>() < static_cast<uint64_t>(spec.vocab_size)
                                                   : v.get<int64_t>() >= 0 && v.get<int64_t>() < spec.vocab_size;
        if (!inside) {
            reject(param, str_cat("'", param, "' has token id ", v.dump(), ", outside the vocabulary (0 to ",
                                  spec.vocab_size - 1, ")"));
        }
        ids.push_back(static_cast<int32_t>(v.get<int64_t>()));
    }
    return ids;
}

std::string base64_floats(const std::vector<float>& values) {
    static_assert(std::endian::native == std::endian::little, "base64 embeddings are little-endian float32");
    return base64_encode({reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float)});
}

}  // namespace

EmbeddingsRequest parse_embeddings_request(const RequestJson& body, const EmbeddingSpec& spec,
                                           const std::function<std::vector<int32_t>(std::string_view)>& tokenize) {
    EmbeddingsRequest r;
    if (const Json* format = find_field(body, "encoding_format")) {
        if (!format->is_string() || (*format != "float" && *format != "base64")) {
            reject("encoding_format", "'encoding_format' must be 'float' or 'base64'");
        }
        r.base64 = *format == "base64";
    }
    if (const Json* d = find_field(body, "dimensions")) {
        if (!d->is_number_integer() || d->get<int64_t>() != spec.dimension) {
            reject("dimensions", str_cat("'dimensions' must be ", spec.dimension,
                                         ", the model's: it makes no shorter embeddings"));
        }
    }
    const Json* input = find_field(body, "input");
    if (input == nullptr) reject("input", "'input' is required");
    const auto add = [&](std::vector<int32_t> ids, const std::string& param) {
        if (const auto problem = sequence_problem(spec, ids)) reject(param, str_cat("'", param, "' has ", *problem));
        r.sequences.push_back(std::move(ids));
    };
    const auto text = [&](const Json& value, const std::string& param) {
        const std::string& s = value.get_ref<const std::string&>();
        if (s.empty()) reject(param, str_cat("'", param, "' is an empty string"));
        std::vector<int32_t> ids;
        try {
            ids = tokenize(s);
        } catch (const std::exception& e) {
            reject(param, str_cat("could not tokenize '", param, "': ", e.what()));
        }
        add(std::move(ids), param);
    };
    if (input->is_string()) {
        text(*input, "input");
        return r;
    }
    if (!input->is_array()) {
        reject("input", "'input' must be a string, an array of strings, an array of token ids or an array of arrays "
                        "of token ids");
    }
    if (input->empty()) reject("input", "'input' must not be empty");
    const Json& first = input->front();
    if (first.is_number()) {  // one sequence of token ids
        add(token_ids(*input, "input", spec), "input");
        return r;
    }
    for (size_t i = 0; i < input->size(); ++i) {
        const std::string param = str_cat("input[", i, "]");
        const Json& item = (*input)[i];
        if (first.is_string()) {
            if (!item.is_string()) reject(param, str_cat("'", param, "' must be a string, as 'input[0]' is"));
            text(item, param);
        } else if (first.is_array()) {
            if (!item.is_array()) reject(param, str_cat("'", param, "' must be an array of token ids, as 'input[0]' is"));
            add(token_ids(item, param, spec), param);
        } else {
            reject(param, str_cat("'", param, "' must be a string, a token id or an array of token ids"));
        }
    }
    return r;
}

nlohmann::ordered_json embeddings_response(const EmbeddingResult& result, bool base64, const std::string& model) {
    nlohmann::ordered_json data = nlohmann::ordered_json::array();
    for (size_t i = 0; i < result.embeddings.size(); ++i) {
        const std::vector<float>& e = result.embeddings[i];
        data.push_back({{"object", "embedding"},
                        {"index", i},
                        {"embedding", base64 ? nlohmann::ordered_json(base64_floats(e)) : nlohmann::ordered_json(e)}});
    }
    return {{"object", "list"},
            {"data", std::move(data)},
            {"model", model},
            {"usage", {{"prompt_tokens", result.tokens}, {"total_tokens", result.tokens}}}};
}

void EmbeddingHandlers::embed(const httplib::Request& req, httplib::Response& res) const {
    const EmbeddingsRequest request = parse_embeddings_request(
        parse_json_body(req.body), service_.spec(), [this](std::string_view text) { return service_.tokenize(text); });
    const EmbeddingCall call = service_.embed(request.sequences);
    const EmbeddingTiming& t = call.result.timing;
    char timing[128];
    std::snprintf(timing, sizeof timing, "queue;dur=%.2f, device;dur=%.2f, total;dur=%.2f", call.queue_ms,
                  t.device_ms, t.total_ms);
    res.set_header("Server-Timing", timing);
    log::info(req.method, " ", req.path, ": ", request.sequences.size(), " inputs, ", call.result.tokens, " tokens, ",
              static_cast<int>(t.total_ms * 10) / 10.0, " ms (device ", static_cast<int>(t.device_ms * 10) / 10.0,
              ")");
    res.set_content(embeddings_response(call.result, request.base64, service_.spec().id).dump(), "application/json");
}

void EmbeddingHandlers::models(const httplib::Request&, httplib::Response& res, int64_t created) const {
    const EmbeddingSpec& spec = service_.spec();
    res.set_content(model_list_json(spec.id, spec.max_tokens, created).dump(), "application/json");
}

}  // namespace xs::server
