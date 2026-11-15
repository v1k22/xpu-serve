#include "models/config_util.h"

#include <fstream>

#include "tensor/tensor.h"
#include "util/error.h"

namespace xs::cfg {
namespace {
const nlohmann::json& field(const nlohmann::json& obj, const char* key, const std::string& where) {
    if (!obj.is_object() || !obj.contains(key) || obj.at(key).is_null()) fail(where, ": missing field '", key, "'");
    return obj.at(key);
}
}  // namespace

nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) fail("cannot read '", path.string(), "'");
    try {
        return nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        fail("'", path.string(), "' is not valid JSON: ", e.what());
    }
}

int64_t get_int(const nlohmann::json& obj, const char* key, const std::string& where) {
    const auto& v = field(obj, key, where);
    if (!v.is_number_integer()) fail(where, ": field '", key, "' must be an integer");
    return v.get<int64_t>();
}

double get_float(const nlohmann::json& obj, const char* key, const std::string& where) {
    const auto& v = field(obj, key, where);
    if (!v.is_number()) fail(where, ": field '", key, "' must be a number");
    return v.get<double>();
}

std::string get_string(const nlohmann::json& obj, const char* key, const std::string& where) {
    const auto& v = field(obj, key, where);
    if (!v.is_string()) fail(where, ": field '", key, "' must be a string");
    return v.get<std::string>();
}

bool get_bool_or(const nlohmann::json& obj, const char* key, bool fallback, const std::string& where) {
    if (!obj.contains(key) || obj.at(key).is_null()) return fallback;
    if (!obj.at(key).is_boolean()) fail(where, ": field '", key, "' must be a boolean");
    return obj.at(key).get<bool>();
}

int64_t get_int_or(const nlohmann::json& obj, const char* key, int64_t fallback, const std::string& where) {
    if (!obj.contains(key) || obj.at(key).is_null()) return fallback;
    return get_int(obj, key, where);
}

std::vector<int32_t> get_int_list(const nlohmann::json& obj, const char* key, const std::string& where) {
    if (!obj.contains(key) || obj.at(key).is_null()) return {};
    const auto& v = obj.at(key);
    if (v.is_number_integer()) return {v.get<int32_t>()};
    if (!v.is_array()) fail(where, ": field '", key, "' must be an integer or a list of integers");
    std::vector<int32_t> out;
    for (const auto& e : v) {
        if (!e.is_number_integer()) fail(where, ": field '", key, "' must contain only integers");
        out.push_back(e.get<int32_t>());
    }
    return out;
}

namespace {

// transformers' fine-grained FP8 (the -FP8 releases of Qwen): E4M3 codes, a scale per block of kFp8Block rows and
// columns.
void check_fp8_blocks(const nlohmann::json& q, const std::string& where) {
    if (q.contains("fmt") && get_string(q, "fmt", where) != "e4m3") {
        fail(where, ": fmt '", get_string(q, "fmt", where), "' is not supported (supported: e4m3)");
    }
    const std::vector<int32_t> block = get_int_list(q, "weight_block_size", where);
    if (block != std::vector<int32_t>{static_cast<int32_t>(kFp8Block), static_cast<int32_t>(kFp8Block)}) {
        fail(where, ": weight_block_size must be [", kFp8Block, ", ", kFp8Block, "]");
    }
    // "dynamic" has the reference runtime round activations to FP8 as it goes; here they stay unquantized. A static
    // scheme would bring calibrated activation scales, which nothing reads.
    if (q.contains("activation_scheme") && get_string(q, "activation_scheme", where) != "dynamic") {
        fail(where, ": activation_scheme '", get_string(q, "activation_scheme", where),
             "' is not supported (supported: dynamic)");
    }
}

bool absent(const nlohmann::json& obj, const char* key) {
    return !obj.contains(key) || obj.at(key).is_null();
}

// compressed-tensors' float-quantized format (llm-compressor's FP8_DYNAMIC): the weights of Linear layers as E4M3
// codes with a scale per output channel; activations rounded as the reference runtime goes, which stay unquantized
// here.
void check_compressed_tensors(const nlohmann::json& q, const std::string& where) {
    if (const std::string format = get_string(q, "format", where); format != "float-quantized") {
        fail(where, ": format '", format, "' is not supported (supported: float-quantized)");
    }
    if (q.contains("quantization_status") && get_string(q, "quantization_status", where) != "compressed") {
        fail(where, ": quantization_status '", get_string(q, "quantization_status", where),
             "' is not supported (supported: compressed)");
    }
    for (const char* key : {"kv_cache_scheme", "sparsity_config"}) {
        if (!absent(q, key)) fail(where, ": ", key, " is not supported");
    }
    if (q.contains("transform_config") && !q.at("transform_config").empty()) {
        fail(where, ": transform_config is not supported");
    }
    if (!q.contains("config_groups") || !q.at("config_groups").is_object() || q.at("config_groups").empty()) {
        fail(where, ": config_groups must name the quantized layers' scheme");
    }
    for (const auto& [name, group] : q.at("config_groups").items()) {
        const std::string in = where + " config_groups." + name;
        if (absent(group, "weights")) fail(in, ": no weights scheme");
        const nlohmann::json& w = group.at("weights");
        const std::string type = get_string(w, "type", in + ".weights");
        const int64_t bits = get_int(w, "num_bits", in + ".weights");
        const std::string strategy = get_string(w, "strategy", in + ".weights");
        if (type != "float" || bits != 8 || strategy != "channel" || !get_bool_or(w, "symmetric", true, in) ||
            get_bool_or(w, "dynamic", false, in)) {
            fail(in, ".weights: ", bits, "-bit ", type, " weights with strategy '", strategy,
                 "' are not supported (supported: symmetric static 8-bit float weights, a scale per channel)");
        }
        if (!absent(group, "input_activations")) {
            const nlohmann::json& a = group.at("input_activations");
            if (!a.contains("dynamic") || !a.at("dynamic").is_boolean() || !a.at("dynamic").get<bool>()) {
                fail(in, ".input_activations: only dynamic activation quantization is supported (calibrated ",
                     "activation scales are not read)");
            }
        }
        if (!absent(group, "output_activations")) fail(in, ".output_activations is not supported");
    }
}

}  // namespace

void check_quantization(const nlohmann::json& config) {
    if (!config.contains("quantization_config") || config.at("quantization_config").is_null()) return;
    const nlohmann::json& q = config.at("quantization_config");
    const std::string where = "config.json quantization_config";
    const std::string method = get_string(q, "quant_method", where);
    if (method == "fp8") return check_fp8_blocks(q, where);
    if (method == "compressed-tensors") return check_compressed_tensors(q, where);
    fail(where, ": quant_method '", method, "' is not supported (supported: fp8 with one scale per block of ",
         kFp8Block, " rows and columns; compressed-tensors' float-quantized format with a scale per channel; or the ",
         "checkpoint this one was quantized from)");
}

std::string model_id(const std::filesystem::path& model_dir) {
    std::filesystem::path dir = std::filesystem::absolute(model_dir).lexically_normal();
    if (dir.filename().empty()) dir = dir.parent_path();  // the path ended in a separator
    // A snapshot in the Hugging Face cache, <cache>/models--<org>--<name>/snapshots/<commit>: org/name.
    const std::string repo = dir.parent_path().parent_path().filename().string();
    if (dir.parent_path().filename() == "snapshots" && repo.rfind("models--", 0) == 0) {
        std::string id = repo.substr(8);
        if (const size_t sep = id.find("--"); sep != std::string::npos) id.replace(sep, 2, "/");
        return id;
    }
    return dir.filename().string();
}

SamplingParams read_generation_defaults(const std::filesystem::path& model_dir, std::vector<int32_t>* eos_ids) {
    SamplingParams p;
    const auto path = model_dir / "generation_config.json";
    if (!std::filesystem::exists(path)) return p;
    const auto j = read_json(path);
    const std::string where = "generation_config.json";
    const bool do_sample = get_bool_or(j, "do_sample", false, where);
    if (!do_sample) {
        p.temperature = 0.0f;
    } else if (j.contains("temperature") && j["temperature"].is_number()) {
        p.temperature = j["temperature"].get<float>();
    }
    if (j.contains("top_k") && j["top_k"].is_number_integer()) p.top_k = j["top_k"].get<int>();
    if (j.contains("top_p") && j["top_p"].is_number()) p.top_p = j["top_p"].get<float>();
    if (j.contains("min_p") && j["min_p"].is_number()) p.min_p = j["min_p"].get<float>();
    if (j.contains("repetition_penalty") && j["repetition_penalty"].is_number()) {
        p.repetition_penalty = j["repetition_penalty"].get<float>();
    }
    if (eos_ids != nullptr) *eos_ids = get_int_list(j, "eos_token_id", where);
    return p;
}

std::optional<int32_t> tokenizer_eos_id(const std::filesystem::path& model_dir) {
    const auto path = model_dir / "tokenizer_config.json";
    if (!std::filesystem::exists(path)) return std::nullopt;
    const auto j = read_json(path);
    if (!j.contains("eos_token")) return std::nullopt;
    const nlohmann::json& eos = j.at("eos_token");
    const std::string text = eos.is_string() ? eos.get<std::string>()
                             : eos.is_object() && eos.contains("content") ? eos.at("content").get<std::string>()
                                                                          : std::string();
    if (text.empty() || !j.contains("added_tokens_decoder") || !j.at("added_tokens_decoder").is_object()) {
        return std::nullopt;
    }
    for (const auto& [id, token] : j.at("added_tokens_decoder").items()) {
        if (token.is_object() && token.value("content", "") == text) return static_cast<int32_t>(std::stol(id));
    }
    return std::nullopt;
}

}  // namespace xs::cfg
