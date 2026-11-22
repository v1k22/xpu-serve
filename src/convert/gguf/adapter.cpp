#include "convert/gguf/adapter.h"

#include <map>

#include "convert/gguf/architecture.h"
#include "io/gguf.h"
#include "util/error.h"

namespace xs::gguf_import {
namespace {

constexpr std::string_view kA = ".lora_a";
constexpr std::string_view kB = ".lora_b";

bool ends_with(std::string_view s, std::string_view tail) {
    return s.size() >= tail.size() && s.substr(s.size() - tail.size()) == tail;
}

HostTensor values(const gguf::TensorInfo& t) {
    XS_CHECK(t.dtype == DType::F32 || t.dtype == DType::F16 || t.dtype == DType::BF16, "adapter: ", t.name, " is ",
             t.type_name(), "; an adapter's tensors are floats (F32, F16 or BF16)");
    return HostTensor::from_vector(DType::F32, t.shape(), float_rows(t, 0, t.rows()));
}

}  // namespace

Adapter read_lora(const std::filesystem::path& path, float scale) {
    const gguf::File f(path);
    const gguf::Value* type = f.find("general.type");
    XS_CHECK(type != nullptr && type->as_string() == "adapter", "adapter: ", path.string(),
             " is not an adapter (general.type is ", type != nullptr ? std::string(type->as_string()) : "absent", ")");
    const gguf::Value* kind = f.find("adapter.type");
    XS_CHECK(kind != nullptr && kind->as_string() == "lora", "adapter: ", path.string(), " is a ",
             kind != nullptr ? std::string(kind->as_string()) : "untyped", " adapter; only LoRA adapters are read");
    const gguf::Value* alpha_value = f.find("adapter.lora_alpha");
    const double alpha = alpha_value != nullptr ? alpha_value->as_float() : 0.0;
    Adapter a;
    a.architecture = std::string(f.get_string("general.architecture"));
    const Architecture& arch = find_architecture(a.architecture);
    a.tensors = std::make_unique<HostWeights>();

    std::map<std::string, std::pair<const gguf::TensorInfo*, const gguf::TensorInfo*>> pairs;  // by the file's name
    for (const gguf::TensorInfo& t : f.tensors()) {
        const bool is_a = ends_with(t.name, kA);
        XS_CHECK(is_a || ends_with(t.name, kB), "adapter: ", t.name, " is not a LoRA tensor (<weight>", kA, " or ", kB,
                 ")");
        auto& p = pairs[t.name.substr(0, t.name.size() - kA.size())];
        (is_a ? p.first : p.second) = &t;
    }
    for (const auto& [name, p] : pairs) {
        XS_CHECK(p.first != nullptr && p.second != nullptr, "adapter: ", name, " has ", p.first ? kA : kB,
                 " without ", p.first ? kB : kA);
        const std::optional<std::string> target = arch.adapter_target(name);
        XS_CHECK(target.has_value(), "adapter: ", a.architecture, " has no weight stored whole as ", name,
                 " for a low-rank change to reach");
        HostTensor ta = values(*p.first);
        HostTensor tb = values(*p.second);
        const int64_t rank = tb.shape().back();
        XS_CHECK(rank > 0, "adapter: ", name, kB, " has no rank");
        graph::LowRank change;
        change.a = *target + std::string(kA);
        change.b = *target + std::string(kB);
        change.scale = static_cast<float>(scale * (alpha != 0.0 ? alpha / static_cast<double>(rank) : 1.0));
        a.tensors->add(change.a, std::move(ta));
        a.tensors->add(change.b, std::move(tb));
        a.changes[*target] = std::move(change);
    }
    XS_CHECK(!a.changes.empty(), "adapter: ", path.string(), " changes no weight");
    return a;
}

}  // namespace xs::gguf_import
