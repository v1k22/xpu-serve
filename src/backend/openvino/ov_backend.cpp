#include "backend/openvino/ov_backend.h"

#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <optional>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/properties.hpp>

#include "backend/openvino/ov_session.h"
#include "backend/openvino/ov_translate.h"
#include "util/error.h"
#include "util/log.h"
#include "util/timer.h"

namespace xs {

namespace {

using ov_backend::AttentionImpl;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

// SessionOptions::extra keys this backend understands.
struct ExtraOptions {
    std::optional<float> activations_scale_factor;  // ov::hint::activations_scale_factor (f16 overflow guard)
    AttentionImpl attention = AttentionImpl::Auto;  // "auto", "sdpa" or "matmul"
};

ExtraOptions parse_extra(const std::map<std::string, std::string>& extra) {
    ExtraOptions out;
    for (const auto& [key, text] : extra) {
        if (key == "activations_scale_factor") {
            try {
                size_t used = 0;
                out.activations_scale_factor = std::stof(text, &used);
                XS_CHECK(used == text.size(), "trailing characters");
            } catch (const std::exception&) {
                fail("openvino option activations_scale_factor='", text, "' is not a number");
            }
        } else if (key == "attention") {
            out.attention = ov_backend::parse_attention_impl(text);
        } else {
            fail("openvino option '", key, "' is unknown (supported: activations_scale_factor, attention)");
        }
    }
    return out;
}

// "VmHWM" (peak) or "VmRSS" (current) of this process in bytes, 0 when unavailable.
uint64_t process_memory(const std::string& field) {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == field + ":") {
            uint64_t kib = 0;
            status >> kib;
            return kib * 1024;
        }
        status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;
}

uint64_t system_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page_size > 0 ? static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size) : 0;
}

ov::AnyMap compile_config(const SessionOptions& options, const ExtraOptions& extra) {
    const ov::element::Type precision = ov_backend::to_ov_type(options.precision);
    ov::AnyMap config{ov::hint::inference_precision(precision), ov::hint::kv_cache_precision(precision)};
    if (!options.cache_dir.empty()) config.emplace(ov::cache_dir(options.cache_dir));
    if (extra.activations_scale_factor) {
        config.emplace(ov::hint::activations_scale_factor(*extra.activations_scale_factor));
    }
    return config;
}

// Op type histogram of what the plugin will execute, to see which fusions (SDPA, RMS, KV cache) happened.
std::string runtime_op_summary(const ov::CompiledModel& compiled) {
    std::map<std::string, int> counts;
    for (const auto& op : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& info = op->get_rt_info();
        const auto it = info.find("layerType");  // ov::exec_model_info::LAYER_TYPE (developer API)
        ++counts[it == info.end() ? op->get_type_name() : it->second.as<std::string>()];
    }
    std::string out;
    for (const auto& [type, n] : counts)
        out += str_cat(out.empty() ? "" : ", ", type, " x", n);
    return out;
}

class OpenVinoBackend final : public Backend {
public:
    std::string_view name() const override { return "openvino"; }

    std::vector<DeviceInfo> devices() const override {
        std::vector<DeviceInfo> out;
        for (const std::string& id : core_.get_available_devices()) {
            DeviceInfo d;
            d.id = id;
            d.name = core_.get_property(id, ov::device::full_name);
            if (id.rfind("GPU", 0) == 0) {
                d.memory_bytes = core_.get_property(id, ov::intel_gpu::device_total_mem_size);
                d.integrated = core_.get_property(id, ov::device::type) == ov::device::Type::INTEGRATED;
            } else {
                d.memory_bytes = system_memory();
            }
            out.push_back(std::move(d));
        }
        return out;
    }

    std::unique_ptr<Session> compile(const graph::Graph& graph, const WeightStore& weights,
                                     const SessionOptions& options) override {
        XS_CHECK(options.precision == DType::F16 || options.precision == DType::F32,
                 "openvino session precision must be f16 or f32, got ", dtype_name(options.precision));
        XS_CHECK(options.max_context > 0, "max_context must be positive, got ", options.max_context);
        const ExtraOptions extra = parse_extra(options.extra);
        const std::string device = options.device.empty() ? default_device() : options.device;
        if (extra.activations_scale_factor) require_property(device, ov::hint::activations_scale_factor.name());

        Timer timer;
        ov_backend::Translation t = ov_backend::translate(graph, weights, {options.precision, extra.attention});
        const double translate_ms = timer.elapsed_ms();
        ov::CompiledModel compiled;
        try {
            compiled = core_.compile_model(t.model, device, compile_config(options, extra));
        } catch (const ov::Exception& e) {
            fail("openvino compile_model on ", device, " failed: ", e.what());
        }
        t.model.reset();  // drops our converted weight buffers unless the plugin still shares them
        log::info("openvino: compiled ", graph.nodes().size(), " graph nodes for ", device, " (",
                  dtype_name(options.precision), ") in ", translate_ms, " ms + ", timer.elapsed_ms() - translate_ms,
                  " ms; weights ", t.weight_bytes.shared / kGiB, " GiB shared + ", t.weight_bytes.converted / kGiB,
                  " GiB converted; host RSS ", process_memory("VmRSS") / kGiB, " GiB, peak ",
                  process_memory("VmHWM") / kGiB, " GiB");
        log::debug("openvino: runtime ops: ", runtime_op_summary(compiled));
        return std::make_unique<ov_backend::OvSession>(std::move(compiled), std::move(t.inputs), std::move(t.outputs),
                                                       options.max_context);
    }

private:
    void require_property(const std::string& device, const std::string& property) const {
        std::vector<ov::PropertyName> supported;
        try {
            supported = core_.get_property(device, ov::supported_properties);
        } catch (const ov::Exception& e) {
            fail("openvino device '", device, "' is not usable: ", e.what());
        }
        const bool found = std::find(supported.begin(), supported.end(), property) != supported.end();
        XS_CHECK(found, "openvino device ", device, " does not support ", property);
    }

    std::string default_device() const {
        for (const std::string& id : core_.get_available_devices()) {
            if (id.rfind("GPU", 0) == 0) return "GPU";
        }
        return "CPU";
    }

    ov::Core core_;
};

}  // namespace

std::unique_ptr<Backend> make_openvino_backend() {
    return std::make_unique<OpenVinoBackend>();
}

}  // namespace xs
