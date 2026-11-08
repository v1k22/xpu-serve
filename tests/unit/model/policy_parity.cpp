// End-to-end parity of a robot policy's program against LeRobot reference outputs
// (tests/parity/policy/lerobot_reference.py), for whichever policy XS_TEST_POLICY_DIR holds. Opt-in: needs
// XS_TEST_POLICY_DIR (with the tokenizer.json the policy reads) and XS_POLICY_FIXTURES.
//
//   XS_PARITY_BACKEND   cpu-ref (default) | sycl
//   XS_PARITY_DEVICE    backend device, e.g. GPU.0
//   XS_PARITY_PRECISION f32 (default) | f16

#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

#include "parity_util.h"
#include "policy/runner.h"

namespace {

using xs::test::compare;
using xs::test::Compare;
using xs::test::env_or;
using xs::test::read_f32;

std::filesystem::path policy_dir() { return env_or("XS_TEST_POLICY_DIR", ""); }
std::filesystem::path fixtures() { return env_or("XS_POLICY_FIXTURES", ""); }
bool enabled() {
    return !policy_dir().empty() && std::filesystem::exists(fixtures() / "manifest.json");
}

std::vector<uint8_t> read_u8(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST_CASE("policy matches LeRobot actions for all cameras and for one camera" * doctest::skip(!enabled())) {
    std::ifstream mf(fixtures() / "manifest.json");
    const auto manifest = nlohmann::json::parse(mf);
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    xs::PolicyRunnerOptions o;
    o.backend = env_or("XS_PARITY_BACKEND", "cpu-ref");
    o.session.device = env_or("XS_PARITY_DEVICE", "");
    o.session.precision = f32 ? xs::DType::F32 : xs::DType::F16;
    auto runner = xs::PolicyRunner::create(policy_dir(), o);
    const xs::PolicySpec& spec = runner->spec();
    const std::vector<float> state = read_f32(fixtures() / "state.bin");
    const std::vector<float> noise_values = read_f32(fixtures() / "noise.bin");
    xs::HostTensor noise = xs::HostTensor::from_vector(xs::DType::F32, {spec.horizon, spec.noise_dim}, noise_values);

    for (const auto& c : manifest["cases"]) {
        const std::string name = c["name"];
        CAPTURE(name);
        xs::Observation obs;
        for (const std::string cam : c["cameras"]) {
            xs::image::Image img;
            img.width = spec.cameras[0].width;
            img.height = spec.cameras[0].height;
            img.rgb = read_u8(fixtures() / ("image_" + cam + ".bin"));
            REQUIRE(img.rgb.size() == static_cast<size_t>(img.width * img.height * 3));
            obs.cameras.emplace_back(cam, std::move(img));
        }
        obs.state = state;
        obs.instruction = c["instruction"];
        xs::ActOptions ao;
        ao.noise = &noise;
        ao.steps = c["steps"];
        const xs::ActResult r = runner->act(obs, ao);
        CHECK(r.prompt_ids == c["input_ids"].get<std::vector<int32_t>>());
        const std::vector<float> ours(r.actions.data<float>(), r.actions.data<float>() + r.actions.numel());
        const std::vector<float> ref = read_f32(fixtures() / c["actions"].get<std::string>());
        const Compare cmp = compare(ours, ref);
        MESSAGE(name, ": ", obs.cameras.size(), " camera(s), ", r.prompt_ids.size(), " prompt tokens, actions cosine ",
                cmp.cosine, ", max |diff| ", cmp.max_abs, ", ", r.timing.total_ms, " ms");
        CHECK(cmp.cosine >= (f32 ? 0.99999 : 0.999));
    }
}
