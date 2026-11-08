#include "policy/runner.h"

#include <chrono>
#include <random>

#include "tokenizer/tokenizer.h"
#include "util/error.h"
#include "util/log.h"

namespace xs {
namespace {

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

std::unique_ptr<PolicyRunner> PolicyRunner::create(const std::filesystem::path& dir, const PolicyRunnerOptions& options) {
    std::unique_ptr<PolicyRunner> r(new PolicyRunner());
    r->policy_ = load_policy(dir);
    r->tokenizer_ = tok::Tokenizer::load(dir);
    r->backend_ = make_backend(options.backend);
    r->session_ = options.session;
    const PolicySpec& s = r->policy_->spec();
    log::info("policy ", s.id, " (", s.arch, "): ", s.cameras.size(), " cameras, state ", s.state_dim, ", ", s.horizon,
              " x ", s.action_dim, " actions, ", s.denoise_steps, " flow steps");
    return r;
}

PolicyRunner::~PolicyRunner() = default;

Session& PolicyRunner::program(const std::vector<int>& cameras, int steps, double* compile_ms) {
    const auto key = std::make_pair(cameras, steps);
    auto it = programs_.find(key);
    if (it == programs_.end()) {
        const auto start = std::chrono::steady_clock::now();
        const graph::Graph g = policy_->build(cameras, steps);
        auto session = backend_->compile(g, policy_->weights(), session_);
        const double ms = ms_since(start);
        if (compile_ms != nullptr) *compile_ms = ms;
        log::info("policy: program for ", cameras.size(), " camera(s), ", steps, " steps: ", g.nodes().size(),
                  " nodes, compiled in ", static_cast<int64_t>(ms), " ms");
        it = programs_.emplace(key, std::move(session)).first;
    }
    return *it->second;
}

double PolicyRunner::warm_up(const std::vector<int>& cameras, int steps, int runs) {
    double compile_ms = 0;
    program(cameras, steps, &compile_ms);
    const Observation obs = synthetic_observation(spec(), cameras, "warm up");
    for (int i = 0; i < runs; ++i) {
        ActOptions o;
        o.steps = steps;
        o.seed = static_cast<uint64_t>(i);
        act(obs, o);
    }
    return compile_ms;
}

ActResult PolicyRunner::act(const Observation& obs, const ActOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    const PolicySpec& s = spec();
    const int steps = options.steps > 0 ? options.steps : s.denoise_steps;
    const std::vector<int> cameras = policy_->camera_indices(obs);
    Session& session = program(cameras, steps);

    ActResult r;
    const int32_t bos = policy_->prompt_bos();
    if (bos >= 0) r.prompt_ids.push_back(bos);
    const std::vector<int32_t> ids = tokenizer_->encode(policy_->prompt(obs), false);
    r.prompt_ids.insert(r.prompt_ids.end(), ids.begin(), ids.end());
    HostTensor noise(DType::F32, {s.horizon, s.noise_dim});
    if (options.noise != nullptr) {
        noise = *options.noise;
    } else {
        std::mt19937_64 rng(options.seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float& v : noise.span<float>())
            v = normal(rng);
    }
    TensorMap inputs, outputs;
    policy_->prepare(obs, cameras, r.prompt_ids, noise, steps, inputs);
    r.timing.prepare_ms = ms_since(start);

    const auto device_start = std::chrono::steady_clock::now();
    session.run(inputs, outputs);
    r.timing.device_ms = ms_since(device_start);

    const auto finish_start = std::chrono::steady_clock::now();
    r.actions = policy_->finish(outputs);
    r.timing.finish_ms = ms_since(finish_start);
    r.timing.total_ms = ms_since(start);
    return r;
}

Observation synthetic_observation(const PolicySpec& spec, const std::vector<int>& cameras, const std::string& instruction) {
    Observation obs;
    for (const int c : cameras) {
        const CameraSpec& cam = spec.cameras[static_cast<size_t>(c)];
        image::Image img;
        img.width = cam.width;
        img.height = cam.height;
        img.rgb.resize(static_cast<size_t>(cam.width * cam.height * 3));
        for (int64_t y = 0; y < cam.height; ++y) {
            for (int64_t x = 0; x < cam.width; ++x) {
                for (int ch = 0; ch < 3; ++ch) {
                    img.rgb[static_cast<size_t>((y * cam.width + x) * 3 + ch)] =
                        static_cast<uint8_t>((x * 3 + y * 5 + ch * 40 + c * 17) % 256);
                }
            }
        }
        obs.cameras.emplace_back(cam.name, std::move(img));
    }
    obs.state.assign(static_cast<size_t>(spec.state_dim), 0.0f);
    obs.instruction = instruction;
    return obs;
}

}  // namespace xs
