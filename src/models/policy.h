#pragma once

// A robot policy (a vision-language-action model): an observation in (camera images, robot state, an instruction), a
// chunk of actions out. The model code builds one static graph per call shape (which cameras are present, how many
// denoising steps, how long the padded prompt is): pixels to actions in a single program, so a call runs without host
// round trips. Tokenizing the
// prompt and running the program are the policy runtime's job (src/policy/); docs/design-vla.md has the design.

#include <compare>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "graph/graph.h"
#include "image/image.h"
#include "io/safetensors.h"
#include "tensor/tensor.h"

namespace xs {

struct CameraSpec {
    std::string name;  // the checkpoint's feature name, e.g. "observation.images.base_0_rgb"
    int64_t height = 0;
    int64_t width = 0;
};

// What a client sends and gets back.
struct PolicySpec {
    std::string id;    // display name, the checkpoint directory name by default
    std::string arch;  // e.g. "pi05"
    std::vector<CameraSpec> cameras;  // every camera the policy knows, in the order the network reads them
    int64_t state_dim = 0;            // robot state values a request carries
    int64_t action_dim = 0;           // action values per step returned
    int64_t horizon = 0;              // steps per chunk
    int64_t noise_dim = 0;            // width of the network's action space (the noise is [horizon, noise_dim])
    int denoise_steps = 0;            // the checkpoint's default number of flow steps
};

// One request. Any non-empty subset of the cameras may be present; the program for that subset runs (a missing
// camera costs nothing).
struct Observation {
    std::vector<std::pair<std::string, image::Image>> cameras;  // (camera name, RGB image)
    std::vector<float> state;
    std::string instruction;
};

// What one program is built for: the cameras present (indices into spec().cameras, ascending), the flow steps, and
// the prompt tokens it reads (the prompt is padded up to that many).
struct CallShape {
    std::vector<int> cameras;
    int steps = 0;
    int64_t prompt_tokens = 0;
    auto operator<=>(const CallShape&) const = default;
};

class Policy {
public:
    virtual ~Policy() = default;
    virtual const PolicySpec& spec() const = 0;
    virtual const WeightStore& weights() const = 0;
    // The program for calls of this shape: inputs as prepare() fills them, output "actions" [horizon, noise_dim].
    virtual graph::Graph build(const CallShape& shape) const = 0;
    // The prompt text the network reads (the instruction, plus whatever the model puts there, such as state bins).
    virtual std::string prompt(const Observation& obs) const = 0;
    // The token placed first in the prompt (the runtime prepends it to the tokenized prompt), or -1 for none.
    virtual int32_t prompt_bos() const = 0;
    // The prompt tokens a program reads for a prompt of `tokens` ids: padded up to a bucket (prompts of nearby
    // lengths share a program) and never past the model's maximum (longer prompts are truncated).
    virtual int64_t prompt_tokens(int64_t tokens) const = 0;
    // Host side of a call: images, prompt ids and noise [horizon, noise_dim] -> the inputs of build(shape).
    // `shape.cameras` are the observation's camera indices as camera_indices() returns them.
    virtual void prepare(const Observation& obs, const CallShape& shape, std::span<const int32_t> prompt_ids,
                         const HostTensor& noise, TensorMap& inputs) const = 0;
    // The program's "actions" -> [horizon, action_dim] in the robot's units.
    virtual HostTensor finish(const TensorMap& outputs) const = 0;

    // The ascending spec() indices of the observation's cameras; throws xs::Error naming an unknown or repeated one.
    std::vector<int> camera_indices(const Observation& obs) const;
};

// What a checkpoint directory holds.
enum class ModelKind : uint8_t { Language, Policy };
ModelKind model_kind(const std::filesystem::path& dir);

// Reads <dir>/config.json (LeRobot layout: "type") and dispatches on the policy type. Throws xs::Error naming
// unsupported types and features.
std::unique_ptr<Policy> load_policy(const std::filesystem::path& dir);
std::vector<std::string> supported_policy_types();

}  // namespace xs
