#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "tensor/tensor.h"

namespace xs {

struct SamplingParams {
    float temperature = 1.0f;  // 0 = greedy
    int top_k = 0;             // 0 = off
    float top_p = 1.0f;        // 1 = off
    float min_p = 0.0f;        // 0 = off
    float repetition_penalty = 1.0f;  // 1 = off; applied to tokens of prompt + output
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    std::optional<uint64_t> seed;
};

// One image of a request, preprocessed for the model's vision encoder.
struct MediaInput {
    uint64_t key = 0;          // content hash of the decoded pixels; tells images apart in the prefix cache
    TensorMap encoder_inputs;  // inputs of the encoder graph
    int64_t num_tokens = 0;    // embedding rows the encoder yields = soft tokens the prompt holds for this image
};

struct GenerateRequest {
    std::vector<int32_t> prompt;  // with every image's placeholder already expanded to its token run
    std::vector<MediaInput> media;  // in prompt order; item k fills the k-th run of soft tokens
    SamplingParams sampling;
    int max_new_tokens = 512;
    std::vector<int32_t> stop_token_ids;  // generation ends after emitting none of these (they are not passed on)
};

enum class FinishReason { Stop, Length, Cancelled, Error };

const char* finish_reason_name(FinishReason r);  // "stop", "length", "cancelled", "error"

struct GenerateStats {
    int prompt_tokens = 0;
    int cached_tokens = 0;  // prompt tokens reused from the KV state of the previous request
    int generated_tokens = 0;
    int images_encoded = 0;  // images that ran through the vision encoder (not served from its cache)
    double encode_ms = 0.0;  // time in the vision encoder, not part of prefill_ms
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double prefill_tok_per_s() const { return prefill_ms > 0 ? (prompt_tokens - cached_tokens) * 1000.0 / prefill_ms : 0; }
    double decode_tok_per_s() const {
        return decode_ms > 0 && generated_tokens > 1 ? (generated_tokens - 1) * 1000.0 / decode_ms : 0;
    }
};

struct GenerateResult {
    FinishReason finish = FinishReason::Stop;
    std::vector<int32_t> tokens;  // generated tokens, stop token excluded
    GenerateStats stats;
    std::string error;  // set when finish == Error
};

// Called on the engine thread for every generated token (stop tokens are not passed). Return false to stop
// generation (finish = Cancelled); the caller uses this for stop strings and client disconnects.
using TokenCallback = std::function<bool(int32_t token)>;

}  // namespace xs
