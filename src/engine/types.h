#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
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

// What a request's media item is: which of the model's encoders reads it.
enum class Modality : uint8_t { Image, Audio };
inline constexpr size_t kModalities = 2;
const char* modality_name(Modality m);  // "image", "audio"

// One run of an encoder graph over (a part of) a media item: its inputs, and how many leading rows of its
// "embeddings" output are the item's (the rest pads the graph's static shape).
struct EncoderRun {
    TensorMap inputs;
    int64_t rows = 0;
};

// One media item of a request (an image, an audio clip), preprocessed for the model's encoder of its modality.
struct MediaInput {
    Modality modality = Modality::Image;
    uint64_t key = 0;               // content hash of the decoded item; tells items apart in the prefix cache
    std::vector<EncoderRun> runs;   // in order: the item's rows are each run's leading `rows`, one after another
    int64_t num_tokens = 0;         // the rows of all runs = soft tokens the prompt holds for this item
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
    int media_encoded = 0;   // media items that ran through their encoder (not served from its cache)
    double encode_ms = 0.0;  // time in the media encoders, not part of prefill_ms
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    // Speculative decoding: verify rounds, drafts proposed, drafts the target accepted; of decode_ms, the time in the
    // drafter and in the verify steps.
    int spec_rounds = 0;
    int spec_drafted = 0;
    int spec_accepted = 0;
    double spec_draft_ms = 0.0;
    double spec_verify_ms = 0.0;
    double prefill_tok_per_s() const { return prefill_ms > 0 ? (prompt_tokens - cached_tokens) * 1000.0 / prefill_ms : 0; }
    double decode_tok_per_s() const {
        return decode_ms > 0 && generated_tokens > 1 ? (generated_tokens - 1) * 1000.0 / decode_ms : 0;
    }
    // Tokens a round yields: the accepted drafts and the target's own.
    double tokens_per_round() const { return spec_rounds > 0 ? 1.0 + double(spec_accepted) / spec_rounds : 0; }
};

struct GenerateResult {
    FinishReason finish = FinishReason::Stop;
    std::vector<int32_t> tokens;  // generated tokens, stop token excluded
    GenerateStats stats;
    std::string error;  // set when finish == Error
};

// Adds `more` (another generation's) to `total`.
void accumulate(GenerateStats& total, const GenerateStats& more);

// A speech recognition request (OpenAI's /v1/audio/transcriptions).
struct TranscriptionRequest {
    std::vector<uint8_t> audio;     // the file's bytes (WAV, MP3, FLAC)
    std::string language;           // an ISO 639-1 code or a language name; empty: the model detects it
    std::string prompt;             // context for the model (names, terms): its system turn
    SamplingParams sampling;
    std::optional<int> max_tokens;  // per piece of audio; nullopt: the model's (Transcription::max_reply_tokens)
};

struct TranscriptionResult {
    FinishReason finish = FinishReason::Stop;
    std::string error;     // when finish == Error
    std::string text;
    std::string language;  // as the model names it; pieces in other languages are listed in order, comma-separated
    double seconds = 0;    // of audio
    int pieces = 0;        // long audio is transcribed in pieces
    GenerateStats stats;   // summed over the pieces
};

// Receives transcript text as it decodes; returning false cancels.
using TextCallback = std::function<bool(std::string_view delta)>;


// Called on the engine thread for every generated token (stop tokens are not passed). Return false to stop
// generation (finish = Cancelled); the caller uses this for stop strings and client disconnects.
using TokenCallback = std::function<bool(int32_t token)>;

}  // namespace xs
