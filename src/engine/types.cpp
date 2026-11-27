#include "engine/types.h"

namespace xs {

void accumulate(GenerateStats& total, const GenerateStats& more) {
    total.prompt_tokens += more.prompt_tokens;
    total.cached_tokens += more.cached_tokens;
    total.generated_tokens += more.generated_tokens;
    total.media_encoded += more.media_encoded;
    total.encode_ms += more.encode_ms;
    total.prefill_ms += more.prefill_ms;
    total.decode_ms += more.decode_ms;
    total.spec_rounds += more.spec_rounds;
    total.spec_drafted += more.spec_drafted;
    total.spec_accepted += more.spec_accepted;
    total.spec_draft_ms += more.spec_draft_ms;
    total.spec_verify_ms += more.spec_verify_ms;
    total.constrained_rows += more.constrained_rows;
    total.constraint_ms += more.constraint_ms;
}

const char* modality_name(Modality m) {
    switch (m) {
        case Modality::Image: return "image";
        case Modality::Audio: return "audio";
    }
    return "?";
}

const char* finish_reason_name(FinishReason r) {
    switch (r) {
        case FinishReason::Stop: return "stop";
        case FinishReason::Length: return "length";
        case FinishReason::Cancelled: return "cancelled";
        case FinishReason::Error: return "error";
    }
    return "error";
}

}  // namespace xs
