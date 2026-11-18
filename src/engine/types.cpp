#include "engine/types.h"

namespace xs {

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
