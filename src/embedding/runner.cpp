#include "embedding/runner.h"

#include <algorithm>

#include "embedding/pooling.h"
#include "util/error.h"
#include "util/log.h"
#include "util/timer.h"

namespace xs {

std::optional<std::string> sequence_problem(const EmbeddingSpec& spec, std::span<const int32_t> ids) {
    if (ids.empty()) return std::string("no token");
    if (static_cast<int64_t>(ids.size()) > spec.max_tokens) {
        return str_cat(ids.size(), " tokens, more than the ", spec.max_tokens,
                       " the model reads (inputs are not truncated)");
    }
    for (const int32_t id : ids) {
        if (id < 0 || id >= spec.vocab_size) {
            return str_cat("token id ", id, ", outside the vocabulary (0 to ", spec.vocab_size - 1, ")");
        }
    }
    return std::nullopt;
}

EmbeddingRunner::~EmbeddingRunner() = default;

std::unique_ptr<EmbeddingRunner> EmbeddingRunner::create(const std::filesystem::path& dir,
                                                         const EmbeddingRunnerOptions& options) {
    std::unique_ptr<EmbeddingRunner> r(new EmbeddingRunner());
    Timer timer;
    r->model_ = load_embedding_model(dir);
    r->spec_ = r->model_->spec();
    if (!options.model_id.empty()) r->spec_.id = options.model_id;
    SessionOptions session = options.session;
    r->spec_.max_tokens = std::min(r->spec_.max_tokens, session.max_context);
    XS_CHECK(r->spec_.max_tokens >= 1, "the context must hold at least one token, got ", session.max_context);
    session.max_context = r->spec_.max_tokens;
    const EmbeddingSpec& s = r->spec_;
    log::info("embedding model ", s.id, " (", s.arch, "): ", s.dimension, " values, ", pooling_name(s.pooling),
              " pooling", s.normalize ? ", normalized" : "", "; inputs of up to ", s.max_tokens, " tokens");
    Timer compile;
    r->session_ = make_backend(options.backend)->compile(r->model_->graph(), r->model_->weights(), session);
    log::info("compiled on ", options.backend, " ", session.device.empty() ? "(default device)" : session.device,
              " in ", static_cast<int>(compile.elapsed_ms()), " ms");
    // The first run pays the backend's kernel setup: pay it here.
    const std::vector<int32_t> warm(static_cast<size_t>(std::min<int64_t>(8, s.max_tokens)), 0);
    r->hidden_states(warm);
    log::info("ready in ", static_cast<int>(timer.elapsed_ms()), " ms");
    return r;
}

HostTensor EmbeddingRunner::hidden_states(std::span<const int32_t> ids) {
    if (const auto problem = sequence_problem(spec_, ids)) fail("an embedding input has ", *problem);
    TensorMap outputs;
    session_->run(model_->prepare_inputs(ids), outputs);
    return std::move(outputs.at("hidden"));
}

EmbeddingResult EmbeddingRunner::embed(const std::vector<std::vector<int32_t>>& sequences) {
    EmbeddingResult result;
    Timer total;
    for (const std::vector<int32_t>& ids : sequences) {
        Timer device;
        const HostTensor hidden = hidden_states(ids);
        result.timing.device_ms += device.elapsed_ms();
        result.embeddings.push_back(pool(hidden, spec_.pooling, spec_.normalize));
        result.tokens += static_cast<int64_t>(ids.size());
    }
    result.timing.total_ms = total.elapsed_ms();
    return result;
}

}  // namespace xs
