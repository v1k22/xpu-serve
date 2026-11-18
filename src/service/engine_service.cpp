#include "service/engine_service.h"

#include <algorithm>
#include <array>
#include <future>

#include "audio/audio.h"
#include "chat/chat_template.h"
#include "engine/media.h"
#include "image/image.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"
#include "util/log.h"
#include "util/timer.h"

namespace xs {

namespace {

// A neutral item of modality `m` (a gray square, a second of silence), for an encoder's first run.
MediaInput warm_up_item(const Model& model, Modality m) {
    if (m == Modality::Image) {
        image::Image gray;
        gray.width = gray.height = 768;
        gray.rgb.assign(static_cast<size_t>(gray.width * gray.height * 3), 128);
        return model.vision()->prepare(gray);
    }
    audio::Audio silence;
    silence.sample_rate = 16000;
    silence.samples.assign(16000, 0.0f);
    return model.audio()->prepare(silence);
}

}  // namespace

struct EngineService::Job {
    const GenerateRequest& request;
    const TokenCallback& on_token;
    std::promise<GenerateResult> done;
};

std::unique_ptr<EngineService> EngineService::create(const std::filesystem::path& model_dir,
                                                     const ServiceOptions& options) {
    std::unique_ptr<EngineService> s(new EngineService());
    Timer timer;
    s->tokenizer_ = tok::Tokenizer::load(model_dir);
    s->template_ = chat::ChatTemplate::load(model_dir);
    const bool speculate = options.engine.spec_k != 0;
    const bool own_count = options.engine.spec_k == kDraftersCount;
    int spec_k = options.engine.spec_k;
    if (own_count) spec_k = options.spec == "dflash" || options.spec == "dspine" ? kBlockDrafts : kChainDrafts;
    auto model = load_model(model_dir, LoadOptions{.vision = options.vision,
                                                   .speculation = speculate,
                                                   .drafter = options.drafter,
                                                   .draft_shortlist = options.draft_shortlist,
                                                   .drafts = spec_k});
    const ModelInfo& info = model->info();
    log::info("model ", info.id, " (", info.arch, "): graph of ", model->graph().nodes().size(), " nodes, ",
              model->graph().kv_slots().size(), " KV slots", model->vision() ? "; vision encoder" : "",
              model->audio() ? "; audio encoder" : "");

    EngineOptions engine_options = options.engine;
    engine_options.spec_k = spec_k;
    if (engine_options.max_context > info.max_position) {
        log::warn("context ", engine_options.max_context, " exceeds the model's ", info.max_position,
                  "; using ", info.max_position);
        engine_options.max_context = info.max_position;
    }
    SessionOptions session_options = options.session;
    session_options.max_context = engine_options.max_context;

    // Speculating, the session also runs the model's prefill, verify and draft programs (spec::Program order) and
    // cuts a verify back to its accepted tokens.
    std::vector<const graph::Graph*> programs{&model->graph()};
    if (speculate) {
        const Speculation* spec = model->speculation();
        if (spec == nullptr) {
            fail("model ", info.id, " has no drafter: speculative decoding needs one (an MTP head in the checkpoint, or "
                 "--drafter <an assistant checkpoint>)");
        }
        // --spec names a kind of drafter: the checkpoints must hold one of it
        const std::string_view name = spec->drafter_name();
        const bool kind = options.spec.empty() || (options.spec == "mtp" && (name == "mtp" || name == "assistant")) ||
                          (options.spec == "dflash" && (name == "dflash" || name == "dflash2")) || options.spec == name;
        if (!kind) fail("--spec ", options.spec, " was asked for, but the drafter of ", info.id, " is ", name);
        const int block = spec->block() != nullptr ? static_cast<int>(spec->block()->block_drafts()) : 0;
        if (block > 0 && engine_options.spec_k > block) {
            if (!own_count) {
                fail("--spec-k ", engine_options.spec_k, " is more than the ", block, " drafts a block of ", name, " holds");
            }
            engine_options.spec_k = block;
        }
        programs.insert(programs.end(), {&spec->prefill_graph(), &spec->verify_graph(), &spec->draft_graph()});
        session_options.rewind_tokens = engine_options.spec_k + 1;
        log::info("speculative decoding: ", name, ", ", engine_options.spec_k, " drafts per round, drafter graph of ",
                  spec->draft_graph().nodes().size(), " nodes");
    }
    auto backend = make_backend(options.backend);
    Timer compile_timer;
    auto session = backend->compile(programs, model->weights(), session_options);
    log::info("compiled on ", options.backend, " ", session_options.device.empty() ? "(default device)" : session_options.device,
              " in ", static_cast<int>(compile_timer.elapsed_ms()), " ms");
    EncoderSessions encoders;
    for (size_t k = 0; k < kModalities; ++k) {
        const auto m = static_cast<Modality>(k);
        const MediaEncoder* encoder = model->encoder(m);
        if (encoder == nullptr) continue;
        SessionOptions encoder_options = session_options;
        encoder_options.max_context = int64_t{1} << 20;  // stateless: bounds one run's rows only
        Timer encoder_timer;
        encoders[k] = backend->compile(encoder->graph(), model->weights(), encoder_options);
        // The first run of an encoder pays the plugin's kernel setup (about 2 s on a GPU): pay it here, on a neutral
        // item, rather than on the first request with media.
        for (const EncoderRun& run : warm_up_item(*model, m).runs) {
            TensorMap warm;
            encoders[k]->run(run.inputs, warm);
        }
        log::info("compiled and warmed up the ", modality_name(m), " encoder (", encoder->graph().nodes().size(),
                  " nodes) in ", static_cast<int>(encoder_timer.elapsed_ms()), " ms");
    }

    s->card_.id = options.model_id.empty() ? info.id : options.model_id;
    s->card_.arch = info.arch;
    s->card_.context_length = engine_options.max_context;
    s->card_.dialect = info.dialect;
    s->card_.default_sampling = info.default_sampling;
    s->card_.stop_token_ids = info.stop_token_ids;
    s->engine_ = std::make_unique<Engine>(std::move(model), std::move(session), std::move(encoders), engine_options);
    s->worker_ = std::thread([raw = s.get()] { raw->worker_loop(); });
    log::info("ready in ", static_cast<int>(timer.elapsed_ms()), " ms");
    return s;
}

EngineService::~EngineService() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

GenerateResult EngineService::generate(const GenerateRequest& request, const TokenCallback& on_token) {
    Job job{request, on_token, {}};
    auto result = job.done.get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        XS_CHECK(!stopping_, "the engine is shutting down");
        queue_.push_back(&job);
    }
    cv_.notify_one();
    return result.get();
}

MediaInput EngineService::prepare_media(Modality m, std::span<const uint8_t> encoded) const {
    if (!accepts(m)) fail("model '", card_.id, "' does not take ", modality_name(m), " input");
    const Model& model = engine_->model();
    switch (m) {
        case Modality::Image: {
            const image::Image img = image::decode(encoded);
            MediaInput media = model.vision()->prepare(img);
            media.key = image::content_hash(img);
            return media;
        }
        case Modality::Audio: {
            const audio::Audio clip = audio::decode(encoded);
            MediaInput media = model.audio()->prepare(clip);
            media.key = audio::content_hash(clip);
            return media;
        }
    }
    fail("unknown modality");
}

std::vector<int32_t> EngineService::expand_media(std::vector<int32_t> prompt,
                                                 const std::vector<MediaInput>& media) const {
    std::array<const MediaEncoder*, kModalities> encoders{};
    for (size_t k = 0; k < kModalities; ++k) {
        if (accepts(static_cast<Modality>(k))) encoders[k] = engine_->model().encoder(static_cast<Modality>(k));
    }
    return expand_media_placeholders(prompt, media, encoders);
}

void EngineService::worker_loop() {
    for (;;) {
        Job* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) return;  // stopping and drained
            job = queue_.front();
            queue_.pop_front();
        }
        job->done.set_value(engine_->generate(job->request, job->on_token));
    }
}

}  // namespace xs
