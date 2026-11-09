#include "service/engine_service.h"

#include <future>

#include "chat/chat_template.h"
#include "engine/media.h"
#include "image/image.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"
#include "util/log.h"
#include "util/timer.h"

namespace xs {

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
    const bool speculate = options.engine.spec_k > 0;
    auto model = load_model(model_dir, LoadOptions{options.vision, speculate, options.drafter});
    const ModelInfo& info = model->info();
    log::info("model ", info.id, " (", info.arch, "): graph of ", model->graph().nodes().size(), " nodes, ",
              model->graph().kv_slots().size(), " KV slots", model->vision() ? "; vision encoder" : "");

    EngineOptions engine_options = options.engine;
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
        programs.insert(programs.end(), {&spec->prefill_graph(), &spec->verify_graph(), &spec->draft_graph()});
        session_options.rewind_tokens = options.engine.spec_k + 1;
        log::info("speculative decoding: ", options.engine.spec_k, " drafts per round, drafter graph of ",
                  spec->draft_graph().nodes().size(), " nodes");
    }
    auto backend = make_backend(options.backend);
    Timer compile_timer;
    auto session = backend->compile(programs, model->weights(), session_options);
    log::info("compiled on ", options.backend, " ", session_options.device.empty() ? "(default device)" : session_options.device,
              " in ", static_cast<int>(compile_timer.elapsed_ms()), " ms");
    std::unique_ptr<Session> vision_session;
    if (const VisionEncoder* vision = model->vision()) {
        SessionOptions vision_options = session_options;
        vision_options.max_context = int64_t{1} << 20;  // stateless: bounds one image's patches only
        Timer vision_timer;
        vision_session = backend->compile(vision->graph(), model->weights(), vision_options);
        // The first run of the encoder pays the plugin's kernel setup (about 2 s on a GPU): pay it here, with a gray
        // square, rather than on the first image request.
        image::Image gray;
        gray.width = gray.height = 768;
        gray.rgb.assign(static_cast<size_t>(gray.width * gray.height * 3), 128);
        TensorMap warm;
        vision_session->run(vision->prepare(gray).encoder_inputs, warm);
        log::info("compiled and warmed up the vision encoder (", vision->graph().nodes().size(), " nodes) in ",
                  static_cast<int>(vision_timer.elapsed_ms()), " ms");
    }

    s->card_.id = options.model_id.empty() ? info.id : options.model_id;
    s->card_.arch = info.arch;
    s->card_.context_length = engine_options.max_context;
    s->card_.dialect = info.dialect;
    s->card_.default_sampling = info.default_sampling;
    s->card_.stop_token_ids = info.stop_token_ids;
    s->engine_ = std::make_unique<Engine>(std::move(model), std::move(session), std::move(vision_session),
                                          engine_options);
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

MediaInput EngineService::prepare_image(std::span<const uint8_t> encoded) const {
    const VisionEncoder* vision = accepts_images() ? engine_->model().vision() : nullptr;
    if (vision == nullptr) fail("model '", card_.id, "' does not take images");
    const image::Image img = image::decode(encoded);
    MediaInput media = vision->prepare(img);
    media.key = image::content_hash(img);
    return media;
}

std::vector<int32_t> EngineService::expand_media(std::vector<int32_t> prompt,
                                                 const std::vector<MediaInput>& media) const {
    const VisionEncoder* vision = accepts_images() ? engine_->model().vision() : nullptr;
    if (vision == nullptr) return InferenceService::expand_media(std::move(prompt), media);
    return expand_media_placeholders(prompt, media, *vision);
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
