#include "service/engine_service.h"

#include <future>

#include "chat/chat_template.h"
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
    auto model = load_model(model_dir);
    const ModelInfo& info = model->info();
    log::info("model ", info.id, " (", info.arch, "): graph of ", model->graph().nodes().size(), " nodes, ",
              model->graph().kv_slots().size(), " KV slots");

    EngineOptions engine_options = options.engine;
    if (engine_options.max_context > info.max_position) {
        log::warn("context ", engine_options.max_context, " exceeds the model's ", info.max_position,
                  "; using ", info.max_position);
        engine_options.max_context = info.max_position;
    }
    SessionOptions session_options = options.session;
    session_options.max_context = engine_options.max_context;

    auto backend = make_backend(options.backend);
    Timer compile_timer;
    auto session = backend->compile(model->graph(), model->weights(), session_options);
    log::info("compiled on ", options.backend, " ", session_options.device.empty() ? "(default device)" : session_options.device,
              " in ", static_cast<int>(compile_timer.elapsed_ms()), " ms");

    s->card_.id = options.model_id.empty() ? info.id : options.model_id;
    s->card_.arch = info.arch;
    s->card_.context_length = engine_options.max_context;
    s->card_.dialect = info.dialect;
    s->card_.default_sampling = info.default_sampling;
    s->card_.stop_token_ids = info.stop_token_ids;
    s->engine_ = std::make_unique<Engine>(std::move(model), std::move(session), engine_options);
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
