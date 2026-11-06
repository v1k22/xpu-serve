#pragma once

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "backend/backend.h"
#include "engine/engine.h"
#include "engine/service.h"

namespace xs {

struct ServiceOptions {
    std::string backend = "openvino";
    SessionOptions session;
    EngineOptions engine;
    std::string model_id;  // empty = the model directory name
};

// The real InferenceService: loads model, tokenizer and chat template from a model directory, compiles the step
// graph on a backend, and runs every generation on one dedicated engine thread (the only thread that touches the
// device).
class EngineService final : public InferenceService {
public:
    static std::unique_ptr<EngineService> create(const std::filesystem::path& model_dir, const ServiceOptions& options);
    ~EngineService() override;
    EngineService(const EngineService&) = delete;
    EngineService& operator=(const EngineService&) = delete;

    const ModelCard& model() const override { return card_; }
    const tok::Tokenizer& tokenizer() const override { return *tokenizer_; }
    const chat::ChatTemplate& chat_template() const override { return *template_; }
    GenerateResult generate(const GenerateRequest& request, const TokenCallback& on_token) override;

private:
    struct Job;

    EngineService() = default;
    void worker_loop();

    ModelCard card_;
    std::unique_ptr<tok::Tokenizer> tokenizer_;
    std::unique_ptr<chat::ChatTemplate> template_;
    std::unique_ptr<Engine> engine_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Job*> queue_;
    bool stopping_ = false;
    std::thread worker_;
};

}  // namespace xs
