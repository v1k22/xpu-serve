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
    std::string backend = "sycl";
    SessionOptions session;
    EngineOptions engine;
    std::string model_id;  // empty = the model directory name
    bool vision = true;    // load the vision encoder when the checkpoint has one
    std::filesystem::path drafter;  // a drafter checkpoint for speculative decoding (a Gemma 4 assistant)
    int64_t draft_shortlist = 64;   // LoadOptions::draft_shortlist
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
    bool accepts_images() const override { return engine_->accepts_images(); }
    MediaInput prepare_image(std::span<const uint8_t> encoded) const override;
    std::vector<int32_t> expand_media(std::vector<int32_t> prompt, const std::vector<MediaInput>& media) const override;

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
