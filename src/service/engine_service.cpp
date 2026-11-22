#include "service/engine_service.h"

#include <algorithm>
#include <array>
#include <future>

#include "audio/audio.h"
#include "audio/split.h"
#include "chat/chat_template.h"
#include "convert/gguf/adapter.h"
#include "engine/media.h"
#include "image/image.h"
#include "models/low_rank_model.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"
#include "util/log.h"
#include "util/timer.h"

namespace xs {

namespace {

// A neutral item of modality `m` (a gray square, a second of silence), for an encoder's first run.
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
    const bool block_drafter = options.spec == "dflash" || options.spec == "dspine";
    if (own_count) spec_k = options.spec == "dspark" ? kDsparkDrafts : block_drafter ? kBlockDrafts : kChainDrafts;
    auto model = load_model(model_dir, LoadOptions{.vision = options.vision,
                                                   .image_max_pixels = options.image_max_pixels,
                                                   .speculation = speculate,
                                                   .drafter = options.drafter,
                                                   .draft_shortlist = options.draft_shortlist,
                                                   .drafts = spec_k});
    if (!options.lora.empty()) {
        gguf_import::Adapter adapter = gguf_import::read_lora(options.lora, options.lora_scale);
        log::info("adapter ", options.lora.filename().string(), " (", adapter.architecture, "): ",
                  adapter.changes.size(), " weights changed, scale ", options.lora_scale);
        model = with_low_rank(std::move(model), std::move(adapter.tensors), adapter.changes);
    }
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
    // The media encoders first: what they hold of the device is then part of the bill the decoder's session checks.
    EncoderSessions encoders;
    for (size_t k = 0; k < kModalities; ++k) {
        const auto m = static_cast<Modality>(k);
        const MediaEncoder* encoder = model->encoder(m);
        if (encoder == nullptr) continue;
        SessionOptions encoder_options = session_options;
        encoder_options.max_context = int64_t{1} << 20;  // stateless: bounds one run's rows only
        Timer encoder_timer;
        const std::vector<const graph::Graph*> encoder_programs = encoder->programs();
        encoders[k] = backend->compile(encoder_programs, model->weights(), encoder_options);
        // The first run of a program pays the plugin's kernel setup (about 2 s on a GPU): pay it here, on a neutral
        // item, rather than on the first request with media.
        for (const EncoderRun& run : encoder->warm_up_item().runs) {
            TensorMap warm;
            encoders[k]->run(run.program, run.inputs, warm);
        }
        log::info("compiled and warmed up the ", modality_name(m), " encoder (", encoder_programs.size(),
                  " program(s), ", encoder_programs.front()->nodes().size(), " nodes) in ",
                  static_cast<int>(encoder_timer.elapsed_ms()), " ms");
    }
    // The cache of the whole context and the largest steps reserved at load: --ctx is refused now, not halfway into
    // a long prompt. Speculating, prefill chunks run the prefill program and verifies the verify program
    // (spec::Program); a drafter's own steps are small.
    const int64_t chunk = engine_options.prefill_chunk;
    session_options.max_step_tokens =
        speculate ? std::vector<int64_t>{0, chunk, engine_options.spec_k + 1, 0} : std::vector<int64_t>{chunk};
    Timer compile_timer;
    std::unique_ptr<Session> session;
    try {
        session = backend->compile(programs, model->weights(), session_options);
    } catch (const Error& e) {
        const bool media = std::any_of(encoders.begin(), encoders.end(), [](const auto& e) { return e != nullptr; });
        if (!media) throw;
        fail(e.what(), " (the media encoders hold some of that memory: --no-vision leaves them out)");
    }
    log::info("compiled on ", options.backend, " ", session_options.device.empty() ? "(default device)" : session_options.device,
              " in ", static_cast<int>(compile_timer.elapsed_ms()), " ms");

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

bool EngineService::transcribes() const {
    return accepts(Modality::Audio) && engine_->model().transcription() != nullptr;
}

TranscriptionResult EngineService::transcribe(const TranscriptionRequest& request, const TextCallback& on_text) {
    if (!transcribes()) return InferenceService::transcribe(request, on_text);
    const Model& model = engine_->model();
    const Transcription& format = *model.transcription();
    const std::string prefix = format.language_prefix(request.language);
    const audio::Audio& clip = request.clip;
    XS_CHECK(clip.sample_rate > 0 && !clip.samples.empty(), "transcription: an empty audio clip");
    // The reference package's turn: a system message (empty without a prompt), the user's audio.
    const nlohmann::json messages = nlohmann::json::array(
        {{{"role", "system"}, {"content", request.prompt}},
         {{"role", "user"}, {"content", nlohmann::json::array({{{"type", "audio"}}})}}});
    const std::vector<int32_t> turn =
        tokenizer_->encode(template_->render(messages, nullptr, true, nlohmann::json::object()) + prefix,
                           /*parse_special=*/true);
    // Pieces short enough that each one's prompt and longest reply fit the context (a cut may land past the limit).
    const int reply_tokens = request.max_tokens.value_or(format.max_reply_tokens());
    const double fits = static_cast<double>(card_.context_length - static_cast<int64_t>(turn.size()) - reply_tokens) /
                            format.audio_tokens_per_second() -
                        audio::kQuietSearchSeconds;
    XS_CHECK(fits >= 1.0, "transcription: a context of ", card_.context_length, " tokens holds too little audio after ",
             "the ", turn.size(), "-token turn and a reply of ", reply_tokens, " tokens: serve with a larger --ctx");
    TranscriptionResult result;
    result.seconds = static_cast<double>(clip.samples.size()) / clip.sample_rate;
    std::vector<std::string> languages;
    bool cancelled = false;
    for (const auto& [first, last] : audio::split_at_quiet(clip, std::min(format.max_piece_seconds(), fits))) {
        audio::Audio piece{clip.sample_rate, {clip.samples.begin() + first, clip.samples.begin() + last}};
        MediaInput media = model.audio()->prepare(piece);
        media.key = audio::content_hash(piece);
        GenerateRequest g;
        g.media.push_back(std::move(media));
        g.prompt = expand_media(turn, g.media);
        g.sampling = request.sampling;
        g.max_new_tokens = reply_tokens;
        g.stop_token_ids = card_.stop_token_ids;

        tok::StreamDecoder decoder(*tokenizer_, /*skip_special=*/true);
        std::string reply;
        size_t sent = 0;  // of this piece's transcript, the part passed to on_text
        const auto pass_on = [&](bool finished) {
            const std::string piece_text = format.parse(prefix, reply, finished).text;
            if (piece_text.size() <= sent) return true;
            const std::string delta = piece_text.substr(sent);
            sent = piece_text.size();
            return !on_text || on_text(delta);
        };
        const GenerateResult r = generate(g, [&](int32_t token) {
            reply += decoder.push(token);
            return pass_on(false);
        });
        reply += decoder.flush();
        const Transcription::Transcript t = format.parse(prefix, reply, true);
        if (r.finish != FinishReason::Cancelled) pass_on(true);
        result.text += t.text;
        if (!t.language.empty() && (languages.empty() || languages.back() != t.language)) {
            languages.push_back(t.language);
        }
        accumulate(result.stats, r.stats);
        ++result.pieces;
        if (r.finish == FinishReason::Error) {
            result.finish = FinishReason::Error;
            result.error = r.error;
            return result;
        }
        if (r.finish == FinishReason::Cancelled) {
            cancelled = true;
            break;
        }
        if (r.finish == FinishReason::Length) result.finish = FinishReason::Length;
    }
    for (const std::string& l : languages) result.language += (result.language.empty() ? "" : ",") + l;
    if (cancelled) result.finish = FinishReason::Cancelled;
    return result;
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
        // Anything a request throws past the engine (which turns its errors into a result) reaches the caller, rather
        // than ending the process from this thread.
        try {
            job->done.set_value(engine_->generate(job->request, job->on_token));
        } catch (...) {
            job->done.set_exception(std::current_exception());
        }
    }
}

}  // namespace xs
