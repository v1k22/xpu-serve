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
    // Prompt lookup drafts with no model of its own, and verifies with the batch graph.
    const bool lookup = speculate && options.engine.ngram.has_value();
    const bool own_count = options.engine.spec_k == kDraftersCount;
    int spec_k = options.engine.spec_k;
    const bool block_drafter = options.spec == "dflash" || options.spec == "dspine";
    if (own_count) {
        spec_k = lookup                     ? kLookupDrafts
                 : options.spec == "dspark" ? kDsparkDrafts
                 : block_drafter            ? kBlockDrafts
                                            : kChainDrafts;
    }
    // Several requests in flight decode a step together: speculating, their rounds' verifies.
    const bool batch = options.engine.sequences > 1;
    auto model = load_model(model_dir, LoadOptions{.vision = options.vision,
                                                   .image_max_pixels = options.image_max_pixels,
                                                   .speculation = speculate && !lookup,
                                                   .batch = batch || lookup,
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
    const auto& slots = model->graph().kv_slots();
    engine_options.window_kv = options.window_kv && std::any_of(slots.begin(), slots.end(), [](const graph::KvSlot& k) {
                                   return k.sliding_window > 0;
                               });
    SessionOptions session_options = options.session;
    session_options.max_context = engine_options.max_context;
    session_options.window_kv = engine_options.window_kv;
    if (engine_options.window_kv && !model->graph().has_fixed_state()) {
        log::info("the sliding attention layers keep their window alone in the KV cache; a prompt's prefix is reused ",
                  "from the end of a chat's messages (--sliding-kv full keeps every key and reuses any prefix)");
    }
    session_options.sequences = engine_options.sequences;

    // Speculating, the session also runs the model's prefill, verify and draft programs (spec::Program order) and
    // cuts a verify back to its accepted tokens.
    std::vector<const graph::Graph*> programs{&model->graph()};
    if (lookup) {
        // The step graph prefills; the batch graph (kBatchProgram) verifies the drafts and cuts back to the accepted.
        const graph::Graph* g = model->batch_graph();
        if (g == nullptr) fail("--spec ngram: model ", info.id, " has no batch graph to verify drafts with");
        programs.push_back(g);
        session_options.rewind_tokens = engine_options.spec_k + 1;
        engine_options.batch = batch;
        log::info("speculative decoding: prompt lookup of ", engine_options.ngram->min, " to ",
                  engine_options.ngram->max, "-grams, up to ", engine_options.spec_k, " drafts per round",
                  batch ? str_cat(", the rounds of up to ", engine_options.sequences, " requests in flight verified ",
                                  "as one step of up to ", engine_options.verify_rows, " rows")
                        : std::string());
    } else if (speculate) {
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
        if (batch && spec->verify_graph().segmented()) {
            engine_options.batch = true;
            log::info("the speculative rounds of up to ", engine_options.sequences,
                      " requests in flight verify as one step of up to ", engine_options.verify_rows,
                      " rows (fewer drafts as more requests decode)");
            if (const graph::Graph* drafts = spec->chain() != nullptr ? spec->batch_draft_graph() : nullptr) {
                programs.push_back(drafts);  // spec::kBatchDraft
                engine_options.batch_draft = true;
                log::info("their drafts run as one run of the drafter a depth");
            }
        } else if (batch) {
            log::info("speculating, the ", engine_options.sequences, " requests in flight decode a round each in turn");
        }
    } else if (const graph::Graph* g = model->batch_graph()) {
        // the session also runs the batch graph (kBatchProgram) over the decode tokens of every sequence
        programs.push_back(g);
        engine_options.batch = true;
        log::info("the decode steps of up to ", engine_options.sequences, " requests in flight run as one");
    } else if (batch) {
        log::info(info.id, " has no batch graph: the ", engine_options.sequences, " requests in flight decode in turn");
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
        encoder_options.sequences = 1;
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
    // (spec::Program); a drafter's own steps are small. Batching, a batch step is a token of each sequence, and a
    // verify a round of each.
    const int64_t chunk = engine_options.prefill_chunk;
    const int64_t rounds = engine_options.batch ? engine_options.sequences : 1;
    session_options.max_step_tokens = {chunk};
    if (lookup) {
        session_options.max_step_tokens = {chunk, rounds * (engine_options.spec_k + 1)};
    } else if (speculate) {
        session_options.max_step_tokens = {0, chunk, rounds * (engine_options.spec_k + 1), 0};
        // A batched draft run: each sequence's waiting pairs (a round's accepted tokens and its pending one).
        if (engine_options.batch_draft) session_options.max_step_tokens.push_back(rounds * (engine_options.spec_k + 1));
    }
    if (engine_options.batch && !speculate) session_options.max_step_tokens.push_back(engine_options.sequences);
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
        const GenerateResult r = generate(g, [&](int32_t token, const TokenLogprobs*) {
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

// Requests the engine has a sequence for are in flight together, each advanced a step at a time: a prefill chunk,
// or a decode token (a speculative round). The requests still reading their prompts and those decoding share the
// device's time evenly from the moment both have work (a decoding chat is not left one token per chunk of a long
// prompt), each side's requests in turn.
void EngineService::worker_loop() {
    struct InFlight {
        Job* job;
        Engine::Handle handle;
    };
    std::vector<InFlight> active;
    size_t turn[2] = {0, 0};  // by side: prefilling, decoding
    double spent[2] = {0, 0};  // ms each side ran since both last had work
    bool had[2] = {false, false};
    for (;;) {
        for (;;) {  // admit what the engine has room for
            Job* job = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (active.empty()) cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (active.empty()) return;  // stopping and drained
                    break;
                }
                if (!engine_->has_free_sequence()) break;
                job = queue_.front();
                queue_.pop_front();
            }
            active.push_back({job, *engine_->start(job->request, job->on_token)});
        }
        std::vector<size_t> side[2];
        for (size_t i = 0; i < active.size(); ++i) side[engine_->decoding(active[i].handle) ? 1 : 0].push_back(i);
        for (int k = 0; k < 2; ++k) {
            const bool has = !side[k].empty();
            if (has && !had[k]) spent[0] = spent[1] = 0;  // a side that starts having work starts even
            had[k] = has;
        }
        const int k = side[0].empty() ? 1 : side[1].empty() ? 0 : spent[1] <= spent[0] ? 1 : 0;
        // A turn: one request's step, or with batching one step of every request that decodes.
        std::vector<size_t> turn_of{side[k][turn[k]++ % side[k].size()]};
        if (k == 1 && engine_->options().batch) turn_of = side[1];
        // Anything a request throws past the engine (which turns its errors into a result) reaches its caller,
        // rather than ending the process from this thread.
        std::vector<bool> done(turn_of.size(), true);
        Timer timer;
        try {
            std::vector<Engine::Handle> handles;
            for (const size_t at : turn_of) handles.push_back(active[at].handle);
            done = k == 1 ? engine_->advance_decoding(handles) : std::vector<bool>{engine_->advance(handles[0])};
            for (size_t j = 0; j < turn_of.size(); ++j) {
                if (done[j]) active[turn_of[j]].job->done.set_value(engine_->finish(active[turn_of[j]].handle));
            }
        } catch (...) {
            for (const size_t at : turn_of) {
                try {
                    active[at].job->done.set_exception(std::current_exception());
                } catch (const std::future_error&) {  // its result was already set
                }
            }
            std::fill(done.begin(), done.end(), true);
        }
        spent[k] += timer.elapsed_ms();
        for (size_t j = turn_of.size(); j-- > 0;) {
            if (done[j]) active.erase(active.begin() + static_cast<std::ptrdiff_t>(turn_of[j]));
        }
    }
}

}  // namespace xs
