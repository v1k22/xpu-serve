// A real gemma4 GGUF (XS_GGUF_FILE, e.g. unsloth/gemma-4-E2B-it-GGUF's gemma-4-E2B-it-Q4_K_M.gguf) against the
// Hugging Face checkpoint it was converted from (XS_GEMMA4_DIR: its config.json, generation_config.json,
// tokenizer.json, tokenizer_config.json and chat_template.jinja; the weights are not read). The tokenizer written from
// the file passes the Gemma 4 tokenizer parity cases and matches the checkpoint's token by token; config.json gives the
// model code what the checkpoint's does; the file's chat template renders thinking, tools and a tool round trip; the
// imported pack loads and answers. Each case skips unless both variables are set and the file is a gemma4 one.
//
// A real gemma4-assistant GGUF (XS_GGUF_DRAFTER_FILE, e.g. MTP/mtp-gemma-4-E2B-it-Q8_0.gguf) against the assistant
// checkpoint (XS_GEMMA4_ASSISTANT_DIR: its config.json): the config fields the model code reads; beside the target
// above, the imported drafter loads and drafts.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>

#include "chat/chat_template.h"
#include "convert/gguf/architecture.h"
#include "convert/gguf/gguf_import.h"
#include "convert/gguf/tokenizer.h"
#include "convert/import_test_util.h"
#include "convert/real_gguf.h"
#include "engine/spec/chain_drafter.h"
#include "io/gguf.h"
#include "model/tiny_checkpoints.h"
#include "models/config_util.h"
#include "models/gemma4/gemma4.h"
#include "models/model.h"
#include "tokenizer/parity_cases.h"
#include "tokenizer/tokenizer.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::TempDir;

namespace {

std::optional<xs::test::RealGguf> real_file() {
    return xs::test::real_gguf("XS_GEMMA4_DIR", "gemma4");
}

// A merge as "a b", whether tokenizer.json lists it so or as a pair.
std::string merge_text(const nlohmann::json& m) {
    return m.is_string() ? m.get<std::string>() : m[0].get<std::string>() + " " + m[1].get<std::string>();
}

std::set<int32_t> as_set(const std::vector<int32_t>& v) {
    return {v.begin(), v.end()};
}

// XS_GGUF_DRAFTER_FILE, when it is set and a gemma4-assistant file.
std::optional<fs::path> drafter_file() {
    const char* d = std::getenv("XS_GGUF_DRAFTER_FILE");
    if (d == nullptr || *d == '\0') return std::nullopt;
    if (gguf::File(d).get_string("general.architecture") != "gemma4-assistant") return std::nullopt;
    return fs::path(d);
}

}  // namespace

TEST_SUITE("gguf real file") {
    TEST_CASE("tokenizer parity: a real gemma4 GGUF's tokenizer encodes and decodes as Hugging Face's") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_gemma4_real_tokenizer_test");
        const gguf::File file(real->gguf);
        xs::test::write_gguf_tokenizer(file, work.path);
        const auto imported = tok::Tokenizer::load(work.path);
        const auto original = tok::Tokenizer::load(real->hf);

        // The parity cases (written from google/gemma-4-E2B-it); its eos_token is a tokenizer_config.json setting
        // the file names otherwise.
        nlohmann::json fixture = nlohmann::json::parse(
            std::ifstream(fs::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/gemma4_cases.json"));
        MESSAGE("parity cases: " << fixture["encode_cases"].size() << " encode, " << fixture["decode_cases"].size()
                                 << " decode");
        if (imported->eos_id() != fixture["eos_id"].get<int32_t>()) {
            MESSAGE("eos_token: the file names " << *imported->eos_id() << " ('"
                                                 << imported->token_text(*imported->eos_id())
                                                 << "'), the parity cases' tokenizer " << fixture["eos_id"]);
            fixture["eos_id"] = *imported->eos_id();
        }
        // Streamed, the decode-only cases of invalid byte runs come out otherwise than Hugging Face's whole decode,
        // for the checkpoint's tokenizer as for this one (tests/unit/tokenizer/test_tokenizer_parity.cpp does not
        // stream them): there they are checked against the checkpoint's.
        nlohmann::json decode_cases = std::move(fixture["decode_cases"]);
        fixture["decode_cases"] = nlohmann::json::array();
        xs::test::check_tokenizer_cases(*imported, fixture);
        for (const auto& c : decode_cases) {
            CAPTURE(c["name"].get<std::string>());
            const std::vector<int32_t> ids = c["ids"].get<std::vector<int32_t>>();
            CHECK(imported->decode(ids) == c["decoded"].get<std::string>());
            CHECK(imported->decode(ids, true) == c["decoded_skip_special"].get<std::string>());
            const std::string streamed = xs::test::stream_decode(*imported, ids, false);
            CHECK(streamed == xs::test::stream_decode(*original, ids, false));
            if (streamed != c["decoded"].get<std::string>()) {
                MESSAGE("streamed otherwise than Hugging Face decodes it whole, as the checkpoint's tokenizer does: "
                        << c["name"].get<std::string>());
            }
        }

        // Token by token against the checkpoint's tokenizer: the same text, special flag, encoding of the text and
        // decoding for every id.
        REQUIRE(imported->vocab_size() == original->vocab_size());
        int differ = 0;
        for (int32_t id = 0; id < original->vocab_size(); ++id) {
            const std::string text = original->token_text(id);
            const std::vector<int32_t> one{id};
            if (imported->token_text(id) != text || imported->is_special(id) != original->is_special(id) ||
                imported->encode(text) != original->encode(text) || imported->decode(one) != original->decode(one)) {
                if (++differ <= 10) MESSAGE("token " << id << " differs: '" << text << "'");
            }
        }
        MESSAGE(original->vocab_size() << " tokens compared, " << differ << " differ");
        CHECK(differ == 0);
        CHECK(imported->bos_id() == original->bos_id());

        // The parts the file names only by tokenizer.ggml.model are the checkpoint's.
        const nlohmann::json written = nlohmann::json::parse(std::ifstream(work.path / "tokenizer.json"));
        const nlohmann::json hf = nlohmann::json::parse(std::ifstream(real->hf / "tokenizer.json"));
        for (const char* part : {"normalizer", "pre_tokenizer", "post_processor", "decoder", "added_tokens"}) {
            CAPTURE(part);
            CHECK(written[part] == hf[part]);
        }
        const bool same_vocab = written["model"]["vocab"] == hf["model"]["vocab"];
        CHECK(same_vocab);
        const nlohmann::json& merges = written["model"]["merges"];
        const nlohmann::json& hf_merges = hf["model"]["merges"];
        int merges_differ = 0;
        for (size_t i = 0; i < std::min(merges.size(), hf_merges.size()); ++i)
            merges_differ += merge_text(merges[i]) != merge_text(hf_merges[i]);
        MESSAGE(merges.size() << " merges, the checkpoint's " << hf_merges.size() << "; " << merges_differ
                              << " of the first " << std::min(merges.size(), hf_merges.size()) << " differ");
        CHECK(merges.size() == hf_merges.size());
        CHECK(merges_differ == 0);
        nlohmann::json settings = hf["model"], written_settings = written["model"];
        for (nlohmann::json* m : {&settings, &written_settings}) {
            m->erase("vocab");
            m->erase("merges");
        }
        CHECK(written_settings == settings);
    }

    TEST_CASE("config: what the model code reads from a real gemma4 GGUF equals the checkpoint's config") {
        const auto real = real_file();
        if (!real) return;
        const gguf::File file(real->gguf);
        const gguf_import::ImportPlan plan =
            gguf_import::find_architecture(file.get_string("general.architecture")).plan(file);
        const nlohmann::json ours = nlohmann::json::parse(plan.config.dump());
        const nlohmann::json theirs = cfg::read_json(real->hf / "config.json");
        const gemma4::Config a = gemma4::parse_config(ours);
        const gemma4::Config b = gemma4::parse_config(theirs);
        CHECK(a.checkpoint_dtype == b.checkpoint_dtype);
        CHECK(a.vocab_size == b.vocab_size);
        CHECK(a.hidden_size == b.hidden_size);
        CHECK(a.intermediate_size == b.intermediate_size);
        CHECK(a.num_layers == b.num_layers);
        CHECK(a.num_heads == b.num_heads);
        CHECK(a.num_kv_heads == b.num_kv_heads);
        CHECK(a.num_global_kv_heads == b.num_global_kv_heads);
        CHECK(a.k_eq_v == b.k_eq_v);
        CHECK(a.head_dim == b.head_dim);
        CHECK(a.global_head_dim == b.global_head_dim);
        CHECK(a.sliding_window == b.sliding_window);
        CHECK(a.num_kv_shared_layers == b.num_kv_shared_layers);
        CHECK(a.ple_dim == b.ple_dim);
        CHECK(a.ple_vocab_size == b.ple_vocab_size);
        CHECK(a.double_wide_mlp == b.double_wide_mlp);
        CHECK(a.rms_norm_eps == b.rms_norm_eps);
        CHECK(a.final_logit_softcap == b.final_logit_softcap);
        CHECK(a.num_experts == b.num_experts);
        CHECK(a.max_position == b.max_position);
        CHECK(a.layer_kinds == b.layer_kinds);
        for (const auto& [x, y] : {std::pair{a.rope_sliding, b.rope_sliding}, std::pair{a.rope_full, b.rope_full}}) {
            CHECK(x.theta == y.theta);
            CHECK(x.proportional == y.proportional);
            CHECK(x.partial_rotary_factor == y.partial_rotary_factor);
        }
        const nlohmann::json& text = theirs.contains("text_config") ? theirs["text_config"] : theirs;
        CHECK(ours["tie_word_embeddings"] == text["tie_word_embeddings"]);
        CHECK(ours["pad_token_id"] == text["pad_token_id"]);
        CHECK(ours["bos_token_id"] == text["bos_token_id"]);
        // The vision and audio towers are not in the file: the pack is the text model.
        CHECK_FALSE(ours.contains("vision_config"));

        // Stop tokens: the file names one end-of-sequence token, the import adds Gemma 4's end-of-turn tokens; the
        // checkpoint names its own in config.json and generation_config.json. The same set.
        const std::vector<int32_t> stops = gguf_import::tokenizer_files(file).stop_ids;
        std::vector<int32_t> hf_stops = cfg::get_int_list(theirs, "eos_token_id", "config.json");
        const std::vector<int32_t> text_stops = cfg::get_int_list(text, "eos_token_id", "config.json text_config");
        hf_stops.insert(hf_stops.end(), text_stops.begin(), text_stops.end());
        std::vector<int32_t> generation;
        cfg::read_generation_defaults(real->hf, &generation);
        hf_stops.insert(hf_stops.end(), generation.begin(), generation.end());
        CHECK(as_set(stops) == as_set(hf_stops));
        MESSAGE("stop tokens: the file names " << file.get_int("tokenizer.ggml.eos_token_id") << ", the pack stops at "
                                               << stops.size() << " tokens, the checkpoint at "
                                               << as_set(hf_stops).size());

        // Sampling defaults: the file's general.sampling.* against the checkpoint's generation_config.json.
        const SamplingParams g = cfg::read_generation_defaults(real->hf, nullptr);
        if (const gguf::Value* v = file.find("general.sampling.top_k")) CHECK(v->as_int() == g.top_k);
        if (const gguf::Value* v = file.find("general.sampling.top_p"))
            CHECK(static_cast<float>(v->as_float()) == g.top_p);
        if (const gguf::Value* v = file.find("general.sampling.temp")) {
            CHECK(static_cast<float>(v->as_float()) == g.temperature);
        }
    }

    TEST_CASE("the chat template of a real gemma4 GGUF renders thinking, tools and a tool round trip") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_gemma4_real_template_test");
        const gguf::File file(real->gguf);
        xs::test::write_gguf_tokenizer(file, work.path);
        const std::string source = gguf_import::chat_template(file);
        std::ofstream(work.path / "chat_template.jinja") << source;
        const auto chat = chat::ChatTemplate::load(work.path);
        const nlohmann::json messages = nlohmann::json::array(
            {{{"role", "system"}, {"content", "Be brief."}}, {{"role", "user"}, {"content", "What is 2 + 2?"}}});
        const std::string off = chat->render(messages, nullptr, true);
        CHECK(off == "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nWhat is 2 + 2?<turn|>\n<|turn>model\n");
        const std::string on = chat->render(messages, nullptr, true, {{"enable_thinking", true}});
        CHECK(on.find("<|turn>system\n<|think|>") != std::string::npos);

        const nlohmann::json tools = nlohmann::json::parse(R"([{"type": "function", "function": {
            "name": "get_weather", "description": "The weather in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])");
        const nlohmann::json user = {{"role", "user"}, {"content", "Weather in Paris?"}};
        const std::string declared = chat->render(nlohmann::json::array({user}), tools, true);
        CHECK(declared.find("<|tool>declaration:get_weather{") != std::string::npos);
        const nlohmann::json round_trip = nlohmann::json::array(
            {user,
             {{"role", "assistant"},
              {"content", ""},
              {"tool_calls",
               {{{"type", "function"},
                 {"function", {{"name", "get_weather"}, {"arguments", R"({"city": "Paris"})"}}}}}}},
             {{"role", "tool"}, {"name", "get_weather"}, {"content", R"({"temperature": 21})"}}});
        const std::string answered = chat->render(round_trip, tools, true);
        CHECK(answered.find("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>") != std::string::npos);
        CHECK(answered.find("<|tool_response>response:get_weather{") != std::string::npos);

        // The checkpoint's own renderings (tests/fixtures/chat, from google/gemma-4-E2B-it's template).
        const nlohmann::json cases =
            nlohmann::json::parse(std::ifstream(fs::path(XS_SOURCE_DIR) / "tests/fixtures/chat/gemma4_cases.json"));
        int same = 0;
        for (const nlohmann::json& c : cases) {
            CAPTURE(c["name"].get<std::string>());
            std::string rendered;
            CHECK_NOTHROW(
                rendered = chat->render(c["messages"], c["tools"], c["add_generation_prompt"].get<bool>(), c["extra"]));
            if (rendered == c["expected"].get<std::string>()) {
                ++same;
            } else {
                MESSAGE("renders otherwise than the checkpoint's template: " << c["name"].get<std::string>());
            }
        }
        MESSAGE(same << " of " << cases.size() << " of the checkpoint's renderings are the same");
        CHECK(same == static_cast<int>(cases.size()));
        std::ifstream hf(real->hf / "chat_template.jinja");
        if (hf && std::string(std::istreambuf_iterator<char>(hf), {}) != source) {
            MESSAGE("the file's chat template is not the checkpoint's chat_template.jinja");
        }
    }

    TEST_CASE("a real gemma4 GGUF imports into a pack that loads and answers") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_gemma4_real_import_test");
        const gguf_import::ImportReport report = xs::test::import_into(real->gguf, work.path / "pack");
        CHECK(report.architecture == "gemma4");
        const auto model = load_model(work.path / "pack");
        const gguf::File file(real->gguf);
        CHECK(model->info().vocab_size == file.find_tensor("token_embd.weight")->rows());
        CHECK(model->vision() == nullptr);
        for (const auto& [gguf_name, pack_name] :
             {std::pair{"token_embd.weight", "model.embed_tokens.weight"},
              std::pair{"per_layer_token_embd.weight", "model.embed_tokens_per_layer.weight"}}) {
            const gguf::TensorInfo& t = *file.find_tensor(gguf_name);
            const TensorView& v = model->weights().get(pack_name);
            CHECK(v.dtype == t.dtype);
            CHECK(std::memcmp(v.data, t.data, t.nbytes) == 0);
        }

        // Greedy, on cpu-ref, through the pack's own tokenizer and chat template.
        const auto tokenizer = tok::Tokenizer::load(work.path / "pack");
        const auto chat = chat::ChatTemplate::load(work.path / "pack");
        const std::string prompt = chat->render(
            nlohmann::json::array({{{"role", "user"}, {"content", "What is the capital of France? One word."}}}),
            nullptr, true);
        std::vector<int32_t> tokens = xs::test::greedy(*model, tokenizer->encode(prompt), 4);
        const auto& stops = model->info().stop_token_ids;
        const auto end = std::find_if(tokens.begin(), tokens.end(), [&](int32_t t) {
            return std::find(stops.begin(), stops.end(), t) != stops.end();
        });
        const std::string answer = tokenizer->decode(std::vector<int32_t>(tokens.begin(), end));
        MESSAGE("answer: '" << answer << "'");
        CHECK(answer.find("Paris") != std::string::npos);
        CHECK(end != tokens.end());

        // The imported drafter beside it: it loads (its shapes fit this target) and drafts after the prompt.
        const auto drafter_gguf = drafter_file();
        if (!drafter_gguf) return;
        xs::test::import_into(*drafter_gguf, work.path / "drafter");
        const auto both = load_model(
            work.path / "pack",
            LoadOptions{.vision = false, .speculation = true, .drafter = work.path / "drafter", .draft_shortlist = 0});
        const Speculation* spec = both->speculation();
        REQUIRE(spec != nullptr);
        const graph::Graph* programs[] = {&both->graph(), &spec->prefill_graph(), &spec->verify_graph(),
                                          &spec->draft_graph()};
        SessionOptions options = xs::test::f32_session();
        options.rewind_tokens = 4;
        const auto session = make_cpu_ref_backend()->compile(programs, both->weights(), options);
        const std::vector<int32_t> ids = tokenizer->encode(prompt);
        TensorMap inputs, outputs;
        both->prepare_inputs(ids, 0, {}, inputs);
        session->run(spec::kPrefill, inputs, outputs);
        spec::ChainDrafter chain(*spec, *session);
        chain.add(TensorMap{{"hidden", outputs.at("hidden")}}, outputs.at("hidden").dim(0), 0, ids);
        const std::vector<int32_t> drafted = chain.propose(tokens[0], 3);
        REQUIRE(drafted.size() == 3);
        MESSAGE("after '" << tokenizer->decode(std::vector<int32_t>{tokens[0]}) << "' the drafter proposes '"
                          << tokenizer->decode(drafted) << "'; the target continues '"
                          << tokenizer->decode(std::vector<int32_t>(tokens.begin() + 1, tokens.end())) << "'");
        CHECK(drafted[0] == tokens[1]);
    }

    TEST_CASE("config: what the model code reads from a real gemma4-assistant GGUF equals the assistant's config") {
        const auto drafter_gguf = drafter_file();
        const char* dir = std::getenv("XS_GEMMA4_ASSISTANT_DIR");
        if (!drafter_gguf || dir == nullptr || *dir == '\0') return;
        const gguf::File file(*drafter_gguf);
        const gguf_import::ImportPlan plan = gguf_import::find_architecture("gemma4-assistant").plan(file);
        CHECK(plan.kind == gguf_import::ImportPlan::Kind::Drafter);
        const gemma4::AssistantConfig a = gemma4::parse_assistant_config(nlohmann::json::parse(plan.config.dump()));
        const gemma4::AssistantConfig b = gemma4::parse_assistant_config(cfg::read_json(fs::path(dir) / "config.json"));
        CHECK(a.backbone_hidden_size == b.backbone_hidden_size);
        CHECK(a.text.checkpoint_dtype == b.text.checkpoint_dtype);
        CHECK(a.text.vocab_size == b.text.vocab_size);
        CHECK(a.text.hidden_size == b.text.hidden_size);
        CHECK(a.text.intermediate_size == b.text.intermediate_size);
        CHECK(a.text.num_layers == b.text.num_layers);
        CHECK(a.text.num_heads == b.text.num_heads);
        CHECK(a.text.num_kv_heads == b.text.num_kv_heads);
        CHECK(a.text.num_global_kv_heads == b.text.num_global_kv_heads);
        CHECK(a.text.head_dim == b.text.head_dim);
        CHECK(a.text.global_head_dim == b.text.global_head_dim);
        CHECK(a.text.sliding_window == b.text.sliding_window);
        CHECK(a.text.num_kv_shared_layers == b.text.num_kv_shared_layers);
        CHECK(a.text.rms_norm_eps == b.text.rms_norm_eps);
        CHECK(a.text.final_logit_softcap == b.text.final_logit_softcap);
        CHECK(a.text.max_position == b.text.max_position);
        CHECK(a.text.layer_kinds == b.text.layer_kinds);
        for (const auto& [x, y] :
             {std::pair{a.text.rope_sliding, b.text.rope_sliding}, std::pair{a.text.rope_full, b.text.rope_full}}) {
            CHECK(x.theta == y.theta);
            CHECK(x.proportional == y.proportional);
            CHECK(x.partial_rotary_factor == y.partial_rotary_factor);
        }
        // What differs, by construction: the file has no centroid head, and llama.cpp scores the whole vocabulary.
        CHECK_FALSE(a.ordered_embeddings);
        if (b.ordered_embeddings) {
            MESSAGE("the checkpoint's head: " << b.num_centroids << " centroids, top " << b.centroid_top_k
                                              << "; the file's: the whole vocabulary");
        }
    }
}
