#pragma once

// Decision models in tests without a released checkpoint: prompt tools that need no tokenizer file, a request's run
// on a session, and tiny checkpoints in the released ones' layouts.

#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "backend/backend.h"
#include "models/decision.h"
#include "tiny_checkpoints.h"

namespace xs::test {

// A byte is a token; the chat template writes each message as "<role>\n<text>\n".
class ByteTools final : public PromptTools {
public:
    std::vector<int32_t> encode(std::string_view text) const override {
        std::vector<int32_t> ids;
        for (const char c : text)
            ids.push_back(static_cast<unsigned char>(c));
        return ids;
    }
    std::optional<int32_t> token_id(std::string_view piece) const override {
        return piece.size() == 1 ? std::optional<int32_t>(static_cast<unsigned char>(piece[0])) : std::nullopt;
    }
    std::string chat(const Json& messages) const override {
        std::string out;
        for (const Json& m : messages) {
            out += m["role"].get<std::string>() + "\n";
            if (m["content"].is_string()) {
                out += m["content"].get<std::string>();
            } else {
                for (const Json& part : m["content"])
                    out += part.value("text", "");
            }
            out += "\n";
        }
        return out;
    }
};

// Runs `request` as the runtime does (decision/runner.cpp) and returns each question's probabilities.
inline std::vector<std::vector<double>> decide(const DecisionModel& model, Session& session,
                                               const DecisionRequest& request, DecisionLimits limits,
                                               int* steps = nullptr, size_t* sequences = nullptr) {
    const ByteTools tools;
    limits.held_tokens = session.num_cached(0);
    const std::unique_ptr<DecisionRun> run = model.start(request, tools, limits);
    TensorMap outputs;
    while (std::optional<DecisionRun::Step> step = run->next(outputs)) {
        if (step->keep == 0) session.reset();
        if (step->keep > 0 && session.num_cached(0) > step->keep) session.truncate(0, step->keep);
        outputs.clear();
        session.run(step->program, step->inputs, outputs, step->outputs);
        if (steps != nullptr) ++*steps;
    }
    if (sequences != nullptr) *sequences = run->sequences().size();
    return run->probabilities();
}

inline DecisionQuestion choice(std::string id, std::string instructions,
                               std::vector<std::pair<std::string, Json>> options) {
    DecisionQuestion q;
    q.id = std::move(id);
    q.type = QuestionType::Choice;
    q.instructions = std::move(instructions);
    for (auto& [name, description] : options)
        q.options.push_back({name, std::move(description)});
    return q;
}
inline DecisionQuestion noul(std::string id, std::string instructions) {
    DecisionQuestion q;
    q.id = std::move(id);
    q.type = QuestionType::Noul;
    q.instructions = std::move(instructions);
    q.options = {{"true", std::nullopt}, {"false", std::nullopt}};
    return q;
}
inline DecisionQuestion score(std::string id, std::string instructions, std::vector<std::string> levels) {
    DecisionQuestion q;
    q.id = std::move(id);
    q.type = QuestionType::Score;
    q.instructions = std::move(instructions);
    for (size_t i = 0; i < levels.size(); ++i)
        q.options.push_back({std::to_string(i), Json(levels[i])});
    return q;
}

// A joint schema head over a decoder of width `hidden`, as Cloudflare/clef-flash lays it out (joint_head_config.json,
// joint_head.safetensors with torch's tensor names). Its output scales are large enough for answers to spread.
inline void write_tiny_clef_head(const fs::path& dir, int64_t hidden, uint32_t seed) {
    const int64_t width = 64, ffn = 96;
    std::ofstream(dir / "joint_head_config.json")
        << nlohmann::json{{"hidden_size", hidden}, {"width", width}, {"routing_layers", 2},
                          {"layers", 2},           {"heads", 4},     {"feedforward", ffn}}
               .dump(1);
    std::vector<TinyTensor> specs;
    const auto matrix = [&](const std::string& name, int64_t out, int64_t in) {
        const float range = 1.0f / std::sqrt(static_cast<float>(in));
        specs.push_back({name, {out, in}, -range, range});
    };
    const auto bias = [&](const std::string& name, int64_t n) { specs.push_back({name, {n}, -0.2f, 0.2f}); };
    const auto norm = [&](const std::string& stem, int64_t n) {
        specs.push_back({stem + ".weight", {n}, 0.7f, 1.3f});
        bias(stem + ".bias", n);
    };
    const auto linear = [&](const std::string& stem, int64_t out, int64_t in) {
        matrix(stem + ".weight", out, in);
        bias(stem + ".bias", out);
    };
    const auto attention = [&](const std::string& stem) {
        matrix(stem + ".in_proj_weight", 3 * width, width);
        bias(stem + ".in_proj_bias", 3 * width);
        linear(stem + ".out_proj", width, width);
    };
    norm("hidden_norm", hidden);
    for (const char* p : {"memory_projection", "question_projection", "option_question_projection", "global_projection",
                          "option_context_projection", "option_lexical_projection"}) {
        matrix(std::string(p) + ".weight", width, hidden);
    }
    specs.push_back({"type_embedding.weight", {3, width}, -1.0f, 1.0f});
    for (int i = 0; i < 2; ++i) {
        const std::string L = "evidence_layers." + std::to_string(i);
        norm(L + ".query_norm", width);
        norm(L + ".memory_norm", width);
        attention(L + ".attention");
        norm(L + ".feedforward_norm", width);
        linear(L + ".feedforward.0", ffn, width);
        linear(L + ".feedforward.3", width, ffn);
    }
    norm("option_summary_norm", width);
    for (int i = 0; i < 2; ++i) {
        const std::string L = "layers." + std::to_string(i);
        attention(L + ".self_attn");
        attention(L + ".multihead_attn");
        linear(L + ".linear1", ffn, width);
        linear(L + ".linear2", width, ffn);
        for (const char* n : {".norm1", ".norm2", ".norm3"})
            norm(L + n, width);
    }
    norm("field_norm", width);
    norm("option_norm", width);
    linear("residual_scorer.0", width, 4 * width);
    specs.push_back({"residual_scorer.3.weight", {1, width}, -2.0f, 2.0f});
    bias("residual_scorer.3.bias", 1);
    specs.push_back({"prior_logit_scale", {1}, 1.5f, 1.5f});
    specs.push_back({"joint_logit_scale", {1}, 2.0f, 2.0f});
    specs.push_back({"residual_gate", {1}, 0.7f, 0.7f});
    write_random(dir / "joint_head.safetensors", specs, seed);
}

// A tiny Qwen3.5 with a joint schema head beside it, as Cloudflare/clef-flash is laid out.
inline void write_tiny_clef(const fs::path& dir, uint32_t seed) {
    const TinyQwen qwen;
    write_tiny_qwen(dir, qwen, seed);
    write_tiny_clef_head(dir, qwen.hidden, seed + 1);
}

// A tiny Qwen3.5 with a read-out of answer codes beside it, as perplexity-ai/pplx-decider-flash-v1.2 is laid out.
// The marker is the byte 0x01, a token of ByteTools.
inline void write_tiny_pplx(const fs::path& dir, uint32_t seed, int codes = 30) {
    const TinyQwen qwen;
    write_tiny_qwen(dir, qwen, seed);
    nlohmann::json decision = {{"format_version", 2},
                               {"attention_mode", "noncausal_full_attention"},
                               {"pooling", "last"},
                               {"temperature", 1.5},
                               {"codes", nlohmann::json::array()}};
    for (int i = 0; i < codes; ++i) {
        decision["codes"].push_back(i < 26 ? std::string(1, static_cast<char>('A' + i))
                                           : std::string("A") + static_cast<char>('A' + i - 26));
    }
    std::ofstream(dir / "decision_config.json") << decision.dump(1);
    std::ofstream(dir / "joint_config.json") << nlohmann::json{{"version", 3},
                                                               {"template", "joint-question-inline-readout-v1"},
                                                               {"marker", "\u0001"},
                                                               {"readout_offset", 0}}
                                                    .dump(1);
    const float range = 6.0f / std::sqrt(static_cast<float>(qwen.hidden));
    write_random(dir / "readout.safetensors", {{"weight", {codes, qwen.hidden}, -range, range}}, seed + 1);
}

}  // namespace xs::test
