#include <algorithm>
#include <fstream>
#include <optional>

#include "io/joined_weights.h"
#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/drafters/dflash.h"
#include "models/qwen3_5/qwen3_5.h"
#include "models/qwen3_5/qwen3_5_decoder.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/log.h"
#include "util/timer.h"

namespace xs::qwen3_5 {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

// A mixture-of-experts checkpoint that stores each expert's projections on their own (the FP8 releases): the banks the
// model reads are their rows in expert order, an expert's gate rows before its up rows.
void join_expert_banks(SafetensorsWeights& weights, const Config& cfg) {
    if (cfg.num_experts == 0) return;
    std::vector<std::string> layers;
    const std::string prefix = Decoder(cfg, weights).prefix();
    for (int64_t i = 0; i < cfg.num_layers; ++i) layers.push_back(prefix + "layers." + std::to_string(i));
    for (int64_t i = 0; i < cfg.mtp_layers; ++i) layers.push_back("mtp.layers." + std::to_string(i));
    for (const std::string& layer : layers) {
        const std::string E = layer + ".mlp.experts.";
        if (weights.contains(E + "gate_up_proj") || !weights.contains(E + "0.gate_proj.weight")) continue;
        std::vector<std::string> gate_up, down;
        for (int64_t e = 0; e < cfg.num_experts; ++e) {
            const std::string X = E + std::to_string(e) + ".";
            gate_up.push_back(X + "gate_proj.weight");
            gate_up.push_back(X + "up_proj.weight");
            down.push_back(X + "down_proj.weight");
        }
        const int64_t experts = cfg.num_experts, inter = cfg.moe_intermediate_size;
        weights.join_rows(E + "gate_up_proj", gate_up, {experts, 2 * inter, cfg.hidden_size});
        weights.join_rows(E + "down_proj", down, {experts, cfg.hidden_size, inter});
    }
}

// Where a block drafter's weights sit beside the target's.
constexpr const char* kDrafterPrefix = "drafter.";

// The programs of a Qwen3.5 checkpoint, built from one set of layer builders: the step graph, and with speculation the
// prefill and verify programs (the same decoder, other outputs) and the drafter's. The drafter is the checkpoint's MTP
// head (vLLM's Qwen3.5 MTP: one decoder layer of the target's kind over fc(cat(norm(embed(t_{i+1})), norm(h_i))),
// then the target's lm_head), or, with a drafter checkpoint beside the model, a block drafter of the DFlash family
// (models/drafters/dflash.h), whose context the prefill and verify programs append.
class Qwen35Model final : public Model, public Speculation, public ChainDrafting {
public:
    // `drafter`: a block drafter's config, its weights already in `weights` under kDrafterPrefix.
    Qwen35Model(Config cfg, ModelInfo info, std::unique_ptr<WeightStore> weights, const LoadOptions& options,
                std::optional<dflash::Config> drafter)
        : cfg_(std::move(cfg)), info_(std::move(info)), weights_(std::move(weights)), dec_(cfg_, *weights_) {
        build_target(step_, Head::LastLogits);
        if (!options.speculation) return;
        if (drafter) {
            XS_CHECK(drafter->hidden_size == cfg_.hidden_size && drafter->vocab_size == cfg_.vocab_size &&
                         drafter->num_target_layers == cfg_.num_layers,
                     "qwen3_5: the drafter was trained for a target of ", drafter->num_target_layers, " layers of ",
                     drafter->hidden_size, " and ", drafter->vocab_size, " tokens; this model has ", cfg_.num_layers,
                     " of ", cfg_.hidden_size, " and ", cfg_.vocab_size);
            block_ = std::make_unique<dflash::Drafter>(std::move(*drafter), *weights_, kDrafterPrefix, options.drafts);
            build_target(prefill_, Head::LastLogitsAllHidden);
            build_target(verify_, Head::AllLogitsAllHidden);
            block_->build_block(draft_, dec_.embedding(draft_), dec_.lm_head(draft_));
            return;
        }
        if (cfg_.mtp_layers == 0 || !weights_->contains("mtp.fc.weight")) {
            log::warn("qwen3_5: the checkpoint has no MTP head (mtp.*); speculative decoding is unavailable");
            return;
        }
        if (cfg_.mtp_layers != 1) fail("qwen3_5: an MTP head of ", cfg_.mtp_layers, " layers is not supported (1 is)");
        build_target(prefill_, Head::LastLogitsAllHidden);
        build_target(verify_, Head::AllLogitsAllHidden);
        build_mtp(draft_, options.draft_shortlist);
        has_mtp_ = true;
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return step_; }
    const WeightStore& weights() const override { return *weights_; }
    const Speculation* speculation() const override { return has_mtp_ || block_ ? this : nullptr; }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        XS_CHECK(embeddings.empty(), "qwen3_5: image input is not supported yet");
        dec_.token_inputs(tokens, start_pos, inputs);
    }

    const graph::Graph& prefill_graph() const override { return prefill_; }
    const graph::Graph& verify_graph() const override { return verify_; }
    const graph::Graph& draft_graph() const override { return draft_; }
    const ChainDrafting* chain() const override { return block_ ? nullptr : this; }
    const BlockDrafting* block() const override { return block_.get(); }
    std::string_view drafter_name() const override { return block_ ? dflash::variant_name(block_->config().variant) : "mtp"; }
    std::string_view drafter_weights() const override { return block_ ? kDrafterPrefix : ""; }
    bool drafter_keeps_state() const override { return true; }

    // The pair (h_i, token i + 1) runs at position i, a chained draft one position after the previous one.
    void prepare_draft(std::span<const int32_t> tokens, std::span<const float> hidden, int64_t start_pos, int depth,
                       TensorMap& inputs) const override {
        const auto n = static_cast<int64_t>(tokens.size());
        XS_CHECK(static_cast<int64_t>(hidden.size()) == n * cfg_.hidden_size, "qwen3_5: ", n, " MTP tokens need ",
                 n * cfg_.hidden_size, " hidden values, got ", hidden.size());
        dec_.token_inputs(tokens, start_pos + depth, inputs);
        inputs["hidden"] = HostTensor::from_vector(DType::F32, {n, cfg_.hidden_size},
                                                   std::vector<float>(hidden.begin(), hidden.end()));
    }

private:
    enum class Head {
        LastLogits,           // the step graph
        LastLogitsAllHidden,  // prefill with speculation
        AllLogitsAllHidden,   // verify
    };

    void build_target(Graph& g, Head head) const {
        const Decoder::Inputs in = dec_.inputs(g);
        const std::vector<Value> after = dec_.layers(g, in);  // the state after each layer
        const Value h = after.back();
        if (head == Head::LastLogits) {
            Graph::Scope scope(g, "head");
            const Value logits = g.matmul(dec_.final_norm(g, g.slice(h, 0, -1)), dec_.lm_head(g));
            g.output("logits", logits);
            g.output("argmax", blocks::argmax_rows(g, logits));
        } else {
            const Value hidden = dec_.final_norm(g, h);
            if (block_ != nullptr) {
                // A block drafter's context: Hugging Face's hidden_states after the layers it names, the last of which
                // is the normed one.
                std::vector<Value> states;
                for (const int64_t id : block_->config().target_layer_ids) {
                    states.push_back(id + 1 < cfg_.num_layers ? after[static_cast<size_t>(id)] : hidden);
                }
                block_->append_context(g, states, in.pos);
            }
            Graph::Scope scope(g, "head");
            const Value logits =
                g.matmul(head == Head::AllLogitsAllHidden ? hidden : g.slice(hidden, 0, -1), dec_.lm_head(g));
            g.output("logits", logits);
            g.output("argmax", blocks::argmax_rows(g, logits));
            if (block_ == nullptr) g.output("hidden", hidden);  // what the MTP head takes
        }
        g.validate();
        log::debug("qwen3_5: graph has ", g.nodes().size(), " nodes, ", g.kv_slots().size(), " kv, ",
                   g.conv_slots().size(), " conv and ", g.recurrent_slots().size(), " recurrent slots");
    }

    // `shortlist` > 0: the drafts pick among that many tokens (blocks::shortlist_head) instead of reading the whole
    // head, which is most of a draft step.
    void build_mtp(Graph& g, int64_t shortlist) const {
        const Decoder::Inputs in = dec_.inputs(g);
        const Value hidden = g.input("hidden", DType::F32, {kT, cfg_.hidden_size});
        Graph::Scope scope(g, "mtp");
        const Value e = dec_.norm(g, g.gather(dec_.embedding(g), in.ids), "mtp.pre_fc_norm_embedding.weight");
        const Value x = g.matmul(g.concat({e, dec_.norm(g, hidden, "mtp.pre_fc_norm_hidden.weight")}, 1),
                                 dec_.weight(g, "mtp.fc.weight"));
        const Value h = dec_.layer(g, "mtp.layers.0.", "mtp.layers.0", LayerKind::Full, x, in.pos, dec_.inv_freq(g));
        const Value last = dec_.norm(g, g.slice(h, 0, -1), "mtp.norm.weight");
        if (shortlist > 0 && shortlist < cfg_.vocab_size && cfg_.hidden_size % 32 == 0) {
            Timer timer;
            const quant::SignSketch sketch = quant::sign_sketch(weights_->get(dec_.lm_head_name()));
            log::info("qwen3_5: drafts pick among ", shortlist, " of ", cfg_.vocab_size, " tokens (a one-bit sketch of ",
                      "the head, ", sketch.signs.nbytes() >> 20, " MiB, built in ", static_cast<int>(timer.elapsed_ms()),
                      " ms)");
            const blocks::Shortlist s = blocks::shortlist_head(g, last, dec_.lm_head(g), sketch, shortlist);
            g.output("logits", s.logits);
            g.output("candidates", s.candidates);
        } else {
            g.output("logits", g.matmul(last, dec_.lm_head(g)));
        }
        g.output("hidden", last);
        g.validate();
    }

    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<WeightStore> weights_;
    Decoder dec_;
    graph::Graph step_, prefill_, verify_, draft_;
    bool has_mtp_ = false;
    std::unique_ptr<dflash::Drafter> block_;  // a block drafter instead of the MTP head
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options) {
    Config cfg = parse_config(config_json);
    if (options.vision && config_json.contains("vision_config") && !config_json.at("vision_config").is_null()) {
        log::warn("qwen3_5: the vision tower is not supported yet; loading the text decoder only");
    }

    ModelInfo info;
    info.arch = "qwen3_5";
    info.id = cfg::model_id(dir);
    info.vocab_size = cfg.vocab_size;
    info.hidden_size = cfg.hidden_size;
    info.max_position = cfg.max_position;
    info.dialect = "qwen3_5";

    std::vector<int32_t> gen_eos;
    const bool has_generation_config = std::filesystem::exists(dir / "generation_config.json");
    info.default_sampling = cfg::read_generation_defaults(dir, &gen_eos);
    if (!has_generation_config) {
        // The checkpoint ships no generation_config.json; these are the model card's recommended settings.
        info.default_sampling.temperature = 1.0f;
        info.default_sampling.top_p = 0.95f;
        info.default_sampling.top_k = 20;
    }
    const nlohmann::json& text = config_json.contains("text_config") ? config_json.at("text_config") : config_json;
    std::vector<int32_t> stops = gen_eos;
    for (const nlohmann::json* j : {&config_json, &text}) {
        if (j->contains("eos_token_id")) {
            const auto ids = cfg::get_int_list(*j, "eos_token_id", "config.json");
            stops.insert(stops.end(), ids.begin(), ids.end());
        }
    }
    if (const auto eos = cfg::tokenizer_eos_id(dir)) stops.push_back(*eos);  // <|im_end|>: the end of a turn
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    XS_CHECK(!stops.empty(), "qwen3_5: no end-of-sequence token in config.json or tokenizer_config.json");
    info.stop_token_ids = std::move(stops);

    std::unique_ptr<SafetensorsWeights> checkpoint = SafetensorsWeights::open_dir(dir);
    join_expert_banks(*checkpoint, cfg);
    std::unique_ptr<WeightStore> weights = std::move(checkpoint);
    std::optional<dflash::Config> drafter;
    if (options.speculation && !options.drafter.empty()) {
        std::ifstream in(options.drafter / "config.json");
        XS_CHECK(in.good(), "qwen3_5: cannot read ", (options.drafter / "config.json").string());
        drafter = dflash::parse_config(nlohmann::json::parse(in));
        weights = std::make_unique<JoinedWeights>(std::move(weights), dflash::open_weights(options.drafter), kDrafterPrefix);
    }
    return std::make_unique<Qwen35Model>(std::move(cfg), std::move(info), std::move(weights), options, std::move(drafter));
}

}  // namespace xs::qwen3_5
