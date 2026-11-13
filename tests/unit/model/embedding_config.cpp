// Embedding models' configs: BERT's and NomicBERT's config.json, the sentence-transformers files around them, and
// every variant they refuse by name.

#include <doctest/doctest.h>

#include <functional>
#include <string>

#include "convert/convert.h"
#include "embedding_test_util.h"
#include "models/bert/bert.h"
#include "models/common/sentence_transformers.h"
#include "models/embedding.h"
#include "models/model.h"
#include "models/nomic_bert/nomic_bert.h"
#include "models/policy.h"
#include "util/error.h"

namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using xs::test::TempDir;
using xs::test::TinyEncoder;

// The xs::Error message `f` throws ("" when it does not).
std::string error_of(const std::function<void()>& f) {
    try {
        f();
    } catch (const xs::Error& e) {
        return e.what();
    }
    return "";
}

void write_json(const fs::path& path, const json& value) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << value.dump(1);
}

json modules(std::initializer_list<const char*> kinds) {
    json out = json::array();
    for (const char* kind : kinds) {
        const std::string k = kind;
        const std::string path = k == "Transformer" ? "" : std::to_string(out.size()) + "_" + k;
        out.push_back({{"idx", out.size()}, {"name", std::to_string(out.size())}, {"path", path},
                       {"type", "sentence_transformers.models." + k}});
    }
    return out;
}

}  // namespace

TEST_SUITE("embedding config") {
    TEST_CASE("bert: the fields the encoder reads, and refusals by name") {
        const json c = xs::test::tiny_encoder_config(TinyEncoder{});
        const xs::bert::Config cfg = xs::bert::parse_config(c);
        CHECK(cfg.hidden_size == 64);
        CHECK(cfg.num_layers == 2);
        CHECK(cfg.num_heads == 2);
        CHECK(cfg.head_dim == 32);
        CHECK(cfg.max_positions == 64);
        CHECK(cfg.layer_norm_eps == doctest::Approx(1e-12));
        const std::pair<json, const char*> refused[] = {
            {{{"hidden_act", "gelu_new"}}, "hidden_act 'gelu_new'"},
            {{{"position_embedding_type", "relative_key"}}, "position_embedding_type 'relative_key'"},
            {{{"is_decoder", true}}, "is_decoder"},
            {{{"add_cross_attention", true}}, "add_cross_attention"},
            {{{"type_vocab_size", 0}}, "type_vocab_size 0"},
            {{{"num_attention_heads", 5}}, "not a multiple of num_attention_heads 5"},
        };
        for (const auto& [patch, what] : refused) {
            const std::string refusal = what;
            CAPTURE(refusal);
            json bad = c;
            bad.merge_patch(patch);
            CHECK(error_of([&] { xs::bert::parse_config(bad); }).find(refusal) != std::string::npos);
        }
    }

    TEST_CASE("nomic_bert: the fields the encoder reads; NomicBertConfig's defaults and every refusal by name") {
        TinyEncoder m;
        m.arch = "nomic_bert";
        const json c = xs::test::tiny_encoder_config(m);
        const xs::nomic_bert::Config cfg = xs::nomic_bert::parse_config(c);
        CHECK(cfg.hidden_size == 64);
        CHECK(cfg.num_layers == 2);
        CHECK(cfg.num_heads == 2);
        CHECK(cfg.head_dim == 32);
        CHECK(cfg.max_positions == 2048);
        CHECK(cfg.rope_theta == doctest::Approx(1000));
        CHECK(cfg.layer_norm_eps == doctest::Approx(1e-12));
        json defaults = c;
        defaults.erase("layer_norm_epsilon");
        defaults.erase("rotary_emb_base");
        defaults.erase("max_position_embeddings");
        const xs::nomic_bert::Config d = xs::nomic_bert::parse_config(defaults);
        CHECK(d.layer_norm_eps == doctest::Approx(1e-5));  // GPT2Config's
        CHECK(d.rope_theta == doctest::Approx(10000));
        CHECK(d.max_positions == 8192);  // n_positions
        const std::pair<json, const char*> refused[] = {
            {{{"activation_function", "gelu"}}, "activation_function 'gelu'"},
            {{{"activation_function", nullptr}}, "activation_function 'gelu_new'"},
            {{{"prenorm", true}}, "prenorm"},
            {{{"parallel_block", true}}, "parallel_block"},
            {{{"use_rms_norm", true}}, "use_rms_norm"},
            {{{"causal", true}}, "causal"},
            {{{"rotary_emb_interleaved", true}}, "rotary_emb_interleaved"},
            {{{"norm_mlp", true}}, "norm_mlp"},
            {{{"rotary_head_dim", true}}, "rotary_head_dim"},
            {{{"qkv_proj_bias", true}}, "qkv_proj_bias"},
            {{{"qkv_proj_bias", nullptr}}, "qkv_proj_bias"},  // NomicBertConfig's default has the bias
            {{{"mlp_fc1_bias", true}}, "mlp_fc1_bias"},
            {{{"mlp_fc2_bias", true}}, "mlp_fc2_bias"},
            {{{"rotary_scaling_factor", 2.0}}, "rotary_scaling_factor"},
            {{{"rotary_emb_scale_base", 512}}, "rotary_emb_scale_base"},
            {{{"moe_every_n_layers", 2}}, "mixture of experts"},
            {{{"rotary_emb_fraction", 0.5}}, "rotary_emb_fraction 0.5"},
            {{{"rotary_emb_fraction", nullptr}}, "rotary_emb_fraction 0"},  // the default: a position table
            {{{"num_heads_kv", 1}}, "num_heads_kv"},
            {{{"type_vocab_size", 0}}, "type_vocab_size 0"},
        };
        for (const auto& [patch, what] : refused) {
            const std::string refusal = what;
            CAPTURE(refusal);
            json bad = c;
            bad.merge_patch(patch);
            CHECK(error_of([&] { xs::nomic_bert::parse_config(bad); }).find(refusal) != std::string::npos);
        }
    }

    TEST_CASE("sentence-transformers: the pooling mode in both spellings, Normalize, the longest input") {
        const TempDir dir("xs_st_pipeline");
        xs::test::write_sentence_transformers_files(dir.path, TinyEncoder{});
        xs::st::Pipeline p = xs::st::read_pipeline(dir.path);
        CHECK(p.pooling == xs::Pooling::Mean);
        CHECK(p.dimension == 64);
        CHECK(!p.normalize);
        CHECK(p.max_seq_length == 48);
        CHECK(xs::st::has_pooling(dir.path));

        const fs::path pooling = dir.path / "1_Pooling" / "config.json";
        write_json(pooling, {{"word_embedding_dimension", 64}, {"pooling_mode_cls_token", true}});
        CHECK(xs::st::read_pipeline(dir.path).pooling == xs::Pooling::Cls);
        write_json(pooling, {{"word_embedding_dimension", 64}, {"pooling_mode_mean_tokens", false}});
        CHECK(xs::st::read_pipeline(dir.path).pooling == xs::Pooling::Mean);  // no flag set: sentence-transformers' mean
        write_json(pooling, {{"embedding_dimension", 32}, {"pooling_mode", "cls"}, {"include_prompt", true}});
        p = xs::st::read_pipeline(dir.path);
        CHECK(p.pooling == xs::Pooling::Cls);
        CHECK(p.dimension == 32);
        write_json(pooling, {{"embedding_dimension", 64}, {"pooling_mode", {"mean"}}});
        CHECK(xs::st::read_pipeline(dir.path).pooling == xs::Pooling::Mean);

        write_json(dir.path / "modules.json", modules({"Transformer", "Pooling", "Normalize"}));
        CHECK(xs::st::read_pipeline(dir.path).normalize);  // with or without a config of its own
        write_json(dir.path / "2_Normalize" / "config.json", {{"module_input_name", "sentence_embedding"}});
        CHECK(xs::st::read_pipeline(dir.path).normalize);

        fs::remove(dir.path / "sentence_bert_config.json");
        CHECK(!xs::st::read_pipeline(dir.path).max_seq_length.has_value());
        write_json(dir.path / "sentence_bert_config.json", {{"max_seq_length", nullptr}, {"do_lower_case", false}});
        CHECK(!xs::st::read_pipeline(dir.path).max_seq_length.has_value());

        // do_lower_case: sentence-transformers lowercases unless the tokenizer does; here the tokenizer must.
        write_json(dir.path / "sentence_bert_config.json", {{"max_seq_length", 48}, {"do_lower_case", true}});
        write_json(dir.path / "tokenizer.json",
                   {{"normalizer", {{"type", "BertNormalizer"}, {"lowercase", true}, {"clean_text", true}}}});
        CHECK(xs::st::read_pipeline(dir.path).max_seq_length == 48);
        write_json(dir.path / "tokenizer.json",
                   {{"normalizer", {{"type", "Sequence"}, {"normalizers", {{{"type", "NFC"}}, {{"type", "Lowercase"}}}}}}});
        CHECK_NOTHROW(xs::st::read_pipeline(dir.path));
        write_json(dir.path / "tokenizer.json", {{"normalizer", {{"type", "BertNormalizer"}, {"lowercase", false}}}});
        CHECK(error_of([&] { xs::st::read_pipeline(dir.path); }).find("do_lower_case") != std::string::npos);
    }

    TEST_CASE("sentence-transformers: other pipelines, modes and settings are refused by name") {
        const TempDir dir("xs_st_refused");
        const auto reset = [&] {
            fs::remove_all(dir.path);
            fs::create_directories(dir.path);
            xs::test::write_sentence_transformers_files(dir.path, TinyEncoder{});
        };
        struct Case {
            std::function<void()> edit;
            const char* what;
        };
        const fs::path pooling = dir.path / "1_Pooling" / "config.json";
        const Case cases[] = {
            {[&] { write_json(pooling, {{"word_embedding_dimension", 64}, {"pooling_mode_max_tokens", true}}); },
             "pooling mode 'max'"},
            {[&] { write_json(pooling, {{"word_embedding_dimension", 64}, {"pooling_mode_mean_sqrt_len_tokens", true}}); },
             "pooling mode 'mean_sqrt_len_tokens'"},
            {[&] { write_json(pooling, {{"word_embedding_dimension", 64}, {"pooling_mode_weightedmean_tokens", true}}); },
             "pooling mode 'weightedmean'"},
            {[&] { write_json(pooling, {{"embedding_dimension", 64}, {"pooling_mode", "lasttoken"}}); },
             "pooling mode 'lasttoken'"},
            {[&] {
                 write_json(pooling, {{"word_embedding_dimension", 64}, {"pooling_mode_cls_token", true},
                                      {"pooling_mode_mean_tokens", true}});
             },
             "several pooling modes concatenated (cls, mean)"},
            {[&] { write_json(dir.path / "modules.json", modules({"Transformer", "Pooling", "Dense", "Normalize"})); },
             "the modules Transformer, Pooling, Dense, Normalize are not supported"},
            {[&] { write_json(dir.path / "modules.json", modules({"Transformer", "Normalize", "Pooling"})); },
             "the modules Transformer, Normalize, Pooling"},
            {[&] { write_json(dir.path / "modules.json", modules({"Transformer", "Pooling", "LayerNorm"})); },
             "the modules Transformer, Pooling, LayerNorm"},
            {[&] {
                 json m = modules({"Transformer", "Pooling"});
                 m[0]["type"] = "custom_st.Transformer";
                 write_json(dir.path / "modules.json", m);
             },
             "'custom_st.Transformer' is the checkpoint's own code"},
            {[&] {
                 json m = modules({"Transformer", "Pooling"});
                 m[0]["path"] = "0_Transformer";
                 write_json(dir.path / "modules.json", m);
             },
             "a folder of its own ('0_Transformer')"},
            {[&] {
                 json m = modules({"Transformer", "Pooling"});
                 m[1]["kwargs"] = {"task"};
                 write_json(dir.path / "modules.json", m);
             },
             "the Pooling module's kwargs"},
            {[&] {
                 write_json(dir.path / "modules.json", modules({"Transformer", "Pooling", "Normalize"}));
                 write_json(dir.path / "2_Normalize" / "config.json", {{"module_input_name", "token_embeddings"}});
             },
             "a Normalize module of 'token_embeddings'"},
            {[&] { write_json(dir.path / "sentence_bert_config.json", {{"transformer_task", "text-generation"}}); },
             "transformer_task 'text-generation'"},
            {[&] { write_json(dir.path / "sentence_bert_config.json", {{"max_seq_length", 0}}); },
             "max_seq_length must be positive"},
            {[&] { write_json(dir.path / "sentence_bert_config.json", {{"module_output_name", "cls_embedding"}}); },
             "module_output_name 'cls_embedding'"},
            {[&] { write_json(dir.path / "sentence_bert_config.json", {{"tokenizer_name_or_path", "other/model"}}); },
             "tokenizer_name_or_path is not supported"},
            {[&] {
                 write_json(dir.path / "sentence_bert_config.json",
                            {{"processor_kwargs", {{"add_prefix_space", true}}}});
             },
             "processor_kwargs is not supported"},
            {[&] { write_json(dir.path / "sentence_bert_config.json", {{"config_kwargs", {{"hidden_act", "relu"}}}}); },
             "config_kwargs is not supported"},
            {[&] { write_json(dir.path / "sentence_bert_config.json", {{"query_length", 32}}); },
             "query_length is not supported"},
            {[&] {
                 write_json(dir.path / "config_sentence_transformers.json",
                            {{"prompts", {{"query", "search_query: "}}}, {"default_prompt_name", "query"}});
             },
             "a default prompt ('query')"},
            {[&] { write_json(dir.path / "config_sentence_transformers.json", {{"model_type", "SparseEncoder"}}); },
             "model_type 'SparseEncoder'"},
        };
        for (const Case& c : cases) {
            const std::string refusal = c.what;
            CAPTURE(refusal);
            reset();
            c.edit();
            CHECK(error_of([&] { xs::st::read_pipeline(dir.path); }).find(c.what) != std::string::npos);
        }
        reset();  // what sentence-transformers 6 writes by default changes nothing
        write_json(dir.path / "sentence_bert_config.json",
                   {{"max_seq_length", 48}, {"transformer_task", "feature-extraction"},
                    {"module_output_name", "token_embeddings"}, {"processor_kwargs", json::object()},
                    {"query_length", nullptr}});
        CHECK_NOTHROW(xs::st::read_pipeline(dir.path));
        reset();  // prompts without a default are the client's to send, in the text
        write_json(dir.path / "config_sentence_transformers.json",
                   {{"prompts", {{"query", "search_query: "}}}, {"default_prompt_name", nullptr}});
        CHECK_NOTHROW(xs::st::read_pipeline(dir.path));
    }

    TEST_CASE("an embedding model is told by its Pooling module, and refused where it is not one") {
        const TempDir dir("xs_embedding_kind");
        for (const char* arch : {"bert", "nomic_bert"}) {
            CAPTURE(arch);
            fs::remove_all(dir.path);
            TinyEncoder m;
            m.arch = arch;
            xs::test::write_tiny_encoder(dir.path, m, 3);
            CHECK(xs::model_kind(dir.path) == xs::ModelKind::Embedding);
            CHECK(xs::load_embedding_model(dir.path)->spec().arch == arch);
            // A convert of one is refused by name.
            xs::ConvertOptions o;
            o.source = dir.path;
            o.output = dir.path / "pack";
            CHECK(error_of([&] { xs::convert_model(o); }).find("is an embedding model") != std::string::npos);
            // Without the sentence-transformers modules nothing says how to pool: not an embedding model.
            fs::remove(dir.path / "modules.json");
            CHECK(xs::model_kind(dir.path) == xs::ModelKind::Language);
            CHECK(error_of([&] { xs::load_model(dir.path, {}); }).find("is served as an embedding model") !=
                  std::string::npos);
        }
        // A Pooling module beside an encoder this build does not serve.
        fs::remove_all(dir.path);
        xs::test::write_tiny_encoder(dir.path, TinyEncoder{}, 3);
        json config = xs::test::tiny_encoder_config(TinyEncoder{});
        config["model_type"] = "roberta";
        write_json(dir.path / "config.json", config);
        CHECK(xs::model_kind(dir.path) == xs::ModelKind::Language);
        CHECK(error_of([&] { xs::load_embedding_model(dir.path); })
                  .find("model_type 'roberta' is not an embedding model this build serves (supported: bert, "
                        "nomic_bert)") != std::string::npos);
        // A Pooling module of another width than the encoder's.
        fs::remove_all(dir.path);
        xs::test::write_tiny_encoder(dir.path, TinyEncoder{}, 3);
        write_json(dir.path / "1_Pooling" / "config.json", {{"word_embedding_dimension", 32}});
        CHECK(error_of([&] { xs::load_embedding_model(dir.path); })
                  .find("embedding dimension 32 is not the encoder's hidden size 64") != std::string::npos);
    }

    TEST_CASE("the longest input: max_seq_length, never more than BERT's position table") {
        const TempDir dir("xs_embedding_limit");
        const auto max_tokens = [&](const TinyEncoder& m) {
            fs::remove_all(dir.path);
            xs::test::write_tiny_encoder(dir.path, m, 4);
            return xs::load_embedding_model(dir.path)->spec().max_tokens;
        };
        TinyEncoder bert;
        CHECK(max_tokens(bert) == 48);
        bert.max_seq_length = 1000;
        CHECK(max_tokens(bert) == 64);  // the table's rows
        bert.max_seq_length = 0;
        CHECK(max_tokens(bert) == 64);
        TinyEncoder nomic;
        nomic.arch = "nomic_bert";
        nomic.max_seq_length = 8192;
        CHECK(max_tokens(nomic) == 8192);  // rotary: no table, past max_trained_positions too
        nomic.max_seq_length = 0;
        CHECK(max_tokens(nomic) == 2048);  // max_position_embeddings
    }
}
