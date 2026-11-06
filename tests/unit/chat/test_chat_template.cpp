#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "chat/chat_template.h"
#include "util/error.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

const fs::path kChatFixtures = fs::path(XS_SOURCE_DIR) / "tests/fixtures/chat";

json read_json(const fs::path& path) {
    std::ifstream in(path);
    REQUIRE_MESSAGE(in.good(), "missing fixture ", path.string());
    return json::parse(in);
}

// A scratch model dir holding only the given files; removed on scope exit.
class ScratchModelDir {
public:
    explicit ScratchModelDir(const std::string& name)
        : dir_(fs::temp_directory_path() / ("xs_chat_" + name + "_" + std::to_string(::getpid()))) {
        fs::remove_all(dir_);
        fs::create_directories(dir_);
    }
    ~ScratchModelDir() { fs::remove_all(dir_); }
    ScratchModelDir(const ScratchModelDir&) = delete;
    ScratchModelDir& operator=(const ScratchModelDir&) = delete;

    void write(const std::string& file, const std::string& text) const { std::ofstream(dir_ / file) << text; }
    const fs::path& path() const { return dir_; }

private:
    fs::path dir_;
};

std::string render_one(const std::string& tmpl, const json& messages, const json& extra = json::object()) {
    ScratchModelDir dir("render");
    dir.write("chat_template.jinja", tmpl);
    dir.write("tokenizer_config.json", R"({"bos_token": "<s>", "eos_token": {"content": "</s>"}})");
    return xs::chat::ChatTemplate::load(dir.path())->render(messages, nullptr, false, extra);
}

}  // namespace

TEST_SUITE("chat template") {
    TEST_CASE("gemma4 renders byte-identical to Hugging Face") {
        const auto tmpl = xs::chat::ChatTemplate::load(kChatFixtures / "gemma4");
        const json cases = read_json(kChatFixtures / "gemma4_cases.json");
        REQUIRE(cases.size() >= 20);
        for (const json& c : cases) {
            const std::string name = c["name"].get<std::string>();
            CAPTURE(name);
            std::string rendered;
            CHECK_NOTHROW(
                rendered = tmpl->render(c["messages"], c["tools"], c["add_generation_prompt"].get<bool>(), c["extra"]));
            CHECK(rendered == c["expected"].get<std::string>());
        }
    }

    TEST_CASE("special tokens come from tokenizer_config, as strings or AddedToken objects") {
        CHECK(render_one("{{ bos_token }}|{{ eos_token }}", json::array()) == "<s>|</s>");
    }

    TEST_CASE("extra variables override special tokens") {
        CHECK(render_one("{{ bos_token }}", json::array(), {{"bos_token", "B"}}) == "B");
    }

    TEST_CASE("falls back to tokenizer_config chat_template") {
        ScratchModelDir dir("fallback");
        dir.write("tokenizer_config.json", R"({"chat_template": "{{ messages[0]['content'] }}!"})");
        const auto tmpl = xs::chat::ChatTemplate::load(dir.path());
        CHECK(tmpl->render(json::array({{{"role", "user"}, {"content", "hi"}}}), nullptr, false) == "hi!");
    }

    TEST_CASE("dictsort is case-insensitive like Jinja2") {
        CHECK(render_one("{% for k, v in messages[0] | dictsort %}{{ k }}{% endfor %}",
                         json::array({{{"b", 1}, {"A", 2}, {"c", 3}, {"B2", 4}}})) == "AbB2c");
    }

    TEST_CASE("floats print like Python repr") {
        const json values = json::array({0.1 + 0.2, 1e16, 1e-5, 123456.5, -0.0, 2.0, 1e-4});
        CHECK(render_one("{% for v in messages %}{{ v }} {% endfor %}", values) ==
              "0.30000000000000004 1e+16 1e-05 123456.5 -0.0 2.0 0.0001 ");
    }

    TEST_CASE("map applies a named filter to every item") {
        CHECK(render_one("{{ messages | map('upper') | join(',') }}", json::array({"a", "b"})) == "A,B");
    }

    TEST_CASE("missing template is refused by name") {
        ScratchModelDir dir("missing");
        dir.write("tokenizer_config.json", "{}");
        CHECK_THROWS_WITH_AS(xs::chat::ChatTemplate::load(dir.path()), doctest::Contains("no chat template"),
                             xs::Error);
    }

    TEST_CASE("template syntax error surfaces the engine message") {
        ScratchModelDir dir("syntax");
        dir.write("chat_template.jinja", "{% if %}");
        CHECK_THROWS_WITH_AS(xs::chat::ChatTemplate::load(dir.path()), doctest::Contains("does not parse"), xs::Error);
    }

    TEST_CASE("raise_exception in the template becomes xs::Error with its message") {
        const auto tmpl = xs::chat::ChatTemplate::load(kChatFixtures / "gemma4");
        const json messages = json::array({
            {{"role", "user"}, {"content", "hi"}},
            {{"role", "assistant"},
             {"tool_calls",
              json::array(
                  {{{"id", "c"}, {"type", "function"}, {"function", {{"name", "f"}, {"arguments", "[1, 2]"}}}}})}},
        });
        CHECK_THROWS_WITH_AS(tmpl->render(messages, nullptr, true), doctest::Contains("must be a JSON object"),
                             xs::Error);
        CHECK_THROWS_WITH_AS(tmpl->render(messages, nullptr, true), doctest::Contains("at line 257, column"),
                             xs::Error);
    }

    TEST_CASE("OpenAI tool round trip: JSON-text arguments, no assistant content, tool_call_id only") {
        const auto tmpl = xs::chat::ChatTemplate::load(kChatFixtures / "gemma4");
        auto conversation = [](const json& arguments) {
            return json::array({
                {{"role", "user"}, {"content", "Weather in Paris?"}},
                {{"role", "assistant"},
                 {"tool_calls", json::array({{{"id", "c1"},
                                              {"type", "function"},
                                              {"function", {{"name", "get_weather"}, {"arguments", arguments}}}}})}},
                {{"role", "tool"}, {"tool_call_id", "c1"}, {"content", R"({"temp_c": 18})"}},
            });
        };
        const std::string from_text = tmpl->render(conversation(R"({"city": "Paris"})"), nullptr, true);
        CHECK(from_text == tmpl->render(conversation({{"city", "Paris"}}), nullptr, true));
        CHECK(from_text.find("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
                             "<|tool_response>response:get_weather{") != std::string::npos);
    }

    TEST_CASE("non-array messages are refused") {
        const auto tmpl = xs::chat::ChatTemplate::load(kChatFixtures / "gemma4");
        CHECK_THROWS_AS(tmpl->render(json::object(), nullptr, true), xs::Error);
    }
}
