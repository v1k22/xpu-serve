"""Renders chat-template parity cases with Hugging Face for the C++ ChatTemplate test.

Usage: /root/workspace/xs-ref/venv/bin/python tests/parity/chat/gen_cases.py [model_dir]
Writes tests/fixtures/chat/gemma4_cases.json. The model dir needs only chat_template.jinja and tokenizer_config.json
(plus tokenizer.json for AutoTokenizer); it defaults to the vendored fixture copy's source model.
"""

import copy
import itertools
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tests/fixtures/chat/gemma4_cases.json"
DEFAULT_MODEL = Path("/root/workspace/models/gemma-4-E2B-it")


def tool(name, description, properties, required=None, **extra):
    params = {"type": "object", "properties": properties}
    if required is not None:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params, **extra}}


WEATHER = tool(
    "get_weather",
    "Current weather for a city.",
    {
        "city": {"type": "string", "description": "City name, e.g. Paris"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
    },
    ["city"],
)

SEARCH = tool(
    "search_docs",
    "Search the documentation.",
    {
        "query": {"type": "string"},
        "Limit": {"type": "integer", "description": "Max results"},
        "filters": {
            "type": "object",
            "description": "Optional filters",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "since": {"type": "string", "nullable": True},
                "score": {"type": "number"},
            },
            "required": ["tags"],
        },
        "URL": {"type": "string", "description": "Restrict to this site"},
        "exact": {"type": "boolean"},
    },
    ["query"],
)

EDIT = tool(
    "apply_edits",
    "Apply a list of edits to a file.",
    {
        "path": {"type": "string"},
        "edits": {
            "type": "array",
            "description": "Edits in order",
            "items": {
                "type": "object",
                "properties": {"old": {"type": "string"}, "new": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["old", "new"],
            },
        },
        "mode": {"type": "array", "items": {"type": ["string", "null"], "enum": ["fast", "safe"]}},
    },
    ["path", "edits"],
)

NO_ARGS = {"type": "function", "function": {"name": "list_files", "description": "List the files."}}

WITH_RESPONSE = tool(
    "get_time",
    "Current time.",
    {"tz": {"type": "string"}},
    response={"description": "ISO time", "type": "object"},
)

FLOATS = {"a": 3.0, "b": 0.1, "c": 1e-05, "d": 1e20, "e": -2.5, "f": 12345678901234, "g": 0, "h": 2.5e-7}


def call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def cases():
    u = {"role": "user", "content": "What is the weather in Paris?"}
    sys_msg = {"role": "system", "content": "  You are a helpful assistant.\n"}
    weather_call = call("call_1", "get_weather", {"city": "Paris", "unit": "celsius"})
    yield "user_only", [u], None, True, {}
    yield "user_only_no_generation_prompt", [u], None, False, {}
    yield "system_and_user", [sys_msg, u], None, True, {}
    yield "developer_and_user", [{"role": "developer", "content": "Be terse."}, u], None, True, {}
    yield "multi_turn", [
        sys_msg,
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ], None, True, {}
    yield "assistant_prefill", [u, {"role": "assistant", "content": "Let me check"}], None, False, {}
    yield "assistant_content_strips_thinking", [
        u,
        {"role": "assistant", "content": "<|channel>thought\nhmm<channel|>It is sunny."},
        {"role": "user", "content": "Thanks"},
    ], None, True, {}
    yield "thinking_on", [u], None, True, {"enable_thinking": True}
    yield "thinking_off", [u], None, True, {"enable_thinking": False}
    yield "thinking_on_with_system", [sys_msg, u], None, True, {"enable_thinking": True}
    yield "reasoning_content_after_last_user", [
        u,
        {"role": "assistant", "reasoning_content": "The user wants weather.", "content": "Sunny."},
    ], None, False, {"enable_thinking": True}
    yield "reasoning_dropped_before_last_user", [
        u,
        {"role": "assistant", "reasoning_content": "Old thoughts.", "content": "Sunny."},
        {"role": "user", "content": "And tomorrow?"},
    ], None, True, {"enable_thinking": True}
    yield "content_parts", [
        {"role": "system", "content": [{"type": "text", "text": "Part one."}, {"type": "text", "text": " Part two. "}]},
        {"role": "user", "content": [{"type": "text", "text": "  Look: "}, {"type": "text", "text": "two parts"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Seen."}]},
        {"role": "user", "content": [{"type": "text", "text": "Again"}]},
    ], None, True, {}
    yield "unicode", [
        {"role": "system", "content": "Réponds en français. 日本語も可。"},
        {"role": "user", "content": "Ça va? 👋🏽 Привет — “quotes” \\ back\\slash \t tab"},
        {"role": "assistant", "content": "Très bien 😀"},
        {"role": "user", "content": "中文 العربية עברית"},
    ], None, True, {}
    yield "tools_simple", [u], [WEATHER], True, {}
    yield "tools_nested_enum_array_optional", [sys_msg, u], [WEATHER, SEARCH, EDIT], True, {}
    yield "tools_no_parameters_and_response", [u], [NO_ARGS, WITH_RESPONSE], True, {}
    yield "tools_with_thinking", [u], [WEATHER], True, {"enable_thinking": True}
    yield "tool_call_then_response", [
        u,
        {"role": "assistant", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"},
    ], [WEATHER], True, {}
    yield "tool_call_then_response_thinking", [
        u,
        {"role": "assistant", "reasoning_content": "Need the tool.", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
    ], [WEATHER], True, {"enable_thinking": True}
    yield "tool_call_then_response_then_answer", [
        u,
        {"role": "assistant", "content": "", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
        {"role": "assistant", "content": "It is 21 C in Paris."},
        {"role": "user", "content": "Thanks!"},
    ], [WEATHER], True, {}
    yield "multiple_tool_calls", [
        {"role": "user", "content": "Weather in Paris and docs on rain?"},
        {"role": "assistant", "tool_calls": [
            weather_call,
            call("call_2", "search_docs", {
                "query": "rain \"gear\"", "Limit": 5, "exact": False,
                "filters": {"tags": ["a", "b"], "since": None, "score": 0.75}, "URL": "x.org"}),
            call("call_3", "list_files", {}),
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
        {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "Doc A"},
                                                               {"type": "text", "text": "; Doc B"}]},
        {"role": "tool", "tool_call_id": "call_3", "content": "a.txt\nb.txt"},
    ], [WEATHER, SEARCH, NO_ARGS], True, {}
    yield "tool_call_argument_numbers", [
        u, {"role": "assistant", "tool_calls": [call("c", "nums", FLOATS)]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ], None, True, {}
    yield "tool_call_unicode_arguments", [
        u, {"role": "assistant", "tool_calls": [call("c", "write", {"text": "naïve 日本 🎉\n<b>x</b>", "list": [[1, 2], {"k": True}]})]},
        {"role": "tool", "name": "write", "content": "done ✓"},
    ], None, False, {}
    yield "legacy_tool_responses_mapping", [
        u,
        {"role": "assistant", "tool_calls": [weather_call],
         "tool_responses": [{"name": "get_weather", "response": {"temp": 21, "sky": "clear", "Wind": [3, 4.5]}}]},
        {"role": "assistant", "content": "Clear and 21 degrees."},
    ], [WEATHER], True, {}
    yield "preserve_thinking", [
        u,
        {"role": "assistant", "reasoning_content": "Calling the tool.", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
        {"role": "assistant", "content": "21 C."},
        {"role": "user", "content": "ok"},
    ], [WEATHER], True, {"enable_thinking": True, "preserve_thinking": True}


def openai_to_hf(messages):
    """Mirror of ChatTemplate::render's normalization: tool_calls[].function.arguments given as JSON text (OpenAI)
    become the object HF templates expect when the text parses to a JSON object; everything else is unchanged."""
    out = copy.deepcopy(messages)
    for message in out:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {})
            if isinstance(function.get("arguments"), str):
                try:
                    parsed = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    function["arguments"] = parsed
    return out


def openai_cases():
    user = {"role": "user", "content": "What is the weather in Paris right now? Use the tool."}
    text_call = {"id": "c1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
    tool_reply = {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 18, \"sky\": \"cloudy\"}"}
    yield "openai_round_trip", [user, {"role": "assistant", "tool_calls": [text_call]}, tool_reply], [WEATHER], True, {}
    yield "openai_round_trip_null_content", [
        user, {"role": "assistant", "content": None, "tool_calls": [text_call]}, tool_reply,
    ], [WEATHER], True, {}
    yield "openai_round_trip_thinking", [
        user, {"role": "assistant", "tool_calls": [text_call]}, tool_reply,
    ], [WEATHER], True, {"enable_thinking": True}
    yield "openai_round_trip_answered", [
        user, {"role": "assistant", "tool_calls": [text_call]}, tool_reply,
        {"role": "assistant", "content": "It is 18 C and cloudy."}, {"role": "user", "content": "Thanks"},
    ], [WEATHER], True, {}


def main():
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    tok = AutoTokenizer.from_pretrained(model)
    out = []
    for name, messages, tools, add_gen, extra in itertools.chain(cases(), openai_cases()):
        expected = tok.apply_chat_template(
            openai_to_hf(messages), tools=tools, add_generation_prompt=add_gen, tokenize=False, **extra)
        out.append({"name": name, "messages": messages, "tools": tools, "add_generation_prompt": add_gen,
                    "extra": extra, "expected": expected})
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} cases to {OUT}")


if __name__ == "__main__":
    main()
