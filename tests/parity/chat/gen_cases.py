"""Renders chat-template parity cases with Hugging Face for the C++ ChatTemplate test.

Usage: python -I tests/parity/chat/gen_cases.py gemma4=<model_dir> qwen3_5=<model_dir> lfm3=<model_dir> ...
Writes tests/fixtures/chat/<set>_cases.json for each set named. A model dir needs only chat_template.jinja and
tokenizer_config.json (plus tokenizer.json for AutoTokenizer): the Hugging Face checkpoint named in
tests/fixtures/chat/<set>/SOURCE, or for lfm3 (whose template is not kept here) LiquidAI/LFM3-1.7B.
"""

import copy
import itertools
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests/fixtures/chat"


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


TRANSLATE = tool(
    "traduire_texte",
    "Traduit un texte — 翻訳します 🌐. Use “smart quotes” and a \\ backslash.",
    {
        "texte": {"type": "string", "description": "Texte à traduire"},
        "langue": {"type": "string", "enum": ["fr", "日本語", "español"]},
        "options": {
            "type": "object",
            "properties": {
                "formel": {"type": "boolean", "default": False},
                "glossaire": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"de": {"type": "string"}, "vers": {"type": "string"}}},
                },
                "température": {"type": "number", "minimum": 0.0, "maximum": 1.5, "default": 0.7531246},
            },
        },
    },
    ["texte", "langue"],
)

QWEN_ARGS = {
    "s": "two\nlines <b>&</b> ünï 日本 🎉",
    "i": -12,
    "f": 2.5,
    "whole": 3.0,
    "precise": 0.123456789012,
    "large": 1234567.125,
    "tiny": 1e-05,
    "big": 1e20,
    "t": True,
    "no": False,
    "none": None,
    "obj": {"nested": {"deep": [1, "two", {"x": False}]}, "é": "ü"},
    "list": [[1, 2], {"k": None}],
    "empty_s": "",
    "empty_o": {},
    "empty_l": [],
}


def qwen3_5_cases():
    u = {"role": "user", "content": "What is the weather in Paris?"}
    sys_msg = {"role": "system", "content": "  You are a helpful assistant.\n"}
    weather_call = call("call_1", "get_weather", {"city": "Paris", "unit": "celsius"})
    weather_text_call = call("call_1", "get_weather", "{\"city\": \"Paris\", \"unit\": \"celsius\"}")
    think_off = {"enable_thinking": False}
    think_on = {"enable_thinking": True}

    yield "user_only", [u], None, True, {}
    yield "user_only_no_generation_prompt", [u], None, False, {}
    yield "system_and_user", [sys_msg, u], None, True, {}
    yield "system_and_user_no_generation_prompt", [sys_msg, u], None, False, {}
    yield "thinking_on", [sys_msg, u], None, True, think_on
    yield "thinking_off", [sys_msg, u], None, True, think_off
    yield "multi_turn", [
        sys_msg,
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ], None, True, {}
    yield "multi_turn_thinking_off", [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Again"},
    ], None, True, think_off
    yield "assistant_prefill", [u, {"role": "assistant", "content": "Let me check"}], None, False, {}
    yield "reasoning_content_after_last_user", [
        u, {"role": "assistant", "reasoning_content": "\nThe user wants weather.\n\n", "content": "Sunny."},
    ], None, False, {}
    yield "reasoning_content_before_last_user_dropped", [
        u,
        {"role": "assistant", "reasoning_content": "Old thoughts.", "content": "Sunny."},
        {"role": "user", "content": "And tomorrow?"},
    ], None, True, {}
    yield "think_in_content_after_last_user", [
        u, {"role": "assistant", "content": "<think>\nhmm, Paris\n\n</think>\n\nIt is sunny."},
    ], None, False, {}
    yield "think_in_content_before_last_user_stripped", [
        u,
        {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nIt is sunny."},
        {"role": "user", "content": "Thanks"},
    ], None, True, {}
    yield "think_close_only_in_content", [
        u,
        {"role": "assistant", "content": "thinking without opener</think>\n\nanswer"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "  leading spaces then</think>tail  "},
    ], None, False, {}
    yield "reasoning_content_wins_over_think_in_content", [
        u, {"role": "assistant", "reasoning_content": "Real reasoning.", "content": "<think>x</think>Visible."},
    ], None, False, {}
    yield "reasoning_content_null_with_think_in_content", [
        u, {"role": "assistant", "reasoning_content": None, "content": "<think>r</think>c"},
    ], None, False, {}
    yield "content_parts", [
        {"role": "system", "content": [{"type": "text", "text": "Part one."}, {"type": "text", "text": " Part two. "}]},
        {"role": "user", "content": [{"type": "text", "text": "  Look: "}, {"type": "text", "text": "two parts"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Seen."}]},
        {"role": "user", "content": [{"type": "text", "text": "Again"}]},
    ], None, True, {}
    yield "image_parts", [
        {"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "What is in this image?"}]},
    ], None, True, {}
    yield "image_parts_multi_turn", [
        {"role": "user", "content": [
            {"type": "text", "text": "Compare "}, {"type": "image"}, {"type": "text", "text": " and "},
            {"type": "image"}]},
        {"role": "assistant", "content": "They differ."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "And this one?"}]},
    ], None, True, think_off
    yield "image_parts_with_vision_ids", [
        {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Both?"}]},
    ], None, True, {"add_vision_id": True}
    yield "video_part", [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Summarize."}]},
    ], None, True, {}
    yield "unicode", [
        {"role": "system", "content": "Réponds en français. 日本語も可。"},
        {"role": "user", "content": "Ça va? 👋🏽 Привет — “quotes” \\ back\\slash \t tab"},
        {"role": "assistant", "content": "Très bien 😀"},
        {"role": "user", "content": "中文 العربية עברית"},
    ], None, True, {}
    yield "tools_simple", [u], [WEATHER], True, {}
    yield "tools_several_with_system", [sys_msg, u], [WEATHER, SEARCH, EDIT, TRANSLATE, NO_ARGS], True, {}
    yield "tools_empty_system", [{"role": "system", "content": "   "}, u], [WEATHER], True, {}
    yield "tools_thinking_off", [u], [WEATHER, TRANSLATE], True, think_off
    yield "tool_call_object_arguments", [
        u,
        {"role": "assistant", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"},
    ], [WEATHER], True, {}
    yield "tool_call_string_arguments", [
        u,
        {"role": "assistant", "content": None, "tool_calls": [weather_text_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"},
    ], [WEATHER], True, {}
    yield "tool_call_argument_types_object", [
        u, {"role": "assistant", "tool_calls": [call("c", "kitchen_sink", QWEN_ARGS)]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ], None, True, {}
    yield "tool_call_argument_types_string", [
        u, {"role": "assistant", "tool_calls": [call("c", "kitchen_sink", json.dumps(QWEN_ARGS))]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ], None, True, {}
    yield "tool_call_argument_floats", [
        u, {"role": "assistant", "tool_calls": [call("c", "nums", FLOATS)]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ], None, True, {}
    yield "tool_call_flat_without_function_key", [
        u, {"role": "assistant", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Oslo"}}]},
        {"role": "tool", "content": "-3 C"},
    ], [WEATHER], True, {}
    yield "tool_call_without_arguments", [
        u, {"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "list_files"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "a.txt"},
    ], [NO_ARGS], True, {}
    yield "tool_call_with_preamble_and_reasoning", [
        u,
        {"role": "assistant", "reasoning_content": "Need the tool.", "content": "Let me check.\n",
         "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
    ], [WEATHER], True, {}
    yield "tool_call_whitespace_only_content", [
        u, {"role": "assistant", "content": " \n ", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
    ], [WEATHER], True, {}
    yield "multiple_tool_calls_consecutive_responses", [
        sys_msg,
        {"role": "user", "content": "Weather in Paris, docs on rain, and translate it?"},
        {"role": "assistant", "content": "I will use three tools.", "tool_calls": [
            weather_call,
            call("call_2", "search_docs", json.dumps({
                "query": "rain \"gear\"", "Limit": 5, "exact": False,
                "filters": {"tags": ["a", "b"], "since": None, "score": 0.75}, "URL": "x.org"})),
            call("call_3", "traduire_texte", {"texte": "il pleut", "langue": "日本語"}),
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
        {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "Doc A"},
                                                               {"type": "text", "text": "; Doc B"}]},
        {"role": "tool", "tool_call_id": "call_3", "content": "雨が降っています"},
    ], [WEATHER, SEARCH, TRANSLATE], True, {}
    yield "multi_step_tools_keep_reasoning_after_last_user", [
        u,
        {"role": "assistant", "reasoning_content": "First the weather.", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
        {"role": "assistant", "content": "<think>\nNow the docs.\n</think>\n\n",
         "tool_calls": [call("call_2", "search_docs", {"query": "umbrella"})]},
        {"role": "tool", "tool_call_id": "call_2", "content": "Doc C"},
    ], [WEATHER, SEARCH], True, think_on
    yield "tool_round_trip_answered_then_new_user", [
        u,
        {"role": "assistant", "reasoning_content": "Calling.", "content": "", "tool_calls": [weather_text_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
        {"role": "assistant", "reasoning_content": "Answer now.", "content": "It is 21 C in Paris."},
        {"role": "user", "content": "Thanks!"},
    ], [WEATHER], True, think_off
    yield "user_message_wrapped_in_tool_response", [
        u,
        {"role": "assistant", "tool_calls": [weather_call]},
        {"role": "user", "content": "<tool_response>\n21 C\n</tool_response>"},
    ], [WEATHER], True, {}


def lfm3_cases():
    """LFM3 writes a turn like Qwen 3.5 (<think>, <tool_call> blocks of <function=...>) with its own rules: no newlines
    around the reasoning, tools as JSON lines in the system turn, tools loaded later (defer_loading, tool_reference,
    additional_tools), and consecutive results of one call merged into one turn."""
    u = {"role": "user", "content": "What is the weather in Paris?"}
    sys_msg = {"role": "system", "content": "  You are a helpful assistant.\n"}
    weather_call = call("call_1", "get_weather", {"city": "Paris", "unit": "celsius"})
    weather_text_call = call("call_1", "get_weather", "{\"city\": \"Paris\", \"unit\": \"celsius\"}")
    think_off = {"enable_thinking": False}
    deferred = copy.deepcopy(SEARCH)
    deferred["function"]["defer_loading"] = True
    deferred_outer = {**copy.deepcopy(EDIT), "defer_loading": True}

    yield "user_only", [u], None, True, {}
    yield "user_only_no_generation_prompt", [u], None, False, {}
    yield "system_and_user", [sys_msg, u], None, True, {}
    yield "empty_system", [{"role": "system", "content": ""}, u], None, True, {}
    yield "thinking_off", [sys_msg, u], None, True, think_off
    yield "multi_turn", [
        sys_msg,
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ], None, True, {}
    yield "multi_turn_thinking_off", [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Again"},
    ], None, True, think_off
    yield "assistant_prefill", [u, {"role": "assistant", "content": "Let me check"}], None, False, {}
    yield "reasoning_content_after_last_user", [
        u, {"role": "assistant", "reasoning_content": "The user wants weather.", "content": "Sunny."},
    ], None, False, {}
    yield "reasoning_and_thinking_keys", [
        u, {"role": "assistant", "thinking": "As thinking.", "content": "One."},
        {"role": "assistant", "reasoning": "As reasoning.", "content": "Two."},
    ], None, False, {}
    yield "reasoning_content_before_last_user_dropped", [
        u,
        {"role": "assistant", "reasoning_content": "Old thoughts.", "content": "Sunny."},
        {"role": "user", "content": "And tomorrow?"},
    ], None, True, {}
    yield "preserve_thinking", [
        u,
        {"role": "assistant", "reasoning_content": "Old thoughts.", "content": "Sunny."},
        {"role": "user", "content": "And tomorrow?"},
    ], None, True, {"preserve_thinking": True}
    yield "think_in_content_after_last_user", [
        u, {"role": "assistant", "content": "<think>hmm, Paris</think>It is sunny."},
    ], None, False, {}
    yield "think_in_content_before_last_user_stripped", [
        u,
        {"role": "assistant", "content": "<think>hmm</think>\n\n It is sunny. "},
        {"role": "user", "content": "Thanks"},
    ], None, True, {}
    yield "reasoning_content_null", [
        u, {"role": "assistant", "reasoning_content": None, "content": "Sunny."},
    ], None, False, {}
    yield "content_parts", [
        {"role": "system", "content": [{"type": "text", "text": "Part one."}, {"type": "text", "text": " Part two. "}]},
        {"role": "user", "content": [{"type": "text", "text": "  Look: "}, {"type": "text", "text": "two parts"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Seen."}]},
        {"role": "user", "content": [{"type": "text", "text": "Again"}, "bare string", {"type": "other", "n": 1}]},
    ], None, True, {}
    yield "image_and_video_parts", [
        {"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "What is in this image?"}, {"type": "video"}]},
    ], None, True, {}
    yield "unicode", [
        {"role": "system", "content": "Réponds en français. 日本語も可。"},
        {"role": "user", "content": "Ça va? 👋🏽 Привет — “quotes” \\ back\\slash \t tab"},
        {"role": "assistant", "content": "Très bien 😀"},
        {"role": "user", "content": "中文 العربية עברית"},
    ], None, True, {}
    yield "tools_simple", [u], [WEATHER], True, {}
    yield "tools_several_with_system", [sys_msg, u], [WEATHER, SEARCH, EDIT, TRANSLATE, NO_ARGS], True, {}
    yield "tools_thinking_off", [u], [WEATHER, TRANSLATE], True, think_off
    yield "tools_deferred_are_not_listed", [u], [WEATHER, deferred, deferred_outer], True, {}
    yield "tools_all_deferred", [sys_msg, u], [deferred], True, {}
    yield "tool_call_object_arguments", [
        u,
        {"role": "assistant", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"},
    ], [WEATHER], True, {}
    yield "tool_call_string_arguments", [
        u,
        {"role": "assistant", "content": None, "tool_calls": [weather_text_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"},
    ], [WEATHER], True, {}
    yield "tool_call_argument_types_object", [
        u, {"role": "assistant", "tool_calls": [call("c", "kitchen_sink", QWEN_ARGS)]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ], None, True, {}
    yield "tool_call_argument_types_string", [
        u, {"role": "assistant", "tool_calls": [call("c", "kitchen_sink", json.dumps(QWEN_ARGS))]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ], None, True, {}
    yield "tool_call_argument_floats", [
        u, {"role": "assistant", "tool_calls": [call("c", "nums", FLOATS)]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ], None, True, {}
    yield "tool_call_flat_without_function_key", [
        u, {"role": "assistant", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Oslo"}}]},
        {"role": "tool", "content": "-3 C"},
    ], [WEATHER], True, {}
    yield "tool_call_without_arguments", [
        u, {"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "list_files"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "a.txt"},
    ], [NO_ARGS], True, {}
    yield "tool_call_empty_string_arguments", [
        u, {"role": "assistant", "tool_calls": [call("c", "list_files", "{}")]},
        {"role": "tool", "tool_call_id": "c", "content": None},
    ], [NO_ARGS], True, {}
    yield "tool_call_with_preamble_and_reasoning", [
        u,
        {"role": "assistant", "reasoning_content": "Need the tool.", "content": "Let me check.\n",
         "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
    ], [WEATHER], True, {}
    yield "multiple_tool_calls_and_responses", [
        sys_msg,
        {"role": "user", "content": "Weather in Paris, docs on rain, and translate it?"},
        {"role": "assistant", "content": "I will use three tools.", "tool_calls": [
            weather_call,
            call("call_2", "search_docs", json.dumps({
                "query": "rain \"gear\"", "Limit": 5, "exact": False,
                "filters": {"tags": ["a", "b"], "since": None, "score": 0.75}, "URL": "x.org"})),
            call("call_3", "traduire_texte", {"texte": "il pleut", "langue": "日本語"}),
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
        {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "Doc A"},
                                                               {"type": "text", "text": "; Doc B"}]},
        {"role": "tool", "tool_call_id": "call_3", "content": {"type": "text", "text": "雨が降っています"}},
    ], [WEATHER, SEARCH, TRANSLATE], True, {}
    yield "tool_responses_of_one_call_merge", [
        u,
        {"role": "assistant", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "part one"},
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "part two"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": {"type": "text", "text": "part three"}},
        {"role": "tool", "tool_call_id": "call_9", "content": "another call"},
    ], [WEATHER], True, {}
    yield "tool_response_already_wrapped", [
        u,
        {"role": "assistant", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "<tool_response>\n21 C\n</tool_response>"},
    ], [WEATHER], True, {}
    yield "tool_response_names_deferred_tools", [
        u,
        {"role": "assistant", "tool_calls": [call("s", "tool_search", {"query": "docs"})]},
        {"role": "tool", "tool_call_id": "s", "content": [
            {"type": "text", "text": "Found:"},
            {"type": "tool_reference", "name": "search_docs"},
            {"type": "tool_reference", "name": "apply_edits"},
            {"type": "text", "text": "Use them."},
            {"type": "tool_reference", "name": "get_weather"},
        ]},
    ], [WEATHER, deferred, deferred_outer], True, {}
    yield "tool_response_result_tools", [
        u,
        {"role": "assistant", "tool_calls": [call("s", "tool_search", {"query": "docs"})]},
        {"role": "tool", "tool_call_id": "s", "content": [{"result": {"tools": [WEATHER, NO_ARGS]}}]},
        {"role": "assistant", "tool_calls": [call("t", "tool_search", {"query": "more"})]},
        {"role": "tool", "tool_call_id": "t", "content": [{"result": {"tools": "{\"name\": \"raw\"}"}}]},
    ], [deferred], True, {}
    yield "additional_tools_message", [
        u,
        {"role": "assistant", "tool_calls": [call("s", "tool_search", {"query": "docs"})]},
        {"role": "tool", "tool_call_id": "s", "additional_tools": [WEATHER, NO_ARGS], "content": "ignored"},
        {"role": "assistant", "tool_calls": [call("t", "tool_search", {"query": "more"})]},
        {"role": "tool", "tool_call_id": "t", "additional_tools": True,
         "content": "<additional_tools>\n{\"name\": \"raw\"}\n</additional_tools>"},
    ], [deferred], True, {}
    yield "multi_step_tools_keep_reasoning_after_last_user", [
        u,
        {"role": "assistant", "reasoning_content": "First the weather.", "tool_calls": [weather_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
        {"role": "assistant", "content": "<think>Now the docs.</think>",
         "tool_calls": [call("call_2", "search_docs", {"query": "umbrella"})]},
        {"role": "tool", "tool_call_id": "call_2", "content": "Doc C"},
    ], [WEATHER, SEARCH], True, {}
    yield "tool_round_trip_answered_then_new_user", [
        u,
        {"role": "assistant", "reasoning_content": "Calling.", "content": "", "tool_calls": [weather_text_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "21 C"},
        {"role": "assistant", "reasoning_content": "Answer now.", "content": "It is 21 C in Paris."},
        {"role": "user", "content": "Thanks!"},
    ], [WEATHER], True, think_off


SETS = {
    "gemma4": lambda: itertools.chain(cases(), openai_cases()),
    "qwen3_5": qwen3_5_cases,
    "lfm3": lfm3_cases,
}


def write_set(name, model):
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False)  # a tokenizer needs no checkpoint code
    out = []
    for case_name, messages, tools, add_gen, extra in SETS[name]():
        expected = tok.apply_chat_template(
            openai_to_hf(messages), tools=tools, add_generation_prompt=add_gen, tokenize=False, **extra)
        out.append({"name": case_name, "messages": messages, "tools": tools, "add_generation_prompt": add_gen,
                    "extra": extra, "expected": expected})
    names = [c["name"] for c in out]
    assert len(names) == len(set(names)), "duplicate case names"
    path = FIXTURES / f"{name}_cases.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} cases to {path}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        name, _, model = arg.partition("=")
        if name not in SETS or not model:
            sys.exit(f"expected <set>=<model_dir> with a set in {', '.join(SETS)}, got {arg!r}")
        write_set(name, Path(model))


if __name__ == "__main__":
    main()
