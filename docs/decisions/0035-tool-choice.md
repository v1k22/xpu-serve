# 0035: Forced tool calls in the model's own syntax

Status: **accepted** · 2026-11-28

## Context

`tool_choice` took `auto` and `none`; `required` and a named function were refused, and so was a `response_format`
beside `tools`. Clients of OpenAI's API force calls (agents that must act, extraction through a function), and
the OpenAI Agents SDK marks its tools `strict`. Constrained decoding (decision 0033) can hold a reply to a grammar, but
a model writes calls in its own syntax, with marker tokens (`<tool_call>`, `<|tool_call>`, `<|"|>`) that a grammar
over bytes could not name.

vLLM (main, 2026-11) has two ways:

- **JSON path** (legacy): the reply held to a JSON schema built from the tools (an array of `{name, parameters}` for
  `required`; the parameters for a named function), and the output parsed as that JSON, not as the model's calls.
- **Structural tags** (its default for the parsers that have them, Qwen 3.5's included): a grammar in the model's own
  call syntax through xgrammar, the output read by the model's tool parser as for `auto`. A strict tool's arguments
  follow its schema, a non-strict one's are any. `parallel_tool_calls: false` stops after the first call. With `auto`
  and strict tools, a call is held from its opener (triggered tags).

## Decision

The model's own syntax, as vLLM's structural tags: the model writes calls as it was trained to, and the dialect's
parser reads them as it reads free ones.

- Grammars gain marker tokens (`grammar::token`), text up to a stop (`text_without`), JSON values inside a larger
  grammar (`JsonValues`) and embedded grammars (`Grammar::embed`).
- `server/tool_grammar` builds, per dialect: qwen3_5 (`<tool_call>\n<function=NAME>\n` then `<parameter=K>\nV\n
  </parameter>\n` per argument, a string's value as text, others as JSON), qwen3 (`<tool_call>\n{"name": ..., "arguments":
  {...}}\n</tool_call>`), gemma4 (`<|tool_call>call:NAME{k:<|"|>v<|"|>,...}<tool_call|>`), inkling2 (a message of its
  own, its body `{"name": ..., "args": {...}}`). `plain` refuses forced calls by name.
- `required`: one or more calls, nothing else (`parallel_tool_calls: false`: one); a named function: one call of it,
  `finish_reason` `stop` (OpenAI's). The grammar starts where the dialect's answer starts (after the reasoning a prompt
  leaves open, 0033).
- `auto` beside a `response_format`: the calls or the formatted answer (vLLM's rule when its tags are on). A format
  beside a forced call is refused by name (vLLM drops it with a warning).
- `auto` with strict tools: free, but once the model writes the call's opener (a marker token), the call is held to its
  closer (`OpenerConstraint`).
- Resolution as vLLM's: with tools, `auto` by default; without, `none`; anything else without tools refused. A tool
  needs `function.name`; `tool_choice` naming no tool is refused. `tool_choice: none` still drops the tools from the
  prompt (vLLM keeps them unless `--exclude-tools-when-tool-choice-none`).
- Anthropic's Messages API maps `any` to `required`, `tool` to a named function, `disable_parallel_tool_use` to one
  call, and passes a tool's `strict` on.

## Measured

Arc Pro B60, greedy, two tools (a strict `get_weather(city, unit)` with an enum, a `search_web(query)`):

- Qwen3.5-4B: `required` with thinking off and on (the call after 169 characters of thought), a named `search_web`
  (`finish_reason` stop), `auto`, `auto` beside a json_schema, two calls in parallel and one with
  `parallel_tool_calls: false`, streamed, and Anthropic's `any`: every call well formed, its arguments of the schema.
  The grammar's masks cost 0.3 to 0.5 ms a row (12 to 25 ms a reply).
- Gemma 4 E2B (unsloth Q4_K_M): the same requests, the same calls; 0.17 ms a row.

## Consequences

- A call's arguments in the XML and Gemma syntaxes follow a strict schema property by property; Gemma 4's check a
  string, an enum of strings and the JSON scalars, anything else being any value.
- Gemma 4 and Inkling 2 open their thoughts themselves, so under a forced choice they call at once.
- With strict tools under `auto`, every row reads its logits rather than its argmax (the constraint watches for an
  opener): decoding costs what a constrained one does.
- Not done: the `allowed_tools` choice (Responses API), strict tools' schemas beyond 0033's subset.
