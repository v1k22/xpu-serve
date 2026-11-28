# 0036: vLLM's structured outputs, and grammars shared between requests

Status: **accepted** · 2026-11-28

## Context

Decision 0033 held an answer to OpenAI's `response_format` (any JSON object, or a JSON Schema). vLLM, whose API many
clients target, takes more through `structured_outputs`: a regular expression, a choice among strings, a grammar, a
schema; it removed its older `guided_*` fields (since v0.12 they are accepted and do nothing). Anthropic's API has
JSON outputs (`output_config.format`). And each request built its grammar's automaton anew: the first token of every
string walked the vocabulary's trie again, though agents send the same schema turn after turn.

## Decision

- `structured_outputs` on chat and text completions, exactly one of: `json` (a schema, or its text), `regex`,
  `choice`, `grammar`, `json_object`, as vLLM reads them. Its `structural_tag` and its options
  (`disable_any_whitespace`, `disable_additional_properties`, `whitespace_pattern`, which vLLM ignores per request)
  are refused by name, and so is a request giving both it and `response_format`. The `guided_*` fields are refused by
  name, pointing to `structured_outputs`, where vLLM silently ignores them.
- **Regular expressions** (`grammar/text_grammars`): the whole answer matches, as xgrammar's: alternation, groups
  (capturing, `(?:`, named), `* + ? {n} {n,} {n,m}` (lazy forms the same), classes with ranges and negation, `.` (any
  code point, a newline too), `\d \w \s` and their negations (ECMAScript's), `\n \t \xHH \uHHHH \u{...}`; `^` and `$`
  change nothing. Lookaround, backreferences, `\b`, `\p{...}`, inline flags and an empty class are refused with their
  offset. Classes are sets of code points, compiled into UTF-8 byte sequences (`code_points`).
- **Grammars**: EBNF in xgrammar's dialect, which takes GBNF's: `name ::= body` (a name first on its line), `root`,
  `#` comments, double-quoted literals with C and JSON escapes, classes (`[^]` any), groups, `|`, `? * + {n} {n,}
  {n,m}`. A grammar without `::=` is Lark's (vLLM's test) and refused; xgrammar's extensions (rule options, lookahead,
  macros) are refused by name; errors name their line and column. Left recursion is refused (0033's matcher does not
  take it; xgrammar does).
- **Choices**: the answer is exactly one of the strings.
- **Anthropic**: `output_config.format` `{"type": "json_schema", "schema"}` holds the answer as `response_format`
  does (in a text block), `output_config.effort` reaches the template as `reasoning_effort`; the beta `output_format`
  is refused by name, as Anthropic's API does without its beta header.
- **Automata shared**: every format and tool-call grammar has a name (its kind and text: the schema's JSON, the
  pattern, the grammar, a tool grammar's tools and mode), and the server keeps the automata of the last 32 by name
  (`AutomatonCache`). Requests held to the same grammar share its states and the masks those keep; constraints start
  on the engine thread, the only one that reads them.

## Measured

Arc Pro B60, Qwen3.5-4B, greedy, thinking off: a choice of three answers in 1 token, an EBNF arithmetic expression,
a regular expression (the answer of the pattern given), a schema of three fields. The same schema's second request
spent 2.4 ms on its constraint over 33 rows where the first spent 11.7 over 35 (the first's trie walks kept); a third
through `response_format` with the same schema and one through Anthropic's `output_config` shared it too (2.4 and 1.8
ms).

## Consequences

- vLLM's whitespace options and structural tags are refused rather than ignored.
- An EBNF grammar xgrammar takes with left recursion must be rewritten right-recursive here.
- The cache holds 32 automata; one of a long schema's walks keeps up to 64 masks of the vocabulary (a few MB).
