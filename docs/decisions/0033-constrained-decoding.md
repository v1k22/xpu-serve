# 0033: Constrained decoding: a byte grammar, masks from a trie of the vocabulary

Status: **accepted**, done · 2026-11-27

## Context

OpenAI's `response_format` asks for an answer that is JSON (`json_object`) or the JSON a schema describes
(`json_schema`, strict structured outputs). The answer must parse, so the output is held to the format token by token:
each step, the tokens that cannot continue a document of the format are masked out of the row's logits before it is
sampled. vLLM does this with xgrammar or llguidance, llama.cpp with its GBNF grammars; both compile a JSON Schema into
a grammar, and vLLM starts the constraint after a reasoning parser's end-of-thought marker.

Three things here shape it. Tokens are byte strings that need not end on a character (byte-level BPE, byte
fallback). Qwen3.5's vocabulary has 248,077 tokens, and a decode step of Qwen3.5-4B on the Arc Pro B60 takes some
12 ms from a q4 file (24 ms from the bf16 checkpoint): a mask may cost a few ms at most. And greedy decoding reads
each row's argmax from the device, speculative rounds verify several drafts in one step, and several requests decode
as one step (decisions 0027, 0028): a constraint must fit all three.

## Options

- **A. Our own byte grammar; masks from one walk of a trie of the vocabulary**, over a DFA built lazily from the
  matcher's configurations, with the masks of long walks kept. Small (a grammar, a JSON Schema compiler, a matcher,
  a trie), in the repository's own terms, and every refusal by name.
- **B. xgrammar or llguidance**, what vLLM uses: masks in microseconds from tokens classified ahead of time per
  grammar position. A C++ library with its own build and Python bindings, or a Rust one; neither is a header to
  vendor, and their schema coverage and refusals would be theirs.
- **C. Masks on the device**: the mask's bits uploaded and applied in the step, argmax kept on the device. Saves the
  copy of a row's logits, but the mask is still made on the host, and the step waits for it.

## Decision

A. It meets the budget without a library (below), and what it costs is in one place to make faster.

- **A grammar library of its own** (`src/grammar/`, xs_grammar), which knows nothing of the engine or the server. A
  grammar is rules over bytes: alternatives of byte sets and rule references, written as expressions and lowered
  into rules. Bytes make byte-level tokens and partial UTF-8 no special case. `finish()` drops alternatives that can
  never end (a recursion with no way out), refuses a root that cannot, and refuses left recursion.
- **JSON Schema compiled to it**, in OpenAI's strict subset: type and lists of types, nullable, object properties in
  their order with required and additionalProperties false, items, enum, const, anyOf, `$ref` into the schema
  (`#`, `#/$defs/...`, recursive ones included) and the annotations that constrain nothing. Every other keyword is
  refused by its path (`response_format.json_schema.schema.properties.age.minimum`), never ignored. Two readings are
  ours: an object with properties takes no other keys even where additionalProperties is absent (what OpenAI's strict
  mode asks for anyway), and an object without properties or additionalProperties is any object. `json_object` is any
  JSON object. Whitespace is bounded wherever it may go (at most two newlines, then 20 spaces or tabs), so a model
  cannot loop on it; a document may start with such whitespace (Qwen writes `\n\n` after `</think>`) and ends with its
  value; strings are valid UTF-8 (no surrogates, no overlong forms).
- **The matcher**: a pushdown recognizer whose configurations (sets of stacks, frames interned) are states, each
  state's step by a byte computed once and kept in a table: a DFA built lazily over the configurations reached. A
  rule's last item is a tail position, so a repetition (the characters of a string) loops through one state.
- **Masks from a trie** of the vocabulary's byte strings, built once per model: one depth-first walk per state, a
  subtree skipped at the first byte the state does not take, each step a table lookup. The tokens of a state whose
  walk was long are kept: inside a string every character returns to the same state, so a string's later tokens cost
  nothing. Added tokens (markers: `<|im_end|>`, and Qwen's `<think>` and `<tool_call>`, which are not special) are never
  written by a grammar, so a JSON string cannot end the turn or open a tool call through one.
- **The engine stays grammar-agnostic**: a request carries an `OutputConstraint`, which starts a `TokenConstraint`
  per generation (`engine/constraint.h`): mask a row, accept a token, complete, and how many of a chain of tokens it
  allows without committing. A constrained request reads its rows' logits instead of their argmax (in a batched step
  every row then does, as for a sampled request beside greedy ones). Each row is masked in the state the tokens before
  it leave: the token chosen is accepted at once, so in a speculative round row i is masked after drafts 0..i-1 (a
  draft is accepted only when the masked row takes it). The stop tokens are the engine's: kept only where the
  constraint is complete, so the output cannot end before it parses. Drafts from the first the constraint refuses are
  cut before the verify, which a masked row would never accept: the same round in fewer rows, and rejection sampling
  stays exact (the masked distribution gives them no weight).
- **Reasoning first**: where the rendered prompt leaves the model reasoning (Qwen3.5 to 3.8 with thinking on), the
  answer starts after the dialect's end-of-reasoning marker, as vLLM does; the dialect's parser already knows the
  prompt's state, and names the marker (`OutputParser::reasoning_end`; `chat::answer_start`). The thought is free but
  for the stop tokens (the answer is still to come); the marker is found in the output's bytes, whether the model
  writes it as one token or several. A dialect whose model writes a header before its answer (Inkling 2's text
  message) has it forced as tokens first. The server (`server/grammar_constraint`) puts the three together; it builds
  the vocabulary at the first request with a format.
- **API**: `/v1/chat/completions` and `/v1/completions` take `json_object` and `json_schema` (`name`, `schema`,
  `strict`, `description`); `text` as before. Beside `tools` it is refused for now (an answer is either held to the
  format or free to call a tool). The content is parsed as usual; `timings` carry the rows masked and the time it
  took.

## Measured

Arc Pro B60, Qwen3.5-4B, greedy; an extraction prompt over a nested schema (a string, an integer, an object of two
strings, an array of objects with a nullable integer, an array of objects with an enum, a boolean; all required,
additionalProperties false) and a paragraph to extract from.

| | free | held to the schema |
|---|---|---|
| bf16 checkpoint, thinking off | 42.5 tok/s; the right JSON inside a ```json fence: the reply does not parse | 41.5 tok/s (-2.4 %); the same JSON without the fence, 245 tokens, of the schema |
| the same, `--spec mtp` (3 drafts) | 121 tok/s | 113 tok/s (-7 %); the tokens without a drafter, 2.77 of 3 drafts kept a verify (2.82 free) |
| bf16, thinking on (4,482 tokens, most of them thought) | 42.2 tok/s; bare JSON after `</think>` | 41.5 tok/s; the free reply, byte for byte |
| the same, `--spec mtp` | 100.1 tok/s | 95.7 tok/s (-4.4 %); the same bytes |
| unsloth's Q4_K_M file, thinking off | 84.5 to 85.0 tok/s (11.8 ms a step); fenced | 80.5 to 80.9 tok/s (-4.5 %, 0.55 ms a token); of the schema |

The constraint's own time, from the responses' `timings`: 45 to 57 ms over the answer's 246 rows (0.18 to 0.23 ms a
row, masking the 248,320 logits of the row included); over a thought's rows about 1 us each (only the stop tokens are
masked). The rest of what a constrained token costs is its row's logits copied from the device (1 MB) and its
argmax taken on the host: about 0.35 ms a token, four times that for a verify of four rows.

The grammar's part alone (`test_token_mask.cpp` with `XS_QWEN35_DIR`: a 65-token answer to a four-field schema, a
new automaton as a request has, the trie of 591,553 nodes built in 59 to 82 ms): on the B60 machine's Xeon
Platinum 8468V, 0.051 ms a step on average, 6 us the median, 2.7 ms the worst; on a Xeon Gold 6252, 0.06 ms, 9 us
and 2.8 to 2.9 ms. The worst step is the first token inside a string: its walk visits nearly
every node (about 5 ns each, a chain of dependent loads); each later token of every string in that place reads the
kept mask. A walk that sets its tokens through a loop of zero or one turns took 3.1 ms; walking four parts of the trie
in turn, or keeping the parent's state in a register, took 4.3.

## Consequences

- A constrained step copies the row's logits to the host (about 1 MB over Qwen3.5's vocabulary) and masks it there;
  the drafter is not constrained, only its drafts cut.
- Gemma 4 and Inkling 2, whose models open their thoughts themselves rather than the prompt leaving them inside one,
  answer at once under a format: the grammar starts at the first token.
- The dialect's parser reads its markers in the answer's text: a JSON string that spells `<think>` in plain tokens is
  read as one. Rare, and the same as unconstrained output.
- Not done: other grammars at the API (regular expressions, GBNF), the response_format of Anthropic's API, formats
  beside tools, and sharing one schema's automaton between requests (each request pays its first walks again).
