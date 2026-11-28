# 0034: Prompt lookup for any model, and drafts per round chosen by measured cost

Status: **accepted** · 2026-11-28

## Context

Speculative decoding needed a drafter trained for the model: an MTP head, an assistant, a block drafter. LFM3, the
GGUF files published without a head and every model without a drafter decoded a token a step. Much of what a server
is asked to write repeats its input: a function returned with a change, a JSON document with a field added, sentences
quoted from a text. vLLM, SGLang and llama.cpp all draft such output from the text itself (prompt lookup, n-gram
speculation).

The drafts a round asks for (`--spec-k`) were fixed per model, from six chat prompts. The best count depends on the
text as much as on the model: a copy task accepts seven drafts in a row, a story two, and a chain drafter's later
drafts cost a run each while a block drafter's cost nothing more.

## Decision

- **`--spec ngram`**: drafts copied from the sequence (prompt and output, the pending token last): the longest suffix
  of `--ngram-min` to `--ngram-max` tokens (both 5 by default; either alone sets both, vLLM's
  `prompt_lookup_min` / `prompt_lookup_max`) that occurred before, at its **first** occurrence, and up to `--spec-k`
  (7 by default) of the tokens that followed it there, never past the end. No drafts when nothing matches: the round is
  a plain decode step. vLLM's GPU speculator takes the latest occurrence and pads a round to k drafts with the last
  token (its verify has a fixed width); its earlier CPU proposer took the first, as here. On the copy prompts below the
  first occurrence decodes 0.5 to 2 % faster than the latest on all five models (it starts the longer copies).
- **The verify is the model's batch graph** (`Model::batch_graph`: logits of every row, segments of several
  sequences): every language model with one can speculate, alone or with requests in flight (their rounds verified as
  one step). The session's programs are the step graph, which prefills, and the batch graph; the engine's prefill and
  verify programs are no longer fixed indices (`spec::Program` stays the order of a model's drafter's).
- **`--spec-adaptive`**: each round asks for the count, 1 to `--spec-k`, with the most tokens a millisecond. Tokens: 1 +
  a1 + a1 a2 + ..., ai the request's acceptance of its i-th draft (counts decayed by 0.9 a round; a draft no recent
  round reached is taken as the one before it). Milliseconds: what rounds of that count cost on this device (drafter
  and verify, averaged; each count's first round not counted, since a new step size pays its setup), between the counts
  measured on their line. A request starts from the acceptance of the requests before it. Off by default: a model's
  recipe gives its fixed count.

## Measured

Arc Pro B60, greedy, `bench --prompts` of up to 512 new tokens: six copy-heavy prompts (a Python module returned with
type hints or a class renamed, a C++ fix, a JSON array with a field added, sentences quoted, a table to CSV) and six
chat prompts. Decode tok/s:

| Model | copy: no drafts → ngram | chat: no drafts → ngram | the model's drafter, fixed k: copy / chat |
|---|---|---|---|
| Qwen3.5-4B, unsloth Q4_K_M | 82.5 → 211.3 | 87.8 → 87.7 | – (the file has no MTP head) |
| Qwen3.5-4B, bf16 | 42.0 → 112.2 | 43.0 → 43.5 | MTP k = 3: 124.3 / 80.2 |
| Gemma 4 E2B, unsloth Q4_K_M | 88.3 → 189.2 | 102.1 → 106.1 | assistant k = 1: 123.4 / 120.9 |
| LFM3-1.7B, bf16 | 97.1 → 201.9 | 98.7 → 98.4 | – (none exists) |
| Qwen3.8-27B, unsloth Q4_0 | 23.1 → 72.3 | 23.3 → 24.1 | MTP k = 4: 82.2 / 51.9 |

A verify of up to 8 rows costs about one row's on the B60, so a round without a match costs what a decode step does:
chat output that repeats little loses nothing (within 1 % either way). Copy tasks decode 2.1 to 3.1 times faster.

Adaptive counts with the model's drafter (up to 7; Gemma's assistant up to 4), against fixed counts:

| Model | copy: recipe k / k = max / adaptive | chat: recipe k / k = max / adaptive |
|---|---|---|
| Qwen3.5-4B bf16, MTP | 124.3 / 165.2 / 169.2 | 80.2 / 68.5 / 83.9 |
| Gemma 4 E2B Q4_K_M, assistant | 123.4 / 206.5 / 204.2 | 120.9 / 126.4 / 134.4 |
| Qwen3.8-27B Q4_0, MTP | 82.2 / 109.3 / 110.3 | 51.9 / 50.7 / 54.0 |

No fixed count is best on both kinds of text: copying wants the most drafts, chat with a chain drafter (a run a draft)
three or fewer. The adaptive count is within 1 % of the better fixed count or above it on each, on chat 5 to 11 %
above the recipes' counts (it also drops to fewer drafts within a reply where the text gets harder). With prompt lookup it changes nothing
measurable (the drafts are as many as the text offers, and cost nothing to make).

## Consequences

- Any language model with a batch graph speculates with `--spec ngram`; the decision models and Qwen3-ASR, which have
  none, refuse it by name.
- A request copying a prefix from another sequence (decision 0029) hands its drafter the copied tokens
  (`Drafter::adopt`), so lookups see the shared system prompt and conversation.
- `--spec-adaptive` learns costs from rounds a request runs alone; a batched round takes the smaller of its count and
  its share of the step's rows.
- Not done: n-gram drafts beside a model's own drafter (the better of the two per round), and the drafts of several
  sequences in one run of a model's drafter (TODO).
