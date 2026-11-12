# 0015: Decision models and the System One API

Status: **accepted** · 2026-11-12

## Context

A decision model answers typed questions about a state instead of writing text. A request holds a `state` (a string
or any JSON) and named `questions`, each a `noul` (yes / no), a `choice` among named options or a `score` on ordered
levels; the answer is a probability for every allowed option of every question, read off the model in a forward pass
with no generation and nothing to parse. The API is TypeSafe's System One (`POST /v1/systemone`, OpenAPI 0.2.0;
their hosted model is Jev), and open models now speak it: `Cloudflare/clef-flash`, `perplexity-ai/pplx-decider-flash-v1.2`,
`openjev/openjev`. They share the API and little else:

| | Clef-Flash | pplx-decider-flash | OpenJev |
|---|---|---|---|
| Decoder | Qwen3.5-9B, causal | Qwen3.5-4B, full-attention layers see the whole prompt | Qwen3.8-27B, a language model |
| Prompt | its own layout, one per request | chat template, one per request | chat template, one per question |
| Read out | a transformer head over the final hidden states | a 255-row linear layer at a marker after each question | the output layer's rows of the option letters |

vLLM serves the endpoint (its `structured_decisions` entry points) and Clef-Flash through a pooling model class;
OpenJev ships a helper that asks vLLM for one token's log-probabilities per question.

## Decision

- **A decision model is a third kind of model**, beside language models and robot policies: `DecisionModel`
  (`src/models/decision.h`), found by what sits beside the decoder's `config.json` (`model_kind`). It generates
  nothing, so it is not a `Model` with an unused decode loop.
- **The model says what to run, the runtime runs it.** `DecisionModel::start` lays out a request's prompt and returns
  a `DecisionRun`; the runtime (`src/decision/`) asks it for a step (a program of the session, its inputs, the
  outputs wanted), runs it and hands the outputs back, until there is none. The three read-outs above differ in how
  many programs and steps they need; none of that reaches the runtime, the server or the backends. Model code still
  never touches a backend, and tokenizing and the chat template reach it through `PromptTools`.
- **The API layer knows nothing of models.** A model gives probabilities per option in the request's order; the
  answer (`choice`, expected `score`, `confidence`) is made from them once (`decision/answers.cpp`), with TypeSafe's
  published confidence formulas for every model. Clef's own Python helper reports the top probability as confidence
  instead; here an API client gets the same definition whatever model answers.
- **The endpoint is the API's, bodies included**: 422 `{"detail": [{"loc", "msg", "type"}]}` for a body that does
  not follow the schema, 400 for one that is not JSON or too long (`max_tokens_exceeded`). Probabilities are not
  rounded. Timings travel in a `Server-Timing` header, so the body stays the specified one.
- **Prompts are the reference code's, token for token.** A model was trained on what its Python wrote, down to
  `json.dumps`'s separators and `repr`'s floats, so that text is reproduced (`models/common/py_text.h`), not
  approximated with another JSON writer.
- **Nothing is truncated.** Clef's reference cuts a long state to fit; here a request over the limit is refused by
  name, as the API specifies.
- **Clef's head runs as programs of the decoder's session.** The prompt is encoded once (in chunks if long): the
  encode program ends in the head's view of every token and appends, for each of the head's attentions over the
  sequence, the keys and values to a KV slot. Three small programs follow, with one row per option or per question
  (the symbolic step dimension): the options' evidence layers, the questions' decoder layers, the scorer; they read
  those slots with `KvRead`. Between them the host does what has a per-request shape: span means, each question's
  softmax over its own options, cosines. The head needed one new op, the exact GELU.
- **pplx-decider reads its whole prompt in one step.** Its full-attention layers see every token of the prompt,
  the later ones too (the Gated DeltaNet layers stay causal), so the decoder's builders take the way a
  full-attention layer sees the sequence: through a KV slot, or over the step. The program has no KV slots; its
  read-out (a 255-row matmul) runs on every row and the host takes the rows at the markers, divides by the
  checkpoint's temperature and applies the softmax over a question's own options. A request over the limit is split
  by question as the reference splits it, each part with the whole state; text that spells a special token loses its
  angle brackets as in the reference, which leaves the chat markers alone, as that code does.
- **OpenJev is a language model behind the same interface.** `letter_readout::make` wraps any `Model` as a
  `DecisionModel`: its step graph is the one program, a question is a prompt through the chat template, and the
  answer is the logits of the option letters in the last row (a softmax over the letters at a temperature is the
  same whether it starts from logits or from log-probabilities, so nothing of the rest of the vocabulary is
  needed). The prompt, the letters, the yes / no recalibration and the passes for more than 52 options are the
  helper's (`helper/shim.py`); its measured settings are a preset by name. `--readout openjev|letters` asks for it,
  since nothing in such a checkpoint says it is more than a language model.
- **A session can undo its last step whole.** Questions about one state share the start of their prompts, but a
  Qwen3.5 decoder's recurrent state cannot be cut back to a prefix. The state before a step is already kept (the
  buffer the step did not write, which makes a failed step harmless), so `Session::truncate` now also takes the
  token count before the last step, whatever that step's size and without `rewind_tokens`. The letter read-out
  feeds the shared part, then each question's own part as one step, and undoes it before the next; the model keeps
  a record of the prompt the session holds, so a request about the state of the one before starts there too.
- **The Qwen3.5 decoder's builders are shared** (`qwen3_5::Decoder`): the language model and the decision models
  built on that decoder make their programs from the same layer code.
- **Images are refused by name** (`assets`, `images`, box questions): the Qwen3.5 vision tower is not loaded yet.

## What was measured

**Each model against its own code** (Clef: `joint_schema_model.py`; pplx-decider: its `pplx_decider` package;
OpenJev: its helper, with the one call it makes to a server answered by transformers; float32 on the CPU;
`tests/parity/decision/reference.py`), on 14 requests chosen to break a prompt (33 questions: every type, nested JSON
with floats and key orders, text in four scripts, literal special tokens, 40 and 60 options, a single option, missing
and empty instructions, a prompt of about 2,600 tokens). OpenJev's helper refuses a question without instructions,
which leaves it 11 requests and 24 questions, a prompt each:

| Checkpoint | Backend | Tokens | Largest probability difference | Top option |
|---|---|---|---|---|
| tiny Clef (random, the released layout) | cpu-ref f32 | equal on 14 of 14 | 6e-7 | equal on 33 of 33 |
| tiny Clef, prompt fed 7 tokens a step | Arc Pro B60 f16 | equal | 0.0006 | equal |
| `Cloudflare/clef-flash` | Arc Pro B60 f16 | equal on 14 of 14 | 0.0007 | equal on 33 of 33 |
| tiny pplx-decider | cpu-ref f32 | equal on 14 of 14 | 1.6e-6 | equal on 33 of 33 |
| tiny pplx-decider | Arc Pro B60 f16 | equal | 0.0052 | equal |
| `perplexity-ai/pplx-decider-flash-v1.2` | Arc Pro B60 f16 | equal on 14 of 14 | 0.0009 | equal on 33 of 33 |
| tiny language model, letter read-out | cpu-ref f32 | equal on 11 of 11 | 1.5e-6 | equal on 24 of 24 |
| tiny language model, state fed 64 tokens a step | Arc Pro B60 f16 | equal | 0.0024 | equal |

The first run of the pplx-decider prompt failed this test on one request: an empty `instructions` string must fall
back to the model's default wording, as Python's `or` does.

**End to end through the server** (`tests/e2e/decisions25.json`: 25 questions with one accepted answer each, 12
choice, 8 noul, 5 score; `tests/e2e/systemone_eval.py`), Arc Pro B60; Clef-Flash and pplx-decider at f16, OpenJev
as a q4 pack (g32, round to nearest) against the float32 answers of the checkpoint it was made from:

| Model | On the device | Accepted answer | Same answer as its reference | Largest probability difference | Latency p50 / p95 | The same state again, p50 | Prompt |
|---|---|---|---|---|---|---|---|
| Clef-Flash | 15.2 GiB | 25 of 25 | 25 of 25 | 0.00005 | 89 / 109 ms | 89 ms | 150 to 262 tokens |
| pplx-decider-flash-v1.2 | 7.8 GiB | 25 of 25 | 25 of 25 | 0.00006 | 39 / 57 ms | 39 ms | 92 to 230 tokens |
| OpenJev, q4 | 16.2 GiB | 25 of 25 | 25 of 25 | 0.0095 | 266 / 478 ms | 143 ms | 67 to 198 tokens |

Latency is the client's wall time for one request at a time, a request that follows another state; the server's own
time is about 2 ms less. "The same state again" is the request sent once more: only the letter read-out keeps a
state, and reads just the question's part. Its times fall in two groups, about 140 ms and about 345 ms: a question's
part of up to 64 tokens runs the quantized matmuls on the few-rows path, a longer one on the path that dequantizes
each weight for a GEMM (`few_rows`, at most 64).

**Latency by prompt size** (`bench --state-tokens`; three questions with eight options; a new state every run),
Arc Pro B60, p50. OpenJev reads its three questions as three prompts of that size, sharing the state:

| Prompt tokens | sample | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,000 |
|---|---|---|---|---|---|---|---|
| Clef-Flash | 123 ms (325 tokens) | 165 ms | 323 ms | 643 ms | 1.34 s | 2.74 s | 5.89 s |
| pplx-decider-flash-v1.2 | 46 ms (170 tokens) | 104 ms | 214 ms | 456 ms | 1.07 s | 2.53 s | 7.31 s |
| OpenJev (q4), a new state | 538 ms (84 tokens) | 1.03 s | 1.46 s | 2.54 s | 4.75 s | 9.47 s | 19.6 s |
| OpenJev (q4), the same state again | 429 ms | 434 ms | 434 ms | 440 ms | 455 ms | 475 ms | 523 ms |

For Clef-Flash all but 3 ms is the decoder reading the prompt (about 3,000 tokens a second, the engine's prefill
speed for this decoder); the head's three programs take 1.5, 1.0 and 0.4 ms. pplx-decider's time grows faster than
its prompt: about 4,800 tokens a second up to 1,024 tokens, 2,200 at 16,000, one step holding the whole prompt.
OpenJev's q4 pack reads a new state at 900 to 950 tokens a second (the engine's prefill speed for the 27B at q4) and
then answers each question in about 145 ms; when the request before read the state, three questions take 0.43 to
0.52 s whatever its length, up to 16,000 tokens.

## Consequences

- A decision model needs no flags: `serve`, `run --request` and `bench` take its directory. A language model read
  at the option letters needs `--readout`, and on a 24 GB card OpenJev needs a q4 pack first (`convert`, as for any
  language model). Its weights are licensed CC BY-NC 4.0, unlike the two Apache-2.0 flash models.
- A decision's latency on a device is the device's prefill time for the prompt; a faster prefill is the way to a
  faster decision. Requests run one at a time.
- Packs (`convert`) of decision checkpoints are not written yet: the two that have a head fit a 24 GB card at f16,
  and a prefill does not get faster from quantized weights on a discrete GPU.
- pplx-decider's long prompts are the place to gain: its single step is slower per token past a few thousand
  tokens, and nothing of a state can be kept for the next request.
- The letter read-out keeps the state of the last request, so an agent that asks question after question about one
  page pays for the page once. Clef-Flash's decoder is causal and could keep a state the same way; it does not yet.
- `Session::truncate` to the start of the last step is a contract of both backends now, tested on its own.
- A new decision model is a `DecisionModel` and, when it has a new read-out, its own programs; the runtime, the
  server and the tests' harness stay as they are.
