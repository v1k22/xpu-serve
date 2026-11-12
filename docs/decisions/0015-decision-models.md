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
- **The Qwen3.5 decoder's builders are shared** (`qwen3_5::Decoder`): the language model and the decision models
  built on that decoder make their programs from the same layer code.
- **Images are refused by name** (`assets`, `images`, box questions): the Qwen3.5 vision tower is not loaded yet.

## What was measured

**Clef-Flash against its own code** (`joint_schema_model.py`, float32 on the CPU; `tests/parity/decision/reference.py`),
on 13 requests chosen to break a prompt (32 questions: every type, nested JSON with floats and key orders, text in
four scripts, literal special tokens, 40 options, a single option, missing instructions, a prompt of 2,665 tokens):

| Checkpoint | Backend | Tokens | Largest probability difference | Top option |
|---|---|---|---|---|
| tiny (random, the released layout) | cpu-ref f32 | equal on 13 of 13 | 6e-7 | equal on 32 of 32 |
| tiny, prompt fed 7 tokens a step | cpu-ref f32 | equal | 6e-7 | equal |
| tiny | Arc Pro B60 f32 / f16 | equal | 6e-7 / 0.0010 | equal |
| `Cloudflare/clef-flash` | Arc Pro B60 f16 | equal on 13 of 13 | 0.0007 | equal on 32 of 32 |

**End to end through the server** (`tests/e2e/decisions25.json`: 25 questions with one accepted answer each, 12
choice, 8 noul, 5 score; `tests/e2e/systemone_eval.py`), Arc Pro B60, f16, 15.2 GiB on the device:

| Model | Accepted answer | Same answer as its reference | Largest probability difference | Latency p50 / p95 | Prompt |
|---|---|---|---|---|---|
| Clef-Flash | 25 of 25 | 25 of 25 | 0.00005 | 89 / 96 ms | 150 to 262 tokens |

Latency is the client's wall time for one request at a time; the server's own time is 2 ms less.

**Latency by prompt size** (`bench --state-tokens`; three questions with eight options), Clef-Flash, Arc Pro B60, f16:

| Prompt tokens | 320 | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,000 |
|---|---|---|---|---|---|---|---|
| Latency p50 | 118 ms | 165 ms | 323 ms | 644 ms | 1.31 s | 2.74 s | 5.86 s |

All but 3 ms is the decoder reading the prompt (about 3,000 tokens a second, the engine's prefill speed for this
decoder); the head's three programs take 1.5, 1.0 and 0.4 ms.

## Consequences

- A decision model needs no flags: `serve`, `run --request` and `bench` take its directory.
- A decision's latency on a device is the device's prefill time for the prompt; a faster prefill is the way to a
  faster decision. Requests run one at a time.
- Packs (`convert`) of decision checkpoints are not written yet: the two that have a head fit a 24 GB card at f16,
  and a prefill does not get faster from quantized weights on a discrete GPU.
- A new decision model is a `DecisionModel` and, when it has a new read-out, its own programs; the runtime, the
  server and the tests' harness stay as they are.
