# 0027: Several sequences, interleaved step by step

Status: **accepted** for requests in flight on sequences of their own; batching their steps into one is still to come
· 2026-11-22

## Context

A session held one sequence, and the engine ran one request from its first prefill chunk to its last token before the
next one started. A second user waited behind the first whole request: behind a 90,000-token prompt, minutes. And
the prefix kept between requests was the last request's alone, so two conversations taking turns each prefilled from
the start every time.

DESIGN.md section 9 plans paged KV blocks, a scheduler that batches every decoding sequence into one step, and a
prefix cache of hashed blocks. Batching changes the step graph's contract (a run advances one sequence: every
stateful op, the KV cache's append, attention, the convolutions and the gated delta rule, would read the rows of
several), every backend's kernels for those ops, and every model's inputs. It is the larger half, and the half that
only pays when several requests are in flight; this decision is the half that makes them be in flight.

## Decision

- **A session holds several sequences** (`SessionOptions::sequences`). Each has state slots of its own (KV, conv and
  recurrent state, the same shapes, in buffers of its own); `Session::use(s)` makes one current, and runs, truncate
  and reset act on it. The programs, their plans and their step memory are shared: one step runs at a time. Swapping
  the current sequence moves the slots' bookkeeping, not their memory.
- **The engine runs requests as generations advanced a step at a time** (`Engine::start`, `advance`, `finish`): a
  prefill chunk (the first token with the last one), a decode token, or a speculative round. Each request takes a free
  sequence, the one whose cached prompt it shares the longest prefix with (the oldest free one on a tie), and its
  drafter's state is the sequence's.
- **The service advances the requests in flight in turn**, admitting a queued one whenever a sequence is free. A long
  prompt gives way to the others at the end of each chunk (`--prefill-chunk`); with one sequence nothing changes.
- **The reservation (decision record of the load-time bill, `SessionOptions::max_step_tokens`) bills every
  sequence's KV cache**: `--sequences 2` takes twice the KV memory of one.

## Consequences

- A conversation's next turn finds the sequence that holds its last one, and a model whose conv or recurrent state
  cannot be cut back (Qwen3.5, LFM3, Inkling 2) still reuses it when the turn extends what is cached.
- Throughput does not grow: each step is one sequence's. Batching the decode steps of the sequences in flight is the
  next step, then KV blocks shared between sequences (a prefix cache across them).
- `--sequences` defaults to 1: more sequences are more KV memory, which the bill shows at load.
