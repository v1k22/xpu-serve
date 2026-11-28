# 0037: sliding attention layers keep their window alone, in attention-only models too

Status: **accepted** · 2026-11-28

## Context

Gemma 4's layers mostly attend over a sliding window: 50 of 31B's 60 layers look 1,024 tokens back, 25 of 26B-A4B's
30, 40 of 12B's 48. The GPU backend kept only that window of such a layer (and room for a step) in models with conv or
recurrent state, whose prefix reuse already goes through a snapshot (decision 0029); in an attention-only model it
kept every key, so that a sequence could be cut back to any prefix. On a 24 GB Arc Pro B60 that left Gemma 4 31B
(Q4_K_M with its vision tower, 18.9 GiB of weights) room for 8,192 tokens with an int8 cache, 901 KB a token at f16,
of which its 10 full-attention layers need 82 KB.

## Options

A. Keep every key (as before): any shared prefix is reused, the context is what the whole cache leaves.
B. Keep the window in attention-only models too, the prefix coming back from the snapshot taken where a chat's
   messages end, as for conv and recurrent state.
C. B, and more snapshots a sequence (where a system prompt ends, for instance), or vLLM's way: a paged cache whose
   blocks past the window stay cached until evicted, so any prefix whose window is still cached comes back.

## Decision

B, by default: `--sliding-kv window` (`EngineOptions::window_kv`, `SessionOptions::window_kv`), for a model with
sliding layers; `--sliding-kv full` keeps every key and reuses any shared prefix (A). The service turns it on only when
the model's own graph has a sliding KV slot, so a drafter's sliding slots (DFlash, DSpark) beside a target without any
change nothing. The engine treats such a session as it treats conv and recurrent state (`Engine::snapshot_prefix_`):
a prefill chunk ends at `GenerateRequest::snapshot_at`, the state is saved there (a sliding slot's window of keys and
values, in host memory), and a later prompt that shares the tokens up to it starts from it. cpu-ref keeps every key
whatever the option says; the engine reuses prefixes the same way on both backends.

C waits: a paged cache is a different engine, and a second snapshot a sequence doubles the host memory of a part
that is already the largest (31B's window is 800 MiB at f16).

## Measured

Arc Pro B60 (22.7 GiB), 31B Q4_K_M with its vision tower, decode of 128 tokens after a 4,096-token prompt:

| | Context that fits | Prefill | Decode |
|---|---|---|---|
| every key, `--kv-dtype i8` | 8,192 (457 KB a token) | 670 tok/s | 15.7 tok/s |
| window alone, `--kv-dtype i8` | 65,536 (41 KB a token, 0.9 GiB of windows) | 678 tok/s | 15.7 tok/s |
| window alone, f16 | 22,528 | 729 tok/s | 15.6 tok/s |

A 32,768-token prompt: 512 tok/s, then 12.5 tok/s. 26B-A4B (UD-Q4_K_M with its tower) and 12B (q8) reserve the
model's whole 262,144 tokens with an i8 cache, 2.1 and 8.5 GiB to spare. Parity against transformers fp32 is what it
was with every key (31B: text logits 0.947 to 0.9994, images 0.987 to 0.992 with every greedy answer equal).

Two chat turns on 12B with a 4,284-token system prompt: the second turn reused 4,277 tokens from the snapshot (4,280
keeping every key) and answered the same; 0.96 s against 0.77 s to its answer, the difference being the window's
restore from host memory.

A first cut reserved a windowed slot's window and a step but let it grow a window more when it moved: 31B at 24k
tokens with an f16 cache then ran out of memory in its first prompt, and at 64k with an i8 cache decoded at 10.1
tok/s with 0.2 GiB to spare. A slot now slides its window back to the front of its own buffers (256 rows past its
window and a step), and a device test checks that a session reserved at compile time allocates nothing more.

## Consequences

- A conversation's next turn starts from where its messages ended, as before; a new conversation sharing only a
  system prompt with the last one, and raw `/v1/completions` prompts, prefill from the start (`--sliding-kv full`
  keeps the old behaviour).
- A snapshot holds every sliding layer's window: 800 MiB for 31B at f16 (400 MiB with `--kv-dtype i8`), 320 MiB for
  12B, in host memory, one a sequence; saving and restoring it crosses PCIe once.
- Truncate on such a slot reaches back only as far as the window it holds before the cut, the last step at least
  (`Session::truncate`): a speculative round's verify is cut back as before.
