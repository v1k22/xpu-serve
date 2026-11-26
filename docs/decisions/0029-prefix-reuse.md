# 0029: Reusing a prompt's prefix: state snapshots, not paged KV blocks

Status: **accepted**: snapshots of a sequence's state (option A), done; copies of a prefix between sequences still to
come; paged KV blocks shared between sequences (option B) deferred · 2026-11-26

## Context

A request reuses the KV of the free sequence whose cached prompt it shares the longest prefix with (0027). A model
with conv or recurrent state (Qwen3.5 to 3.8, SenseNova Loop, LFM3, Inkling 2) can only extend that cache: the state
summarizes the whole sequence and cannot be cut back into it. When the new prompt departs from the cached tokens
anywhere, the sequence starts over. A chat template that drops the reasoning of earlier turns (Qwen's, with thinking
on) does that on every new user turn, and a tool call the template renders otherwise than the model wrote it does it
on every tool round: each turn prefills the whole conversation again, minutes for a long one on a 27B model.

xpu-serve runs mostly on one machine for one or a few users, not in a data centre: past the weights and the KV cache
of the context, it should hold little memory.

## Options

- **A. Snapshots of a sequence's state, and copies between sequences.** A sequence saves its conv and recurrent state
  (and what a windowed KV slot holds that a cut cannot give back) at the end of each prompt's conversation, before the
  generation prompt: where the next turn's prompt departs from it. A new prompt that shares the sequence's tokens up
  to there restores the snapshot, cuts the KV back to it and prefills the rest. A request may take a prefix and its
  snapshot from another sequence as a copy. The attention kernels and the memory bill stay as they are.
- **B. Paged KV blocks shared between sequences** (DESIGN.md 9.4; vLLM's automatic prefix caching): the cache in
  fixed blocks keyed by a hash chain of their tokens, reference-counted, freed blocks kept in an LRU pool; a prompt
  takes the cached blocks of its longest prefix. vLLM keeps a hybrid model's recurrent state in the same pool at block
  boundaries only (`mamba_cache_mode=align`), so a hit resumes at the last boundary, which is thousands of tokens
  back for some models; caching the state at a request's exact end is still an open RFC there (vllm#66844, 2026-11).
  llama.cpp, built for local use, keeps a cache per slot and context checkpoints of the state in system RAM
  (`--ctx-checkpoints`, `--cache-ram`): the family of A.

| | A | B |
|---|---|---|
| Memory beyond the model and the context | one snapshot a sequence: Qwen3.8-27B about 150 MB (48 layers of 48 x 128 x 128 f32 state), Inkling 2 a few MB, none for attention-only models; in host memory | a pool sized up front, with a hybrid's states per block in it |
| Where a hit resumes | the snapshot's exact position | a block boundary |
| Many users sharing a long prefix | each sequence holds its own copy | held once |
| Cost to build | the engine and two session calls | every attention path through block tables, the bill, the hybrid states |
| Speed with no hits | unchanged | vLLM reports 11 to 16 % lost on a hybrid model (vllm#60008) |

## Decision

A. The snapshot lives in host memory (restoring 150 MB over PCIe takes some 10 to 20 ms, against seconds to minutes
of prefill), one a sequence, taken where the prompt's conversation ends; models without conv or recurrent state need
none, their KV is cut back as before. B waits until several users sharing long prefixes make memory the limit, which
on one machine they rarely do.

## Measured

Arc Pro B60, Qwen3.8-27B Q4_0, a chat with thinking on: a first turn of 7,140 tokens, then the second, which sends the
first reply back without its reasoning (7,159 tokens). Before, the second turn found its sequence departing from the
prompt at the reply and prefilled all of it: 8.3 s to its first token. With the snapshot taken where the first turn's
messages ended (7,135 tokens; 149 MiB, saved in 77 ms), it restored that state in 62 ms and prefilled 24 tokens: 0.42
s. The first turn took 0.3 s longer (the save, and the prefill chunk that ends at the snapshot).

## Consequences

- A session can save and restore a sequence's state (`Session::save_state`, `restore_state`: the conv and recurrent
  state, and on the GPU what a KV slot that keeps only its window holds); the engine ends a prefill chunk at
  `GenerateRequest::snapshot_at`, saves the state there and restores it in prefix reuse, and a free sequence is chosen
  by the prefix a request can use of it; the chat and messages APIs set `snapshot_at` where the template's rendering
  without the generation prompt ends.
- A snapshot is the latest prompt's only: an earlier point of a conversation (an edited old message) still prefills
  from where the two prompts agree, or from the start.
- Revisit B when many sequences share long prefixes and their copies crowd out the context, or when the server runs
  for many users at once.
