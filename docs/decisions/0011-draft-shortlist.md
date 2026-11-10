# 0011: an MTP head drafts over a shortlist of the vocabulary

Status: **accepted** · 2026-11-10

## Context

A draft step of a Qwen3.5-family MTP head runs one decoder layer and then the target's `lm_head` to pick a token. For
Qwen3.8-27B at q4 the layer is 0.26 GB and the head 0.79 GB (248,320 tokens x 5,120): on a GPU bound by memory
bandwidth the head is most of a draft step (Arc B390: about 7 of 9.8 ms), and a round of k drafts pays it k times.
The pick only has to be the head's argmax often enough: the target verifies every draft, so a wrong pick costs a
draft, never output.

## What was measured

1,872 drafts of Qwen3.8-27B (the MTP head's inputs to the `lm_head` over the six bench prompts, k = 4), fp32 head: how
often the head's argmax survives a cheaper scoring of all rows followed by the exact product over the best N.

| Scoring of every row | Bytes read | argmax in the best 32 | 64 | 256 |
|---|---|---|---|---|
| sign of each weight, one scale per row | 159 MB | 99.5% | 99.8% | 99.9% |
| the same, x rounded to 8 / 6 bits | 159 MB | 99.5% / 99.4% | 99.8% | 99.9% |
| two sign planes (the residual's signs) | 318 MB | 99.8% | 100% | 100% |
| affine 2-bit, groups of 64 | 397 MB | 100% | 100% | 100% |
| affine 3-bit, groups of 64 | 556 MB | 100% | 100% | 100% |
| signs of every second column | 79 MB | 94.1% | 96.0% | 98.2% |
| truncated SVD of the head, rank 512 / 1024 | 127 / 254 MB | 55% / 63% | 64% / 70% | 78% / 82% |

Without scoring at all: the 64,000 lowest token ids hold 94.6% of the picks (the 32,000 lowest 89.9%), which matches
the acceptance lost when drafts were restricted to them (3.21 to 3.02 tokens per verify), and such a list depends on
the language of the text. The best row of each block of 64 consecutive rows by sign score keeps 97.9%: neighbours in
the vocabulary are similar tokens, so the shortlist is the best N of all rows.

## Decision

Drafts score every row of the head with its one-bit sketch (`quant::sign_sketch`: a bit per weight, a scale per row;
`Graph::sign_matmul`), take the 64 best (`TopK`) and compute their exact logits (`Gather` on the head, `MatMul`):
`blocks::shortlist_head`. The draft program outputs `logits` over the `candidates`, as a Gemma 4 assistant's centroid
head does. The sketch is built from the head when the model loads (under a second) and costs one bit per weight of
device memory (151 MiB for Qwen3.8-27B). `--draft-shortlist N` sets the count; 0 reads the whole head.

On the device SignMatMul rounds x to 12 bits of its largest magnitude and counts bits: per 32 columns a popcount per
bit plane of x, where selecting and adding floats takes two operations a column and would bind the kernel by its
arithmetic instead of the memory. TopK over a row as wide as a vocabulary keeps the best of each block of 256 and
picks among the blocks (a pick used to scan the row: 10 ms for 64 picks of 248,320).

## Result

Qwen3.8-27B q4, k = 4, six prompts: 1,038 of 1,876 drafts accepted against 1,040 of 1,872 with the whole head (3.20
and 3.21 tokens per verify). A draft step: 9.8 to about 5 ms on the Arc B390, 3.6 to 2.3 ms on the Arc Pro B60.

## Rejected

- A list of frequent tokens: no scoring to pay for, but it depends on the text (its language most of all) and on
  data the repository does not have.
- A low-rank head: the head's spectrum is flat (the top 1,024 of 5,120 directions hold 37% of its energy).
- Two or three bits per weight (what the MLX reference engine drafts with): closer scores than needed for a
  shortlist of 64, at two to four times the bytes.
