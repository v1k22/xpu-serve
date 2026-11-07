# 0009: Quantized packs (q8, q4) and `convert`

Date: 2026-11-07 · Status: accepted

**What it is.** `xpu-serve convert <dir> --out <pack> --quant q8|q4` writes a pack: the checkpoint with its matmul
weights quantized by round to nearest (q8 symmetric; q4 affine with an f16 scale and bias per group of 32, 64 or 128
columns, DESIGN.md 6.3), everything else copied, and `xpu_pack.json` naming each quantized tensor's format. Every
command takes a pack where it takes a model directory.

**No model code per format.** `convert` builds the model's own step graph and quantizes what the graph reads as a
matmul weight; gather tables (and so a tied LM head) take q8; matrices under 1M elements and every other tensor stay as
they are. Loading a pack, the weight store presents each quantized tensor as one `Q8` / `Q4` view, so the same graph
runs on it; cpu-ref dequantizes at compile, the SYCL backend runs an int4 / int8 GEMV for decode and dequantizes to f16
for each prefill GEMM. `--q8 name,...` keeps chosen weights at q8.

**Results** on the Intel Arc Pro B60 discrete GPU (SYCL, f16 activations), 2026-11-07.

| Model | Pack | Size | Logits cosine vs HF fp32 (6 prompts) | Greedy = HF | MMLU-Pro (25) | Decode tok/s (128 / 4096) | TTFT 1024 |
|---|---|---|---|---|---|---|---|
| Gemma 4 E2B | bf16 checkpoint (runs as f16) | 10.25 GB | 1.0 | 6 / 6 | 52 % | 61.5 / 58.1 | 155 ms |
| | q8 | 8.08 GB | ≥ 0.99998 | 6 / 6 | 56 % | 92.4 / 83.4 | 183 ms |
| | q4 g64 | 7.22 GB | 0.9916 – 0.99997 | 1 / 6 | 52 % | 116.5 / 105.4 | 183 ms |
| Qwen3.5-4B | bf16 checkpoint (runs as f16) | 9.32 GB | ≥ 0.999998 | 6 / 6 | 68 % | 41.9 / 41.1 | 222 ms |
| | q8 | 5.25 GB | 0.9989 – 0.9999 | 6 / 6 | 68 % | 70.7 / 67.7 | 268 ms |
| | q4 g64 | 3.58 GB | 0.87 – 0.99 | 1 / 6 | 68 % | 92.0 / 86.9 | 269 ms |

- MMLU-Pro: evalscope (outside the repo) against `xpu-serve serve` over the OpenAI API, temperature 0, the first 5
  questions of math, physics, computer science, biology and history; Qwen3.5 with thinking off. At 25 questions one
  answer is 4 points, so this shows no loss that size can see, not that there is none. All 150 requests served cleanly.
- Gemma's pack stays large because its per-layer embedding table (4.7 GB, read on the host) and vision tower are not
  quantized yet.
- `convert` takes 6 to 8 s per model; the GPU and cpu-ref agree on every pack (cosine 0.999999 on the test model; the
  Qwen q4 numbers above are identical on cpu-ref, so they are quantization error, not a kernel bug).

**Findings.**
- q8 is the safe default: decode 1.5 to 1.7 times faster, quality indistinguishable from f16 in every measure here.
- q4 doubles decode speed (1.9 to 2.2 times) but round to nearest costs Qwen3.5-4B a lot of logit fidelity (cosine
  0.87 to 0.99 at groups of 64; groups of 32 lift it to about 0.92, keeping the MLP at q8 to about 0.95). The weights
  themselves quantize as well as Gemma's (about 9 % relative error per tensor, similar outliers), so the error grows
  through the activations: the case activation-aware quantization (AWQ) addresses (below). q4 defaults to groups of
  32.
- Prefill copies each quantized weight to f16 before its GEMM: about 40 ms per 1024-token chunk on Qwen3.5-4B. A GEMM
  that dequantizes in registers (oneDNN weight decompression or a SYCL*TLA mixed-input GEMM) removes it.

## AWQ (added the same day)

`--method awq` runs an AWQ-style search for q4 (round to nearest stays the default, see the eval below): the
model's graph is copied with each quantized matmul's input as an extra output, calibration text runs through our own
engine, and each weight gets column scales `s = rms(x)^alpha` and per-group clipping chosen to minimize
`sum_c mean(x_c^2) * (w_eff - w)^2`; `1 / s` is stored as `.input_scales` and applied to `x` by the kernels. The
built-in calibration text is about 2,200 tokens; `--calib file` takes more.

| Model | q4 pack | Weighted error vs RTN | Logits cosine vs HF fp32 (5 chat prompts) | Greedy tokens = HF (of 192 / 128) |
|---|---|---|---|---|
| Qwen3.5-4B | RTN, group 64 | — | 0.865 – 0.934 (mean 0.891) | 85 |
| | RTN, group 32 | — | 0.895 – 0.955 (mean 0.919) | |
| | AWQ, group 64 | -25 % | 0.839 – 0.938 (mean 0.902) | 130 |
| | **AWQ, group 32** (default) | -25 % | 0.917 – 0.962 (mean 0.940) | 110 |
| Gemma 4 E2B | RTN, group 64 | — | 0.9917 – 0.99997 (mean 0.9969) | 59 |
| | AWQ, group 64 | -46 % | 0.9984 – 0.99999 (mean 0.9988) | 44 |

Calibration and search take 50 to 80 s per model on the Arc Pro B60. AWQ narrows the gap to q8 (cosine 0.999) but does
not close it for Qwen3.5-4B; greedy agreement is a noisy measure for quantized models (one early different token
changes the rest), which is why the eval below matters more.

MMLU-Pro (the same 25 questions through `xpu-serve serve`):

| Model | q4 pack | MMLU-Pro | Answers that opened Gemma's thinking channel unasked |
|---|---|---|---|
| Qwen3.5-4B | RTN, group 64 | 68 % | — |
| | AWQ, group 64 | 68 % | — |
| | AWQ, group 32 | 68 % | — |
| Gemma 4 E2B | RTN, group 64 (host table q8, scale fix) | 56 % | 0 |
| | AWQ, group 32 | 52 % | 14 |
| | AWQ, group 64 | 40 % | 23 |

AWQ helps Qwen3.5 (closer logits, same score) but shifts Gemma 4 toward its thinking channel, which the f16, q8 and
RTN packs never open unasked, and costs it MMLU-Pro points at this sample size. The likely cause is calibration on
plain prose, which never exercises the chat-template tokens that decide that first token; chat-formatted calibration
is the next step. Until it is measured, round to nearest stays the default and AWQ is opt-in.

The serving checks behind these runs, on 0.0.0.0 through the OpenAI API: thinking on by default for Qwen3.5 (off for
Gemma 4) and switchable with `chat_template_kwargs.enable_thinking`, tool calls (plain and streamed) with
`finish_reason: tool_calls`, and a tool result turned into the final answer, on both models.

Pack sizes after quantizing Gemma 4's per-layer embedding table (q8, dequantized on the host per token) and its vision
tower (q8): Gemma 4 E2B q8 5.69 GB, q4 4.83 GB (group 64) or 5.04 GB (group 32), down from 8.08 / 7.22 GB.
