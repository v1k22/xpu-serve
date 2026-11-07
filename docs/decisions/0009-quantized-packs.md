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
  0.87 to 0.99; groups of 32 lift it to about 0.92, keeping the MLP at q8 to about 0.95). The weights themselves
  quantize as well as Gemma's (about 9 % relative error per tensor, similar outliers), so the error grows through the
  activations: the case activation-aware quantization (AWQ) addresses. q4 needs AWQ before it is the default.
- Prefill copies each quantized weight to f16 before its GEMM: about 40 ms per 1024-token chunk on Qwen3.5-4B. A GEMM
  that dequantizes in registers (oneDNN weight decompression or a SYCL*TLA mixed-input GEMM) removes it.
