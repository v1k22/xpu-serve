# 0010: int8 KV cache (`--kv-dtype i8`)

Date: 2026-11-08 · Status: accepted (opt-in)

**What it is.** `--kv-dtype i8` (on `serve`, `run` and `bench`; `SessionOptions::kv_dtype`) stores each KV slot as
int8 codes with one f32 scale per (token, KV head): `scale = max|x| / 127`, `code = rint(x * 127 / max|x|)`. The
default, `auto`, keeps f16 (f32 at f32 precision). The KV memory roughly halves: Gemma 4 E2B goes from 18 KiB to
9.1 KiB per token, Qwen3.5-4B from 32 KiB to 16.3 KiB per token, so 32k tokens take 0.29 GB instead of 0.58 GB and
0.52 GB instead of 1.0 GB.

**How it runs.**
- One rounding everywhere: `quant::quantize_kv_row_i8` on the host, `quantize_kv_i8` on the device. cpu-ref stores
  the rounded values, so it stays the exact oracle (the device matches it to 1e-7 at f32).
- Decode attention (the per-row kernel) reads the codes. It applies the k scale to the score and the v scale to the
  probability, and each lane loads 4 bytes at a time.
- Prefill attention (the GEMM path) dequantizes to f16 only the keys the chunk can see (the window on sliding layers),
  then runs the usual two batched GEMMs.
- A slot is int8 only when Attention ops are its only readers. Anything else (a graph output, a view) keeps the
  slot f16, and the log says how many slots are int8.

**Results** on the Intel Arc Pro B60 discrete GPU, f16 compute, 2026-11-08.

| Model | Decode tok/s f16 KV → int8 KV (4k / 16k / 32k) | Prefill 32k | Logits cosine vs HF fp32 (6 prompts) | Greedy = HF |
|---|---|---|---|---|
| Gemma 4 E2B | 58.7 → 58.9 / 51.8 → 54.8 / 45.0 → 49.1 | 7.36 → 7.96 s | 0.99979 – 1.0 (f16 KV: 1.0) | 5 / 6 (f16 KV: 6 / 6) |
| Qwen3.5-4B | 41.3 → 41.1 / 38.7 → 39.4 / 35.7 → 37.0 | 11.58 → 11.79 s | 0.99947 – 0.99995 (f16 KV: ≥ 0.999998) | 6 / 6 |

MMLU-Pro (the same 25 questions as 0009, through `xpu-serve serve`, Qwen3.5 with thinking off):

| Model | Weights | f16 KV | int8 KV | Answers that changed | Right ↔ wrong |
|---|---|---|---|---|---|
| Qwen3.5-4B | bf16 checkpoint | 68 % | 68 % | 2 | 0 |
| | q4 AWQ g32 | 68 % | 68 % | 2 | 0 |
| Gemma 4 E2B | bf16 checkpoint | 52 % | 48 % | 4 | 1 |
| | q4 g64 | 56 % | 52 % | 5 | 3 |

Needle in a haystack at about 7k and 27k tokens: both models find the fact with either KV dtype.

**Findings.**
- int8 KV is about as close to f16 as q8 weights are (0009: Gemma q8 ≥ 0.99998, Qwen q8 0.9989 – 0.9999). Qwen3.5
  shows no change in any measure. Gemma 4 moves more answers on questions it was unsure about: one answer net in
  each pair, which this sample size cannot tell apart from noise. One likely reason is that 20 of its 35 layers
  read an earlier layer's cache, so each rounding error reaches more layers.
- Speed: decode at long context is 2 to 9 % faster (less KV to read), prefill 1 to 8 % slower (the f16 copy of the
  visible keys per chunk). On these small models decode mostly reads weights, so the gain is mainly memory: twice
  the context in the same VRAM, which matters most for the 27B to 35B targets on 24 to 32 GB GPUs.
- It stays opt-in until a larger eval (more questions, long-context tasks) shows Gemma 4 does not lose accuracy.
  Per-channel key scales (KIVI-style) are the next step if it does.
