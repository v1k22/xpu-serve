# Manual (model-aware) quantization: results

Branch-only experiment: what a quantizer that reads the model, its calibration activations and measured sensitivities
can get out of Qwen3.6-35B-A3B and π0.5, against the generic per-tensor q4 / q8 packs. Hardware: Intel Arc Pro B60
discrete GPU (24 GB), xpu-serve SYCL backend.

## Qwen3.6-35B-A3B

Evaluation: 12 x 256 tokens (8 windows of WikiText-2 test, 4 Python standard-library files), next-token distribution at
every position against a float32 reference (the bfloat16 checkpoint run in float32 on the CPU). KL(ref || ours),
top-1 agreement, perplexity change. For scale: the same model computed in bfloat16 is at KL 0.0094, top-1 96%.
Calibration: 96 x 2048 tokens (C4 validation, Python from installed packages, C headers; never the evaluation files).

| pack | GPU memory | KL | top-1 | PPL WikiText | PPL Python | decode |
|---|---|---|---|---|---|---|
| generic q4 g64, round to nearest (baseline) | 18.42 GiB | 0.132 | 86.0% | +3.1% | +17.9% | 54.0 tok/s |
| non-experts q8, experts q4 g64 RTN | 19.25 GiB | 0.052 | 92.2% | +1.0% | +7.0% | |
| **non-experts q8, embedding q4, experts q4 g128 (GPTQ, light prior)** | **18.09 GiB** | **0.052** | **91.9%** | **+0.9%** | **+3.3%** | 48.9 tok/s |
| non-experts q8, embedding q4, experts q4 g64 (regularized GPTQ) | 19.03 GiB | 0.045 | 92.1% | +1.3% | +2.6% | |
| ... experts 16 GiB, one format per bank (sensitivity allocation) | 18.12 GiB | 0.055 | 91.4% | +1.4% | +5.3% | |
| ... experts 15 GiB, one format per bank | 17.15 GiB | 0.073 | 90.5% | +2.2% | +7.9% | 48.2 tok/s |
| ... experts 14 GiB, one format per bank | 16.11 GiB | 0.098 | 89.5% | +3.0% | +8.6% | |
| ... experts 12 GiB, a format per expert | 14.15 GiB | 0.174 | 85.8% | +5.9% | +24.0% | |
| non-experts q8, experts q3 g64 | 15.50 GiB | 0.151 | 88.0% | +3.8% | +22.2% | |
| non-experts q8, experts q2 g64 | 11.75 GiB | 0.419 | 77.3% | +20.9% | +90.7% | |

At the baseline's size the KL is 2.5x lower (perplexity +1.7% instead of +7.8%); at the baseline's KL the pack is about
15.2 GiB (-17%). The accuracy costs 9% of decode speed: the q8 non-expert weights are read every token.

What mattered, measured:

1. **The always-on 10% of the weights.** Attention, DeltaNet projections, shared expert and LM head at q8 instead of q4
   took KL from 0.132 to 0.052 for +0.85 GiB. q4 GPTQ there stays at 0.109, q3 GPTQ reaches 0.205.
2. **The q4 experts are the remaining loss.** f32 compute changes nothing (0.0482 vs 0.0485); moving 6 of 40 layers'
   experts to q8 lowers KL by 15%. The token embedding takes q4 for free (+0.0006 KL, -0.27 GiB).
3. **Calibration must cover the evaluation domains.** GPTQ calibrated on C4 alone halved the WikiText KL and did
   nothing for Python; code in the calibration set fixed that.
4. **GPTQ overfits per-expert Hessians.** An expert sees 30 to 45 000 of the 196 608 calibration tokens. On held-out
   tokens GPTQ's error was 1.5x (busy experts) to 5x (rare ones) its in-sample error, and with the usual damping it was
   worse than round to nearest. A strong prior (the layer's input statistics worth 16 384 tokens) and damping 0.1 keep it
   off the input directions calibration did not exercise: 4-9% better than RTN held-out, KL 0.049 -> 0.045 at q4.
5. **Per-layer sensitivity is uneven but allocation buys little at q4.** A layer's experts at q2 cost +0.005 to +0.028
   KL; down_proj errors cost 3-500x more per unit than gate_up's. Allocating formats per bank from these measurements
   matches uniform q4 g128 at the same size; below ~17 GiB every GiB removed costs a lot.
6. **A format per expert did not pay.** The top 10% of experts carry 38-54% of the router-weighted output energy, and on
   held-out calibration tokens a per-expert choice lowers output error 1.5-1.8x. On the evaluation text it measured the
   same as one format per bank (KL 0.107 vs 0.108 at 14 GiB): expert usage correlates only 0.63 between calibration
   and evaluation text. The engine supports it (`qx` banks: q8 / q4 / q3 / q2 per expert) at no speed cost.
7. **Rotations and act-order: no.** A Hadamard rotation of the expert inputs lowered GPTQ's error 5-14%, act-order with
   static groups +-5%; the expert weights are already Gaussian (kurtosis 3.2).

"Almost lossless" (KL near the bfloat16 compute noise, 0.01-0.02) would need experts above 4 bits, more than the 24 GB
card holds with room for a KV cache; q4-class quality (KL 0.05, top-1 92%) is where this model's size bottoms out.

## π0.5 (pi05_base)

Evaluation: 64 held-out LIBERO frames (2 cameras, robot state, task instruction) with fixed flow noise; relative L2 error
of the 50-step action chunk against the float32 engine. For scale: another noise draw moves the actions by 0.376 (the
policy's own sampling spread); f16 is at 0.0005. Calibration: 64 other LIBERO frames (every linear map's input).

| pack | GPU memory | action error mean | p90 | cosine |
|---|---|---|---|---|
| f16 | 4.86 GiB | 0.0005 | 0.0007 | 1.00000 |
| everything q8 (GPTQ) | 2.63 GiB | 0.0035 | 0.0049 | 0.99999 |
| allocated for 1% | 2.53 GiB | 0.0071 | 0.0115 | 0.99997 |
| allocated for 2% | 2.37 GiB | 0.0172 | 0.0253 | 0.99983 |
| allocated for 3% | 2.22 GiB | 0.0286 | 0.0479 | 0.99955 |
| everything q4 | 1.55 GiB | 0.051 | 0.077 | 0.99865 |
| everything q3 | 1.26 GiB | 0.103 | 0.158 | 0.99431 |

- q8 is lossless and halves the weights. Errors of the blocks add in quadrature (vision 0.020, VLM 0.045, action expert
  0.021 at q4 -> 0.054 predicted, 0.051 measured), so the allocation predicts its error to within 10%.
- The VLM holds 75% of the bytes and is the cheapest to quantize per byte; the vision tower and action expert are 3x
  more expensive per byte saved and stay q8.
- The action expert reads only each VLM layer's keys and values, so the VLM's last layers barely matter (layer 17 at q3
  costs nothing; layers 15-16 little) while layers 3-13 are the sensitive ones.

## Reproduce

    python tools/manual_quant/ref_logits.py --model-dir <hf dir> --data <dir> --out ref32 --dtype float32
    python tools/manual_quant/calib_capture.py tokens --model-dir <hf dir> --data <dir> --seqs 56 --out calib.bin \
        --code <site-packages>:*.py:28 --code /usr/include:*.h:12
    xpu-serve quant-capture <q4 pack> --tokens calib.bin --seq 2048 --out capture --no-vision
    python tools/manual_quant/calib_capture.py stats --model-dir <hf dir> --capture capture --out stats
    python tools/manual_quant/sweep.py --model-dir <hf dir> --stats stats --capture capture --prior-tokens 16384 \
        --damp 0.1 --rtn-errors --match 'language_model\.layers\.[0-9]+\.mlp\.experts\.' --configs q4g64,q4g128 \
        --out sweeps/ex
    XS_PACK_OVERLAY=<non-experts q8>:<embedding q4>:sweeps/ex/q4g128 \
        xpu-serve quant-eval <q4 pack> --ref ref32 --prefill --no-vision

Policies: `policy_cases.py` (observations), `xpu-serve policy-eval --capture`, `calib_capture.py linear`,
`sweep.py --from-stats`, `policy_compare.py`.
