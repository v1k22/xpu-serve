# GGUF experiment (2026-11-12)

Not part of the engine: two scripts used to size what reading GGUF would take.

- `parse_gguf.py <file or its first bytes>` prints a GGUF's metadata and its tensors by storage type.
- `gguf_to_pack.py --gguf X.gguf --side <checkpoint dir> --out <pack> [--hf <checkpoint dir>]` rewrites a
  Qwen3.5-family GGUF as an xpu-serve pack, undoing what llama.cpp's converter does to the tensors (norm weights
  stored as 1 + w, `A_log` as `-exp`, value heads tiled instead of grouped by key head, the MTP block as the last
  block with `nextn.*` names). With `--hf` it compares every tensor with the original checkpoint.

What it showed, on `unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-Q4_0.gguf` and an Arc Pro B60:

- Q4_0 and Q4_1 blocks are the pack's q4 with groups of 32 without loss (the nibbles repacked; Q4_0's offset is
  -8 times its scale), Q8_0 is q8 without loss, and a Q4_0 embedding table fits q8 without loss.
- That file is not all Q4_0: by weight 87 % Q4_0, 5.5 % Q5_K, 4.7 % Q6_K, 2.6 % Q4_1. Q5_K and Q6_K have no
  counterpart in the pack formats; the script decodes them and rounds them again to q8 (their error against the
  checkpoint goes from 3.68 to 3.72 % and from 1.82 to 1.90 %).
- Every other tensor comes back exactly equal to the checkpoint's after the transforms above.
- The unchanged engine serves the result: 17.3 GiB on the device (its own q4 pack: 16.2), decode 20.2 tok/s (22.2),
  prefill 830 tok/s (830), last-token logits against Hugging Face fp32 at cosine 0.79 to 0.999 on six prompts (0.67 to
  0.996), and the MTP head from the file's block 64 drafts.
