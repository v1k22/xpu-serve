# 0025: Qwen3-Omni, its thinker first

Status: **accepted** for the thinker with text, audio and images; video still to be decided · 2026-11-20

## Context

Qwen/Qwen3-Omni-30B-A3B-Instruct (`qwen3_omni_moe`, 70.5 GB in bf16) is a thinker that reads text, audio, images and
video and writes text, and a talker that turns the thinker's states into speech (a mixture-of-experts decoder of
codec tokens, a code predictor for the other codebooks, and Code2Wav, a convolutional decoder to a waveform). The
thinker is three parts:

- Qwen's audio encoder, the same code as Qwen3-ASR's at a larger size (32 layers of 1,280).
- Qwen3-VL's vision tower: patches of 16 pixels and 2 frames, a learned position table interpolated to the image,
  2-D rotary positions inside the tower, a 2 x 2 merger, and "deepstack" features from three of its layers added to
  the decoder's hidden states at the decoder's first three layers, at the image's tokens.
- A Qwen3 mixture-of-experts decoder (48 layers, 128 experts, 8 a token, no shared expert) with multimodal RoPE: each
  frequency of a head takes the time, height or width position of the token (sections 24, 20, 20, interleaved).

## Decision

- **The thinker first, with text and audio in.** With text and audio only, the reference gives every token one
  position on all three axes (transformers takes the plain branch of `get_rope_index` when there is no image or video
  grid), so the multimodal RoPE is plain RoPE and the engine needs nothing new: the model is Qwen3-ASR's audio
  encoder (moved to `models/qwen3_audio`), the Qwen3 decoder with mixture-of-experts layers (experts joined at load
  from one tensor each, as the release stores them), and the `qwen3` output dialect (tool calls as JSON).
- **The talker is not read**: speech out is a later step.
- **Images: rotary positions apart from cache positions.** An image's tokens take (t, h, w) from its grid, and the
  text after it resumes at the largest position + 1, so a token's rotary position is no longer its index in the KV
  cache. Built as:
  - `MediaInput::positions`, each token's offsets on three axes; the engine computes a prompt's positions
    (`rotary_positions`), keeps the offset of the next one from the cache length, and hands the model each token's
    three positions (`Model::prepare_positioned_inputs`; a model that takes one position a token refuses media that
    bring positions of their own). Prefix reuse, truncation and speculation keep using cache indices.
  - The Rope op names a position axis per frequency (interleaved sections, Qwen3-VL's and Qwen3-Omni's; the
    non-interleaved layout is refused by name) and rotates rows of several a token, which the vision tower's 2-D
    rotary uses too.
  - The vision tower runs with T = its merged tokens and each token's four patches as rows, so no new symbolic
    dimension: attention over the rows as one sequence, the mergers reshape a token's four rows into one. Its output
    row is the merged row and the deepstack rows side by side; the decoder takes the first at `<|image_pad|>` and adds
    the others after its first layers.
  - `--image-max-pixels` lowers the processor's pixel bound: the checkpoint's 12.8 megapixels (12,544 tokens) do not
    fit a 24 GB card beside the weights.
- **Video: to be decided.** Its frames come from a container (MP4 / H.264), which this repository has no decoder for
  (audio's codecs are header-only libraries; a video decoder means a dependency such as FFmpeg's libraries, or frames
  sent as images); its t positions are time-scaled, and its audio is interleaved with its frames by time.

## What was measured

On the released weights (model_recipes.md has the numbers): cpu-ref in f32 equals transformers in fp32 on the
mel features, the encoder's rows and every greedy token, the last logits within 1.5e-5 except one prompt's at cosine
0.99939, which is what transformers' own float32 and float64 differ by there: a mixture-of-experts router's top 8
decided within float32 rounding, which a reference in float32 does not settle. The q4 pack on an Arc Pro B60: logits
cosine 0.984 to 0.986, same top token, 64 tok/s of decode, 16k tokens of context in 24 GB. Images on the released
weights (a photo and a drawn invoice): cpu-ref in f32 equals transformers in fp32 on the patches, the vision
tower's rows, the last logits (cosine 1) and every greedy token.

On tiny checkpoints of transformers' own classes, in the release's layout (a dense layer among the sparse ones, one
tensor per expert): the log-mel features, the encoder's rows, the prompt, the last logits and greedy text equal to
transformers in fp32 on cpu-ref and an Arc Pro B60 at f32 (logits within 1.3e-6, greedy 32 of 32) and f16 (cosine 1,
greedy 27 to 32 of 32), clips of one to nine windows. Images of four sizes (one the processor keeps, one it resizes
down to a multiple of 32, a 1:4 one, one it resizes up to its least pixels): the processor's patches within 6e-8, the
vision tower's rows (merged and deepstack) and the last logits at cosine 1, greedy 32 of 32 on cpu-ref and at f32.
