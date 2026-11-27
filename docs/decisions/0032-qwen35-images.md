# 0032: Images for the Qwen3.5 family

Status: **accepted**, done · 2026-11-27

## Context

Qwen3.5, Qwen3.6 and Qwen3.8 ship one vision tower design (transformers' Qwen3_5VisionModel): Qwen3-VL's tower, which
Qwen3-Omni already uses here (`models/qwen3_vision/`, decision 0025), without its deepstack rows, and with its mergers
under Qwen3-VL's names. The decoder reads an image through multimodal RoPE: the full-attention layers rotate a quarter
of each head by positions on three axes (time, height, width), the frequencies interleaved over the axes
(mrope_section [11, 11, 10]); the linear-attention layers read no positions. llama.cpp keeps the tower in a file of its
own (mmproj, projector type `qwen3vl_merger`), which unsloth publishes beside every GGUF file of the three models.

## Decision

- The tower is `qwen3_vision`'s, read under either naming of its mergers; the language model splices its rows in
  place of the image's token embeddings and its decoder (`qwen3_5::Decoder` with `three_axes`) takes the engine's
  positions on three axes. Text alone reads the same RoPE as before (all three axes equal).
- The MTP head keeps reading one position a token (the cache's) and the image token's embedding at an image's rows.
  Drafts change no output, only how many a verify keeps: on an image prompt of Qwen3.5-4B its drafts kept 2.7 tokens a
  verify (k = 3). Giving the drafter positions on three axes means the drafter interface carrying rotary positions;
  left for when it is measured to matter (TODO).
- Beside a block drafter (DFlash, DSpine: trained on text) the tower is not loaded, with a warning; images are then
  refused. SenseNova Loop's tower is not loaded either.
- A GGUF import reads the mmproj file (`convert --mmproj`, `-hf`, or the file beside a model given by its path): its
  tensors under the checkpoint's names, config.json in the release's layout (text_config, vision_config,
  image_token_id), and the release's preprocessor_config.json, which the file does not hold (the three releases share
  one). llama.cpp's converter cuts the patch embedding, a 3-D convolution over two frames, into one 2-D kernel a frame:
  the pack keeps the two (`patch_embed.proj.frames.<t>.weight`, never re-encoded) and the tower sums each frame's
  matmul, as CLAUDE.md 8.8 asks for a tensor split into pieces that are not runs of its rows. Whether a model's
  importer reads such companion files is the importer's answer (`Architecture::reads_projector`, `drafter_apart`), not
  a list in the hub.

## Measured

Qwen3.5-4B against transformers fp32 (Qwen3_5ForConditionalGeneration), four images and a text prompt: on the Arc
Pro B60 at f32 the tower's rows and the last logits at cosine 1 and 24 greedy tokens equal on all five; at f16 rows
0.99999, logits 0.999998, greedy equal. Tiny dense and mixture-of-experts checkpoints with the tower: the same on
cpu-ref and the GPU (f16 flips one near tie of the MoE's). unsloth's Q4_K_M file with its mmproj-F16 file: the tower's
rows at cosine 0.99999 against the reference's.

## Consequences

- A GGUF pack imported without the mmproj file is text-only, as before; serving the file again with the projector
  beside it imports it again with the tower.
- Video stays out: the template's video parts and transformers' per-frame timestamps are not read.
