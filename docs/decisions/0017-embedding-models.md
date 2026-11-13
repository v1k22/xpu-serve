# 0017: Embedding models and /v1/embeddings

Status: **accepted** · 2026-11-13

## Context

An embedding model reads a text whole and returns one vector; a search compares a query's vector with its documents'.
The API is OpenAI's `POST /v1/embeddings`, which vLLM (its pooling models) and llama.cpp (`--embedding`) serve too.
The first three checkpoints are sentence-transformers checkpoints: an encoder, then the modules `modules.json` lists.

| | all-MiniLM-L6-v2 | bge-small-en-v1.5 | nomic-embed-text-v1.5 |
|---|---|---|---|
| Encoder (`model_type`) | BERT (`bert`), 6 layers, 384 wide | BERT, 12 layers, 384 wide | NomicBERT (`nomic_bert`), 12 layers, 768 wide |
| Positions | a table of 512 | a table of 512 | rotary over the whole head, base 1000 |
| MLP | exact GELU, biases | exact GELU, biases | SwiGLU, no biases |
| Pooling, Normalize | mean, yes | CLS, yes | mean, no |
| Longest input (`max_seq_length`) | 256 | 512 | 8192 (trained on 2048) |

## Decision

- **A fourth kind of model**, beside language, policy and decision models: `EmbeddingModel` (`models/embedding.h`), a
  spec (dimension, longest input, vocabulary, pooling, normalize), one encoder program without state (its output the
  final hidden states) and the inputs of a token sequence. `model_kind` calls a checkpoint one when `modules.json`
  lists a Pooling module beside a `model_type` served as an encoder (`bert`, `nomic_bert`). It generates nothing, so
  it is not a `Model`; it lays out no prompt and runs one program once, so it is not a `DecisionModel` either.
- **The pipeline is the checkpoint's, as sentence-transformers 6 reads it** (`models/common/sentence_transformers`):
  the Pooling mode (its legacy `pooling_mode_*` flags or its `pooling_mode`), L2 normalization exactly when a Normalize
  module is listed (vLLM does the same), the longest input from `sentence_bert_config.json` (for BERT never more than
  its position table). Everything else that would change an embedding is refused by name: other pooling modes (max,
  mean_sqrt_len_tokens, weightedmean, lasttoken) or several at once, Dense and other modules, a module of the
  checkpoint's own code, a default prompt, `do_lower_case` with a tokenizer that does not lowercase.
- **The encoders share BERT's post-norm layer** (`models/common/encoder`) and differ in their parts: BERT with its
  position table and biases; NomicBERT with one fused QKV projection (sliced), rotary queries and keys, SwiGLU. Past
  the positions NomicBERT was trained on, its reference rotates on unchanged unless a `rotary_scaling_factor` asks for
  dynamic NTK scaling: so does the engine, and a scaling factor is refused. No new graph op.
- **Pooling is host work**, from the hidden states the program returns: one sequence and no padding, so the mean is
  over every row, [CLS] and [SEP] included (sentence-transformers with a mask of ones), and CLS is row 0; normalizing
  divides by the norm as torch's `F.normalize` does.
- **The runtime takes token ids** (`src/embedding/`: `EmbeddingRunner`, the model and one session; `EmbeddingService`,
  the tokenizer and a device thread). A text becomes `tokenizer.with_special_tokens(tokenizer.encode(text))`: the ids
  with the special tokens the tokenizer's post-processor puts around a sequence, as sentence-transformers tokenizes.
- **The API follows OpenAI's**: `input` as a string, strings, token ids or lists of token ids (ids are read as given,
  without special tokens, as vLLM reads them); `encoding_format` `float` or `base64` (float32, little-endian: what the
  OpenAI Python client asks for by default); OpenAI's error bodies, 400 naming the field (`input[2]`); `model` not
  checked, as for chat; `user` ignored; `GET /v1/models` with the longest input as `context_length`. The server of an
  embedding model has no chat routes. Where servers differ:
  - **Normalization is the checkpoint's.** llama.cpp L2-normalizes every embedding; here, as in vLLM, a checkpoint
    without a Normalize module (nomic-embed-text-v1.5) returns its pooled vector as it is.
  - **`dimensions` is refused unless it is the model's own.** None of these checkpoints declares shorter embeddings;
    vLLM refuses them for such models too. nomic-embed-text-v1.5 was trained for them, but its card makes them with a
    layer norm before the cut, which a plain cut would not reproduce.
  - **An input longer than the model reads is refused, not cut** (400 with its token count and the limit), as OpenAI
    and vLLM do; sentence-transformers would cut it silently.
  - **Each input is one run of the encoder**, in order, and requests run one at a time.

## What was measured

Against sentence-transformers 6.3 in float32 on the CPU (`tests/parity/embedding/reference.py`; nomic-embed-text-v1.5
with its own modeling code), the engine on cpu-ref at f32 (`tests/unit/model/embedding_parity.cpp`): 11 texts of 3 to
50 tokens (accents, CJK, emoji, code, literal special tokens) and filler passages cut to the lengths below. The worst
case of each checkpoint:

| Checkpoint | Longest inputs | Hidden states: 1 - cosine, max \|difference\| (largest value) | Embedding: 1 - cosine, max \|difference\| | cpu-ref, longest input |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 256 | 6.0e-13, 2.6e-6 (6.2) | 3.1e-13, 1.5e-7 | 0.46 s |
| bge-small-en-v1.5 | 512 | 3.6e-13, 2.4e-6 (4.3) | 2.1e-13, 1.2e-7 | 1.5 s |
| nomic-embed-text-v1.5, up to 50 tokens | 50 | 9.8e-13, 1.4e-5 (11.4) | 6.5e-13, 2.6e-6 | |
| nomic-embed-text-v1.5 | 2,000 | 2.2e-11, 2.2e-4 (13.3) | 5.2e-13, 2.4e-6 | 7.4 s |
| nomic-embed-text-v1.5 | 8,192 | 6.1e-10, 6.6e-3 (14.7) | 6.0e-11, 2.4e-5 | 69 s |

The tiny checkpoints of `make_tiny.py` (BERT with mean pooling, BERT with CLS and Normalize, NomicBERT read to 256
tokens after training on 64) agree to 3.3e-6. The difference that grows with nomic's length comes from the
reference's float32: it rounds each rotation angle (position times frequency) to float32, up to 2.4e-4 radians at
position 8,000, and sums 8,192 keys in float32, where cpu-ref works in double. With the angles in float64 the
reference's largest hidden-state difference at 8,192 tokens halves (3.0e-3, 1 - cosine 2.1e-10). The GPU was not
measured yet.

## Consequences

- `serve`, `run --prompt` and `bench` take an embedding model's directory and need no flag; `--ctx` can lower the
  longest input. Text needs the checkpoint's tokenizer: WordPiece for these three, which the tokenizer gains
  separately; token ids work without it.
- Not done: several inputs in one step (one padded or block-diagonal run instead of a run per input), shorter
  embeddings for checkpoints that declare them, the rerank and score APIs, packs of embedding models (`convert`
  refuses them by name).
