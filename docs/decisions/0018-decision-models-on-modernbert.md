# 0018: Decision models on ModernBERT (Laya, Julia-1)

Status: **accepted** · 2026-11-13

## Context

Two more open decision models speak the System One API (docs/decisions/0015), and both are the same model class
under two names: an encoder read at a mask token per option.

| | Laya (`convaiinnovations/laya`) | Julia-1 (`SupersonicLabs/Julia-1`) |
|---|---|---|
| Encoder | ModernBERT-large: 28 layers, 1024 wide (the root model and `typed-decisions/`); mmBERT-base, 22 x 768 (`multilingual/`) | mmBERT-small: 22 layers, 384 wide, Gemma's vocabulary of 256,000 |
| Head | the question type's embedding on every row, two `nn.TransformerEncoderLayer` (norm first, ReLU, 4 x wide, no positions) over the whole sequence, a scorer (LayerNorm, Linear, GELU, Linear) at each option's mask token | the same class, copied |
| Reference | `rl_common.py` (`build_sequence`, `DecisionModel`), `rl_agent_api.py` (`system_one`) | `julia/data.py` (`sequence`), `julia/typed.py` (`predict_typed`), `inference-policy.json` |
| An option | `name: description` (a name alone without one), `level i: text`, `false: ...` / `true: ...` in the model's wording | its description alone; a noul's literal `false` / `true` without criteria |
| Probabilities | softmax of the logits over a temperature fitted per question type and option count | plain softmax |
| What does not fit | cut: an option to 48 tokens, the options and question evenly to the question budget (192), the state to `max_len` (512) | refused (`strict_encoding`): 2 to 20 options, text instructions and descriptions, at most 48 tokens an option, 512 for the question and options, 8,192 in all |

A question is a sequence of its own: `[CLS] "<type> question: <instructions>" [SEP]`, then per option the mask token
and its text, then `[SEP] state [SEP]`, each piece tokenized alone. ModernBERT is pre-norm, with bias-free LayerNorms,
rotary positions with a base per layer kind, a GeGLU MLP, and attention over the whole sequence in every third layer
and over the 64 tokens on either side of each token in the others. llama.cpp serves both from GGUF files
(`ggml-org/Laya-GGUF`, `ggml-org/Julia-1-GGUF`: architecture `modern-bert`, `decision.type` laya).

## Decision

- **A symmetric window is the non-causal window.** The graph's non-causal attention took a window that bounded only
  the keys before a query (a block drafter's block over its cache, where nothing after the block exists). It now
  reaches as far after a query as before it, `|p_i - j| < sliding_window`, which is the same for the drafter and is
  ModernBERT's local attention (`sliding_window = local_attention / 2 + 1`). No new attribute; cpu-ref and the three
  SYCL lowerings (the per-row kernel, the GEMM path's key range and softmax mask, the fused matrix-engine kernel)
  bound the later keys as they bound the earlier ones.
- **ModernBERT is a graph builder, not a model** (`models/modern_bert`): the encoder over weights under a prefix, its
  config parsed in both of transformers' forms and every variant it does not implement refused by name. torch.nn's
  Linear and LayerNorm (with or without a bias) moved to `models/common/torch_nn`, which Clef and the BERT encoders
  now share; the blocks gained the exact GELU for the GeGLU.
- **One `DecisionModel` for both, the differences data** (`models/laya`, `Policy`): option wording, budgets,
  temperatures, strictness and whether a question must leave the state a token. One stateless program: the encoder,
  the type row, the head, the scorer on every row; a step per question; the logits at the markers through the
  softmax on the host. Found by `encoder/config.json` beside `rl_agent_config.json` or `julia_config.json`, as
  llama.cpp's converter finds them (Laya's variant folders have no `config.json`); all three Laya variants share the
  code path (the multilingual one with mmBERT's tokenizer).
- **Each model's own policy.** Laya cuts as its reference does and refuses only what it refuses (options past the
  sequence; a noul description that is not text, which its reference fails on); a cut state is logged at info, since
  the response cannot say so. Julia-1 follows its `inference-policy.json`: strict encoding refuses by name what would
  be cut and a request that spells its mask token. A question without instructions reads as null, which Laya writes
  `null` as its reference writes an explicit null (the request schema does not tell a missing one from a null one;
  the reference's API wrapper fails on a missing one). Answers (choice, score, confidence) come from
  `decision/answers.cpp` as for every model, not from Laya's `act_probability` or Julia's `max_probability`.
- **The tokenizer sits in the checkpoint's `tokenizer/`, with no chat template.** `DecisionSpec` names the folder and
  whether the model renders a template; the runtime loads the model before its prompt tools. The warm-up's sample
  request now describes every option (Julia-1 refuses one without).
- **GGUF: `modern-bert` with `decision.type` laya** (`convert/gguf/modern_bert.cpp`) imports into the original
  checkpoint's layout (`ImportPlan::Kind::Decision`: configuration files, the tokenizer in `tokenizer/`, no chat
  template), the head's blocks mapped back to torch's names. The two models are told apart by the prompt template
  llama.cpp's converter writes for each, rebuilt from the file's [CLS], [SEP] and mask tokens; any other is refused.
  What the file does not hold comes from the published checkpoint it identifies: Laya's state budget (`max_len` 512
  with a question budget of 192, 1,024 with 256; any other budget refused) and Julia-1's inference policy. A Julia-1
  file's `max_head_tokens` is the converter's default, 256 (its `julia_config.json` names none): the pack takes Julia's
  own 512, which under strict encoding only accepts more, encoded the same. The tokenizer table gained `modern-bert`
  and `mmbert`: a BERT-style post-processor over the file's tokens, the mask's lstrip, ModernBERT's normalized plain
  added tokens (its runs of spaces never take the spaces its mask strips), mmBERT's special tokens by name (the
  converter types its unused placeholders control) and its runs of U+2581 (typed normal by the converter, added tokens
  in the original).
- `convert` refuses a q8 / q4 pack of either by name: a pack does not carry the checkpoint's folders. The GGUF files
  are their quantized forms.

## What was measured

Against each model's own Python code in float32 on the CPU (`tests/parity/decision/reference.py`; Julia's encoder
forward written for transformers 5.0 is taken off under 5.20, ModernBERT's own computes the same; its configure()'s
reduced-precision matmuls set back to highest), the engine on cpu-ref at f32 (`decision_parity.cpp`), over
`requests.json` (14 requests, 33 questions) and `requests_laya_julia.json` (16 requests: the budgets, cuts, wordings
and refusals of both). Every sequence the references read is the engine's, to the token, and every refusal agrees.

| Checkpoint | Requests answered | Refused alike | Questions | Largest probability difference |
|---|---|---|---|---|
| tiny Laya (`make_tiny.py`, the released layout) | 28 | 2 | 59 | 2.5e-5 |
| tiny Laya typed-decisions (`requests_laya_julia.json` only) | 15 | 1 | 27 | 1.6e-5 |
| tiny Laya multilingual (mmBERT's tokenizer; the same) | 15 | 1 | 27 | 2.9e-6 |
| tiny Julia-1 | 9 | 21 | 17 | 3.3e-6 |
| `convaiinnovations/laya` | 28 | 2 | 59 | 3.2e-6 |
| `SupersonicLabs/Julia-1` | 9 | 21 | 17 | 4.6e-6 |

Laya's reference rounds its softmax in float32, which is most of what is left. Through the server on cpu-ref, the 25
test decisions (`tests/e2e/decisions25.json`, `systemone_eval.py --reference`): Laya answers 19 right, the same 25
answers as its reference (largest difference 2.0e-6); Julia-1 refuses 8 (choices named without descriptions) as its
reference does and answers 8 of the other 17 right (its literal-noul questions are its weak point, as its card says),
the same 17 as its reference (7.1e-6).

GGUF (`gguf_modern_bert_real_file.cpp`, ggml-org's files): the tokenizers written from `Laya-*.gguf` and
`Julia-1-*.gguf` equal the checkpoints' on every one of 50,368 and 256,000 tokens and on the tokenizer parity cases;
the configuration the import writes is the checkpoint's; over both parity request sets (but the state of 10,000
tokens) each pack reads every request as its checkpoint does, to the token, and refuses the same ones. The answers
move with the file's rounding, on cpu-ref, per question the largest probability difference from the checkpoint:

| File | Questions | Median | Largest | Most likely option differs |
|---|---|---|---|---|
| Laya-BF16 (from f16) | 58 | 0.0008 | 0.019 | 0 |
| Laya-Q8_0 | 58 | 0.0026 | 0.044 | 0 |
| Julia-1-BF16 (from f32) | 17 | 0.0020 | 0.039 | 0 |
| Julia-1-Q8_0 | 17 | 0.015 | 0.24 | 0 |

Julia-1's largest is a noul whose state spells `<bos>`, `<eos>` and `<pad>` (special tokens to its tokenizer, so the
sequence is far from what it was trained on): 0.14 for true from the checkpoint, 0.38 from the Q8_0 file.

The GPU was not measured yet.

## Consequences

- `serve`, `run --request` and `bench` take a Laya or Julia-1 directory, a variant folder of Laya, or the pack of a
  GGUF file, with no flag.
- A Laya answer for a state past 512 tokens reads its first 480 or so, as the reference does; a client that needs the
  whole state read must ask a model that reads it.
- A sequence per question re-reads the state for every question; nothing is shared between questions or requests.
- Not done: batching the questions of a request into one step; q8 / q4 packs of the originals.
