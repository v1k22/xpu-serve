# Adding a model

CLAUDE.md 8.1 is the checklist. This page is the long form, written while adding Qwen 3 (`qwen3`, `qwen3_moe`) as a
language model of its own: what each step looked like, the commands, and what went wrong on the way. Qwen 3 was the
"add a model in one change" test: its decoder already existed (Qwen3-ASR and Qwen3-Omni's thinker read it), so the
work was everything around a decoder.

What it took:

| Part | Files | Lines |
|---|---|---|
| The model (a `Model` over the decoder, stop tokens, experts joined) | `src/models/qwen3/qwen3_model.cpp`, `qwen3.h` | 110 |
| Registration | `src/models/registry.cpp` | 4 |
| A config key transformers 5 renamed (`num_local_experts`) | `src/models/qwen3/qwen3_config.cpp` | 4 |
| Tiny checkpoints for parity (dense, MoE in the release's layout) | `tests/parity/model/make_tiny.py` | 55 |
| Chat template fixtures and their test | `tests/parity/chat/gen_cases.py`, `tests/fixtures/chat/qwen3*`, `test_chat_template.cpp` | 30 + fixtures |
| A thought the model opens itself before a forced answer (a server change, any model benefits) | `src/chat/`, `src/server/grammar_constraint.*` | 140 |

No op, kernel, tokenizer or parser was new. A model whose layers are new needs those first (CLAUDE.md 8.2).

## 1. Read the model before writing code

- `config.json`: the keys the model reads. Note the ones whose values change the math (RoPE type and scaling, sliding
  windows, biases, tied embeddings, norms with a `1 +`, shared experts). Each one you do not implement is refused by
  name in `parse_config`, never served silently wrong.
- The modeling file in transformers (`transformers/models/<arch>/modeling_<arch>.py`): attention, norms, MLP, where
  the residuals are.
- The weight names and shapes, from the safetensors headers alone:

  ```bash
  python - <<'PY'
  import json, struct, glob
  for f in sorted(glob.glob("<dir>/*.safetensors")):
      with open(f, "rb") as fh:
          n = struct.unpack("<Q", fh.read(8))[0]
          for k, v in json.loads(fh.read(n)).items():
              if k != "__metadata__" and ".layers.1." in k: print(k, v["dtype"], v["shape"])
  PY
  ```

  Qwen3-30B-A3B stores each expert's `gate_proj`, `up_proj` and `down_proj` apart; transformers 5 saves the same model
  as fused banks (`experts.gate_up_proj`). The loader takes both (`join_expert_banks`, `models/common/expert_banks.h`).

Then find the closest model already here. `models/lfm3/` is the shortest whole model; `models/qwen3/qwen3_model.cpp`
is shorter still because its layers live in a shared `Decoder`.

## 2. The model file

A `Model` (`src/models/model.h`) gives:

- `info()`: `arch`, `id`, `vocab_size`, `hidden_size`, `max_position`, the chat `dialect`, `stop_token_ids` and
  `default_sampling` (`cfg::read_generation_defaults` reads `generation_config.json`).
- `graph()`: the step graph, inputs `token_ids` and `positions`, outputs `logits` (the last row) and `argmax`
  (`blocks::argmax_rows`). Greedy decoding reads `argmax` and never copies the logits back.
- `batch_graph()`: the same graph with logits for every row (`g.allow_segments()`). Build it when
  `LoadOptions::batch` asks: batched decoding, prompt lookup (`--spec ngram`) and several sequences at once use it.
- `weights()`, and `prepare_inputs()` filling every graph input.

Stop tokens: take every id `generation_config.json` lists, then `config.json`'s `eos_token_id`, then the tokenizer's
eos (`cfg::tokenizer_eos_id`), sorted and deduplicated. Qwen 3's generation config lists `<|im_end|>` and
`<|endoftext|>`; a model that stops only on one of them runs on past its turn.

Refuse what the model does not have by name: Qwen 3 refuses `--drafter` (no drafter is published for it), and a
`qwen3_moe` config without experts.

## 3. Register it

`src/models/registry.cpp`: the `model_type` in `supported_model_types()` and a line in `load_model`. Nothing else
lists models: the CLI, `convert`, the server and `-hf` all go through the registry.

## 4. Parity against Hugging Face

Two levels, both with `tests/unit/model/model_parity.cpp` (model-agnostic):

1. A tiny checkpoint (`make_tiny.py --arch <arch>`), seconds to run, on cpu-ref and the GPU at f32. Give it the real
   model's ratios (query heads per KV head, head_dim against hidden size), an untied head if the release has one,
   norms away from 1 (a norm the model forgets to apply then shows), and for a mixture of experts decisive routers
   (two experts close in score flip on rounding). Save it in the release's layout, not transformers': the
   per-expert split is what the loader meets in the wild.
2. The smallest real checkpoint, f16 and f32 on the GPU. Greedy text must match token for token (32 at f32, 8 at f16).

```bash
PY=<venv with transformers and torch>
$PY -I tests/parity/model/make_tiny.py --arch qwen3_moe --tokenizer-from <Qwen3-0.6B dir> --out /tmp/tiny_qwen3_moe
$PY -I tests/parity/model/hf_reference.py --model-dir /tmp/tiny_qwen3_moe --out /tmp/fx_qwen3_moe --gen 24
XS_TEST_MODEL_DIR=/tmp/tiny_qwen3_moe XS_PARITY_FIXTURES=/tmp/fx_qwen3_moe XS_PARITY_BACKEND=sycl \
  XS_PARITY_DEVICE=GPU.1 XS_PARITY_PRECISION=f32 build/tests/xs_tests -tc='model matches*'
```

The cpu-ref backend takes `XS_PARITY_DEVICE=CPU` (it refuses a GPU name). What went wrong for Qwen 3: transformers 5
writes `num_local_experts` where the released configs say `num_experts`, so the first tiny MoE was refused as having
no experts. Read both.

Results: the tiny dense and MoE models match at cosine 1 (max |diff| 2e-6) on both backends; Qwen3-0.6B matches 32 of
32 greedy tokens on all six prompts at f16 and f32.

## 5. The chat template

`ChatTemplate` renders the checkpoint's own Jinja template. Check it renders byte for byte what transformers renders:

```bash
$PY -I tests/parity/chat/gen_cases.py qwen3=<Qwen3-0.6B dir>   # writes tests/fixtures/chat/qwen3_cases.json
```

Copy the template's file into `tests/fixtures/chat/<set>/` with a `SOURCE` line (repository, license, revision), and
add a test case to `tests/unit/chat/test_chat_template.cpp`. A case the template itself fails on (Qwen 3's template
does not take content as a list of parts, nor a tool call without arguments) is skipped by `gen_cases.py` and
printed; check each skip is the template's limit and not the case's mistake.

## 6. The output dialect

`ModelInfo::dialect` names the parser that splits the output into thinking, content and tool calls
(`src/chat/`). Reuse one when the syntax matches: Qwen 3 writes `<think>` and JSON calls in `<tool_call>` blocks,
which the `qwen3` dialect already read (Qwen3-Omni). A new syntax needs a parser
(`src/chat/<dialect>_parser.cpp`, registered in `make_output_parser`), its calls in `server/tool_grammar.cpp`
(`tool_choice` required or named holds the model to the dialect's syntax) and its start in `answer_start`.

Who opens the thought matters. Qwen 3.5's template ends the prompt inside `<think>\n`, so the parser starts in
reasoning and a forced answer waits for `</think>`. Qwen 3's template ends the prompt at `<|im_start|>assistant\n`
and the model writes `<think>` itself; with nothing to wait for, `tool_choice: required` forced the call at once and
Qwen3-0.6B, kept from thinking, called the same tool 14 times. The fix is general: the parser reports a thought the
model may open (`OutputParser::own_thought`), and the constraint (`AnswerGate`) lets the opener through beside the
answer's first tokens, then waits for the closer. After it, the same request thinks for 431 characters and makes one
call. vLLM's Qwen 3 parser does the same (it starts in reasoning and waits for `</think>` when thinking is on).

## 7. On the GPU, through the server

- Tool calls and formats: `tool_choice` required (thinking on and off), named, auto, auto beside a `json_schema`,
  parallel and not, streamed, Anthropic's `any`. Every call parsed, its arguments of the schema.
- Packs: `xpu-serve convert <dir> --out <pack> --quant q8|q4` needs nothing from the model; it reads the graph to find
  the matmul weights. Check each pack against the original's fp32 reference with the same parity test
  (`XS_TEST_MODEL_DIR=<pack>`).
- Speed: `xpu-serve bench <model> --device GPU.1 --prompt-tokens 4096 --gen-tokens 128 --runs 2`.

## Llama 3: the second model, with less to reuse

Llama 3 had no decoder here, so it took the parts Qwen 3 found ready:

| Part | Where | Lines |
|---|---|---|
| Config, model, graph (layers from `blocks::attention` and `blocks::gated_mlp`) | `src/models/llama/` | 260 |
| Llama 3's RoPE scaling (frequencies in three bands) | `blocks::llama3_rope_inv_freq` | 30 |
| A post-processor `Sequence` (ByteLevel, then TemplateProcessing adding `<|begin_of_text|>`) | `src/tokenizer/post_processor.cpp` | 15 |
| The `llama3` dialect: JSON calls with no marker before them | `src/chat/llama3_parser.cpp` | 155 |
| Its forced calls (`{"name": ..., "parameters": ...}`, `"; "` between) | `src/server/tool_grammar.cpp` | 20 |
| Whether the request offers tools, for the parser | `make_output_parser(..., tools)` | 15 |

What was different:

- The tokenizer: run its parity first (`gen_llama3_cases.py`, `XS_LLAMA3_DIR`). The Split regex and `ignore_merges`
  were there from LFM3; the post-processor was not, and only `with_special_tokens` (the begin-of-text token) showed it.
- The chat template writes today's date (`strftime_now`): the fixtures pass `date_string`, and one test checks today's
  date is rendered without it.
- The calls are bare JSON. With markers a call cannot be mistaken for an answer; here a JSON answer and a call look
  alike, so the parser reads calls only when the request offers tools (vLLM runs no tool parser without them), and
  only from the reply's first character, as vLLM does. A forced call's grammar starts the reply with `{` for that
  reason.
- Gated checkpoints: Unsloth's copies (`unsloth/Llama-3.2-1B-Instruct`) are the same files without the gate; the
  fixtures name the copy they came from.

Results: the tiny checkpoint at cosine 1 (cpu-ref and the GPU, f32, a rope_parameters config as transformers 5 writes
it); Llama-3.2-1B and Llama-3.1-8B (the released config layout) equal to transformers on all six prompts at f16.

## 8. Write it down

- README: the model table (name, what it does, repositories, forms).
- model_recipes.md: the reference and optimized commands, what was measured, what it was checked against.
- TODO.md: tick the item; add what was left out (Qwen 3's GGUF files are another architecture of the importer).
- CLAUDE.md: the map if a file or directory is new, and a test table row if a test file is.
