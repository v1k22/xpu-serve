#!/usr/bin/env python3
"""Reference outputs for Gemma 4 text models from Hugging Face transformers (test-only).

    python -I tests/parity/model/hf_reference.py --model-dir <hf dir> [--out tests/fixtures/generated/gemma4] [--gen 32]

Runs the checkpoint in float32 on the CPU with eager attention and writes, per prompt:
  * the prompt token ids,
  * the full-vocabulary logits of the last prompt position (float32, <name>.last_logits.bin),
  * the top-5 token ids at every prompt position,
  * the greedy continuation (--gen tokens, stop tokens included if hit).

One deliberate deviation from a plain float32 load: the main embedding scale sqrt(hidden_size) is rounded to bfloat16,
as the checkpoint's own dtype does in transformers (and in the original JAX code). Without it a float32 load would
use 39.19 instead of 39.25 for E2B and every comparison would carry that 0.15% error.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

CHAT_PROMPTS = {
    "chat_short": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    "chat_system": [
        {"role": "system", "content": "You are a terse assistant who answers in exactly one sentence."},
        {"role": "user", "content": "Explain what a KV cache is in a transformer."},
    ],
    "chat_code": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}],
    "chat_hindi": [{"role": "user", "content": "नमस्ते! भारत की राजधानी क्या है?"}],
}

# Longer than the 512-token sliding window, so sliding layers must mask and shared layers must reuse KV correctly.
LONG_TEXT = " ".join(
    f"Paragraph {i}: the engine reads weights, writes keys and values, and attends over a window of recent tokens."
    for i in range(70)
)


def build_prompts(tok):
    prompts = {}
    for name, messages in CHAT_PROMPTS.items():
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts[name] = tok(text, add_special_tokens=False)["input_ids"]
    long_messages = [{"role": "user", "content": LONG_TEXT + "\n\nSummarize the text above in one sentence."}]
    text = tok.apply_chat_template(long_messages, tokenize=False, add_generation_prompt=True)
    prompts["chat_long"] = tok(text, add_special_tokens=False)["input_ids"]
    prompts["raw_text"] = tok("<bos>The theory of relativity", add_special_tokens=False)["input_ids"]
    return prompts


def bf16(x):
    return torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default=os.path.join("tests", "fixtures", "generated", "gemma4"))
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    lm = model.model.language_model
    hidden = model.config.text_config.hidden_size
    lm.embed_tokens.embed_scale.fill_(bf16(math.sqrt(hidden)))
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]

    manifest = {"model_dir": os.path.abspath(args.model_dir), "dtype": "float32", "cases": []}
    for name, ids in build_prompts(tok).items():
        input_ids = torch.tensor([ids])
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0].float()
            out = model.generate(input_ids=input_ids, max_new_tokens=args.gen, do_sample=False, eos_token_id=eos)
        greedy = out[0, len(ids):].tolist()
        last = logits[-1].numpy().astype(np.float32)
        last.tofile(os.path.join(args.out, f"{name}.last_logits.bin"))
        top5 = torch.topk(logits, 5, dim=-1).indices.tolist()
        manifest["cases"].append({
            "name": name,
            "input_ids": ids,
            "last_logits": f"{name}.last_logits.bin",
            "top5": top5,
            "greedy": greedy,
            "greedy_text": tok.decode(greedy),
        })
        print(f"{name}: {len(ids)} prompt tokens -> {tok.decode(greedy)!r}", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
