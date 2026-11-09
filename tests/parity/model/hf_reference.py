#!/usr/bin/env python3
"""Reference outputs for text models from Hugging Face transformers (test-only).

    python -I tests/parity/model/hf_reference.py --model-dir <hf dir> [--out tests/fixtures/generated/<model_type>]
        [--gen 32]

Runs the checkpoint in float32 on the CPU with eager attention and writes, per prompt:
  * the prompt token ids,
  * the full-vocabulary logits of the last prompt position (float32, <name>.last_logits.bin),
  * the top-5 token ids at every prompt position,
  * the greedy continuation (--gen tokens, stop tokens included if hit),
  * with an MTP head (Qwen3.5 mtp.*, wired as vLLM's Qwen3.5 MTP): its logits after the prompt's pairs and the first
    greedy token (<name>.mtp_logits.bin) and the 3 drafts it chains from there;
  * with --assistant <dir> (a Gemma 4 assistant checkpoint): the same for the assistant, as transformers'
    SinglePositionMultiTokenCandidateGenerator drafts after the prompt.

One deliberate deviation from a plain float32 load, for Gemma 4 only: the main embedding scale sqrt(hidden_size) is
rounded to bfloat16, as the checkpoint's own dtype does in transformers (and in the original JAX code). Without it a
float32 load would use 39.19 instead of 39.25 for E2B and every comparison would carry that 0.15% error.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

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
    prompts["raw_text"] = tok((tok.bos_token or "") + "The theory of relativity", add_special_tokens=False)["input_ids"]
    return prompts


def load_mtp(model, model_dir):
    """vLLM's Qwen3.5 MTP head from the checkpoint's mtp.* weights, built from the target's own layer classes, as a
    function (next tokens [T], target hidden [T, H], positions [T]) -> (normed outputs [T, H], last-row logits);
    None when the checkpoint has no head."""
    import torch.nn as nn
    from safetensors import safe_open

    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    sd = {}
    for file in files:
        with safe_open(os.path.join(model_dir, file), "pt") as f:
            for name in f.keys():
                if name.startswith("mtp."):
                    sd[name[len("mtp."):]] = f.get_tensor(name).float()
    if not sd:
        return None
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    hidden = cfg.hidden_size
    norm_cls = type(lm.norm)
    head = nn.ModuleDict({
        "layers": nn.ModuleList([type(lm.layers[0])(cfg, cfg.layer_types.index("full_attention"))]),
        "fc": nn.Linear(2 * hidden, hidden, bias=False),
        "pre_fc_norm_embedding": norm_cls(hidden, eps=cfg.rms_norm_eps),
        "pre_fc_norm_hidden": norm_cls(hidden, eps=cfg.rms_norm_eps),
        "norm": norm_cls(hidden, eps=cfg.rms_norm_eps),
    }).float().eval()
    head.load_state_dict(sd, strict=True)

    def forward(tokens, hidden_rows, positions):
        x = torch.cat([head["pre_fc_norm_embedding"](lm.embed_tokens(tokens)[None]),
                       head["pre_fc_norm_hidden"](hidden_rows[None])], dim=-1)
        x = head["fc"](x)
        t = tokens.shape[0]
        mask = torch.full((t, t), float("-inf")).triu(1)[None, None]  # causal (eager attention takes no default)
        pe = lm.rotary_emb(x, positions.view(1, 1, -1).expand(3, 1, -1))
        x = head["layers"][0](x, position_embeddings=pe, attention_mask=mask, position_ids=positions[None])
        out = head["norm"](x)[0]
        return out, model.lm_head(out[-1:]).float()[0]

    return lm, forward


def mtp_case(lm, mtp, ids, first, k=3):
    """The head over the prompt's pairs (h_i, token i + 1) with the first greedy token, then k chained drafts."""
    with torch.no_grad():
        hidden = lm(input_ids=torch.tensor([ids])).last_hidden_state[0]  # post-norm
        tokens = torch.tensor(ids[1:] + [first])
        positions = torch.arange(len(ids))
        drafts, logits0 = [], None
        for s in range(k):
            out, logits = mtp(tokens, hidden, positions)
            logits0 = logits if logits0 is None else logits0
            drafts.append(int(logits.argmax()))
            tokens = torch.cat([tokens, torch.tensor([drafts[-1]])])
            hidden = torch.cat([hidden, out[-1:]])
            positions = torch.cat([positions, torch.tensor([len(ids) + s])])
    return logits0.numpy().astype(np.float32), drafts


def assistant_case(model, assistant, ids, first, k=3):
    """The assistant's first drafting round after the prompt: (the token after the prompt, the target's last hidden
    state) at the position of that token, reading the target's KV, k drafts chained on its own projected state."""
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True, return_shared_kv_states=True)
        hidden = out.hidden_states[-1][:, -1:]
        kv = out.shared_kv_states
        token = torch.tensor([[first]])
        position = torch.tensor([[len(ids)]])
        drafts, logits0 = [], None
        for _ in range(k):
            embeds = torch.cat([model.get_input_embeddings()(token), hidden], dim=-1)
            a = assistant(inputs_embeds=embeds, position_ids=position, shared_kv_states=kv, use_cache=False)
            logits = a.logits[0, -1].float()
            logits0 = logits if logits0 is None else logits0
            token = logits.argmax().view(1, 1)
            drafts.append(int(token))
            hidden = a.last_hidden_state
    return logits0.numpy().astype(np.float32), drafts


def bf16(x):
    return torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", help="default: tests/fixtures/generated/<model_type>")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--assistant", help="a Gemma 4 assistant checkpoint to draft with")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    model_type = config["model_type"]
    args.out = args.out or os.path.join("tests", "fixtures", "generated", model_type)
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    auto = AutoModelForImageTextToText if "vision_config" in config else AutoModelForCausalLM  # text-only: tiny models
    model = auto.from_pretrained(args.model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    if model_type.startswith("gemma4"):
        lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
        lm.embed_tokens.embed_scale.fill_(bf16(math.sqrt(model.config.get_text_config().hidden_size)))
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]

    mtp = load_mtp(model, args.model_dir)
    assistant = None
    if args.assistant:
        assistant = AutoModelForCausalLM.from_pretrained(args.assistant, dtype=torch.float32,
                                                         attn_implementation="eager").eval()
    manifest = {"model_dir": os.path.abspath(args.model_dir), "model_type": model_type, "dtype": "float32", "cases": []}
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
        if mtp is not None:
            mtp_logits, drafts = mtp_case(*mtp, ids, greedy[0])
            mtp_logits.tofile(os.path.join(args.out, f"{name}.mtp_logits.bin"))
            manifest["cases"][-1].update({"mtp_logits": f"{name}.mtp_logits.bin", "mtp_drafts": drafts})
        if assistant is not None:
            draft_logits, drafts = assistant_case(model, assistant, ids, greedy[0])
            draft_logits.tofile(os.path.join(args.out, f"{name}.mtp_logits.bin"))
            manifest["cases"][-1].update({"mtp_logits": f"{name}.mtp_logits.bin", "mtp_drafts": drafts})
        print(f"{name}: {len(ids)} prompt tokens -> {tok.decode(greedy)!r}", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
