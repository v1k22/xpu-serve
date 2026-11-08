#!/usr/bin/env python3
"""Calibration statistics from engine activations (branch exp/manual-quant): the fast path to what calibrate.py makes.

    python calib_capture.py tokens --model-dir <hf dir> --data <dir> --seqs 56 --len 2048 --out calib_tokens.bin \
        [--code <site-packages>:*.py:28 --code /usr/include:*.h:12]
    xpu-serve quant-capture <pack> --tokens calib_tokens.bin --seq 2048 --out <capture dir>
    python calib_capture.py stats --model-dir <hf dir> --capture <capture dir> --out <stats dir> [--device xpu]

quant-capture writes, per layer, the inputs of its linear maps as f16 [tokens, width]: attn_in.N (input_layernorm
out), moe_in.N (post_attention_layernorm out), attn_out.N (out_proj / o_proj input) and final (the LM head input).
`stats` recomputes the routing and the expert intermediates from the bfloat16 checkpoint and writes the same files
calibrate.py does (layerNN.pt, linear.pt), so quantize.py reads either. `linear` makes linear.pt from the per-weight
captures of policy-eval --capture (any model whose linear maps are plain matmuls).
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import mq

P = "model.language_model."


def code_tokens(tok, root, pattern, seqs, length, seed=0):
    """`seqs` windows of `length` tokens from source files under root (shuffled, concatenated)."""
    files = sorted(str(p) for p in Path(root).rglob(pattern) if p.is_file() and 8 << 10 < p.stat().st_size < 512 << 10)
    random.Random(seed).shuffle(files)
    ids, out = [], []
    for f in files:
        try:
            text = open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        ids += tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        while len(ids) >= length and len(out) < seqs:
            out.append(ids[:length])
            ids = ids[length:]
        if len(out) == seqs:
            break
    return out


def cmd_tokens(args):
    from transformers import AutoTokenizer

    from calibrate import calibration_tokens

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    seqs = calibration_tokens(tok, args.data, args.seqs, args.len)
    for spec in args.code:  # <root>:<glob>:<seqs>
        root, pattern, n = spec.rsplit(":", 2)
        seqs += code_tokens(tok, root, pattern, int(n), args.len)
    np.asarray(seqs, dtype=np.int32).tofile(args.out)
    print(f"wrote {len(seqs)} x {args.len} tokens to {args.out}")


def load_capture(path, width, device):
    a = np.fromfile(path, dtype=np.float16)
    return torch.from_numpy(a.reshape(-1, width)).to(device).float()


def gram(x):
    return x.T @ x


def cmd_stats(args):
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    ckpt = mq.Checkpoint(args.model_dir)
    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    n_layers, top_k = cfg["num_hidden_layers"], cfg["num_experts_per_tok"]
    hidden = cfg["hidden_size"]
    w = lambda n: ckpt.get(n).to(dev).float()
    linear = {}
    tokens = None
    t0 = time.time()
    for li in range(n_layers):
        L = f"model.layers.{li}."
        cap = lambda n, width: load_capture(os.path.join(args.capture, f"{n}.{li}.f16"), width, dev)
        x = cap("attn_in", hidden)
        tokens = x.shape[0]
        h = gram(x).cpu()
        o = load_capture(os.path.join(args.capture, f"attn_out.{li}.f16"), 1, dev).reshape(tokens, -1)
        if f"{P}layers.{li}.linear_attn.out_proj.weight" in ckpt.where:
            for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
                linear[L + "linear_attn." + n] = h
            linear[L + "linear_attn.out_proj"] = gram(o).cpu()
        else:
            for n in ("q_proj", "k_proj", "v_proj"):
                linear[L + "self_attn." + n] = h
            linear[L + "self_attn.o_proj"] = gram(o).cpu()
        del x, o

        y = cap("moe_in", hidden)
        hy = gram(y).cpu()
        for n in ("mlp.gate", "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert_gate"):
            linear[L + n] = hy
        sg, su = w(f"{P}layers.{li}.mlp.shared_expert.gate_proj.weight"), w(f"{P}layers.{li}.mlp.shared_expert.up_proj.weight")
        act = torch.nn.functional.silu(y @ sg.T) * (y @ su.T)
        linear[L + "mlp.shared_expert.down_proj"] = gram(act).cpu()
        del act, sg, su

        # Routing as Qwen3_5MoeTopKRouter: softmax over the experts, top-k, renormalized.
        probs = torch.softmax(y @ w(f"{P}layers.{li}.mlp.gate.weight").T, dim=-1)
        top_w, top_i = probs.topk(top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        gu = ckpt.get(f"{P}layers.{li}.mlp.experts.gate_up_proj").to(dev)  # [E, 2I, H] bf16
        e_count, i2, _ = gu.shape
        st = {"experts_gate_up": torch.zeros(e_count, hidden, hidden), "experts_down": torch.zeros(e_count, i2 // 2, i2 // 2),
              "counts": torch.zeros(e_count), "weight_sum": torch.zeros(e_count)}
        # Group the (token, slot) pairs by expert with one sort (two-index gathers trip an XPU indexing assert).
        flat = top_i.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok_all, rw_all = order // top_k, top_w.reshape(-1)[order]
        ends = torch.bincount(flat, minlength=e_count).cumsum(0).tolist()
        for e in range(e_count):
            lo, hi = (ends[e - 1] if e else 0), ends[e]
            if hi == lo:
                continue
            tok_idx = tok_all[lo:hi]
            xs = y[tok_idx]
            rw = rw_all[lo:hi]
            gate, up = (xs @ gu[e].float().T).chunk(2, dim=-1)
            a = torch.nn.functional.silu(gate) * up
            st["experts_gate_up"][e] = gram(xs * rw[:, None]).cpu()
            st["experts_down"][e] = gram(a * rw[:, None]).cpu()
            st["counts"][e] = len(tok_idx)
            st["weight_sum"][e] = rw.sum().cpu()
        torch.save(st, os.path.join(args.out, f"layer{li:02d}.pt"))
        c = st["counts"]
        print(f"layer {li}: {tokens} tokens, routed per expert min {c.min():.0f} median {c.median():.0f} "
              f"max {c.max():.0f} ({time.time() - t0:.0f} s)", flush=True)
        del y, gu, st, probs
    f = load_capture(os.path.join(args.capture, "final.f16"), hidden, dev)
    linear["lm_head"] = gram(f).cpu()
    torch.save({"linear": linear, "tokens": tokens}, os.path.join(args.out, "linear.pt"))
    print(f"done: {tokens} tokens in {time.time() - t0:.0f} s", flush=True)


def cmd_linear(args):
    """Hessians of every weight from policy-eval --capture files (<weight>.f16, f16 rows of its input), keyed by the
    weight's name."""
    dev = torch.device(args.device)
    ckpt = mq.Checkpoint(args.model_dir)
    linear = {}
    for f in sorted(os.listdir(args.capture)):
        name = f[: -len(".f16")]
        k = ckpt.shape(name)[-1]
        x = load_capture(os.path.join(args.capture, f), k, dev)
        linear[name] = gram(x).cpu()
        print(f"{name}: {x.shape[0]} rows", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save({"linear": linear}, os.path.join(args.out, "linear.pt"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("--model-dir", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--seqs", type=int, default=64)
    t.add_argument("--len", type=int, default=2048)
    t.add_argument("--out", required=True)
    t.add_argument("--code", action="append", default=[],
                   help="<root>:<glob>:<seqs>: also windows of source code (keep the evaluation's files out)")
    s = sub.add_parser("stats")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--capture", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    lin = sub.add_parser("linear")
    lin.add_argument("--model-dir", required=True)
    lin.add_argument("--capture", required=True)
    lin.add_argument("--out", required=True)
    lin.add_argument("--device", default="xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    {"tokens": cmd_tokens, "stats": cmd_stats, "linear": cmd_linear}[args.cmd](args)


if __name__ == "__main__":
    main()
