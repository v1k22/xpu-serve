#!/usr/bin/env python3
"""Calibration statistics for the manual-quantization experiment (branch exp/manual-quant).

    python calibrate.py --model-dir <hf dir> --data <dir with c4_val0.json.gz> --out <dir> [--seqs 16 --len 2048]

Runs the bfloat16 checkpoint on the CPU over C4 validation text and writes, per decoder layer, the second moments the
quantizer needs (float32, summed over tokens):
  experts_gate_up [E, H, H]  sum over each expert's routed tokens of (w x)(w x)^T, w the token's routing weight
  experts_down    [E, I, I]  the same over the expert's intermediate activations act(gate) * up
  counts / weight_sum [E]    tokens routed to each expert and the sum of their routing weights
and for every other linear layer (attention, linear attention, shared expert, LM head) sum x x^T of its input, keyed
by module name. Routing weights scale an expert's output, so weighting its inputs by them makes the quantizer
minimize the expert's contribution to the layer output rather than its raw output.
"""

import argparse
import gzip
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def calibration_tokens(tok, data_dir, seqs, length, seed=0):
    g = torch.Generator().manual_seed(seed)
    docs = [json.loads(l)["text"] for l in gzip.open(os.path.join(data_dir, "c4_val0.json.gz"), "rt")]
    order = torch.randperm(len(docs), generator=g).tolist()
    ids, out = [], []
    for i in order:
        ids += tok(docs[i], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        while len(ids) >= length and len(out) < seqs:
            out.append(ids[:length])
            ids = ids[length:]
        if len(out) == seqs:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--len", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    seqs = calibration_tokens(tok, args.data, args.seqs, args.len)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16)
    model.eval()
    print(f"loaded in {time.time() - t0:.0f} s; {len(seqs)} x {args.len} tokens", flush=True)

    layers = model.model.layers
    n_layers = len(layers)
    stats = []
    for li, layer in enumerate(layers):
        ex = layer.mlp.experts
        e, i2, h = ex.gate_up_proj.shape
        stats.append({"experts_gate_up": torch.zeros(e, h, h), "experts_down": torch.zeros(e, i2 // 2, i2 // 2),
                      "counts": torch.zeros(e), "weight_sum": torch.zeros(e)})

        def experts_forward(self, hidden_states, top_k_index, top_k_weights, _li=li):
            st = stats[_li]
            out = torch.zeros_like(hidden_states)
            x32 = hidden_states.float()
            for expert in torch.unique(top_k_index).tolist():
                tok_idx, pos = torch.where(top_k_index == expert)
                xs = hidden_states[tok_idx]
                w = top_k_weights[tok_idx, pos].float()
                gate, up = torch.nn.functional.linear(xs, self.gate_up_proj[expert]).chunk(2, dim=-1)
                act = self.act_fn(gate) * up
                y = torch.nn.functional.linear(act, self.down_proj[expert])
                out.index_add_(0, tok_idx, (y * top_k_weights[tok_idx, pos, None]).to(out.dtype))
                xw = x32[tok_idx] * w[:, None]
                st["experts_gate_up"][expert] += xw.T @ xw
                aw = act.float() * w[:, None]
                st["experts_down"][expert] += aw.T @ aw
                st["counts"][expert] += len(tok_idx)
                st["weight_sum"][expert] += w.sum()
            return out

        ex.forward = experts_forward.__get__(ex)

    linear_h = {}

    def hook(name):
        def f(module, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            if name not in linear_h:
                linear_h[name] = torch.zeros(x.shape[1], x.shape[1])
            linear_h[name] += x.T @ x
        return f

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "visual" not in name and "mtp" not in name and name != "lm_head":
            module.register_forward_pre_hook(hook(name))
    # The LM head runs on the last position only (logits_to_keep): take its input at every position from the norm.
    model.model.norm.register_forward_hook(lambda m, inp, out: hook("lm_head")(m, (out,)))

    tokens = 0
    with torch.no_grad():
        for i, s in enumerate(seqs):
            t1 = time.time()
            model(input_ids=torch.tensor([s]), logits_to_keep=1)
            tokens += len(s)
            print(f"seq {i}: {time.time() - t1:.0f} s", flush=True)
    for li in range(n_layers):
        torch.save(stats[li], os.path.join(args.out, f"layer{li:02d}.pt"))
    torch.save({"linear": linear_h, "tokens": tokens}, os.path.join(args.out, "linear.pt"))
    print(f"done: {tokens} tokens in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
