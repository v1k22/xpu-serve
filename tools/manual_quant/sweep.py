#!/usr/bin/env python3
"""Quantizes the matching tensors in several formats at once (branch exp/manual-quant).

    python sweep.py --model-dir <hf dir> --stats <stats dir> --match <regex> --configs q2g64,q3g64,q4g64 --out <dir>

Each format's tensors go to <out>/<format>/, an overlay pack (XS_PACK_OVERLAY), and <out>/<format>/errors.json holds
the output error of every tensor (per expert for expert banks): tr(dW H dW^T) next to the output energy tr(W H W^T),
both summed over the calibration tokens. GPTQ's Cholesky factor depends only on H, so it is computed once per
tensor (per batch of experts) and shared by all formats. assemble.py builds a pack from a per-tensor choice.

Rarely routed experts have few calibration tokens (late layers have experts with none), so each expert's Hessian gets
a prior worth --prior-tokens tokens: the second moment of its inputs over a random sample of all the layer's tokens
(for down_proj, the expert's own intermediate activations on that sample), scaled by the mean squared routing weight.
"""

import numpy as np

import argparse
import concurrent.futures
import json
import os
import re
import time

import torch

import mq
from quantize import EXPERT, module_key


def parse_config(c):
    m = re.fullmatch(r"q(\d)g(\d+)", c)
    return int(m.group(1)), int(m.group(2))


def sample_inputs(capture, li, width, n=8192, seed=0):
    a = np.memmap(os.path.join(capture, f"moe_in.{li}.f16"), dtype=np.float16, mode="r").reshape(-1, width)
    idx = np.sort(np.random.default_rng(seed + li).choice(a.shape[0], n, replace=False))
    return torch.from_numpy(np.asarray(a[idx], dtype=np.float32))


def with_prior(h, st, bank, ckpt, name, tokens, dev):
    """h + tokens * mean(w^2) * E[x x^T] over the sample, per expert (for down_proj x = act(gate) * up of the expert)."""
    mean_w = (st["weight_sum"].sum() / st["counts"].sum()).item()
    scale = tokens * mean_w ** 2
    x = st["sample"].to(dev)
    if bank == "gate_up_proj":
        return h + (scale / len(x)) * (x.T @ x).cpu().unsqueeze(0)
    gu = ckpt.get(name.replace("down_proj", "gate_up_proj")).to(dev)
    out = h.clone()
    for e in range(gu.shape[0]):
        gate, up = (x @ gu[e].float().T).chunk(2, dim=-1)
        a = torch.nn.functional.silu(gate) * up
        out[e] += (scale / len(x)) * (a.T @ a).cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64, help="experts per GPTQ batch")
    ap.add_argument("--capture", help="quant-capture dir (moe_in) for the expert Hessian prior")
    ap.add_argument("--prior-tokens", type=float, default=32)
    ap.add_argument("--device", default="xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    dev = torch.device(args.device)
    configs = args.configs.split(",")
    ckpt = mq.Checkpoint(args.model_dir)
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    writers = {c: mq.PackWriter(os.path.join(args.out, c), os.path.basename(os.path.normpath(args.model_dir)),
                                config.get("model_type", "")) for c in configs}
    errors = {c: {} for c in configs}
    linear = None
    layer = (None, None)
    pool = concurrent.futures.ThreadPoolExecutor(1)
    t0 = time.time()
    for name in sorted(n for n in ckpt.names() if re.search(args.match, n)):
        t1 = time.time()
        w = ckpt.get(name).to(dev).float()
        shape = list(w.shape)
        k = shape[-1]
        m = EXPERT.search(name)
        if m:
            li = int(m.group(1))
            if layer[0] != li:
                layer = (li, None)
                layer = (li, torch.load(os.path.join(args.stats, f"layer{li:02d}.pt")))
                if args.capture:
                    layer[1]["sample"] = sample_inputs(args.capture, li, config.get("text_config", config)["hidden_size"])
            st = layer[1]
            h = st["experts_gate_up" if m.group(2) == "gate_up_proj" else "experts_down"]
            if args.capture:
                h = with_prior(h, st, m.group(2), ckpt, name, args.prior_tokens, dev)
            batch = args.batch
        else:
            if linear is None:
                linear = torch.load(os.path.join(args.stats, "linear.pt"))["linear"]
            h = (linear[name] if name in linear else linear[module_key(name)]).unsqueeze(0)
            w = w.reshape(1, -1, k)
            batch = 1
        out = {c: ([], [], [], [], []) for c in configs}  # codes, scales, biases, err, energy
        # The Cholesky factorizations run on the CPU (no XPU kernel), one batch ahead of the GPU work.
        w_cpu, h_cpu = w.cpu(), h.cpu()
        starts = list(range(0, w.shape[0], batch))
        prep = lambda b0: mq.gptq_prepare(w_cpu[b0:b0 + batch], h_cpu[b0:b0 + batch])
        pending = pool.submit(prep, starts[0])
        for j, b0 in enumerate(starts):
            wp, hinv, diag = (t.to(dev) for t in pending.result())
            if j + 1 < len(starts):
                pending = pool.submit(prep, starts[j + 1])
            wb, hb = w[b0:b0 + batch], h[b0:b0 + batch].to(dev)
            for c in configs:
                bits, group = parse_config(c)
                codes, scales, biases = mq.gptq_run(wp, hinv, diag, bits, group)
                deq = mq.dequant(codes.reshape(-1, k), scales.reshape(-1, k // group),
                                 None if biases is None else biases.reshape(-1, k // group), bits, group).reshape(wb.shape)
                err, energy = mq.output_error(wb, deq, hb)
                o = out[c]
                o[0].append(codes.cpu()), o[1].append(scales.cpu()), o[2].append(None if biases is None else biases.cpu())
                o[3].append(err.cpu()), o[4].append(energy.cpu())
                del codes, scales, biases, deq
            del wp, hinv, diag, hb
        line = []
        for c in configs:
            bits, group = parse_config(c)
            codes, scales, biases, err, energy = out[c]
            rows = w.numel() // k
            writers[c].quantized(name, shape, bits, group, torch.cat(codes).reshape(rows, k),
                                 torch.cat(scales).reshape(rows, -1),
                                 None if bits == 8 else torch.cat(biases).reshape(rows, -1))
            err, energy = torch.cat(err), torch.cat(energy)
            errors[c][name] = {"err": err.tolist(), "energy": energy.tolist()}
            line.append(f"{c} {err.sum() / energy.sum():.2e}")
        print(f"{name}: {', '.join(line)} ({time.time() - t1:.0f} s)", flush=True)
    for c in configs:
        total = writers[c].close()
        json.dump(errors[c], open(os.path.join(args.out, c, "errors.json"), "w"))
        print(f"{c}: {total / 2**30:.2f} GiB", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
