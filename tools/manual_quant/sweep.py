#!/usr/bin/env python3
"""Quantizes the matching tensors in several formats at once (branch exp/manual-quant).

    python sweep.py --model-dir <hf dir> --stats <stats dir> --match <regex> --configs q2g64,q3g64,q4g64 --out <dir>

Each format's tensors go to <out>/<format>/, an overlay pack (XS_PACK_OVERLAY), and <out>/<format>/errors.json holds
the output error of every tensor (per expert for expert banks): tr(dW H dW^T) next to the output energy tr(W H W^T),
both summed over the calibration tokens. GPTQ's Cholesky factor depends only on H, so it is computed once per
tensor (per batch of experts) and shared by all formats. assemble.py builds a pack from a per-tensor choice.
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64, help="experts per GPTQ batch")
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
            h = layer[1]["experts_gate_up" if m.group(2) == "gate_up_proj" else "experts_down"]
            batch = args.batch
        else:
            if linear is None:
                linear = torch.load(os.path.join(args.stats, "linear.pt"))["linear"]
            h = linear[module_key(name)].unsqueeze(0)
            w = w.reshape(1, -1, k)
            batch = 1
        out = {c: ([], [], [], [], []) for c in configs}  # codes, scales, biases, err, energy
        for b0 in range(0, w.shape[0], batch):
            wb, hb = w[b0:b0 + batch], h[b0:b0 + batch].to(dev)
            wp, hinv, diag = mq.gptq_prepare(wb, hb)
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
