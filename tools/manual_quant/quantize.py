#!/usr/bin/env python3
"""Writes a pack from a plan (branch exp/manual-quant).

    python quantize.py --model-dir <hf dir> --stats <calibrate.py out> --plan plan.json --out <dir> [--overlay]

plan.json: {"rules": [{"match": "<regex>", "bits": 3, "group": 64, "method": "gptq" | "rtn"}, ...]}: the first rule
whose regex matches a tensor name decides its format; tensors no rule matches are copied as they are (with
--overlay they are left out, and the pack overrides only the tensors it holds: XS_PACK_OVERLAY). GPTQ uses the
calibration second moments: per expert (router-weighted) for expert banks, per input for other linear layers; RTN
weights its clip search by their diagonal when they exist.
"""

import argparse
import json
import os
import re
import shutil
import time

import torch

import mq

EXPERT = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")


def module_key(name):
    """Checkpoint tensor name -> the transformers module name calibrate.py keyed its statistics by."""
    n = name.replace("model.language_model.", "model.")
    return n[: -len(".weight")] if n.endswith(".weight") else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--only", default="", help="regex: quantize only matching tensors (debugging)")
    ap.add_argument("--device", default="xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    rules = json.load(open(args.plan))["rules"]
    ckpt = mq.Checkpoint(args.model_dir)
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    writer = mq.PackWriter(args.out, os.path.basename(os.path.normpath(args.model_dir)), config.get("model_type", ""))
    linear = torch.load(os.path.join(args.stats, "linear.pt"))["linear"] if os.path.exists(
        os.path.join(args.stats, "linear.pt")) else {}
    layer_stats = {}
    report = {"bits": {}, "err": {}, "bytes": 0}
    dev = torch.device(args.device)
    t0 = time.time()
    for name in sorted(ckpt.names()):
        rule = next((r for r in rules if re.search(r["match"], name)), None)
        if args.only and not re.search(args.only, name):
            rule = None if args.overlay else rule
        if rule is None or rule.get("bits") is None:
            if not args.overlay:
                writer.copy(name, ckpt.get(name))
            continue
        bits, group, method = rule["bits"], rule["group"], rule.get("method", "rtn")
        w = ckpt.get(name).to(dev).float()
        shape = list(w.shape)
        t1 = time.time()
        m = EXPERT.search(name)
        if m:
            li = int(m.group(1))
            if li not in layer_stats:
                layer_stats.clear()
                p = os.path.join(args.stats, f"layer{li:02d}.pt")
                layer_stats[li] = torch.load(p) if os.path.exists(p) else None
            st = layer_stats[li]
            h = None if st is None else st["experts_gate_up" if m.group(2) == "gate_up_proj" else "experts_down"]
            batch = 64 if dev.type != "cpu" else 32
            e, r, k = w.shape
            if method == "gptq" and h is not None:
                codes, scales, biases = [], [], []
                for b0 in range(0, e, batch):  # batches of experts bound the working memory
                    c, s, bb = mq.gptq(w[b0:b0 + batch], h[b0:b0 + batch].to(dev), bits, group)
                    codes.append(c), scales.append(s), biases.append(bb)
                codes = torch.cat(codes).reshape(e * r, k)
                scales = torch.cat(scales).reshape(e * r, -1)
                biases = None if bits == 8 else torch.cat(biases).reshape(e * r, -1)
            else:
                cw = None if h is None else torch.diagonal(h, dim1=1, dim2=2).to(dev).unsqueeze(1).expand(e, r, k).reshape(e * r, k)
                codes, scales, biases = mq.quantize_rtn(w.reshape(e * r, k), bits, group, col_weight=cw)
        else:
            k = shape[-1]
            w2 = w.reshape(-1, k)
            h = linear.get(module_key(name))
            h = None if h is None else h.to(dev)
            if method == "gptq" and h is not None:
                codes, scales, biases = mq.gptq(w2, h, bits, group)
            else:
                cw = None if h is None else torch.diagonal(h).unsqueeze(0).expand_as(w2)
                codes, scales, biases = mq.quantize_rtn(w2, bits, group, col_weight=cw)
        if h is not None:  # output error per expert (or one value), from the dequantized weights
            hb = h.to(dev) if h.dim() == 3 else h.to(dev).unsqueeze(0)
            wb = w.reshape(hb.shape[0], -1, k)
            deq = mq.dequant(codes, scales, biases, bits, group).reshape(wb.shape)
            err, energy = zip(*(mq.output_error(wb[i:i + 32], deq[i:i + 32], hb[i:i + 32]) for i in range(0, len(hb), 32)))
            err, energy = torch.cat(err).cpu(), torch.cat(energy).cpu()
            report["err"][name] = {"err": err.tolist(), "energy": energy.tolist()}
            del deq, hb
        writer.quantized(name, shape, bits, group, codes.cpu(), scales.cpu(), None if biases is None else biases.cpu())
        report["bits"][name] = f"q{bits}g{group}/{method}"
        rel = sum(report["err"][name]["err"]) / max(sum(report["err"][name]["energy"]), 1e-30) if name in report["err"] else float("nan")
        print(f"{name}: q{bits} g{group} {method}, relative output error {rel:.2e} ({time.time() - t1:.0f} s)", flush=True)
    total = writer.close()
    if not args.overlay:
        for f in os.listdir(args.model_dir):
            if not f.endswith(".safetensors") and f != "model.safetensors.index.json" and os.path.isfile(
                    os.path.join(args.model_dir, f)):
                shutil.copy(os.path.join(args.model_dir, f), args.out)
    report["bytes"] = total
    json.dump(report, open(os.path.join(args.out, "quant_report.json"), "w"), indent=1)
    print(f"wrote {args.out}: {total / 2**30:.2f} GiB in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
