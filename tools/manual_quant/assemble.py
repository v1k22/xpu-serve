#!/usr/bin/env python3
"""Builds an overlay pack from per-tensor format choices over sweep.py outputs (branch exp/manual-quant).

    python assemble.py --alloc alloc.json --out <dir>

alloc.json: {"<tensor name>": "<sweep format dir>", ...}, e.g. {"model.language_model.layers.0.mlp.experts.down_proj":
"sweeps/experts/q3g64"}; each tensor (codes, scales, biases) is copied from that overlay pack. A list of format dirs
(one per expert, all of one group size, q8 / q4 / q3 / q2) makes the expert bank a qx tensor: expert e from list[e].
"""

import argparse
import json
import os

import torch
from safetensors import safe_open

import mq


class Pack:
    def __init__(self, path):
        self.path = path
        self.where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.manifest = json.load(open(os.path.join(path, "xpu_pack.json")))["tensors"]
        self.files = {}

    def get(self, name):
        f = self.where[name]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.path, f), framework="pt")
        return self.files[f].get_tensor(name)


def mixed(writer, name, per_expert):
    """Expert e of `name` from per_expert[e]: codes expert after expert, scales / biases [E, N, K / group]."""
    metas = [p.manifest[name] for p in per_expert]
    group = metas[0]["group"]
    assert all(m["group"] == group and m["format"] in ("q8", "q4", "q3", "q2") for m in metas), name
    tensors = {}
    for p in set(per_expert):
        scales = p.get(name + ".scales")
        biases = p.get(name + ".biases") if name + ".biases" in p.where else torch.zeros_like(scales)  # q8: none
        tensors[id(p)] = (p.get(name).view(torch.uint8), scales, biases)
    codes = torch.cat([tensors[id(p)][0][e].reshape(-1) for e, p in enumerate(per_expert)])
    scales = torch.stack([tensors[id(p)][1][e] for e, p in enumerate(per_expert)])
    biases = torch.stack([tensors[id(p)][2][e] for e, p in enumerate(per_expert)])
    writer.copy(name, codes)
    writer.copy(name + ".scales", scales)
    writer.copy(name + ".biases", biases)
    writer.manifest["tensors"][name] = {"format": "qx", "group": group, "shape": metas[0]["shape"],
                                        "expert_bits": [int(m["format"][1]) for m in metas]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    alloc = json.load(open(args.alloc))
    packs = {}
    writer = mq.PackWriter(args.out)
    for name in sorted(alloc):
        src = alloc[name]
        if isinstance(src, list):
            for s in set(src):
                packs.setdefault(s, Pack(s))
            mixed(writer, name, [packs[s] for s in src])
            continue
        if src not in packs:
            packs[src] = Pack(src)
        p = packs[src]
        for part in (name, name + ".scales", name + ".biases"):
            if part in p.where:
                writer.copy(part, p.get(part))
        writer.manifest["tensors"][name] = p.manifest[name]
    total = writer.close()
    print(f"wrote {args.out}: {len(alloc)} tensors, {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
