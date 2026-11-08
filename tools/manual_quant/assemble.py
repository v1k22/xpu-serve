#!/usr/bin/env python3
"""Builds an overlay pack from per-tensor format choices over sweep.py outputs (branch exp/manual-quant).

    python assemble.py --alloc alloc.json --out <dir>

alloc.json: {"<tensor name>": "<sweep format dir>", ...}, e.g. {"model.language_model.layers.0.mlp.experts.down_proj":
"sweeps/experts/q3g64"}; each tensor (codes, scales, biases) is copied from that overlay pack.
"""

import argparse
import json
import os

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
