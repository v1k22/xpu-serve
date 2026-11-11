#!/usr/bin/env python3
"""A Qwen3.5 checkpoint rewritten the way an FP8 release stores it, and the model those bytes decode to (test-only).

    python -I tests/parity/model/to_fp8.py --model-dir <checkpoint> --out <fp8 dir> --decoded <dir>

`--out` gets the layout of the -FP8 releases (transformers' fine-grained FP8): each projection of the decoder and of
the MTP head as float8_e4m3fn beside `<name>_scale_inv`, one bf16 scale per block of 128 rows and columns
(w = code * scale), expert banks expert by expert (experts.<e>.gate_proj.weight, up_proj, down_proj), and
quantization_config in config.json. Embeddings, the head, norms, routers and the small linear-attention projections
are copied, as the releases leave them.

`--decoded` gets the original layout with the values those codes and scales stand for, in float32. Run
hf_reference.py on it, then the parity tests on `--out` with those fixtures: the FP8 checkpoint must match as the
decoded one matches itself. torch rounds to E4M3 and decodes here, so the comparison does not rest on this
repository's decoder.
"""

import argparse
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file

BLOCK = 128
FP8_MAX = 448.0
PROJECTIONS = ("_proj.weight", "in_proj_qkv.weight", "in_proj_z.weight")


def to_fp8(w):
    """[R, C] float -> (E4M3 codes [R, C], bf16 scales [ceil(R / 128), ceil(C / 128)], the float32 values they mean)."""
    rows, cols = w.shape
    padded = torch.nn.functional.pad(w.float(), (0, -cols % BLOCK, 0, -rows % BLOCK))
    blocks = padded.view(padded.shape[0] // BLOCK, BLOCK, padded.shape[1] // BLOCK, BLOCK)
    peak = blocks.abs().amax(dim=(1, 3))
    scale = torch.where(peak > 0, peak / FP8_MAX, torch.ones_like(peak)).to(torch.bfloat16)
    s = scale.float()[:, None, :, None]
    codes = (blocks / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    decoded = (codes.float() * s).reshape(padded.shape)[:rows, :cols].contiguous()
    return codes.reshape(padded.shape)[:rows, :cols].contiguous(), scale, decoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True, help="the FP8 layout")
    ap.add_argument("--decoded", required=True, help="the original layout, decoded values in float32")
    args = ap.parse_args()

    tensors = {}
    for f in sorted(os.listdir(args.model_dir)):
        if f.endswith(".safetensors"):
            tensors.update(load_file(os.path.join(args.model_dir, f)))
    fp8, decoded = {}, {}

    def store(name, w):
        codes, scale, values = to_fp8(w)
        fp8[name] = codes
        fp8[name + "_scale_inv"] = scale
        return values

    for name, t in tensors.items():
        bank = re.match(r"(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$", name)
        if bank and bank.group(2) == "gate_up_proj":  # [E, 2 I, H], an expert's gate rows first
            half = t.shape[1] // 2
            decoded[name] = torch.stack([
                torch.cat([store(f"{bank.group(1)}.{e}.gate_proj.weight", t[e, :half]),
                           store(f"{bank.group(1)}.{e}.up_proj.weight", t[e, half:])]) for e in range(t.shape[0])])
        elif bank:  # [E, H, I]
            decoded[name] = torch.stack(
                [store(f"{bank.group(1)}.{e}.down_proj.weight", t[e]) for e in range(t.shape[0])])
        elif t.dim() == 2 and name.endswith(PROJECTIONS) and "visual" not in name:
            decoded[name] = store(name, t)
        else:
            fp8[name] = decoded[name] = t

    for out, weights in ((args.out, fp8), (args.decoded, decoded)):
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(args.model_dir):
            if not f.endswith(".safetensors") and f != "model.safetensors.index.json" and not f.startswith("."):
                shutil.copy(os.path.join(args.model_dir, f), os.path.join(out, f))
        save_file(weights, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(args.out, "config.json")) as f:
        config = json.load(f)
    config["quantization_config"] = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                     "weight_block_size": [BLOCK, BLOCK]}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    n = sum(1 for t in fp8.values() if t.dtype == torch.float8_e4m3fn)
    print(f"{n} fp8 tensors of {len(fp8)} in {args.out}; decoded twin in {args.decoded}")


if __name__ == "__main__":
    main()
