#!/usr/bin/env python3
"""Reference next-token distributions for the manual-quantization experiment (branch exp/manual-quant).

    python ref_logits.py --model-dir <hf dir> --data <dir with wikitext2_test.parquet> --out <dir>

Builds the evaluation set (8 windows of WikiText-2 test, 4 of Python source, 256 tokens each), runs the checkpoint
on the CPU (bfloat16, or --dtype float32 for a reference free of bfloat16 rounding) and writes tokens.bin (int32 [N, 256]) and logprobs.bin (float16 [N, 256, vocab]: log-softmax
at every position), plus manifest.json with the reference perplexity per sequence.
"""

import argparse
import glob
import json
import os
import sysconfig
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEQ = 256


def eval_set(tok, data_dir):
    text = "\n".join(pd.read_parquet(os.path.join(data_dir, "wikitext2_test.parquet"))["text"].tolist())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    seqs = []
    for i in range(8):  # spread across the test set
        at = (len(ids) - SEQ) * i // 8
        seqs.append(ids[at : at + SEQ])
    stdlib = sysconfig.get_paths()["stdlib"]
    for name in ["textwrap.py", "argparse.py", "fractions.py", "heapq.py"]:
        code = tok(open(os.path.join(stdlib, name)).read(), add_special_tokens=False)["input_ids"]
        seqs.append(code[200 : 200 + SEQ])
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    seqs = eval_set(tok, args.data)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=getattr(torch, args.dtype))
    model.eval()
    print(f"loaded {type(model).__name__} in {time.time() - t0:.0f} s", flush=True)
    tokens = np.array(seqs, dtype=np.int32)
    tokens.tofile(os.path.join(args.out, "tokens.bin"))
    vocab = None
    ppl = []
    with open(os.path.join(args.out, "logprobs.bin"), "wb") as f, torch.no_grad():
        for i, s in enumerate(seqs):
            t1 = time.time()
            logits = model(input_ids=torch.tensor([s])).logits[0].float()
            lp = torch.log_softmax(logits, dim=-1)
            vocab = lp.shape[-1]
            f.write(lp.to(torch.float16).numpy().tobytes())
            nll = -lp[:-1].gather(1, torch.tensor(s[1:]).unsqueeze(1)).mean().item()
            ppl.append(float(np.exp(nll)))
            print(f"seq {i}: ppl {ppl[-1]:.3f} ({time.time() - t1:.1f} s)", flush=True)
    json.dump({"seq": SEQ, "count": len(seqs), "vocab": vocab, "ppl": ppl,
               "sources": ["wikitext2"] * 8 + ["python"] * 4}, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
