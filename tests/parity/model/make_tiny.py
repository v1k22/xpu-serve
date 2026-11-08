#!/usr/bin/env python3
"""A tiny random checkpoint of a supported architecture, for parity tests (test-only).

    python -I tests/parity/model/make_tiny.py --arch qwen3_5_moe --tokenizer-from <Qwen3.5 dir> --out <dir>

The model keeps the real layer pattern and tensor names (Qwen3.5 MoE: three Gated DeltaNet layers then one full
attention layer, fused gate_up expert banks, a gated shared expert) at a width a CPU runs in seconds. The tokenizer
files are copied from a real checkpoint, so the vocabulary is the real one. Router weights are scaled up so each
token's top-k experts win by a clear margin: a near tie could route differently at f16 and fail parity for a reason
that is not a bug. Then run hf_reference.py on the output.
"""

import argparse
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"]


def qwen3_5_moe(vocab_size):
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    config = Qwen3_5MoeTextConfig(
        vocab_size=vocab_size, hidden_size=64, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=48, max_position_embeddings=4096, tie_word_embeddings=False,
        rms_norm_eps=1e-6, initializer_range=0.05,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
    )
    model = Qwen3_5MoeForCausalLM(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("mlp.gate.weight"):
                p.mul_(40.0)
    return model


ARCHS = {"qwen3_5_moe": qwen3_5_moe}


def fuse_experts(path):
    """Rewrites per-expert tensors (what save_pretrained writes) as the fused banks the released checkpoints ship:
    experts.gate_up_proj [E, 2 I, H] (gate rows first) and experts.down_proj [E, H, I]."""
    tensors = load_file(path)
    experts = {}
    for name in list(tensors):
        m = re.match(r"(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
        if m:
            experts.setdefault(m.group(1), {})[(int(m.group(2)), m.group(3))] = tensors.pop(name)
    for prefix, parts in experts.items():
        n = 1 + max(e for e, _ in parts)
        tensors[prefix + ".gate_up_proj"] = torch.stack(
            [torch.cat([parts[(e, "gate_proj")], parts[(e, "up_proj")]]) for e in range(n)]).contiguous()
        tensors[prefix + ".down_proj"] = torch.stack([parts[(e, "down_proj")] for e in range(n)]).contiguous()
    save_file(tensors, path, metadata={"format": "pt"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--tokenizer-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_from)
    model = ARCHS[args.arch](len(tok))
    model.config.eos_token_id = tok.eos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    model.save_pretrained(args.out, safe_serialization=True)
    fuse_experts(os.path.join(args.out, "model.safetensors"))
    for f in TOKENIZER_FILES:
        src = os.path.join(args.tokenizer_from, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters -> {args.out}")


if __name__ == "__main__":
    main()
