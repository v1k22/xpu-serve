#!/usr/bin/env python3
"""A tiny random checkpoint of a supported architecture, for parity tests (test-only).

    python -I tests/parity/model/make_tiny.py --arch qwen3_5|qwen3_5_moe|gemma4_moe --tokenizer-from <dir> --out <dir>

The model keeps the real layer pattern and tensor names (Qwen3.5 MoE: three Gated DeltaNet layers then one full
attention layer, fused gate_up expert banks, a gated shared expert; an MTP head) at a width a CPU runs in seconds. The tokenizer
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


def qwen3_5(vocab_size):
    """Dense, with Qwen3.8-27B's ratios: three linear value heads per key head, six query heads per KV head, gated
    attention output, a quarter of the head rotary with interleaved multimodal sections, an untied head."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=vocab_size, hidden_size=96, intermediate_size=192, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4, num_attention_heads=6, num_key_value_heads=1, head_dim=64, attn_output_gate=True,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, max_position_embeddings=4096, tie_word_embeddings=False, rms_norm_eps=1e-6,
        initializer_range=0.05,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [3, 3, 2]},
    )
    return Qwen3_5ForCausalLM(config)


def gemma4_moe(vocab_size):
    """Gemma 4 26B-A4B's layout: five sliding layers then a full one, experts beside every layer's MLP (a router with
    per-expert scales), global layers with fewer KV heads whose values are the key projection, softcapped logits.
    The sliding window is shorter than the longest prompt, so the masks are exercised."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    config = Gemma4TextConfig(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=48, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, num_global_key_value_heads=1, head_dim=32, global_head_dim=64, sliding_window=16,
        layer_types=["sliding_attention"] * 5 + ["full_attention"], hidden_size_per_layer_input=0,
        enable_moe_block=True, num_experts=8, top_k_experts=2, moe_intermediate_size=32, attention_k_eq_v=True,
        final_logit_softcapping=30.0, max_position_embeddings=4096, tie_word_embeddings=True, rms_norm_eps=1e-6,
        initializer_range=0.05,
        rope_parameters={"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                                            "rope_theta": 1000000.0},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
    )
    model = Gemma4ForCausalLM(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("router.proj.weight"):
                p.normal_(0.0, 40.0)  # decisive routing: a near tie could route differently at f16
            elif name.endswith("router.per_expert_scale") or name.endswith("router.scale"):
                p.uniform_(0.5, 1.5)
            elif p.dim() == 1 and "norm" in name:
                p.uniform_(0.5, 1.5)  # norms away from 1, so a skipped norm shows
    return model


ARCHS = {"qwen3_5": qwen3_5, "qwen3_5_moe": qwen3_5_moe, "gemma4_moe": gemma4_moe}


def mtp_head(model):
    """A random MTP head as the released checkpoints carry it (mtp.*: fc over [embedding | hidden], the two pre-fc
    norms, one full-attention decoder layer of the target's kind, the final norm), built from the target's classes."""
    import torch.nn as nn

    cfg = model.config
    layer_cls, norm_cls = type(model.model.layers[0]), type(model.model.norm)
    hidden = cfg.hidden_size
    head = nn.ModuleDict({
        "layers": nn.ModuleList([layer_cls(cfg, cfg.layer_types.index("full_attention"))]),
        "fc": nn.Linear(2 * hidden, hidden, bias=False),
        "pre_fc_norm_embedding": norm_cls(hidden, eps=cfg.rms_norm_eps),
        "pre_fc_norm_hidden": norm_cls(hidden, eps=cfg.rms_norm_eps),
        "norm": norm_cls(hidden, eps=cfg.rms_norm_eps),
    })
    with torch.no_grad():
        for name, p in head.named_parameters():
            if p.dim() == 1:
                p.normal_(0.0, 0.2)  # zero-centered norms: (1 + w) away from 1, so a skipped norm shows
            elif name.endswith("mlp.gate.weight"):
                p.normal_(0.0, 2.0)  # decisive routing, as for the target's routers
            else:
                p.normal_(0.0, cfg.initializer_range)
    return {"mtp." + k: v.contiguous() for k, v in head.state_dict().items()}


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
    qwen = args.arch.startswith("qwen")
    if qwen:
        model.config.mtp_num_hidden_layers = 1
    model.save_pretrained(args.out, safe_serialization=True)
    weights = os.path.join(args.out, "model.safetensors")
    if qwen:
        save_file({**load_file(weights), **mtp_head(model)}, weights, metadata={"format": "pt"})
    fuse_experts(weights)
    for f in TOKENIZER_FILES:
        src = os.path.join(args.tokenizer_from, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters -> {args.out}")


if __name__ == "__main__":
    main()
