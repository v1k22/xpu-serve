#!/usr/bin/env python3
"""A tiny random decision model in the layout of a released one, for parity tests (test-only).

    python -I tests/parity/decision/make_tiny.py --from <a clef-flash checkpoint> --out <dir>

The decoder is a Qwen3.5 of four layers (three Gated DeltaNet, one full attention) with a small vision tower, as the
released checkpoints carry one; the head has the released head's structure at a small width. The tokenizer, the
processor and chat template files and the checkpoint's own Python code are copied from `--from`, so the vocabulary is
the real one and reference.py runs the tiny model with the code the real one runs with. Random weights give
near-uniform answers, which would hide a wrong option order or span: the head's output scales are raised so that
answers spread.
"""

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

COPIED = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json", "vocab.json",
          "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json", "joint_schema_model.py",
          "joint_config.json"]


def decoder_config(vocab_size, hidden):
    from transformers import Qwen3_5Config

    return Qwen3_5Config(
        text_config=dict(
            vocab_size=vocab_size, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            full_attention_interval=4, num_attention_heads=6, num_key_value_heads=1, head_dim=64,
            attn_output_gate=True, linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=32,
            linear_value_head_dim=32, linear_conv_kernel_dim=4, max_position_embeddings=4096,
            tie_word_embeddings=False, rms_norm_eps=1e-6, initializer_range=0.05, mtp_num_hidden_layers=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                             "mrope_interleaved": True, "mrope_section": [3, 3, 2]}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=hidden,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64),
        tie_word_embeddings=False)


def clef(source, out, vocab_size):
    """Qwen3_5ForConditionalGeneration + joint_head.safetensors, as Cloudflare/clef-flash."""
    from transformers import Qwen3_5ForConditionalGeneration

    hidden = 96
    config = decoder_config(vocab_size, hidden)
    with open(os.path.join(source, "config.json")) as f:
        released = json.load(f)
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        setattr(config, key, released[key])
    model = Qwen3_5ForConditionalGeneration(config)
    model.save_pretrained(out, safe_serialization=True)

    sys.path.insert(0, os.path.abspath(source))  # the head's class is the checkpoint's own
    import joint_schema_model as code

    head_config = {"hidden_size": hidden, "width": 64, "routing_layers": 2, "layers": 2, "heads": 4, "feedforward": 128}
    head = code.JointSchemaHead(**head_config)
    with torch.no_grad():
        for name, p in head.named_parameters():
            if name.endswith("bias"):
                p.normal_(0.0, 0.1)  # torch zeroes the attention biases and the LayerNorm biases: a dropped one would not show
            elif "norm" in name and name.endswith("weight"):
                p.normal_(1.0, 0.2)
        head.prior_logit_scale.fill_(1.5)
        head.joint_logit_scale.fill_(2.0)
        head.residual_gate.fill_(0.7)
        head.residual_scorer[3].weight.mul_(8.0)
    save_file(head.state_dict(), os.path.join(out, "joint_head.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out, "joint_head_config.json"), "w") as f:
        json.dump(head_config, f, indent=2)
    return sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", required=True, help="a released checkpoint of the kind to imitate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    vocab_size = len(AutoTokenizer.from_pretrained(args.source))
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.source, "joint_head_config.json")):
        kind, count = "clef", clef(args.source, args.out, vocab_size)
    else:
        ap.error(f"{args.source} is not a checkpoint this script imitates (clef)")
    for f in COPIED:
        src = os.path.join(args.source, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print(f"{kind}: {count / 1e6:.1f} M parameters -> {args.out}")


if __name__ == "__main__":
    main()
