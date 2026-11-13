#!/usr/bin/env python3
"""A tiny random decision model in the layout of a released one, for parity tests (test-only).

    python -I tests/parity/decision/make_tiny.py --from <a clef-flash, pplx-decider, openjev, laya or julia-1
        checkpoint> --out <dir>

The decoder is a Qwen3.5 of four layers (three Gated DeltaNet, one full attention) with a small vision tower, as the
released checkpoints carry one; the head has the released head's structure at a small width (Clef's joint schema
head; pplx-decider's read-out of 255 answer codes; none for OpenJev, a language model with its helper). Laya (its
root or a variant's folder) and Julia-1 get a ModernBERT encoder of four layers, a global one each side of two local
ones with a window of 16, and their decision head, as their own code builds it. The tokenizer, the processor and chat
template files and the checkpoint's own Python code are copied from `--from`, so the vocabulary is the real one and
reference.py runs the tiny model with the code the real one runs with. Random weights give near-uniform answers,
which would hide a wrong option order or span: the head's output scales are raised so that answers spread.
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
          "joint_config.json", "decision_config.json", "source", "helper", "generation_config.json",
          "rl_agent_config.json", "rl_common.py", "rl_agent_api.py", "inference-policy.json", "julia", "pyproject.toml",
          "tokenizer"]


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


def pplx(source, out, vocab_size):
    """Qwen3_5Model + readout.safetensors, as perplexity-ai/pplx-decider-flash-v1.2."""
    from transformers import Qwen3_5Model

    hidden = 96
    config = decoder_config(vocab_size, hidden)
    with open(os.path.join(source, "config.json")) as f:
        released = json.load(f)
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        setattr(config, key, released[key])
    model = Qwen3_5Model(config)
    model.save_pretrained(out, safe_serialization=True)
    with open(os.path.join(source, "decision_config.json")) as f:
        codes = len(json.load(f)["codes"])
    readout = torch.nn.Linear(hidden, codes, bias=False)
    with torch.no_grad():
        readout.weight.mul_(4.0)
    save_file({"weight": readout.weight.detach()}, os.path.join(out, "readout.safetensors"), metadata={"format": "pt"})
    return sum(p.numel() for p in model.parameters()) + readout.weight.numel()


def letters(source, out, vocab_size):
    """Qwen3_5ForConditionalGeneration and nothing else, as openjev/openjev: the answers are read off its output layer."""
    from transformers import Qwen3_5ForConditionalGeneration

    config = decoder_config(vocab_size, 96)
    with open(os.path.join(source, "config.json")) as f:
        released = json.load(f)
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        setattr(config, key, released[key])
    model = Qwen3_5ForConditionalGeneration(config)
    with torch.no_grad():
        model.lm_head.weight.mul_(6.0)  # the letters' scores apart
    model.save_pretrained(out, safe_serialization=True)
    return sum(p.numel() for p in model.parameters())


def modern_bert_config(source):
    """The released encoder's config.json at a small size; local layers see 8 tokens on either side."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(os.path.join(source, "encoder"))
    config.hidden_size, config.intermediate_size, config.num_hidden_layers = 128, 96, 4
    config.num_attention_heads, config.local_attention = 4, 16
    config.layer_types = ["full_attention", "sliding_attention", "sliding_attention", "full_attention"]
    config.initializer_range = 0.2
    return config


def spread(model):
    """Biases and norms off their defaults (a dropped one would not show), the scores apart."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.normal_(0.0, 0.1)
            elif "norm" in name and name.endswith("weight") or name == "scorer.0.weight":
                p.normal_(1.0, 0.2)
        model.scorer[3].weight.mul_(8.0)


def laya(source, out, vocab_size):
    """rl_common.DecisionModel on a tiny ModernBERT, as convaiinnovations/laya stores it (f16, the temperature buffer
    f32), with the root's config.json and the code beside it (a variant's folder has the code one level up)."""
    from transformers import AutoModel

    code = source if os.path.exists(os.path.join(source, "rl_common.py")) else os.path.dirname(source)
    sys.path.insert(0, os.path.abspath(code))
    import rl_common

    with open(os.path.join(source, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    encoder = AutoModel.from_config(modern_bert_config(source), attn_implementation="sdpa")
    model = rl_common.DecisionModel(encoder, cfg["head_layers"], len(cfg["act_costs"]) + 1)
    spread(model)
    state = {k: v if k == "temperature" else v.half() for k, v in model.state_dict().items()}
    save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(out, "model.safetensors"),
              metadata={"format": "pt"})
    encoder.config.save_pretrained(os.path.join(out, "encoder"))
    for f in ("rl_common.py", "rl_agent_api.py", "config.json"):
        if os.path.exists(os.path.join(code, f)):
            shutil.copy(os.path.join(code, f), os.path.join(out, f))
    return sum(p.numel() for p in model.parameters())


def julia(source, out, vocab_size):
    """julia.model.JuliaDecisionModel on a tiny ModernBERT, saved by its own save_pretrained (f32)."""
    from transformers import AutoModel

    sys.path.insert(0, os.path.abspath(source))
    from julia.model import JuliaDecisionModel

    encoder = AutoModel.from_config(modern_bert_config(source), attn_implementation="sdpa")
    model = JuliaDecisionModel(encoder, head_layers=2)
    spread(model)
    model.save_pretrained(out)
    shutil.copy(os.path.join(source, "config.json"), os.path.join(out, "config.json"))
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", required=True, help="a released checkpoint of the kind to imitate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = os.path.join(args.source, "tokenizer")  # Laya and Julia-1 keep it in a folder
    vocab_size = len(AutoTokenizer.from_pretrained(tokenizer if os.path.isdir(tokenizer) else args.source))
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.source, "joint_head_config.json")):
        kind, count = "clef", clef(args.source, args.out, vocab_size)
    elif os.path.exists(os.path.join(args.source, "decision_config.json")):
        kind, count = "pplx_decider", pplx(args.source, args.out, vocab_size)
    elif os.path.exists(os.path.join(args.source, "helper", "shim.py")):
        kind, count = "letter_readout", letters(args.source, args.out, vocab_size)
    elif os.path.exists(os.path.join(args.source, "rl_agent_config.json")):
        kind, count = "laya", laya(args.source, args.out, vocab_size)
    elif os.path.exists(os.path.join(args.source, "julia_config.json")):
        kind, count = "julia", julia(args.source, args.out, vocab_size)
    else:
        ap.error(f"{args.source} is not a checkpoint this script imitates (clef, pplx-decider, openjev, laya, julia-1)")
    for f in COPIED:
        src = os.path.join(args.source, f)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(args.out, f), dirs_exist_ok=True)
        elif os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print(f"{kind}: {count / 1e6:.1f} M parameters -> {args.out}")


if __name__ == "__main__":
    main()
