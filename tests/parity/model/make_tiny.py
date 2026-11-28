#!/usr/bin/env python3
"""A tiny random checkpoint of a supported architecture, for parity tests (test-only).

    python -I tests/parity/model/make_tiny.py --arch llama|qwen3|qwen3_moe|qwen3_5|qwen3_5_moe|qwen3_5_vl|qwen3_5_moe_vl|gemma4_moe|lfm3|
        sensenova_loop|qwen3_asr|inkling2_loop --tokenizer-from <dir> --out <dir>
    python -I tests/parity/model/make_tiny.py --arch bert|nomic_bert --tokenizer-from <dir> --out <dir>
        [--pooling mean|cls] [--normalize] [--max-seq-length N]
    python -I tests/parity/model/make_tiny.py --drafter dflash|dflash2|dspine --target <tiny dir> --dflash-repo <clone
        of github.com/z-lab/dflash> --out <dir>

The model keeps the real layer pattern and tensor names (Qwen3.5 MoE: three Gated DeltaNet layers then one full
attention layer, fused gate_up expert banks, a gated shared expert; an MTP head) at a width a CPU runs in seconds.
The tokenizer files are copied from a real checkpoint, so the vocabulary is the real one; an architecture
transformers does not ship (lfm3, sensenova_loop) is built from the modeling code in that checkpoint, which is copied
too. Router weights are scaled up so each token's top-k experts win by a clear margin: a near tie could route
differently at f16 and fail parity for a reason that is not a bug. Then run hf_reference.py on the output.

An embedding model (bert, nomic_bert) is the encoder of a sentence-transformers checkpoint, with its modules.json,
1_Pooling/config.json and sentence_bert_config.json: --pooling and --normalize pick the pipeline, --max-seq-length the
longest input. nomic_bert's modeling code comes from the --tokenizer-from checkpoint (nomic-ai/nomic-bert-2048 holds
both); the tiny one is trained on fewer positions than it reads (max_trained_positions), as nomic-embed-text-v1.5 reads
8192 tokens after training on 2048. Norms and the token type rows are away from their defaults, so a skipped one
shows. Then run tests/parity/embedding/reference.py on the output.

A drafter is a block drafter for a tiny target, in the released checkpoints' layout (config keys and tensor names):
z-lab's DFlash model, DFlash 2 with its convolutions and candidate selector, or DSpine with its code bridge. Its
window is shorter than most prompts and its bridge writes and selector terms are as large as the states they change,
so a wrong mask, write or pairing shows. Then run block_drafter_reference.py on target and drafter.
"""

import argparse
import importlib.util
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json", "vocab.json",
                   "merges.txt", "preprocessor_config.json", "processor_config.json"]


def qwen3_5_moe_text(vocab_size):
    return dict(
        vocab_size=vocab_size, hidden_size=64, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=48, max_position_embeddings=4096, tie_word_embeddings=False,
        rms_norm_eps=1e-6, initializer_range=0.05,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25})


def qwen3(vocab_size, source):
    """Dense Qwen3 with Qwen3-8B's ratios: four query heads per KV head, q and k normed per head, an untied head;
    its norms away from 1, so a skipped one shows."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(vocab_size=vocab_size, hidden_size=64, intermediate_size=160, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=1, head_dim=32, max_position_embeddings=4096,
                         rope_theta=1000000.0, rms_norm_eps=1e-6, tie_word_embeddings=False, initializer_range=0.05)
    return off_one_norms(Qwen3ForCausalLM(config))


def qwen3_moe(vocab_size, source):
    """Qwen3's mixture of experts, as Qwen3-30B-A3B: eight experts, two a token, renormalized; a dense first layer
    (mlp_only_layers), so both kinds of layer run. Saved with each expert's tensors apart, as the release stores them."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                            num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
                            mlp_only_layers=[0], num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=32, max_position_embeddings=4096, rope_theta=1000000.0, rms_norm_eps=1e-6,
                            tie_word_embeddings=False, initializer_range=0.05)
    return decisive_routers(off_one_norms(Qwen3MoeForCausalLM(config)))


def llama(vocab_size, source):
    """Llama 3.2-1B's ratios (four query heads per KV head, the head tied to the embedding) and its RoPE: theta 500000
    and Llama 3's scaling, whose three bands of wavelengths (divided by the factor, blended, kept) a head of 32
    channels all reaches; norms away from 1."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=vocab_size, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=1, head_dim=32, max_position_embeddings=4096,
                         rms_norm_eps=1e-5, tie_word_embeddings=True, initializer_range=0.05,
                         rope_parameters={"rope_type": "llama3", "rope_theta": 500000.0, "factor": 32.0,
                                          "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                          "original_max_position_embeddings": 8192})
    return off_one_norms(LlamaForCausalLM(config))


def off_one_norms(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and "norm" in name:
                p.uniform_(0.5, 1.5)
    return model


def decisive_routers(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("mlp.gate.weight"):
                p.mul_(40.0)
    return model


def qwen3_5_moe(vocab_size, source):
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    return decisive_routers(Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**qwen3_5_moe_text(vocab_size))))


def qwen3_5_text(vocab_size):
    return dict(
        vocab_size=vocab_size, hidden_size=96, intermediate_size=192, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4, num_attention_heads=6, num_key_value_heads=1, head_dim=64, attn_output_gate=True,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, max_position_embeddings=4096, tie_word_embeddings=False, rms_norm_eps=1e-6,
        initializer_range=0.05,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [3, 3, 2]})


def qwen3_5(vocab_size, source):
    """Dense, with Qwen3.8-27B's ratios: three linear value heads per key head, six query heads per KV head, gated
    attention output, a quarter of the head rotary with interleaved multimodal sections, an untied head."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    return Qwen3_5ForCausalLM(Qwen3_5TextConfig(**qwen3_5_text(vocab_size)))


def qwen3_5_vision(model):
    """A tiny Qwen3-VL tower without deepstack rows, as Qwen3.5's checkpoints carry it (model.visual.*), with its
    norms and biases away from their initial values so a skipped one shows."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "visual" not in name or p.dim() != 1:
                continue
            if "norm" in name:
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0.0, 0.2)
    return model


def qwen3_5_vl_config(config_cls, text, source):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(source)
    ids = {key: tok.convert_tokens_to_ids(token) for key, token in [
        ("image_token_id", "<|image_pad|>"), ("video_token_id", "<|video_pad|>"),
        ("vision_start_token_id", "<|vision_start|>"), ("vision_end_token_id", "<|vision_end|>")]}
    # out_hidden_size is the decoder's width; the multimodal sections must cover the MoE's rotary pairs too.
    text = dict(text, rope_parameters=dict(text["rope_parameters"], mrope_interleaved=True,
                                           mrope_section=[2, 1, 1] if text["head_dim"] == 32 else [3, 3, 2]))
    vision = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=text["hidden_size"],
                  patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64)
    return config_cls(text_config=text, vision_config=vision, tie_word_embeddings=False, **ids)


def qwen3_5_vl(vocab_size, source):
    """qwen3_5's decoder with Qwen3.5's vision tower (Qwen3_5ForConditionalGeneration): images in."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = qwen3_5_vl_config(Qwen3_5Config, qwen3_5_text(vocab_size), source)
    return qwen3_5_vision(Qwen3_5ForConditionalGeneration(config))


def qwen3_5_moe_vl(vocab_size, source):
    """qwen3_5_moe's decoder with the vision tower (Qwen3_5MoeForConditionalGeneration), as Qwen3.6-35B-A3B."""
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

    config = qwen3_5_vl_config(Qwen3_5MoeConfig, qwen3_5_moe_text(vocab_size), source)
    return qwen3_5_vision(decisive_routers(Qwen3_5MoeForConditionalGeneration(config)))


def gemma4_moe(vocab_size, source):
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


def lfm3(vocab_size, source):
    """LFM3-1.7B's layout: gated short convolutions (seven taps) around full and sliding attention layers, four query
    heads per KV head, sandwich norms, a tied head. The sliding window is shorter than the longest prompt, and the RoPE
    bases are small: at the released 1e6 the unrotated half of a full-attention head would barely differ from a
    rotated one within a prompt, and a wrong layout would pass. The head gates are decisive (a constant gate would
    vanish in the norm after the attention)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_cls = get_class_from_dynamic_module("configuration_lfm3.Lfm3Config", source)
    model_cls = get_class_from_dynamic_module("modeling_lfm3.Lfm3ForCausalLM", source)
    config = config_cls(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=96, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=1, head_dim=32, conv_L_cache=7, sliding_window=16, max_position_embeddings=4096,
        layer_types=["conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"],
        norm_eps=1e-5, initializer_range=0.05, tie_word_embeddings=True,
        rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 200.0, "partial_rotary_factor": 0.5},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 50.0}},
    )
    model = model_cls(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("gate_proj.weight"):
                p.normal_(0.0, 1.0)
            elif name.endswith("conv.conv.weight"):
                p.normal_(0.0, 0.4)
            elif p.dim() == 1 and "norm" in name:
                p.uniform_(0.5, 1.5)  # norms away from 1, so a skipped norm shows
    return model


def sensenova_loop(vocab_size, source):
    """SenseNova-Loop-9B's layout from its own modeling code (the --tokenizer-from checkpoint's): twelve Qwen3.5 layers
    in three linear / linear / linear / full cycles, the middle four run twice (loop_settings) inside two-stream
    hyper-connections (mhc), a tiny vision tower (the class builds one; the engine does not load it). The routers'
    weights are as large as a trained router's, so read, write and mix weights differ by token and stream and a wrong
    order of streams or mix shows; norms away from their defaults."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_cls = get_class_from_dynamic_module("modeling_sensenova_loop.SenseNovaLoopConfig", source)
    model_cls = get_class_from_dynamic_module("modeling_sensenova_loop.SenseNovaLoopForConditionalGeneration", source)
    with open(os.path.join(source, "config.json")) as f:
        config = json.load(f)
    for key in ("auto_map", "model_name"):
        config.pop(key, None)
    config["text_config"].update(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=12, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, max_position_embeddings=4096, initializer_range=0.05,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 3,
    )
    config["text_config"]["rope_parameters"]["mrope_section"] = [1, 1, 2]
    config["vision_config"].update(hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, depth=2)
    config["hidden_size"] = 64
    config["loop_settings"] = {"counts": [2], "start": 4, "end": 8, "eval_count": 2}
    config["mhc"] = {"start": 4, "end": 8, "streams": 2, "iterations": 20, "fixed_identity": False, "pre_scale": 1.0}
    model = model_cls(config_cls.from_dict(config)).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".mhc_" in name and name.endswith("weight"):
                p.normal_(0.0, 0.1)
            elif ".mhc_" in name:
                p.normal_(0.0, 1.0)
            elif p.dim() == 1 and "norm" in name and "visual" not in name:
                p.normal_(0.0, 0.2)  # zero-centered norms: (1 + w) away from 1, so a skipped norm shows
    return model


def inkling2_loop(vocab_size, source):
    """Inkling 2 Mini's layout, from the released checkpoint's own config.json (the --tokenizer-from one): twelve
    layers in blocks of three, the sixth and the twelfth attending over the whole context, layers 3 to 8 run three
    times with messages between the passes, a tiny vision tower (two layers over the release's patches). Everything
    the released values would hide at this size is brought into reach: clip thresholds near the typical norm of a query, key or
    relative state, softcaps near the experts' and the logits' range, a floor of the position scaling below a
    prompt's length, a window and a bias extent shorter than a prompt. Parameters the class starts at zero or one
    (the residual and message queries, the stored taps of the dynamic convolutions, the clip scales) are random, as
    are the routers' selection biases, so each shows when skipped. The scales are as large as keeps the model well
    conditioned: at twice these an error of one part in a million in the weights moved the logits a thousand times
    as much, and no f16 run could match."""
    from transformers import Inkling2LoopConfig, Inkling2LoopForConditionalGeneration

    with open(os.path.join(source, "config.json")) as f:
        released = json.load(f)
    released.pop("quantization_config", None)  # released_inkling2_layout stores the experts quantized and restores it
    text = released["text_config"]
    layers = 12
    text.update(
        vocab_size=max(vocab_size, text["vocab_size"]), hidden_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, swa_num_attention_heads=4,
        swa_num_key_value_heads=2, swa_head_dim=32, d_rel=8, rel_extent=24, sliding_window_size=16,
        log_scaling_n_floor=20, log_scaling_alpha=0.4, qk_l2_clip_threshold=2.25, rel_l2_clip_threshold=1.1,
        local_layer_ids=[i for i in range(layers) if (i + 1) % 6], n_layers_per_attn_res_block=3, n_routed_experts=8,
        num_experts_per_tok=2, intermediate_size=32, route_scale=2.0, mlp_gate_softcap=0.75, mlp_up_softcap=1.0,
        final_logit_softcapping=0.75, model_max_length=4096,
        loop_config={"core_start": 3, "core_end": 9, "num_passes": 3, "gate_rank": 8, "is_varloop": False},
    )
    released["vision_config"].update(text_hidden_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                     projection_hidden_dim=64, posemb_grid_size=4)
    config = Inkling2LoopConfig.from_dict(json.loads(json.dumps(released)))
    config.text_config.initializer_range = 0.05
    model = Inkling2LoopForConditionalGeneration(config).float()
    model.released_config = released
    spread = {"attention_input.query": 0.75, "mlp_input.query": 0.75, "final_input.query": 0.75,
              "message_input.query": 0.75, "sconv.static_weight": 0.15, "sconv.weight_proj.weight": 0.025,
              "mlp.gate.weight": 0.25, "sink_proj.weight": 0.1, "relative_position_bias.proj": 0.5,
              "_gate_proj.weight": 0.15}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "vision_tower" in name:  # what the class starts at one, zero or near zero, so a skipped one shows
                if "layernorm" in name or "lambda_" in name:
                    p.normal_(0.0, 0.2) if name.endswith(".bias") else p.uniform_(0.5, 1.5)
                elif name.endswith(("class_embedding", "cls_position_embedding")):
                    p.normal_(0.0, 0.5)
                continue
            scale = next((v for k, v in spread.items() if name.endswith(k)), None)
            if scale is not None:
                p.normal_(0.0, scale)
            elif name.endswith("norm.scale") or (p.dim() == 1 and "norm" in name):
                p.uniform_(0.5, 1.5)  # clip scales and norms away from 1, so a skipped one shows
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.normal_(0.0, 0.2)
    return model


def nvfp4(w):
    """A bank [E, rows, cols] as the release stores NVFP4: E2M1 codes two a byte (the even column in the low nibble),
    an E4M3 scale per 16 columns of a row, each expert's largest magnitude and the denominator its scales divide by."""
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    amax = w.abs().amax(dim=(1, 2))
    denominator = torch.tensor([6.0 * 448.0])
    factor = (amax / denominator)[:, None, None, None]
    blocks = w.unflatten(-1, (-1, 16))
    scale = (blocks.abs().amax(-1, keepdim=True) / 6.0 / factor).to(torch.float8_e4m3fn)
    scaled = blocks / (scale.float() * factor).clamp_min(1e-30)
    codes = (scaled.abs()[..., None] - values).abs().argmin(-1) + 8 * (scaled < 0)
    codes = codes.flatten(-2).to(torch.uint8)
    return {"": codes[..., 0::2] | (codes[..., 1::2] << 4), "_scale": scale.squeeze(-1), "_amax": amax,
            "_outer_scale_denominator": denominator}


def mxfp8(w):
    """A bank as the release stores MXFP8: E4M3 codes and, per 32 columns of a row, a power of two as its exponent
    plus 127 in a byte."""
    blocks = w.unflatten(-1, (-1, 32))
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-30) / 448.0)).clamp(-127, 127)
    codes = (blocks / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn).flatten(-2)
    return {"": codes, "_scale": (exponent + 127).to(torch.uint8)}


def released_inkling2_layout(out, released):
    """The saved checkpoint as Inkling 2 Mini is released: its own config.json keys, one shard under an index, and the
    experts quantized, the last third of the layers in MXFP8 and the others in NVFP4 (the release's last four of 24).
    transformers decodes both on load, so the reference runs on the values these bytes hold."""
    path = os.path.join(out, "model.safetensors")
    tensors = load_file(path)
    layers = released["text_config"]["num_hidden_layers"]
    for i in range(layers):
        for bank in ("w13_weight", "w2_weight"):
            name = f"model.llm.layers.{i}.mlp.experts.{bank}"
            stored = (mxfp8 if i >= layers - layers // 3 else nvfp4)(tensors.pop(name).float())
            tensors.update({name + suffix: t for suffix, t in stored.items()})
    os.remove(path)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, os.path.join(out, shard), metadata={"format": "pt"})
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": {name: shard for name in sorted(tensors)}}, f, indent=2)
    released["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    released["mtp_config"] = inkling2_drafter(out, released["text_config"])
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(released, f, indent=2)


def inkling2_drafter(out, text):
    """The checkpoint's block drafter (DSpark) as the release ships one: mtp.safetensors beside the model and an
    mtp_config, which is returned. Three layers, the first and the last seeing a window of context shorter than a
    prompt; states read at the input of the last layer of a block before the loop, in its second pass and in its
    third, and after the final norm, as the release's are; every tensor random, norms away from 1."""
    hidden, heads, kv_heads, head_dim, width, rank, layers = text["hidden_size"], 4, 2, 32, 128, 16, 3
    block, loop = text["n_layers_per_attn_res_block"], text["loop_config"]
    core = loop["core_end"] - loop["core_start"]
    visits = text["num_hidden_layers"] + (loop["num_passes"] - 1) * core
    taps = [block - 1, loop["core_start"] + core + block - 1, loop["core_start"] + 3 * core - 1, visits]
    config = dict(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv_heads,
                  head_dim=head_dim, intermediate_size=width, max_position_embeddings=text["model_max_length"],
                  torch_dtype="bfloat16", q_bias=False, o_bias=False, rms_norm_eps=1e-6, rope_theta=1000000.0,
                  local_layer_ids=[0, 2], sliding_window_size=6, block_size=8, markov_rank=rank,
                  markov_head_type="rnn_hidden", target_layer_ids=taps)
    g = torch.Generator().manual_seed(7)
    normal = lambda *shape, std=0.05: torch.randn(*shape, generator=g) * std
    norm = lambda n: torch.rand(n, generator=g) + 0.5
    t = {"fuse_weight": normal(layers + 1, len(taps), std=1.0), "mask_embed": normal(hidden, std=1.0),
         "norm.weight": norm(hidden), "markov_head.dflare_norm.weight": norm(hidden),
         "markov_head.markov_w1.weight": normal(text["vocab_size"], rank, std=1.0),
         "markov_head.rnn_proj.weight": normal(3 * rank, 2 * rank + 2 * hidden, std=0.1),
         "markov_head.rnn_proj.bias": normal(3 * rank, std=0.5),
         "markov_head.hidden_out.weight": normal(hidden, rank, std=0.5),
         "markov_head.hidden_out.bias": normal(hidden, std=0.2)}
    for l in range(layers):
        a = f"layers.{l}.self_attn."
        t.update({
            f"layers.{l}.input_layernorm.weight": norm(hidden), f"layers.{l}.post_attention_layernorm.weight": norm(hidden),
            f"layers.{l}.dflare_norm.weight": norm(hidden),
            a + "q_proj.weight": normal(heads * head_dim, hidden, std=0.25),
            a + "k_proj.weight": normal(kv_heads * head_dim, hidden, std=0.25),
            a + "v_proj.weight": normal(kv_heads * head_dim, hidden),
            a + "o_proj.weight": normal(hidden, heads * head_dim),
            a + "k_proj_ctx.weight": normal(kv_heads * head_dim, hidden, std=0.25),
            a + "v_proj_ctx.weight": normal(kv_heads * head_dim, hidden),
            a + "q_norm.weight": norm(head_dim), a + "k_norm.weight": norm(head_dim), a + "k_norm_ctx.weight": norm(head_dim),
            f"layers.{l}.mlp.gate_proj.weight": normal(width, hidden), f"layers.{l}.mlp.up_proj.weight": normal(width, hidden),
            f"layers.{l}.mlp.down_proj.weight": normal(hidden, width)})
    save_file({k: v.to(torch.bfloat16) for k, v in t.items()}, os.path.join(out, "mtp.safetensors"),
              metadata={"format": "pt"})
    return config


def qwen3_asr(vocab_size, source):
    """Qwen3-ASR's layout from transformers' own classes: a Whisper-style audio encoder (three stride-2 convolutions
    over chunks of 100 mel frames, attention within windows of 800 frames, a two-layer projector) and a Qwen3
    decoder; saved in the released checkpoint's layout (weights under thinker.*, a thinker_config) by
    released_asr_layout(). The audio token is the --tokenizer-from checkpoint's <|audio_pad|>."""
    from transformers import AutoTokenizer, Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

    audio_token = AutoTokenizer.from_pretrained(source).convert_tokens_to_ids("<|audio_pad|>")
    config = Qwen3ASRConfig(
        audio_config=dict(num_mel_bins=128, encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=128,
                          d_model=64, n_window=50, n_window_infer=800, downsample_hidden_size=32, output_dim=96,
                          max_position_embeddings=13, activation_function="gelu"),
        text_config=dict(vocab_size=vocab_size, hidden_size=96, intermediate_size=192, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6,
                         max_position_embeddings=4096, rope_theta=1000000.0, tie_word_embeddings=False,
                         hidden_act="silu"),
        audio_token_id=audio_token, tie_word_embeddings=False, initializer_range=0.05)
    model = Qwen3ASRForConditionalGeneration(config).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and "norm" in name:
                p.uniform_(0.5, 1.5)  # norms away from 1, so a skipped norm shows
            elif p.dim() == 1:
                p.normal_(0.0, 0.2)
    return model


def released_asr_layout(out):
    """Rewrites a Qwen3-ASR saved by transformers into the released checkpoint's layout: the configs nested under
    thinker_config, the weights under thinker.* with the projector inside the audio tower (proj1, proj2)."""
    path = os.path.join(out, "model.safetensors")
    renames = [("model.multi_modal_projector.linear_1.", "thinker.audio_tower.proj1."),
               ("model.multi_modal_projector.linear_2.", "thinker.audio_tower.proj2."),
               ("model.audio_tower.", "thinker.audio_tower."), ("model.language_model.", "thinker.model."),
               ("lm_head.", "thinker.lm_head.")]
    tensors = {}
    for name, t in load_file(path).items():
        for old, new in renames:
            if name.startswith(old):
                name = new + name[len(old):]
                break
        tensors[name] = t
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(out, "config.json")) as f:
        flat = json.load(f)
    audio = dict(flat["audio_config"], max_source_positions=flat["audio_config"].pop("max_position_embeddings"))
    thinker = {"audio_config": audio, "text_config": flat["text_config"], "audio_token_id": flat["audio_token_id"],
               "model_type": "qwen3_asr"}
    released = {"architectures": ["Qwen3ASRForConditionalGeneration"], "model_type": "qwen3_asr",
                "thinker_config": thinker}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(released, f, indent=2)


def qwen3_omni(vocab_size, source):
    """Qwen3-Omni's thinker from transformers' own classes: Qwen's audio encoder, a Qwen3-VL vision tower (saved, not
    read yet), and a Qwen3 mixture-of-experts decoder whose second layer is dense (mlp_only_layers), with interleaved
    multimodal RoPE; saved in the released checkpoint's layout (weights under thinker.*, one tensor per expert, a
    thinker_config) by released_omni_layout(). Token ids are the --tokenizer-from checkpoint's."""
    from transformers import AutoTokenizer, Qwen3OmniMoeThinkerConfig, Qwen3OmniMoeThinkerForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(source)
    ids = {key: tok.convert_tokens_to_ids(token) for key, token in [
        ("audio_token_id", "<|audio_pad|>"), ("image_token_id", "<|image_pad|>"), ("video_token_id", "<|video_pad|>"),
        ("audio_start_token_id", "<|audio_start|>"), ("audio_end_token_id", "<|audio_end|>"),
        ("vision_start_token_id", "<|vision_start|>"), ("vision_end_token_id", "<|vision_end|>")]}
    config = Qwen3OmniMoeThinkerConfig(
        audio_config=dict(num_mel_bins=128, encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=128,
                          d_model=64, n_window=50, n_window_infer=800, downsample_hidden_size=32, output_dim=96,
                          max_source_positions=1500, activation_function="gelu"),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=96,
                           deepstack_visual_indexes=[0, 1], patch_size=16, spatial_merge_size=2, temporal_patch_size=2),
        text_config=dict(vocab_size=vocab_size, hidden_size=96, intermediate_size=192, moe_intermediate_size=48,
                         num_experts=4, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
                         mlp_only_layers=[1], num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=32, rms_norm_eps=1e-6, max_position_embeddings=4096, hidden_act="silu",
                         rope_parameters={"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [6, 5, 5],
                                          "interleaved": True, "mrope_interleaved": True},
                         tie_word_embeddings=False),
        position_id_per_seconds=13, seconds_per_chunk=2, initializer_range=0.05, **ids)
    model = Qwen3OmniMoeThinkerForConditionalGeneration(config).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and "norm" in name:
                p.uniform_(0.5, 1.5)  # norms away from 1, so a skipped norm shows
            elif p.dim() == 1:
                p.normal_(0.0, 0.2)
    return model


# A Qwen3-Omni thinker's names as save_pretrained writes them (or as the module holds them) -> the released
# checkpoint's.
OMNI_RELEASED_PREFIXES = [("model.language_model.", "thinker.model."), ("model.audio_tower.", "thinker.audio_tower."),
                          ("model.visual.", "thinker.visual."), ("audio_tower.", "thinker.audio_tower."),
                          ("visual.", "thinker.visual."), ("model.", "thinker.model."), ("lm_head.", "thinker.lm_head.")]


def released_omni_layout(out):
    """Rewrites a Qwen3-Omni thinker saved by transformers into the released checkpoint's layout: its config as the
    thinker_config of a qwen3_omni_moe config, its weights under thinker.* (one tensor per expert, as saved)."""
    path = os.path.join(out, "model.safetensors")
    tensors = {}
    for name, t in load_file(path).items():
        for old, new in OMNI_RELEASED_PREFIXES:
            if name.startswith(old):
                name = new + name[len(old):]
                break
        assert name.startswith("thinker."), name
        tensors[name] = t
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(out, "config.json")) as f:
        thinker = json.load(f)
    released = {"architectures": ["Qwen3OmniMoeForConditionalGeneration"], "model_type": "qwen3_omni_moe",
                "thinker_config": thinker, "enable_audio_output": False}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(released, f, indent=2)


def bert(vocab_size, source):
    """BERT as all-MiniLM-L6-v2 and bge-small-en-v1.5 are: post-norm layers, biases everywhere, the exact GELU,
    absolute positions; head_dim 32 as theirs."""
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=vocab_size, hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256,
        hidden_act="gelu", max_position_embeddings=192, type_vocab_size=2, layer_norm_eps=1e-12,
        initializer_range=0.05,
    )
    return perturbed(BertModel(config, add_pooling_layer=False))


def nomic_bert(vocab_size, source):
    """NomicBERT as nomic-embed-text-v1.5 is: post-norm, fused Wqkv, rotary over the whole head (base 1000), SwiGLU, no
    biases but the norms'; trained on 64 positions, read up to the sentence-transformers limit."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_cls = get_class_from_dynamic_module("configuration_hf_nomic_bert.NomicBertConfig", source)
    model_cls = get_class_from_dynamic_module("modeling_hf_nomic_bert.NomicBertModel", source)
    config = config_cls(
        vocab_size=vocab_size, n_embd=128, n_head=4, n_layer=2, n_inner=256, activation_function="swiglu",
        rotary_emb_fraction=1.0, rotary_emb_base=1000, qkv_proj_bias=False, mlp_fc1_bias=False, mlp_fc2_bias=False,
        prenorm=False, causal=False, layer_norm_epsilon=1e-12, n_positions=8192, max_trained_positions=64,
        type_vocab_size=2, pad_vocab_size_multiple=64, initializer_range=0.05, attn_pdrop=0.0, resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    return perturbed(model_cls(config, add_pooling_layer=False))


def perturbed(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and name.endswith("weight"):
                p.uniform_(0.5, 1.5)  # norms away from 1, so a skipped norm shows
            elif p.dim() == 1:
                p.normal_(0.0, 0.2)
            elif "token_type_embeddings" in name:
                p.normal_(0.0, 0.5)  # every token adds row 0: it must show
    return model.eval()


ARCHS = {"llama": llama, "qwen3": qwen3, "qwen3_moe": qwen3_moe, "qwen3_5": qwen3_5, "qwen3_5_moe": qwen3_5_moe, "qwen3_5_vl": qwen3_5_vl, "qwen3_5_moe_vl": qwen3_5_moe_vl,
         "gemma4_moe": gemma4_moe, "lfm3": lfm3, "sensenova_loop": sensenova_loop, "qwen3_asr": qwen3_asr,
         "qwen3_omni": qwen3_omni, "inkling2_loop": inkling2_loop, "bert": bert, "nomic_bert": nomic_bert}
ENCODERS = {"bert", "nomic_bert"}
ENCODER_TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]


def embedding_model(args):
    """An encoder in a sentence-transformers checkpoint's layout."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_from)
    model = ARCHS[args.arch](len(tok), args.tokenizer_from)
    os.makedirs(args.out, exist_ok=True)
    model.config.save_pretrained(args.out)
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    if args.arch == "nomic_bert":  # the modeling code beside the weights, named in config.json
        for f in ("configuration_hf_nomic_bert.py", "modeling_hf_nomic_bert.py"):
            shutil.copy(os.path.join(args.tokenizer_from, f), os.path.join(args.out, f))
        path = os.path.join(args.out, "config.json")
        with open(path) as f:
            config = json.load(f)
        config["auto_map"] = {"AutoConfig": "configuration_hf_nomic_bert.NomicBertConfig",
                              "AutoModel": "modeling_hf_nomic_bert.NomicBertModel"}
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    for f in ENCODER_TOKENIZER_FILES:
        src = os.path.join(args.tokenizer_from, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    modules = [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
               {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}]
    if args.normalize:
        modules.append({"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"})
    with open(os.path.join(args.out, "modules.json"), "w") as f:
        json.dump(modules, f, indent=2)
    os.makedirs(os.path.join(args.out, "1_Pooling"), exist_ok=True)
    hidden = model.config.hidden_size
    with open(os.path.join(args.out, "1_Pooling", "config.json"), "w") as f:
        json.dump({"word_embedding_dimension": hidden, "pooling_mode_cls_token": args.pooling == "cls",
                   "pooling_mode_mean_tokens": args.pooling == "mean", "pooling_mode_max_tokens": False,
                   "pooling_mode_mean_sqrt_len_tokens": False}, f, indent=2)
    max_seq_length = args.max_seq_length or (192 if args.arch == "bert" else 256)
    with open(os.path.join(args.out, "sentence_bert_config.json"), "w") as f:
        json.dump({"max_seq_length": max_seq_length, "do_lower_case": False}, f, indent=2)
    print(f"{args.arch}: {sum(v.numel() for v in tensors.values()) / 1e6:.1f} M parameters, {args.pooling} pooling"
          f"{', normalized' if args.normalize else ''}, up to {max_seq_length} tokens -> {args.out}")


def mtp_head(model):
    """A random MTP head as the released checkpoints carry it (mtp.*: fc over [embedding | hidden], the two pre-fc
    norms, one full-attention decoder layer of the target's kind, the final norm), built from the target's classes."""
    import torch.nn as nn

    cfg = model.config.get_text_config()
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    layer_cls, norm_cls = type(lm.layers[0]), type(lm.norm)
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


def drafter(kind, target_dir, dflash_repo, out, layer_types=None):
    """A random block drafter for the target in `target_dir`; `layer_types`: its layers' kinds, else the variant's."""
    spec = importlib.util.spec_from_file_location("dflash_model", os.path.join(dflash_repo, "dflash", "model.py"))
    dflash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dflash)
    with open(os.path.join(target_dir, "config.json")) as f:
        target = json.load(f)
    target = target.get("text_config", target)
    hidden, vocab, layers = target["hidden_size"], target["vocab_size"], 3
    if layer_types is None:
        layer_types = ["sliding_attention" if kind != "dflash" else "full_attention"] * layers
    layers = len(layer_types)
    window = "sliding_attention" in layer_types
    config = dflash.Qwen3Config(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=4096, rms_norm_eps=1e-6,
        initializer_range=0.05, tie_word_embeddings=False, use_sliding_window=window, sliding_window=24,
        max_window_layers=layers, layer_types=layer_types,
        rope_parameters={"rope_theta": 10000000, "rope_type": "default"},
    )
    config.num_target_layers = target["num_hidden_layers"]
    mask = min(248070, vocab - 1)
    ids = [0, 1, target["num_hidden_layers"] - 1]
    if kind == "dflash":
        config.block_size = 6
        config.dflash_config = {"mask_token_id": mask, "target_layer_ids": ids}
    elif kind == "dflash2":
        config.is_causal = False
        config.dflash_config = {"block_size": 6, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": mask,
                                "selector_rank": 16, "selector_top_k": 4, "target_layer_ids": ids}
    else:
        config.is_causal = True
        config.block_size = 6
        config.code_bridge_second_pass = {"first_exact": layers - 1, "head_topk": 4, "mode": "serial_table"}
        config.dflash_config = {
            "context_fusion_checkpointing": True, "decoder_checkpointing": True, "draft_block_causal": True,
            "mask_token_id": mask, "selected_prefix_code_bridge": {"code_size": 32, "eps": 1e-6, "kappa": 10.0, "rank": 16},
            "selected_prefix_decode_mode": "serial", "selected_prefix_feature_gt_training": True,
            "selected_prefix_feature_layers": list(range(1, layers + 1)),
            "selected_prefix_feature_write_mode": "independent", "selected_prefix_feature_writes": True,
            "selected_prefix_last_write_refine": {"head_topk": 4, "kind": "exact_sender", "version": 1, "weight": 1.0},
            "selected_prefix_terminal": False, "target_layer_ids": ids,
        }
    config.architectures = ["DFlash2DraftModel" if kind == "dflash2" else "DFlashDraftModel"]
    model = (dflash.DFlash2DraftModel if kind == "dflash2" else dflash.DFlashDraftModel)(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:
                p.uniform_(0.5, 1.5)  # norms away from 1, so a skipped norm shows
            elif name.endswith("base_kernel"):
                p.normal_(0.0, 0.5)
                p[:, 0] += 1.0
            elif name.startswith("candidate_selector."):
                p.normal_(0.0, 0.3)
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    for name in ("predecessor_codebook", "successor_codebook"):  # the released layout: the tables without ".weight"
        if f"candidate_selector.{name}.weight" in tensors:
            tensors[f"candidate_selector.{name}"] = tensors.pop(f"candidate_selector.{name}.weight")
    if kind == "dspine":
        code, rank, b = 32, 16, "selected_prefix_bridge."
        for name in ("draft_block_causal_version", "last_write_refine_version", b + "independent_write_version",
                     b + "terminal_disabled_version"):
            tensors[name] = torch.tensor(1, dtype=torch.int64)
        tensors.update({
            b + "codebook": torch.nn.functional.normalize(torch.randn(vocab, code), dim=-1),
            b + "early_read_weights": 0.2 * torch.randn(layers - 1, code, hidden),
            b + "read_weight": 0.2 * torch.randn(code, hidden),
            b + "feature_write_weights": 0.2 * torch.randn(layers, hidden, rank),
            b + "gate.weight": 0.3 * torch.randn(rank, code), b + "gate.bias": 0.5 * torch.randn(rank),
            b + "predecessor_gate.weight": 0.3 * torch.randn(rank, code), b + "prev.weight": 0.3 * torch.randn(rank, code),
            b + "out.weight": torch.zeros(code, rank),
        })
    os.makedirs(out, exist_ok=True)
    config.save_pretrained(out)
    save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    print(f"{kind} drafter: {sum(v.numel() for v in tensors.values()) / 1e6:.1f} M parameters -> {out}")


def split_experts(path):
    """Rewrites fused expert banks (what transformers 5 saves) as one tensor an expert and projection, as Qwen3's MoE
    releases store them; per-expert tensors are left as they are."""
    tensors = load_file(path)
    for name in list(tensors):
        m = re.match(r"(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$", name)
        if not m:
            continue
        bank = tensors.pop(name)
        for e in range(bank.shape[0]):
            if m.group(2) == "down_proj":
                tensors[f"{m.group(1)}.{e}.down_proj.weight"] = bank[e].contiguous()
            else:
                half = bank.shape[1] // 2
                tensors[f"{m.group(1)}.{e}.gate_proj.weight"] = bank[e, :half].contiguous()
                tensors[f"{m.group(1)}.{e}.up_proj.weight"] = bank[e, half:].contiguous()
    save_file(tensors, path, metadata={"format": "pt"})


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
    ap.add_argument("--arch", choices=sorted(ARCHS))
    ap.add_argument("--tokenizer-from")
    ap.add_argument("--drafter", choices=["dflash", "dflash2", "dspine"])
    ap.add_argument("--target", help="--drafter: the tiny target it drafts for")
    ap.add_argument("--dflash-repo", help="--drafter: a clone of github.com/z-lab/dflash")
    ap.add_argument("--layer-types", help="--drafter: its layers' kinds, separated by commas (sliding_attention, "
                    "full_attention), instead of the variant's three of one kind")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean", help="bert, nomic_bert: the Pooling mode")
    ap.add_argument("--normalize", action="store_true", help="bert, nomic_bert: a Normalize module after the pooling")
    ap.add_argument("--max-seq-length", type=int, help="bert, nomic_bert: the longest input")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.drafter:
        if not args.target or not args.dflash_repo:
            ap.error("--drafter needs --target and --dflash-repo")
        drafter(args.drafter, args.target, args.dflash_repo, args.out,
                args.layer_types.split(",") if args.layer_types else None)
        return
    if not args.arch or not args.tokenizer_from:
        ap.error("--arch and --tokenizer-from are required")
    if args.arch in ENCODERS:
        embedding_model(args)
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_from, trust_remote_code=False)
    model = ARCHS[args.arch](len(tok), args.tokenizer_from)
    model.config.eos_token_id = tok.eos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    qwen = args.arch in ("qwen3_5", "qwen3_5_moe", "qwen3_5_vl", "qwen3_5_moe_vl", "sensenova_loop")  # an MTP head
    if qwen:
        model.config.get_text_config().mtp_num_hidden_layers = 1
    model.save_pretrained(args.out, safe_serialization=True)
    weights = os.path.join(args.out, "model.safetensors")
    if qwen:
        save_file({**load_file(weights), **mtp_head(model)}, weights, metadata={"format": "pt"})
    if args.arch == "qwen3_omni":
        released_omni_layout(args.out)  # experts stay one tensor each, as the release has them
    elif args.arch == "inkling2_loop":
        released_inkling2_layout(args.out, model.released_config)
    elif args.arch == "qwen3_moe":
        split_experts(weights)  # Qwen3-30B-A3B's release: each expert's gate_proj, up_proj and down_proj apart
    else:
        fuse_experts(weights)
    if args.arch == "qwen3_asr":
        released_asr_layout(args.out)
    with open(os.path.join(args.tokenizer_from, "config.json")) as f:
        auto_map = json.load(f).get("auto_map")
    code = [f for f in os.listdir(args.tokenizer_from) if f.endswith(".py")] if auto_map else []
    if auto_map:  # the checkpoint's own modeling code: the tiny model loads with it too
        with open(os.path.join(args.out, "config.json")) as f:
            config = json.load(f)
        config["auto_map"] = auto_map
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    for f in TOKENIZER_FILES + code:
        src = os.path.join(args.tokenizer_from, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters -> {args.out}")


if __name__ == "__main__":
    main()
