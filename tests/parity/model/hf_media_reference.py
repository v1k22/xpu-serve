#!/usr/bin/env python3
"""Reference outputs for models that read audio and images from Hugging Face transformers (test-only): Qwen3-ASR,
Qwen3-Omni's thinker (text, audio and images in), the Qwen3.5 family with its vision tower (qwen3_5,
qwen3_5_moe: text and images in), and Gemma 4 12B's audio (gemma4_unified: raw 640-sample frames, no tower; its
images are hf_vision_reference.py's).

    python -I tests/parity/model/hf_media_reference.py --model-dir <checkpoint> --out <dir> [--audio a.wav ...]
        [--image a.png ...] [--gen 32] [--converted-out <dir>]

Runs transformers' Qwen3ASRForConditionalGeneration, Qwen3OmniMoeThinkerForConditionalGeneration or the Qwen3.5
family's ForConditionalGeneration class in float32 on the CPU. A checkpoint in the released layout (weights under
thinker.*, a thinker_config; Qwen3-Omni's experts one tensor each) is converted in memory as transformers' conversion
does; for Qwen3-ASR --converted-out also writes that
converted copy, so the engine's reading of both layouts can be checked against the same outputs. Qwen3-ASR hears each
clip as its own turn; Qwen3-Omni is asked about it. The clips are the --audio files, or, without any, synthetic ones
(tones, noise, a chirp) at several lengths and sample rates. Per case it writes:
  * the clip as a 32-bit float WAV at its own sample rate (<name>.wav),
  * the log-mel features (float32 [128, frames], <name>.mel.bin), from the model's feature extractor
    (Qwen3ASRFeatureExtractor; Qwen3-Omni's WhisperFeatureExtractor) on the clip brought to 16 kHz by
    torchaudio.functional.resample (the engine's resampler); the manifest's mel_min_length is the shortest clip the
    features pad to,
  * the encoder's rows after the projector (float32 [rows, hidden], <name>.audio_rows.bin),
  * the messages and the prompt token ids (the chat template's turn, <|audio_pad|> once per row),
  * the full-vocabulary logits of the last prompt position (<name>.last_logits.bin) and the greedy continuation, with
    each step's five best tokens and logit range (where f16 may flip a near tie).
For Qwen3-Omni and the Qwen3.5 family, image cases too (the --image files, or synthetic ones of sizes the processor
resizes in each of its ways): the image as PNG, the processor's patches (float32 [patches, values],
<name>.pixels.bin, in its merge-block order), the vision tower's rows (float32 [tokens, (1 + deepstack) * hidden]:
the merged row, then each deepstack row, <name>.image_rows.bin), the prompt with <|image_pad|> once per token, the
last logits and the greedy text.
A text-only case checks the decoder alone.
"""

import argparse
import json
import math
import os

import numpy as np
import torch

RELEASED_RENAMES = [("thinker.audio_tower.proj1.", "model.multi_modal_projector.linear_1."),
                    ("thinker.audio_tower.proj2.", "model.multi_modal_projector.linear_2."),
                    ("thinker.audio_tower.", "model.audio_tower."), ("thinker.model.", "model.language_model."),
                    ("thinker.lm_head.", "lm_head.")]


OMNI_RENAMES = [("thinker.model.", "model.language_model."), ("thinker.audio_tower.", "model.audio_tower."),
                ("thinker.visual.", "model.visual."), ("thinker.lm_head.", "lm_head.")]


def load_omni(model_dir):
    """Qwen3-Omni's thinker in float32 from the released layout: names mapped, each expert's projections copied into
    the banks transformers holds (gate rows before up rows). Tensors are copied into the model one at a time, so the
    30B thinker takes its own size in memory once."""
    import re

    from safetensors import safe_open
    from transformers import Qwen3OmniMoeThinkerConfig, Qwen3OmniMoeThinkerForConditionalGeneration

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    thinker = {k: v for k, v in config["thinker_config"].items() if k != "model_type"}
    model = Qwen3OmniMoeThinkerForConditionalGeneration(Qwen3OmniMoeThinkerConfig(**thinker)).float().eval()
    params = dict(model.named_parameters())
    filled = {name: 0 for name in params}
    inter = model.config.text_config.moe_intermediate_size
    index = os.path.join(model_dir, "model.safetensors.index.json")
    files = ["model.safetensors"]
    if os.path.exists(index):
        files = sorted(set(json.load(open(index))["weight_map"].values()))
    with torch.no_grad():
        for file in files:
            with safe_open(os.path.join(model_dir, file), "pt") as f:
                for name in f.keys():
                    if not name.startswith("thinker."):
                        continue  # the talker and Code2Wav
                    target = name
                    for old, new in OMNI_RENAMES:
                        if target.startswith(old):
                            target = new + target[len(old):]
                            break
                    t = f.get_tensor(name)
                    m = re.match(r"(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", target)
                    if m:
                        e, part = int(m.group(2)), m.group(3)
                        bank = m.group(1) + (".down_proj" if part == "down_proj" else ".gate_up_proj")
                        rows = {"gate_proj": slice(0, inter), "up_proj": slice(inter, 2 * inter)}.get(part, slice(None))
                        params[bank][e, rows].copy_(t)
                        filled[bank] += 1
                    else:
                        params[target].copy_(t)
                        filled[target] += 1
    experts = model.config.text_config.num_experts
    def expected(name):
        return 2 * experts if name.endswith("gate_up_proj") else experts if name.endswith("experts.down_proj") else 1

    unfilled = [n for n, c in filled.items() if c != expected(n)]
    assert not unfilled, unfilled[:5]
    return model, config


def load_model(model_dir, converted_out=None):
    """The model in float32; a released-layout checkpoint converted in memory (and written to converted_out)."""
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    if "thinker_config" not in config:
        return Qwen3ASRForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval(), config
    thinker = config["thinker_config"]
    audio = dict(thinker["audio_config"])
    audio["max_position_embeddings"] = audio.pop("max_source_positions", audio.get("max_position_embeddings"))
    keep = set(Qwen3ASRConfig().audio_config.to_dict()) - {"model_type"}  # the release names it otherwise
    audio = {k: v for k, v in audio.items() if k in keep}
    flat = Qwen3ASRConfig(audio_config=audio, text_config=thinker["text_config"],
                          audio_token_id=thinker["audio_token_id"],
                          tie_word_embeddings=thinker["text_config"].get("tie_word_embeddings", False))
    index = os.path.join(model_dir, "model.safetensors.index.json")
    files = ["model.safetensors"]
    if os.path.exists(index):
        files = sorted(set(json.load(open(index))["weight_map"].values()))
    state = {}
    for file in files:
        for name, t in load_file(os.path.join(model_dir, file)).items():
            for old, new in RELEASED_RENAMES:
                if name.startswith(old):
                    name = new + name[len(old):]
                    break
            state[name] = t.float()
    model = Qwen3ASRForConditionalGeneration(flat).float().eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if not (m == "lm_head.weight" and flat.text_config.tie_word_embeddings)]
    assert not missing and not unexpected, (missing, unexpected)
    if converted_out:
        os.makedirs(converted_out, exist_ok=True)
        flat.save_pretrained(converted_out)
        save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                  os.path.join(converted_out, "model.safetensors"), metadata={"format": "pt"})
        for name in os.listdir(model_dir):
            skip = ("config.json", "model.safetensors.index.json")
            if name.endswith((".json", ".txt", ".jinja")) and name not in skip:
                with open(os.path.join(model_dir, name), "rb") as src:
                    with open(os.path.join(converted_out, name), "wb") as dst:
                        dst.write(src.read())
    return model, config


def synthetic_clips():
    """(name, sample rate, float32 samples): tones and noise at several lengths, two windows long, nine windows long
    (more than one encoder run), below the 0.5 s minimum, and at 44.1 kHz (resampled)."""
    rng = np.random.default_rng(7)

    def clip(seconds, rate):
        t = np.arange(int(seconds * rate)) / rate
        tone = 0.3 * np.sin(2 * math.pi * (220 + 180 * t) * t) + 0.2 * np.sin(2 * math.pi * 997 * t)
        return (tone + 0.05 * rng.standard_normal(t.size)).astype(np.float32)

    return [("short_16k", 16000, clip(2.37, 16000)), ("tiny_16k", 16000, clip(0.3, 16000)),
            ("two_windows_16k", 16000, clip(9.5, 16000)), ("nine_windows_16k", 16000, clip(70.3, 16000)),
            ("cd_44k", 44100, clip(3.1, 44100))]


def synthetic_images():
    """(name, PIL image): colour gradients with noise; one whose sides are a multiple of 32, one the processor resizes,
    one whose height rounds half to even (48 / 32), one below min_pixels (scaled up)."""
    from PIL import Image

    rng = np.random.default_rng(11)

    def image(w, h):
        y, x = np.mgrid[0:h, 0:w]
        a = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x + y) * 127 // max(w + h - 2, 1)], -1)
        return Image.fromarray((a + rng.integers(0, 40, a.shape)).clip(0, 255).astype(np.uint8))

    return [("grid_64x96", image(96, 64)), ("odd_75x100", image(100, 75)), ("half_48x200", image(200, 48)),
            ("small_30x30", image(30, 30))]


def read_wav(path):
    import soundfile

    data, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return rate, data.mean(axis=1).astype(np.float32)


def write_wav_f32(path, rate, samples):
    import struct

    data = samples.astype("<f4").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32) + b"data" + struct.pack("<I", len(data))
    with open(path, "wb") as f:
        f.write(header + data)



def suppressed(model, logits):
    """The logits as the model gives them to sampling: generation_config's suppress_tokens at -inf (transformers'
    generate and vLLM's Gemma 4 rule them out; the engine does it in the graph)."""
    ids = getattr(model.generation_config, "suppress_tokens", None)
    if ids:
        logits[..., ids] = float("-inf")
    return logits

def gemma4_unified(args):
    """Gemma 4 12B's audio cases: the processor chunks the clip (brought to 16 kHz by torchaudio, the engine's
    resampler) into 640-sample frames (<name>.frames.bin, float32 [frames, 640]), each frame one row through the
    multimodal embedder (<name>.audio_rows.bin); the prompt the processor expands; last logits and greedy text."""
    import torchaudio
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32,
                                                        attn_implementation="eager").eval()
    hidden = model.config.text_config.hidden_size
    scale = torch.tensor(math.sqrt(hidden), dtype=torch.bfloat16).float()  # the engine rounds it as the checkpoint
    model.model.language_model.embed_tokens.embed_scale.fill_(scale.item())
    tok = processor.tokenizer
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    clips = [(os.path.splitext(os.path.basename(p))[0], *read_wav(p)) for p in args.audio] or synthetic_clips()
    manifest = {"model_dir": os.path.abspath(args.model_dir), "dtype": "float32", "cases": []}
    for name, rate, samples in clips:
        write_wav_f32(os.path.join(args.out, f"{name}.wav"), rate, samples)
        pcm = torch.from_numpy(samples)
        if rate != 16000:
            pcm = torchaudio.functional.resample(pcm, rate, 16000)
        messages = [{"role": "user", "content": [{"type": "audio"},
                                                 {"type": "text", "text": "What do you hear? Answer in one sentence."}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], audio=[pcm.numpy()], return_tensors="pt")
        frames = inputs["input_features"][0][inputs["input_features_mask"][0]]
        frames.numpy().astype(np.float32).tofile(os.path.join(args.out, f"{name}.frames.bin"))
        with torch.no_grad():
            rows = model.model.get_audio_features(inputs["input_features"], inputs["input_features_mask"],
                                                  return_dict=True).pooler_output[0]
            rows.numpy().astype(np.float32).tofile(os.path.join(args.out, f"{name}.audio_rows.bin"))
            logits = suppressed(model, model(**inputs).logits[0, -1].float())
            out = model.generate(**inputs, max_new_tokens=args.gen, do_sample=False, eos_token_id=eos,
                                 repetition_penalty=1.0, output_scores=True, return_dict_in_generate=True)
        ids = inputs["input_ids"][0].tolist()
        greedy = out.sequences[0, len(ids):].tolist()
        top = []
        for step in out.scores:
            values, indices = step[0].float().topk(5)
            finite = step[0][torch.isfinite(step[0])].float()  # suppressed tokens are -inf
            top.append({"ids": indices.tolist(), "logits": values.tolist(),
                        "range": float(finite.max() - finite.min())})
        logits.numpy().astype(np.float32).tofile(os.path.join(args.out, f"{name}.last_logits.bin"))
        manifest["cases"].append({
            "name": name, "wav": f"{name}.wav", "sample_rate": rate, "frames": f"{name}.frames.bin",
            "audio_rows": f"{name}.audio_rows.bin", "rows": int(rows.shape[0]), "messages": messages,
            "input_ids": ids, "last_logits": f"{name}.last_logits.bin", "greedy": greedy,
            "greedy_text": tok.decode(greedy), "greedy_top": top})
        print(f"{name}: {len(ids)} prompt tokens, {rows.shape[0]} audio rows -> {tok.decode(greedy)!r}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", nargs="*", default=[])
    ap.add_argument("--image", nargs="*", default=[])
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--converted-out")
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    import torchaudio
    from transformers import AutoTokenizer, Qwen3ASRFeatureExtractor, WhisperFeatureExtractor

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.model_dir, "config.json")) as f:
        model_type = json.load(f)["model_type"]
    omni = model_type == "qwen3_omni_moe"
    vl = model_type in ("qwen3_5", "qwen3_5_moe")  # images, no audio
    if vl and args.audio:
        ap.error(f"{model_type} takes no audio")
    if model_type == "gemma4_unified":
        return gemma4_unified(args)
    if vl:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    else:
        model, config = load_omni(args.model_dir) if omni else load_model(args.model_dir, args.converted_out)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    # Qwen3-Omni's processor: the checkpoint's WhisperFeatureExtractor, clips as they are; Qwen3-ASR's package pads a
    # clip under half a second.
    extractor = WhisperFeatureExtractor.from_pretrained(args.model_dir) if omni else Qwen3ASRFeatureExtractor()
    min_length = 0 if omni else 8000
    pad = tok.convert_tokens_to_ids("<|audio_pad|>")
    gen = json.load(open(os.path.join(args.model_dir, "generation_config.json"))) \
        if os.path.exists(os.path.join(args.model_dir, "generation_config.json")) else {}
    eos = gen.get("eos_token_id", tok.eos_token_id)
    eos = eos if isinstance(eos, list) else [eos]
    clips = [] if vl else [(os.path.splitext(os.path.basename(p))[0], *read_wav(p)) for p in args.audio] \
        or synthetic_clips()

    # AutoTokenizer does not read chat_template.json (the processor does): pass it.
    chat_template = None
    if os.path.exists(os.path.join(args.model_dir, "chat_template.json")):
        chat_template = json.load(open(os.path.join(args.model_dir, "chat_template.json")))["chat_template"]

    def template(messages):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       chat_template=chat_template)

    def audio_turn():  # what the model is asked about a clip
        content = [{"type": "audio"}]
        if omni:
            content.append({"type": "text", "text": "What do you hear? Answer in one sentence."})
        return [{"role": "user", "content": content}]

    manifest = {"model_dir": os.path.abspath(args.model_dir), "dtype": "float32", "mel_min_length": min_length,
                "cases": []}
    images = []
    if omni or vl:
        from PIL import Image
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(args.model_dir)
        images = [(os.path.splitext(os.path.basename(p))[0], Image.open(p).convert("RGB")) for p in args.image] \
            or synthetic_images()
        image_pad = tok.convert_tokens_to_ids("<|image_pad|>")
    for name, rate, samples in [("text_only", 0, None)] + clips + [(n, -1, im) for n, im in images]:
        case = {"name": name}
        with torch.no_grad():
            if rate == -1:  # an image
                samples.save(os.path.join(args.out, f"{name}.png"))
                feats = processor(images=samples, return_tensors="pt")
                pixels, grid = feats["pixel_values"], feats["image_grid_thw"]
                pixels.numpy().astype(np.float32).tofile(os.path.join(args.out, f"{name}.pixels.bin"))
                vision = model.model.visual(pixels, grid_thw=grid)
                deepstack = getattr(vision, "deepstack_features", None) or []
                rows = torch.cat([vision.pooler_output] + list(deepstack), dim=1)
                rows.numpy().astype(np.float32).tofile(os.path.join(args.out, f"{name}.image_rows.bin"))
                messages = [{"role": "user", "content": [{"type": "image"},
                                                         {"type": "text", "text": "Describe the image in one line."}]}]
                ids = tok(template(messages), add_special_tokens=False)["input_ids"]
                at = ids.index(image_pad)
                ids = ids[:at] + [image_pad] * rows.shape[0] + ids[at + 1:]
                inputs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                          "pixel_values": pixels, "image_grid_thw": grid}
                if vl:  # which tokens are an image's, for the multimodal positions
                    inputs["mm_token_type_ids"] = (inputs["input_ids"] == image_pad).long()
                case.update({"image": f"{name}.png", "pixels": f"{name}.pixels.bin",
                             "image_rows": f"{name}.image_rows.bin", "rows": int(rows.shape[0]),
                             "grid": grid[0].tolist()})
            elif samples is None:
                messages = [{"role": "user", "content": "Say hello in one word."}]
                ids = tok(template(messages), add_special_tokens=False)["input_ids"]
                inputs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
            else:
                write_wav_f32(os.path.join(args.out, f"{name}.wav"), rate, samples)
                pcm = torch.from_numpy(samples)
                if rate != 16000:
                    pcm = torchaudio.functional.resample(pcm, rate, 16000)
                feats = extractor(pcm.numpy(), sampling_rate=16000, padding=True, truncation=False,
                                  return_attention_mask=True, return_tensors="pt")
                mel, mask = feats["input_features"], feats["attention_mask"]
                frames = int(mask.sum())
                mel[0, :, :frames].numpy().astype(np.float32).tofile(os.path.join(args.out, f"{name}.mel.bin"))
                rows = model.get_audio_features(mel, mask, return_dict=True).pooler_output
                rows.numpy().astype(np.float32).tofile(os.path.join(args.out, f"{name}.audio_rows.bin"))
                messages = audio_turn()
                ids = tok(template(messages), add_special_tokens=False)["input_ids"]
                at = ids.index(pad)
                ids = ids[:at] + [pad] * rows.shape[0] + ids[at + 1:]
                inputs = {"input_ids": torch.tensor([ids]), "input_features": mel,
                          "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
                inputs["feature_attention_mask" if omni else "input_features_mask"] = mask
                case.update({"wav": f"{name}.wav", "sample_rate": rate, "mel": f"{name}.mel.bin", "mel_frames": frames,
                             "audio_rows": f"{name}.audio_rows.bin", "rows": int(rows.shape[0])})
            logits = suppressed(model, model(**inputs).logits[0, -1].float())
            out = model.generate(**inputs, max_new_tokens=args.gen, do_sample=False, eos_token_id=eos,
                                 repetition_penalty=1.0, output_scores=True, return_dict_in_generate=True)
        greedy = out.sequences[0, len(ids):].tolist()
        # Each step's five best tokens and its logit range: where an f16 run may flip a near tie.
        top = []
        for step in out.scores:
            values, indices = step[0].float().topk(5)
            finite = step[0][torch.isfinite(step[0])].float()  # suppressed tokens are -inf
            top.append({"ids": indices.tolist(), "logits": values.tolist(),
                        "range": float(finite.max() - finite.min())})
        case["greedy_top"] = top
        logits.numpy().astype(np.float32).tofile(os.path.join(args.out, f"{name}.last_logits.bin"))
        case.update({"messages": messages, "input_ids": ids, "last_logits": f"{name}.last_logits.bin",
                     "greedy": greedy, "greedy_text": tok.decode(greedy)})
        manifest["cases"].append(case)
        print(f"{name}: {len(ids)} prompt tokens -> {tok.decode(greedy)!r}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
