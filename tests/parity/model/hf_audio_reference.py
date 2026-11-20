#!/usr/bin/env python3
"""Reference outputs for models that hear from Hugging Face transformers (test-only): Qwen3-ASR, and Qwen3-Omni's
thinker (text and audio in).

    python -I tests/parity/model/hf_audio_reference.py --model-dir <checkpoint> --out <dir> [--audio a.wav ...]
        [--gen 32] [--converted-out <dir>]

Runs transformers' Qwen3ASRForConditionalGeneration or Qwen3OmniMoeThinkerForConditionalGeneration in float32 on
the CPU. A checkpoint in the released layout (weights under thinker.*, a thinker_config; Qwen3-Omni's experts one
tensor each) is converted in memory as transformers' conversion does; for Qwen3-ASR --converted-out also writes that
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
  * the full-vocabulary logits of the last prompt position (<name>.last_logits.bin) and the greedy continuation.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", nargs="*", default=[])
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--converted-out")
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    import torchaudio
    from transformers import AutoTokenizer, Qwen3ASRFeatureExtractor, WhisperFeatureExtractor

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.model_dir, "config.json")) as f:
        omni = json.load(f)["model_type"] == "qwen3_omni_moe"
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
    clips = [(os.path.splitext(os.path.basename(p))[0], *read_wav(p)) for p in args.audio] or synthetic_clips()

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
    for name, rate, samples in [("text_only", 0, None)] + clips:
        case = {"name": name}
        with torch.no_grad():
            if samples is None:
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
            logits = model(**inputs).logits[0, -1].float()
            out = model.generate(**inputs, max_new_tokens=args.gen, do_sample=False, eos_token_id=eos,
                                 repetition_penalty=1.0)
        greedy = out[0, len(ids):].tolist()
        logits.numpy().astype(np.float32).tofile(os.path.join(args.out, f"{name}.last_logits.bin"))
        case.update({"messages": messages, "input_ids": ids, "last_logits": f"{name}.last_logits.bin",
                     "greedy": greedy, "greedy_text": tok.decode(greedy)})
        manifest["cases"].append(case)
        print(f"{name}: {len(ids)} prompt tokens -> {tok.decode(greedy)!r}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
