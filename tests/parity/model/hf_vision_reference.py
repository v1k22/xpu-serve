#!/usr/bin/env python3
"""Reference outputs for Gemma 4 image input from Hugging Face transformers (test-only).

    python -I tests/parity/model/hf_vision_reference.py --model-dir <hf dir> \\
        [--out tests/fixtures/generated/gemma4_vision] [--gen 32] [--image photo.jpg ...]

Draws a few synthetic test images (different aspect ratios, one that must be upscaled), adds any --image files, and
writes for each image:
  * the image file itself (<name>.png or the original bytes), which the C++ test decodes,
  * the processor's pixel values of the real patches ([patches, 768] float32, <name>.pixels.bin) and positions,
  * the vision tower's soft tokens after the multimodal projection ([tokens, hidden] float32, <name>.features.bin);
and for each chat case (one or two images and a question):
  * the expanded prompt ids, the last-position logits (<case>.last_logits.bin) and the greedy continuation.

The model runs in float32 on the CPU with eager attention, with the text embedding scale rounded to bfloat16 as in
hf_reference.py.
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor


def synthetic_images():
    shapes = Image.new("RGB", (640, 427), (255, 255, 255))
    d = ImageDraw.Draw(shapes)
    d.ellipse((40, 100, 240, 300), fill=(220, 30, 30))
    d.rectangle((300, 110, 470, 280), fill=(30, 60, 220))
    d.polygon([(520, 300), (600, 120), (630, 300)], fill=(40, 170, 60))

    sign = Image.new("RGB", (360, 600), (250, 220, 40))
    d = ImageDraw.Draw(sign)
    font = ImageFont.load_default(size=120)
    d.text((40, 120), "XPU", fill=(0, 0, 0), font=font)
    d.text((80, 330), "42", fill=(0, 0, 0), font=font)

    y, x = np.mgrid[0:512, 0:512]
    gradient = np.stack([x * 255 // 511, y * 255 // 511, (x + y) * 255 // 1022], axis=-1).astype(np.uint8)

    tiny = Image.new("RGB", (40, 24))
    tiny.putdata([((255, 128, 0) if (i // 4 + j // 4) % 2 else (0, 0, 0)) for j in range(24) for i in range(40)])

    return {
        "shapes": shapes,
        "sign": sign,
        "gradient": Image.fromarray(gradient),
        "tiny": tiny,
    }


def chat_cases(image_names):
    def user(images, text):
        return [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]}]

    cases = {
        "image_shapes": (["shapes"], user(["shapes"], "What shapes and colors are in this image? Answer briefly.")),
        "image_sign": (["sign"], user(["sign"], "What does the text in the image say?")),
        "image_gradient": (["gradient"], user(["gradient"], "Describe this image in one sentence.")),
        "two_images": (["shapes", "tiny"],
                       user(["shapes", "tiny"], "How many images are there? Describe each in a few words.")),
    }
    for name in image_names:
        cases[f"photo_{name}"] = ([name], user([name], "Describe this image in one sentence."))
    return cases


def bf16(x):
    return torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default=os.path.join("tests", "fixtures", "generated", "gemma4_vision"))
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--image", action="append", default=[], help="extra image file (JPEG/PNG), described in one case")
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    hidden = model.config.text_config.hidden_size
    model.model.language_model.embed_tokens.embed_scale.fill_(bf16(math.sqrt(hidden)))
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]

    files, pil = {}, {}
    for name, img in synthetic_images().items():
        files[name] = f"{name}.png"
        img.save(os.path.join(args.out, files[name]))
    photos = []
    for i, path in enumerate(args.image):
        name = f"photo{i}"
        files[name] = name + os.path.splitext(path)[1].lower()
        shutil.copyfile(path, os.path.join(args.out, files[name]))
        photos.append(name)
    for name, f in files.items():
        pil[name] = Image.open(os.path.join(args.out, f)).convert("RGB")

    manifest = {"model_dir": os.path.abspath(args.model_dir), "dtype": "float32", "images": [], "cases": []}
    for name in files:
        feats = processor.image_processor(images=[pil[name]], return_tensors="pt")
        pos = feats["image_position_ids"][0]
        real = int((pos[:, 0] >= 0).sum())
        feats["pixel_values"][0, :real].numpy().astype(np.float32).tofile(os.path.join(args.out, f"{name}.pixels.bin"))
        with torch.no_grad():
            out = model.model.get_image_features(feats["pixel_values"], feats["image_position_ids"], return_dict=True)
        features = out.pooler_output[0].float().numpy().astype(np.float32)
        features.tofile(os.path.join(args.out, f"{name}.features.bin"))
        manifest["images"].append({
            "name": name, "file": files[name], "width": pil[name].width, "height": pil[name].height,
            "patches": real, "positions": pos[:real].tolist(), "soft_tokens": int(feats["num_soft_tokens_per_image"][0]),
            "pixels": f"{name}.pixels.bin", "features": f"{name}.features.bin",
        })
        print(f"{name}: {pil[name].width}x{pil[name].height} -> {real} patches, {features.shape[0]} soft tokens",
              flush=True)

    for case, (images, messages) in chat_cases(photos).items():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[pil[n] for n in images], return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        with torch.no_grad():
            logits = model(**inputs).logits[0].float()
            out = model.generate(**inputs, max_new_tokens=args.gen, do_sample=False, eos_token_id=eos)
        greedy = out[0, len(ids):].tolist()
        logits[-1].numpy().astype(np.float32).tofile(os.path.join(args.out, f"{case}.last_logits.bin"))
        manifest["cases"].append({
            "name": case, "images": images, "messages": messages, "input_ids": ids,
            "last_logits": f"{case}.last_logits.bin", "greedy": greedy,
            "greedy_text": processor.tokenizer.decode(greedy),
        })
        print(f"{case}: {len(ids)} prompt tokens -> {processor.tokenizer.decode(greedy)!r}", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
