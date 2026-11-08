#!/usr/bin/env python3
"""Reference outputs for robot policies from LeRobot (test-only; needs lerobot installed, Python >= 3.12).

    python -I tests/parity/policy/lerobot_reference.py --policy-dir <lerobot checkpoint> --tokenizer <dir with
        tokenizer.json> [--out tests/fixtures/generated/<type>]

Runs the policy in float32 on the CPU, once per camera set (all cameras, then the first camera alone: LeRobot pads
the missing cameras with masked images), with fixed images, state, instruction and noise, and writes per case:
  * the inputs: images (uint8 HWC), state (f32, already normalized), instruction, noise (f32 [horizon, max dim]),
  * the prompt token ids (BOS included, unpadded),
  * each present camera's image features after the projector (f32 [tokens, width]),
  * the velocity of the first denoising step and the final actions (f32 [horizon, max action dim]).

LeRobot's tokenizer step loads the PaliGemma tokenizer by name; --tokenizer points it at a local tokenizer.json of the
same SentencePiece model instead (the Hugging Face repository is gated).
"""

import argparse
import json
import os

import numpy as np
import torch


def synthetic_image(seed, size):
    """Smooth gradients plus seeded noise: structured enough that SigLIP features are not degenerate."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base = np.stack([np.sin(6.28 * (x + seed * 0.1)), np.cos(6.28 * y * (1 + seed)), x * y], axis=-1)
    img = (base * 0.5 + 0.5) * 200 + rng.integers(0, 55, size=(size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out")
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import TokenizerProcessorStep
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    with open(os.path.join(args.policy_dir, "config.json")) as f:
        config = json.load(f)
    policy_type = config["type"]
    args.out = args.out or os.path.join("tests", "fixtures", "generated", policy_type)
    os.makedirs(args.out, exist_ok=True)

    policy = PI05Policy.from_pretrained(args.policy_dir)
    policy.eval()
    cfg = policy.config
    cameras = list(cfg.image_features)
    size = cfg.image_resolution[0]

    # Record what the network sees: image features and the velocity of every denoising step.
    record = {"features": [], "velocities": []}
    embed_image = policy.model.paligemma_with_expert.embed_image
    denoise_step = policy.model.denoise_step

    def embed_image_hook(image, **kw):
        out = embed_image(image, **kw)
        record["features"].append(out.detach().clone())
        return out

    def denoise_step_hook(**kw):
        out = denoise_step(**kw)
        record["velocities"].append(out.detach().clone())
        return out

    policy.model.paligemma_with_expert.embed_image = embed_image_hook
    policy.model.denoise_step = denoise_step_hook

    state_step = Pi05PrepareStateTokenizerProcessorStep()
    tok_step = TokenizerProcessorStep(tokenizer_name=args.tokenizer, max_length=cfg.tokenizer_max_length,
                                      padding="max_length", padding_side="right", truncation=True)

    rng = np.random.default_rng(7)
    state = rng.uniform(-1.0, 1.0, size=cfg.max_state_dim).astype(np.float32)
    instruction = "pick up the red cup and place it on the plate"
    noise = torch.from_numpy(rng.standard_normal((1, cfg.chunk_size, cfg.max_action_dim)).astype(np.float32))
    images = {cam: synthetic_image(i + 1, size) for i, cam in enumerate(cameras)}

    manifest = {"policy_dir": os.path.abspath(args.policy_dir), "type": policy_type, "dtype": "float32",
                "cameras": cameras, "cases": []}
    state.tofile(os.path.join(args.out, "state.bin"))
    noise[0].numpy().tofile(os.path.join(args.out, "noise.bin"))
    for cam, img in images.items():
        img.tofile(os.path.join(args.out, f"image_{cam}.bin"))

    for name, present in [("all_cameras", cameras), ("one_camera", cameras[:1])]:
        record["features"].clear()
        record["velocities"].clear()
        # The prompt as LeRobot builds it: state bins in the text, then the PaliGemma tokenizer (BOS, padded).
        transition = {"observation": {OBS_STATE: torch.from_numpy(state)[None]},
                      "complementary_data": {"task": [instruction]}}
        transition = state_step(transition)
        transition = tok_step(transition)
        obs = transition["observation"]
        tokens, mask = obs[OBS_LANGUAGE_TOKENS], obs[OBS_LANGUAGE_ATTENTION_MASK]
        batch = {OBS_LANGUAGE_TOKENS: tokens, OBS_LANGUAGE_ATTENTION_MASK: mask, OBS_STATE: torch.from_numpy(state)[None]}
        for cam in present:
            batch[cam] = torch.from_numpy(images[cam]).permute(2, 0, 1)[None].float() / 255.0
        with torch.no_grad():
            actions = policy.model.sample_actions(*policy._preprocess_images(batch), tokens, mask, noise=noise.clone())
        ids = tokens[0][mask[0].bool()].tolist()
        case = {"name": name, "cameras": present, "instruction": instruction,
                "prompt": transition["complementary_data"]["task"][0] if "task" in transition["complementary_data"] else None,
                "input_ids": ids, "steps": cfg.num_inference_steps, "features": [], "velocity0": f"{name}.velocity0.bin",
                "actions": f"{name}.actions.bin"}
        for cam, feat in zip(present, record["features"][0].unbind(0) if len(record["features"]) == 1 else
                             [f[0] for f in record["features"]]):
            path = f"{name}.features_{cam}.bin"
            feat.numpy().astype(np.float32).tofile(os.path.join(args.out, path))
            case["features"].append(path)
        record["velocities"][0][0].numpy().astype(np.float32).tofile(os.path.join(args.out, case["velocity0"]))
        actions[0].numpy().astype(np.float32).tofile(os.path.join(args.out, case["actions"]))
        manifest["cases"].append(case)
        print(f"{name}: {len(ids)} prompt tokens, actions[0][:4] = {actions[0, 0, :4].tolist()}", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
