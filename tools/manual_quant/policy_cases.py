#!/usr/bin/env python3
"""Policy observations from a LeRobot v2 dataset with embedded images, for xpu-serve policy-eval (exp/manual-quant).

    python policy_cases.py --dataset <dir> --out <dir> --episodes 0:32 --frames 4 [--seed 0]

Writes <out>/manifest.json and per case the camera images (PNG, resized to --size), and fixed noise (f32 [50, 32]). Cameras:
`image` -> observation.images.base_0_rgb, `wrist_image` -> observation.images.left_wrist_0_rgb (π0.5's features). The state is mapped to [-1, 1] with the
dataset's min / max and zero-padded to 32 values.
"""

import argparse
import glob
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CAMERAS = {"image": "observation.images.base_0_rgb", "wrist_image": "observation.images.left_wrist_0_rgb"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="0:32", help="slice of the downloaded episode files")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--noise-dim", type=int, default=32)
    ap.add_argument("--state-dim", type=int, default=32)
    ap.add_argument("--size", type=int, default=224, help="image side the policy reads (bicubic resize)")
    ap.add_argument("--seed", type=int, default=0, help="frame choice")
    ap.add_argument("--noise-seed", type=int, default=0, help="the flow noise (same frames, another draw)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tasks = {}
    for line in open(os.path.join(args.dataset, "meta", "tasks.jsonl")):
        t = json.loads(line)
        tasks[t["task_index"]] = t["task"]
    st = json.load(open(os.path.join(args.dataset, "meta", "stats.json")))["state"]
    lo, hi = np.array(st["min"]), np.array(st["max"])
    files = sorted(glob.glob(os.path.join(args.dataset, "data", "*", "*.parquet")))
    a, b = (int(x) for x in args.episodes.split(":"))
    rng = np.random.default_rng(args.seed)
    noise_rng = np.random.default_rng([args.seed, args.noise_seed])
    cases = []
    for f in files[a:b]:
        t = pq.read_table(f)
        for i in np.sort(rng.choice(t.num_rows, args.frames, replace=False)):
            r = t.slice(int(i), 1).to_pylist()[0]
            name = f"e{r['episode_index']:04d}_f{r['frame_index']:04d}"
            cams = {}
            for col, cam in CAMERAS.items():
                path = f"{name}_{cam.rsplit('.', 1)[-1]}.png"
                img = Image.open(io.BytesIO(r[col]["bytes"])).convert("RGB")
                img.resize((args.size, args.size), Image.BICUBIC).save(os.path.join(args.out, path))
                cams[cam] = path
            state = np.clip(2 * (np.array(r["state"]) - lo) / np.maximum(hi - lo, 1e-8) - 1, -1, 1)
            state = np.concatenate([state, np.zeros(args.state_dim - len(state))]).astype(np.float32)
            noise = noise_rng.standard_normal((args.horizon, args.noise_dim)).astype(np.float32)
            noise.tofile(os.path.join(args.out, f"{name}_noise.f32"))
            cases.append({"cameras": cams, "state": state.tolist(), "instruction": tasks[r["task_index"]],
                          "noise": f"{name}_noise.f32"})
    json.dump({"cases": cases}, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
