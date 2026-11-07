"""Expected outputs of torchvision's antialiased bicubic resize on uint8 images, the resize Hugging Face image
processors run. Writes tests/fixtures/image/resize_cases.json (pixels as base64 RGB bytes, row-major) for
tests/unit/image/test_image.cpp.

    python -I tests/parity/image/gen_cases.py        # needs torch and torchvision
"""
import base64
import json
import pathlib

import torch
import torchvision
import torchvision.transforms.v2.functional as tvF
from torchvision.transforms import InterpolationMode


def source(width, height, seed):
    # A smooth gradient plus a hash-like pattern: smooth areas test rounding, sharp edges test negative lobes.
    g = torch.Generator().manual_seed(seed)
    noise = torch.randint(0, 256, (height, width, 3), generator=g)
    y = torch.arange(height).view(-1, 1, 1)
    x = torch.arange(width).view(1, -1, 1)
    c = torch.arange(3).view(1, 1, -1)
    smooth = (x * 255 // max(width - 1, 1) + y * 3 + c * 40) % 256
    edges = ((x // 3 + y // 2) % 2) * 255
    img = torch.where((x + y) % 7 < 3, smooth, torch.where(noise > 128, edges, noise))
    return img.to(torch.uint8)


def resize(img_hwc, width, height):
    chw = img_hwc.permute(2, 0, 1).contiguous()
    out = tvF.resize(chw, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return out.permute(1, 2, 0).contiguous()


def b64(img):
    return base64.b64encode(img.flatten().numpy().tobytes()).decode()


def main():
    sources, cases = [], []
    for (w, h, seed), targets in [
        ((37, 23, 1), [(60, 40), (16, 11), (37, 50), (20, 23), (9, 5)]),
        ((64, 40, 2), [(32, 32), (31, 17)]),
    ]:
        src = source(w, h, seed)
        sources.append({"width": w, "height": h, "rgb": b64(src)})
        for tw, th in targets:
            cases.append({"source": len(sources) - 1, "width": tw, "height": th, "rgb": b64(resize(src, tw, th))})
    out = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "image" / "resize_cases.json"
    out.write_text(json.dumps({
        "generator": "tests/parity/image/gen_cases.py",
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "sources": sources, "cases": cases,
    }, separators=(",", ":")))
    print(f"wrote {len(cases)} cases to {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
