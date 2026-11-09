#!/usr/bin/env python3
"""Depth Anything 3 (DA3MONO-LARGE and the like: DINOv2 ViT and a DPT head) on the CPU with
the official modules: the reference hwnpu's NPU depth (cmd/hwnpu-depth, tools/amd/iron_da3.py)
is checked against.

Needs the official repository's src/ (github.com/ByteDance-Seed/Depth-Anything-3, Apache-2.0)
for its DinoV2 and DPT modules (--src), and addict and einops besides torch. The image is
resized to exactly --width x --height (multiples of 14; give it an image of that size to
compare with the NPU, as the resizes differ) and ImageNet-normalized.

Writes OUT.npz (the normalized image, the patch embedding's input rows, the interpolated
position embedding, the four backbone features, the head's stages, depth and sky) and OUT.png
(the depth, inverted and normalized, as 8-bit gray).

  python tools/ref/ref_da3.py ~/models/DA3MONO-LARGE IMAGE OUT --src DA3/src --width 504 --height 504
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_image(path: Path, w: int, h: int) -> np.ndarray:
    """The image as (h, w, 3) normalized; resized (PIL's bicubic, where the official processor
    takes OpenCV's cubic or area) only when not already w x h."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BICUBIC)
    return (np.asarray(img).astype(np.float32) / 255 - MEAN) / STD


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a DA3 repository (config.json, model.safetensors)")
    ap.add_argument("image", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, required=True, help="the official repository's src/")
    ap.add_argument("--width", type=int, default=504)
    ap.add_argument("--height", type=int, default=504)
    a = ap.parse_args()
    sys.path.insert(0, str(a.src))
    from safetensors.torch import load_file

    from depth_anything_3.model.dinov2.dinov2 import DinoV2
    from depth_anything_3.model.dpt import DPT

    cfg = json.loads((a.root / "config.json").read_text())["config"]
    net, head = dict(cfg["net"]), dict(cfg["head"])
    net.pop("__object__")
    head.pop("__object__")
    assert head.get("output_dim", 1) == 1 and net["alt_start"] == -1, "a mono model (DPT head) only"
    bb = DinoV2(**net).eval()
    dpt = DPT(**head).eval()
    sd = load_file(a.root / "model.safetensors")
    for m, p in ((bb, "model.backbone."), (dpt, "model.head.")):
        part = {k.removeprefix(p): v for k, v in sd.items() if k.startswith(p)}
        m.load_state_dict(part, strict=True)
    rec = {}
    img = load_image(a.image, a.width, a.height)
    rec["image"] = img
    x = torch.from_numpy(img).permute(2, 0, 1)[None, None]  # (1, 1, 3, h, w)
    vit = bb.pretrained
    rec["pos_embed"] = vit.interpolate_pos_encoding(torch.zeros(1, 1 + (a.width // 14) * (a.height // 14), vit.embed_dim), a.width, a.height)[0].numpy()
    feats, _ = bb(x)
    rec["feats"] = np.stack([f[0][0, 0].numpy() for f in feats])  # (4, N, C)
    rec["cls"] = np.stack([f[1][0, 0].numpy() for f in feats])
    hooks = {}
    for name in ("layer1_rn", "layer2_rn", "layer3_rn", "layer4_rn", "refinenet4", "refinenet3", "refinenet2",
                 "refinenet1", "output_conv1", "output_conv2", "sky_output_conv2"):
        getattr(dpt.scratch, name).register_forward_hook(lambda m, i, o, n=name: hooks.__setitem__(n, o[0].permute(1, 2, 0).numpy()))
    for i, r in enumerate(dpt.resize_layers):
        r.register_forward_hook(lambda m, i_, o, n=f"resize{i}": hooks.__setitem__(n, o[0].permute(1, 2, 0).numpy()))
    out = dpt(feats, a.height, a.width, patch_start_idx=0)
    rec.update(hooks)
    rec["depth"] = out["depth"][0, 0].numpy()
    rec["sky"] = out["sky"][0, 0].numpy()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out.with_suffix(".npz"), **rec)
    from PIL import Image

    d = 1 / rec["depth"]
    g = ((d - d.min()) / (d.max() - d.min()) * 255).round().astype(np.uint8)
    Image.fromarray(g).save(a.out.with_suffix(".png"))
    print(f"{a.out.with_suffix('.png')}: {a.width}x{a.height}, depth {rec['depth'].min():.3f}..{rec['depth'].max():.3f}, "
          f"sky {float((rec['sky'] > 0.3).mean()):.3f} of the pixels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
