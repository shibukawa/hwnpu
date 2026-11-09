#!/usr/bin/env python3
"""MoGe-2 (Ruicheng/moge-2-*-normal, MIT: DINOv2 ViT, a ConvStack neck, points, normal and mask
heads and a metric scale head) on the CPU with the official modules: the reference hwnpu's NPU
depth (cmd/hwnpu-depth, tools/amd/iron_moge.py) is checked against.

Needs the official repository (github.com/microsoft/MoGe, MIT) for its modules (--src, the
directory holding moge/) and scipy (its focal and shift solver); utils3d, which only the
official infer() uses, is stubbed. The picture is given at the size the ViT takes (base rows
and columns times 14: resize it first, as the NPU's check does), so forward()'s resizes are
identities; the post-processing is infer()'s: the focal length and the z shift solved on the
point map, the depth z + shift, times the metric scale, where the mask is set.

Writes OUT.npz (the normalized image, the class token, the heads' outputs at their last level,
the outputs at the picture's size, focal, shift, metric scale, depth, mask) and OUT.png (the
inverse depth as 8-bit gray, the masked pixels black).

  python tools/ref/ref_moge.py ~/models/moge-2-vits-normal IMAGE OUT --src MoGe --base 50x50
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a MoGe-2 repository (model.pt)")
    ap.add_argument("image", type=Path, help="a picture of (base rows x 14) x (base columns x 14) pixels")
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--base", default="50x50", help="the ViT's patches, COLUMNSxROWS")
    a = ap.parse_args()
    for m in ("utils3d", "cv2"):  # used only by paths not taken here
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.path.insert(0, str(a.src))
    from PIL import Image

    from moge.model.v2 import MoGeModel
    from moge.utils.geometry_torch import recover_focal_shift

    ck = torch.load(a.root / "model.pt", map_location="cpu", weights_only=True)
    model = MoGeModel(**ck["model_config"]).eval()
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    bw, bh = (int(v) for v in a.base.split("x"))
    img = Image.open(a.image).convert("RGB")
    assert img.size == (bw * 14, bh * 14), f"{a.image}: {img.size}, give it {bw * 14}x{bh * 14}"
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255).permute(2, 0, 1)[None]
    rec = {"image": ((np.asarray(img).astype(np.float32) / 255 - MEAN) / STD)}
    hooks = {}
    for name in ("points_head", "normal_head", "mask_head"):
        getattr(model, name).output_blocks[-1].register_forward_hook(lambda m, i, o, n=name: hooks.__setitem__(n, o[0].permute(1, 2, 0).numpy()))
    model.scale_head.register_forward_hook(lambda m, i, o: hooks.__setitem__("scale_logit", o.reshape(-1).numpy()))
    out = model(x, num_tokens=bw * bh)
    rec.update(hooks)
    points, normal, mask = out["points"], out["normal"], out["mask"]
    rec["points"], rec["normal"], rec["mask"] = points[0].numpy(), normal[0].numpy(), mask[0].numpy()
    scale = out["metric_scale"]
    mask_b = mask > 0.5
    focal, shift = recover_focal_shift(points, mask_b)
    depth = (points[..., 2] + shift[..., None, None]) * scale[:, None, None]
    rec["focal"], rec["shift"], rec["scale"] = float(focal[0]), float(shift[0]), float(scale[0])
    rec["depth"] = depth[0].numpy()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out.with_suffix(".npz"), **rec)
    d = 1 / np.where(rec["mask"] > 0.5, rec["depth"], np.inf)
    g = ((d - d.min()) / (d.max() - d.min()) * 255).round().astype(np.uint8)
    Image.fromarray(g).save(a.out.with_suffix(".png"))
    print(f"{a.out.with_suffix('.png')}: base {bw}x{bh}, focal {rec['focal']:.3f}, shift {rec['shift']:.3f}, "
          f"scale {rec['scale']:.3f}, depth {np.median(rec['depth'][rec['mask'] > 0.5]):.2f} m median, "
          f"{float((rec['mask'] > 0.5).mean()):.3f} valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
