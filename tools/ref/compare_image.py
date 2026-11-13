#!/usr/bin/env python3
"""Compare an NPU picture with a reference: the final latents' cosine (raw float32 against an npz's
``latents``, the same layout) and the pictures' PSNR over RGB.

  python tools/ref/compare_image.py REF.npz NPU.latents.f32 REF.png NPU.png
"""

import sys

import numpy as np
from PIL import Image


def main(argv) -> int:
    ref, lat, rpng, npng = argv[1:5]
    z = np.load(ref)["latents"].astype(np.float64).ravel()
    x = np.fromfile(lat, np.float32).astype(np.float64)
    if x.size != z.size:
        print(f"{x.size} latents for the reference's {z.size}")
        return 1
    cos = float(z @ x / np.linalg.norm(z) / np.linalg.norm(x))
    a = np.asarray(Image.open(rpng).convert("RGB"), np.float64)
    b = np.asarray(Image.open(npng).convert("RGB"), np.float64)
    mse = float(((a - b) ** 2).mean())
    psnr = 10 * np.log10(255**2 / mse) if mse else float("inf")
    print(f"latents cosine {cos:.4f}, PSNR {psnr:.1f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
