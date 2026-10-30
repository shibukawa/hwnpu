"""Golden image resizes, for checking internal/llm/lfm3vl's Resize.

    python resize_golden.py OUT_DIR

Writes OUT_DIR/resize.png, a small RGB image of gradients and noise, and OUT_DIR/resize.json with
PIL's antialiased bilinear and bicubic resizes of it (Image.resize, as the LFM3-VL image
processor and llama.cpp resize) to sizes smaller and larger than the image, as RGB bytes in hex.
Needs Pillow and numpy.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    h, w = 23, 37
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 / (w - 1), y * 255 / (h - 1), (x * y) % 256], axis=-1)
    img = np.clip(img + rng.integers(-40, 40, size=img.shape), 0, 255).astype(np.uint8)
    path = os.path.join(a.out_dir, "resize.png")
    Image.fromarray(img, "RGB").save(path)
    src = Image.open(path)
    cases = []
    for filt, name in ((Image.BILINEAR, "bilinear"), (Image.BICUBIC, "bicubic")):
        for size in ((16, 16), (64, 32), (40, 24), (8, 30)):
            out = np.asarray(src.resize(size, filt))
            cases.append({"filter": name, "width": size[0], "height": size[1], "rgb": out.tobytes().hex()})
    with open(os.path.join(a.out_dir, "resize.json"), "w") as f:
        json.dump({"cases": cases}, f)
        f.write("\n")


if __name__ == "__main__":
    main()
