#!/usr/bin/env python3
"""Compare two vaeprobe runs stage by stage: the same bytes, or the PSNR and largest difference.

    python3 cmp.py OUT_A OUT_B     (OUT_B is the reference of the PSNR's peak)
"""
import glob
import os
import sys

import numpy as np

a, b = sys.argv[1], sys.argv[2]
for fa in sorted(glob.glob(f"{a}/*.bin")):
    name = os.path.basename(fa)
    stage = name.split(".")[0]
    # the same stage's output in b, whatever the programs are called there (vae_s3, dec_s3)
    tail = stage[stage.rfind("_s"):] + "." + name.split(".", 1)[1]
    fbs = [p for p in glob.glob(f"{b}/*.bin") if os.path.basename(p).endswith(tail)]
    if not fbs:
        print(stage, "missing in", b)
        continue
    x = np.fromfile(fa, np.float16).astype(np.float64)
    y = np.fromfile(fbs[0], np.float16).astype(np.float64)
    same = np.array_equal(np.fromfile(fa, np.uint16), np.fromfile(fbs[0], np.uint16))
    mse = np.mean((x - y) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10(np.max(np.abs(y)) ** 2 / mse)
    print(f"{stage} same={same} psnr={psnr:.1f} dB maxdiff={np.max(np.abs(x - y)):.4g} nan={int(np.isnan(x).sum())}")
