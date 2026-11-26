#!/usr/bin/env python3
"""Check a vaeprobe run of Anima's VAE decoder (mil.VaeStages) stage by stage.

    python3 ref_anima.py CKPT OUT [STAGE...]

CKPT holds vae/diffusion_pytorch_model.safetensors (circlestone-labs/Anima-Base-v1.0-Diffusers);
OUT is vaeprobe's output folder. Each stage (1 to 9, or those named) is computed again in float64
from the run's own output of the stage before, and compared with the run's: its PSNR (peak: the
reference's largest magnitude) and the rows whose error is over 20 times the median row's. A right
stage is 75 to 89 dB with no such row; a stage with bands 20 to 30 dB with the rows listed.
Needs numpy alone.
"""
import glob
import json
import re
import struct
import sys

import numpy as np

ckpt, out = sys.argv[1], sys.argv[2]
f = open(ckpt + "/vae/diffusion_pytorch_model.safetensors", "rb")
n = struct.unpack("<Q", f.read(8))[0]
header = json.loads(f.read(n))
base = 8 + n


def get(name):
    t = header[name]
    a, b = t["data_offsets"]
    f.seek(base + a)
    raw = f.read(b - a)
    if t["dtype"] == "BF16":
        v = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif t["dtype"] == "F32":
        v = np.frombuffer(raw, np.float32)
    else:
        v = np.frombuffer(raw, np.float16).astype(np.float32)
    return v.reshape(t["shape"]).astype(np.float64)


def conv(x, path):
    """A causal 3 x 3 x 3 convolution's last temporal tap, or a 2 D one, zero padded."""
    w, b = get(path + ".weight"), get(path + ".bias")
    if w.ndim == 5:
        w = w[:, :, -1]
    O, C, k, _ = w.shape
    H, W = x.shape[1:]
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    y = np.zeros((O, H * W))
    for dy in range(k):
        for dx in range(k):
            y += w[:, :, dy, dx] @ xp[:, dy:dy + H, dx:dx + W].reshape(C, -1)
    return y.reshape(O, H, W) + b[:, None, None]


def norm(x, gamma):
    return x / np.sqrt((x * x).mean(0, keepdims=True) + 1e-12) * get(gamma).reshape(-1, 1, 1)


def silu(x):
    return x / (1 + np.exp(-x))


D = "decoder."


def res(x, path, cout):
    h = x if x.shape[0] == cout else conv(x, D + path + ".conv_shortcut")
    a = conv(silu(norm(x, D + path + ".norm1.gamma")), D + path + ".conv1")
    a = conv(silu(norm(a, D + path + ".norm2.gamma")), D + path + ".conv2")
    return a + h


def up(x, path):
    return conv(x.repeat(2, 1).repeat(2, 2), D + path + ".upsamplers.0.resample.1")


def head(x):
    return conv(silu(norm(x, D + "norm_out.gamma")), D + "conv_out")


def R(path, cout):
    return ("res", path, cout)


def U(path):
    return ("up", path, 0)


STAGES = {
    1: [R("up_blocks.0.resnets.0", 384), R("up_blocks.0.resnets.1", 384), R("up_blocks.0.resnets.2", 384), U("up_blocks.0")],
    2: [R("up_blocks.1.resnets.0", 384), R("up_blocks.1.resnets.1", 384), R("up_blocks.1.resnets.2", 384)],
    3: [U("up_blocks.1"), R("up_blocks.2.resnets.0", 192)],
    4: [R("up_blocks.2.resnets.1", 192), R("up_blocks.2.resnets.2", 192)],
    5: [U("up_blocks.2")],
    6: [R("up_blocks.3.resnets.0", 96)],
    7: [R("up_blocks.3.resnets.1", 96)],
    8: [R("up_blocks.3.resnets.2", 96)],
    9: [("head", "", 0)],
}


def load(i):
    name = glob.glob(f"{out}/*_s{i}.y.*.bin")[0]
    dims = [int(v) for v in re.search(r"\[(.*)\]", name).group(1).split()]
    return np.fromfile(name, np.float16).astype(np.float64).reshape(dims[1:])


for i in [int(s) for s in sys.argv[3:]] or range(1, 10):
    x = load(i - 1)
    for kind, path, cout in STAGES[i]:
        x = res(x, path, cout) if kind == "res" else up(x, path) if kind == "up" else head(x)
    d = load(i) - x
    rows = np.sqrt((d ** 2).mean(axis=(0, 2)))
    med = np.median(rows)
    psnr = 10 * np.log10(np.abs(x).max() ** 2 / (d ** 2).mean())
    bad = np.where(rows > 20 * med)[0]
    print(f"s{i} {x.shape} psnr {psnr:.1f} dB  row-rms median {med:.2g} max {rows.max():.2g}  rows >20x median: {len(bad)} {bad[:16]}", flush=True)
