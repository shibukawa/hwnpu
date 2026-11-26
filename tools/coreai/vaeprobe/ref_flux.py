#!/usr/bin/env python3
"""Check a vaeprobe run of FLUX.2 klein's VAE decoder (fluxvae.VaeStages) stage by stage.

    python3 ref_flux.py CKPT OUT [STAGE...]

CKPT holds vae/diffusion_pytorch_model.safetensors (black-forest-labs/FLUX.2-klein-4B); OUT is
vaeprobe's output folder (the decoder's stages; an encoder's outputs beside them are passed over).
Each stage (1 to 9, or those named) is computed again in float64 from the run's own output of the
stage before, and compared with the run's: its PSNR (peak: the reference's largest magnitude) and
the rows whose error is over 20 times the median row's. A right stage is 74 to 89 dB with no such
row; a stage with bands 20 to 30 dB with the rows listed (every 112 rows at 512 x 512).
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
    w, b = get(path + ".weight"), get(path + ".bias")
    O, C, k, _ = w.shape
    H, W = x.shape[1:]
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    y = np.zeros((O, H * W))
    for dy in range(k):
        for dx in range(k):
            y += w[:, :, dy, dx] @ xp[:, dy:dy + H, dx:dx + W].reshape(C, -1)
    return y.reshape(O, H, W) + b[:, None, None]


def group_norm(x, path, groups=32, eps=1e-6):
    y = x.reshape(groups, -1)
    y = (y - y.mean(1, keepdims=True)) / np.sqrt(y.var(1, keepdims=True) + eps)
    return y.reshape(x.shape) * get(path + ".weight")[:, None, None] + get(path + ".bias")[:, None, None]


def silu(x):
    return x / (1 + np.exp(-x))


D = "decoder"


def res(x, p):
    h = conv(x, p + ".conv_shortcut") if p + ".conv_shortcut.weight" in header else x
    a = conv(silu(group_norm(x, p + ".norm1")), p + ".conv1")
    a = conv(silu(group_norm(a, p + ".norm2")), p + ".conv2")
    return a + h


def up(x, p):
    return conv(x.repeat(2, 1).repeat(2, 2), p + ".upsamplers.0.conv")


def head(x):
    return conv(silu(group_norm(x, f"{D}.conv_norm_out")), f"{D}.conv_out")


STAGES = {
    1: [("r", f"{D}.up_blocks.0.resnets.{j}") for j in range(3)] + [("u", f"{D}.up_blocks.0")],
    2: [("r", f"{D}.up_blocks.1.resnets.{j}") for j in range(3)] + [("u", f"{D}.up_blocks.1")],
    3: [("r", f"{D}.up_blocks.2.resnets.0")],
    4: [("r", f"{D}.up_blocks.2.resnets.1")],
    5: [("r", f"{D}.up_blocks.2.resnets.2"), ("u", f"{D}.up_blocks.2")],
    6: [("r", f"{D}.up_blocks.3.resnets.0")],
    7: [("r", f"{D}.up_blocks.3.resnets.1")],
    8: [("r", f"{D}.up_blocks.3.resnets.2")],
    9: [("h", "")],
}


def load(i):
    names = [p for p in glob.glob(f"{out}/*_s{i}.y.*.bin") if "enc" not in p.split("/")[-1]]
    dims = [int(v) for v in re.search(r"\[(.*)\]", names[0]).group(1).split()]
    return np.fromfile(names[0], np.float16).astype(np.float64).reshape(dims[1:])


for i in [int(s) for s in sys.argv[3:]] or range(1, 10):
    x = load(i - 1)
    for kind, p in STAGES[i]:
        x = res(x, p) if kind == "r" else up(x, p) if kind == "u" else head(x)
    d = load(i) - x
    rows = np.sqrt((d ** 2).mean(axis=(0, 2)))
    med = np.median(rows)
    psnr = 10 * np.log10(np.abs(x).max() ** 2 / (d ** 2).mean())
    bad = np.where(rows > 20 * med)[0]
    print(f"s{i} {x.shape} psnr {psnr:.1f} dB  row-rms median {med:.2g} max {rows.max():.2g}  rows >20x median: {len(bad)} {bad[:12]}", flush=True)
