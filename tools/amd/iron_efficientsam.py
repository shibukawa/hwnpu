#!/usr/bin/env python3
"""EfficientSAM's image encoder (a plain ViT at 1024 x 1024 and its neck) as one IRON graph,
from the ONNX export of yunyangx/EfficientSAM (efficientsam_{ti,s}_encoder.onnx, Apache-2.0):
what cmd/hwnpu-image segment runs on AMD's NPU. The prompt's mask decoder runs on the host
(internal/onnx, per point or box).

  input   patches (4096, 768): the picture stretched to 1024 x 1024, normalized (ImageNet's
          mean and std), each 16 x 16 patch's 3 x 16 x 16 values (channel, row, column) in
          raster order
  embed   the patch embedding (a GEMM), its bias and the position embedding (14 x 14 resized
          bicubically to 64 x 64, as the ONNX graph does) added
  blocks  pre-norm: layer norm, attention (heads of 64, IRON's MHA over the 4096 tokens), the
          projection, added; layer norm, fc1, the erf GELU, fc2, added
  neck    a 1 x 1 convolution (a GEMM), layer norm over the channels, a 3 x 3 convolution (the
          tokens into a zero-padded grid, three im2col copies, a GEMM), layer norm
  output  (4096, 256): the image embedding, token-major (the decoder's (256, 64, 64) transposed)

  python tools/amd/iron_efficientsam.py efficientsam_ti_encoder.onnx OUT [--check]

See docs/amd-npu-models.md 22.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import iron
import iron.functional as F
from iron import Precision
from iron.operators.copy import Copy
from iron.operators.mha import MHA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_anima import bf, command_boundary, tap  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

PREC = Precision.HIGHEST
SIZE, PATCH = 1024, 16
GRID = SIZE // PATCH  # 64
T = GRID * GRID  # 4096 tokens
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def cubic_resize(x, H, W, a=-0.75):
    """ONNX's Resize (cubic, half_pixel, exclude_outside 0) of x (C, h, w) to (C, H, W): torch's
    bicubic interpolate without align_corners, indices clamped at the borders."""

    def weights(n_out, n_in):
        s = n_in / n_out
        m = np.zeros((n_out, n_in))
        for o in range(n_out):
            f = (o + 0.5) * s - 0.5
            i0 = math.floor(f)
            t = f - i0
            ws = [((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a,
                  ((a + 2) * t - (a + 3)) * t * t + 1,
                  ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1]
            ws.append(1 - sum(ws))
            for k, w in enumerate(ws):
                m[o, min(max(i0 - 1 + k, 0), n_in - 1)] += w
        return m

    my, mx = weights(H, x.shape[1]), weights(W, x.shape[2])
    return np.einsum("Yy,cyx,Xx->cYX", my, x.astype(np.float64), mx).astype(np.float32)


def load(path: Path):
    """The encoder's weights from its ONNX graph, by the nodes' order."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path))
    g = m.graph
    init = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            init[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)

    def bias_of(n):
        for u in users.get(n.output[0], []):
            if u.op_type == "Add":
                b = [init[i] for i in u.input if i in init and init[i].ndim == 1]
                if b:
                    return b[0]
        return None

    convs = [n for n in g.node if n.op_type == "Conv"]
    lns = [n for n in g.node if n.op_type == "LayerNormalization"]
    mms = [n for n in g.node if n.op_type == "MatMul" and n.input[1] in init]
    resize = next(n for n in g.node if n.op_type == "Resize" and n.input[0] in init)
    pe, neck0, neck2 = convs
    assert init[pe.input[1]].shape[2:] == (PATCH, PATCH), init[pe.input[1]].shape
    E = init[pe.input[1]].shape[0]
    L = len(lns) // 2
    assert len(mms) == 4 * L
    heads = next(int(init[n.input[1]].reshape(-1)[0]) for n in g.node if n.op_type == "Div" and "attn" in n.name and n.input[1] in init)
    w = dict(E=E, layers=L, heads=heads, pe=init[pe.input[1]].reshape(E, -1), peb=init[pe.input[2]],
             pos=cubic_resize(init[resize.input[0]][0], GRID, GRID).reshape(E, T).T, blocks=[])
    for i in range(L):
        n1, n2 = lns[2 * i], lns[2 * i + 1]
        qkv, proj, f1, f2 = mms[4 * i : 4 * i + 4]
        b = dict(n1=init[n1.input[1]], n1b=init[n1.input[2]], n2=init[n2.input[1]], n2b=init[n2.input[2]],
                 eps=onnx.helper.get_attribute_value(next(a for a in n1.attribute if a.name == "epsilon")))
        for k, n in (("qkv", qkv), ("proj", proj), ("f1", f1), ("f2", f2)):
            b[k], b[k + "b"] = init[n.input[1]].T, bias_of(n)  # (out, in)
        w["blocks"].append(b)
    # the neck's layer norms: Mul by gamma (256, 1, 1), Add of beta
    ga = [init[i].reshape(-1) for n in g.node if n.op_type == "Mul" and "neck" in n.name for i in n.input if i in init and init[i].ndim == 3]
    be = [init[i].reshape(-1) for n in g.node if n.op_type == "Add" and "neck" in n.name for i in n.input if i in init and init[i].ndim == 3]
    w.update(neck0=init[neck0.input[1]][:, :, 0, 0], neck2=init[neck2.input[1]], ln=list(zip(ga, be)))
    assert len(w["ln"]) == 2 and len(neck0.input) == 2 and len(neck2.input) == 2
    return w


class Encoder(iron.Graph):
    def __init__(self, w):
        E = self.E = w["E"]
        self.heads = w["heads"]
        p = lambda x: iron.param(bf(x))  # noqa: E731
        self.pe = p(w["pe"])
        self.pos = p(w["pos"] + w["peb"])
        self.blocks = []
        for b in w["blocks"]:
            q, k, v = np.split(b["qkv"], 3, axis=0)
            qb, kb, vb = np.split(b["qkvb"], 3)
            self.blocks.append({n: p(x) for n, x in dict(
                n1=b["n1"], n1b=b["n1b"], n2=b["n2"], n2b=b["n2b"], q=q, qb=qb, k=k, kb=kb, v=v, vb=vb,
                o=b["proj"], ob=b["projb"], f1=b["f1"], f1b=b["f1b"], f2=b["f2"], f2b=b["f2b"]).items()})
        self.eps = [b["eps"] for b in w["blocks"]]
        self.neck0 = p(w["neck0"])
        self.neck2 = p(w["neck2"].transpose(0, 2, 3, 1).reshape(256, -1))  # K: (dy, dx, c)
        (g0, b0), (g1, b1) = w["ln"]
        self.ln0, self.ln0b, self.ln1, self.ln1b = p(g0), p(b0), p(g1), p(b1)
        self.Wp = GRID + 2
        self.grid = iron.state(((GRID + 2) * self.Wp + 2, 256))  # the 3 x 3's input, zero-padded by 1
        self.cols = iron.state((T, 9 * 256))

    def body(self, patches):
        E, H = self.E, self.heads
        D = E // H
        x = F.linear(patches, self.pe, precision=PREC) + self.pos
        for i, b in enumerate(self.blocks):
            if i and i % 2 == 0:
                command_boundary()
            h = F.layer_norm(x, (E,), b["n1"], b["n1b"], eps=self.eps[i])
            q = F.linear(h, b["q"], b["qb"], precision=PREC)
            k = F.linear(h, b["k"], b["kb"], precision=PREC)
            v = F.linear(h, b["v"], b["vb"], precision=PREC)
            o = MHA.step(q.reshape(T, H, D), k.reshape(T, H, D), v.reshape(T, H, D), heads_interleaved=True,
                         kv_interleaved=True, causal=False, d=D, num_pipelines=8)
            x = x + F.linear(o.reshape(T, E), b["o"], b["ob"], precision=PREC)
            h = F.layer_norm(x, (E,), b["n2"], b["n2b"], eps=self.eps[i])
            h = F.linear(F.gelu(F.linear(h, b["f1"], b["f1b"], precision=PREC), approximate="none"), b["f2"], b["f2b"], precision=PREC)
            x = x + h
        command_boundary()
        h = F.layer_norm(F.linear(x, self.neck0, precision=PREC), (256,), self.ln0, self.ln0b, eps=1e-6)
        C, Wp = 256, self.Wp
        Copy.step(h, self.grid, src=tap((T, C), 0, [GRID, GRID * C], [GRID * C, 1]),
                  dst=tap(tuple(self.grid.shape), (Wp + 1) * C, [GRID, GRID * C], [Wp * C, 1]))
        for dy in range(3):  # output (y, x)'s kernel row dy: grid pixels (y + dy, x .. x + 2)
            Copy.step(self.grid, self.cols, src=tap(tuple(self.grid.shape), dy * Wp * C, [GRID, GRID, 3 * C], [Wp * C, C, 1]),
                      dst=tap((T, 9 * C), dy * 3 * C, [GRID, GRID, 3 * C], [GRID * 9 * C, 9 * C, 1]))
        h = F.linear(self.cols[0:T], self.neck2, precision=PREC)
        return F.layer_norm(h, (256,), self.ln1, self.ln1b, eps=1e-6)


def patch_rows(img):
    """The graph's input of a normalized picture (3, 1024, 1024)."""
    return img.reshape(3, GRID, PATCH, GRID, PATCH).transpose(1, 3, 0, 2, 4).reshape(T, 3 * PATCH * PATCH)


def layer_norm(x, g, b, eps):
    m = x.mean(-1, keepdims=True)
    d = x - m
    return d / np.sqrt((d * d).mean(-1, keepdims=True) + eps) * g + b


def erf(x):
    """erf to 1.5e-7 (Abramowitz and Stegun 7.1.26)."""
    t = 1 / (1 + 0.3275911 * np.abs(x))
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return np.sign(x) * y


def reference(w, rows):
    """The encoder in float32 on the patch rows: (4096, 256)."""

    E, H = w["E"], w["heads"]
    D = E // H
    x = rows @ w["pe"].T + w["peb"] + w["pos"]
    for b in w["blocks"]:
        h = layer_norm(x, b["n1"], b["n1b"], b["eps"])
        qkv = h @ b["qkv"].T + b["qkvb"]
        q, k, v = (a.reshape(T, H, D).transpose(1, 0, 2) for a in np.split(qkv, 3, axis=1))
        o = np.empty((H, T, D), np.float32)
        for j in range(H):
            s = (q[j] @ k[j].T) / math.sqrt(D)
            s = np.exp(s - s.max(-1, keepdims=True))
            o[j] = (s / s.sum(-1, keepdims=True)) @ v[j]
        x = x + o.transpose(1, 0, 2).reshape(T, E) @ b["proj"].T + b["projb"]
        h = layer_norm(x, b["n2"], b["n2b"], b["eps"]) @ b["f1"].T + b["f1b"]
        x = x + (0.5 * h * (1 + erf(h / math.sqrt(2)))) @ b["f2"].T + b["f2b"]
    (g0, b0), (g1, b1) = w["ln"]
    h = layer_norm(x @ w["neck0"].T, g0, b0, 1e-6).T.reshape(256, GRID, GRID)
    hp = np.pad(h, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(h)
    for dy in range(3):
        for dx in range(3):
            out += np.einsum("oc,chw->ohw", w["neck2"][:, :, dy, dx], hp[:, dy : dy + GRID, dx : dx + GRID])
    return layer_norm(out.reshape(256, T).T, g1, b1, 1e-6)


def main(argv=None) -> int:
    global PREC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path, help="efficientsam_{ti,s}_encoder.onnx")
    ap.add_argument("out", type=Path)
    ap.add_argument("--precision", choices=["MEDIUM", "HIGHEST"], default="HIGHEST")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-image", type=Path, help="a .npy (3, 1024, 1024) in [0, 1] for --check (default: random)")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    PREC = getattr(Precision, a.precision)
    bind_device(a.device, a.cols)
    t0 = time.time()
    w = load(a.onnx)
    g = Encoder(w)
    cg = g.compile(patches=(T, 3 * PATCH * PATCH))
    meta = dict(kind="efficientsam", size=SIZE, patch=PATCH, grid=GRID, tokens=T, embed=256, mean=MEAN, std=STD,
                hidden=w["E"], layers=w["layers"], heads=w["heads"], precision=a.precision)
    m = export_graph(cg, a.out, "encoder", meta)
    print(f"compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    (a.out / "model.json").write_text(json.dumps(meta, indent=1) + "\n")
    if a.check:
        img = np.load(a.check_image) if a.check_image else np.random.default_rng(0).uniform(0, 1, (3, SIZE, SIZE))
        img = ((img - np.array(MEAN)[:, None, None]) / np.array(STD)[:, None, None]).astype(np.float32)
        x = bf(patch_rows(img))
        write_check(g, m, a.out, dict(patches=x))
        want = reference(w, np.asarray(x, np.float32))
        got = np.asarray(g.reference(x), np.float32)
        cos = (got * want).sum() / math.sqrt((got * got).sum() * (want * want).sum())
        print(f"  IRON's reference against float32: cos {cos:.5f}, max err {np.abs(got - want).max():.4f} of {np.abs(want).max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
