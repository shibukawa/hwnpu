#!/usr/bin/env python3
"""Depth Anything 3's monocular depth (depth-anything/DA3MONO-LARGE, Apache-2.0) on IRON: the
graphs cmd/hwnpu-depth runs on AMD's NPU.

DA3MONO-LARGE is DINOv2's ViT-L/14 (24 blocks, hidden 1024) and a DPT head: the backbone's
layers 4, 11, 17 and 23 (after the final norm) projected and resized to four scales (4x, 2x,
1x and 1/2x of the patch grid), fused top-down by residual convolution units with bilinear
upsampling, then a 3 x 3 convolution, a bilinear resize to the image's size and two small
heads: depth (exp of the logit) and sky (ReLU). Two graphs, each exported by iron_graph.py:

  vit   the patch embedding (the host gives each 14 x 14 patch's 588 values) and the 24 blocks:
        the four features (tokens, 1024), the class token's row first
  head  the DPT head: two channels of the image's pixels, depth's logit and sky's

The size is fixed per export (the position embedding is interpolated to it, bicubic as
DINOv2's). The host (internal/vision/depth) resizes and normalizes the image, cuts the
patches, takes exp of the depth logit and writes the picture.

Convolutions are iron_anima's: a channel-last zero-padded grid (Grid here: rows of Wp pixels,
Wp >= W + 1 and a multiple of 16, and rows to a multiple of 16), a GEMM over the im2col rows of
16 or more rows at a time, the interior copied into the next grid. Transposed convolutions
(kernel = stride) are one GEMM and a copy of each kernel position's columns into its pixels.
Bilinear resizes (align_corners) are two GEMMs with the interpolation matrices, the dense
pixels transposed between them by a copy.

  python tools/amd/iron_da3.py ~/models/DA3MONO-LARGE OUT --width 504 --height 504

See docs/amd-npu-models.md.
"""

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import iron
import iron.functional as F
from iron import Precision, Scratchpad
from iron.operators.copy import Copy
from iron.operators.gemm import GEMM
from iron.operators.mha import MHA
from iron.operators.relu import ReLU

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_anima import RandomWeights, Safetensors, bf, command_boundary, tap  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

PATCH = 14
PE_IN = 640  # the patch's 588 values, padded


@dataclasses.dataclass
class Precisions:
    vit: Precision = Precision.MEDIUM
    head: Precision = Precision.MEDIUM
    resize: Precision = Precision.HIGHEST  # the interpolation matrices' GEMMs


PREC = Precisions()


def f32(x) -> np.ndarray:
    return np.array(x, np.float32)


@dataclasses.dataclass(frozen=True)
class Config:
    width: int = 504
    height: int = 504
    hidden: int = 1024
    heads: int = 16
    layers: int = 24
    mlp: int = 4096
    out_layers: tuple = (4, 11, 17, 23)
    features: int = 256
    out_channels: tuple = (256, 512, 1024, 1024)
    split: int = 6  # blocks a command runs

    @property
    def gh(self) -> int:
        return self.height // PATCH

    @property
    def gw(self) -> int:
        return self.width // PATCH

    @property
    def patches(self) -> int:
        return self.gh * self.gw

    @property
    def tokens(self) -> int:
        """The class token and the patches, padded to MHA's 8 pipelines of 64 rows."""
        return -(-(1 + self.patches) // 512) * 512

    @classmethod
    def of(cls, root: Path, width: int, height: int) -> "Config":
        cfg = json.loads((root / "config.json").read_text())["config"]
        net, head = cfg["net"], cfg["head"]
        if net["name"] != "vitl" or net.get("alt_start", -1) != -1 or head.get("output_dim", 1) != 1:
            raise SystemExit(f"{root}: only DA3's mono ViT-L with a DPT head (DA3MONO-LARGE) is supported")
        if width % PATCH or height % PATCH:
            raise SystemExit(f"{width}x{height}: not whole {PATCH}-pixel patches")
        return cls(width=width, height=height, out_layers=tuple(net["out_layers"]), features=head["features"],
                   out_channels=tuple(head["out_channels"]))


# --- the ViT ----------------------------------------------------------------------------------


def bicubic_weights(n_out: int, n_in: int, scale: float) -> np.ndarray:
    """torch's bicubic (A = -0.75, align_corners False, the given scale factor, clamped
    borders) as a matrix (n_out, n_in)."""
    A = -0.75

    def cubic(x):
        x = abs(x)
        if x <= 1:
            return ((A + 2) * x - (A + 3)) * x * x + 1
        if x < 2:
            return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
        return 0.0

    m = np.zeros((n_out, n_in), np.float64)
    for o in range(n_out):
        src = (o + 0.5) / scale - 0.5
        i0 = math.floor(src)
        t = src - i0
        for k in range(-1, 3):
            m[o, min(max(i0 + k, 0), n_in - 1)] += cubic(k - t)
    return m


def pos_embed(w: Safetensors, c: Config) -> np.ndarray:
    """DINOv2's interpolate_pos_encoding: the patches' table bicubically resized (scale
    (n + 0.1) / M, interpolate_offset), the class token's row first. (1 + patches, hidden)."""
    pe = f32(w["model.backbone.pretrained.pos_embed"])[0]
    M = int(math.isqrt(pe.shape[0] - 1))
    grid = pe[1:].reshape(M, M, -1)
    if (c.gh, c.gw) == (M, M):
        return pe
    # (DINOv2 names the image's rows w0 and its columns h0; each axis is scaled by its own)
    ah = bicubic_weights(c.gh, M, (c.gh + 0.1) / M)
    aw = bicubic_weights(c.gw, M, (c.gw + 0.1) / M)
    out = np.einsum("ai,bj,ijc->abc", ah, aw, grid)  # (gh, gw, C)
    return np.concatenate([pe[:1], out.reshape(-1, pe.shape[1])]).astype(np.float32)


class VitBlock(iron.Graph):
    """A DINOv2 block: layer norm, attention (16 heads of 64), layer scale; layer norm, GELU
    MLP, layer scale (the layer scales folded into the output projections)."""

    def __init__(self, c: Config, w, i: int):
        E, M = c.hidden, c.mlp
        p = f"model.backbone.pretrained.blocks.{i}."
        self.c = c
        if w is None:
            for n, s in (("n1", (E,)), ("n1b", (E,)), ("n2", (E,)), ("n2b", (E,)), ("q", (E, E)), ("k", (E, E)),
                         ("v", (E, E)), ("qb", (E,)), ("kb", (E,)), ("vb", (E,)), ("o", (E, E)), ("ob", (E,)),
                         ("f1", (M, E)), ("f1b", (M,)), ("f2", (E, M)), ("f2b", (E,))):
                setattr(self, n, iron.param(*s))
            return
        g = lambda n: f32(w[p + n])  # noqa: E731
        self.n1, self.n1b = iron.param(bf(g("norm1.weight"))), iron.param(bf(g("norm1.bias")))
        self.n2, self.n2b = iron.param(bf(g("norm2.weight"))), iron.param(bf(g("norm2.bias")))
        qkv, qkvb = g("attn.qkv.weight"), g("attn.qkv.bias")
        for j, n in enumerate("qkv"):
            setattr(self, n, iron.param(bf(qkv[j * E : (j + 1) * E])))
            setattr(self, n + "b", iron.param(bf(qkvb[j * E : (j + 1) * E])))
        ls1, ls2 = g("ls1.gamma"), g("ls2.gamma")
        self.o = iron.param(bf(g("attn.proj.weight") * ls1[:, None]))
        self.ob = iron.param(bf(g("attn.proj.bias") * ls1))
        self.f1, self.f1b = iron.param(bf(g("mlp.fc1.weight"))), iron.param(bf(g("mlp.fc1.bias")))
        self.f2 = iron.param(bf(g("mlp.fc2.weight") * ls2[:, None]))
        self.f2b = iron.param(bf(g("mlp.fc2.bias") * ls2))

    def body(self, x, nk):
        c = self.c
        E, H = c.hidden, c.heads
        D = E // H
        T = x.shape[0]
        P = PREC.vit
        h = F.layer_norm(x, (E,), self.n1, self.n1b, eps=1e-6)
        q = F.linear(h, self.q, self.qb, precision=P)
        k = F.linear(h, self.k, self.kb, precision=P)
        v = F.linear(h, self.v, self.vb, precision=P)
        o = MHA.step(q.reshape(T, H, D), k.reshape(T, H, D)[:nk], v.reshape(T, H, D)[:nk], heads_interleaved=True,
                     kv_interleaved=True, causal=False, d=D, num_pipelines=8)
        x = x + F.linear(o.reshape(T, E), self.o, self.ob, precision=P)
        h = F.layer_norm(x, (E,), self.n2, self.n2b, eps=1e-6)
        h = F.linear(F.gelu(F.linear(h, self.f1, self.f1b, precision=P)), self.f2, self.f2b, precision=P)
        return x + h


class Vit(iron.Graph):
    """The patch embedding and the blocks.

    Input ``patches`` (tokens, 640): row 0 zeros (the class token's), then each patch's 3 x 14
    x 14 values (channel, row, column; normalized), zeros past them. Per call: ``nk``, the
    rows that are tokens (1 + patches). Outputs the four features (tokens, hidden) after the
    final norm.
    """

    def __init__(self, c: Config, w=None):
        E = c.hidden
        self.c = c
        self.blocks = [VitBlock(c, w, i) for i in range(max(c.out_layers) + 1)]
        if w is None:
            self.pe, self.pos = iron.param(E, PE_IN), iron.param(c.tokens, E)
            self.norm, self.normb = iron.param(E), iron.param(E)
            return
        pw = f32(w["model.backbone.pretrained.patch_embed.proj.weight"]).reshape(E, -1)
        full = np.zeros((E, PE_IN), np.float32)
        full[:, : pw.shape[1]] = pw
        self.pe = iron.param(bf(full))
        pos = np.zeros((c.tokens, E), np.float32)
        pe = pos_embed(w, c)
        pos[: pe.shape[0]] = pe
        pos[1 : pe.shape[0]] += f32(w["model.backbone.pretrained.patch_embed.proj.bias"])
        pos[0] += f32(w["model.backbone.pretrained.cls_token"]).reshape(-1)
        self.pos = iron.param(bf(pos))
        self.norm = iron.param(bf(w["model.backbone.pretrained.norm.weight"]))
        self.normb = iron.param(bf(w["model.backbone.pretrained.norm.bias"]))

    def body(self, patches, *, nk: Scratchpad[np.int32]):
        c = self.c
        x = F.linear(patches, self.pe, precision=PREC.vit) + self.pos
        outs = []
        for i, b in enumerate(self.blocks):
            if i and i % c.split == 0:
                command_boundary()
            x = b(x, nk)
            if i in c.out_layers:
                outs.append(F.layer_norm(x, (c.hidden,), self.norm, self.normb, eps=1e-6))
        return tuple(outs)


# --- grids and the head --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Grid:
    """A channel-last grid zero-padded for 3 x 3 convolutions: Hp = H rounded up to 16 rows of
    Wp pixels (Wp >= W + 1, a multiple of 16; pixel x of a row at 1 + x, zeros around), a zero
    row above and below, a zero pixel before the first, rounded up to 256 rows."""

    H: int
    W: int
    C: int

    @property
    def Wp(self) -> int:
        return -(-(self.W + 1) // 16) * 16

    @property
    def Hp(self) -> int:
        return -(-self.H // 16) * 16

    @property
    def rows(self) -> int:
        return -(-((self.Hp + 2) * self.Wp + 2) // 256) * 256

    @property
    def band(self) -> int:
        return 1 + self.Wp

    def pixel(self, y: int, x: int) -> int:
        return self.band + y * self.Wp + 1 + x

    def chunk(self, cin: int) -> int:
        """Rows a convolution takes at once: a multiple of 16 dividing Hp, its im2col rows at
        most 64 MiB."""
        best = 16
        for r in range(16, self.Hp + 1, 16):
            if self.Hp % r == 0 and r * self.Wp * 9 * cin * 2 <= 64 << 20:
                best = r
        return best


def interp_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Bilinear interpolation with align_corners=True as a matrix (n_out, n_in)."""
    m = np.zeros((n_out, n_in), np.float64)
    for o in range(n_out):
        s = o * (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
        i0 = min(int(math.floor(s)), n_in - 1)
        t = s - i0
        m[o, i0] += 1 - t
        if t > 0:
            m[o, i0 + 1] += t
    return m


def pad64(n: int) -> int:
    return -(-n // 64) * 64


class Head(iron.Graph):
    """DPT's head on the four features: input ``f0``..``f3`` (tokens, hidden), the class
    token's row first; output the image's pixels (H W, 64), channel 0 depth's logit, 1 sky's."""

    def __init__(self, c: Config, w=None):
        self.c = c
        self.params = []
        self._index = {}
        self.states = []
        self._cols = {}
        self._dense = {}
        H, W = c.gh, c.gw
        Fe = c.features
        p = "model.head."
        g = (lambda n: f32(w[p + n])) if w is not None else None
        oc = c.out_channels
        for i in range(4):
            self._param(f"proj{i}", g(f"projects.{i}.weight")[:, :, 0, 0] if g else np.zeros((oc[i], c.hidden)))
            self._param(f"proj{i}.b", g(f"projects.{i}.bias") if g else np.zeros(oc[i]))
        for i, k in ((0, 4), (1, 2)):
            wt = g(f"resize_layers.{i}.weight") if g else np.zeros((oc[i], oc[i], k, k))  # (cin, cout, k, k)
            self._param(f"up{i}", wt.transpose(2, 3, 1, 0).reshape(k * k * oc[i], oc[i]))  # rows (ky, kx, cout)
            self._param(f"up{i}.b", np.tile(g(f"resize_layers.{i}.bias") if g else np.zeros(oc[i]), k * k))
        self._conv3("down3", g("resize_layers.3.weight") if g else np.zeros((oc[3], oc[3], 3, 3)),
                    g("resize_layers.3.bias") if g else np.zeros(oc[3]))
        for i in range(4):
            self._conv3(f"rn{i}", g(f"scratch.layer{i + 1}_rn.weight") if g else np.zeros((Fe, oc[i], 3, 3)), None)
        for r in range(1, 5):
            for u in ((1, 2) if r < 4 else (2,)):
                for j in (1, 2):
                    n = f"scratch.refinenet{r}.resConfUnit{u}.conv{j}"
                    self._conv3(f"ref{r}.{u}.{j}", g(n + ".weight") if g else np.zeros((Fe, Fe, 3, 3)),
                                g(n + ".bias") if g else np.zeros(Fe))
            self._param(f"ref{r}.out", g(f"scratch.refinenet{r}.out_conv.weight")[:, :, 0, 0] if g else np.zeros((Fe, Fe)))
            self._param(f"ref{r}.out.b", g(f"scratch.refinenet{r}.out_conv.bias") if g else np.zeros(Fe))
        self._conv3("oc1", g("scratch.output_conv1.weight") if g else np.zeros((Fe // 2, Fe, 3, 3)),
                    g("scratch.output_conv1.bias") if g else np.zeros(Fe // 2))
        # the depth and the sky heads side by side: 3 x 3 to 32 + 32, ReLU, 1 x 1 to 2 (of 64)
        if g:
            w3 = np.concatenate([g("scratch.output_conv2.0.weight"), g("scratch.sky_output_conv2.0.weight")])
            b3 = np.concatenate([g("scratch.output_conv2.0.bias"), g("scratch.sky_output_conv2.0.bias")])
            w1 = np.zeros((64, 64), np.float32)
            w1[0, :32] = g("scratch.output_conv2.2.weight")[0, :, 0, 0]
            w1[1, 32:] = g("scratch.sky_output_conv2.2.weight")[0, :, 0, 0]
            b1 = np.zeros(64, np.float32)
            b1[0] = g("scratch.output_conv2.2.bias")[0]
            b1[1] = g("scratch.sky_output_conv2.2.bias")[0]
        else:
            w3, b3, w1, b1 = np.zeros((64, Fe // 2, 3, 3)), np.zeros(64), np.zeros((64, 64)), np.zeros(64)
        self._conv3("oc2", w3, b3)
        self._param("oc2.1", w1)
        self._param("oc2.1.b", b1)
        # the interpolation matrices, padded to 64 rows and columns
        self.sizes = [(H // 2, W // 2), (H, W), (2 * H, 2 * W), (4 * H, 4 * W), (8 * H, 8 * W), (c.height, c.width)]
        for a, b in zip(self.sizes, self.sizes[1:]):
            for n_in, n_out in ((a[0], b[0]), (a[1], b[1])):
                key = f"interp{n_out}x{n_in}"
                if key not in self._index:
                    m = np.zeros((pad64(n_out), pad64(n_in)), np.float32)
                    m[:n_out, :n_in] = interp_matrix(n_out, n_in)
                    self._param(key, m)

    def _param(self, name: str, x) -> None:
        self._index[name] = len(self.params)
        self.params.append(iron.param(bf(x)))

    def w(self, name: str):
        return self.params[self._index[name]] if name in self._index else None

    def _conv3(self, name: str, wt: np.ndarray, b) -> None:
        co = wt.shape[0]
        self._param(f"{name}.w", np.asarray(wt, np.float32).transpose(0, 2, 3, 1).reshape(co, -1))
        if b is not None:
            self._param(f"{name}.b", b)

    def state(self, shape):
        s = iron.state(shape)
        self.states.append(s)
        return s

    @staticmethod
    def view(s):
        return s[0 : s.shape[0]]

    # --- moving between dense pixels (H W, C) and grids ---

    def to_grid(self, x, g: Grid, first: int = 0):
        """The grid of dense pixels x (rows from ``first``: (H W, C) or more rows)."""
        out = self.state((g.rows, g.C))
        Copy.step(x, out, src=tap(tuple(x.shape), first * g.C, [g.H * g.W * g.C], [1]),
                  dst=tap((g.rows, g.C), g.pixel(0, 0) * g.C, [g.H, g.W * g.C], [g.Wp * g.C, 1]))
        return self.view(out)

    def to_dense(self, x, g: Grid):
        """The pixels (H W, C), in a state of whole 256 rows (zeros past them, for the GEMMs)."""
        C, n = g.C, g.H * g.W
        out = self.state((-(-n // 256) * 256, C))
        Copy.step(x, out, src=tap((g.rows, C), g.pixel(0, 0) * C, [g.H, g.W * C], [g.Wp * C, 1]),
                  dst=tap(out.shape, 0, [n * C], [1]))
        return self.view(out)

    def cols(self, n: int, k: int):
        key = (n, k)
        if key not in self._cols:
            self._cols[key] = self.state((n, k))
        return self._cols[key]

    def conv3(self, x, g: Grid, name: str, cout: int, post=None):
        """The 3 x 3 convolution (stride 1, zero padded) of grid x into a new grid. ``post(y,
        r0)`` takes each chunk's GEMM result (band rows from image row r0) before the copy."""
        go = Grid(g.H, g.W, cout)
        out = self.state((go.rows, cout))
        Wp, C = g.Wp, g.C
        R = g.chunk(C)
        for r0 in range(0, g.H, R):
            n = R * Wp
            start = (g.band + r0 * Wp - Wp - 1) * C
            if 3 * C <= 2046:
                cols = Copy.step(x, src=tap((g.rows, C), start, [R, Wp, 3, 3 * C], [Wp * C, C, Wp * C, 1])).reshape(n, 9 * C)
            else:  # a DMA dimension is at most 1023 words
                cs = self.cols(n, 9 * C)
                for dy in range(3):
                    Copy.step(x, cs, src=tap((g.rows, C), start + dy * Wp * C, [R, Wp, 3, C], [Wp * C, C, C, 1]),
                              dst=tap((n, 9 * C), dy * 3 * C, [n, 3, C], [9 * C, C, 1]))
                cols = cs[0:n]
            y = F.linear(cols, self.w(f"{name}.w"), self.w(f"{name}.b"), precision=PREC.head)
            if post is not None:
                y = post(y, r0)
            rows = min(R, g.H - r0)
            Copy.step(y, out, src=tap((n, cout), cout, [rows, g.W * cout], [Wp * cout, 1]),
                      dst=tap((go.rows, cout), go.pixel(r0, 0) * cout, [rows, g.W * cout], [Wp * cout, 1]))
        return self.view(out), go

    def conv3_s2(self, x, g: Grid, name: str, cout: int):
        """The 3 x 3 convolution with stride 2 (padding 1): dense (H/2 W/2, cout)."""
        Ho, Wo, C = g.H // 2, g.W // 2, g.C
        n = Ho * Wo
        m = -(-n // 256) * 256
        cs = self.cols(m, 9 * C)
        for dy in range(3):
            for oy in range(Ho):
                # output row oy's window rows 2 oy - 1 + dy; pixel 2 ox - 1 + dx
                Copy.step(x, cs, src=tap((g.rows, C), (g.band + (2 * oy - 1 + dy) * g.Wp) * C, [Wo, 3, C], [2 * C, C, 1]),
                          dst=tap((m, 9 * C), oy * Wo * 9 * C + dy * 3 * C, [Wo, 3, C], [9 * C, C, 1]))
        y = F.linear(cs[0:m], self.w(f"{name}.w"), self.w(f"{name}.b"), precision=PREC.head)
        return y[0:n]

    def deconv(self, x, H: int, W: int, i: int, k: int, cout: int, first: int = 0):
        """ConvTranspose2d with kernel = stride = k of dense x (rows from ``first``, cin): a
        grid (kH, kW, cout)."""
        y = F.linear(x, self.w(f"up{i}"), self.w(f"up{i}.b"), precision=PREC.head)  # (rows, k k cout)
        go = Grid(k * H, k * W, cout)
        out = self.state((go.rows, cout))
        kk = k * k * cout
        for ky in range(k):
            for kx in range(k):
                Copy.step(y, out, src=tap(tuple(y.shape), (first * kk) + (ky * k + kx) * cout, [H, W, cout], [W * kk, kk, 1]),
                          dst=tap((go.rows, cout), go.pixel(ky, kx) * cout, [H, W, cout], [k * go.Wp * cout, k * cout, 1]))
        return self.view(out), go

    def resize(self, x, H: int, W: int, H2: int, W2: int, C: int):
        """Bilinear (align_corners) of dense x (H W, C) to (H2 W2, C)."""
        # along the columns: transpose to (W, H C), the matrix (W2, W) on the left
        t = self.state((pad64(W), H * C))
        Copy.step(x, t, src=tap(tuple(x.shape), 0, [W, H, C], [C, W * C, 1]), dst=tap((pad64(W), H * C), 0, [W, H * C], [H * C, 1]))
        u = GEMM.step(self.w(f"interp{W2}x{W}"), self.view(t), b_col_maj=False, precision=PREC.resize)  # (pad W2, H C)
        # back to rows: (H, W2 C), the matrix (H2, H) on the left
        v = self.state((pad64(H), W2 * C))
        Copy.step(u, v, src=tap((pad64(W2), H * C), 0, [H, W2, C], [C, H * C, 1]), dst=tap((pad64(H), W2 * C), 0, [H, W2 * C], [W2 * C, 1]))
        y = GEMM.step(self.w(f"interp{H2}x{H}"), self.view(v), b_col_maj=False, precision=PREC.resize)  # (pad H2, W2 C)
        return y[0:H2].reshape(H2 * W2, C)

    def rcu(self, x, g: Grid, name: str):
        """A residual conv unit: ReLU, conv, ReLU, conv, plus x (grids)."""
        a, ga = self.conv3(ReLU.step(x), g, f"{name}.1", g.C, post=lambda y, r0: ReLU.step(y))

        def post(y, r0):
            s = g.band + r0 * g.Wp
            R = y.shape[0] // g.Wp
            return y + x[s : s + R * g.Wp]

        return self.conv3(a, ga, f"{name}.2", g.C, post=post)

    def refine(self, r: int, x, g: Grid, lateral=None, size=None):
        """A fusion block: (x + RCU1(lateral)), RCU2, the 1 x 1 out_conv, then the bilinear
        resize (the 1 x 1 first: both are linear per pixel); dense (H2 W2, features)."""
        if lateral is not None:
            lx, lg = lateral
            y, _ = self.rcu(lx, lg, f"ref{r}.1")
            x = x + y
        x, g = self.rcu(x, g, f"ref{r}.2")
        d = F.linear(self.to_dense(x, g), self.w(f"ref{r}.out"), self.w(f"ref{r}.out.b"), precision=PREC.head)
        H2, W2 = size
        return self.resize(d, g.H, g.W, H2, W2, g.C)

    def body(self, f0, f1, f2, f3):
        c = self.c
        H, W, Fe = c.gh, c.gw, c.features
        oc = c.out_channels
        feats = []
        for i, f in enumerate((f0, f1, f2, f3)):
            # every token's row (whole 256 rows for the GEMMs); the patches' from row 1
            p = F.linear(f, self.w(f"proj{i}"), self.w(f"proj{i}.b"), precision=PREC.head)  # (tokens, oc)
            if i == 0:
                feats.append(self.deconv(p, H, W, 0, 4, oc[0], first=1))
            elif i == 1:
                feats.append(self.deconv(p, H, W, 1, 2, oc[1], first=1))
            elif i == 2:
                g = Grid(H, W, oc[2])
                feats.append((self.to_grid(p, g, first=1), g))
            else:
                g = Grid(H, W, oc[3])
                d = self.conv3_s2(self.to_grid(p, g, first=1), g, "down3", oc[3])
                g = Grid(H // 2, W // 2, oc[3])
                feats.append((self.to_grid(d, g), g))
        rn = [self.conv3(x, g, f"rn{i}", Fe) for i, (x, g) in enumerate(feats)]
        s = self.sizes
        d = self.refine(4, *rn[3], size=s[1])
        g = rn[2][1]
        d = self.refine(3, self.to_grid(d, g), g, lateral=rn[2], size=s[2])
        g = rn[1][1]
        d = self.refine(2, self.to_grid(d, g), g, lateral=rn[1], size=s[3])
        g = rn[0][1]
        d = self.refine(1, self.to_grid(d, g), g, lateral=rn[0], size=s[4])
        g = Grid(*s[4], Fe)
        x, g = self.conv3(self.to_grid(d, g), g, "oc1", Fe // 2)
        d = self.resize(self.to_dense(x, g), g.H, g.W, c.height, c.width, Fe // 2)
        g = Grid(c.height, c.width, Fe // 2)
        x, g = self.conv3(self.to_grid(d, g), g, "oc2", 64, post=lambda y, r0: ReLU.step(y))
        return F.linear(self.to_dense(x, g), self.w("oc2.1"), self.w("oc2.1.b"), precision=PREC.head)


# --- export -----------------------------------------------------------------------------------


def patches(img: np.ndarray, c: Config) -> np.ndarray:
    """The Vit graph's input of a normalized image (H, W, 3)."""
    out = np.zeros((c.tokens, PE_IN), np.float32)
    p = img.reshape(c.gh, PATCH, c.gw, PATCH, 3).transpose(0, 2, 4, 1, 3).reshape(c.patches, -1)
    out[1 : 1 + c.patches, : p.shape[1]] = p
    return out


def export(root: Path, out: Path, c: Config, check: bool) -> None:
    w = (RandomWeights if ARGS.random else Safetensors)(root / "model.safetensors")
    meta = dict(config=dataclasses.asdict(c), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], patch=PATCH,
                tokens=c.tokens, rows=1 + c.patches)
    rng = np.random.default_rng(0)
    img = rng.standard_normal((c.height, c.width, 3)).astype(np.float32)
    feats = None
    if "vit" in ARGS.only:
        t0 = time.time()
        g = Vit(c, w)
        cg = g.compile(patches=(c.tokens, PE_IN))
        m = export_graph(cg, out / "vit", "vit", meta)
        print(f"vit: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            write_check(g, m, out / "vit", dict(patches=bf(patches(img, c))), dict(nk=1 + c.patches))
            feats = [np.asarray(f, np.float32) for f in g.reference(bf(patches(img, c)), nk=1 + c.patches)]
    if "head" in ARGS.only:
        t0 = time.time()
        g = Head(c, w)
        cg = g.compile(**{f"f{i}": (c.tokens, c.hidden) for i in range(4)})
        m = export_graph(cg, out / "head", "head", meta)
        print(f"head: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            if feats is None:
                feats = [rng.standard_normal((c.tokens, c.hidden)).astype(np.float32) for _ in range(4)]
            write_check(g, m, out / "head", {f"f{i}": bf(f) for i, f in enumerate(feats)})
    (out / "model.json").write_text(json.dumps(dict(name=root.name, kind="depth", **meta), indent=1) + "\n")


ARGS = None


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a DA3 mono repository (config.json, model.safetensors)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--width", type=int, default=504)
    ap.add_argument("--height", type=int, default=504)
    ap.add_argument("--only", nargs="*", default=["vit", "head"])
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    bind_device(a.device, a.cols)
    c = Config.of(a.root, a.width, a.height)
    a.out.mkdir(parents=True, exist_ok=True)
    export(a.root, a.out, c, a.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
