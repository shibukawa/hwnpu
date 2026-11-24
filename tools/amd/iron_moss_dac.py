#!/usr/bin/env python3
"""MOSS-SoundEffect's DAC decoder (vae/vae_128d_48k.pth) on IRON: the five stages of
internal/sfx's DACStages as graphs, which bonsaiai-audio generate -m sfx runs on AMD's NPU.

  python tools/amd/iron_moss_dac.py ~/models/MOSS-SoundEffect-v2.0 OUT

writes OUT/dac-N/ for the stages N = 0..4 (iron_graph.py), each over the frames of FRAMES[N] at
its input rate: input x (frames, channels) time-major; output the stage's (frames x rate,
channels out), the last stage's the output layer's convolution before its tanh (in column 0 of
64), whose tanh the host takes (internal/sfx's amdDAC).

internal/sfx's stages zero every convolution's input outside the clip by masks. These graphs have
none: each convolution pads its input with zeros (the states' PAD rows), and the host places its
windows so that a window's frames before the clip's start or past its end are never read (a
window starts at the clip's first frame or reads past the halo it throws away).

The layout: a tensor is rows of time, its channels the columns. A convolution is an im2col of
IRON's Copy (its taps, dilated: rows t + j d of a padded state) and one GEMM a chunk of rows.
The transposed convolution of stride s (kernel 2 s) is a convolution of three taps (the input's
frames q - 1, q, q + 1) to s x out channels, phase major: its rows are the output's frames q s ..
q s + s - 1, the same bytes. Snake is float32 (iron_snake.py: IRON has no sin). The weight norms
are folded on the host.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import iron
from iron import Precision
from iron.operators.copy import Copy
from iron.operators.gemm import GEMM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_snake  # noqa: E402
from iron_anima import bf, command_boundary, tap  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402
from iron_moge import TorchZip  # noqa: E402

PREC = [Precision.HIGHEST]  # the GEMMs' of the stage being built (export's medium)
# the stages whose GEMMs are MEDIUM (bfp16), the rest HIGHEST (bf16): none. Against the CPU's
# decoder, all MEDIUM was at 25.8 dB (its partial sums in float32 too: the bfp16 inputs' error),
# stages 2 to 4 at 32.3 dB, none at 34.6 dB, for 20 % and 4 % less time
MEDIUM = ()
RATES = [8, 5, 4, 3, 2]  # decoder_rates
DIM, LATENT = 2048, 128
FRAMES = [768, 2048, 2048, 5120, 8192]  # each stage's input frames a call (multiples of 32; the flipped GEMMs' 512)
PAD = 32  # zero rows before and after a padded state's frames (the dilated convolution's 27)
CHUNK_BYTES = 16 << 20  # an im2col's most
BLOCK = 1024  # a state's most columns (Snake's, and a DMA's innermost 1023 words)
OUT_COLS = 64  # the output layer's one channel, padded for the GEMM
# IRON's GEMM splits its N across the array's columns in tiles of 64: to fewer channels than
# 512 (the eight columns' 64 each) the GEMM runs flipped, the channels its M and the rows its N
FLIP_UNDER = 512


def flipped(co: int) -> bool:
    """Whether a GEMM to co channels runs flipped (gemm)."""
    return co < FLIP_UNDER


def chunk_rows(T: int, K: int, co: int) -> int:
    """The rows of a convolution's chunk dividing T, its im2col at most CHUNK_BYTES: a multiple of
    512 flipped (the array's columns' N), else of 256 where one is (the array's four rows of 64:
    stage 0 over 544 frames, chunks of 1088 rows, took 410 ms a call, over 768, chunks of 1024,
    287 ms), else of 32."""
    for step in (512,) if flipped(co) else (256, 32):
        best = [r for r in range(step, T + 1, step) if T % r == 0 and r * K * 2 <= CHUNK_BYTES]
        if best:
            return best[-1]
    raise ValueError(f"no chunk of {T} rows for K {K}")


def gemm(cols, w, b):
    """cols (rows, K) @ w.T (w (out, K)) + b, its partial sums in float32 (one rounding to bf16 at
    MEDIUM's bfp16 too). Flipped (to fewer than FLIP_UNDER channels, over rows a multiple of 512:
    the array's columns' N) it is w @ cols.T, written column-major: the same (rows, out) in
    memory."""
    if flipped(w.shape[0]) and cols.shape[0] % 512 == 0:
        y = GEMM.step(w, cols, b_col_maj=True, c_col_maj=True, precision=PREC[0], accumulate_f32=True)
    else:
        y = GEMM.step(cols, w, b_col_maj=True, precision=PREC[0], accumulate_f32=True)
    return y + b


class Weights:
    """The VAE's decoder weights (the .pth's state_dict), the weight norms folded."""

    def __init__(self, path: Path):
        self.sd = TorchZip(path).obj["state_dict"]

    def __getitem__(self, name):
        return np.asarray(self.sd[name], np.float64)

    def norm(self, name):
        """weight_g weight_v / |weight_v|, the norm over all but the first axis."""
        v, g = self[name + ".weight_v"], self[name + ".weight_g"]
        n = np.sqrt((v.reshape(v.shape[0], -1) ** 2).sum(1))
        return v * (g.reshape(-1) / n).reshape((-1,) + (1,) * (v.ndim - 1))


class Feat:
    """A tensor of T frames as column blocks: each a state of PAD + T + PAD rows (padded) or T
    rows (dense), of at most BLOCK channels."""

    def __init__(self, blocks: list, T: int, padded: bool):
        self.blocks, self.T, self.padded = blocks, T, padded

    @property
    def C(self) -> int:
        return sum(b.shape[1] for b in self.blocks)

    def frames(self, i: int):
        b = self.blocks[i]
        return b[PAD : PAD + self.T] if self.padded else b[0 : self.T]


class Stage(iron.Graph):
    """internal/sfx's DACStage i over FRAMES[i] input frames."""

    def __init__(self, w: Weights, i: int):
        self.i, self.r = i, RATES[i]
        self.In = FRAMES[i]
        self.T = self.In * self.r
        self.params, self.states = [], []
        self.cin = DIM >> i
        self.cout = self.cin // 2
        blk = f"decoder.model.{i + 1}.block."
        if i == 0:
            pq = w["post_quant_conv.weight"][:, :, 0]
            self.pq = (self.p(pq), self.p(w["post_quant_conv.bias"]))
            self.first = self.conv_w(w.norm("decoder.model.0"), w["decoder.model.0.bias"])
        self.snake0 = self.snake_rows(w[blk + "0.alpha"])
        self.up = self.transposed_w(w.norm(blk + "1"), w[blk + "1.bias"], self.r)
        self.units = []
        for u, d in enumerate((1, 3, 9)):
            q = f"{blk}{u + 2}.block."
            self.units.append(dict(
                d=d, s1=self.snake_rows(w[q + "0.alpha"]), c1=self.conv_w(w.norm(q + "1"), w[q + "1.bias"]),
                s2=self.snake_rows(w[q + "2.alpha"]), c2=self.conv_w(w.norm(q + "3"), w[q + "3.bias"])))
        if i == len(RATES) - 1:
            self.snake_out = self.snake_rows(w["decoder.model.6.alpha"])
            k = w.norm("decoder.model.7")  # (1, 64, 7)
            kp = np.zeros((OUT_COLS,) + k.shape[1:])
            kp[:1] = k
            bp = np.zeros(OUT_COLS)
            bp[:1] = w["decoder.model.7.bias"]
            self.out_w = self.conv_w(kp, bp)

    # --- weights ---

    def p(self, x):
        h = iron.param(bf(x))
        self.params.append(h)
        return h

    def conv_w(self, k, bias):
        """A Conv1d's (out, k in) matrices for im2col's (tap, channel) order, in column blocks of
        the output, and its bias's."""
        co, ci, kk = k.shape
        m = k.transpose(0, 2, 1).reshape(co, kk * ci)
        blocks = [(self.p(m[o : o + BLOCK]), self.p(bias[o : o + BLOCK])) for o in range(0, co, BLOCK)]
        return dict(blocks=blocks, k=kk, ci=ci, co=co)

    def transposed_w(self, k, bias, s: int):
        """The ConvTranspose1d (in, out, 2 s) of stride s, padding ceil(s / 2), output padding s %
        2, as a convolution of 3 taps (frames q - 1, q, q + 1) to s out channels, phase major: the
        output's frame q s + b is x[q + j] k[:, :, b + p - j s] summed over the valid taps."""
        ci, co, kk = k.shape
        assert kk == 2 * s, k.shape
        p = (s + 1) // 2
        m = np.zeros((s, co, 3, ci))
        for b in range(s):
            for j in (-1, 0, 1):
                t = b + p - j * s
                if 0 <= t < kk:
                    m[b, :, j + 1, :] = k[:, :, t].T
        return dict(w=self.p(m.reshape(s * co, 3 * ci)), b=self.p(np.tile(bias, s)), s=s, ci=ci, co=co)

    def snake_rows(self, alpha):
        """Snake's rows (iron_snake.row) for each column block of the channels."""
        a = np.asarray(alpha).reshape(-1)
        out = []
        for c in range(0, len(a), BLOCK):
            part = a[c : c + BLOCK]
            out.append((self.p(iron_snake.row(part, iron_snake.line_pixels(len(part)) * len(part))), len(part)))
        return out

    # --- tensors ---

    def state(self, rows: int, cols: int):
        s = iron.state((rows, cols))
        self.states.append(s)
        return s

    def new(self, T: int, C: int, padded: bool) -> Feat:
        rows = T + 2 * PAD if padded else T
        return Feat([self.state(rows, min(BLOCK, C - c)) for c in range(0, C, BLOCK)], T, padded)

    def snake(self, x: Feat, rows: list, padded: bool) -> Feat:
        out = self.new(x.T, x.C, padded)
        for bi, (r, C) in enumerate(rows):
            y = iron_snake.snake(x.frames(bi), r, C)
            if padded:
                out.blocks[bi][PAD : PAD + x.T] = y
            else:
                out.blocks[bi][0 : x.T] = y
        return out

    def im2col(self, x: Feat, t0: int, R: int, k: int, d: int, pad: int):
        """The rows t0..t0 + R of the im2col of x (padded) for k taps of dilation d, the first at
        t - pad: (R, k C), tap major."""
        C = x.C
        K = k * C
        if len(x.blocks) == 1:
            b = x.blocks[0]
            return Copy.step(b, src=tap(tuple(b.shape), (PAD + t0 - pad) * C, [R, k, C], [C, d * C, 1])).reshape(R, K)
        cs = self.state(R, K)
        c0 = 0
        for b in x.blocks:
            cb = b.shape[1]
            Copy.step(b, cs, src=tap(tuple(b.shape), (PAD + t0 - pad) * cb, [R, k, cb], [cb, d * cb, 1]),
                      dst=tap((R, K), c0, [R, k, cb], [K, C, 1]))
            c0 += cb
        return cs[0:R]

    def conv(self, x: Feat, w: dict, d: int, padded: bool) -> Feat:
        """The Conv1d w ("same": padding d (k - 1) / 2) over x (padded) into a new Feat."""
        k = w["k"]
        pad = d * (k - 1) // 2
        T = x.T
        out = self.new(T, sum(min(BLOCK, w["co"] - o) for o in range(0, w["co"], BLOCK)), padded)
        R = chunk_rows(T, k * x.C, min(BLOCK, w["co"]))
        off = PAD if padded else 0
        for t0 in range(0, T, R):
            cols = self.im2col(x, t0, R, k, d, pad) if k > 1 else x.frames(0)[t0 : t0 + R] if len(x.blocks) == 1 else None
            if cols is None:
                raise ValueError("a 1 x 1 convolution over column blocks")
            for bi, (wm, bb) in enumerate(w["blocks"]):
                out.blocks[bi][off + t0 : off + t0 + R] = gemm(cols, wm, bb)
        command_boundary()
        return out

    def transposed(self, x: Feat, t: dict) -> Feat:
        """The transposed convolution t over x (padded): a dense Feat of x.T s frames."""
        s, co = t["s"], t["co"]
        out = self.new(x.T * s, co, False)
        R = chunk_rows(x.T, 3 * x.C, s * co)
        o = out.blocks[0]
        for t0 in range(0, x.T, R):
            cols = self.im2col(x, t0, R, 3, 1, 1)
            o[t0 * s : (t0 + R) * s] = gemm(cols, t["w"], t["b"]).reshape(R * s, co)
        command_boundary()
        return out

    def body(self, x):
        T = self.T
        if self.i == 0:
            p0 = self.new(self.In, LATENT, True)
            p0.blocks[0][PAD : PAD + self.In] = gemm(x, self.pq[0], self.pq[1])
            h = self.conv(p0, self.first, 1, False)
        else:
            h = Feat([x], self.In, False)
        h = self.transposed(self.snake(h, self.snake0, True), self.up)
        for u in self.units:
            y = self.conv(self.snake(h, u["s1"], True), u["c1"], u["d"], False)
            y = self.conv(self.snake(y, u["s2"], False), u["c2"], 1, False)
            n = self.new(T, h.C, False)
            n.blocks[0][0:T] = h.frames(0) + y.frames(0)
            command_boundary()
            h = n
        if self.i == len(RATES) - 1:
            h = self.conv(self.snake(h, self.snake_out, True), self.out_w, 1, False)
        return Copy.step(h.frames(0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the MOSS-SoundEffect repository (vae/vae_128d_48k.pth)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--stages", type=int, nargs="*", default=list(range(len(RATES))))
    ap.add_argument("--medium", type=int, nargs="*", default=list(MEDIUM),
                    help="the stages whose GEMMs are MEDIUM (bfp16); the others' are HIGHEST (bf16)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    bind_device(a.device, a.cols)
    a.out.mkdir(parents=True, exist_ok=True)
    export(a.root / "vae" / "vae_128d_48k.pth", a.out, a.stages, a.check, a.medium)
    return 0


def halo(i: int) -> int:
    """sfx.DACStage's Halo."""
    h = (3 + 9 + 27 + RATES[i] - 1) // RATES[i] + 2
    return h + 3 if i == 0 else h


def export(pth: Path, out: Path, stages: list, check: bool = False, medium=MEDIUM) -> None:
    """The stages' graphs into out/dac-N."""
    w = Weights(pth)
    rng = np.random.default_rng(0)
    for i in stages:
        t0 = time.time()
        PREC[0] = Precision.MEDIUM if i in medium else Precision.HIGHEST
        g = Stage(w, i)
        cin = LATENT if i == 0 else g.cin
        cg = g.compile(x=(g.In, cin))
        cout = OUT_COLS if i == len(RATES) - 1 else g.cout
        meta = dict(stage=i, precision=PREC[0].name.lower(), frames=g.In, rate=g.r, channels=cin, out=cout, halo=halo(i))
        name = f"dac-{i}"
        m = export_graph(cg, out / name, name, meta)
        print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan'].splitlines()[0]}", flush=True)
        if check:
            x = rng.standard_normal((g.In, cin)).astype(np.float32) * (1.0 if i == 0 else 0.1)
            write_check(g, m, out / name, dict(x=bf(x)))


if __name__ == "__main__":
    sys.exit(main())
