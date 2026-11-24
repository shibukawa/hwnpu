#!/usr/bin/env python3
"""Magenta RealTime 2 (google/magenta-realtime-2: code Apache-2.0, weights CC-BY 4.0) on IRON:
SpectroStream's decoder, which bonsaiai-audio generate runs on AMD's NPU (the language model runs
on the CPU: internal/music).

  python tools/amd/iron_magenta.py ~/models/magenta-realtime-2 ~/models/magenta-realtime-2-small.amd-npu2

writes decoder/, a graph (iron_graph.py) of internal/music's Decoder over a window of up to 48
frames (internal/music's 41, DecodeFrames and decodeContext): input x (64, 256), each frame's
sum of its codes' quantizer embeddings, the window's first frame in row 0 (the rows after its
frames anything: the decoder is causal in time); outputs out0 and out1 (192 x 480, 2), the
left and the right channel's STFT frames (time-major, then the bin), the real part in column
0 and the imaginary in 1.

The program of internal/music zeroes, by its masks, every convolution's input before the clip
(the window's frames before frame 0). Here there are no masks: the host puts the clip's first
frame in row 0, and the zero rows above every grid are the causal convolutions' padding, which
gives the same values. bf16 needs none of the fp16 program's rescaling (decoderScales).

The layout: a tensor is rows of (time, frequency), its channels the columns. A convolution reads
a padded grid (zero rows above, zero columns either side) by an im2col of IRON's Copy (taps),
then one GEMM. A transposed convolution is a convolution for each phase of its strides (the
output's time and frequency index modulo them) over the input, its output scattered into the
phases' places. The nearest upsampling of the shortcuts is the same scatter. The ELUs are
float32 (iron_elu.py).
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

import iron
import iron.functional as F
from iron import Precision
from iron.operators.copy import Copy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_elu  # noqa: E402
from iron_anima import RandomWeights, Safetensors, bf, command_boundary, tap  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

PREC = Precision.HIGHEST
FRAMES = 48  # the window's frames (internal/music's 41, padded)
IN_ROWS = 64  # the input's rows (the 1 x 1 convolutions' GEMMs take multiples of 32)
CHUNK_BYTES = 16 << 20  # an im2col's most
DMA_INNER = 2046  # a DMA's innermost size, in bf16 values (1023 words)
SLACK = 40  # zero rows after a grid's: a convolution's last chunk may run past its rows
DEC = "decoder/"

# internal/music's decoderLevels: name, strides, the transposed kernel, in, out, shortcut conv
LEVELS = [
    ("decoder_1", 2, 2, "conv2dtranspose_4x4", 512, 256, True),
    ("decoder_2", 1, 2, "conv2dtranspose_3x4", 256, 256, False),
    ("decoder_3", 1, 2, "conv2dtranspose_3x4", 256, 256, False),
    ("decoder_4", 1, 3, "conv2dtranspose_3x6", 256, 128, True),
    ("decoder_5", 1, 2, "conv2dtranspose_3x4", 128, 128, False),
    ("decoder_6", 1, 2, "conv2dtranspose_3x4", 128, 64, True),
]


def f32(x) -> np.ndarray:
    return np.array(x, np.float32)


def round32(n: int) -> int:
    return -(-n // 32) * 32


@dataclasses.dataclass(frozen=True)
class Grid:
    """A padded grid's shape: T time rows below ``top`` zero rows, F bins between ``side`` zero
    columns, C channels; a zero row after. Its state is rows x C."""

    T: int
    F: int
    C: int
    top: int = 2
    side: int = 1

    @property
    def Fp(self) -> int:
        return self.F + 2 * self.side

    @property
    def rows(self) -> int:
        # SLACK rows after: a chunk of a convolution may read past T (conv_on)
        return (self.top + self.T + SLACK) * self.Fp

    def pixel(self, t: int, f: int) -> int:
        return (self.top + t) * self.Fp + self.side + f


class Decoder(iron.Graph):
    """SpectroStream's decoder (internal/music's Decoder) over FRAMES frames."""

    def __init__(self, w):
        self.params, self.states = [], []
        self.w = w
        k = lambda n: f32(w[DEC + n + "/kernel"])  # noqa: E731
        b = lambda n: f32(w[DEC + n + "/bias"])  # noqa: E731
        il = "input_layer/"
        self.first = self.p(k(il + "conv1x1_first/conv")[0, 0].T), self.p(b(il + "conv1x1_first/conv"))
        self.b1 = self.p(k(il + "shortcut_layer/conv1x1_b1/conv")[0, 0].T), self.p(b(il + "shortcut_layer/conv1x1_b1/conv"))
        self.b2 = self.p(k(il + "shortcut_layer/conv1x1_b2/conv")[0, 0].T), self.p(b(il + "shortcut_layer/conv1x1_b2/conv"))
        self.res_a = self.conv_w("input_layers_residual_unit/conv2d_3x3_a/conv")
        self.res_b = self.conv_w("input_layers_residual_unit/conv2d_3x3/conv")
        self.d0_t = self.transposed_w("decoder_0/conv2dtranspose_4x3/conv", 2, 1)
        # decoder_0's 3 x 3 and shortcut in two halves of the output channels (left, right)
        kc, bc = k("decoder_0/conv2d_3x3/conv"), b("decoder_0/conv2d_3x3/conv")
        ks, bs = k("decoder_0/shortcut_layer/conv1x1/conv"), b("decoder_0/shortcut_layer/conv1x1/conv")
        self.d0_c = [self.conv_w(None, kc[..., h * 512 : (h + 1) * 512], bc[h * 512 : (h + 1) * 512]) for h in range(2)]
        self.d0_s = [(self.p(ks[0, 0][:, h * 512 : (h + 1) * 512].T), self.p(bs[h * 512 : (h + 1) * 512])) for h in range(2)]
        self.levels = []
        for name, st, sf, tname, ci, co, sc in LEVELS:
            lv = dict(name=name, st=st, sf=sf, ci=ci, co=co,
                      t=self.transposed_w(f"{name}/{tname}/conv", st, sf), c=self.conv_w(f"{name}/conv2d_3x3/conv"))
            if sc:
                lv["s"] = self.p(k(f"{name}/shortcut_layer/conv1x1/conv")[0, 0].T), self.p(b(f"{name}/shortcut_layer/conv1x1/conv"))
            self.levels.append(lv)
        # the output layer: a 7 x 7 convolution to the real and imaginary parts. Its im2col takes
        # the time taps alone (K 7 x 64), its GEMM the bin taps as columns (bin tap e, part o in
        # column 2 e + o: 14 of 64), which out_sum shifts and adds; the bias in tap 3's
        ko, bo = k("output_layer/base_conv_last/conv"), b("output_layer/base_conv_last/conv")
        kt, kf, ci, _ = ko.shape
        kp = np.zeros((kt, 1, ci, 64), np.float32)
        for e in range(kf):
            kp[:, 0, :, 2 * e : 2 * e + 2] = ko[:, e]
        bp = np.zeros(64, np.float32)
        bp[2 * (kf // 2) : 2 * (kf // 2) + 2] = bo
        self.out_w = self.conv_w(None, kp, bp)
        self.out_kf = kf

    # --- weights ---

    def p(self, x):
        h = iron.param(bf(x))
        self.params.append(h)
        return h

    def conv_w(self, name, kernel=None, bias=None):
        """A convolution's (out, kt kf in) matrix of a Flax kernel (kt, kf, in, out) and its bias."""
        if kernel is None:
            kernel, bias = f32(self.w[DEC + name + "/kernel"]), f32(self.w[DEC + name + "/bias"])
        kt, kf, ci, co = kernel.shape
        return dict(w=self.p(kernel.reshape(kt * kf * ci, co).T), b=self.p(bias), kt=kt, kf=kf, ci=ci, co=co)

    def transposed_w(self, name, st, sf):
        """A transposed convolution's phases: for each (pt, pf), the (out, Mt Nf in) matrix of the
        taps over the input it reads, in im2col's order (time rows q - Mt + 1 .. q, bins q' -
        Nf + 1 .. q'), from a Flax ConvTranspose kernel (kt, kf, in, out), the true transposed
        convolution's kernel reversed in both axes; and where its output goes."""
        k = f32(self.w[DEC + name + "/kernel"])
        bias = f32(self.w[DEC + name + "/bias"])
        kt, kf, ci, co = k.shape
        Mt, Nf = -(-kt // st), -(-kf // sf)
        fl = max(0, kf - sf) // 2
        phases = []
        for pt in range(st):
            for pf in range(sf):
                m = np.zeros((Mt, Nf, ci, co), np.float32)
                for a2 in range(Mt):
                    for e2 in range(Nf):
                        mt, nf = Mt - 1 - a2, Nf - 1 - e2
                        a, e = mt * st + pt, nf * sf + pf
                        if a < kt and e < kf:
                            m[a2, e2] = k[kt - 1 - a, kf - 1 - e]
                # the output bin f = (q + qf0) sf + pf - fl of the input's bin q: qf0 the first q'
                # whose bin is not before 0
                qf0 = 0 if pf >= fl else 1
                phases.append(dict(pt=pt, pf=pf, qf0=qf0, Mt=Mt, Nf=Nf, fl=fl,
                                   w=self.p(m.reshape(Mt * Nf * ci, co).T), b=self.p(bias)))
        return dict(st=st, sf=sf, ci=ci, co=co, phases=phases)

    # --- operations ---

    def grid(self, g: Grid):
        s = iron.state((g.rows, g.C))
        self.states.append(s)
        return s

    def to_grid(self, x, T: int, F_: int, C: int, top: int = 2, side: int = 1):
        """x (rows of T x F, C), dense, into the interior of a new padded grid."""
        g = Grid(T, F_, C, top, side)
        s = self.grid(g)
        Copy.step(x, s, src=tap((x.shape[0], C), 0, [T, F_ * C], [F_ * C, 1]),
                  dst=tap((g.rows, C), g.pixel(0, 0) * C, [T, F_ * C], [g.Fp * C, 1]))
        return s[0 : g.rows], g

    def chunk_rows(self, T: int, F_: int, K: int) -> int:
        """Output time rows a convolution's im2col takes at once: rows x F a multiple of 32, at
        most CHUNK_BYTES, the chunks covering T with as little past it as may be (the last may
        run past T, by less than SLACK rows)."""
        u = 32 // np.gcd(F_, 32)
        best = None
        for r in range(u, -(-T // u) * u + 1, u):
            if r * F_ * K * 2 > CHUNK_BYTES:
                break
            over = -(-T // r) * r - T
            if over < SLACK and (best is None or over <= best[1]):
                best = (r, over)
        if best is None:
            raise ValueError(f"no chunk of {T} rows of {F_} bins for K {K}")
        return best[0]

    def conv_on(self, gs, g: Grid, w: dict, t_back: int, f_left: int, Tout: int, Fout: int, out=None):
        """y[t, f] = sum over the taps of w (kt x kf) of x[t - t_back + a, f - f_left + e], for t
        < Tout, f < Fout, x the grid gs; into out (a dense state, rows of Tout x Fout)."""
        K = w["kt"] * w["kf"] * g.C
        R = self.chunk_rows(Tout, Fout, K)
        if out is None:
            out = iron.state((max(round32(Tout * Fout), -(-Tout // R) * R * Fout), w["co"]))
            self.states.append(out)
        split = w["kf"] * g.C > DMA_INNER
        if split:
            # a tap's C values at a time: a DMA's innermost size is at most 1023 words
            cs = iron.state((R * Fout, K))
            self.states.append(cs)
        for t0 in range(0, Tout, R):
            if split:
                for a in range(w["kt"]):
                    for e in range(w["kf"]):
                        Copy.step(gs, cs, src=tap((g.rows, g.C), g.pixel(t0 - t_back + a, e - f_left) * g.C, [R, Fout, g.C], [g.Fp * g.C, g.C, 1]),
                                  dst=tap((R * Fout, K), (a * w["kf"] + e) * g.C, [R * Fout, g.C], [K, 1]))
                cols = cs[0 : R * Fout]
            else:
                cols = Copy.step(gs, src=tap((g.rows, g.C), g.pixel(t0 - t_back, -f_left) * g.C,
                                             [R, Fout, w["kt"], w["kf"] * g.C], [g.Fp * g.C, g.C, g.Fp * g.C, 1])).reshape(R * Fout, K)
            out[t0 * Fout : (t0 + R) * Fout] = F.linear(cols, w["w"], w["b"], precision=PREC)
        return out

    def conv3(self, x, T: int, F_: int, w: dict):
        """ELU, then the causal 3 x 3 convolution (time padded 2 before, the bins 1 either side)."""
        gs, g = self.to_grid(iron_elu.elu(x), T, F_, w["ci"])
        out = self.conv_on(gs, g, w, 2, 1, T, F_)
        command_boundary()
        return out[0 : round32(T * F_)]

    def transposed(self, x, T: int, F_: int, t: dict):
        """ELU, then the transposed convolution of t: (T st) x (F sf) out, causal in time (its
        last kt - st steps off), "same" in the bins."""
        st, sf, co = t["st"], t["sf"], t["co"]
        To, Fo = T * st, F_ * sf
        gs, g = self.to_grid(iron_elu.elu(x), T, F_, t["ci"], top=2, side=2)
        out = iron.state((round32(To * Fo), co))
        self.states.append(out)
        for ph in t["phases"]:
            w = dict(w=ph["w"], b=ph["b"], kt=ph["Mt"], kf=ph["Nf"], co=co)
            y = self.conv_on(gs, g, w, ph["Mt"] - 1, ph["Nf"] - 1 - ph["qf0"], T, F_)
            off = (ph["pt"] * Fo + ph["qf0"] * sf + ph["pf"] - ph["fl"]) * co
            Copy.step(y[0 : y.shape[0]], out, src=tap((y.shape[0], co), 0, [T, F_ * co], [F_ * co, 1]),
                      dst=tap((out.shape[0], co), off, [T, F_, co], [st * Fo * co, sf * co, 1]))
        command_boundary()
        return out[0 : out.shape[0]]

    def upsample(self, x, T: int, F_: int, C: int, st: int, sf: int):
        """The nearest upsampling of x (T x F) by st x sf."""
        if st == 1 and sf == 1:
            return x
        To, Fo = T * st, F_ * sf
        out = iron.state((round32(To * Fo), C))
        self.states.append(out)
        for dt in range(st):
            for df in range(sf):
                Copy.step(x, out, src=tap((x.shape[0], C), 0, [T, F_ * C], [F_ * C, 1]),
                          dst=tap((out.shape[0], C), (dt * Fo + df) * C, [T, F_, C], [st * Fo * C, sf * C, 1]))
        return out[0 : out.shape[0]]

    def out_sum(self, y, T: int, F_: int, Fp: int):
        """The output layer's sum over its bin taps: y holds, for each pixel (t, f) of the padded
        bins (Fp: F and 3 either side), column 2 e + o of bin tap e; out (T x F, 2) is y[t, f + e,
        2 e + o] summed over e."""
        C = y.shape[1]
        total = None
        for e in range(self.out_kf):
            part = Copy.step(y, src=tap(tuple(y.shape), e * C + 2 * e, [T, F_, 2], [Fp * C, C, 1]))
            total = part if total is None else total + part
        return total.reshape(T * F_, 2)

    def body(self, x):
        T = FRAMES
        lin = lambda y, wb: F.linear(y, wb[0], wb[1], precision=PREC)  # noqa: E731
        # input_layer: conv1x1_first(x) + conv1x1_b2(elu(conv1x1_b1(x))): (rows, 2560), the
        # channel f 512 + c: as rows of (t, f), 512 channels
        h = lin(x, self.first) + lin(iron_elu.elu(lin(x, self.b1)), self.b2)
        h = h.reshape(IN_ROWS * 5, 512)
        command_boundary()
        # input_layers_residual_unit (over the input's rows: IN_ROWS frames)
        r = self.conv3(self.conv3(h, IN_ROWS, 5, self.res_a), IN_ROWS, 5, self.res_b)
        h = h + r
        # decoder_0: time stride 2, 1024 channels; from here on the first FRAMES frames
        r = self.transposed(h, T, 5, self.d0_t)  # (2T x 5, 1024)
        gs, g = self.to_grid(iron_elu.elu(r), 2 * T, 5, 1024)
        halves = []
        for hh in range(2):
            rc = self.conv_on(gs, g, self.d0_c[hh], 2, 1, 2 * T, 5)
            s = lin(h, self.d0_s[hh])  # (rows of IN_ROWS x 5, 512)
            halves.append(rc[0 : round32(2 * T * 5)] + self.upsample(s, T, 5, 512, 2, 1))
            command_boundary()
        # the levels, each half with the same weights
        Tc, Fc = 2 * T, 5
        outs = []
        for hh in range(2):
            h, Tl, Fl = halves[hh], Tc, Fc
            for lv in self.levels:
                st, sf = lv["st"], lv["sf"]
                r = self.transposed(h, Tl, Fl, lv["t"])
                Tn, Fn = Tl * st, Fl * sf
                r = self.conv3(r, Tn, Fn, lv["c"])
                s = lin(h, lv["s"]) if "s" in lv else h
                h = r + self.upsample(s, Tl, Fl, lv["co"], st, sf)
                Tl, Fl = Tn, Fn
                command_boundary()
            # the output layer: ELU, a causal 7 x 7 convolution (time 6 before, the bins 3 either side)
            gs, g = self.to_grid(iron_elu.elu(h), Tl, Fl, 64, top=6, side=3)
            outs.append(self.out_sum(self.conv_on(gs, g, self.out_w, 6, 3, Tl, g.Fp), Tl, Fl, g.Fp))
            command_boundary()
        return tuple(outs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the magenta-realtime-2 repository (checkpoints/mrt2_small.safetensors)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    bind_device(a.device, a.cols)
    a.out.mkdir(parents=True, exist_ok=True)
    w = (RandomWeights if a.random else Safetensors)(a.root / "checkpoints" / "mrt2_small.safetensors", prefix="params/soundstream/")
    t0 = time.time()
    g = Decoder(w)
    cg = g.compile(x=(IN_ROWS, 256))
    meta = dict(frames=FRAMES, rows=IN_ROWS, time_out=4 * FRAMES, bins=480)
    m = export_graph(cg, a.out / "decoder", "decoder", meta)
    print(f"decoder: compiled in {time.time() - t0:.0f} s; {m['graph']['plan'].splitlines()[0]}", flush=True)
    if a.check:
        x = np.random.default_rng(0).standard_normal((IN_ROWS, 256)).astype(np.float32) * 0.3
        write_check(g, m, a.out / "decoder", dict(x=bf(x)))
    (a.out / "model.json").write_text(json.dumps(dict(name="magenta-realtime-2-small", kind="magenta-rt2", **meta), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
