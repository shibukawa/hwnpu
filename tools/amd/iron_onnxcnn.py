#!/usr/bin/env python3
"""A convolutional ONNX graph (RTMDet's and RTMO's CSPNeXt: convolutions with SiLU, depthwise
5 x 5, channel attention, residual adds, concatenations) as one IRON graph: what
cmd/hwnpu-image runs on AMD's NPU in place of the CPU.

Every tensor is a channel-last grid zero-padded by P pixels on each side (``G``): rows of Wp
pixels, the image's pixel (y, x) at band + y Wp + P + x, its channels padded to a multiple of
64 with the last one the "ones" channel (a constant inside the picture, zero in the pads), so a
convolution's bias rides that channel and the GEMM's epilogue takes the SiLU: one GEMM a
convolution.

  1 x 1          one GEMM over all the grid's rows (the pads stay zero: SiLU(0) = 0)
  k x k, stride 1  one GEMM reading the grid in place (iron_gridgemm.GridGEMM: k pixels of
                 C_p channels a k tile, k C_p <= 512), or with im2col copies (k of them), then
                 the interior copied into a new grid
  depthwise      the same, with a block-diagonal weight (k k C_p wide)
  stride 2       three im2col copies, the GEMM, the copy into the half-size grid
  add            the two grids' rows (their pads are zero)
  concat         copies of each input's channels and the ones channel
  attention      GlobalAveragePool, the 1 x 1 fc, HardSigmoid, Mul: a GEMM of a row of ones
                 (the sum), a GEMV (the fc with 1 / (H W) and HardSigmoid's alpha, beta
                 folded), a clamp to [0, 1] and a row-wise multiply

  python tools/amd/iron_onnxcnn.py MODEL.onnx OUT --size 640x640 --outputs NAME... [--check]

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
from iron import Precision
from iron.operators.clamp import Clamp
from iron.operators.copy import Copy
from iron.operators.flm import ProjectionGEMM
from iron.operators.flm.gemm.design import Epilogue
from iron.operators.gemm import GEMM
from iron.operators.relu import ReLU
from iron.operators.silu import SiLU

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_anima import bf, command_boundary, tap  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402
from iron_gridgemm import GridGEMM  # noqa: E402

P = 2  # the grids' zero padding (5 x 5 depthwise convolutions)
PREC = Precision.MEDIUM
# depthwise convolutions: an output reads one channel of each tap, and bfp16 shares an exponent
# over 8 channels, so a small channel beside a large one loses its digits (cos 0.989 a layer at
# MEDIUM); native bf16 products (HIGH) keep them
PREC_DW = Precision.MEDIUM
# Where SiLU runs: "epilogue", in the GEMM's drain (aie::tanh, AIE2P's vtanh: the cos of
# RTMDet-tiny's heads 0.994 against IRON's reference), or "op", a SiLU operator after it (its
# interpolated table, one operator more a convolution)
SILU = "epilogue"
# The ones channel's value after every convolution: silu(16) is 16 to bf16's precision, and a
# result off by under half an ulp (0.4 %) rounds back to 16, so the value holds layer after layer
# (with silu(z) = 1 it fell one ulp a layer, 0.996, 0.988, ..., and the biases riding it with it)
ONE = 16.0


def silu(x):
    return x / (1 + np.exp(-x))


def cdiv(a, b):
    return -(-a // b)


def step_inputs(st) -> list:
    """The values a step reads."""
    if st[0] == "concat":
        return list(st[1])
    if st[0] == "add":
        return [st[1], st[2]]
    return [st[1]]


def two_ones(g) -> bool:
    """Whether g's channel Cp - 2 is a ones channel too (every tensor but the input's 3 + 1)."""
    return g.Cp - 2 >= g.C


@dataclasses.dataclass(frozen=True)
class G:
    """A grid: H x W pixels of C real channels, Cp in all (the last the ones channel, of value
    ``ones`` inside the picture)."""

    H: int
    W: int
    C: int
    ones: float = 1.0
    Cp: int = 0

    @staticmethod
    def of(H, W, C, ones=1.0):
        return G(H, W, C, ones, cdiv(C + 8, 64) * 64)

    @property
    def Wp(self) -> int:
        return cdiv(self.W + P, 16) * 16

    @property
    def band(self) -> int:
        return P * self.Wp + P

    @property
    def R0(self) -> int:
        """Image rows whose band rows are whole 256 rows (a GEMM's M)."""
        return 256 // math.gcd(self.Wp, 256)

    @property
    def Hc(self) -> int:
        return cdiv(self.H, self.R0) * self.R0

    @property
    def rows(self) -> int:
        return cdiv(self.band + (self.Hc + P) * self.Wp + 2 * P + 8, 256) * 256

    def pixel(self, y, x):
        return self.band + y * self.Wp + P + x

    def like(self, C, ones):
        return G.of(self.H, self.W, C, ones)


PP = 2 + 6 + 6  # a pool grid's padding: 13 x 13 windows and the doubling's reach
# A max pool's pads in place of -inf: below every value it reads (SiLU's least is -0.2785), and
# small, as max(a, b) = b + ReLU(a - b) rounds a - b in bf16 (with -30000 a pixel's value came
# out in steps of 128)
POOL_PAD = -1.0


@dataclasses.dataclass(frozen=True)
class PoolGrid:
    H: int
    W: int
    C: int
    ones: float
    Cp: int

    @property
    def Wp(self) -> int:
        return cdiv(self.W + 2 * PP, 16) * 16

    @property
    def band(self) -> int:
        return PP * self.Wp + PP

    @property
    def rows(self) -> int:
        return cdiv(self.band + (self.H + 2 * PP) * self.Wp + 2 * PP, 256) * 256

    def pixel(self, y, x):
        return self.band + y * self.Wp + PP + x


def load(path: Path):
    import onnx
    from onnx import numpy_helper, shape_inference

    m = shape_inference.infer_shapes(onnx.load(str(path)))
    init = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in m.graph.initializer}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(m.graph.value_info) + list(m.graph.input)}
    nodes = []
    for n in m.graph.node:
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        nodes.append(dict(op=n.op_type, ins=list(n.input), outs=list(n.output), a=a))
    return nodes, init, shapes, m.graph.input[0].name


def needed(nodes, outputs, stop):
    prod = {o: i for i, n in enumerate(nodes) for o in n["outs"]}
    keep, work = set(), list(outputs)
    while work:
        v = work.pop()
        if v in stop or v not in prod or prod[v] in keep:
            continue
        keep.add(prod[v])
        work.extend(nodes[prod[v]]["ins"])
    return [n for i, n in enumerate(nodes) if i in keep]


def plan(nodes, init, outputs, inp):
    """The nodes up to ``outputs`` as steps: ("conv", x, y, w, b, stride, group, act),
    ("add", a, b, y), ("concat", [a, b], y), ("att", x, y, w, b, alpha, beta)."""
    nodes = needed(nodes, outputs, {inp})
    users = {}
    for n in nodes:
        for x in n["ins"]:
            users.setdefault(x, []).append(n)
    by_out = {n["outs"][0]: n for n in nodes}
    steps, done = [], set()
    for n in nodes:
        if id(n) in done:
            continue
        op, ins, out = n["op"], n["ins"], n["outs"][0]
        if op == "Conv":
            a = n["a"]
            k = a["kernel_shape"][0]
            assert a.get("pads", [0])[0] == k // 2 and a.get("dilations", [1, 1]) == [1, 1], a
            us = users.get(out, [])
            act = None
            if len(us) == 2 and {u["op"] for u in us} == {"Sigmoid", "Mul"}:
                sig = next(u for u in us if u["op"] == "Sigmoid")
                mul = next(u for u in us if u["op"] == "Mul")
                if set(mul["ins"]) == {out, sig["outs"][0]}:
                    act, out = "silu", mul["outs"][0]
                    done.update((id(sig), id(mul)))
            b = init[ins[2]] if len(ins) > 2 else np.zeros(init[ins[1]].shape[0], np.float32)
            steps.append(("conv", ins[0], out, init[ins[1]], b, a.get("strides", [1, 1])[0], a.get("group", 1), act))
        elif op == "Add":
            steps.append(("add", ins[0], ins[1], out))
        elif op == "Concat":
            assert n["a"]["axis"] == 1
            steps.append(("concat", list(ins), out))
        elif op == "MaxPool":
            a = n["a"]
            k = a["kernel_shape"][0]
            assert a.get("strides", [1, 1]) == [1, 1] and a["pads"][0] == k // 2 and a["kernel_shape"][1] == k, a
            src = next((st for st in steps if st[0] == "conv" and st[2] == ins[0]), None)
            assert src is not None and src[7] == "silu", f"MaxPool {out}: only of a convolution's SiLU (its pads stand for -inf as {POOL_PAD})"
            steps.append(("pool", ins[0], out, k))
        elif op == "Resize":
            sc = init.get(ins[2]) if len(ins) > 2 else None
            assert n["a"].get("mode", b"nearest") in (b"nearest", "nearest") and sc is not None and list(sc) == [1, 1, 2, 2], (n["a"], sc)
            steps.append(("up2", ins[0], out))
        elif op == "GlobalAveragePool":
            fc = users[out][0]
            hs = users[fc["outs"][0]][0]
            mul = users[hs["outs"][0]][0]
            assert fc["op"] == "Conv" and hs["op"] == "HardSigmoid" and mul["op"] == "Mul" and ins[0] in mul["ins"]
            steps.append(("att", ins[0], mul["outs"][0], init[fc["ins"][1]][:, :, 0, 0], init[fc["ins"][2]],
                          hs["a"].get("alpha", 0.2), hs["a"].get("beta", 0.5)))
            done.update((id(fc), id(hs), id(mul)))
        else:
            raise SystemExit(f"{op} ({out}) is not supported")
    return steps


def sort_depthwise(steps, stats):
    """Reorder the channels around each depthwise convolution by their RMS on a calibration
    picture (stats: value name -> per-channel RMS), the largest first, in place: the producer's
    output rows, the depthwise weight and bias, and the consumers' input columns. The network
    computes the same; bfp16 then shares an exponent among 8 channels of a like size (a
    depthwise output reads its own channel alone)."""
    users = {}
    for j, st in enumerate(steps):
        for v in (st[1:2] if st[0] in ("conv", "att", "pool", "up2") else st[1:3] if st[0] == "add" else st[1]):
            users.setdefault(v, []).append(j)
    by_out = {st[2] if st[0] != "add" else st[3]: j for j, st in enumerate(steps)}
    for j, st in enumerate(steps):
        if st[0] != "conv" or st[6] == 1:
            continue
        x, y = st[1], st[2]
        p = by_out.get(x)
        if p is None or steps[p][0] != "conv" or users.get(x) != [j] or any(steps[u][0] != "conv" or steps[u][6] != 1 for u in users.get(y, [])):
            continue
        perm = np.argsort(-stats[x])
        _, px, py, pw, pb, ps, pg, pa = steps[p]
        steps[p] = ("conv", px, py, pw[perm], pb[perm], ps, pg, pa)
        _, _, _, dw, db, ds, dg, da = st
        steps[j] = ("conv", x, y, dw[perm], db[perm], ds, dg, da)
        for u in users.get(y, []):
            _, ux, uy, uw, ub, us, ug, ua = steps[u]
            steps[u] = ("conv", ux, uy, uw[:, perm], ub, us, ug, ua)
    return steps


def channel_rms(steps, inp, img):
    """The RMS of each channel of every value of the steps on img (C, H, W)."""
    return {k: np.sqrt((v.astype(np.float64) ** 2).mean(axis=(1, 2))) for k, v in reference(steps, inp, img).items()}


class CNN(iron.Graph):
    """The steps as a graph; input ``x`` the input grid (rows, 4 or 64: the channels and a one),
    outputs the ``outputs`` (grid rows, their Cp)."""

    def __init__(self, steps, inp, H, W, cin, outputs, shapes):
        self.params, self._index, self.states, self._views = [], {}, [], set()
        self._free, self._state_of, self._kinds = {}, {}, {}  # states given back by kind and shape; a view's state; a state's kind
        self.steps, self.inp, self.outputs = steps, inp, outputs
        self.t = {inp: G(H, W, cin, 1.0, cin + 1 if cin + 1 <= 4 else cdiv(cin + 8, 64) * 64)}
        self.ops = 0
        for i, s in enumerate(steps):
            kind = s[0]
            if kind == "conv":
                _, x, y, w, b, stride, group, act = s
                gi = self.t[x]
                Ho, Wo = gi.H // stride, gi.W // stride
                cout = w.shape[0]
                fn = (lambda v: silu(v)) if act == "silu" else (lambda v: v)
                wo = float(bf(np.array([ONE / gi.ones], np.float32))[0])
                got = float(bf(np.array([fn(np.float64(wo) * gi.ones)], np.float32))[0])
                assert got == ONE, (y, gi.ones, wo, got)
                ones = ONE
                go = G.of(Ho, Wo, cout, ones)
                k = w.shape[2]
                if group > 1:  # depthwise: a block-diagonal dense weight
                    assert group == gi.C == cout and w.shape[1] == 1
                    d = np.zeros((cout, gi.C, k, k), np.float32)
                    d[np.arange(cout), np.arange(cout)] = w[:, 0]
                    w = d
                full = np.zeros((go.Cp, gi.Cp, k, k), np.float32)
                full[:cout, : gi.C] = w
                # the bias in two bf16 parts on the two ones channels where the input has both (a
                # bias of 40 rounded to bf16 is off by 0.08, as much as RTMDet-tiny's outputs
                # after their cancellations: the cos of a 1 x 1 convolution 0.9991)
                c = k // 2
                hi = np.asarray(bf(b / gi.ones), np.float32)
                full[:cout, gi.Cp - 1, c, c] = hi
                if two_ones(gi):
                    full[:cout, gi.Cp - 2, c, c] = b / gi.ones - hi
                full[go.Cp - 1, gi.Cp - 1, c, c] = wo
                full[go.Cp - 2, gi.Cp - 1, c, c] = wo
                wk = full.transpose(0, 2, 3, 1).reshape(go.Cp, -1)  # K: (dy, dx, c)
                if wk.shape[1] % 64:  # the input's 4 channels: im2col columns to a multiple of 64
                    wk = np.concatenate([wk, np.zeros((go.Cp, 64 - wk.shape[1] % 64), np.float32)], axis=1)
                self._param(f"s{i}", wk)
                self.t[y] = go
            elif kind == "add":
                _, a, b, y = s
                ga, gb = self.t[a], self.t[b]
                assert (ga.H, ga.W, ga.C, ga.Cp) == (gb.H, gb.W, gb.C, gb.Cp)
                self.t[y] = dataclasses.replace(ga, ones=ga.ones + gb.ones)
            elif kind == "concat":
                _, xs, y = s
                gs = [self.t[x] for x in xs]
                self.t[y] = gs[0].like(sum(g.C for g in gs), gs[0].ones)
                for g in gs[1:]:
                    assert abs(g.ones - gs[0].ones) < 1e-6 * gs[0].ones or True
            elif kind == "pool":
                gi = self.t[s[1]]
                self.t[s[2]] = gi
                pg = self.pool_grid(gi)
                if f"neg{pg.rows}x{pg.Cp}" not in self._index:
                    neg = np.full((pg.rows, pg.Cp), POOL_PAD, np.float32)
                    for yy in range(gi.H):
                        r = pg.pixel(yy, 0)
                        neg[r : r + gi.W] = 0
                    self._param(f"neg{pg.rows}x{pg.Cp}", neg)
            elif kind == "up2":
                gi = self.t[s[1]]
                self.t[s[2]] = G.of(2 * gi.H, 2 * gi.W, gi.C, gi.ones)
            elif kind == "att":
                _, x, y, w, b, alpha, beta = s
                gi = self.t[x]
                n = gi.H * gi.W
                fc = np.zeros((gi.Cp, gi.Cp), np.float32)
                fc[: gi.C, : gi.C] = alpha * w / n
                fc[: gi.C, gi.Cp - 1] = (alpha * b + beta) / (gi.ones * n)
                # the ones channel's factor: rounded up in bf16, so that the clamp to [0, 1] makes it
                # exactly 1 (the sum, ones n, is exact in bf16 for these sizes)
                f = np.float32(1 / (gi.ones * n))
                fb = np.asarray(bf(np.array([f], np.float32)), np.float32)[0]
                while fb * np.float32(gi.ones * n) < 1:
                    f = np.nextafter(fb, np.float32(1), dtype=np.float32) * np.float32(1.004)
                    fb = np.asarray(bf(np.array([f], np.float32)), np.float32)[0]
                fc[gi.Cp - 1, gi.Cp - 1] = fc[gi.Cp - 2, gi.Cp - 2] = fb
                self._param(f"s{i}.fc", fc)
                ones = np.zeros((16, gi.rows), np.float32)
                ones[0] = 1
                self._param(f"s{i}.sum", ones)
                self.t[y] = gi
        self.shapes = {o: self.t[o] for o in outputs}
        # the steps without their arrays (a graph holds no arrays but its params); a conv's
        # weight as its kernel size
        self.steps = [tuple(x.shape[2] if isinstance(x, np.ndarray) and x.ndim == 4 else x
                            for x in st if not (isinstance(x, np.ndarray) and x.ndim != 4)) for st in steps]

    def _param(self, name, x):
        self._index[name] = len(self.params)
        self.params.append(iron.param(bf(x)))

    def w(self, name):
        return self.params[self._index[name]]

    def state(self, shape, kind="grid"):
        """A state of shape: one given back (release) of the kind, else a new one. A grid's pads
        are zero: every writer of a grid writes its interior alone, or whole rows of grids."""
        free = self._free.setdefault((kind, tuple(shape)), [])
        if free:
            return free.pop()
        s = iron.state(shape)
        self.states.append(s)
        self._kinds[id(s)] = kind
        return s

    def release(self, s) -> None:
        """Give state s back, read no more, for the next of its kind and shape."""
        self._free.setdefault((self._kinds[id(s)], tuple(s.shape)), []).append(s)

    def view(self, s):
        v = s[0 : s.shape[0]]
        self._views.add(id(v))
        self._state_of[id(v)] = s
        return v

    def gemm(self, x, w, act, prec=None):
        self.ops += 1
        ep = Epilogue.SILU if act == "silu" and SILU == "epilogue" else Epilogue.NONE
        y = ProjectionGEMM.step(x, w, b_col_maj=True, epilogue=ep, epilogue_modes=(Epilogue.SILU,), precision=prec or PREC)
        return self.act_op(y, act)

    def act_op(self, y, act):
        """SiLU as an operator of its own (SILU "op")."""
        if act == "silu" and SILU == "op":
            self.ops += 1
            return SiLU.step(y)
        return y

    def interior(self, y, go: G, n_rows):
        """A new grid of go from the band rows y (image rows 0.., Wp a row)."""
        out = self.state((go.rows, go.Cp))
        C, Wp = go.Cp, go.Wp
        Copy.step(y, out, src=tap((n_rows, C), P * C, [go.H, go.W * C], [Wp * C, 1]),
                  dst=tap((go.rows, C), go.pixel(0, 0) * C, [go.H, go.W * C], [Wp * C, 1]))
        self.ops += 1
        return self.view(out)

    def conv(self, x, gi: G, i, s):
        _, _, y, k, stride, group, act = s
        go = self.t[y]
        prec = PREC_DW if group > 1 else PREC
        W = self.w(f"s{i}")
        C, Wp = gi.Cp, gi.Wp
        if k == 1 and stride == 1:
            return self.gemm(x, W, act)
        h = k // 2
        if stride == 1:
            n = gi.Hc * Wp
            s0 = gi.band - h * Wp - h
            MA = n + (k - 1) * Wp + (k - 1)
            if k * C <= 512:
                ep = Epilogue.SILU if act == "silu" and SILU == "epilogue" else Epilogue.NONE
                yv = GridGEMM.step(x[s0 : s0 + MA], W, M=n, MA=MA, CA=C, k_stride=Wp * C, k_tile=k * C,
                                   b_col_maj=True, epilogue=ep, epilogue_modes=(Epilogue.SILU,), precision=prec)
                self.ops += 1
                yv = self.act_op(yv, act)
            else:  # im2col: a copy a kernel row
                cs = self.state((n, k * k * C), "cols")
                for dy in range(k):
                    Copy.step(x, cs, src=tap((gi.rows, C), (s0 + dy * Wp) * C, [gi.Hc, Wp, k * C], [Wp * C, C, 1]),
                              dst=tap((n, k * k * C), dy * k * C, [gi.Hc, Wp, k * C], [Wp * k * k * C, k * k * C, 1]))
                    self.ops += 1
                yv = self.gemm(cs[0 : cs.shape[0]], W, act, prec)
                self.release(cs)
            return self.interior(yv, go, n)
        # stride 2 (k 3): output (oy, ox)'s window row dy: pixels 2 ox - 1 .. 2 ox + 1 of row 2 oy - 1 + dy
        assert stride == 2 and k == 3
        Ho, Wo = go.H, go.W
        m = cdiv(Ho * Wo, 256) * 256
        K = cdiv(9 * C, 64) * 64
        cs = self.state((m, K), "cols")
        for dy in range(3):
            Copy.step(x, cs, src=tap((gi.rows, C), gi.pixel(dy - 1, -1) * C, [Ho, Wo, 3 * C], [2 * Wp * C, 2 * C, 1]),
                      dst=tap((m, K), dy * 3 * C, [Ho, Wo, 3 * C], [Wo * K, K, 1]))
            self.ops += 1
        yv = self.gemm(cs[0 : cs.shape[0]], W, act)
        self.release(cs)
        out = self.state((go.rows, go.Cp))
        Co = go.Cp
        Copy.step(yv, out, src=tap((m, Co), 0, [Ho, Wo * Co], [Wo * Co, 1]),
                  dst=tap((go.rows, Co), go.pixel(0, 0) * Co, [Ho, Wo * Co], [go.Wp * Co, 1]))
        self.ops += 1
        return self.view(out)

    def concat(self, xs, gs, go: G):
        out = self.state((go.rows, go.Cp))
        at = 0
        for x, g in zip(xs, gs):
            Copy.step(x, out, src=tap((g.rows, g.Cp), 0, [g.rows, g.C], [g.Cp, 1]),
                      dst=tap((go.rows, go.Cp), at, [go.rows, g.C], [go.Cp, 1]))
            at += g.C
            self.ops += 1
        g = gs[0]
        # the ones channel with the zero before it (a DMA moves whole 4 bytes)
        Copy.step(xs[0], out, src=tap((g.rows, g.Cp), g.Cp - 2, [g.rows, 2], [g.Cp, 1]),
                  dst=tap((go.rows, go.Cp), go.Cp - 2, [go.rows, 2], [go.Cp, 1]))
        self.ops += 1
        return self.view(out)

    # --- max pools (stride 1, padding k // 2, the pads -inf) ---

    @staticmethod
    def pool_grid(gi: G):
        """The grid a max pool reads: padded by 2 + 6 + 6 pixels (13 x 13 windows and their
        doubling), as a G of the same channels."""
        return PoolGrid(gi.H, gi.W, gi.C, gi.ones, gi.Cp)

    def pool_input(self, x, gi: G):
        """x's interior in a pool grid, its pads POOL_PAD (as -inf): (handle, start, end, memo)."""
        pg = self.pool_grid(gi)
        st = self.state((pg.rows, pg.Cp), "pool")
        C = gi.Cp
        Copy.step(x, st, src=tap((gi.rows, C), gi.pixel(0, 0) * C, [gi.H, gi.W * C], [gi.Wp * C, 1]),
                  dst=tap((pg.rows, C), pg.pixel(0, 0) * C, [gi.H, gi.W * C], [pg.Wp * C, 1]))
        y = st[0 : st.shape[0]] + self.w(f"neg{pg.rows}x{pg.Cp}")
        self.release(st)
        self.ops += 2
        return (y, 0, pg.rows, {})

    def mx(self, a, b):
        self.ops += 3
        return b + ReLU.step(a - b)

    def window(self, T, k, step, lo, hi):
        """The max of T (handle, start, end, memo) over q + j step, |j| <= k // 2, for q in [lo,
        hi): pieces of the doubling maxima D_m (memoized in T)."""
        h = k // 2
        pieces, rest, off = [], k, 0
        for m in (8, 4, 2, 1):
            while rest >= m:
                pieces.append((m, off))
                off += m
                rest -= m
        handle, t0, t1, memo = T
        # every D_m over all it can cover of T (shared by the windows that read the same T)
        need = {m: (t0, t1 - (m - 1) * step) for m in (2, 4, 8)}

        def get(m, a, b):
            """D_m over [a, b) as a handle slice."""
            if m == 1:
                assert t0 <= a and b <= t1, (a, b, t0, t1)
                return handle[a - t0 : b - t0]
            key = (m, step)
            if key not in memo:
                a2, b2 = need[m]
                d = self.mx(get(m // 2, a2, b2), get(m // 2, a2 + (m // 2) * step, b2 + (m // 2) * step))
                memo[key] = (d, a2, b2)
            d, a2, b2 = memo[key]
            assert a2 <= a and b <= b2, (m, a, b, a2, b2)
            return d[a - a2 : b - a2]

        out = None
        for m, o in pieces:
            p = get(m, lo - h * step + o * step, hi - h * step + o * step)
            out = p if out is None else self.mx(out, p)
        return out

    def pool(self, T, gi: G, k):
        """The k x k max pool of the pool grid T into a new grid (zero pads)."""
        pg = self.pool_grid(gi)
        Wp = pg.Wp
        # rows: the image's rows (all of each, pads included), from row 0 to row H - 1
        lo, hi = pg.pixel(0, -PP), pg.pixel(gi.H - 1, -PP) + Wp
        hz = self.window(T, k, 1, lo - (k // 2) * Wp, hi + (k // 2) * Wp)
        vt = self.window((hz, lo - (k // 2) * Wp, hi + (k // 2) * Wp, {}), k, Wp, lo, hi)
        out = self.state((gi.rows, gi.Cp))
        C = gi.Cp
        Copy.step(vt, out, src=tap((hi - lo, C), PP * C, [gi.H, gi.W * C], [Wp * C, 1]),
                  dst=tap((gi.rows, C), gi.pixel(0, 0) * C, [gi.H, gi.W * C], [gi.Wp * C, 1]))
        self.ops += 1
        return self.view(out)

    def up2(self, x, gi: G, go: G):
        """Nearest 2x: each input pixel into the four of its 2 x 2 block."""
        out = self.state((go.rows, go.Cp))
        C = gi.Cp
        for ky in range(2):
            for kx in range(2):
                Copy.step(x, out, src=tap((gi.rows, C), gi.pixel(0, 0) * C, [gi.H, gi.W, C], [gi.Wp * C, C, 1]),
                          dst=tap((go.rows, C), go.pixel(ky, kx) * C, [gi.H, gi.W, C], [2 * go.Wp * C, 2 * C, 1]))
                self.ops += 1
        return self.view(out)

    def att(self, x, gi: G, i):
        sums = GEMM.step(self.w(f"s{i}.sum"), x, b_col_maj=False, precision=Precision.HIGHEST)  # (16, Cp): row 0
        a = F.linear(sums[0:1], self.w(f"s{i}.fc"), precision=PREC)
        a = Clamp.step(a, low=0.0, high=1.0)
        self.ops += 4
        return x * a.reshape(gi.Cp)

    def body(self, x):
        v = {self.inp: x}
        pools = {}
        last = {}
        for i, s in enumerate(self.steps):
            for name in step_inputs(s):
                last[name] = i
        for i, s in enumerate(self.steps):
            kind = s[0]
            if kind == "conv":
                v[s[2]] = self.conv(v[s[1]], self.t[s[1]], i, s)
            elif kind == "add":
                v[s[3]] = v[s[1]] + v[s[2]]
                self.ops += 1
            elif kind == "concat":
                v[s[2]] = self.concat([v[a] for a in s[1]], [self.t[a] for a in s[1]], self.t[s[2]])
            elif kind == "att":
                v[s[2]] = self.att(v[s[1]], self.t[s[1]], i)
            elif kind == "pool":
                key = (s[1],)
                if key not in pools:
                    pools[key] = self.pool_input(v[s[1]], self.t[s[1]])
                v[s[2]] = self.pool(pools[key], self.t[s[1]], s[3])
            elif kind == "up2":
                v[s[2]] = self.up2(v[s[1]], self.t[s[1]], self.t[s[2]])
            for name in step_inputs(s):  # grids read no more go back
                h = v.get(name)
                if last.get(name) == i and name not in self.outputs and h is not None and id(h) in self._state_of:
                    self.release(self._state_of.pop(id(h)))
            if i % 12 == 11:
                command_boundary()
        # a graph returns handles made inside it, not views of its states
        return tuple(Copy.step(v[o]) if id(v[o]) in self._views else v[o] for o in self.outputs)


# --- the reference: the ONNX steps in numpy -------------------------------------------------


def conv_np(x, w, b, stride=1, group=1):
    Co, Cg, k, _ = w.shape
    Cin, H, W = x.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    Ho, Wo = (H + 2 * p - k) // stride + 1, (W + 2 * p - k) // stride + 1
    out = np.zeros((Co, Ho, Wo), np.float32)
    og = Co // group
    for gi in range(group):
        xs = xp[gi * Cg : (gi + 1) * Cg]
        for ky in range(k):
            for kx in range(k):
                patch = xs[:, ky : ky + stride * Ho : stride, kx : kx + stride * Wo : stride]
                out[gi * og : (gi + 1) * og] += np.einsum("oc,chw->ohw", w[gi * og : (gi + 1) * og, :, ky, kx], patch)
    return out + b[:, None, None]


def reference(steps, inp, img):
    v = {inp: img}
    for s in steps:
        kind = s[0]
        if kind == "conv":
            _, x, y, w, b, stride, group, act = s
            r = conv_np(v[x], w, b, stride, group)
            v[y] = silu(r) if act == "silu" else r
        elif kind == "add":
            v[s[3]] = v[s[1]] + v[s[2]]
        elif kind == "concat":
            v[s[2]] = np.concatenate([v[a] for a in s[1]])
        elif kind == "pool":
            _, x, y, k = s
            h = k // 2
            xp = np.pad(v[x], ((0, 0), (h, h), (h, h)), constant_values=-np.inf)
            Hh, Ww = v[x].shape[1:]
            out = np.full_like(v[x], -np.inf)
            for dy in range(k):
                for dx in range(k):
                    out = np.maximum(out, xp[:, dy : dy + Hh, dx : dx + Ww])
            v[y] = out
        elif kind == "up2":
            v[s[2]] = v[s[1]].repeat(2, axis=1).repeat(2, axis=2)
        elif kind == "att":
            _, x, y, w, b, alpha, beta = s
            m = v[x].mean(axis=(1, 2))
            a = np.clip(alpha * (w @ m + b) + beta, 0, 1)
            v[y] = v[x] * a[:, None, None]
    return v


def to_grid(img, g: G):
    """The grid of img (C, H, W) with the ones channel."""
    out = np.zeros((g.rows, g.Cp), np.float32)
    for y in range(g.H):
        r = g.pixel(y, 0)
        out[r : r + g.W, : g.C] = img[:, y].T
        out[r : r + g.W, g.Cp - 1] = g.ones
        if two_ones(g):
            out[r : r + g.W, g.Cp - 2] = g.ones
    return out


def from_grid(a, g: G):
    a = np.asarray(a, np.float32).reshape(-1, g.Cp)
    out = np.zeros((g.C, g.H, g.W), np.float32)
    for y in range(g.H):
        r = g.pixel(y, 0)
        out[:, y] = a[r : r + g.W, : g.C].T
    return out


def picture(path, W, H, mean, std):
    """A picture stretched to W x H, normalized, (3, H, W) RGB; or a .npy of that."""
    if str(path).endswith(".npy"):
        return np.load(path).astype(np.float32)
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR), np.float32)
    return ((img - np.array(mean, np.float32)) / np.array(std, np.float32)).transpose(2, 0, 1).copy()


def main(argv=None) -> int:
    global PREC, PREC_DW, SILU
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", default="640x640")
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--calib", type=Path, help="a picture to sort the depthwise channels by (default: random)")
    ap.add_argument("--check-image", type=Path, help="the picture --check runs (default: --calib's)")
    ap.add_argument("--mean", type=float, nargs=3, default=[123.675, 116.28, 103.53], help="the input's normalization (RGB)")
    ap.add_argument("--std", type=float, nargs=3, default=[58.395, 57.12, 57.375])
    ap.add_argument("--precision", choices=["MEDIUM", "HIGH"], default="MEDIUM",
                    help="the convolutions' GEMMs (but the depthwise ones'): MEDIUM (bfp16 activations) or HIGH (bf16)")
    ap.add_argument("--silu", choices=["epilogue", "op"], default=SILU, help="where SiLU runs (see SILU)")
    ap.add_argument("--dw-precision", choices=["MEDIUM", "HIGH"], default="MEDIUM", help="the depthwise convolutions' GEMMs")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    PREC, PREC_DW, SILU = getattr(Precision, a.precision), getattr(Precision, a.dw_precision), a.silu
    bind_device(a.device, a.cols)
    W, H = (int(v) for v in a.size.split("x"))
    nodes, init, shapes, inp = load(a.onnx)
    steps = plan(nodes, init, a.outputs, inp)
    t0 = time.time()
    calib = picture(a.calib, W, H, a.mean, a.std) if a.calib else np.random.default_rng(1).standard_normal((3, H, W)).astype(np.float32)
    steps = sort_depthwise(steps, channel_rms(steps, inp, calib))
    print(f"sorted the depthwise channels in {time.time() - t0:.0f} s", flush=True)
    g = CNN(steps, inp, H, W, 3, a.outputs, shapes)
    gi = g.t[inp]
    cg = g.compile(x=(gi.rows, gi.Cp))
    meta = dict(width=W, height=H, P=P, input=dict(rows=gi.rows, cp=gi.Cp, wp=gi.Wp),
                outputs=[dict(name=o, C=t.C, Cp=t.Cp, H=t.H, W=t.W, wp=t.Wp, rows=t.rows, band=t.band) for o, t in g.shapes.items()])
    m = export_graph(cg, a.out, "cnn", meta)
    print(f"compiled in {time.time() - t0:.0f} s, {len(steps)} steps, {g.ops} operators; {m['graph']['plan']}", flush=True)
    (a.out / "model.json").write_text(json.dumps(dict(kind="onnx-cnn", **meta), indent=1) + "\n")
    if a.check:
        img = picture(a.check_image, W, H, a.mean, a.std) if a.check_image else calib
        x = bf(to_grid(img, gi))
        write_check(g, m, a.out, dict(x=x))
        ref = reference(steps, inp, np.asarray(x, np.float32)[:, :3].reshape(-1)[:0].reshape(0) if False else from_grid(x, gi))
        outs = g.reference(x)
        outs = outs if isinstance(outs, (tuple, list)) else [outs]
        for o, y in zip(a.outputs, outs):
            got, want = from_grid(y, g.shapes[o]), ref[o]
            cos = (got * want).sum() / math.sqrt((got * got).sum() * (want * want).sum())
            print(f"  {o}: IRON's reference against the ONNX steps: cos {cos:.5f}, max err {np.abs(got - want).max():.4f} of {np.abs(want).max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
