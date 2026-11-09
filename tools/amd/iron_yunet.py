#!/usr/bin/env python3
"""YuNet (OpenCV Zoo's face_detection_yunet_2023mar.onnx, MIT) on IRON: the graph
cmd/hwnpu-image face-detect runs on AMD's NPU.

YuNet is a tiny face detector (53 thousand parameters): a 3 x 3 convolution with stride 2,
then blocks of a 1 x 1 convolution and a 3 x 3 depthwise one (and ReLU), 2 x 2 max pools
down to 1/32 of the picture, a top-down neck (nearest 2x upsampling and an add) and, at
strides 8, 16 and 32, heads of a 1 x 1 and a depthwise 3 x 3 convolution: the class's and
the objectness' logits, the box (4) and five landmarks (10). The host (internal/vision/face)
takes the sigmoids, decodes the boxes and landmarks and suppresses the overlaps.

A 1 x 1 convolution followed by a depthwise 3 x 3 one is one 3 x 3 convolution here: weight
dw[o, ky, kx] w1[o, i] and the 1 x 1's bias b1 times the depthwise taps that fall inside the
picture (zero padding: the 1 x 1's output is zero outside, its bias too). Where the input has
fewer than 64 channels, channel 63 is one inside the picture and zero in the pads, and b1
rides it as one more input channel; else the graph adds a per-pixel bias map. Convolutions
run as iron_da3's (ConvOps: channel-last zero-padded grids, GEMMs over im2col rows), every
channel count padded to 64. A max pool is three maxima of the four strided views (max(a, b) =
b + ReLU(a - b)).

Input ``input`` (H W, 4): the picture's B, G, R over 255 (OpenCV's blobFromImage gives 0..255:
the first convolution's weight is times 255) and a one (the biases'); H and W multiples of 32 (the picture at the top left, zeros past it).
Outputs (cells, 64) at strides 8, 16 and 32 (the cells H/s W/s, row major):
per cell cls, obj (logits), the box's dx, dy, log w, log h and the landmarks' x, y (in strides
from the cell).

  python tools/amd/iron_yunet.py ~/models/face_detection_yunet ~/models/yunet.amd-npu2   # --sizes 320x256 640x480

See docs/amd-npu-models.md.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_da3  # noqa: E402
from iron_anima import bf, tap  # noqa: E402
from iron_da3 import ConvOps, Grid  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

import iron.functional as F  # noqa: E402
from aie.utils.compile.jit.params import per_call  # noqa: E402
from iron.operators.copy import Copy  # noqa: E402
from iron.operators.flm import ProjectionGEMM  # noqa: E402
from iron.operators.flm.gemm.design import Epilogue  # noqa: E402
from iron.operators.relu import ReLU  # noqa: E402

C = 64  # every tensor's channels, padded
ONES = 63  # the channel that is one inside the picture (where a tensor has fewer)
HEAD = ["cls", "obj", "bbox", "kps"]  # a head's outputs' order in its 16 channels
STRIDES = (8, 16, 32)


def load_onnx(path: Path):
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path))
    init = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in m.graph.initializer}
    nodes = []
    for n in m.graph.node:
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        nodes.append(dict(op=n.op_type, inputs=list(n.input), outputs=list(n.output), attrs=a))
    return nodes, init


def plan(nodes, init) -> list:
    """The ONNX graph as this graph's steps:

    ("conv0", in, out, w, b)              the 3 x 3 stride 2 convolution and ReLU
    ("block", in, out, w1, b1, dw, b2)    1 x 1, depthwise 3 x 3, ReLU
    ("pool", in, out)                     the 2 x 2 max pool
    ("upadd", small, lateral, out)        nearest 2x of small, plus lateral
    ("head", in, stride, {name: (w1, b1, dw, b2)})
    """
    by_out = {n["outputs"][0]: n for n in nodes}
    users = {}
    for n in nodes:
        for i in n["inputs"]:
            users.setdefault(i, []).append(n)
    steps, done = [], set()
    # a head's outputs: Conv -> Transpose -> Reshape [-> Sigmoid] -> "cls_8" ...
    head_of = {}
    for n in nodes:
        if n["op"] == "Transpose":
            r = users[n["outputs"][0]][0]
            name = r["outputs"][0]
            if r["outputs"][0] in users:
                name = users[r["outputs"][0]][0]["outputs"][0]
            head_of[n["inputs"][0]] = name  # the depthwise conv's output
    for n in nodes:
        if id(n) in done:
            continue
        op = n["op"]
        if op == "Conv" and n["outputs"][0] in head_of:
            continue  # taken with its 1 x 1
        if op == "Conv" and n["attrs"]["strides"] == [2, 2]:
            relu = users[n["outputs"][0]][0]
            steps.append(("conv0", n["inputs"][0], relu["outputs"][0], init[n["inputs"][1]], init[n["inputs"][2]]))
            done.add(id(relu))
        elif op == "Conv" and n["attrs"]["kernel_shape"] == [1, 1]:
            dw = users[n["outputs"][0]][0]
            assert dw["op"] == "Conv" and dw["attrs"]["kernel_shape"] == [3, 3] and dw["attrs"]["group"] == init[dw["inputs"][1]].shape[0]
            w1, b1 = init[n["inputs"][1]][:, :, 0, 0], init[n["inputs"][2]]
            d, b2 = init[dw["inputs"][1]][:, 0], init[dw["inputs"][2]]
            done.add(id(dw))
            if dw["outputs"][0] in head_of:
                name = head_of[dw["outputs"][0]]  # cls_8
                kind, stride = name.split("_")
                steps.append(("headpart", n["inputs"][0], int(stride), kind, (w1, b1, d, b2)))
            else:
                relu = users[dw["outputs"][0]][0]
                assert relu["op"] == "Relu"
                done.add(id(relu))
                steps.append(("block", n["inputs"][0], relu["outputs"][0], w1, b1, d, b2))
        elif op == "MaxPool":
            assert n["attrs"]["kernel_shape"] == [2, 2] and n["attrs"]["strides"] == [2, 2]
            steps.append(("pool", n["inputs"][0], n["outputs"][0]))
        elif op == "Resize":
            assert n["attrs"]["mode"] == b"nearest"
            add = users[n["outputs"][0]][0]
            assert add["op"] == "Add"
            lateral = [i for i in add["inputs"] if i != n["outputs"][0]][0]
            done.add(id(add))
            steps.append(("upadd", n["inputs"][0], lateral, add["outputs"][0]))
        elif op in ("Transpose", "Reshape", "Sigmoid"):
            continue
        else:
            raise SystemExit(f"YuNet: no step for {op} ({n['outputs']})")
    # the heads' parts by stride, in HEAD's order
    out, heads = [], {}
    for s in steps:
        if s[0] == "headpart":
            h = heads.setdefault(s[2], dict(input=s[1], parts={}))
            assert h["input"] == s[1]
            h["parts"][s[3]] = s[4]
        else:
            out.append(s)
    for stride in STRIDES:
        out.append(("head", heads[stride]["input"], stride, heads[stride]["parts"]))
    by_out.clear()
    return out


class ReluGEMM(ProjectionGEMM):
    """ProjectionGEMM with its epilogue's clamp from 0: ReLU in the GEMM (the kernel clamps
    every mode's output to runtime bounds, the identity by default), one operator less."""

    clamp_min = per_call(derive=lambda op: 0)  # +0.0's bits

    def reference(self, A, B):
        return np.maximum(super().reference(A, B), 0).astype(A.dtype)


def gemm(x, w, relu: bool):
    if relu:
        return ReluGEMM.step(x, w, b_col_maj=True, epilogue_modes=(Epilogue.NONE,), precision=iron_da3.PREC.head)
    return F.linear(x, w, precision=iron_da3.PREC.head)


@dataclasses.dataclass(frozen=True)
class T:
    """A tensor of the graph: H x W pixels of ``real`` channels in a grid ``width`` wide (64, or
    128 for 64 real), its last channel ``ones`` inside the picture (0 in the pads)."""

    H: int
    W: int
    real: int
    ones: float = 1.0

    @property
    def width(self) -> int:
        return 64 if self.real < 64 else 128

    @property
    def grid(self) -> Grid:
        return Grid(self.H, self.W, self.width)


class YuNet(ConvOps):
    """The graph. Every bias rides a ones channel (a tensor's last, one inside the picture
    and zero in the pads; the input's fourth), so each convolution is one GEMM, with ReLU in
    its epilogue (ReluGEMM): an im2col copy, the GEMM and the copy into the next grid."""

    def __init__(self, steps: list, width: int, height: int):
        self.init_ops()
        self.width, self.height = width, height
        self.t = {}  # tensor name -> T
        H, W = height, width
        for i, s in enumerate(steps):
            kind = s[0]
            if kind == "conv0":
                _, src, dst, w, b = s
                cout = w.shape[0]
                to = T(H // 2, W // 2, cout)
                # columns dy 12 + dx 4 + c: the pixels (over 255); 62 and 63 the centre pixel's R
                # and ones, apart from the pixels: NPU2's GEMM takes A in bfp16, 8 columns sharing
                # an exponent, and a one beside 0..255 rounded away
                w0 = np.zeros((to.width, C), np.float32)
                for dy in range(3):
                    for dx in range(3):
                        w0[:cout, dy * 12 + dx * 4 : dy * 12 + dx * 4 + 3] = w[:, :, dy, dx] * 255
                w0[:cout, 63] = b
                w0[to.width - 1, 63] = 1
                self._param("conv0.w", w0)
                self.t[dst] = to
            elif kind == "block":
                _, src, dst, w1, b1, d, b2 = s
                ti = self.t[src]
                to = T(ti.H, ti.W, w1.shape[0])
                self.fused(f"s{i}", w1, b1, d, b2, ti, to.width, ones_out=True)
                self.t[dst] = to
            elif kind == "pool":
                ti = self.t[s[1]]
                self.t[s[2]] = T(ti.H // 2, ti.W // 2, ti.real, ti.ones)
            elif kind == "upadd":
                a, lat = self.t[s[1]], self.t[s[2]]
                assert a.width == lat.width and (2 * a.H, 2 * a.W) == (lat.H, lat.W)
                self.t[s[3]] = T(lat.H, lat.W, lat.real, a.ones + lat.ones)
            elif kind == "head":
                _, src, stride, parts = s
                w1 = np.concatenate([parts[k][0] for k in HEAD])
                b1 = np.concatenate([parts[k][1] for k in HEAD])
                d = np.concatenate([parts[k][2] for k in HEAD])
                b2 = np.concatenate([parts[k][3] for k in HEAD])
                self.fused(f"s{i}", w1, b1, d, b2, self.t[src], C, ones_out=False)
        # the steps without their weights (a graph holds no arrays but its params)
        self.steps = [tuple(x for x in s if isinstance(x, (str, int))) for s in steps]

    def fused(self, name: str, w1, b1, d, b2, ti: T, wout: int, ones_out: bool) -> None:
        """The 1 x 1 (w1 (Co, cin), b1) then the depthwise 3 x 3 (d (Co, 3, 3), b2) as one 3 x 3
        convolution (Co, cin, 3, 3) = d w1, b1 times the taps on the ones channel (zero in the
        pads, as the 1 x 1's output is) and b2 on its centre tap."""
        Co, cin = w1.shape
        win, o = ti.width, ti.width - 1
        wt = np.zeros((wout, win, 3, 3), np.float32)
        wt[:Co, :cin] = d[:, None] * w1[:, :, None, None]
        wt[:Co, o] = d * b1[:, None, None] / ti.ones
        wt[:Co, o, 1, 1] += b2 / ti.ones
        if ones_out:
            assert Co < wout
            wt[wout - 1, o, 1, 1] = 1 / ti.ones
        self._param(f"{name}.w", wt.transpose(0, 2, 3, 1).reshape(wout, -1))

    def conv(self, x, ti: T, name: str, wout: int, relu: bool):
        """The 3 x 3 convolution (one GEMM a chunk of rows) of grid x into a new grid."""
        g = ti.grid
        go = Grid(g.H, g.W, wout)
        out = self.state((go.rows, wout))
        Wp, Ci = g.Wp, g.C
        R = g.chunk(Ci)
        for r0 in range(0, g.H, R):
            n = R * Wp
            start = (g.band + r0 * Wp - Wp - 1) * Ci
            cols = Copy.step(x, src=tap((g.rows, Ci), start, [R, Wp, 3, 3 * Ci], [Wp * Ci, Ci, Wp * Ci, 1])).reshape(n, 9 * Ci)
            y = gemm(cols, self.w(f"{name}.w"), relu)
            rows = min(R, g.H - r0)
            Copy.step(y, out, src=tap((n, wout), wout, [rows, g.W * wout], [Wp * wout, 1]),
                      dst=tap((go.rows, wout), go.pixel(r0, 0) * wout, [rows, g.W * wout], [Wp * wout, 1]))
        return self.view(out)

    def maxpool(self, x, ti: T):
        """The 2 x 2 max pool: the four strided views as two stacks, [v00; v10] and [v01;
        v11], their maximum (max(a, b) = b + ReLU(a - b)), then its halves'."""
        g, to = ti.grid, T(ti.H // 2, ti.W // 2, ti.real, ti.ones)
        Cw = g.C
        Ho, Wo = to.H, to.W
        n = Ho * Wo
        m = -(-n // 256) * 256
        stacks = []
        for dx in range(2):
            v = self.state((2 * m, Cw))
            Copy.step(x, v, src=tap((g.rows, Cw), g.pixel(0, dx) * Cw, [2, Ho, Wo, Cw], [g.Wp * Cw, 2 * g.Wp * Cw, 2 * Cw, 1]),
                      dst=tap((2 * m, Cw), 0, [2, n * Cw], [m * Cw, 1]))
            stacks.append(self.view(v))
        a, b = stacks
        y = b + ReLU.step(a - b)  # (2 m): the maximum over dx, of row dy = 0 then dy = 1
        top, bot = y[0:m], y[m : 2 * m]
        y = bot + ReLU.step(top - bot)
        return self.to_grid(y, to.grid)

    def upadd(self, x, ta: T, lat, tl: T):
        g, gl = ta.grid, tl.grid
        Cw = g.C
        out = self.state((gl.rows, Cw))
        for ky in range(2):
            for kx in range(2):
                Copy.step(x, out, src=tap((g.rows, Cw), g.pixel(0, 0) * Cw, [g.H, g.W, Cw], [g.Wp * Cw, Cw, 1]),
                          dst=tap((gl.rows, Cw), gl.pixel(ky, kx) * Cw, [g.H, g.W, Cw], [2 * gl.Wp * Cw, 2 * Cw, 1]))
        return self.view(out) + lat

    def body(self, input):
        H, W = self.height, self.width
        v = {}
        outs = {}
        for i, s in enumerate(self.steps):
            kind = s[0]
            if ARGS is not None and ARGS.upto is not None and i == ARGS.upto:
                g = self.t[s[1]].grid
                x = v[s[1]]
                return (Copy.step(x, src=tap((g.rows, g.C), 0, [g.rows * g.C], [1])).reshape(g.rows, g.C),)
            if kind == "conv0":
                gi = Grid(H, W, 4)
                xg = self.to_grid(input, gi)
                to = self.t[s[2]]
                Ho, Wo = to.H, to.W
                n = Ho * Wo
                m = -(-n // 256) * 256
                cs = self.state((m, C))
                for dy in range(3):  # output pixel (oy, ox)'s window row dy: pixels 2 ox - 1 .. 2 ox + 1 of row 2 oy - 1 + dy
                    Copy.step(xg, cs, src=tap((gi.rows, 4), (gi.band + (dy - 1) * gi.Wp) * 4, [Ho, Wo, 12], [2 * gi.Wp * 4, 8, 1]),
                              dst=tap((m, C), dy * 12, [Ho, Wo, 12], [Wo * C, C, 1]))
                Copy.step(xg, cs, src=tap((gi.rows, 4), gi.pixel(0, 0) * 4 + 2, [Ho, Wo, 2], [2 * gi.Wp * 4, 8, 1]),
                          dst=tap((m, C), 62, [Ho, Wo, 2], [Wo * C, C, 1]))
                y = gemm(self.view(cs), self.w("conv0.w"), relu=True)
                v[s[2]] = self.to_grid(y, to.grid)
            elif kind == "block":
                to = self.t[s[2]]
                v[s[2]] = self.conv(v[s[1]], self.t[s[1]], f"s{i}", to.width, relu=True)
            elif kind == "pool":
                v[s[2]] = self.maxpool(v[s[1]], self.t[s[1]])
            elif kind == "upadd":
                v[s[3]] = self.upadd(v[s[1]], self.t[s[1]], v[s[2]], self.t[s[2]])
            elif kind == "head":
                ti = self.t[s[1]]
                y = self.conv(v[s[1]], ti, f"s{i}", C, relu=False)
                g = Grid(ti.H, ti.W, C)
                outs[s[2]] = Copy.step(y, src=tap((g.rows, C), g.pixel(0, 0) * C, [g.H, g.W * C], [g.Wp * C, 1])).reshape(g.H * g.W, C)
        return tuple(outs[st] for st in STRIDES)


# --- the reference: the ONNX model in numpy (float32) -----------------------------------------


def conv_np(x, w, b, stride=1, group=1):
    """x (C, H, W); w (Co, Ci/group, k, k); padding k // 2."""
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


def onnx_ref(nodes, init, img: np.ndarray) -> dict:
    """The ONNX graph on img (3, H, W), its outputs by name (cells, channels)."""
    v = {nodes[0]["inputs"][0]: img}
    for n in nodes:
        a, op, ins = n["attrs"], n["op"], n["inputs"]
        x = v.get(ins[0])
        if op == "Conv":
            y = conv_np(x, init[ins[1]], init[ins[2]], a["strides"][0], a.get("group", 1))
        elif op == "Relu":
            y = np.maximum(x, 0)
        elif op == "MaxPool":
            c, h, w = x.shape
            y = x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))
        elif op == "Resize":
            y = x.repeat(2, axis=1).repeat(2, axis=2)
        elif op == "Add":
            y = v[ins[0]] + v[ins[1]]
        elif op == "Transpose":
            y = x.transpose(1, 2, 0)
        elif op == "Reshape":
            y = x.reshape(-1, x.shape[-1])
        elif op == "Sigmoid":
            y = 1 / (1 + np.exp(-x))
        else:
            raise SystemExit(op)
        v[n["outputs"][0]] = y
    return v


def graph_input(img: np.ndarray) -> np.ndarray:
    """The graph's input of img (3, H, W) BGR in 0..255: (H W, 4), over 255, then a one."""
    _, H, W = img.shape
    x = np.ones((H * W, 4), np.float32)  # the fourth channel: the biases' ones
    x[:, :3] = img.transpose(1, 2, 0).reshape(-1, 3) / 255
    return x


def compare(graph_outs, ref: dict, width: int, height: int) -> None:
    off = 0
    for st, y in zip(STRIDES, graph_outs):
        y = np.asarray(y, np.float32)
        n = (width // st) * (height // st)
        for k, (a, b) in dict(cls=(0, 1), obj=(1, 2), bbox=(2, 6), kps=(6, 16)).items():
            r = ref[f"{k}_{st}"]
            g = y[:n, a:b]
            if k in ("cls", "obj"):
                g = 1 / (1 + np.exp(-g))
            err = np.abs(g - r).max()
            cos = (g * r).sum() / np.sqrt((g * g).sum() * (r * r).sum() + 1e-30)
            print(f"  {k}_{st}: max err {err:.4f}, cos {cos:.5f}")
        off += n


SIZES = ["320x256", "640x480"]
ARGS = None


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path, help="face_detection_yunet_2023mar.onnx, or the directory holding it (opencv/face_detection_yunet)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--sizes", nargs="*", default=SIZES, help="WxH (multiples of 32), each a directory of the export")
    ap.add_argument("--check", action="store_true", help="write check/ and compare the reference with the ONNX model")
    ap.add_argument("--upto", type=int, help=argparse.SUPPRESS)  # time the steps before this one
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    bind_device(a.device, a.cols)
    if a.onnx.is_dir():
        a.onnx = a.onnx / "face_detection_yunet_2023mar.onnx"
    nodes, init = load_onnx(a.onnx)
    steps = plan(nodes, init)
    sizes = []
    for wh in a.sizes:
        width, height = (int(v) for v in wh.split("x"))
        if width % 32 or height % 32:
            raise SystemExit(f"{wh}: not multiples of 32")
        t0 = time.time()
        g = YuNet(steps, width, height)
        cg = g.compile(input=(width * height, 4))
        out = a.out / wh
        meta = dict(width=width, height=height, strides=list(STRIDES), channels=C, head=dict(cls=0, obj=1, bbox=2, kps=6))
        m = export_graph(cg, out, "yunet", meta)
        print(f"{wh}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if a.check:
            img = np.random.default_rng(0).uniform(0, 255, (3, height, width)).astype(np.float32)
            x = bf(graph_input(img))
            write_check(g, m, out, dict(input=x))
            ref = onnx_ref(nodes, init, np.asarray(x, np.float32)[:, :3].reshape(height, width, 3).transpose(2, 0, 1) * 255)
            compare(g.reference(x), ref, width, height)
        (out / "model.json").write_text(json.dumps(dict(kind="face-detect-size", **meta), indent=1) + "\n")
        sizes.append([width, height])
    m = dict(name="YuNet", kind="face-detect", model="yunet", source=a.onnx.name, sizes=sizes,
             strides=list(STRIDES), channels=C, head=dict(cls=0, obj=1, bbox=2, kps=6))
    (a.out / "model.json").write_text(json.dumps(m, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
