"""A depthwise 1-D convolution of K dilated taps and a bias as one operator (kernels/dwconv1d.cc).

``y = DWConv1d(x, w)``: x (rows + (K - 1) d, C) the input already padded (time-major, channels
last), w (K + 1, C) the taps and then the bias; y (rows, C) with y[t] = bias + sum_j w[j] x[t + j d].
A "same" convolution pads x by (K - 1) d / 2 rows on each side, a causal one by (K - 1) d before.
Each of ``columns`` columns' core takes a block of BLOCK channels and a part of the rows, an
object TILE output rows and the (K - 1) d rows of halo after them (a sliding window of rows took a
buffer descriptor a row: 33 for 5 taps of dilation 8, past a channel's 16).

IRON's DepthwiseConv1d is 5 causal taps without a bias or dilation; Supertonic's ConvNeXt blocks
(iron_supertonic.py) take 5 taps of dilation 1, 2, 4, 8 and 7 of 1, 2, 4.

    python tools/amd/iron_dwconv1d.py check OUT [--rows 128] [--taps 5] [--dilation 8]
    go run ./tools/amd/irongraph OUT
"""

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

COLUMNS = 8
MAX_K = 7
BLOCK = 128  # a core's channels
TILE = 16  # a core's output rows an object


def reference(x, w, rows: int, taps: int, dilation: int) -> np.ndarray:
    """DWConv1d in numpy, a float32 sum rounded once to bf16."""
    x = np.asarray(x, np.float32)
    w = np.asarray(w, np.float32)
    y = np.zeros((rows, x.shape[1]), np.float32) + w[taps]
    for j in range(taps):
        y += w[j] * x[j * dilation : j * dilation + rows]
    return y.astype(bfloat16)


@functools.cache
def operator(n: int = COLUMNS):
    """The DWConv1d operator class over ``n`` columns."""
    from aie.iron import In, ObjectFifo, Out, TaskGroup, Worker
    from aie.iron.controlflow import range_
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    class DWConv1d(Operator):
        """``y = DWConv1d(x, w)``: the module's convolution. The channels are cut in blocks of
        BLOCK, the rows in tiles of TILE: each column takes one block and a part of the tiles
        (columns / blocks parts), an object a tile's rows and halo of its block."""

        rows: int = param(array=True)
        channels: int = param(array=True)
        taps: int = param(array=True)
        dilation: int = param(array=True)
        rows_in: int = param(array=True)
        kp: int = param(derive=lambda op: op.taps + 1, repr=False)
        columns: int = param(default=n, init=False, array=True)
        xo: int = param(derive=lambda op: (TILE + (op.taps - 1) * op.dilation) * BLOCK, repr=False)
        wo: int = param(derive=lambda op: (op.taps + 1) * BLOCK, repr=False)
        yo: int = param(default=TILE * BLOCK, init=False, repr=False)

        x = In(rows_in, channels, obj_shape=(xo,), per=(columns,))
        w = In(kp, channels, obj_shape=(wo,), per=(columns,))
        y = Out(rows, channels, obj_shape=(yo,), per=(columns,))

        def parts(self) -> int:
            return n // (self.channels // BLOCK)

        def validate(self):
            C = self.channels
            if C % BLOCK or n % (C // BLOCK) or not 1 <= self.taps <= MAX_K or self.rows % (TILE * (n // (C // BLOCK))):
                raise ValueError(f"DWConv1d: {C} channels (blocks of {BLOCK} over {n} columns), {self.taps} "
                                 f"taps (1 to {MAX_K}), {self.rows} rows (tiles of {TILE} in each part)")
            if self.rows_in != self.rows + (self.taps - 1) * self.dilation:
                raise ValueError(f"DWConv1d: {self.rows_in} input rows for {self.rows} out of {self.taps} "
                                 f"taps of dilation {self.dilation}")

        def build(self) -> Built:
            K, d = self.taps, self.dilation
            tiles = self.rows // TILE // self.parts()
            here = Path(__file__).resolve().parent / "kernels"
            x_t = np.ndarray[(self.xo,), np.dtype[bfloat16]]
            w_t = np.ndarray[(self.wo,), np.dtype[bfloat16]]
            y_t = np.ndarray[(self.yo,), np.dtype[bfloat16]]
            kern = ExternalFunction(
                "dwconv1d_bias", arg_types=[w_t, x_t, y_t], source_file=str(here / "dwconv1d.cc"),
                include_dirs=[str(Path(config.aie_kernels_dir())), config.cxx_header_path()],
                compile_flags=[f"-DDW_C={BLOCK}", f"-DDW_K={K}", f"-DDW_D={d}", f"-DDW_R={TILE}"],
                object_file_name=f"dwconv1d_{BLOCK}_{K}_{d}_{TILE}.o")
            x_f = [ObjectFifo(x_t, depth=2) for _ in range(n)]
            w_f = [ObjectFifo(w_t, depth=1) for _ in range(n)]
            y_f = [ObjectFifo(y_t, depth=2) for _ in range(n)]

            def core(xf, wf, yf, k):
                wt = wf.acquire(1)
                for _ in range_(tiles):
                    xo, yo = xf.acquire(1), yf.acquire(1)
                    k(wt, xo, yo)
                    xf.release(1)
                    yf.release(1)
                wf.release(1)

            workers = [Worker(core, [x_f[c].cons(), w_f[c].cons(), y_f[c].prod(), kern]) for c in range(n)]
            self.x.bind([f.prod() for f in x_f])
            self.w.bind([f.prod() for f in w_f])
            self.y.bind([f.cons() for f in y_f])
            return Built(workers)

        def sequence(self, rt):
            from aie.helpers.taplib import TensorAccessPattern as TAP

            C, K, R = self.channels, self.taps, self.rows
            halo = (K - 1) * self.dilation
            nb, parts = C // BLOCK, self.parts()
            tiles = R // TILE // parts
            tg = TaskGroup()
            for c in range(n):
                b, p = c % nb, c // nb
                r0 = p * tiles * TILE
                self.w.fill(TAP((K + 1, C), b * BLOCK, [K + 1, BLOCK], [C, 1]), lane=c, group=tg)
                self.x.fill(TAP((self.rows_in, C), r0 * C + b * BLOCK, [tiles, TILE + halo, BLOCK], [TILE * C, C, 1]),
                            lane=c, group=tg)
            for c in range(n):
                b, p = c % nb, c // nb
                r0 = p * tiles * TILE
                self.y.drain(TAP((R, C), r0 * C + b * BLOCK, [tiles, TILE, BLOCK], [TILE * C, C, 1]), lane=c,
                             group=tg, wait=True)
            tg.finish()

        def ops(self) -> int:
            return 2 * (self.taps + 1) * self.rows * self.channels

        def reference(self, x, w):
            return reference(x, w, self.rows, self.taps, self.dilation)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(2**-7, 1e-3, note="a float32 sum rounded once to bf16")

    return DWConv1d


def dwconv(x, w, rows: int, taps: int, dilation: int):
    """DWConv1d.step of x (rows + (taps - 1) dilation, C), w (taps + 1, C)."""
    return operator().step(x, w, rows=rows, taps=taps, dilation=dilation, rows_in=x.shape[0])


def _check(a) -> int:
    import iron

    from iron_graph import bind_device, export_graph, write_check

    bind_device(a.device, COLUMNS)
    rows_in = a.rows + (a.taps - 1) * a.dilation

    class G(iron.Graph):
        def __init__(self, w):
            self.wh = iron.param(w)

        def body(self, x):
            return dwconv(x, self.wh, a.rows, a.taps, a.dilation)

    rng = np.random.default_rng(0)
    w = rng.standard_normal((a.taps + 1, a.channels)).astype(bfloat16)
    g = G(w)
    cg = g.compile(x=(rows_in, a.channels))
    m = export_graph(cg, a.out, "dwconv1d", dict(rows=a.rows, taps=a.taps, dilation=a.dilation))
    x = rng.standard_normal((rows_in, a.channels)).astype(bfloat16)
    write_check(g, m, a.out, dict(x=x))
    print(m["graph"]["plan"].splitlines()[0])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of one DWConv1d and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--rows", type=int, default=128)
    c.add_argument("--channels", type=int, default=512)
    c.add_argument("--taps", type=int, default=5)
    c.add_argument("--dilation", type=int, default=8)
    c.add_argument("--device", default="npu2")
    a = ap.parse_args(argv)
    return _check(a)


if __name__ == "__main__":
    sys.exit(main())
