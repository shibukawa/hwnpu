"""A k x k depthwise convolution (3 or 5) as one IRON operator (kernels/dwconv3.cc).

Sana's GLUMBConv runs a 3 x 3 depthwise convolution on each of its two branches. iron_sana built
it from IRON's elementwise operators: the branch copied into a zero-padded grid, then each tap
a row product of a shifted slice of the grid and a sum (9 + 9 operators), the bias, and a copy
dropping the grid's pad column: with SiLU and the gate's product, 14.6 ms of a 42 ms Sana 600M
block at 1024 x 1024 (design.md 16.200).

``DWConv3`` reads the same grid (iron_sana's DwGrid: rows of W + p pixels, p = k // 2, the pad
pixels shared by a row's right and the next row's left, p zero rows above and below) and writes
the output in token order (H W, channels): no pad column, no copy after. Each column's core takes
a band of ``C`` channels at a time (``S`` bands), slides a window of k padded rows (W + k - 1
pixels: the grid rows r (W + p) .. r (W + p) + W + k - 2) down the image and writes a row a call.
The channels must be whole bands over the columns (iron_sana pads the DiT's, with zero weights).
The VAE's EfficientViT runs it 5 x 5 on its q, k and v and 3 x 3 in its GLUMBConv.
"""

from pathlib import Path

import numpy as np
from aie.iron import In, Out
from aie.utils.compile.jit.params import param, tunable, widths
from ml_dtypes import bfloat16

from iron.common import Built, Operator

KERNELS = Path(__file__).resolve().parent / "kernels"
QUEUE = 4  # transfers a shim channel takes queued at most (more hung the device, npu-dma-queue-wedge)


def grid_rows(h: int, w: int, k: int = 3) -> int:
    """iron_sana.DwGrid(h, w, k).rows: (H + 2p) rows of W + p pixels and 2p more."""
    p = k // 2
    return (h + 2 * p) * (w + p) + 2 * p


# A core's data memory left for the windows and the output rows (of 64 KiB: the taps, the stack)
CORE_BYTES = 54 << 10


def depths(w: int, c: int, k: int) -> tuple[int, int] | None:
    """The window's and the output's fifo depths for a band of c channels: k + 1 and 2 where they
    fit a core, else k (the next row lands while the window waits) or one output; None if not."""
    row, out = (w + k - 1) * c * 2, w * c * 2
    for di, do in ((k + 1, 2), (k, 2), (k, 1)):
        if di * row + do * out + (k * k + 1) * c * 2 <= CORE_BYTES:
            return di, do
    return None


class DWConv3(Operator):
    """``y`` (H W, Ch): each channel of the padded ``grid`` convolved with its K x K taps plus its
    bias (``taps`` (K K + 1, Ch): row dy K + dx a tap, the last row the bias)."""

    H: int = param(array=True)
    W: int = param(array=True)
    Ch: int = param(array=True)
    C: int = param(default=96, array=True)
    K: int = param(default=3, array=True)
    rows: int = param(derive=lambda op: grid_rows(op.H, op.W, op.K), array=True)
    W2: int = param(derive=lambda op: op.W + op.K - 1, array=True)
    HW: int = param(derive=lambda op: op.H * op.W, array=True)
    T: int = param(derive=lambda op: op.K * op.K + 1, array=True)
    num_aie_columns: int = tunable(domain=widths)

    grid = In(rows, Ch, obj_shape=(W2, C), per=(num_aie_columns,))
    taps = In(T, Ch, obj_shape=(T, C), per=(num_aie_columns,), depth=1)
    y = Out(HW, Ch, obj_shape=(W, C), per=(num_aie_columns,))

    def validate(self) -> None:
        if self.C % 32:
            raise ValueError(f"DWConv3: bands of {self.C} channels (whole 32-lane vectors)")
        if self.K not in (3, 5):
            raise ValueError(f"DWConv3: {self.K} x {self.K} (3 or 5)")
        if depths(self.W, self.C, self.K) is None:
            raise ValueError(f"DWConv3: rows of {self.W} pixels in bands of {self.C} do not fit a core")

    def resolve(self, dev):
        cols = self.resolve_columns(dev, self.num_aie_columns, fits=lambda c: self.Ch % (c * self.C) == 0)
        return self.replace(num_aie_columns=cols)

    def compatible(self) -> None:
        if self.Ch % (self.num_aie_columns * self.C):
            raise ValueError(f"DWConv3: {self.Ch} channels are no whole bands of {self.C} over "
                             f"{self.num_aie_columns} columns")

    @property
    def S(self) -> int:
        """The bands each column's core takes in turn."""
        return self.Ch // (self.num_aie_columns * self.C)

    def kernel(self):
        from aie.iron.kernel import ExternalFunction
        from aie.utils import config

        row = np.ndarray[(self.W2 * self.C,), np.dtype[bfloat16]]
        taps = np.ndarray[(self.T * self.C,), np.dtype[bfloat16]]
        out = np.ndarray[(self.W * self.C,), np.dtype[bfloat16]]
        return ExternalFunction(
            f"dwconv{self.K}_row", source_file=str(KERNELS / "dwconv3.cc"),
            include_dirs=[str(Path(config.aie_kernels_dir())), config.cxx_header_path()],
            compile_flags=[f"-DDW_W={self.W}", f"-DDW_C={self.C}", f"-DDW_K={self.K}"],
            object_file_name=f"dwconv{self.K}_{self.W}_{self.C}.o",
            arg_types=[row] * self.K + [taps, out])

    def build(self) -> Built:
        from aie.iron import ObjectFifo, Worker
        from aie.iron.controlflow import range_

        cols, S, H, K = self.num_aie_columns, self.S, self.H, self.K
        di, do = depths(self.W, self.C, K)
        kern = self.kernel()
        of_in = [ObjectFifo(self.grid.obj_type, depth=di) for _ in range(cols)]
        of_t = [ObjectFifo(self.taps.obj_type, depth=1) for _ in range(cols)]
        of_o = [ObjectFifo(self.y.obj_type, depth=do) for _ in range(cols)]

        def core(of_in, of_t, of_o, conv):
            for _ in range_(S):
                t = of_t.acquire(1)
                for _ in range_(H):
                    win = of_in.acquire(K)
                    o = of_o.acquire(1)
                    conv(*[win[j] for j in range(K)], t, o)
                    of_in.release(1)
                    of_o.release(1)
                of_in.release(K - 1)
                of_t.release(1)

        workers = [Worker(core, [of_in[i].cons(), of_t[i].cons(), of_o[i].prod(), kern]) for i in range(cols)]
        self.grid.bind([f.prod() for f in of_in])
        self.taps.bind([f.prod() for f in of_t])
        self.y.bind([f.cons() for f in of_o])
        return Built(workers, [])

    def sequence(self, rt):
        """Each core's bands in turn: the band's padded rows and its taps in, its output rows out,
        at most QUEUE bands a group (a shim channel takes QUEUE transfers queued)."""
        from aie.helpers.taplib import TensorAccessPattern as TAP
        from aie.iron import TaskGroup

        cols, S, C, Ch, H, W, K = self.num_aie_columns, self.S, self.C, self.Ch, self.H, self.W, self.K
        Wp, per, T = W + K // 2, S * C, self.T
        for g0 in range(0, S, QUEUE):
            tg = TaskGroup()
            for k in range(cols):
                for j in range(g0, min(S, g0 + QUEUE)):
                    c0 = k * per + j * C
                    self.taps.fill(TAP((T, Ch), c0, [T, C], [Ch, 1]), lane=k, group=tg)
                    self.grid.fill(TAP((self.rows, Ch), c0, [H + K - 1, W + K - 1, C], [Wp * Ch, Ch, 1]),
                                   lane=k, group=tg)
                    self.y.drain(TAP((self.HW, Ch), c0, [H, W, C], [W * Ch, Ch, 1]), lane=k, group=tg, wait=True)
            tg.finish()

    def ops(self) -> int:
        return 2 * self.K * self.K * self.HW * self.Ch

    def reference(self, grid, taps):
        H, W, K = self.H, self.W, self.K
        Wp = W + K // 2
        g = np.asarray(grid, np.float32)
        t = np.asarray(taps, np.float32)
        y = np.zeros((H, W, self.Ch), np.float32) + t[K * K]
        for dy in range(K):
            for dx in range(K):
                # pixel (y, x)'s neighbour (y + dy - p, x + dx - p) is grid row (y + dy) Wp + x + dx
                idx = (np.arange(H)[:, None] + dy) * Wp + np.arange(W)[None, :] + dx
                y += g[idx] * t[dy * K + dx]
        return y.reshape(H * W, self.Ch).astype(bfloat16)

    def tolerance(self):
        from aie.utils.verify import Tolerance

        return Tolerance.relative(0.02, 1e-2, note="nine products summed in float32, rounded once")


def band(ch: int, w: int, k: int, cols: int = 8) -> int | None:
    """The widest band (96, 64, 32) that divides ch over the columns and fits a core."""
    return next((b for b in (96, 64, 32) if ch % (cols * b) == 0 and depths(w, b, k)), None)


def dwconv3(grid, taps, h: int, w: int, c: int = 96, k: int = 3):
    """The k x k depthwise convolution of the padded ``grid`` (grid_rows(h, w, k), Ch) by
    ``taps`` (k k + 1, Ch): (h w, Ch) in token order."""
    return DWConv3.step(grid, taps, H=h, W=w, C=c, K=k)
