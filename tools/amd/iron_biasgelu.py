"""GELU of rows plus a bias row as one operator (kernels/bias_gelu.cc): what a RowwiseAdd and
iron_gelu.GELUEven did in two.

``y = BiasGELU.step(x, b)``: x (rows, C), b (C,); y (rows, C) = gelu(x + b) (the tanh form,
iron_gelu.gelu_tanh_ref). Each of ``columns`` columns' core takes C / columns channels of every
row, TILE rows an object.

    python tools/amd/iron_biasgelu.py check OUT [--rows 192] [--channels 2304]
    go run ./tools/amd/irongraph OUT
"""

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_gelu import gelu_tanh_ref  # noqa: E402

COLUMNS = 8
TILE = 16
KERNELS = Path(__file__).resolve().parent / "kernels"


def reference(x, b) -> np.ndarray:
    """BiasGELU in numpy: x + b rounded to bf16, the tanh GELU."""
    s = (np.asarray(x, np.float32) + np.asarray(b, np.float32)).astype(bfloat16).astype(np.float64)
    return gelu_tanh_ref(s)


@functools.cache
def operator(n: int = COLUMNS):
    """The BiasGELU operator class over ``n`` columns."""
    from aie.iron import In, ObjectFifo, Out, TaskGroup, Worker
    from aie.iron.controlflow import range_
    from aie.iron.kernels import activation as act
    from aie.iron.kernels._common import _make_extern  # noqa: F401
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    class BiasGELU(Operator):
        """``y = BiasGELU.step(x, b)``: the module's operator."""

        rows: int = param(array=True)
        channels: int = param(array=True)
        columns: int = param(default=n, init=False, array=True)
        cb: int = param(derive=lambda op: op.channels // n, array=True)
        xo: int = param(derive=lambda op: TILE * (op.channels // n), array=True)

        x = In(rows, channels, obj_shape=(xo,), per=(columns,))
        b = In(channels, obj_shape=(cb,), per=(columns,))
        y = Out(rows, channels, obj_shape=(xo,), per=(columns,))

        def validate(self):
            if self.channels % (32 * n) or self.rows % TILE:
                raise ValueError(f"BiasGELU: {self.channels} channels over {n} columns of whole vectors, "
                                 f"{self.rows} rows in tiles of {TILE}")

        def build(self) -> Built:
            CB = self.cb
            tiles = self.rows // TILE
            kdir = Path(config.aie_kernels_dir())
            x_t = np.ndarray[(self.xo,), np.dtype[bfloat16]]
            b_t = np.ndarray[(CB,), np.dtype[bfloat16]]
            include = [str(kdir), config.cxx_header_path(), str(KERNELS),
                       str(Path(config.aie_runtime_lib_dir()) / act._detect_arch().upper())]
            kern = ExternalFunction(
                "bias_gelu_bf16", arg_types=[x_t, b_t, x_t], source_file=str(kdir / "common" / "lut_kernel.cc"),
                include_dirs=include,
                compile_flags=["-DACTIVATIONS_TANH_LUT=1", f'-DAIE_LUT_KERNEL_SOURCE="{KERNELS / "bias_gelu.cc"}"',
                               f"-DBG_C={CB}", f"-DBG_R={TILE}"],
                object_file_name=f"bias_gelu_{CB}_{TILE}.o")
            x_f = [ObjectFifo(x_t, depth=2) for _ in range(n)]
            b_f = [ObjectFifo(b_t, depth=1) for _ in range(n)]
            y_f = [ObjectFifo(x_t, depth=2) for _ in range(n)]

            def core(xf, bf_, yf, k):
                bo = bf_.acquire(1)
                for _ in range_(tiles):
                    xo, yo = xf.acquire(1), yf.acquire(1)
                    k(xo, bo, yo)
                    xf.release(1)
                    yf.release(1)
                bf_.release(1)

            workers = [Worker(core, [x_f[c].cons(), b_f[c].cons(), y_f[c].prod(), kern]) for c in range(n)]
            self.x.bind([f.prod() for f in x_f])
            self.b.bind([f.prod() for f in b_f])
            self.y.bind([f.cons() for f in y_f])
            return Built(workers)

        def sequence(self, rt):
            from aie.helpers.taplib import TensorAccessPattern as TAP

            R, C, CB = self.rows, self.channels, self.cb
            tiles = R // TILE
            tg = TaskGroup()
            for c in range(n):
                self.b.fill(TAP((C,), c * CB, [CB], [1]), lane=c, group=tg)
                self.x.fill(TAP((R, C), c * CB, [tiles, TILE, CB], [TILE * C, C, 1]), lane=c, group=tg)
            for c in range(n):
                self.y.drain(TAP((R, C), c * CB, [tiles, TILE, CB], [TILE * C, C, 1]), lane=c, group=tg, wait=True)
            tg.finish()

        def ops(self) -> int:
            return 20 * self.rows * self.channels

        def reference(self, x, b):
            return reference(x, b)

        def tolerance(self) -> Tolerance:
            return Tolerance.bf16_ulps(2, atol=1e-2)

    return BiasGELU


def bias_gelu(x, b):
    """BiasGELU.step of x (rows, C), b (C,)."""
    return operator().step(x, b, rows=x.shape[0], channels=x.shape[1])


def _check(a) -> int:
    import iron

    from iron_graph import bind_device, export_graph, write_check

    bind_device(a.device, COLUMNS)
    rng = np.random.default_rng(0)

    class G(iron.Graph):
        def __init__(self):
            self.b = iron.param(rng.standard_normal(a.channels).astype(bfloat16))

        def body(self, x):
            return bias_gelu(x, self.b)

    g = G()
    cg = g.compile(x=(a.rows, a.channels))
    m = export_graph(cg, a.out, "bias_gelu", dict(rows=a.rows))
    write_check(g, m, a.out, dict(x=(rng.standard_normal((a.rows, a.channels)) * 2).astype(bfloat16)))
    print(m["graph"]["plan"].splitlines()[0])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of one BiasGELU and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--rows", type=int, default=192)
    c.add_argument("--channels", type=int, default=2304)
    c.add_argument("--device", default="npu2")
    return _check(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
