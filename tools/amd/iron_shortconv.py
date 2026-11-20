"""LFM2/LFM3's gated short convolution of a decode step as one operator (kernels/short_conv.cc).

A conv layer's mixer is ``out_proj(C * conv(B * x))``, B, C and x the three parts of in_proj's
row and conv a depth-wise causal filter of ``K`` taps (``conv_L_cache``) over the ``K - 1`` rows
before. IRON's graph (``F.causal_conv1d``) runs a decode step's as about 16 operators: copies into
its window, a multiply and an add a tap, and the gate's two multiplies. Each is a
reconfiguration of the array (about 40 us and 2.3 us a KB of its configuration, docs/design.md
16.264), together about 1.1 ms of a layer's. ``ShortConvStep`` is one: each of ``n`` columns' core
takes its ``width / n`` channels of B, C, x and the taps on one channel and the state's rows on
the other, and gives back y and the state's next rows.

    python tools/amd/iron_shortconv.py check OUT [--width 2048] [--taps 7]
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


def bf16(x) -> np.ndarray:
    return np.asarray(x, np.float32).astype(bfloat16)


def short_conv_reference(p, taps, state):
    """ShortConvStep in numpy: ``p`` in_proj's row (B, C, x), ``taps`` (K, width), ``state`` the K - 1
    rows before (oldest first). y = C * conv(B * x) and the K - 1 rows ending with B * x, rounded
    as the kernel does (u, conv(u) and y once each)."""
    p = np.asarray(p, np.float32).reshape(3, -1)
    b, c, x = p
    t = np.asarray(taps, np.float32)
    s = np.asarray(state, np.float32)
    u = bf16(b * x).astype(np.float32)
    conv = bf16(t[-1] * u + (t[:-1] * s).sum(axis=0)).astype(np.float32)
    y = bf16(c * conv).reshape(1, -1)
    fin = bf16(np.concatenate([s[1:], u[None]]))
    return y, fin


@functools.cache
def short_conv_operator(n: int = COLUMNS):
    """``y, final = ShortConvStep(p, taps, state)``: a decode step's gated short convolution over
    ``n`` columns of one core each."""
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import In, ObjectFifo, Out, Worker
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    class ShortConvStep(Operator):
        """``y, final = ShortConvStep(p, taps, state)``: ``p`` (1, 3 width) in_proj's row (B, C, x),
        ``taps`` (K, width), ``state`` (K - 1, width) the rows before; ``y`` (1, width) C *
        conv(B * x), ``final`` (K - 1, width) the rows ending with B * x."""

        width: int = param(array=True)
        taps_n: int = param(array=True)
        columns: int = param(default=n, init=False, array=True)
        w: int = param(derive=lambda op: op.width // n, array=True)
        three: int = param(derive=lambda op: 3 * op.width, repr=False)
        rest: int = param(derive=lambda op: op.taps_n - 1, repr=False)
        # a core's object on the first channel: B, C and x, then the taps
        xo: int = param(derive=lambda op: (3 + op.taps_n) * (op.width // n), repr=False)
        so: int = param(derive=lambda op: (op.taps_n - 1) * (op.width // n), repr=False)

        p = In(1, three, obj_shape=(xo,), per=(columns,))
        taps = In(taps_n, width, obj_shape=(xo,), per=(columns,))  # on p's channel, after the row
        state = In(rest, width, obj_shape=(so,), per=(columns,), depth=1)
        y = Out(1, width, obj_shape=(w,), per=(columns,))
        final = Out(rest, width, obj_shape=(so,), per=(columns,), depth=1)

        def validate(self):
            if self.width % (n * 16) or self.taps_n < 2:
                raise ValueError(f"ShortConvStep: {self.width} channels over {n} columns of whole "
                                 f"vectors, {self.taps_n} taps")

        def build(self) -> Built:
            W, K = self.w, self.taps_n
            kdir = Path(config.aie_kernels_dir())
            here = Path(__file__).resolve().parent / "kernels"
            x_t = np.ndarray[((3 + K) * W,), np.dtype[bfloat16]]
            s_t = np.ndarray[((K - 1) * W,), np.dtype[bfloat16]]
            y_t = np.ndarray[(W,), np.dtype[bfloat16]]
            kern = ExternalFunction(
                "short_conv_step", arg_types=[x_t, s_t, y_t, s_t], source_file=str(here / "short_conv.cc"),
                include_dirs=[str(kdir), config.cxx_header_path()],
                compile_flags=[f"-DSC_W={W}", f"-DSC_K={K}"], object_file_name=f"short_conv_{W}_{K}.o")
            x_f = [ObjectFifo(x_t, depth=1) for _ in range(n)]
            s_f = [ObjectFifo(s_t, depth=1) for _ in range(n)]
            y_f = [ObjectFifo(y_t, depth=1) for _ in range(n)]
            f_f = [ObjectFifo(s_t, depth=1) for _ in range(n)]

            def core(xf, sf, yf, ff, k):
                xo, so, yo, fo = xf.acquire(1), sf.acquire(1), yf.acquire(1), ff.acquire(1)
                k(xo, so, yo, fo)
                xf.release(1)
                sf.release(1)
                yf.release(1)
                ff.release(1)

            workers = [Worker(core, [x_f[c].cons(), s_f[c].cons(), y_f[c].prod(), f_f[c].prod(), kern])
                       for c in range(n)]
            self.p.bind([f.prod() for f in x_f])
            self.taps.bind([f.prod() for f in x_f])
            self.state.bind([f.prod() for f in s_f])
            self.y.bind([f.cons() for f in y_f])
            self.final.bind([f.cons() for f in f_f])
            return Built(workers, [])

        def sequence(self, rt):
            E, W, K = self.width, self.w, self.taps_n
            for c in range(n):
                # B, C and x of the column's channels, then its taps: one object
                self.p.fill(TAP((1, 3 * E), c * W, [3, W], [E, 1]), lane=c, managed=False)
                self.taps.fill(TAP((K, E), c * W, [K, W], [E, 1]), lane=c, managed=False)
                self.state.fill(TAP((K - 1, E), c * W, [K - 1, W], [E, 1]), lane=c, managed=False)
            drains = []
            for c in range(n):
                drains.append(self.y.drain(TAP((1, E), c * W, [W], [1]), lane=c, wait=True, managed=False))
                drains.append(self.final.drain(TAP((K - 1, E), c * W, [K - 1, W], [E, 1]), lane=c,
                                               wait=True, managed=False))
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * (self.taps_n + 2) * self.width

        def reference(self, p, taps, state):
            return short_conv_reference(p, taps, state)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.01, 1e-3, note="float32 sums, rounded once to bf16")

    return ShortConvStep


def _check(args) -> int:
    import iron
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    E, K = args.width, args.taps
    Op = short_conv_operator()
    rng = np.random.default_rng(5)
    taps = iron.param(bf16(rng.standard_normal((K, E)) * 0.3))

    class Graph(iron.Graph):
        def body(self, p, state):
            y, final = Op.step(p, taps, state, width=E, taps_n=K)
            if args.plus:  # a second design after it: the array configured again (its cost)
                y = y + y
            return y, final

        def reference(self, p, state):
            y, final = short_conv_reference(p, np.asarray(taps), state)
            if args.plus:
                y = bf16(np.asarray(y, np.float32) * 2)
            return y, final

    g = Graph()
    cg = g.compile(p=(1, 3 * E), state=(K - 1, E))
    m = export_graph(cg, args.out, "shortconv", dict(width=E, taps=K))
    print(cg.plan.report("shortconv"))
    p = bf16(rng.standard_normal((1, 3 * E)))
    state = bf16(rng.standard_normal((K - 1, E)))
    write_check(g, m, args.out, dict(p=p, state=state), None)
    print(f"{args.out}: ShortConvStep of {E} channels, {K} taps")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of one ShortConvStep and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--width", type=int, default=2048)
    c.add_argument("--taps", type=int, default=7)
    c.add_argument("--plus", action="store_true", help="y + y after it (a second design)")
    return _check(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
