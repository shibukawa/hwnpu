"""A sandwich-normed block's residual and the next norm of a decode step as one operator
(kernels/sandwich_add.cc).

After each block of a sandwich-normed layer (LFM3's, Gemma's), ``y = x + rms(h) * w_post`` and
the next block's input ``rms(y) * w_pre``: IRON's graph runs a decode step's as three operators
(RMSNorm, ElementwiseAdd, RMSNorm), each a reconfiguration of the array (about 40 us and 2.3 us
a KB of its configuration, docs/design/amd.md). ``SandwichAdd`` is one, on one core: h and the
two weights on one channel, x on the other; y and the next input out.

    python tools/amd/iron_sandwich.py check OUT [--width 2048]
    go run ./tools/amd/irongraph OUT
"""

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))


def bf16(x) -> np.ndarray:
    return np.asarray(x, np.float32).astype(bfloat16)


def sandwich_reference(h, x, w_post, w_pre, eps: float, scale=None):
    """SandwichAdd in numpy (float64 inside): y = x + rms(h) w_post (then, rounded, times scale),
    n = rms(y) w_pre, each bf16."""
    def norm(v, w):
        v = np.asarray(v, np.float64)
        return v / np.sqrt((v * v).mean(axis=-1, keepdims=True) + eps) * np.asarray(w, np.float64)

    p = bf16(norm(h, w_post)).astype(np.float64)
    y = bf16(np.asarray(x, np.float64) + p)
    if scale is not None:
        y = bf16(y.astype(np.float64) * np.asarray(scale, np.float64))
    return y, bf16(norm(y.astype(np.float64), w_pre))


@functools.cache
def sandwich_operator(scaled: bool = False):
    """``y, n = SandwichAdd(h, x, w_post, w_pre[, scale])``: one row, one core; with ``scaled``, y
    (rounded) times ``scale`` before the next norm."""
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import In, ObjectFifo, Out, Worker
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    class SandwichAdd(Operator):
        """``y, n = SandwichAdd(h, x, w_post, w_pre)``: ``h`` the block's output and ``x`` the
        residual (1, width); ``y`` = x + rms(h) w_post, ``n`` = rms(y) w_pre."""

        width: int = param(array=True)
        eps: float = param(array=True)
        lanes: int = param(default=1, init=False, array=True)
        hw: int = param(derive=lambda op: (3 + scaled) * op.width, repr=False)

        h = In(1, width, obj_shape=(hw,), per=(lanes,))
        x = In(1, width, obj_shape=(width,), per=(lanes,))
        w_post = In(width, obj_shape=(hw,), per=(lanes,))  # on h's channel, after it
        w_pre = In(width, obj_shape=(hw,), per=(lanes,))
        if scaled:  # y times scale (Gemma 4's layer_scalar), on h's channel after w_pre
            scale = In(width, obj_shape=(hw,), per=(lanes,))
        y = Out(1, width, obj_shape=(width,), per=(lanes,))
        n = Out(1, width, obj_shape=(width,), per=(lanes,))

        def validate(self):
            if self.width % 32:
                raise ValueError(f"SandwichAdd: {self.width} channels, whole vectors of 32")

        def build(self) -> Built:
            E = self.width
            kdir = Path(config.aie_kernels_dir())
            here = Path(__file__).resolve().parent / "kernels"
            hw_t = np.ndarray[((3 + scaled) * E,), np.dtype[bfloat16]]
            r_t = np.ndarray[(E,), np.dtype[bfloat16]]
            kern = ExternalFunction(
                "sandwich_add", arg_types=[hw_t, r_t, r_t, r_t], source_file=str(here / "sandwich_add.cc"),
                include_dirs=[str(kdir), config.cxx_header_path()],
                compile_flags=[f"-DSA_N={E}", f"-DSA_EPS={self.eps!r}f"] + (["-DSA_SCALE=1"] if scaled else []),
                object_file_name=f"sandwich_add_{E}_{self.eps!r}{'_s' if scaled else ''}.o")
            hf, xf = ObjectFifo(hw_t, depth=1), ObjectFifo(r_t, depth=1)
            yf, nf = ObjectFifo(r_t, depth=1), ObjectFifo(r_t, depth=1)

            def core(hf, xf, yf, nf, k):
                a, b, c, d = hf.acquire(1), xf.acquire(1), yf.acquire(1), nf.acquire(1)
                k(a, b, c, d)
                hf.release(1)
                xf.release(1)
                yf.release(1)
                nf.release(1)

            workers = [Worker(core, [hf.cons(), xf.cons(), yf.prod(), nf.prod(), kern])]
            self.h.bind([hf.prod()])
            self.w_post.bind([hf.prod()])
            self.w_pre.bind([hf.prod()])
            if scaled:
                self.scale.bind([hf.prod()])
            self.x.bind([xf.prod()])
            self.y.bind([yf.cons()])
            self.n.bind([nf.cons()])
            return Built(workers, [])

        def sequence(self, rt):
            E = self.width
            self.h.fill(TAP((1, E), 0, [E], [1]), lane=0, managed=False)
            self.w_post.fill(TAP((E,), 0, [E], [1]), lane=0, managed=False)
            self.w_pre.fill(TAP((E,), 0, [E], [1]), lane=0, managed=False)
            if scaled:
                self.scale.fill(TAP((E,), 0, [E], [1]), lane=0, managed=False)
            self.x.fill(TAP((1, E), 0, [E], [1]), lane=0, managed=False)
            drains = [self.y.drain(TAP((1, E), 0, [E], [1]), lane=0, wait=True, managed=False),
                      self.n.drain(TAP((1, E), 0, [E], [1]), lane=0, wait=True, managed=False)]
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 8 * self.width

        if scaled:
            def reference(self, h, x, w_post, w_pre, scale):
                return sandwich_reference(h, x, w_post, w_pre, self.eps, scale)
        else:
            def reference(self, h, x, w_post, w_pre):
                return sandwich_reference(h, x, w_post, w_pre, self.eps)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.01, 1e-3, note="float32 sums, rounded once to bf16")

    return SandwichAdd


def sandwich_add(h, x, w_post, w_pre, eps: float, scale=None):
    """``(y, n)``: the residual ``x + rms(h) w_post`` (times ``scale``, a row) and the next input
    ``rms(y) w_pre``, a decode step's row (SandwichAdd)."""
    E = h.shape[-1]
    if scale is not None:
        return sandwich_operator(True).step(h.reshape(1, E), x.reshape(1, E), w_post, w_pre, scale, width=E,
                                            eps=float(eps))
    return sandwich_operator().step(h.reshape(1, E), x.reshape(1, E), w_post, w_pre, width=E, eps=float(eps))


def _check(args) -> int:
    import iron
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    E, eps = args.width, 1e-5
    rng = np.random.default_rng(7)
    w_post = iron.param(bf16(rng.uniform(0.5, 1.5, E)))
    w_pre = iron.param(bf16(rng.uniform(0.5, 1.5, E)))
    scale = iron.param(bf16(np.full(E, 0.73))) if args.scale else None

    class Graph(iron.Graph):
        def body(self, h, x):
            y, n = sandwich_add(h, x, w_post, w_pre, eps, scale)
            if args.plus:  # a second design after it: the array configured again
                n = n + n
            return y, n

        def reference(self, h, x):
            y, n = sandwich_reference(h, x, np.asarray(w_post), np.asarray(w_pre), eps,
                                      None if scale is None else np.asarray(scale))
            return y, (bf16(np.asarray(n, np.float32) * 2) if args.plus else n)

    g = Graph()
    cg = g.compile(h=(1, E), x=(1, E))
    m = export_graph(cg, args.out, "sandwich", dict(width=E))
    print(cg.plan.report("sandwich"))
    h = bf16(rng.standard_normal((1, E)) * 3)
    x = bf16(rng.standard_normal((1, E)) * 10)
    write_check(g, m, args.out, dict(h=h, x=x), None)
    print(f"{args.out}: SandwichAdd of {E} channels")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of one SandwichAdd and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--width", type=int, default=2048)
    c.add_argument("--plus", action="store_true", help="n + n after it (a second design)")
    c.add_argument("--scale", action="store_true", help="y times a scale row (Gemma 4's layer_scalar)")
    return _check(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
