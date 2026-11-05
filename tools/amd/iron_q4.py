"""4-bit weights (q4nx) for IRON graphs: the quantizer, the decode GEMV and its check.

q4nx is FastFlowLM's format (mlir-aie's flm_gemma4/q4nx.h, IRON's iron.operators.flm.q4nx):
a block holds 32 out-features by 256 in-features, a bf16 scale and minimum for each 32
in-features of an out-feature, then the 4-bit codes; a weight is ``min + scale * code``.
5 bits a weight with the scales, 31 % of bf16.

The blocks of a weight (N out-features, K in-features) are in the order
iron.operators.flm.DequantBFP reads (its README, layers 6 to 9): column blocks of 64
out-features, each K / 512 objects of two pairs of blocks (in-features 0-255, then 256-511),
a pair's blocks the out-features 0-31, then 32-63. A column block is contiguous, so the GEMV's
cores each read a run of whole column blocks, and prefill can dequantize the same bytes into
the bfp16 that flm.ProjectionGEMM reads.

Q4GEMV is ``y = W x`` for one row: one core a column, each with the column blocks of
``N / columns`` out-features and all of x. Its kernel (kernels/q4_mv.cc) is FastFlowLM's
q4nx_accumulate.

    python tools/amd/iron_q4.py check OUT --n 1536 --k 1536   # a graph of one Q4GEMV, with check/
    go run ./tools/amd/irongraph OUT
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

M_TILE, K_TILE, GROUP = 32, 256, 32
BLOCK_BYTES = M_TILE * K_TILE // 2 + 4 * (M_TILE * K_TILE // GROUP)  # 5120
CB = 2 * M_TILE  # out-features a column block


def q4_bytes(n: int, k: int) -> int:
    return n * k // (M_TILE * K_TILE) * BLOCK_BYTES


def _fit(g):
    """Each group's (scale, minimum) of the codes 0..15 for g (groups, 32) float32: the clip range
    of least squared error among a few shrinks of [min, max], then the scale and minimum fitted to
    the codes by least squares (as llama.cpp's make_qkx quantizers), rounded to bf16."""
    lo, hi = g.min(-1), g.max(-1)
    span = hi - lo
    best = (np.full(len(g), np.inf, np.float32), lo, span / 15)
    for t_lo in (0.0, 0.03, 0.07):
        for t_hi in (0.0, 0.03, 0.07):
            a, b = lo + t_lo * span, hi - t_hi * span
            sc = np.where(b > a, (b - a) / 15, 1)
            c = np.clip(np.rint((g - a[:, None]) / sc[:, None]), 0, 15)
            # least squares of g ~ m + s * c over the group
            n = g.shape[1]
            sc_ = c.sum(1)
            scc = (c * c).sum(1)
            sg = g.sum(1)
            scg = (c * g).sum(1)
            det = n * scc - sc_ * sc_
            ok = det > 0
            s_ls = np.where(ok, (n * scg - sc_ * sg) / np.where(ok, det, 1), sc)
            m_ls = np.where(ok, (sg - s_ls * sc_) / n, a)
            s_ls = s_ls.astype(bfloat16).astype(np.float32)
            m_ls = m_ls.astype(bfloat16).astype(np.float32)
            safe = np.where(s_ls != 0, s_ls, 1)
            c = np.clip(np.rint((g - m_ls[:, None]) / safe[:, None]), 0, 15)
            err = ((m_ls[:, None] + s_ls[:, None] * c - g) ** 2).sum(1)
            better = err < best[0]
            best = (np.where(better, err, best[0]), np.where(better, m_ls, best[1]), np.where(better, s_ls, best[2]))
    return best[2].astype(bfloat16).astype(np.float32), best[1].astype(bfloat16).astype(np.float32)


def quantize(w) -> np.ndarray:
    """q4nx of w (N, K), N a multiple of 64 and K of 512, in DequantBFP's block order:
    uint8 (N / 64 * K / 256 * 2, BLOCK_BYTES).

    Each group of 32 in-features of an out-feature gets the scale and minimum of _fit; the codes
    are chosen against the bf16 scale and minimum.
    """
    w = np.asarray(w, np.float32)
    N, K = w.shape
    if N % CB or K % 512:
        raise ValueError(f"q4nx: ({N}, {K}) is not 64 x 512 blocks")
    g = w.reshape(N, K // GROUP, GROUP)
    scale = np.empty((N, K // GROUP), np.float32)
    mn = np.empty_like(scale)
    rows = max(1, (1 << 20) // K)  # a few MB of groups at a time
    for r in range(0, N, rows):
        part = g[r : r + rows].reshape(-1, GROUP)
        sc, m = _fit(part)
        scale[r : r + rows] = sc.reshape(-1, K // GROUP)
        mn[r : r + rows] = m.reshape(-1, K // GROUP)
    safe = np.where(scale != 0, scale, 1)
    codes = np.clip(np.rint((g - mn[..., None]) / safe[..., None]), 0, 15).astype(np.uint8)
    codes = codes.reshape(N, K)
    # blocks [N/32, K/256]: scales and minima [k // 32, m], codes [m // 16, k, m % 16]
    nb, kb = N // M_TILE, K // K_TILE
    s = scale.reshape(nb, M_TILE, kb, K_TILE // GROUP).transpose(0, 2, 3, 1).astype(bfloat16)
    m = mn.reshape(nb, M_TILE, kb, K_TILE // GROUP).transpose(0, 2, 3, 1).astype(bfloat16)
    c = codes.reshape(nb, M_TILE // 16, 16, kb, K_TILE).transpose(0, 3, 1, 4, 2).reshape(nb, kb, -1)
    packed = (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)
    blocks = np.concatenate(
        [s.reshape(nb, kb, -1).view(np.uint8), m.reshape(nb, kb, -1).view(np.uint8), packed], axis=-1
    )  # [N/32, K/256, BLOCK_BYTES]
    # DequantBFP's order: [N/64][K/512][2 k-halves][2 n-halves]
    blocks = blocks.reshape(N // CB, 2, K // 512, 2, BLOCK_BYTES).transpose(0, 2, 3, 1, 4)
    return np.ascontiguousarray(blocks.reshape(-1, BLOCK_BYTES))


def dequantize(q, n: int, k: int) -> np.ndarray:
    """The (n, k) float32 weights of quantize's blocks."""
    q = np.asarray(q, np.uint8).reshape(n // CB, k // 512, 2, 2, BLOCK_BYTES)
    q = q.transpose(0, 3, 1, 2, 4).reshape(n // M_TILE, k // K_TILE, BLOCK_BYTES)
    sm = M_TILE * K_TILE // GROUP
    s = q[..., : 2 * sm].copy().view(bfloat16).astype(np.float32).reshape(n // M_TILE, k // K_TILE, -1, M_TILE)
    m = q[..., 2 * sm : 4 * sm].copy().view(bfloat16).astype(np.float32).reshape(n // M_TILE, k // K_TILE, -1, M_TILE)
    p = q[..., 4 * sm :]
    c = np.empty((*p.shape[:2], p.shape[2] * 2), np.uint8)
    c[..., 0::2], c[..., 1::2] = p & 15, p >> 4
    c = c.reshape(n // M_TILE, k // K_TILE, M_TILE // 16, K_TILE, 16).transpose(0, 2, 4, 1, 3).reshape(n, k)
    s = s.transpose(0, 3, 1, 2).reshape(n, k // GROUP)
    m = m.transpose(0, 3, 1, 2).reshape(n, k // GROUP)
    return c.astype(np.float32) * np.repeat(s, GROUP, 1) + np.repeat(m, GROUP, 1)


def _kernel(k: int):
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config

    kernels = Path(config.aie_kernels_dir())
    here = Path(__file__).resolve().parent / "kernels"
    blk = np.ndarray[(BLOCK_BYTES,), np.dtype[np.uint8]]
    x = np.ndarray[(k,), np.dtype[bfloat16]]
    sums = np.ndarray[(k // GROUP,), np.dtype[bfloat16]]
    acc = np.ndarray[(CB,), np.dtype[np.float32]]
    y = np.ndarray[(CB,), np.dtype[bfloat16]]
    common = dict(
        source_file=str(here / "q4_mv.cc"),
        include_dirs=[str(kernels / "flm_gemma4"), str(kernels), config.cxx_header_path()],
        compile_flags=[f"-DQ4_K={k}"],
    )
    obj = f"q4_mv_{k}.o"
    begin = ExternalFunction("q4_mv_begin", arg_types=[x, sums, acc], object_file_name=obj, **common)
    block = ExternalFunction("q4_mv_block", arg_types=[blk, x, sums, acc, np.int32, np.int32],
                             object_file_name=obj, **common)
    end = ExternalFunction("q4_mv_end", arg_types=[acc, y], object_file_name=obj, **common)
    return begin, block, end


def _operator():
    """Q4GEMV, defined on first use: IRON's modules load only in its environment."""
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.iron import Buffer, In, ObjectFifo, Out, Worker
    from aie.iron.controlflow import range_
    from aie.utils.compile.jit.params import param, tunable, widths
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    class Q4GEMV(Operator):
        """``y = W x``: W as q4nx blocks (quantize), x one row of K, y N.

        One core a column, each with N / columns out-features (whole column blocks of 64).
        """

        # array=True: the counts are the cores' loops
        blocks: int = param(array=True)  # N / 64 * K / 256 * 2
        K: int = param(array=True)
        N: int = param(derive=lambda op: op.blocks * CB // (2 * op.K // K_TILE), array=True)
        num_aie_columns: int = tunable(domain=widths)

        W = In(blocks, BLOCK_BYTES, dtype=np.uint8, obj_shape=(1, BLOCK_BYTES), per=(num_aie_columns,))
        x = In(K, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
        y = Out(N, obj_shape=(CB,), per=(num_aie_columns,))

        def validate(self):
            if self.K % 512:
                raise ValueError(f"Q4GEMV: K ({self.K}) must be a multiple of 512")
            if self.blocks % (2 * self.K // K_TILE):
                raise ValueError(f"Q4GEMV: {self.blocks} blocks are no whole column blocks of K {self.K}")

        def resolve(self, dev):
            cbs = self.N // CB
            cols = self.resolve_columns(dev, self.num_aie_columns, fits=lambda c: cbs % c == 0)
            return self.replace(num_aie_columns=cols)

        def compatible(self):
            if (self.N // CB) % self.num_aie_columns:
                raise ValueError(f"Q4GEMV: {self.N // CB} column blocks over {self.num_aie_columns} columns")

        def build(self) -> Built:
            K, n = self.K, self.num_aie_columns
            cbs = self.N // CB // n
            kbs = K // K_TILE
            begin, block, end = _kernel(K)
            w_f = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            x_f = [ObjectFifo(self.x.obj_type, depth=1) for _ in range(n)]
            y_f = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            sums = [Buffer(np.ndarray[(K // GROUP,), np.dtype[bfloat16]]) for _ in range(n)]
            accs = [Buffer(np.ndarray[(CB,), np.dtype[np.float32]]) for _ in range(n)]

            def core(w_f, x_f, y_f, sums, acc, begin, block, end):
                x = x_f.acquire(1)
                begin(x, sums, acc)
                for _ in range_(cbs):
                    for kb in range_(kbs):
                        kb32 = index.casts(T.i32(), kb)
                        for h in range(2):
                            w = w_f.acquire(1)
                            block(w, x, sums, acc, kb32, h)
                            w_f.release(1)
                    y = y_f.acquire(1)
                    end(acc, y)
                    y_f.release(1)
                x_f.release(1)

            workers = [
                Worker(core, [w_f[i].cons(), x_f[i].cons(), y_f[i].prod(), sums[i], accs[i], begin, block, end])
                for i in range(n)
            ]
            self.W.bind([f.prod() for f in w_f])
            self.x.bind([f.prod() for f in x_f])
            self.y.bind([f.cons() for f in y_f])
            return Built(workers, [])

        def ops(self) -> int:
            return 2 * self.N * self.K

        def reference(self, W, x):
            w = dequantize(W, self.N, self.K)
            return (w.astype(np.float64) @ np.asarray(x, np.float64)).astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="32-column partial sums narrowed to bf16 before the scale")

    return Q4GEMV


_Q4GEMV = None


def Q4GEMV():
    global _Q4GEMV
    if _Q4GEMV is None:
        _Q4GEMV = _operator()
    return _Q4GEMV


def q4_linear(x, q, n: int, k: int):
    """``x @ W.T`` for x (k,), (1, k) or (rows, k) and q, W's q4nx blocks as a 1-D iron.param.

    One row runs Q4GEMV on the blocks. Several (a prompt chunk) dequantize them on the device
    into bfp16 (iron.operators.flm.DequantBFP, into the graph's scratch) for
    flm.ProjectionGEMM, as IRON's Gemma 4 at Precision.MEDIUM multiplies anyway.
    """
    one_row = len(x.shape) == 1 or x.shape[0] == 1
    if one_row:
        y = Q4GEMV().step(q.reshape(-1, BLOCK_BYTES), x.reshape(k))
        return y if len(x.shape) == 1 else y.reshape(1, n)
    from iron import Precision
    from iron.operators.flm import DequantBFP, ProjectionGEMM
    from iron.operators.flm.gemm.op import Epilogue

    packed = DequantBFP.step(q, K=k, N=n)
    return ProjectionGEMM.step(x, packed, K=k, N=n, bfp16_b=True, precision=Precision.MEDIUM,
                               epilogue_modes=(Epilogue.NONE,))


# q4nx weights by id of their iron.param: (the param, out-features, in-features). The param is
# held, so no other object takes its id.
_Q4: dict[int, tuple[object, int, int]] = {}


def q4_param(w):
    """An iron.param of w's q4nx blocks (quantize), known to the patched F.linear."""
    import iron

    n, k = np.shape(w)
    p = iron.param(quantize(w).reshape(-1))
    _Q4[id(p)] = (p, n, k)
    return p


def install_linear() -> None:
    """Make iron.functional.linear run q4_linear on the weights q4_param made."""
    import iron.functional as F

    if getattr(F.linear, "hwnpu_q4", False):
        return
    base = F.linear

    def linear(x, weight, bias=None, *, precision, **gemv):
        entry = _Q4.get(id(weight))
        if entry is None or entry[0] is not weight:
            return base(x, weight, bias, precision=precision, **gemv)
        y = q4_linear(x, weight, entry[1], entry[2])
        if bias is None:
            return y
        rows = int(np.prod(y.shape)) // bias.shape[0]
        return (y.reshape(rows, bias.shape[0]) + bias).reshape(y.shape)

    linear.hwnpu_q4 = True
    F.linear = linear
    _fix_bfp_fits()


def _fix_bfp_fits() -> None:
    """IRON's Operand.fits hands the device a block-float marker class (v8bfp16ebs8) as the
    dtype, which np.dtype refuses: flm.ProjectionGEMM with a packed bfp16 B and a per-call row
    bound (a prompt chunk) fails there. The same check on a dtype of the block's bytes."""
    from aie.utils import bfp

    from iron.common.declare.operand import Operand

    base = Operand.fits
    if getattr(base, "hwnpu", False):
        return

    def fits(self, tap):
        try:
            np.dtype(self.dtype)
        except TypeError:
            # the pattern counts blocks of b bytes: the same walk in bytes, a block a run
            from aie.helpers.taplib import TensorAccessPattern

            b = bfp.itemsize(self.dtype)
            sizes, strides = list(tap.sizes), [x * b for x in tap.strides]
            if tap.strides[-1] == 1:
                sizes[-1] *= b
                strides[-1] = 1
            else:
                sizes, strides = sizes + [b], strides + [1]
            dims = (int(np.prod(tap.tensor_dims)) * b,)
            byte_tap = TensorAccessPattern(dims, tap.offset * b, sizes, strides)
            return self.op.dev.dma_bd_fits(0, 0, byte_tap, np.uint8)
        return base(self, tap)

    fits.hwnpu = True
    Operand.fits = fits


def _check(args) -> int:
    import iron
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    n, k = args.n, args.k
    rng = np.random.default_rng(1)
    w = rng.standard_normal((n, k)).astype(np.float32) * 0.05
    q = quantize(w)
    deq = dequantize(q, n, k)
    err = np.abs(deq - w).max() / np.abs(w).max()
    print(f"quantization: largest error {err:.4f} of the largest weight")

    from iron import Scratchpad

    class G(iron.Graph):
        def __init__(self):
            self.q = iron.param(q.reshape(-1))

        def body(self, x, *, rows: Scratchpad[np.int32] = np.int32(0)):
            if args.rows:  # a prompt chunk's: bounded per call
                return q4_linear(x[:rows], self.q, n, k)
            return q4_linear(x, self.q, n, k)

        def reference(self, x, rows=0):
            x = np.asarray(x, np.float64)
            y = (x @ deq.astype(np.float64).T).astype(bfloat16)
            if rows:
                y[rows:] = 0
            return y

    g = G()
    shape = (k,) if args.rows == 0 else (args.rows, k)
    cg = g.compile(x=shape)
    m = export_graph(cg, args.out, "q4linear", dict(n=n, k=k))
    x = rng.standard_normal(shape).astype(bfloat16)
    write_check(g, m, args.out, dict(x=x), dict(rows=args.valid or args.rows) if args.rows else None)
    print(f"{args.out}: Q4GEMV {n} x {k}, {q.nbytes >> 10} KiB of weights")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of one Q4GEMV and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--n", type=int, default=1536)
    c.add_argument("--k", type=int, default=1536)
    c.add_argument("--rows", type=int, default=0, help="rows of x (0: one, (k,)): several run the prefill path")
    c.add_argument("--valid", type=int, default=0, help="rows the call bounds x to (default all)")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return _check(a)
    return 1


if __name__ == "__main__":
    sys.exit(main())
