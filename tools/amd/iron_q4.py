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


# --- q8: 8.5 bits a weight -------------------------------------------------------------------

# A q8 block: 32 out-features by 256 in-features, bf16 scales [k / 32, m], then int8 codes
# [m / 16, k, m % 16] (q4nx's layout, a byte a code, no minima). w = scale * code.
Q8_BLOCK_BYTES = 2 * (M_TILE * K_TILE // GROUP) + M_TILE * K_TILE  # 8704
FORMATS = {"q4": BLOCK_BYTES, "q8": Q8_BLOCK_BYTES}


def quantize8(w) -> np.ndarray:
    """q8 of w (N, K) in q4nx's block order: uint8 (blocks, Q8_BLOCK_BYTES). Each group of 32
    in-features of an out-feature: scale = its largest magnitude / 127 in bf16, codes rounded."""
    w = np.asarray(w, np.float32)
    N, K = w.shape
    if N % CB or K % 512:
        raise ValueError(f"q8: ({N}, {K}) is not 64 x 512 blocks")
    g = w.reshape(N, K // GROUP, GROUP)
    scale = (np.abs(g).max(-1) / 127).astype(bfloat16).astype(np.float32)
    safe = np.where(scale > 0, scale, 1)
    codes = np.clip(np.rint(g / safe[..., None]), -127, 127).astype(np.int8).reshape(N, K)
    nb, kb = N // M_TILE, K // K_TILE
    sc = scale.reshape(nb, M_TILE, kb, K_TILE // GROUP).transpose(0, 2, 3, 1).astype(bfloat16)
    c = codes.reshape(nb, M_TILE // 16, 16, kb, K_TILE).transpose(0, 3, 1, 4, 2).reshape(nb, kb, -1)
    blocks = np.concatenate([sc.reshape(nb, kb, -1).view(np.uint8), c.view(np.uint8)], axis=-1)
    blocks = blocks.reshape(N // CB, 2, K // 512, 2, Q8_BLOCK_BYTES).transpose(0, 2, 3, 1, 4)
    return np.ascontiguousarray(blocks.reshape(-1, Q8_BLOCK_BYTES))


def dequantize8(q, n: int, k: int) -> np.ndarray:
    """The (n, k) float32 weights of quantize8's blocks."""
    q = np.asarray(q, np.uint8).reshape(n // CB, k // 512, 2, 2, Q8_BLOCK_BYTES)
    q = q.transpose(0, 3, 1, 2, 4).reshape(n // M_TILE, k // K_TILE, Q8_BLOCK_BYTES)
    sm = M_TILE * K_TILE // GROUP
    s = q[..., : 2 * sm].copy().view(bfloat16).astype(np.float32).reshape(n // M_TILE, k // K_TILE, -1, M_TILE)
    c = q[..., 2 * sm :].copy().view(np.int8)
    c = c.reshape(n // M_TILE, k // K_TILE, M_TILE // 16, K_TILE, 16).transpose(0, 2, 4, 1, 3).reshape(n, k)
    s = s.transpose(0, 3, 1, 2).reshape(n, k // GROUP)
    return c.astype(np.float32) * np.repeat(s, GROUP, 1)


def quantize_fmt(w, fmt: str) -> np.ndarray:
    return quantize(w) if fmt == "q4" else quantize8(w)


def dequantize_fmt(q, n: int, k: int, fmt: str) -> np.ndarray:
    return dequantize(q, n, k) if fmt == "q4" else dequantize8(q, n, k)


# --- the kernels and the operators ----------------------------------------------------------


def _kernels(fmt: str, k: int, out: int):
    """The GEMV's begin, block and end calls for fmt at k in-features, out out-features a core."""
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config

    kernels = Path(config.aie_kernels_dir())
    here = Path(__file__).resolve().parent / "kernels"
    blk = np.ndarray[(FORMATS[fmt],), np.dtype[np.uint8]]
    x = np.ndarray[(k,), np.dtype[bfloat16]]
    sums = np.ndarray[(k // GROUP,), np.dtype[bfloat16]]
    acc = np.ndarray[(out,), np.dtype[np.float32]]
    y = np.ndarray[(out,), np.dtype[bfloat16]]
    common = dict(
        source_file=str(here / f"{fmt}_mv.cc"),
        include_dirs=[str(kernels / "flm_gemma4"), str(kernels), config.cxx_header_path()],
        compile_flags=[f"-D{fmt.upper()}_K={k}", f"-DQ_OUT={out}"],
    )
    obj = f"{fmt}_mv_{k}_{out}.o"
    begin = ExternalFunction(f"{fmt}_mv_begin", arg_types=[x, sums, acc], object_file_name=obj, **common)
    block = ExternalFunction(f"{fmt}_mv_block", arg_types=[blk, x, sums, acc, np.int32, np.int32],
                             object_file_name=obj, **common)
    end = ExternalFunction(f"{fmt}_mv_end", arg_types=[acc, y], object_file_name=obj, **common)
    return begin, block, end


def _dequant_kernel(k: int):
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config

    kernels = Path(config.aie_kernels_dir())
    here = Path(__file__).resolve().parent / "kernels"
    return ExternalFunction(
        "q8_dequant",
        source_file=str(here / "q8_mv.cc"),
        arg_types=[np.ndarray[(Q8_BLOCK_BYTES,), np.dtype[np.uint8]],
                   np.ndarray[(CB * K_TILE,), np.dtype[bfloat16]], np.int32],
        include_dirs=[str(kernels / "flm_gemma4"), str(kernels), config.cxx_header_path()],
        compile_flags=[f"-DQ8_K={k}"],
        object_file_name=f"q8_dequant_{k}.o",
    )


def _operators():
    """QGEMV and Q8Dequant, defined on first use: IRON's modules load only in its environment."""
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import Buffer, In, ObjectFifo, Out, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.utils.compile.jit.params import param, tunable, widths
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    class QGEMV(Operator):
        """``y = W x``: W as q4nx or q8 blocks (quantize_fmt), x one row of K, y N.

        Each column takes N / columns out-features (whole column blocks of 64) and all of x.
        With ``cores`` 2 its memtile splits each pair of blocks between two cores, one for the
        column block's out-features 0-31, the other for 32-63, gives both x, and joins their
        outputs.
        """

        fmt: str = param(array=True)
        blocks: int = param(array=True)  # N / 64 * K / 256 * 2
        K: int = param(array=True)
        N: int = param(derive=lambda op: op.blocks * CB // (2 * op.K // K_TILE), array=True)
        cores: int = param(default=2, array=True)
        block_bytes: int = param(derive=lambda op: FORMATS[op.fmt], array=True)
        num_aie_columns: int = tunable(domain=widths)

        W = In(blocks, block_bytes, dtype=np.uint8, obj_shape=(cores, block_bytes), per=(num_aie_columns,))
        x = In(K, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
        y = Out(N, obj_shape=(CB,), per=(num_aie_columns,))

        def validate(self):
            if self.K % 512:
                raise ValueError(f"QGEMV: K ({self.K}) must be a multiple of 512")
            if self.blocks % (2 * self.K // K_TILE):
                raise ValueError(f"QGEMV: {self.blocks} blocks are no whole column blocks of K {self.K}")
            if self.cores not in (1, 2):
                raise ValueError(f"QGEMV: {self.cores} cores a column (1 or 2)")

        def resolve(self, dev):
            cbs = self.N // CB
            cols = self.resolve_columns(dev, self.num_aie_columns, fits=lambda c: cbs % c == 0)
            return self.replace(num_aie_columns=cols)

        def compatible(self):
            if (self.N // CB) % self.num_aie_columns:
                raise ValueError(f"QGEMV: {self.N // CB} column blocks over {self.num_aie_columns} columns")

        def build(self) -> Built:
            K, n, cores, B = self.K, self.num_aie_columns, self.cores, self.block_bytes
            cbs = self.N // CB // n
            kbs = K // K_TILE
            out = CB // cores
            begin, block, end = _kernels(self.fmt, K, out)
            blk_t = np.ndarray[(B,), np.dtype[np.uint8]]
            y_t = np.ndarray[(out,), np.dtype[bfloat16]]
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            x_l3 = [ObjectFifo(self.x.obj_type, depth=1) for _ in range(n)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            if cores == 1:
                w_l1 = [[f] for f in w_l3]
                x_l1 = x_l3
                y_l1 = [[f] for f in y_l3]
            else:
                w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
                x_l1 = [f.cons().forward(obj_type=self.x.obj_type, tile=AnyMemTile) for f in x_l3]
                y_l1 = [f.prod().join([0, out], obj_types=[y_t, y_t], depths=[2, 2], tile=AnyMemTile)
                        for f in y_l3]
            sums = [[Buffer(np.ndarray[(K // GROUP,), np.dtype[bfloat16]]) for _ in range(cores)] for _ in range(n)]
            accs = [[Buffer(np.ndarray[(out,), np.dtype[np.float32]]) for _ in range(cores)] for _ in range(n)]

            def core(w_f, x_f, y_f, sums, acc, begin, block, end):
                x = x_f.acquire(1)
                begin(x, sums, acc)
                for _ in range_(cbs):
                    for kb in range_(kbs):
                        kb32 = index.casts(T.i32(), kb)
                        for h in range(2 if cores == 1 else 1):
                            w = w_f.acquire(1)
                            block(w, x, sums, acc, kb32, h)
                            w_f.release(1)
                    y = y_f.acquire(1)
                    end(acc, y)
                    y_f.release(1)
                x_f.release(1)

            workers = [
                Worker(core, [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), sums[i][j], accs[i][j],
                              begin, block, end])
                for i in range(n) for j in range(cores)
            ]
            self.W.bind([f.prod() for f in w_l3])
            self.x.bind([f.prod() for f in x_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def ops(self) -> int:
            return 2 * self.N * self.K

        def reference(self, W, x):
            w = dequantize_fmt(W, self.N, self.K, self.fmt)
            return (w.astype(np.float64) @ np.asarray(x, np.float64)).astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="32-column partial sums narrowed to bf16 before the scale")

    class Q8Dequant(Operator):
        """q8 blocks to the (N, K) bf16 weight, for a prompt chunk's GEMM: a core a column, each
        with N / columns out-features, writing a column block's 64 rows 256 in-features at a time."""

        blocks: int = param(array=True)
        K: int = param(array=True)
        N: int = param(derive=lambda op: op.blocks * CB // (2 * op.K // K_TILE), array=True)
        num_aie_columns: int = tunable(domain=widths)

        W = In(blocks, Q8_BLOCK_BYTES, dtype=np.uint8, obj_shape=(1, Q8_BLOCK_BYTES), per=(num_aie_columns,))
        w = Out(N, K, obj_shape=(CB, K_TILE), per=(num_aie_columns,))

        def resolve(self, dev):
            cbs = self.N // CB
            cols = self.resolve_columns(dev, self.num_aie_columns, fits=lambda c: cbs % c == 0)
            return self.replace(num_aie_columns=cols)

        def compatible(self):
            if (self.N // CB) % self.num_aie_columns:
                raise ValueError(f"Q8Dequant: {self.N // CB} column blocks over {self.num_aie_columns} columns")

        def build(self) -> Built:
            n = self.num_aie_columns
            cbs, kbs = self.N // CB // n, self.K // K_TILE
            deq = _dequant_kernel(self.K)
            w_f = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            o_f = [ObjectFifo(self.w.obj_type, depth=1) for _ in range(n)]

            def core(w_f, o_f, deq):
                for _ in range_(cbs * kbs):
                    o = o_f.acquire(1)
                    for h in range(2):
                        w = w_f.acquire(1)
                        deq(w, o, h)
                        w_f.release(1)
                    o_f.release(1)

            workers = [Worker(core, [w_f[i].cons(), o_f[i].prod(), deq]) for i in range(n)]
            self.W.bind([f.prod() for f in w_f])
            self.w.bind([f.cons() for f in o_f])
            return Built(workers, [])

        def sequence(self, rt):
            n, N, K = self.num_aie_columns, self.N, self.K
            rows = N // n
            drains = []
            for lane in range(n):
                # the lane's column blocks, 256 in-features at a time, 64 rows of 256 each
                tap = TensorAccessPattern((N, K), lane * rows * K, [rows // CB, K // K_TILE, CB, K_TILE],
                                          [CB * K, K_TILE, K, 1])
                drains.append(self.w.drain(tap, lane=lane, wait=True, managed=False))
            for lane, tap, _, _ in self.W.transfers():
                self.W.fill(tap, lane=lane, managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return self.N * self.K

        def reference(self, W):
            return dequantize8(W, self.N, self.K).astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.01, 1e-3, note="a product rounded to bf16")

    return QGEMV, Q8Dequant


_OPS = None


def operators():
    global _OPS
    if _OPS is None:
        _OPS = _operators()
    return _OPS


def q_linear(x, q, n: int, k: int, fmt: str = "q4", cores: int = 2):
    """``x @ W.T`` for x (k,), (1, k) or (rows, k) and q, W's blocks in fmt as a 1-D iron.param.

    One row runs QGEMV on the blocks. Several (a prompt chunk) dequantize them on the device:
    q4nx into bfp16 (iron.operators.flm.DequantBFP) for flm.ProjectionGEMM, as IRON's Gemma 4
    at Precision.MEDIUM multiplies anyway; q8 into bf16 (Q8Dequant) for IRON's own F.linear.
    """
    QGEMV, Q8Dequant = operators()
    one_row = len(x.shape) == 1 or x.shape[0] == 1
    if one_row:
        y = QGEMV.step(q.reshape(-1, FORMATS[fmt]), x.reshape(k), fmt=fmt, cores=cores)
        return y if len(x.shape) == 1 else y.reshape(1, n)
    from iron import Precision

    if fmt == "q8":
        w = Q8Dequant.step(q.reshape(-1, Q8_BLOCK_BYTES), K=k)
        return _BASE["linear"](x, w, precision=Precision.MEDIUM)
    from iron.operators.flm import DequantBFP, ProjectionGEMM
    from iron.operators.flm.gemm.op import Epilogue

    packed = DequantBFP.step(q, K=k, N=n)
    return ProjectionGEMM.step(x, packed, K=k, N=n, bfp16_b=True, precision=Precision.MEDIUM,
                               epilogue_modes=(Epilogue.NONE,))


def q4_linear(x, q, n: int, k: int):
    return q_linear(x, q, n, k, "q4")


# Quantized weights by id of their iron.param: (the param, out-features, in-features, format).
# The param is held, so no other object takes its id.
_Q4: dict[int, tuple[object, int, int, str]] = {}
_BASE: dict[str, object] = {}


def q_param(w, fmt: str = "q4"):
    """An iron.param of w's blocks in fmt (quantize_fmt), known to the patched F.linear."""
    import iron

    n, k = np.shape(w)
    p = iron.param(quantize_fmt(w, fmt).reshape(-1))
    _Q4[id(p)] = (p, n, k, fmt)
    return p


def q4_param(w):
    return q_param(w, "q4")


def install_linear() -> None:
    """Make iron.functional.linear run q_linear on the weights q_param made."""
    import iron.functional as F

    if getattr(F.linear, "hwnpu_q4", False):
        return
    base = F.linear
    _BASE["linear"] = base

    def linear(x, weight, bias=None, *, precision, **gemv):
        entry = _Q4.get(id(weight))
        if entry is None or entry[0] is not weight:
            return base(x, weight, bias, precision=precision, **gemv)
        y = q_linear(x, weight, entry[1], entry[2], entry[3])
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
    install_linear()  # the base F.linear q8's prompt path calls, and the bfp16 fix
    n, k, fmt = args.n, args.k, args.fmt
    rng = np.random.default_rng(1)
    w = rng.standard_normal((n, k)).astype(np.float32) * 0.05
    q = quantize_fmt(w, fmt)
    deq = dequantize_fmt(q, n, k, fmt)
    err = np.abs(deq - w).max() / np.abs(w).max()
    print(f"quantization ({fmt}): largest error {err:.4f} of the largest weight")

    from iron import Scratchpad

    class G(iron.Graph):
        def __init__(self):
            self.q = iron.param(q.reshape(-1))

        def body(self, x, *, rows: Scratchpad[np.int32] = np.int32(0)):
            if args.rows:  # a prompt chunk's: bounded per call
                return q_linear(x[:rows], self.q, n, k, fmt, args.cores)
            return q_linear(x, self.q, n, k, fmt, args.cores)

        def reference(self, x, rows=0):
            x = np.asarray(x, np.float64)
            y = (x @ deq.astype(np.float64).T).astype(bfloat16)
            if rows:
                y[rows:] = 0
            return y

    g = G()
    shape = (k,) if args.rows == 0 else (args.rows, k)
    cg = g.compile(x=shape)
    m = export_graph(cg, args.out, "qlinear", dict(n=n, k=k, fmt=fmt))
    x = rng.standard_normal(shape).astype(bfloat16)
    write_check(g, m, args.out, dict(x=x), dict(rows=args.valid or args.rows) if args.rows else None)
    print(f"{args.out}: {fmt} {n} x {k}, {q.nbytes >> 10} KiB of weights")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of one quantized linear and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--n", type=int, default=1536)
    c.add_argument("--k", type=int, default=1536)
    c.add_argument("--fmt", choices=sorted(FORMATS), default="q4")
    c.add_argument("--cores", type=int, default=2, help="cores a column of the GEMV (1 or 2)")
    c.add_argument("--rows", type=int, default=0, help="rows of x (0: one, (k,)): several run the prefill path")
    c.add_argument("--valid", type=int, default=0, help="rows the call bounds x to (default all)")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return _check(a)
    return 1


if __name__ == "__main__":
    sys.exit(main())
