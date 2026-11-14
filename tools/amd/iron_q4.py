"""4-bit weights (q4nx) for IRON graphs: the quantizer, the decode GEMV and its check.

q4nx is FastFlowLM's format (mlir-aie's flm_gemma4/q4nx.h, IRON's iron.operators.flm.q4nx):
a block holds 32 out-features by 256 in-features, a bf16 scale and minimum for each 32
in-features of an out-feature, then the 4-bit codes; a weight is ``min + scale * code``.
5 bits a weight with the scales, 31 % of bf16.

The blocks of a weight (N out-features, K in-features) are in the order
iron.operators.flm.DequantBFP reads (its README, layers 6 to 9): column blocks of 64
out-features, each K / 512 objects of two pairs of blocks (in-features 0-255, then 256-511),
a pair's blocks the out-features 0-31, then 32-63. A column block is contiguous, so the GEMV's
cores each read a run of whole column blocks, and prefill dequantizes the same bytes for
the GEMM (DequantBFP into bfp16, or QDequant into bf16: q_linear).

The kernels set the cores' rounding to nearest even: a core starts in floor mode, where every
narrowing to bf16 or bfp16 rounds toward negative infinity, and over K in-features of inputs
with a mean that bias grows to several times the quantization's own error (DequantBFP's).

Q4GEMV is ``y = W x`` for one row: one core a column, each with the column blocks of
``N / columns`` out-features and all of x. Its kernel (kernels/q4_mv.cc) is FastFlowLM's
q4nx_accumulate.

    python tools/amd/iron_q4.py check OUT --n 1536 --k 1536   # a graph of one Q4GEMV, with check/
    go run ./tools/amd/irongraph OUT
"""

import argparse
import hashlib
import json
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
    return pack(codes.reshape(N, K), scale, mn)


def pack(codes, scale, mn) -> np.ndarray:
    """q4nx blocks of codes (N, K) 0..15 with each group's scale and minimum (N, K / 32), as
    quantize writes them: a quantizer of its own (GPTQ) chooses them, this lays them out."""
    codes = np.asarray(codes, np.uint8)
    scale, mn = np.asarray(scale, np.float32), np.asarray(mn, np.float32)
    N, K = codes.shape
    if N % CB or K % 512:
        raise ValueError(f"q4nx: ({N}, {K}) is not 64 x 512 blocks")
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
# A q6 block: q8's scales, then its codes (-31..31, stored as code + 32) in q8's order, in two
# planes: the low 4 bits as q4nx's nibbles, then the high 2 bits, 32 bytes for each 128 codes
# (code 32 j + i of the 128 in bits 2 j and 2 j + 1 of byte i). 6.5 bits a weight.
Q6_BLOCK_BYTES = 2 * (M_TILE * K_TILE // GROUP) + M_TILE * K_TILE * 3 // 4  # 6656
Q6_UNIT = 128  # codes a high-bit run covers (kernels/q6_mv.cc's load)
FORMATS = {"q4": BLOCK_BYTES, "q6": Q6_BLOCK_BYTES, "q8": Q8_BLOCK_BYTES}


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
    return pack8(codes, scale)


def pack8(codes, scale) -> np.ndarray:
    """q8 blocks of codes (N, K) int8 with each group's scale (N, K / 32), as quantize8 writes them."""
    codes = np.asarray(codes, np.int8)
    scale = np.asarray(scale, np.float32)
    N, K = codes.shape
    if N % CB or K % 512:
        raise ValueError(f"q8: ({N}, {K}) is not 64 x 512 blocks")
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


# --- q6: 6.5 bits a weight -------------------------------------------------------------------


def quantize6(w) -> np.ndarray:
    """q6 of w (N, K) in q4nx's block order: uint8 (blocks, Q6_BLOCK_BYTES). Each group of 32
    in-features of an out-feature: scale = its largest magnitude / 31 in bf16, codes rounded."""
    w = np.asarray(w, np.float32)
    N, K = w.shape
    if N % CB or K % 512:
        raise ValueError(f"q6: ({N}, {K}) is not 64 x 512 blocks")
    g = w.reshape(N, K // GROUP, GROUP)
    scale = (np.abs(g).max(-1) / 31).astype(bfloat16).astype(np.float32)
    safe = np.where(scale > 0, scale, 1)
    codes = np.clip(np.rint(g / safe[..., None]), -31, 31).astype(np.int8).reshape(N, K)
    return pack6(codes, scale)


def pack6(codes, scale) -> np.ndarray:
    """q6 blocks of codes (N, K) -31..31 with each group's scale (N, K / 32)."""
    codes = np.asarray(codes, np.int8)
    scale = np.asarray(scale, np.float32)
    N, K = codes.shape
    if N % CB or K % 512:
        raise ValueError(f"q6: ({N}, {K}) is not 64 x 512 blocks")
    nb, kb = N // M_TILE, K // K_TILE
    sc = scale.reshape(nb, M_TILE, kb, K_TILE // GROUP).transpose(0, 2, 3, 1).astype(bfloat16)
    u = (codes.astype(np.int16) + 32).astype(np.uint8)
    c = u.reshape(nb, M_TILE // 16, 16, kb, K_TILE).transpose(0, 3, 1, 4, 2).reshape(nb, kb, -1)
    lo = c & 15
    lo = (lo[..., 0::2] | (lo[..., 1::2] << 4)).astype(np.uint8)
    h = (c >> 4).reshape(nb, kb, -1, 4, Q6_UNIT // 4)
    hi = (h[..., 0, :] | (h[..., 1, :] << 2) | (h[..., 2, :] << 4) | (h[..., 3, :] << 6)).astype(np.uint8)
    blocks = np.concatenate([sc.reshape(nb, kb, -1).view(np.uint8), lo, hi.reshape(nb, kb, -1)], axis=-1)
    blocks = blocks.reshape(N // CB, 2, K // 512, 2, Q6_BLOCK_BYTES).transpose(0, 2, 3, 1, 4)
    return np.ascontiguousarray(blocks.reshape(-1, Q6_BLOCK_BYTES))


def dequantize6(q, n: int, k: int) -> np.ndarray:
    """The (n, k) float32 weights of quantize6's blocks."""
    q = np.asarray(q, np.uint8).reshape(n // CB, k // 512, 2, 2, Q6_BLOCK_BYTES)
    q = q.transpose(0, 3, 1, 2, 4).reshape(n // M_TILE, k // K_TILE, Q6_BLOCK_BYTES)
    sm = M_TILE * K_TILE // GROUP
    nc = M_TILE * K_TILE
    s = q[..., : 2 * sm].copy().view(bfloat16).astype(np.float32).reshape(n // M_TILE, k // K_TILE, -1, M_TILE)
    lo = q[..., 2 * sm : 2 * sm + nc // 2]
    hi = q[..., 2 * sm + nc // 2 :].reshape(*q.shape[:2], -1, Q6_UNIT // 4)
    c = np.empty((*q.shape[:2], nc), np.uint8)
    c[..., 0::2], c[..., 1::2] = lo & 15, lo >> 4
    h = np.stack([(hi >> (2 * j)) & 3 for j in range(4)], axis=-2).reshape(*q.shape[:2], nc)
    c = (c | (h << 4)).astype(np.int16) - 32
    c = c.reshape(n // M_TILE, k // K_TILE, M_TILE // 16, K_TILE, 16).transpose(0, 2, 4, 1, 3).reshape(n, k)
    s = s.transpose(0, 3, 1, 2).reshape(n, k // GROUP)
    return c.astype(np.float32) * np.repeat(s, GROUP, 1)


def weight_key(w) -> str:
    """A weight's name among chosen codes (load_codes): a hash of its shape and bf16 values, the
    same whichever program loaded it from the checkpoint."""
    b = np.ascontiguousarray(np.asarray(w, np.float32).astype(bfloat16))
    return hashlib.sha1(repr(b.shape).encode() + b.view(np.uint16).tobytes()).hexdigest()


# Codes another quantizer chose (q4_gptq.py's GPTQ): by weight_key, (format, the .npy path).
_CODES: dict[str, tuple[str, Path]] = {}
CODES_USED = {"hit": 0, "miss": 0}


def load_codes(path) -> int:
    """Make quantize_fmt take the blocks q4_gptq.py wrote into path (index.json and a .npy a
    weight) for the weights they were chosen for, instead of rounding them. Returns how many."""
    path = Path(path)
    index = json.loads((path / "index.json").read_text())
    for k, e in index.items():
        _CODES[k] = (e["fmt"], path / f"{k}.npy")
    return len(index)


def quantize_fmt(w, fmt: str) -> np.ndarray:
    if _CODES:
        hit = _CODES.get(weight_key(w))
        if hit is not None and hit[0] == fmt:
            CODES_USED["hit"] += 1
            q = np.load(hit[1])
            n, k = np.shape(w)
            if q.shape != (n * k // (M_TILE * K_TILE), FORMATS[fmt]):
                raise ValueError(f"codes {hit[1]}: {q.shape} for a ({n}, {k}) weight in {fmt}")
            return q
        CODES_USED["miss"] += 1
    return {"q4": quantize, "q6": quantize6, "q8": quantize8}[fmt](w)


def dequantize_fmt(q, n: int, k: int, fmt: str) -> np.ndarray:
    return {"q4": dequantize, "q6": dequantize6, "q8": dequantize8}[fmt](q, n, k)


# --- the kernels and the operators ----------------------------------------------------------


def _kernels(fmt: str, k: int, out: int, rows: int = 0):
    """The GEMV's begin, block and end calls for fmt at k in-features, out out-features a core.
    rows > 0: the calls of a GEMV of that many rows (``{fmt}_mvr_*``, q8 and q6)."""
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config

    kernels = Path(config.aie_kernels_dir())
    here = Path(__file__).resolve().parent / "kernels"
    blk = np.ndarray[(FORMATS[fmt],), np.dtype[np.uint8]]
    r = max(rows, 1)
    x = np.ndarray[(rows * K_TILE if rows else k,), np.dtype[bfloat16]]  # rows: x a k-block of every row
    sums = np.ndarray[(k // GROUP,), np.dtype[bfloat16]]
    acc = np.ndarray[(r * (CB if rows else out),), np.dtype[np.float32]]
    y = np.ndarray[(r * out,), np.dtype[bfloat16]]
    flags = [f"-D{fmt.upper()}_K={k}", f"-DQ_OUT={out}"] + ([f"-DQ_ROWS={rows}"] if rows else [])
    common = dict(
        source_file=str(here / f"{fmt}_mv.cc"),
        include_dirs=[str(kernels / "flm_gemma4"), str(kernels), config.cxx_header_path()],
        compile_flags=flags,
    )
    name = f"{fmt}_mvr" if rows else f"{fmt}_mv"
    obj = f"{fmt}_mvr{rows}_{k}_{out}.o" if rows else f"{fmt}_mv_{k}_{out}.o"
    begin = ExternalFunction(f"{name}_begin", arg_types=[x, sums, acc], object_file_name=obj, **common)
    block = ExternalFunction(f"{name}_block", arg_types=[blk, x, sums, acc, np.int32, np.int32],
                             object_file_name=obj, **common)
    end = ExternalFunction(f"{name}_end", arg_types=[acc, y], object_file_name=obj, **common)
    return begin, block, end


def _dequant_kernel(fmt: str, k: int):
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config

    kernels = Path(config.aie_kernels_dir())
    here = Path(__file__).resolve().parent / "kernels"
    return ExternalFunction(
        f"{fmt}_dequant",
        source_file=str(here / f"{fmt}_mv.cc"),
        arg_types=[np.ndarray[(FORMATS[fmt],), np.dtype[np.uint8]],
                   np.ndarray[(CB * K_TILE,), np.dtype[bfloat16]], np.int32],
        include_dirs=[str(kernels / "flm_gemma4"), str(kernels), config.cxx_header_path()],
        compile_flags=[f"-D{fmt.upper()}_K={k}"],
        object_file_name=f"{fmt}_dequant_{k}.o",
    )


def _operators():
    """QGEMV and QDequant, defined on first use: IRON's modules load only in its environment."""
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

    class QDequant(Operator):
        """q4nx or q8 blocks to the (N, K) bf16 weight (rounded to nearest even), for a prompt
        chunk's GEMM: a core a column, each with N / columns out-features, writing a column
        block's 64 rows 256 in-features at a time."""

        fmt: str = param(array=True)
        blocks: int = param(array=True)
        K: int = param(array=True)
        N: int = param(derive=lambda op: op.blocks * CB // (2 * op.K // K_TILE), array=True)
        block_bytes: int = param(derive=lambda op: FORMATS[op.fmt], array=True)
        num_aie_columns: int = tunable(domain=widths)

        W = In(blocks, block_bytes, dtype=np.uint8, obj_shape=(1, block_bytes), per=(num_aie_columns,))
        w = Out(N, K, obj_shape=(CB, K_TILE), per=(num_aie_columns,))

        def resolve(self, dev):
            cbs = self.N // CB
            cols = self.resolve_columns(dev, self.num_aie_columns, fits=lambda c: cbs % c == 0)
            return self.replace(num_aie_columns=cols)

        def compatible(self):
            if (self.N // CB) % self.num_aie_columns:
                raise ValueError(f"QDequant: {self.N // CB} column blocks over {self.num_aie_columns} columns")

        def build(self) -> Built:
            n = self.num_aie_columns
            cbs, kbs = self.N // CB // n, self.K // K_TILE
            deq = _dequant_kernel(self.fmt, self.K)
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
            return dequantize_fmt(W, self.N, self.K, self.fmt).astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.01, 1e-3, note="a product rounded to bf16")

    return QGEMV, QDequant


_ADD_OPS: dict = {}


def add_operator(adds: int):
    """QGEMVAdd: QGEMV whose output has ``adds`` vectors (1 or 2: a bias, a residual) added as it
    leaves the cores (kernels/mv_add.h), ``y = W x + a1 (+ a2)``: one operator in place of the GEMV
    and an add a vector. Each vector's slice of a column's out-features rides on x's channel after
    x (a core has two input channels), x's object K + adds * N / columns long."""
    if adds in _ADD_OPS:
        return _ADD_OPS[adds]
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import Buffer, In, ObjectFifo, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param

    QGEMV, _ = operators()

    # The vectors' objects are the column's slices; the sequence fills them itself (their
    # declared object a column block's, a shape known before the columns are)
    body = dict(
        adds=param(default=adds, array=True),
        a1=In(QGEMV.N, obj_shape=(CB,), per=(QGEMV.num_aie_columns,)),
    )
    if adds == 2:
        body["a2"] = In(QGEMV.N, obj_shape=(CB,), per=(QGEMV.num_aie_columns,))
    body["__annotations__"] = {"adds": int}

    def build(self):
        K, n, cores, B = self.K, self.num_aie_columns, self.cores, self.block_bytes
        cbs = self.N // CB // n
        kbs = K // K_TILE
        out = CB // cores
        R = self.N // n
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        xa_t = np.ndarray[(K + adds * R,), np.dtype[bfloat16]]
        y_t = np.ndarray[(out,), np.dtype[bfloat16]]
        acc_t = np.ndarray[(out,), np.dtype[np.float32]]
        blk_t = np.ndarray[(B,), np.dtype[np.uint8]]
        sums_t = np.ndarray[(K // GROUP,), np.dtype[bfloat16]]
        # one object of the three calls (begin and block read x's first K)
        common = dict(source_file=str(here / f"{self.fmt}_mv.cc"),
                      include_dirs=[str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
                      compile_flags=[f"-D{self.fmt.upper()}_K={K}", f"-DQ_OUT={out}", f"-DQ_ADDS={adds}",
                                     f"-DQ_R={R}"],
                      object_file_name=f"{self.fmt}_mv_add{adds}_{K}_{out}_{R}.o")
        begin = ExternalFunction(f"{self.fmt}_mv_begin", arg_types=[xa_t, sums_t, acc_t], **common)
        block = ExternalFunction(f"{self.fmt}_mv_block", arg_types=[blk_t, xa_t, sums_t, acc_t, np.int32, np.int32],
                                 **common)
        end_add = ExternalFunction(f"{self.fmt}_mv_end_add", arg_types=[acc_t, xa_t, np.int32, np.int32, y_t],
                                   **common)
        w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
        x_l3 = [ObjectFifo(xa_t, depth=1) for _ in range(n)]
        y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
        if cores == 1:
            w_l1, x_l1, y_l1 = [[f] for f in w_l3], x_l3, [[f] for f in y_l3]
        else:
            w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
            x_l1 = [f.cons().forward(obj_type=xa_t, tile=AnyMemTile) for f in x_l3]
            y_l1 = [f.prod().join([0, out], obj_types=[y_t, y_t], depths=[2, 2], tile=AnyMemTile) for f in y_l3]
        sums = [[Buffer(sums_t) for _ in range(cores)] for _ in range(n)]
        accs = [[Buffer(acc_t) for _ in range(cores)] for _ in range(n)]

        def core(w_f, x_f, y_f, sums, acc, begin, block, end_add, half):
            x = x_f.acquire(1)
            begin(x, sums, acc)
            for cb in range_(cbs):
                cb32 = index.casts(T.i32(), cb)
                for kb in range_(kbs):
                    kb32 = index.casts(T.i32(), kb)
                    for h in range(2 if cores == 1 else 1):
                        w = w_f.acquire(1)
                        block(w, x, sums, acc, kb32, h)
                        w_f.release(1)
                y = y_f.acquire(1)
                end_add(acc, x, cb32, half, y)
                y_f.release(1)
            x_f.release(1)

        workers = [
            Worker(lambda w_f, x_f, y_f, s, a, b, bl, e, half=(j if cores > 1 else 0): core(
                w_f, x_f, y_f, s, a, b, bl, e, half),
                   [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), sums[i][j], accs[i][j],
                    begin, block, end_add])
            for i in range(n) for j in range(cores)
        ]
        self.W.bind([f.prod() for f in w_l3])
        for name in ("x", "a1", "a2")[: 2 + (adds == 2)]:
            getattr(self, name).bind([f.prod() for f in x_l3])
        self.y.bind([f.cons() for f in y_l3])
        from iron.common import Built

        return Built(workers, [])

    def sequence(self, rt):
        n, N, K = self.num_aie_columns, self.N, self.K
        R = N // n
        drains = [d for d in (self.y.drain(tap, lane=lane, wait=True, managed=False)
                              for lane, tap, _, _ in self.y.transfers())]
        for lane in range(n):
            self.x.fill(TensorAccessPattern((K,), 0, [K], [1]), lane=lane, managed=False)
            for name in ("a1", "a2")[:adds]:
                getattr(self, name).fill(TensorAccessPattern((N,), lane * R, [R], [1]), lane=lane, managed=False)
        for lane, tap, _, _ in self.W.transfers():
            self.W.fill(tap, lane=lane, managed=False)
        for d in drains:
            d.await_()

    def reference(self, W, x, a1, a2=None):
        w = dequantize_fmt(W, self.N, self.K, self.fmt)
        y = w.astype(np.float64) @ np.asarray(x, np.float64) + np.asarray(a1, np.float64).reshape(-1)
        if a2 is not None:
            y = y + np.asarray(a2, np.float64).reshape(-1)
        return y.astype(bfloat16)

    body.update(build=build, sequence=sequence, reference=reference,
                __doc__=add_operator.__doc__)
    cls = type("QGEMV", (QGEMV,), body)
    _ADD_OPS[adds] = cls
    return cls


_QKV_OPS = {}


def qkv_operator():
    """QGEMVQKV: a fused q, k, v projection for one row (kernels/mv_add.h's Q_QKV), ``y =
    rope(s W x + b)``: the input RMSNorm folded in (its weight in W's columns, s = 1 / sqrt(mean
    (x^2) + eps) from x on each core), the bias added and the first ``rope_rows`` rows (q's and
    k's, their table in interleaved pairs) turned by RoPE with the call's angles (cos and sin
    interleaved, a head's 64), which ride on x's channel after the bias. One operator for the
    norm, the GEMV, the bias and RoPE."""
    if "op" in _QKV_OPS:
        return _QKV_OPS["op"]
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import Buffer, In, ObjectFifo, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param

    Add = add_operator(1)
    QGEMV, _ = operators()
    D = 64  # a head's angles: the rows of a column block

    body = dict(
        rope_rows=param(array=True),
        eps=param(array=True),
        ang=In(D, obj_shape=(D,), per=(QGEMV.num_aie_columns,), replicate=True),
        gw=In(QGEMV.K, obj_shape=(QGEMV.K,), per=(QGEMV.num_aie_columns,), replicate=True),
    )
    body["__annotations__"] = {"rope_rows": int, "eps": float}

    def build(self):
        K, n, cores, B = self.K, self.num_aie_columns, self.cores, self.block_bytes
        if cores != 2:
            raise ValueError("QGEMVQKV: two cores a column (a core's 32 out-features of a head)")
        cbs, kbs, out, R = self.N // CB // n, K // K_TILE, CB // cores, self.N // n
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        xa_t = np.ndarray[(K + R + D + K,), np.dtype[bfloat16]]  # x, the bias, the angles, the norm's weight
        y_t = np.ndarray[(out,), np.dtype[bfloat16]]
        acc_t = np.ndarray[(out,), np.dtype[np.float32]]
        blk_t = np.ndarray[(B,), np.dtype[np.uint8]]
        sums_t = np.ndarray[(K // GROUP,), np.dtype[bfloat16]]
        flags = [f"-D{self.fmt.upper()}_K={K}", f"-DQ_OUT={out}", "-DQ_QKV=1", f"-DQ_R={R}",
                 f"-DQ_ROPE_ROWS={self.rope_rows}", f"-DQ_EPS={float(self.eps)!r}f"]
        common = dict(source_file=str(here / f"{self.fmt}_mv.cc"),
                      include_dirs=[str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
                      compile_flags=flags,
                      object_file_name=f"{self.fmt}_mv_qkvg_{K}_{out}_{R}_{self.rope_rows}_{abs(hash(self.eps)) % 10**8}.o")
        begin = ExternalFunction(f"{self.fmt}_mv_begin_norm", arg_types=[xa_t, sums_t, acc_t], **common)
        block = ExternalFunction(f"{self.fmt}_mv_block", arg_types=[blk_t, xa_t, sums_t, acc_t, np.int32, np.int32],
                                 **common)
        end = ExternalFunction(f"{self.fmt}_mv_end_qkv",
                               arg_types=[acc_t, xa_t, np.int32, np.int32, np.int32, y_t], **common)
        w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
        x_l3 = [ObjectFifo(xa_t, depth=1) for _ in range(n)]
        y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
        w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
        x_l1 = [f.cons().forward(obj_type=xa_t, tile=AnyMemTile) for f in x_l3]
        y_l1 = [f.prod().join([0, out], obj_types=[y_t, y_t], depths=[2, 2], tile=AnyMemTile) for f in y_l3]

        def core(w_f, x_f, y_f, sums, acc, begin, block, end, half, lane):
            x = x_f.acquire(1)
            begin(x, sums, acc)
            for cb in range_(cbs):
                cb32 = index.casts(T.i32(), cb)
                for kb in range_(kbs):
                    kb32 = index.casts(T.i32(), kb)
                    w = w_f.acquire(1)
                    block(w, x, sums, acc, kb32, 0)  # a core's own half of the block (the split's)
                    w_f.release(1)
                y = y_f.acquire(1)
                end(acc, x, cb32, half, lane, y)
                y_f.release(1)
            x_f.release(1)

        workers = [
            Worker(lambda w_f, x_f, y_f, sm, a, b, bl, e, half=j, lane=i: core(w_f, x_f, y_f, sm, a, b, bl, e, half, lane),
                   [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), Buffer(sums_t), Buffer(acc_t),
                    begin, block, end])
            for i in range(n) for j in range(cores)
        ]
        self.W.bind([f.prod() for f in w_l3])
        for name in ("x", "a1", "ang", "gw"):
            getattr(self, name).bind([f.prod() for f in x_l3])
        self.y.bind([f.cons() for f in y_l3])
        from iron.common import Built

        return Built(workers, [])

    def sequence(self, rt):
        n, N, K = self.num_aie_columns, self.N, self.K
        R = N // n
        drains = [self.y.drain(tap, lane=lane, wait=True, managed=False) for lane, tap, _, _ in self.y.transfers()]
        for lane in range(n):
            self.x.fill(TensorAccessPattern((K,), 0, [K], [1]), lane=lane, managed=False)
            self.a1.fill(TensorAccessPattern((N,), lane * R, [R], [1]), lane=lane, managed=False)
            self.ang.fill(TensorAccessPattern((D,), 0, [D], [1]), lane=lane, managed=False)
            self.gw.fill(TensorAccessPattern((K,), 0, [K], [1]), lane=lane, managed=False)
        for lane, tap, _, _ in self.W.transfers():
            self.W.fill(tap, lane=lane, managed=False)
        for d in drains:
            d.await_()

    def reference(self, W, x, a1, ang, gw):
        w = dequantize_fmt(W, self.N, self.K, self.fmt).astype(np.float64)
        xv = np.asarray(x, np.float64)
        s = 1 / np.sqrt(np.mean(xv * xv) + float(self.eps))
        xn = (xv * np.asarray(gw, np.float64) * s).astype(bfloat16).astype(np.float64)
        y = w @ xn + np.asarray(a1, np.float64).reshape(-1)
        t = np.asarray(ang, np.float64).reshape(-1)
        c, sn = t[0::2], t[1::2]
        r = y[: self.rope_rows].reshape(-1, D // 2, 2)
        a, b = r[..., 0].copy(), r[..., 1].copy()
        r[..., 0], r[..., 1] = a * c - b * sn, a * sn + b * c
        y[: self.rope_rows] = r.reshape(-1)
        return y.astype(bfloat16)

    body.update(build=build, sequence=sequence, reference=reference, __doc__=qkv_operator.__doc__)
    cls = type("QGEMV", (Add,), body)
    _QKV_OPS["op"] = cls
    return cls


def qkv_linear(x, weight, bias, angles, norm_weight, rope_rows: int, eps: float):
    """QGEMVQKV on one row of x (the layer's input, not normed), the norm's weight and a q_param
    weight (q and k in interleaved pairs)."""
    entry = _Q4.get(id(weight))
    if entry is None or entry[0] is not weight:
        raise ValueError("qkv_linear: the weight is not a q_param")
    _, n, k, fmt = entry
    op = qkv_operator()
    given = dict(W=weight.reshape(-1, FORMATS[fmt]), x=x.reshape(k), a1=bias.reshape(n), ang=angles.reshape(64),
                 gw=norm_weight.reshape(k))
    names = [m.name for m in op.declaration.members if m.name in given]
    return op.step(*(given[nm] for nm in names), fmt=fmt, cores=2, rope_rows=rope_rows, eps=eps)


def _rows_operator():
    """QGEMVRows: QGEMV's ``Y = X W.T`` for a few rows of X (a call checking drafts), q8 or q6,
    one or two cores a column as QGEMV's. The weights stream once and multiply every row
    (kernels/mv_rows.h); each core's column blocks come out a block's rows at a time, which the
    drains put in Y's rows."""
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.iron import Buffer, ObjectFifo, Worker
    from aie.iron.controlflow import range_
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import In, Out
    from aie.iron.device import AnyMemTile
    from aie.utils.compile.jit.params import param, tunable, widths
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    class QGEMVRows(Operator):
        fmt: str = param(array=True)
        blocks: int = param(array=True)
        K: int = param(array=True)
        rows: int = param(array=True)
        cores: int = param(default=1, array=True)
        N: int = param(derive=lambda op: op.blocks * CB // (2 * op.K // K_TILE), array=True)
        block_bytes: int = param(derive=lambda op: FORMATS[op.fmt], array=True)
        num_aie_columns: int = tunable(domain=widths)

        W = In(blocks, block_bytes, dtype=np.uint8, obj_shape=(cores, block_bytes), per=(num_aie_columns,))
        RK: int = param(derive=lambda op: op.rows * op.K, array=True)
        RKT: int = param(derive=lambda op: op.rows * K_TILE, array=True)
        RCB: int = param(derive=lambda op: op.rows * CB, array=True)

        x = In(RK, obj_shape=(RKT,), per=(num_aie_columns,), replicate=True)
        y = Out(rows, N, obj_shape=(RCB,), per=(num_aie_columns,))

        def validate(self):
            if self.K % 512:
                raise ValueError(f"QGEMVRows: K ({self.K}) must be a multiple of 512")
            if self.fmt not in ("q8", "q6"):
                raise ValueError(f"QGEMVRows: {self.fmt} (q8 and q6 only)")
            if self.cores not in (1, 2):
                raise ValueError(f"QGEMVRows: {self.cores} cores a column (1 or 2)")

        def resolve(self, dev):
            cbs = self.N // CB
            cols = self.resolve_columns(dev, self.num_aie_columns, fits=lambda c: cbs % c == 0)
            return self.replace(num_aie_columns=cols)

        def compatible(self):
            if (self.N // CB) % self.num_aie_columns:
                raise ValueError(f"QGEMVRows: {self.N // CB} column blocks over {self.num_aie_columns} columns")

        def build(self) -> Built:
            K, n, R, cores, B = self.K, self.num_aie_columns, self.rows, self.cores, self.block_bytes
            cbs, kbs = self.N // CB // n, K // K_TILE
            out = CB // cores
            begin, block, end = _kernels(self.fmt, K, out, rows=R)
            blk_t = np.ndarray[(B,), np.dtype[np.uint8]]
            y_t = np.ndarray[(R * out,), np.dtype[bfloat16]]
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            x_l3 = [ObjectFifo(self.x.obj_type, depth=1) for _ in range(n)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            if cores == 1:
                w_l1, x_l1, y_l1 = [[f] for f in w_l3], x_l3, [[f] for f in y_l3]
            else:  # a core the column block's out-features 0-31, the other 32-63: Y (N / 64, 2, rows, 32)
                w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
                x_l1 = [f.cons().forward(obj_type=self.x.obj_type, tile=AnyMemTile) for f in x_l3]
                y_l1 = [f.prod().join([0, R * out], obj_types=[y_t, y_t], depths=[2, 2], tile=AnyMemTile)
                        for f in y_l3]
            sums = [[Buffer(np.ndarray[(K // GROUP,), np.dtype[bfloat16]]) for _ in range(cores)] for _ in range(n)]
            accs = [[Buffer(np.ndarray[(R * CB,), np.dtype[np.float32]]) for _ in range(cores)] for _ in range(n)]

            def core(w_f, x_f, y_f, sums, acc, begin, block, end):
                # x a k-block of every row at a time, again for each column block
                for _ in range_(cbs):
                    for kb in range_(kbs):
                        kb32 = index.casts(T.i32(), kb)
                        x = x_f.acquire(1)
                        for h in range(2 if cores == 1 else 1):
                            w = w_f.acquire(1)
                            block(w, x, sums, acc, kb32, h)
                            w_f.release(1)
                        x_f.release(1)
                    y = y_f.acquire(1)
                    end(acc, y)
                    y_f.release(1)

            workers = [Worker(core, [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), sums[i][j], accs[i][j],
                                     begin, block, end])
                       for i in range(n) for j in range(cores)]
            self.W.bind([f.prod() for f in w_l3])
            self.x.bind([f.prod() for f in x_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            n, N, R = self.num_aie_columns, self.N, self.rows
            cbs = N // CB // n
            drains = []
            for lane in range(n):
                # Y (rows, N): the lane's column blocks, each its rows in turn (a core's 32
                # out-features of each, then the other's, on two)
                if self.cores == 1:
                    dims, strides = [1, cbs, R, CB], [0, CB, N, 1]
                else:
                    dims, strides = [cbs, 2, R, CB // 2], [CB, CB // 2, N, 1]
                tap = TensorAccessPattern((R, N), lane * cbs * CB, dims, strides)
                drains.append(self.y.drain(tap, lane=lane, wait=True, managed=False))
            K, kbs = self.K, self.K // K_TILE
            for lane in range(n):
                # every row's k-block in turn, the whole of x again for each column block
                self.x.fill(TensorAccessPattern((R, K), 0, [cbs, kbs, R, K_TILE], [0, K_TILE, K, 1]),
                            lane=lane, managed=False)
            for lane, tap, _, _ in self.W.transfers():
                self.W.fill(tap, lane=lane, managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * self.N * self.K * self.rows

        def reference(self, W, x):
            w = dequantize_fmt(W, self.N, self.K, self.fmt).astype(np.float64)
            return (np.asarray(x, np.float64).reshape(self.rows, self.K) @ w.T).astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="32-column partial sums narrowed to bf16 before the scale")

    return QGEMVRows


_ROWS_OP = None


def rows_operator():
    global _ROWS_OP
    if _ROWS_OP is None:
        _ROWS_OP = _rows_operator()
    return _ROWS_OP


_OPS = None


def operators():
    global _OPS
    if _OPS is None:
        _OPS = _operators()
    return _OPS


def q_linear(x, q, n: int, k: int, fmt: str = "q4", cores: int = 2, adds=()):
    """``x @ W.T`` for x (k,), (1, k) or (rows, k) and q, W's blocks in fmt as a 1-D iron.param.

    One row runs QGEMV on the blocks. Several (a prompt chunk) dequantize them on the device:
    q8 into bf16 (QDequant) for IRON's own F.linear at Precision.MEDIUM; q4nx by PREFILL["q4"],
    "bfp16" (the default) IRON's flm.DequantBFP straight into flm.ProjectionGEMM's bfp16, or
    "bf16" QDequant as q8. DequantBFP's cores round toward negative infinity, a bias that adds
    up over the in-features when the inputs have a mean: Anima's DiT 0.24 of the velocity
    against 0.09 for the quantization alone, so iron_anima.py takes "bf16". Language models'
    inputs (normalized, about zero mean) show no difference (KL from bf16 the same on Gemma 4
    E2B and LFM3), and "bf16" costs their prompt chunk 5 to 9 % (LFM3 2048: 628 -> 684 ms).
    """
    QGEMV, QDequant = operators()
    one_row = len(x.shape) == 1 or x.shape[0] == 1
    if one_row:
        if adds:  # vectors added as the output leaves (QGEMVAdd: a bias, a residual)
            op = add_operator(len(adds))
            # in the operator's own order of operands (the subclass's vectors come first)
            given = dict(W=q.reshape(-1, FORMATS[fmt]), x=x.reshape(k),
                         **{f"a{i + 1}": a.reshape(n) for i, a in enumerate(adds)})
            names = [m.name for m in op.declaration.members if m.name in given]
            y = op.step(*(given[nm] for nm in names), fmt=fmt, cores=cores)
        else:
            y = QGEMV.step(q.reshape(-1, FORMATS[fmt]), x.reshape(k), fmt=fmt, cores=cores)
        return y if len(x.shape) == 1 else y.reshape(1, n)
    if adds:
        raise ValueError("q_linear: vectors to add are for one row (a prompt chunk adds them itself)")
    from iron import Precision

    if fmt == "bfp16":
        return bfp16_linear(x, q, n, k)
    if x.shape[0] <= ROWS_GEMV and fmt in ("q6", "q8"):
        # a few rows (a call checking drafts): the weights stream once through all of them
        R = x.shape[0]
        return rows_operator().step(q.reshape(-1, FORMATS[fmt]), x.reshape(R * k), fmt=fmt, K=k, rows=R,
                                    cores=cores)
    if fmt in ("q6", "q8") or PREFILL["q4"] == "bf16":
        w = QDequant.step(q.reshape(-1, FORMATS[fmt]), fmt=fmt, K=k)
        return _BASE["linear"](x, w, precision=Precision.MEDIUM)
    from iron.operators.flm import DequantBFP, ProjectionGEMM
    from iron.operators.flm.gemm.op import Epilogue

    packed = DequantBFP.step(q, K=k, N=n)
    return ProjectionGEMM.step(x, packed, K=k, N=n, bfp16_b=True, precision=Precision.MEDIUM,
                               epilogue_modes=(Epilogue.NONE,))


# flm.ProjectionGEMM's tiling of a B packed on the host ("bfp16"): tile_n 64 and a compute tile's
# k slice 128 (the order DequantBFP writes), k tiles of 512 or, where K is no multiple of it, the
# largest multiple of 128 below it that divides K (Sana 600M's 1152: 384). Pinned on both sides so
# the packing and the GEMM cannot resolve apart (pack_b says a wrong value packs silently).
BFP16_TILES = dict(tile_n=64, ct_k=128)


def bfp16_tiles(k: int, n: int) -> dict | None:
    """The tiling bfp16_pack and bfp16_linear take for a (n, k) weight; None where none fits."""
    if n % BFP16_TILES["tile_n"]:
        return None
    kt = next((t for t in (512, 384, 256, 128) if k % t == 0), None)
    return None if kt is None else dict(BFP16_TILES, k_tile=kt)


def bfp16_pack(w) -> np.ndarray:
    """W (out, in) as flm.ProjectionGEMM's B in bfp16ebs8 blocks (9 bytes for 8 in-features of an
    out-feature: a shared exponent byte, 8 mantissa bytes), rounded once to nearest even on the
    host: the format AIE2P's matrix instruction multiplies (Precision.MEDIUM rounds bf16 weights
    to it in every call). 9 bits a weight."""
    from iron.operators.flm.gemm.design import S, T
    from iron.operators.flm.packing import pack_b

    w = np.asarray(w, np.float32)
    n, k = w.shape
    t = bfp16_tiles(k, n)
    if t is None:
        raise ValueError(f"bfp16: a ({n}, {k}) weight is no whole tiles (out-features by 64, in-features by 128)")
    b = np.ascontiguousarray(w.T).astype(bfloat16)
    return np.asarray(pack_b(b, k_tile=t["k_tile"], n_tile=t["tile_n"], s=S, t=T, ct_k=t["ct_k"], bfp16=True,
                             round_conv_even=True), np.uint8).reshape(-1)


def bfp16_linear(x, q, n: int, k: int):
    """``x @ W.T`` with W packed by bfp16_pack: flm.ProjectionGEMM reads it as it is (no dequantize)."""
    from iron import Precision
    from iron.operators.flm import ProjectionGEMM
    from iron.operators.flm.gemm.op import Epilogue

    if len(x.shape) != 2 or x.shape[0] == 1:
        raise ValueError("bfp16: a weight packed for the GEMM takes rows, not one row")
    return ProjectionGEMM.step(x, bfp16_operand(q), K=k, N=n, bfp16_b=True, precision=Precision.MEDIUM,
                               epilogue_modes=(Epilogue.NONE,), **bfp16_tiles(k, n))


def bfp16_operand(q):
    """The packed bytes q (an iron.param) as a graph weight of bfp16ebs8 blocks.

    ProjectionGEMM's B is typed v8bfp16ebs8, which numpy has no dtype for, so the recorder would
    take a uint8 param as bytes; the weight is registered as the recorder does a param (its name
    the graph's attribute path, so variant banks line up), with the block type. Its bytes are
    stored as they are (Recording stores weights as their raw bytes)."""
    from aie.helpers.npdtypes import v8bfp16ebs8
    from aie.utils import bfp
    from iron.common.declare.operator import graph_recorder
    from iron.common.graph.handle import Handle

    rec = graph_recorder.get()
    key = id(q)
    if key not in rec.weights:
        name = rec._names.get(key) or rec.fresh(f"w{len(rec.weights)}")
        rec.weights[key] = (q, Handle.whole((q.size // bfp.BLOCK_BYTES,), v8bfp16ebs8, name, "weight"))
    return rec.weights[key][1]


# Rows up to which a linear runs QGEMVRows (q8, q6), beyond it the prompt chunk's path.
ROWS_GEMV = 8

# How q4nx's prompt path dequantizes: "bfp16" (IRON's flm.DequantBFP) or "bf16" (QDequant).
PREFILL = {"q4": "bfp16"}


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
    p = iron.param(bfp16_pack(w) if fmt == "bfp16" else quantize_fmt(w, fmt).reshape(-1))
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
        one_row = len(x.shape) == 1 or x.shape[0] == 1
        if bias is not None and one_row and FUSE_BIAS["on"]:
            return q_linear(x, weight, entry[1], entry[2], entry[3], adds=(bias,))
        y = q_linear(x, weight, entry[1], entry[2], entry[3])
        if bias is None:
            return y
        rows = int(np.prod(y.shape)) // bias.shape[0]
        return (y.reshape(rows, bias.shape[0]) + bias).reshape(y.shape)

    linear.hwnpu_q4 = True
    F.linear = linear
    _install_split()
    _install_residual()
    _fix_bfp_fits()


# Whether a one-row linear's bias is added in its GEMV (QGEMVAdd) rather than by an add after it.
FUSE_BIAS = {"on": True}

import contextlib  # noqa: E402
import contextvars  # noqa: E402

# A residual the next one-row call of a given Linear adds in its GEMV (into_linear).
_RESIDUAL: contextvars.ContextVar = contextvars.ContextVar("iron_q4_residual", default=None)


@contextlib.contextmanager
def into_linear(linear, residual):
    """Inside, ``linear``'s one-row call (a decode step's) on a q_param weight adds ``residual`` in
    its GEMV (QGEMVAdd, with its bias if it has one); ``state["used"]`` says whether it did, so
    the caller adds it itself if not (a layer's last projection: o_proj, out_proj, down_proj)."""
    state = {"linear": linear, "residual": residual, "used": False}
    token = _RESIDUAL.set(state)
    try:
        yield state
    finally:
        _RESIDUAL.reset(token)


def _install_residual() -> None:
    """Make iron.lm's Linear.body take into_linear's residual."""
    from iron.lm.blocks import Linear

    base = Linear.body
    if getattr(base, "hwnpu_residual", False):
        return

    def body(self, x):
        st = _RESIDUAL.get()
        one_row = len(x.shape) == 1 or x.shape[0] == 1
        entry = _Q4.get(id(self.weight))
        if (st is not None and st["linear"] is self and not st["used"] and one_row and entry is not None
                and entry[0] is self.weight):
            st["used"] = True
            r = st["residual"]
            return linear_add(x, self.weight, self.bias, r.reshape(r.shape[-1]) if len(r.shape) > 1 else r).reshape(
                *x.shape[:-1], entry[1])
        return base(self, x)

    body.hwnpu_residual = True
    Linear.body = body


def linear_add(x, weight, *adds):
    """``x @ W.T + adds...`` for one row of x and a q_param weight, the vectors (a bias, a
    residual, up to two) added in the GEMV (QGEMVAdd)."""
    entry = _Q4.get(id(weight))
    if entry is None or entry[0] is not weight:
        raise ValueError("linear_add: the weight is not a q_param")
    return q_linear(x, weight, entry[1], entry[2], entry[3], adds=tuple(a for a in adds if a is not None))


def q_rows(q, begin: int, end: int, k: int, fmt: str):
    """The bytes of out-features begin..end (multiples of 64) of q, a weight's blocks in fmt with
    k in-features: a column block (64 out-features) is contiguous, K / 128 blocks."""
    if begin % CB or end % CB:
        raise ValueError(f"q_rows: out-features {begin}..{end} are not whole column blocks of {CB}")
    per = k // (M_TILE * K_TILE // CB) * FORMATS[fmt]
    return q[begin // CB * per : end // CB * per]


def _install_split() -> None:
    """Make iron.lm's Linear.split (a fused projection's runs of rows, PLaMo 3's qkv_proj and
    gate_up_proj) on a q_param weight run one QGEMV and cut its output (decode), or q_linear on
    each run's column blocks (a prompt chunk)."""
    from iron.lm.blocks import Linear

    base = Linear.split
    if getattr(base, "hwnpu_q4", False):
        return

    def split(self, x, *rows: int, **gemv) -> list:
        entry = _Q4.get(id(self.weight))
        if entry is None or entry[0] is not self.weight:
            return base(self, x, *rows, **gemv)
        _, n, k, fmt = entry
        if sum(rows) != n:
            raise ValueError(f"Linear.split: rows {rows} of {n} out-features")
        one_row = len(x.shape) == 1 or x.shape[0] == 1
        if one_row:
            # decode: one QGEMV over the whole weight, its output cut (a contiguous run each);
            # a GEMV a run costs a command's worth of start-up each (qkv 0.61 -> 0.44 ms on 2B)
            whole = q_linear(x.reshape(k), self.weight, n, k, fmt)
        out, end = [], 0
        for r in rows:
            if one_row:
                y = whole[end : end + r]
                y = y if len(x.shape) == 1 else y.reshape(1, r)
            else:
                y = q_linear(x, q_rows(self.weight, end, end + r, k, fmt), r, k, fmt)
            if self.bias is not None:
                b = self.bias[end : end + r]
                lead = int(np.prod(y.shape)) // r
                y = (y.reshape(lead, r) + b).reshape(y.shape)
            out.append(y)
            end += r
        return out

    split.hwnpu_q4 = True
    Linear.split = split


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
    PREFILL["q4"] = args.prefill
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

        def body(self, x, a1=None, a2=None, ang=None, gw=None, *, rows: Scratchpad[np.int32] = np.int32(0)):
            if args.qkv:
                ang, gw = (a2, ang) if gw is None else (ang, gw)
                op = qkv_operator()
                given = dict(W=self.q.reshape(-1, FORMATS[fmt]), x=x, a1=a1, ang=ang, gw=gw)
                names = [m.name for m in op.declaration.members if m.name in given]
                return op.step(*(given[nm] for nm in names), fmt=fmt, cores=2, rope_rows=args.qkv, eps=1e-5)
            if args.adds:
                return q_linear(x, self.q, n, k, fmt, args.cores, adds=tuple(a for a in (a1, a2) if a is not None))
            if args.gemv_rows:
                return rows_operator().step(self.q.reshape(-1, FORMATS[fmt]), x.reshape(args.gemv_rows * k), fmt=fmt, K=k,
                                            rows=args.gemv_rows, cores=args.cores)  # (rows, n)
            if args.rows:  # a prompt chunk's: bounded per call
                return q_linear(x[:rows], self.q, n, k, fmt, args.cores)
            return q_linear(x, self.q, n, k, fmt, args.cores)

        def reference(self, x, a1=None, a2=None, ang=None, gw=None, rows=0):
            if args.qkv:
                ang, gw = (a2, ang) if gw is None else (ang, gw)  # the inputs in order: x, a1, ang, gw
                x = np.asarray(x, np.float64)
                sc = 1 / np.sqrt(np.mean(x * x) + 1e-5)
                xn = (x * np.asarray(gw, np.float64) * sc).astype(bfloat16).astype(np.float64)
                y = deq.astype(np.float64) @ xn + np.asarray(a1, np.float64)
                t = np.asarray(ang, np.float64)
                r = y[: args.qkv].reshape(-1, 32, 2)
                a, b = r[..., 0].copy(), r[..., 1].copy()
                r[..., 0], r[..., 1] = a * t[0::2] - b * t[1::2], a * t[1::2] + b * t[0::2]
                y[: args.qkv] = r.reshape(-1)
                return y.astype(bfloat16)
            x = np.asarray(x, np.float64)
            y = x @ deq.astype(np.float64).T
            for a in (a1, a2):
                if a is not None:
                    y = y + np.asarray(a, np.float64)
            y = y.astype(bfloat16)
            if rows:
                y[rows:] = 0
            return y

    g = G()
    shape = (args.gemv_rows, k) if args.gemv_rows else (k,) if args.rows == 0 else (args.rows, k)
    add_names = [f"a{i + 1}" for i in range(1 if args.qkv else args.adds)]
    cg = g.compile(x=shape, **{a: (n,) for a in add_names}, **(dict(ang=(64,), gw=(k,)) if args.qkv else {}))
    m = export_graph(cg, args.out, "qlinear", dict(n=n, k=k, fmt=fmt))
    # a mean in x shows a rounding bias that zero-mean inputs average away
    x = (rng.standard_normal(shape) + args.x_mean).astype(bfloat16)
    inputs = dict(x=x, **{a: rng.standard_normal(n).astype(bfloat16) for a in add_names})
    if args.qkv:
        th = rng.uniform(-np.pi, np.pi, 32)
        ang = np.empty(64, np.float32)
        ang[0::2], ang[1::2] = np.cos(th), np.sin(th)
        inputs["ang"] = ang.astype(bfloat16)
        inputs["gw"] = (1 + 0.3 * rng.standard_normal(k)).astype(bfloat16)
    write_check(g, m, args.out, inputs, dict(rows=args.valid or args.rows) if args.rows and not args.gemv_rows else None)
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
    c.add_argument("--gemv-rows", type=int, default=0, help="rows of x through QGEMVRows (a call checking drafts; q8)")
    c.add_argument("--valid", type=int, default=0, help="rows the call bounds x to (default all)")
    c.add_argument("--adds", type=int, default=0, help="vectors added in the GEMV (QGEMVAdd: 1 or 2)")
    c.add_argument("--qkv", type=int, default=0, help="QGEMVQKV: the rows turned by RoPE (the norm and a bias too)")
    c.add_argument("--x-mean", type=float, default=0.0, help="the mean of x's random values")
    c.add_argument("--prefill", choices=("bfp16", "bf16"), default=PREFILL["q4"], help="q4nx's prompt path (q_linear)")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return _check(a)
    return 1


if __name__ == "__main__":
    sys.exit(main())


# --- a gated MLP's gate and up as one table (decode: one operator) ---------------------------

# Glu tables made by glu_param: by id of the param, (the param, out-features I, in-features, format).
_GLU: dict[int, tuple[object, int, int, str]] = {}


def glu_table(gate, up) -> np.ndarray:
    """gate (I, K) and up (I, K) as one (2 I, K) weight, their column blocks of 64 out-features
    alternating (gate first): QGEMVGlu reads a gate block and the up block of the same
    out-features in turn; a prompt chunk's or a rows GEMV's reads one of the two (parity)."""
    gate, up = np.asarray(gate, np.float32), np.asarray(up, np.float32)
    I, K = gate.shape
    if I % CB:
        raise ValueError(f"glu_table: {I} out-features are not whole column blocks of {CB}")
    return np.stack([gate.reshape(I // CB, CB, K), up.reshape(I // CB, CB, K)], 1).reshape(2 * I, K)


def glu_param(gate, up, fmt: str):
    """An iron.param of glu_table(gate, up) in fmt, known to glu_linear and glu_part."""
    import iron

    I, K = np.shape(gate)
    p = iron.param(quantize_fmt(glu_table(gate, up), fmt).reshape(-1))
    _GLU[id(p)] = (p, I, K, fmt)
    return p


def _cb_chunk(cb_bytes: int) -> int:
    """The bytes of a DMA run within a column block's bytes: whole words, at most 1023 of them."""
    for m in range(1, cb_bytes + 1):
        c = cb_bytes // m
        if cb_bytes % m == 0 and c % 4 == 0 and c <= 4 * 1023:
            return c
    raise ValueError(f"no run of a column block of {cb_bytes} bytes fits a DMA dimension")


_GLU_OPS: dict = {}


def glu_operators():
    """QGEMVGlu (one row: y = silu(G x) * (U x) from a glu_table) and the parity-select forms of
    QDequant and QGEMVRows (the gate's or the up's column blocks of a glu_table: a prompt chunk's
    and a rows GEMV's, the table read once, no second copy)."""
    if _GLU_OPS:
        return _GLU_OPS
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import Buffer, In, ObjectFifo, Out, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param, tunable, widths
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    class QGEMVGlu(Operator):
        """``y = silu(G x) * (U x)``: G and U a glu_table's (I out-features), one row of x. Each
        column takes I / columns out-features: for each of its column blocks the gate's 64 rows
        then the up's, a core's 32 of each held until both have passed (kernels/mv_add.h)."""

        fmt: str = param(array=True)
        blocks: int = param(array=True)  # the table's: 2 I / 64 * K / 256 * 2
        K: int = param(array=True)
        N: int = param(derive=lambda op: op.blocks * CB // (2 * op.K // K_TILE) // 2, array=True)
        cores: int = param(default=2, array=True)
        block_bytes: int = param(derive=lambda op: FORMATS[op.fmt], array=True)
        num_aie_columns: int = tunable(domain=widths)

        W = In(blocks, block_bytes, dtype=np.uint8, obj_shape=(cores, block_bytes), per=(num_aie_columns,))
        x = In(K, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
        y = Out(N, obj_shape=(CB,), per=(num_aie_columns,))

        def resolve(self, dev):
            cbs = self.N // CB
            cols = self.resolve_columns(dev, self.num_aie_columns, fits=lambda c: cbs % c == 0)
            return self.replace(num_aie_columns=cols)

        def build(self) -> Built:
            K, n, cores, B = self.K, self.num_aie_columns, self.cores, self.block_bytes
            pairs, kbs, out = self.N // CB // n, K // K_TILE, CB // cores
            kdir = Path(config.aie_kernels_dir())
            here = Path(__file__).resolve().parent / "kernels"
            x_t = np.ndarray[(K,), np.dtype[bfloat16]]
            acc_t = np.ndarray[(out,), np.dtype[np.float32]]
            y_t = np.ndarray[(out,), np.dtype[bfloat16]]
            blk_t = np.ndarray[(B,), np.dtype[np.uint8]]
            sums_t = np.ndarray[(K // GROUP,), np.dtype[bfloat16]]
            common = dict(source_file=str(here / f"{self.fmt}_mv.cc"),
                          include_dirs=[str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
                          compile_flags=[f"-D{self.fmt.upper()}_K={K}", f"-DQ_OUT={out}", "-DQ_GLU=1"],
                          object_file_name=f"{self.fmt}_mv_glu_{K}_{out}.o")
            begin = ExternalFunction(f"{self.fmt}_mv_begin", arg_types=[x_t, sums_t, acc_t], **common)
            block = ExternalFunction(f"{self.fmt}_mv_block", arg_types=[blk_t, x_t, sums_t, acc_t, np.int32, np.int32],
                                     **common)
            end = ExternalFunction(f"{self.fmt}_mv_end_glu", arg_types=[acc_t, acc_t, y_t], **common)
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            x_l3 = [ObjectFifo(self.x.obj_type, depth=1) for _ in range(n)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            if cores == 1:
                w_l1, x_l1, y_l1 = [[f] for f in w_l3], x_l3, [[f] for f in y_l3]
            else:
                w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
                x_l1 = [f.cons().forward(obj_type=self.x.obj_type, tile=AnyMemTile) for f in x_l3]
                y_l1 = [f.prod().join([0, out], obj_types=[y_t, y_t], depths=[2, 2], tile=AnyMemTile)
                        for f in y_l3]

            def core(w_f, x_f, y_f, sums, g, u, begin, block, end):
                x = x_f.acquire(1)
                begin(x, sums, g)
                begin(x, sums, u)
                for _ in range_(pairs):
                    for acc in (g, u):
                        for kb in range_(kbs):
                            kb32 = index.casts(T.i32(), kb)
                            for h in range(2 if cores == 1 else 1):
                                w = w_f.acquire(1)
                                block(w, x, sums, acc, kb32, h)
                                w_f.release(1)
                    y = y_f.acquire(1)
                    end(g, u, y)
                    y_f.release(1)
                x_f.release(1)

            workers = [Worker(core, [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), Buffer(sums_t),
                                     Buffer(acc_t), Buffer(acc_t), begin, block, end])
                       for i in range(n) for j in range(cores)]
            self.W.bind([f.prod() for f in w_l3])
            self.x.bind([f.prod() for f in x_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def ops(self) -> int:
            return 4 * self.N * self.K

        def reference(self, W, x):
            w = dequantize_fmt(W, 2 * self.N, self.K, self.fmt).astype(np.float64)
            w = w.reshape(self.N // CB, 2, CB, self.K)
            x = np.asarray(x, np.float64)
            g = w[:, 0].reshape(self.N, self.K) @ x
            u = w[:, 1].reshape(self.N, self.K) @ x
            return (g / (1 + np.exp(-g)) * u).astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="partial sums narrowed to bf16 before the scale")

    QGEMV, QDequant = operators()
    Rows = rows_operator()

    def select(base):
        """``base`` (QDequant, QGEMVRows) over the gate's (parity 0) or the up's (1) column blocks of
        a glu_table: its W the whole table, read every other column block."""

        def sequence(self, rt):
            n, B, K = self.num_aie_columns, self.block_bytes, self.K
            cb_bytes = 2 * (K // K_TILE) * B
            cbs = self.N // CB // n
            c = _cb_chunk(cb_bytes)
            total = 2 * self.blocks * B
            taps = [(lane, TensorAccessPattern((total,), (2 * lane * cbs + self.parity) * cb_bytes,
                                               [cbs, cb_bytes // c, c], [2 * cb_bytes, c, 1]), None, None)
                    for lane in range(n)]
            w = self.W
            made = w.transfers
            w.transfers = lambda: taps
            try:
                return base.sequence(self, rt)
            finally:
                w.transfers = made

        def reference(self, W, *rest):
            t = dequantize_fmt(W, 2 * self.N, self.K, self.fmt).reshape(self.N // CB, 2, CB, self.K)
            half = t[:, self.parity].reshape(self.N, self.K)
            q = quantize_fmt(half, self.fmt)  # exact: the same values in the same groups
            return base.reference(self, q, *rest)

        cores = getattr(base, "cores", None)
        shape = (base.cores, base.block_bytes) if cores is not None and not isinstance(cores, int) else (1, base.block_bytes)
        body = dict(
            parity=param(default=0, array=True),
            tblocks=param(derive=lambda op: 2 * op.blocks, array=True, repr=False),
            sequence=sequence, reference=reference,
            __annotations__={"parity": int, "tblocks": int},
        )
        body["W"] = In(body["tblocks"], base.block_bytes, dtype=np.uint8, obj_shape=shape, per=(base.num_aie_columns,))
        return type(base.__name__, (base,), body)

    _GLU_OPS.update(glu=QGEMVGlu, dequant=select(QDequant), rows=select(Rows))
    return _GLU_OPS


def glu_linear(x, gu):
    """silu(x G^T) * (x U^T) for a glu_param: one row through QGEMVGlu; a few rows (a call
    checking drafts) or a prompt chunk through the gate's and the up's column blocks in turn."""
    p, I, K, fmt = _GLU[id(gu)]
    ops = glu_operators()
    one_row = len(x.shape) == 1 or x.shape[0] == 1
    q = gu.reshape(-1, FORMATS[fmt])
    if one_row:
        y = ops["glu"].step(q, x.reshape(K), fmt=fmt)
        return y if len(x.shape) == 1 else y.reshape(1, I)
    import iron.functional as F
    from iron import Precision

    def part(parity):
        if x.shape[0] <= ROWS_GEMV and fmt in ("q6", "q8"):
            R = x.shape[0]
            return ops["rows"].step(q, x.reshape(R * K), fmt=fmt, K=K, rows=R, parity=parity, blocks=q.shape[0] // 2)
        w = ops["dequant"].step(q, fmt=fmt, K=K, parity=parity, blocks=q.shape[0] // 2)
        return _BASE["linear"](x, w, precision=Precision.MEDIUM)

    return F.silu(part(0)) * part(1)


def install_glu() -> None:
    """Make iron.lm's GatedMLP run glu_linear on a ``gu`` table (glu_param) where it holds one."""
    from iron.lm.blocks import GatedMLP

    base = GatedMLP.body
    if getattr(base, "hwnpu_glu", False):
        return

    def body(self, x):
        gu = getattr(self, "gu", None)
        if gu is None:
            return base(self, x)
        return self.down_proj(glu_linear(x, gu))

    body.hwnpu_glu = True
    GatedMLP.body = body
