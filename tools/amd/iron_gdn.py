"""Gated DeltaNet's recurrence as an IRON operator (Qwen3.5 / Qwen3.6's linear attention).

IRON has the kernel (mlir-aie's ssm/gated_delta_rule.cc) but no operator; this is one, in
hwnpu as iron_q4's operators are. Its kernel is kernels/gdn.cc: the same recurrence, with
the block gathered in the core from a row of the column and silu applied there.

GatedDeltaRule takes the short convolution's output (before silu) of the model's fused
projection, laid out by column (``gdn_layout``): column ``c`` of ``n`` holds, a row at a time,

    q of its key heads, k of them, v of its value heads, then (a, b) of each value head

``KH / n`` key heads and ``H / n`` value heads (value head ``h`` reads key head
``h // (H / KH)``, as transformers' repeat_interleave). Two cores a column take ``H / 2n``
value heads each and walk them 16 value columns (a slice) at a time: a slice's state
(key_dim x 16, float32) comes in, every row of the call runs through it a block at a time,
and the state after the last valid row goes out. The memtile broadcasts the column's rows to
both cores and splits (joins) the two cores' slices of state (of y, of the final state).

The state is the operator's own layout, a slice's run contiguous in the order the column
streams them: (column, head of the core, slice, core, key_dim, 16), float32. ``state_order``
maps it to (H, key_dim, value_dim). Each value head's dt_bias and A reach its core as
runtime parameter words (two 16-bit halves of each float), fixed when the step is built.

    python tools/amd/iron_gdn.py check OUT [--rows 1|64] [--valid N]
    go run ./tools/amd/irongraph -tol 0.05 OUT
"""

import argparse
import functools
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

SLICE = 16  # value columns a core's kernel call keeps (gdn.cc's GDR_VALUE_DIM)
BLOCK = 16  # rows a prompt chunk's kernel call runs (GDR_ROWS)
QUEUED = 3  # drains a shim channel holds queued at once (its task queue is 4 deep)


def column_widths(heads: int, key_heads: int, key_dim: int, value_dim: int, n: int) -> int:
    """A column's part of a row: its key heads' q and k, its value heads' v and (a, b)."""
    return 2 * key_heads // n * key_dim + heads // n * (value_dim + 2)


def gdn_layout(heads: int, key_heads: int, key_dim: int, value_dim: int, n: int) -> np.ndarray:
    """The order of the fused projection's rows for ``n`` columns: index i of the result is the
    row of transformers' [q (KH dk), k (KH dk), v (H dv), a (H), b (H)] that row i of ours is."""
    q0, k0, v0 = 0, key_heads * key_dim, 2 * key_heads * key_dim
    a0 = v0 + heads * value_dim
    b0 = a0 + heads
    kh, vh = key_heads // n, heads // n
    order = []
    for c in range(n):
        ks = range(c * kh, (c + 1) * kh)
        vs = range(c * vh, (c + 1) * vh)
        order += [q0 + k * key_dim + i for k in ks for i in range(key_dim)]
        order += [k0 + k * key_dim + i for k in ks for i in range(key_dim)]
        order += [v0 + h * value_dim + i for h in vs for i in range(value_dim)]
        for h in vs:
            order += [a0 + h, b0 + h]
    return np.array(order)


def state_order(heads: int, key_dim: int, value_dim: int, n: int) -> np.ndarray:
    """Index i of the result is the (h, k, v) state element (flat, h-major) at i of ours."""
    hc, S = heads // (2 * n), value_dim // SLICE
    idx = np.arange(heads * key_dim * value_dim).reshape(heads, key_dim, S, SLICE)
    out = np.empty((n, hc, S, 2, key_dim, SLICE), np.int64)
    for c in range(n):
        for i in range(hc):
            for j in range(2):
                h = c * 2 * hc + j * hc + i
                out[c, i, :, j] = idx[h].transpose(1, 0, 2)
    return out.reshape(-1)


def halves(x: float) -> tuple[int, int]:
    bits = int(np.float32(x).view(np.uint32))
    return bits >> 16, bits & 0xFFFF


def head_words(dt_bias, A) -> tuple[int, ...]:
    """Each value head's (dt_bias, A) as GatedDeltaRule's ``words``: four 16-bit halves a head."""
    out = []
    for d, a in zip(np.asarray(dt_bias, np.float32), np.asarray(A, np.float32)):
        out += [*halves(d), *halves(a)]
    return tuple(out)


def conv_hist_words(key_dim: int) -> int:
    """bf16 words of a core's half of a slice pair: what holds the convolution's rows (conv)."""
    return 2 * key_dim * SLICE


def conv_reference(x, taps, state, *, heads, key_heads, key_dim, value_dim, n):
    """GatedDeltaRule's conv in numpy: the convolution's output u (1, W) and the state with each
    column's first pair (the rows before) taken out (the plain operator's layout), and a function
    putting the rows ending with this one back in front of a final state of that layout."""
    H, KH, dk, dv = heads, key_heads, key_dim, value_dim
    seg = column_widths(H, KH, dk, dv, n)
    K = np.asarray(taps).shape[0]
    pair, run = 2 * dk * SLICE, (1 + H // (2 * n) * (dv // SLICE)) * 2 * dk * SLICE
    st = np.asarray(state, np.float32).reshape(n, run)
    hist = st[:, : dk * SLICE].copy().view(bfloat16)[:, : (K - 1) * seg].reshape(n, K - 1, seg)
    x = np.asarray(x, bfloat16).reshape(n, seg).astype(np.float32)
    t = np.asarray(taps, bfloat16).astype(np.float32).reshape(K, n, seg).transpose(1, 0, 2)
    u = t[:, K - 1] * x
    for k in range(K - 1):
        u = u + t[:, k] * hist[:, k].astype(np.float32)
    rest = st[:, pair:].reshape(-1)
    new = np.concatenate([hist[:, 1:].astype(np.float32), x[:, None]], axis=1).astype(bfloat16)

    def put(final):
        out = np.zeros((n, run), np.float32)
        out[:, pair:] = np.asarray(final, np.float32).reshape(n, run - pair)
        half = np.zeros((n, 2 * dk * SLICE), bfloat16)
        half[:, : (K - 1) * seg] = new.reshape(n, -1)
        out[:, : dk * SLICE] = half.view(np.float32)
        out[:, dk * SLICE : pair] = half.view(np.float32)
        return out.reshape(-1)

    return u.astype(bfloat16).reshape(1, n * seg), rest, put


def gdn_reference(x, state, words, *, heads, key_heads, key_dim, value_dim, n, valid):
    """numpy float32, as transformers' torch_recurrent_gated_delta_rule (silu on q, k, v first,
    the l2 norm in the rule): y (rows, H dv) and the final state, both in the operator's layouts."""
    H, KH, dk, dv = heads, key_heads, key_dim, value_dim
    rows = x.shape[0]
    x = np.asarray(x, np.float32)
    inv = np.argsort(gdn_layout(H, KH, dk, dv, n))
    # back to transformers' order: q, k, v, a, b
    hf = x[:, inv]
    silu = lambda v: v / (1 + np.exp(-v))  # noqa: E731
    qkv = silu(hf[:, : 2 * KH * dk + H * dv]).astype(bfloat16).astype(np.float32)
    q = qkv[:, : KH * dk].reshape(rows, KH, dk)
    k = qkv[:, KH * dk : 2 * KH * dk].reshape(rows, KH, dk)
    v = qkv[:, 2 * KH * dk :].reshape(rows, H, dv)
    a = hf[:, 2 * KH * dk + H * dv : 2 * KH * dk + H * dv + H]
    b = hf[:, 2 * KH * dk + H * dv + H :]
    w = np.asarray(words, np.int64).reshape(H, 4)
    f = lambda hi, lo: np.array((hi << 16) | lo, np.uint32).view(np.float32)  # noqa: E731
    dt_bias = f(w[:, 0], w[:, 1])
    A = f(w[:, 2], w[:, 3])
    order = state_order(H, dk, dv, n)
    s = np.empty(H * dk * dv, np.float32)
    s[order] = np.asarray(state, np.float32).reshape(-1)
    s = s.reshape(H, dk, dv)
    y = np.zeros((rows, H, dv), np.float32)
    rep = H // KH
    for t in range(min(valid, rows)):
        for h in range(H):
            qt, kt = q[t, h // rep], k[t, h // rep]
            qt = qt / np.sqrt((qt**2).sum() + np.float32(1e-6)) / np.sqrt(np.float32(dk))
            kt = kt / np.sqrt((kt**2).sum() + np.float32(1e-6))
            g = A[h] * np.logaddexp(np.float32(0), a[t, h] + dt_bias[h])
            beta = 1 / (1 + np.exp(-b[t, h]))
            s[h] = np.exp(g) * s[h]
            s[h] = s[h] + np.outer(kt, beta * (v[t, h] - kt @ s[h]))
            y[t, h] = qt @ s[h]
    final = s.reshape(-1)[order]
    return y.reshape(rows, H * dv).astype(bfloat16), final.astype(np.float32)


@functools.cache
def gdn_operator(n: int = 8, hc: int = 2, conv: int = 0):
    """The GatedDeltaRule operator for ``n`` columns of two cores, each core ``hc`` value heads:
    each core's words (four a head) are values of their own, so it is made per shape.

    ``conv`` > 0 (a decode step only): the operator runs the short convolution of ``conv`` taps
    itself. Its ``x`` is the projection's row (before the convolution), ``taps`` the convolution's
    (conv, W) in x's columns, and each column's run of the state starts with a pair more, the
    ``conv - 1`` rows before the call (each core's half a copy: bf16 rows of the column's
    width, packed); the final state's holds the rows ending with the call."""
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.extras.dialects import arith
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import In, ObjectFifo, Out, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param, per_call
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator, Rtp

    cores = 2 * n

    def kernels(R: int, blocks: int, dk: int, seg: int):
        flags = [f"-DGDR_SEG={seg}", f"-DGDR_CONV={conv}"] if conv else []
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        common = dict(
            source_file=str(here / "gdn.cc"),
            include_dirs=[str(kdir), config.cxx_header_path()],
            compile_flags=["-ffp-contract=off", "-fno-fast-math", "-fno-unroll-loops",
                           f"-DGDR_ROWS={R}", f"-DGDR_KEY_DIM={dk}", f"-DGDR_VALUE_DIM={SLICE}",
                           f"-DGDR_BLOCKS={blocks}", *flags],
            object_file_name=f"gdn_{R}_{dk}_{blocks}" + (f"_c{conv}_{seg}" if conv else "") + ".o",
        )
        seg_t = np.ndarray[(seg,), np.dtype[bfloat16]]
        st_t = np.ndarray[(dk * SLICE,), np.dtype[np.float32]]
        y_t = np.ndarray[(R * SLICE,), np.dtype[bfloat16]]
        i32 = np.int32
        gather = ExternalFunction("gdn_gather", arg_types=[seg_t, i32, i32, i32, i32, i32], **common)
        rule = ExternalFunction("gdn_rule", arg_types=[st_t, st_t, y_t, i32, i32, i32, i32, i32, i32],
                                **common)
        if conv:
            xt_t = np.ndarray[((1 + conv) * seg,), np.dtype[bfloat16]]
            convk = ExternalFunction("gdn_conv", arg_types=[xt_t, st_t, st_t], **common)
            gather = ExternalFunction("gdn_gather_conv", arg_types=[i32, i32, i32, i32, i32], **common)
            return gather, rule, convk
        return gather, rule, None

    def i32c(v: int):
        return arith.constant(v, T.i32())

    ns = {}

    class _GatedDeltaRule(Operator):
        """``y, final = rule(x, state)`` over ``rows`` rows: Gated DeltaNet's recurrence for every
        value head, the short convolution's output ``x`` in gdn_layout's columns (silu not yet
        applied), the state in the operator's layout (module docstring)."""

        # 2 * heads / cores slices' runs of drains a column for a prompt chunk
        aiecc_flags: ClassVar[tuple[str, ...]] = ("--reclaim-runtime-bds",)

        rows: int = param(array=True)
        columns: int = param(default=n, init=False, array=True)
        heads: int = param(array=True)
        key_heads: int = param(array=True)
        key_dim: int = param(array=True)
        value_dim: int = param(array=True)
        words: tuple = param(repr=False)  # head_words: the cores read them as residents
        seg: int = param(derive=lambda op: column_widths(op.heads, op.key_heads, op.key_dim,
                                                         op.value_dim, n), array=True)
        width: int = param(derive=lambda op: n * op.seg, repr=False)
        hv: int = param(derive=lambda op: op.heads * op.value_dim, repr=False)
        # with conv, a pair more a column: the rows before the call
        states: int = param(derive=lambda op: op.heads * op.key_dim * op.value_dim
                            + (n * 2 * op.key_dim * SLICE if conv else 0), repr=False)
        pair: int = param(derive=lambda op: 2 * op.key_dim * SLICE, repr=False)
        block: int = param(derive=lambda op: 1 if op.rows == 1 else BLOCK, array=True)
        ypair: int = param(derive=lambda op: 2 * op.block * SLICE, repr=False)

        x = In(rows, width, obj_shape=(seg,), per=(columns,))
        if conv:
            taps = In(conv, width, obj_shape=(seg,), per=(columns,))  # x's channel, after the row
        state = In(states, dtype=np.float32, obj_shape=(pair,), per=(columns,), depth=1)
        y = Out(rows, hv, obj_shape=(ypair,), per=(columns,))
        final = Out(states, dtype=np.float32, obj_shape=(pair,), per=(columns,), depth=1)
        valid = per_call(bounds=rows)
        count = per_call(derive=lambda op: op.valid)

        def validate(self):
            H, KH = self.heads, self.key_heads
            if H % cores or KH % n or H % KH or (H // n) % (H // KH):
                raise ValueError(f"GatedDeltaRule: {H} value and {KH} key heads over {n} columns "
                                 f"of two cores")
            if self.value_dim % SLICE or self.key_dim % 16:
                raise ValueError("GatedDeltaRule: value_dim whole slices of 16, key_dim of 16")
            if self.rows != 1 and self.rows % BLOCK:
                raise ValueError(f"GatedDeltaRule: rows ({self.rows}) one or whole blocks of {BLOCK}")
            if H != cores * hc:
                raise ValueError(f"GatedDeltaRule: {H} value heads, made for {cores * hc}")
            if len(self.words) != 4 * H:
                raise ValueError(f"GatedDeltaRule: {len(self.words)} words for {H} heads")
            if conv and self.rows != 1:
                raise ValueError("GatedDeltaRule: the convolution in the core is a decode step's")
            if conv and (conv - 1) * self.seg > conv_hist_words(self.key_dim):
                raise ValueError(f"GatedDeltaRule: {conv - 1} rows of {self.seg} past a pair's half")

        def geometry(self):
            H, KH, dk, dv = self.heads, self.key_heads, self.key_dim, self.value_dim
            hc, S, R = H // cores, dv // SLICE, self.block
            return H, KH, dk, dv, hc, S, R, self.rows // R

        def build(self) -> Built:
            H, KH, dk, dv, hc, S, R, blocks = self.geometry()
            gather, rule, convk = kernels(R, blocks, dk, self.seg)
            seg_t = np.ndarray[((1 + conv) * self.seg,), np.dtype[bfloat16]]
            st_t = np.ndarray[(dk * SLICE,), np.dtype[np.float32]]
            y_t = np.ndarray[(R * SLICE,), np.dtype[bfloat16]]
            xd = 1 if conv else 2
            x_l3 = [ObjectFifo(seg_t, depth=xd) for _ in range(n)]
            x_l1 = [f.cons().forward(obj_type=seg_t, depth=xd, tile=AnyMemTile) for f in x_l3]
            s_l3 = [ObjectFifo(self.state.obj_type, depth=1) for _ in range(n)]
            s_l1 = [f.cons().split([0, dk * SLICE], obj_types=[st_t, st_t], depths=[1, 1],
                                   tile=AnyMemTile) for f in s_l3]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            y_l1 = [f.prod().join([0, R * SLICE], obj_types=[y_t, y_t], depths=[2, 2], tile=AnyMemTile)
                    for f in y_l3]
            f_l3 = [ObjectFifo(self.final.obj_type, depth=1) for _ in range(n)]
            f_l1 = [f.prod().join([0, dk * SLICE], obj_types=[st_t, st_t], depths=[1, 1], tile=AnyMemTile)
                    for f in f_l3]
            Hg, KHg, rep = H // n, KH // n, H // KH
            rtps = [Rtp([self.count, *(self.value(f"hp{k}_{w}") for w in range(4 * hc))])
                    for k in range(cores)]

            def core(x_f, s_f, y_f, f_f, gather, rule, convk, rtp, j, k):
                w = rtp.read()
                valid = w[self.count]
                hp = [w[self.value(f"hp{k}_{u}")] for u in range(4 * hc)]
                zero, size = i32c(0), i32c(R)
                if conv:
                    xt, st, fin = x_f.acquire(1), s_f.acquire(1), f_f.acquire(1)
                    convk(xt, st, fin)
                    x_f.release(1)
                    s_f.release(1)
                    f_f.release(1)
                for i in range(hc):
                    hl = j * hc + i  # the value head within the column
                    kl = hl // rep
                    qo, ko = i32c(kl * dk), i32c(KHg * dk + kl * dk)
                    vbase = 2 * KHg * dk + hl * dv
                    abo = i32c(2 * KHg * dk + Hg * dv + 2 * hl)
                    for s in range_(S):
                        vo = arith.addi(i32c(vbase), arith.muli(index.casts(T.i32(), s), i32c(SLICE)))
                        st, fin = s_f.acquire(1), f_f.acquire(1)
                        for b in range_(blocks):
                            for t in range_(R):
                                if conv:
                                    gather(index.casts(T.i32(), t), qo, ko, vo, abo)
                                else:
                                    row = x_f.acquire(1)
                                    gather(row, index.casts(T.i32(), t), qo, ko, vo, abo)
                                    x_f.release(1)
                            b32 = index.casts(T.i32(), b)
                            left = arith.subi(valid, arith.muli(b32, size))
                            live = arith.minsi(arith.maxsi(left, zero), size)
                            y = y_f.acquire(1)
                            rule(st, fin, y, live, b32, hp[4 * i], hp[4 * i + 1], hp[4 * i + 2],
                                 hp[4 * i + 3])
                            y_f.release(1)
                        s_f.release(1)
                        f_f.release(1)

            workers = []
            for c in range(n):
                for j in range(2):
                    if conv:
                        fn = lambda x_f, s_f, y_f, f_f, g, r, ck, rtp, j=j, k=2 * c + j: core(  # noqa: E731
                            x_f, s_f, y_f, f_f, g, r, ck, rtp, j, k)
                        ks = [gather, rule, convk]
                    else:
                        fn = lambda x_f, s_f, y_f, f_f, g, r, rtp, j=j, k=2 * c + j: core(  # noqa: E731
                            x_f, s_f, y_f, f_f, g, r, None, rtp, j, k)
                        ks = [gather, rule]
                    workers.append(Worker(fn, [x_l1[c].cons(), s_l1[c][j].cons(), y_l1[c][j].prod(),
                                               f_l1[c][j].prod(), *ks, rtps[2 * c + j]]))
            self.x.bind([f.prod() for f in x_l3])
            if conv:
                self.taps.bind([f.prod() for f in x_l3])
            self.state.bind([f.prod() for f in s_l3])
            self.y.bind([f.cons() for f in y_l3])
            self.final.bind([f.cons() for f in f_l3])
            return Built(workers, rtps)

        def sequence(self, rt):
            H, KH, dk, dv, hc, S, R, blocks = self.geometry()
            rows, W, seg, HV = self.rows, self.width, self.seg, self.hv
            run = (hc * S + (1 if conv else 0)) * self.pair  # a column's states, contiguous
            # The fills first: a prompt chunk's drains are 2 hc S descriptors a column, past what a
            # channel's task queue holds before the data they wait for has come in.
            for c in range(n):
                self.state.fill(TAP((self.states,), c * run, [run], [1]), lane=c, managed=False)
                if conv:  # the row once (the core keeps the convolution's output), then the taps
                    self.x.fill(TAP((rows, W), c * seg, [seg], [1]), lane=c, managed=False)
                    self.taps.fill(TAP((conv, W), c * seg, [conv, seg], [W, 1]), lane=c, managed=False)
                else:
                    self.x.fill(TAP((rows, W), c * seg, [hc * S, rows, seg], [0, W, 1]), lane=c,
                                managed=False)
            drains = [self.final.drain(TAP((self.states,), c * run, [run], [1]), lane=c, wait=True,
                                       managed=False) for c in range(n)]
            if R == 1:
                for c in range(n):
                    h0 = c * 2 * hc  # the column's first head (core 0's), core 1's from h0 + hc
                    tap = TAP((rows, HV), h0 * dv, [hc, S, 2, R * SLICE], [dv, SLICE, hc * dv, 1])
                    drains.append(self.y.drain(tap, lane=c, wait=True, managed=False))
            else:
                # A slice's run of y a drain: hc S of them a column, at most QUEUED in a channel's
                # task queue at once (more overflow it), the columns in turn.
                pending = [[] for _ in range(n)]
                for i in range(hc):
                    for s in range(S):
                        for c in range(n):
                            if len(pending[c]) == QUEUED:
                                pending[c].pop(0).await_()
                            tap = TAP((rows, HV), (c * 2 * hc + i) * dv + s * SLICE, [blocks, 2, R, SLICE],
                                      [R * HV, hc * dv, HV, 1])
                            pending[c].append(self.y.drain(tap, lane=c, wait=True, managed=False))
                drains += [d for p in pending for d in p]
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 6 * self.rows * self.heads * self.key_dim * self.value_dim

        if conv:
            def reference(self, x, taps, state):
                shape = dict(heads=self.heads, key_heads=self.key_heads, key_dim=self.key_dim,
                             value_dim=self.value_dim, n=n)
                u, rest, put = conv_reference(x, taps, state, **shape)
                y, final = gdn_reference(u, rest, self.words, **shape, valid=1)
                return y, put(final)
        else:
            def reference(self, x, state):
                return gdn_reference(x, state, self.words, heads=self.heads, key_heads=self.key_heads,
                                     key_dim=self.key_dim, value_dim=self.value_dim, n=n,
                                     valid=int(self.valid if self.valid is not None else self.rows))

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.02, 1e-3, note="bf16 q, k, v after silu; the state in limbs")

    # A core's words: its value heads' (dt_bias, A) halves, one value each.
    for k in range(cores):
        for u in range(4 * hc):
            ns[f"hp{k}_{u}"] = per_call(derive=lambda op, at=4 * hc * k + u: op.words[at])
    ns["__doc__"] = _GatedDeltaRule.__doc__
    ns["__module__"] = __name__
    return type("GatedDeltaRule", (_GatedDeltaRule,), ns)


def _check(args) -> int:
    import iron
    from iron import Scratchpad
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    n = 8
    H, KH, dk, dv = args.heads, args.key_heads, 128, 128
    rng = np.random.default_rng(3)
    dt_bias = rng.uniform(-4, 1, H).astype(np.float32)
    A = -np.exp(rng.uniform(-3, 2.5, H)).astype(np.float32)
    words = head_words(dt_bias, A)
    K = args.conv
    G = gdn_operator(n, H // (2 * n), K)
    seg = column_widths(H, KH, dk, dv, n)
    W = n * seg
    rows = args.rows
    shape = dict(heads=H, key_heads=KH, key_dim=dk, value_dim=dv)
    if K:
        states = H * dk * dv + n * 2 * dk * SLICE

        class Graph(iron.Graph):
            def body(self, x, taps, state):
                return G.step(x, taps, state, **shape, words=words)

            def reference(self, x, taps, state):
                u, rest, put = conv_reference(x, taps, state, **shape, n=n)
                y, final = gdn_reference(u, rest, words, **shape, n=n, valid=1)
                return y, put(final)

        g = Graph()
        cg = g.compile(x=(1, W), taps=(K, W), state=((states,), np.float32))
        m = export_graph(cg, args.out, "gdn", dict(rows=1, heads=H, key_heads=KH, conv=K))
        print(cg.plan.report("gdn"))
        x = (rng.standard_normal((1, W)) * 1.5).astype(bfloat16)
        taps = (rng.standard_normal((K, W)) * 0.5).astype(bfloat16)
        state = (rng.standard_normal(states) * 0.3).astype(np.float32)
        run = states // n
        hist = (rng.standard_normal((n, (K - 1) * seg)) * 1.5).astype(bfloat16)
        st = state.reshape(n, run)
        half = np.zeros((n, 2 * dk * SLICE), bfloat16)
        half[:, : (K - 1) * seg] = hist
        st[:, : dk * SLICE] = half.view(np.float32)
        st[:, dk * SLICE : 2 * dk * SLICE] = half.view(np.float32)
        write_check(g, m, args.out, dict(x=x, taps=taps, state=state), None)
        print(f"{args.out}: GatedDeltaRule with its convolution ({K} taps), {H} value heads")
        return 0

    class Graph(iron.Graph):
        def body(self, x, state, *, valid: Scratchpad[np.int32] = np.int32(rows)):
            xs = x[:valid] if rows > 1 and not args.unbounded else x
            return G.step(xs, state, heads=H, key_heads=KH, key_dim=dk, value_dim=dv, words=words)

        def reference(self, x, state, valid=rows):
            return gdn_reference(x, state, words, heads=H, key_heads=KH, key_dim=dk, value_dim=dv, n=n,
                                 valid=int(valid))

    g = Graph()
    cg = g.compile(x=(rows, W), state=((H * dk * dv,), np.float32))
    m = export_graph(cg, args.out, "gdn", dict(rows=rows, heads=H, key_heads=KH))
    print(cg.plan.report("gdn"))
    x = (rng.standard_normal((rows, W)) * 1.5).astype(bfloat16)
    state = (rng.standard_normal(H * dk * dv) * 0.3).astype(np.float32)
    write_check(g, m, args.out, dict(x=x, state=state), dict(valid=args.valid or rows) if rows > 1 and not args.unbounded else None)
    print(f"{args.out}: GatedDeltaRule of {rows} rows, {H} value heads")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of one GatedDeltaRule and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--rows", type=int, default=1)
    c.add_argument("--valid", type=int, default=0)
    c.add_argument("--unbounded", action="store_true", help="x not bounded per call (every row valid)")
    c.add_argument("--heads", type=int, default=32)
    c.add_argument("--key-heads", type=int, default=16)
    c.add_argument("--conv", type=int, default=0, help="the convolution's taps in the operator (decode)")
    a = ap.parse_args(argv)
    return _check(a)


if __name__ == "__main__":
    sys.exit(main())
