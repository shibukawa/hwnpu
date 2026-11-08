"""A mixture of experts for prompt chunks (Qwen3.6-35B-A3B's), the host routing the chunk.

A decode step routes on the device (iron_moe.py): command B's scratchpad (16 words) takes the
8 experts' offsets. A prompt chunk of T tokens needs 8 T, so here the host routes it, between
two commands of the layer:

  A    ... the mixer, the residual h, the post-attention norm hn, the router's logits (T, n)
  host route: the top 8 of each token (route_chunk), the 8 T rows sorted by expert (plan), the
       experts' row counts and x's stream (stream_x): each expert's rows' k-blocks in turn
  B    MoEGroupGateUp: every expert's gate and up stream through the cores once, each expert's
       rows through its blocks as they pass (a = silu(G x) U x, the rows sorted by expert); then
       MoEGroupDown: every expert's down the same way (y_e = D a)
  host combine: h + the shared expert's output times its gate + each token's 8 rows of y times
       their routing weights (combine)

The streams are static: 8 T rows whatever the routing, the counts (data, the first objects on x's
channel) telling the cores how many of them each expert has. A core keeps an expert's rows'
accumulators (MOE_CAP rows); the host splits a chunk whose routing gives an expert more.

    python tools/amd/iron_moep.py check OUT [--experts 32] [--tokens 16]
    go run ./tools/amd/irongraph -tol 0.05 OUT
"""

import functools
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOP = 8
CAP = 256  # rows of one expert a core keeps (kernels/moe_rows.cc's MOE_CAP): a 256-token chunk's all
COUNTS = 512  # int32 words of the counts: each expert's (256 at most), then the padding rows' (256)
PAD = 256  # the counts' word of the padding rows


def route_chunk(logits, experts: int, top: int = TOP):
    """Each row's chosen experts (descending logit, ties to the lower index), their routing
    weights (softmax over the chosen, float32) and the shared expert's gate (sigmoid of logit
    ``experts``), as MoERoute does one row."""
    x = np.asarray(logits, np.float32)
    ids = np.argsort(-x[:, :experts], axis=1, kind="stable")[:, :top]
    v = np.take_along_axis(x, ids, axis=1)
    e = np.exp(v - v[:, :1])
    w = (e / e.sum(1, keepdims=True)).astype(bfloat16).astype(np.float32)
    shared = (1 / (1 + np.exp(-x[:, experts]))).astype(bfloat16).astype(np.float32)
    return ids.astype(np.int32), w, shared


def plan(ids, experts: int, C: int | None = None):
    """The rows sorted by expert: ``rows`` (8 C) the token of each, ``slot`` (T, 8) the row of
    each choice, and the counts (COUNTS int32: each expert's rows, then at PAD the padding rows'
    after them, 8 (C - T), whose x is zero and output unread)."""
    T, k = ids.shape
    C = T if C is None else C
    flat = ids.reshape(-1)
    order = np.argsort(flat, kind="stable")  # by expert, then by token
    rows = np.concatenate([order // k, np.full(k * (C - T), -1)]).astype(np.int32)
    slot = np.empty(T * k, np.int32)
    slot[order] = np.arange(T * k)
    counts = np.zeros(COUNTS, np.int32)
    counts[:experts] = np.bincount(flat, minlength=experts)[:experts]
    counts[PAD] = k * (C - T)
    return rows, slot.reshape(T, k), counts


def stream_x(hn, rows, counts, kt: int = 256):
    """MoEGroupGateUp's x: each expert's rows of hn, a k-block of all of them at a time, twice
    (the gate's pass over the expert's k-blocks, then the up's)."""
    hn = np.asarray(hn, bfloat16)
    K = hn.shape[1]
    out, at = [], 0
    for n in list(counts[:PAD]) + [counts[PAD]]:  # the padding rows' x last, zero
        if n:
            r = rows[at : at + n]
            x = np.where((r >= 0)[:, None], hn[np.maximum(r, 0)], bfloat16(0))
            x = x.reshape(n, K // kt, kt).transpose(1, 0, 2).reshape(-1)
            out += [x, x]
            at += n
    return np.concatenate(out) if out else np.zeros(0, bfloat16)


def counts_words(counts) -> np.ndarray:
    """The counts as the bf16 elements the x channel carries (their bytes)."""
    return np.asarray(counts, np.int32).view(bfloat16)


def combine(h, y, slot, w, shared_out, shared_gate):
    """h + shared_gate * shared_out + sum_k w[t, k] y[slot[t, k]], float32, rounded to bf16."""
    out = np.asarray(h, np.float32).copy()
    out += np.asarray(shared_gate, np.float32)[:, None] * np.asarray(shared_out, np.float32)
    yv = np.asarray(y, np.float32)
    for k in range(slot.shape[1]):
        out += w[:, k : k + 1] * yv[slot[:, k]]
    return out.astype(bfloat16)


def rows_of(xs, counts, K: int, kt: int = 256):
    """stream_x's inverse: the rows (8 T, K), sorted by expert (from the gate's pass)."""
    xs = np.asarray(xs)
    out, at = [], 0
    for n in list(counts[:PAD]) + [counts[PAD]]:
        if n:
            out.append(xs[at : at + n * K].reshape(K // kt, n, kt).transpose(1, 0, 2).reshape(n, K))
            at += 2 * n * K
    return np.concatenate(out)


@functools.cache
def group_operators(experts: int = 256, cap: int = CAP):
    """MoEGroupGateUp and MoEGroupDown for tables of ``experts`` routed experts (the shared one
    after them is not read)."""
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import Buffer, In, ObjectFifo, Out, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator
    from iron_q4 import BLOCK_BYTES as B, CB, K_TILE, dequantize
    from iron_moeprobe import silu

    MT = CB // 2
    cols = 8
    blk_t = np.ndarray[(B,), np.dtype[np.uint8]]
    half_t = np.ndarray[(MT,), np.dtype[bfloat16]]
    cnt_t = np.ndarray[(COUNTS,), np.dtype[np.int32]]
    ST = MT // 2  # a gate-up core's strip of a block (moe_rows.cc's ST)
    strip_t = np.ndarray[(ST,), np.dtype[bfloat16]]
    acc_t = np.ndarray[(cap * ST,), np.dtype[np.float32]]

    def kernel(name, args):
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        return ExternalFunction(
            name, arg_types=args, source_file=str(here / "moe_rows.cc"),
            include_dirs=[str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
            compile_flags=[f"-DMOE_CAP={cap}"], object_file_name=f"moe_rows_{cap}.o")

    def i32(v):
        return index.casts(T.i32(), v)

    def i32c(v: int):
        from aie.extras.dialects import arith

        return arith.constant(v, T.i32())

    def count_of(cnt, e):
        return index.casts(T.index(), cnt[e])

    def halves(n):
        """n // 2 and n % 2 (index values)."""
        from aie.extras.dialects import arith

        two = arith.constant(2, index=True)
        return arith.divui(n, two), arith.remui(n, two)

    def arith_mul2(p):
        from aie.extras.dialects import arith

        return arith.muli(p, arith.constant(2, index=True))

    def arith_sub1(n):
        from aie.extras.dialects import arith

        return arith.subi(n, arith.constant(1, index=True))

    class MoEGroupGateUp(Operator):
        """``a = silu(G_e x) * (U_e x)`` for each of the A rows, the rows sorted by expert and
        ``x`` in stream_x's order, ``counts`` the experts' rows. Column c reads every expert's
        column blocks 2c and 2c + 1 (gate and up of out-features 64 c ..), each block to two of
        its four cores, a strip (16 out-features) each: a core keeps 16 accumulators of the gate
        and of the up a row, so cap rows of an expert fit; a is (A, NE)."""

        rows: int = param(array=True)  # blocks in the table
        A: int = param(array=True)  # rows: 8 a token
        K: int = param(array=True)
        NE: int = param(array=True)
        columns: int = param(default=cols, init=False, array=True)
        AK: int = param(derive=lambda op: 2 * op.A * op.K, repr=False)  # x twice
        ANE: int = param(derive=lambda op: op.A * op.NE, repr=False)

        W = In(rows, B, dtype=np.uint8, obj_shape=(2, B), per=(columns,))
        counts = In(2 * COUNTS, obj_shape=(K_TILE,), per=(columns,), replicate=True)  # 4 objects
        x = In(AK, obj_shape=(K_TILE,), per=(columns,), replicate=True)
        a = Out(ANE, obj_shape=(CB,), per=(columns,))

        def validate(self):
            if self.NE != cols * CB:
                raise ValueError(f"MoEGroupGateUp: NE {self.NE} is 64 out-features a column")

        def expert_bytes(self) -> int:
            return 2 * self.NE // CB * (self.K // K_TILE) * 2 * B

        def build(self) -> Built:
            kbs = self.K // K_TILE
            k_counts = kernel("moe_rows_counts", [self.counts.obj_type, cnt_t, np.int32, np.int32])
            k_pad = kernel("moe_gu_pad", [strip_t])
            k_zero = kernel("moe_gu_zero", [acc_t, acc_t])
            k_acc = kernel("moe_gu_acc", [blk_t, self.x.obj_type, acc_t, np.int32, np.int32])
            k_acc2 = kernel("moe_gu_acc2", [blk_t, self.x.obj_type, self.x.obj_type, acc_t, np.int32, np.int32])
            k_glu = kernel("moe_gu_glu", [acc_t, acc_t, np.int32, strip_t])
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(cols)]
            w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], depths=[2, 2], tile=AnyMemTile)
                    for f in w_l3]
            x_l3 = [ObjectFifo(self.x.obj_type, depth=2) for _ in range(cols)]
            x_l1 = [f.cons().forward(obj_type=self.x.obj_type, depth=4, tile=AnyMemTile) for f in x_l3]
            a_l3 = [ObjectFifo(self.a.obj_type, depth=2) for _ in range(cols)]
            a_l1 = [f.prod().join([0, ST, 2 * ST, 3 * ST], obj_types=[strip_t] * 4, depths=[2] * 4,
                                  tile=AnyMemTile) for f in a_l3]

            def core(w_f, x_f, y_f, cnt, g, u, kc, kz, kacc, kacc2, kglu, kpad, h):
                for part in range(4):
                    c = x_f.acquire(1)
                    kc(c, cnt, i32c(part), i32c(COUNTS // 4))
                    x_f.release(1)
                kz(g, u)
                for e in range_(experts):
                    n = count_of(cnt, e)
                    two, one = halves(n)  # rows two at a time, then the odd one
                    for acc in (g, u):  # the table's order: the gate's k-blocks, then the up's
                        for _ in range_(kbs):
                            b = w_f.acquire(1)
                            for p in range_(two):
                                xt = x_f.acquire(2)
                                kacc2(b, xt[0], xt[1], acc, i32(arith_mul2(p)), i32c(h))
                                x_f.release(2)
                            for _ in range_(one):
                                xt = x_f.acquire(1)
                                kacc(b, xt, acc, i32(arith_sub1(n)), i32c(h))
                                x_f.release(1)
                            w_f.release(1)
                    for r in range_(n):
                        y = y_f.acquire(1)
                        kglu(g, u, i32(r), y)
                        y_f.release(1)
                pad = count_of(cnt, PAD)  # the padding rows: their x read past, zeros out
                for _ in range_(2 * kbs):
                    for _ in range_(pad):
                        x_f.acquire(1)
                        x_f.release(1)
                for _ in range_(pad):
                    y = y_f.acquire(1)
                    kpad(y)
                    y_f.release(1)

            workers = []
            for i in range(cols):
                for j in range(2):  # the block, then its strip: out-features 32 j + 16 h ..
                    for h in range(2):
                        cnt, g, u = Buffer(cnt_t), Buffer(acc_t), Buffer(acc_t)
                        fn = lambda w_f, x_f, y_f, cnt, g, u, kc, kz, ka, ka2, kg, kp, h=h: core(  # noqa: E731
                            w_f, x_f, y_f, cnt, g, u, kc, kz, ka, ka2, kg, kp, h)
                        workers.append(Worker(fn, [w_l1[i][j].cons(), x_l1[i].cons(), a_l1[i][2 * j + h].prod(),
                                                   cnt, g, u, k_counts, k_zero, k_acc, k_acc2, k_glu, k_pad]))
            self.W.bind([f.prod() for f in w_l3])
            self.counts.bind([f.prod() for f in x_l3])
            self.x.bind([f.prod() for f in x_l3])
            self.a.bind([f.cons() for f in a_l3])
            return Built(workers, [])

        def sequence(self, rt):
            A, K, NE = self.A, self.K, self.NE
            eb = self.expert_bytes()
            cbb = (K // K_TILE) * 2 * B  # a column block's bytes
            total = self.rows * B
            drains = [self.a.drain(TAP((A * NE,), c * CB, [A, CB], [NE, 1]), lane=c, wait=True,
                                   managed=False) for c in range(cols)]
            for c in range(cols):
                self.counts.fill(TAP((2 * COUNTS,), 0, [2 * COUNTS], [1]), lane=c, managed=False)
                self.x.fill(TAP((2 * A * K,), 0, [2 * A * K], [1]), lane=c, managed=False)
            run, piece = 2 * cbb, 2048  # each expert's two column blocks, in pieces a BD dimension takes
            for c in range(cols):
                self.W.fill(TAP((total,), 2 * c * cbb, [experts, run // piece, piece], [eb, piece, 1]),
                            lane=c, managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * 2 * self.A * self.NE * self.K

        def reference(self, W, counts, x):
            eb = self.expert_bytes()
            flat = np.asarray(W, np.uint8).reshape(-1)
            allc = np.asarray(counts).view(np.int32)
            cnt = allc[:experts]
            xr = rows_of(np.asarray(x, np.float64), allc, self.K)
            out, at = [], 0
            for e, n in enumerate(cnt):
                if not n:
                    continue
                gu = dequantize(flat[e * eb : (e + 1) * eb], 2 * self.NE, self.K).astype(np.float64)
                gu = gu.reshape(self.NE // CB, 2, CB, self.K)
                xe = xr[at : at + n]
                g = xe @ gu[:, 0].reshape(self.NE, self.K).T
                u = xe @ gu[:, 1].reshape(self.NE, self.K).T
                out.append(silu(g) * u)
                at += n
            out.append(np.zeros((allc[PAD], self.NE)))
            return np.concatenate(out).astype(bfloat16).reshape(-1)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="q4nx partial sums in bf16, SiLU's tanh")

    class MoEGroupDown(Operator):
        """``y = D_e a`` for each of the A rows (sorted by expert, a (A, KE)): column c keeps
        out-features 256 c .. 256 c + 255, a column block (64) of every expert at a time, its
        two blocks of 32 each to two of its four cores, a strip (16) each; y is (A, N), without
        the routing weights or the residual (the host's combine)."""

        aiecc_flags = ("--reclaim-runtime-bds",)

        rows: int = param(array=True)
        A: int = param(array=True)
        KE: int = param(array=True)
        N: int = param(array=True)
        columns: int = param(default=cols, init=False, array=True)
        AKE: int = param(derive=lambda op: op.A * op.KE, repr=False)
        AN: int = param(derive=lambda op: op.A * op.N, repr=False)

        W = In(rows, B, dtype=np.uint8, obj_shape=(2, B), per=(columns,))
        counts = In(2 * COUNTS, obj_shape=(KE,), per=(columns,), replicate=True)  # 2 objects
        a = In(AKE, obj_shape=(KE,), per=(columns,), replicate=True)
        y = Out(AN, obj_shape=(CB,), per=(columns,))

        def validate(self):
            if self.KE != 2 * K_TILE:
                raise ValueError(f"MoEGroupDown: KE {self.KE} is two k-blocks")
            if self.N % (cols * CB):
                raise ValueError(f"MoEGroupDown: N {self.N} is whole column blocks a column")

        def expert_bytes(self) -> int:
            return self.N // CB * (self.KE // K_TILE) * 2 * B

        def build(self) -> Built:
            per = self.N // CB // cols  # column blocks a column
            k_counts = kernel("moe_rows_counts", [self.counts.obj_type, cnt_t, np.int32, np.int32])
            k_down = kernel("moe_dn", [blk_t, blk_t, self.a.obj_type, strip_t, np.int32])
            k_down2 = kernel("moe_dn2", [blk_t, blk_t, self.a.obj_type, self.a.obj_type, strip_t, strip_t,
                                         np.int32])
            k_pad = kernel("moe_gu_pad", [strip_t])
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(cols)]
            w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], depths=[2, 2], tile=AnyMemTile)
                    for f in w_l3]
            x_l3 = [ObjectFifo(self.a.obj_type, depth=2) for _ in range(cols)]
            x_l1 = [f.cons().forward(obj_type=self.a.obj_type, depth=4, tile=AnyMemTile) for f in x_l3]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(cols)]
            y_l1 = [f.prod().join([0, ST, 2 * ST, 3 * ST], obj_types=[strip_t] * 4, depths=[4] * 4,
                                  tile=AnyMemTile) for f in y_l3]

            def core(w_f, x_f, y_f, cnt, kc, kd, kd2, kpad, h):
                for part in range(2):
                    c = x_f.acquire(1)
                    kc(c, cnt, i32c(part), i32c(COUNTS // 2))
                    x_f.release(1)
                pad = count_of(cnt, PAD)
                for _ in range_(per):
                    for e in range_(experts):
                        n = count_of(cnt, e)
                        two, one = halves(n)
                        b = w_f.acquire(2)  # k-blocks 0 and 1 of the column block
                        for _ in range_(two):
                            xt, y = x_f.acquire(2), y_f.acquire(2)
                            kd2(b[0], b[1], xt[0], xt[1], y[0], y[1], i32c(h))
                            x_f.release(2)
                            y_f.release(2)
                        for _ in range_(one):
                            xt, y = x_f.acquire(1), y_f.acquire(1)
                            kd(b[0], b[1], xt, y, i32c(h))
                            x_f.release(1)
                            y_f.release(1)
                        w_f.release(2)
                    for _ in range_(pad):  # the padding rows: x read past, zeros out
                        x_f.acquire(1)
                        y = y_f.acquire(1)
                        kpad(y)
                        x_f.release(1)
                        y_f.release(1)

            workers = []
            for i in range(cols):
                for j in range(2):  # the block, then its strip: out-features 32 j + 16 h ..
                    for h in range(2):
                        cnt = Buffer(cnt_t)
                        fn = lambda w_f, x_f, y_f, cnt, kc, kd, kd2, kp, h=h: core(  # noqa: E731
                            w_f, x_f, y_f, cnt, kc, kd, kd2, kp, h)
                        workers.append(Worker(fn, [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][2 * j + h].prod(),
                                                   cnt, k_counts, k_down, k_down2, k_pad]))
            self.W.bind([f.prod() for f in w_l3])
            self.counts.bind([f.prod() for f in x_l3])
            self.a.bind([f.prod() for f in x_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            A, KE, N = self.A, self.KE, self.N
            per = N // CB // cols
            eb = self.expert_bytes()
            cbb = (KE // K_TILE) * 2 * B
            total = self.rows * B
            drains = [self.y.drain(TAP((A * N,), c * per * CB, [per, A, CB], [CB, N, 1]), lane=c,
                                   wait=True, managed=False) for c in range(cols)]
            for c in range(cols):
                self.counts.fill(TAP((2 * COUNTS,), 0, [2 * COUNTS], [1]), lane=c, managed=False)
                self.a.fill(TAP((A * KE,), 0, [per, A, KE], [0, KE, 1]), lane=c, managed=False)
            piece = 2048
            for c in range(cols):  # a column block of every expert at a time
                self.W.fill(TAP((total,), c * per * cbb, [per, experts, cbb // piece, piece],
                                [cbb, eb, piece, 1]), lane=c, managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * self.A * self.N * self.KE

        def reference(self, W, counts, a):
            eb = self.expert_bytes()
            flat = np.asarray(W, np.uint8).reshape(-1)
            allc = np.asarray(counts).view(np.int32)
            cnt = allc[:experts]
            av = np.asarray(a, np.float64).reshape(self.A, self.KE)
            out, at = [], 0
            for e, n in enumerate(cnt):
                if not n:
                    continue
                d = dequantize(flat[e * eb : (e + 1) * eb], self.N, self.KE).astype(np.float64)
                out.append(av[at : at + n] @ d.T)
                at += n
            out.append(np.zeros((allc[PAD], self.N)))
            return np.concatenate(out).astype(bfloat16).reshape(-1)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="q4nx partial sums in bf16")

    return MoEGroupGateUp, MoEGroupDown


def _check(args) -> int:
    """A graph of MoEGroupGateUp then MoEGroupDown on random tables and a random routing."""
    import iron
    from iron_graph import bind_device, export_graph, write_check
    from iron_moeprobe import rand_q4
    from iron_q4 import BLOCK_BYTES

    bind_device("npu2", 8)
    E, T, K, NE, N = args.experts, args.tokens, 2048, 512, 2048
    GU, D = group_operators(E)
    rng = np.random.default_rng(args.seed)
    gu = rand_q4(rng, 2 * NE, K, E, 0.02)
    down = rand_q4(rng, N, NE, E, 0.02)
    logits = rng.standard_normal((T, E + 1)).astype(np.float32)
    logits[:, : args.hot] += 4  # experts nearly every token chooses
    ids, w, sg = route_chunk(logits, E)
    real = T - args.pad
    rows, slot, counts = plan(ids[:real], E, T)
    if counts[:E].max() > CAP:
        raise SystemExit(f"an expert has {counts[:E].max()} rows, past {CAP}")
    hn = (rng.standard_normal((T, K)) * 0.5).astype(bfloat16)
    xs = stream_x(hn, rows, counts)
    A = TOP * T
    gu_rows, d_rows = gu.size // BLOCK_BYTES, down.size // BLOCK_BYTES

    class Graph(iron.Graph):
        def __init__(self):
            self.gu = iron.param(gu)
            self.down = iron.param(down)

        def body(self, counts, x):
            a = GU.step(self.gu, counts, x, rows=gu_rows, A=A, K=K, NE=NE)
            return D.step(self.down, counts, a, rows=d_rows, A=A, KE=NE, N=N)

        def reference(self, counts, x):
            a = GU(rows=gu_rows, A=A, K=K, NE=NE).reference(gu, counts, x)
            return D(rows=d_rows, A=A, KE=NE, N=N).reference(down, counts, a)

    g = Graph()
    cg = g.compile(counts=(2 * COUNTS,), x=(2 * A * K,))
    m = export_graph(cg, args.out, "moep", dict(experts=E, tokens=T))
    print(cg.plan.report("moep"))
    write_check(g, m, args.out, dict(counts=counts_words(counts), x=xs), None)
    print(f"{args.out}: MoEGroupGateUp and MoEGroupDown, {T} tokens over {E} experts (most {counts[:E].max()} rows)")
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of the two operators and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--experts", type=int, default=32)
    c.add_argument("--tokens", type=int, default=16)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--pad", type=int, default=3, help="tokens of the chunk past the call's (padding)")
    c.add_argument("--hot", type=int, default=0, help="experts favoured by every token (rows past 128)")
    return _check(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
