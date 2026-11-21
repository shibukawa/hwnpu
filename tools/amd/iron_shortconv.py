"""LFM2/LFM3's gated short convolution of a decode step as one operator (kernels/short_conv.cc).

A conv layer's mixer is ``out_proj(C * conv(B * x))``, B, C and x the three parts of in_proj's
row and conv a depth-wise causal filter of ``K`` taps (``conv_L_cache``) over the ``K - 1`` rows
before. IRON's graph (``F.causal_conv1d``) runs a decode step's as about 16 operators: copies into
its window, a multiply and an add a tap, and the gate's two multiplies. Each is a
reconfiguration of the array (about 40 us and 2.3 us a KB of its configuration, docs/design/amd.md), together about 1.1 ms of a layer's. ``ShortConvStep`` is one: each of ``n`` columns' core
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
def short_conv_operator(n: int = COLUMNS, slots: int = 0):
    """``y, final = ShortConvStep(p, taps, state)``: a decode step's gated short convolution over
    ``n`` columns of one core each.

    ``slots`` > 0: ``state`` is the convolution's whole checkpoint state of that many slots (each
    K - 1 rows), read at the per-call element offset ``read_at``, and ``final`` the same buffer
    (an InOut) written at ``write_at``, in place of copies of the slot in and out around the
    operator (as iron_gdn's slots). The two may be one slot: a column reads its rows before it
    writes them."""
    from aie.iron import InOut
    from aie.utils.compile.jit.params import per_call
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
        if slots:
            total: int = param(derive=lambda op: slots * (op.taps_n - 1) * op.width, repr=False)
            state = In(total, obj_shape=(so,), per=(columns,), depth=1)
            y = Out(1, width, obj_shape=(w,), per=(columns,))
            final = InOut(total, obj_shape=(so,), per=(columns,), depth=1)
            read_at = per_call(derive=lambda op: 0)  # elements: the slot read
            write_at = per_call(derive=lambda op: 0)  # and the one written
        else:
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
            if slots:
                rd = self.read_at if self.read_at.per_call else None
                wr = self.write_at if self.write_at.per_call else None
                st = lambda c: TAP((self.total,), c * W, [K - 1, W], [E, 1])  # noqa: E731
            else:
                rd = wr = None
                st = lambda c: TAP((K - 1, E), c * W, [K - 1, W], [E, 1])  # noqa: E731
            for c in range(n):
                # B, C and x of the column's channels, then its taps: one object
                self.p.fill(TAP((1, 3 * E), c * W, [3, W], [E, 1]), lane=c, managed=False)
                self.taps.fill(TAP((K, E), c * W, [K, W], [E, 1]), lane=c, managed=False)
                self.state.fill(st(c), lane=c, offset_parameter=rd, managed=False)
            drains = []
            for c in range(n):
                drains.append(self.y.drain(TAP((1, E), c * W, [W], [1]), lane=c, wait=True, managed=False))
                drains.append(self.final.drain(st(c), lane=c, offset_parameter=wr, wait=True, managed=False))
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * (self.taps_n + 2) * self.width

        if slots:
            def reference(self, p, taps, state, final, *, read_at=0, write_at=0):
                K, E = self.taps_n, self.width
                size = (K - 1) * E
                st = np.asarray(state).reshape(-1)[int(read_at) : int(read_at) + size].reshape(K - 1, E)
                y, fin = short_conv_reference(p, taps, st)
                out = np.array(final, copy=True).reshape(-1)
                out[int(write_at) : int(write_at) + size] = np.asarray(fin).reshape(-1)
                return y, out
        else:
            def reference(self, p, taps, state):
                return short_conv_reference(p, taps, state)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.01, 1e-3, note="float32 sums, rounded once to bf16")

    return ShortConvStep


def short_conv_chunk_reference(p, taps, state, n: int):
    """ShortConvChunk in numpy: the first ``n`` rows of ``p`` (rows, 3 width) each a decode step's
    (short_conv_reference), the state carried from row to row. y's rows past n are zero (the
    device's too)."""
    p = np.asarray(p, np.float32)
    y = np.zeros((p.shape[0], p.shape[1] // 3), bfloat16)
    s = np.asarray(state, np.float32)
    for r in range(n):
        yr, fin = short_conv_reference(p[r], taps, s)
        y[r] = yr[0]
        s = np.asarray(fin, np.float32)
    return y, bf16(s)


def short_conv_chunk_silu_reference(x, taps, state, n: int, seg: int, raw: int):
    """ShortConvChunk with ``raw`` in numpy: y = silu(conv(x)) but on each column's last ``raw`` of
    ``seg`` channels, conv's sums float32 rounded once to bf16, silu float32 rounded once."""
    x = np.asarray(x, np.float32)
    t = np.asarray(taps, np.float32)
    K = t.shape[0]
    h = np.concatenate([np.asarray(state, np.float32), x[:n]])
    y = np.zeros(x.shape, bfloat16)
    conv = np.zeros((n, x.shape[1]), np.float32)
    for k in range(K):
        conv += t[k] * h[k : k + n]
    v = bf16(conv).astype(np.float32).reshape(n, -1, seg)
    out = v.copy()
    out[..., : seg - raw] = v[..., : seg - raw] / (1 + np.exp(-v[..., : seg - raw]))
    y[:n] = bf16(out.reshape(n, -1))
    return y, bf16(h[n : n + K - 1])


@functools.cache
def short_conv_chunk_operator(n: int = COLUMNS, tile: int = 8, raw: int | None = None):
    """``y, final = ShortConvChunk(p, taps, state)``: a prompt chunk's gated short convolution, the
    call's rows (bounded) in turn, each of ``n`` columns' core its ``width / n`` channels of every
    row in tiles of ``tile`` rows. The DMAs move the chunk's rows whatever the call's; the cores
    compute the call's.

    With ``raw`` (Qwen3.5's Gated DeltaNet, iron_gdn's layout, a column's channels its ``seg``):
    ``p`` is x alone (rows, width), u = x and y = silu(conv(u)) but on each column's last ``raw``
    channels (a and b), as IRON's convolution then GdnSiLU."""
    plain = raw is not None
    parts = 1 if plain else 3
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import Buffer, In, ObjectFifo, Out, TaskGroup, Worker
    from aie.iron.controlflow import range_
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param, per_call
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator, Rtp

    class ShortConvChunk(Operator):
        """``y, final = ShortConvChunk(p, taps, state)``: ``p`` (rows, 3 width) in_proj's rows (B, C,
        x), bounded per call; ``taps`` (K, width); ``state`` (K - 1, width) the rows before the
        call. ``y`` (rows, width) C * conv(B * x) of the call's rows, ``final`` (K - 1, width) the
        rows ending the call."""

        rows: int = param(array=True)
        width: int = param(array=True)
        taps_n: int = param(array=True)
        columns: int = param(default=n, init=False, array=True)
        tile_rows: int = param(default=tile, init=False, array=True)
        w: int = param(derive=lambda op: op.width // n, array=True)
        three: int = param(derive=lambda op: parts * op.width, repr=False)
        rest: int = param(derive=lambda op: op.taps_n - 1, repr=False)
        po: int = param(derive=lambda op: tile * parts * (op.width // n), repr=False)
        yo: int = param(derive=lambda op: tile * (op.width // n), repr=False)
        # a core's object on the second channel: the taps, then the state
        to: int = param(derive=lambda op: (2 * op.taps_n - 1) * (op.width // n), repr=False)
        so: int = param(derive=lambda op: (op.taps_n - 1) * (op.width // n), repr=False)
        valid = per_call(bounds=rows)
        count = per_call(derive=lambda op: op.valid)

        p = In(rows, three, obj_shape=(po,), per=(columns,))
        taps = In(taps_n, width, obj_shape=(to,), per=(columns,))
        state = In(rest, width, obj_shape=(to,), per=(columns,))  # on taps' channel, after them
        y = Out(rows, width, obj_shape=(yo,), per=(columns,))
        final = Out(rest, width, obj_shape=(so,), per=(columns,))

        def validate(self):
            if self.width % (n * 16) or self.taps_n < 2 or self.rows % tile:
                raise ValueError(f"ShortConvChunk: {self.width} channels over {n} columns of whole "
                                 f"vectors, {self.taps_n} taps, {self.rows} rows of {tile}-row tiles")

        def build(self) -> Built:
            W, K = self.w, self.taps_n
            kdir = Path(config.aie_kernels_dir())
            here = Path(__file__).resolve().parent / "kernels"
            p_t = np.ndarray[(tile * parts * W,), np.dtype[bfloat16]]
            t_t = np.ndarray[((2 * K - 1) * W,), np.dtype[bfloat16]]
            y_t = np.ndarray[(tile * W,), np.dtype[bfloat16]]
            s_t = np.ndarray[((K - 1) * W,), np.dtype[bfloat16]]
            h_t = np.ndarray[((K - 1 + tile) * W,), np.dtype[bfloat16]]
            src, flags = str(here / "short_conv.cc"), [f"-DSC_W={W}", f"-DSC_K={K}", f"-DSC_T={tile}"]
            obj = f"short_conv_chunk_{W}_{K}_{tile}.o"
            if plain:
                flags += ["-DSC_PLAIN", f"-DSC_RAW={raw}", "-ffp-contract=off", "-fno-fast-math"]
                obj = f"short_conv_chunk_{W}_{K}_{tile}_silu{raw}.o"
            inc = [str(kdir), config.cxx_header_path()]
            k_begin = ExternalFunction("short_conv_begin", arg_types=[t_t, h_t], source_file=src,
                                       include_dirs=inc, compile_flags=flags, object_file_name=obj)
            k_tile = ExternalFunction("short_conv_tile", arg_types=[p_t, t_t, h_t, y_t, np.int32, np.int32],
                                      source_file=src, include_dirs=inc, compile_flags=flags,
                                      object_file_name=obj)
            k_end = ExternalFunction("short_conv_end", arg_types=[h_t, s_t], source_file=src,
                                     include_dirs=inc, compile_flags=flags, object_file_name=obj)
            p_f = [ObjectFifo(p_t, depth=2) for _ in range(n)]
            t_f = [ObjectFifo(t_t, depth=1) for _ in range(n)]
            y_f = [ObjectFifo(y_t, depth=2) for _ in range(n)]
            f_f = [ObjectFifo(s_t, depth=1) for _ in range(n)]
            rtps = [Rtp([self.count]) for _ in range(n)]
            tiles = self.rows // tile

            def core(pf, tf, yf, ff, hist, kb, kt, ke, rtp):
                rows = rtp.read()[self.count]
                ts = tf.acquire(1)
                kb(ts, hist)
                for t in range_(tiles):
                    t32 = index.casts(T.i32(), t)
                    po, yo = pf.acquire(1), yf.acquire(1)
                    kt(po, ts, hist, yo, t32, rows)
                    pf.release(1)
                    yf.release(1)
                fo = ff.acquire(1)
                ke(hist, fo)
                ff.release(1)
                tf.release(1)

            workers = [Worker(core, [p_f[c].cons(), t_f[c].cons(), y_f[c].prod(), f_f[c].prod(),
                                     Buffer(h_t, name=f"short_conv_hist_{c}"), k_begin, k_tile, k_end, rtps[c]])
                       for c in range(n)]
            self.p.bind([f.prod() for f in p_f])
            self.taps.bind([f.prod() for f in t_f])
            self.state.bind([f.prod() for f in t_f])
            self.y.bind([f.cons() for f in y_f])
            self.final.bind([f.cons() for f in f_f])
            return Built(workers, rtps)

        def sequence(self, rt):
            R, E, W, K = self.rows, self.width, self.w, self.taps_n
            tiles = R // tile
            tg = TaskGroup()
            for c in range(n):
                self.taps.fill(TAP((K, E), c * W, [K, W], [E, 1]), lane=c, group=tg)
                self.state.fill(TAP((K - 1, E), c * W, [K - 1, W], [E, 1]), lane=c, group=tg)
                # every row's B, C and x of the column's channels, a tile of rows an object
                if plain:
                    self.p.fill(TAP((R, E), c * W, [tiles, tile, W], [tile * E, E, 1]), lane=c, group=tg)
                else:
                    self.p.fill(TAP((R, 3 * E), c * W, [tiles, tile, 3, W], [tile * 3 * E, 3 * E, E, 1]),
                                lane=c, group=tg)
            for c in range(n):
                self.y.drain(TAP((R, E), c * W, [tiles, tile, W], [tile * E, E, 1]), lane=c, group=tg,
                             wait=True)
                self.final.drain(TAP((K - 1, E), c * W, [K - 1, W], [E, 1]), lane=c, group=tg, wait=True)
            tg.finish()

        def ops(self) -> int:
            return 2 * (self.taps_n + 2) * self.width * self.rows

        def reference(self, p, taps, state):
            if plain:
                return short_conv_chunk_silu_reference(p, taps, state, int(self.valid), self.w, raw)
            return short_conv_chunk_reference(p, taps, state, int(self.valid))

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.01, 1e-3, note="float32 sums, rounded once to bf16")

    return ShortConvChunk


@functools.cache
def gdn_conv_chunk_operator(n: int = COLUMNS, parts: int = 2):
    """``y, final = GdnConvChunk(x, taps, state)``: Gated DeltaNet's short convolution of a prompt
    chunk (Qwen3.5's, iron_gdn's layout: a column's ``seg`` channels, the last ``raw`` a and b),
    then silu but on a and b: IRON's convolution (about 9 operators) and GdnSiLU as one. Each of
    ``n`` columns' ``parts`` cores its ``seg / parts`` channels, a row at a time (a memtile cuts a
    column's row into the cores' parts); the call's rows are bounded, the DMAs move the chunk's."""
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import Buffer, In, ObjectFifo, Out, TaskGroup, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param, per_call
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator, Rtp

    P = parts

    class GdnConvChunk(Operator):
        """``y, final = GdnConvChunk(x, taps, state)``: ``x`` (rows, width) the fused projection's
        q, k, v, a, b (bounded per call), ``taps`` (K, width), ``state`` (K - 1, width); ``y``
        silu(conv(x)) but on each column's last ``raw`` channels, ``final`` the rows ending the
        call."""

        rows: int = param(array=True)
        width: int = param(array=True)
        taps_n: int = param(array=True)
        raw: int = param(array=True)
        columns: int = param(default=n, init=False, array=True)
        seg: int = param(derive=lambda op: op.width // n, array=True)
        rest: int = param(derive=lambda op: op.taps_n - 1, repr=False)
        to: int = param(derive=lambda op: (2 * op.taps_n - 1) * (op.width // n), repr=False)
        so: int = param(derive=lambda op: (op.taps_n - 1) * (op.width // n), repr=False)
        valid = per_call(bounds=rows)
        count = per_call(derive=lambda op: op.valid)

        x = In(rows, width, obj_shape=(seg,), per=(columns,))
        taps = In(taps_n, width, obj_shape=(to,), per=(columns,))
        state = In(rest, width, obj_shape=(to,), per=(columns,))  # on taps' channel, after them
        y = Out(rows, width, obj_shape=(seg,), per=(columns,))
        final = Out(rest, width, obj_shape=(so,), per=(columns,))

        def validate(self):
            seg = self.width // n
            if self.width % n or seg % (P * 16) or self.raw % 16 or self.raw > seg // P:
                raise ValueError(f"GdnConvChunk: {self.width} channels over {n} columns of {P} cores of "
                                 f"whole vectors, {self.raw} raw in the last")

        def build(self) -> Built:
            seg, K = self.seg, self.taps_n
            N = seg // P
            kdir = Path(config.aie_kernels_dir())
            here = Path(__file__).resolve().parent / "kernels"
            seg_t = np.ndarray[(seg,), np.dtype[bfloat16]]
            row_t = np.ndarray[(N,), np.dtype[bfloat16]]
            ts_c = np.ndarray[(P * (2 * K - 1) * N,), np.dtype[bfloat16]]
            ts_t = np.ndarray[((2 * K - 1) * N,), np.dtype[bfloat16]]
            fin_c = np.ndarray[(P * (K - 1) * N,), np.dtype[bfloat16]]
            fin_t = np.ndarray[((K - 1) * N,), np.dtype[bfloat16]]
            h_t = np.ndarray[(K * N,), np.dtype[bfloat16]]
            common = dict(source_file=str(here / "short_conv.cc"), include_dirs=[str(kdir), config.cxx_header_path()],
                          compile_flags=[f"-DSC_W={N}", f"-DSC_K={K}", "-DSC_GDN", "-ffp-contract=off",
                                         "-fno-fast-math"],
                          object_file_name=f"gdn_conv_{N}_{K}.o")
            kb = ExternalFunction("gdn_conv_begin", arg_types=[ts_t, h_t], **common)
            kr = ExternalFunction("gdn_conv_row", arg_types=[row_t, ts_t, h_t, row_t, np.int32, np.int32, np.int32],
                                  **common)
            ke = ExternalFunction("gdn_conv_end", arg_types=[h_t, fin_t, np.int32], **common)
            x_l3 = [ObjectFifo(seg_t, depth=2) for _ in range(n)]
            x_l1 = [f.cons().split([j * N for j in range(P)], obj_types=[row_t] * P, depths=[2] * P,
                                   tile=AnyMemTile) for f in x_l3]
            t_l3 = [ObjectFifo(ts_c, depth=1) for _ in range(n)]
            t_l1 = [f.cons().split([j * (2 * K - 1) * N for j in range(P)], obj_types=[ts_t] * P,
                                   depths=[1] * P, tile=AnyMemTile) for f in t_l3]
            y_l3 = [ObjectFifo(seg_t, depth=2) for _ in range(n)]
            y_l1 = [f.prod().join([j * N for j in range(P)], obj_types=[row_t] * P, depths=[2] * P,
                                  tile=AnyMemTile) for f in y_l3]
            f_l3 = [ObjectFifo(fin_c, depth=1) for _ in range(n)]
            f_l1 = [f.prod().join([j * (K - 1) * N for j in range(P)], obj_types=[fin_t] * P, depths=[1] * P,
                                  tile=AnyMemTile) for f in f_l3]
            rtps = [Rtp([self.count]) for _ in range(n * P)]
            R, raw = self.rows, self.raw

            def core(xf, tf, yf, ff, hist, kb, kr, ke, rtp, own_raw):
                rows = rtp.read()[self.count]
                ts = tf.acquire(1)
                kb(ts, hist)
                for t in range_(R):
                    t32 = index.casts(T.i32(), t)
                    xo, yo = xf.acquire(1), yf.acquire(1)
                    kr(xo, ts, hist, yo, t32, rows, own_raw)
                    xf.release(1)
                    yf.release(1)
                fo = ff.acquire(1)
                ke(hist, fo, rows)
                ff.release(1)
                tf.release(1)

            workers = []
            for c in range(n):
                for j in range(P):
                    workers.append(Worker(
                        lambda xf, tf, yf, ff, hist, kb, kr, ke, rtp, last=(j == P - 1):
                            core(xf, tf, yf, ff, hist, kb, kr, ke, rtp, raw if last else 0),
                        [x_l1[c][j].cons(), t_l1[c][j].cons(), y_l1[c][j].prod(), f_l1[c][j].prod(),
                         Buffer(h_t, name=f"gdn_conv_hist_{c}_{j}"), kb, kr, ke, rtps[c * P + j]]))
            self.x.bind([f.prod() for f in x_l3])
            self.taps.bind([f.prod() for f in t_l3])
            self.state.bind([f.prod() for f in t_l3])
            self.y.bind([f.cons() for f in y_l3])
            self.final.bind([f.cons() for f in f_l3])
            return Built(workers, rtps)

        def sequence(self, rt):
            R, E, seg, K = self.rows, self.width, self.seg, self.taps_n
            N = seg // P
            tg = TaskGroup()
            for c in range(n):
                for j in range(P):  # each core's taps, then its state: the memtile cuts them
                    self.taps.fill(TAP((K, E), c * seg + j * N, [K, N], [E, 1]), lane=c, group=tg)
                    self.state.fill(TAP((K - 1, E), c * seg + j * N, [K - 1, N], [E, 1]), lane=c, group=tg)
                self.x.fill(TAP((R, E), c * seg, [R, seg], [E, 1]), lane=c, group=tg)
            for c in range(n):
                self.y.drain(TAP((R, E), c * seg, [R, seg], [E, 1]), lane=c, group=tg, wait=True)
                self.final.drain(TAP((K - 1, E), c * seg, [P, K - 1, N], [N, E, 1]), lane=c, group=tg,
                                 wait=True)
            tg.finish()

        def ops(self) -> int:
            return (2 * self.taps_n + 4) * self.width * self.rows

        def reference(self, x, taps, state):
            return short_conv_chunk_silu_reference(x, taps, state, int(self.valid), self.seg, self.raw)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.01, 1e-3, note="float32 sums and silu, rounded once to bf16")

    return GdnConvChunk


def _check_chunk(args) -> int:
    """Export a graph of one ShortConvChunk of the call's ``n`` rows and its check at ``--n``."""
    import iron
    from iron import Scratchpad
    from iron.operators.copy import Copy
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    E, K, R, raw = args.width, args.taps, args.chunk, args.silu
    Op = short_conv_chunk_operator(tile=args.tile, raw=raw)
    extra = {}
    if args.gdn:
        Op, extra = gdn_conv_chunk_operator(), dict(raw=raw)
    parts = 1 if raw is not None else 3
    rng = np.random.default_rng(5)
    taps_v = bf16(rng.standard_normal((K, E)) * 0.3)
    taps = iron.param(taps_v)

    def ref(p, state, n):
        if raw is not None:
            return short_conv_chunk_silu_reference(p, taps_v, state, n, E // COLUMNS, raw)
        return short_conv_chunk_reference(p, taps_v, state, n)

    if args.baseline:  # IRON's convolution, then GdnSiLU (what Qwen3.5's prompt chunk runs)
        import iron.functional as F
        import iron_gdn

        window, tail, write = iron.state((2, R, E)), iron.state((K - 1, E)), iron.state((K - 1, E))

        class Graph(iron.Graph):
            def body(self, p, state, *, n: Scratchpad[np.int32]):
                y = F.causal_conv1d(p[:n], taps, F.ConvState(state, write, window, tail))
                u = iron_gdn.silu_operator(COLUMNS).step(y, seg=E // COLUMNS, raw=raw)
                return u, Copy.step(write)

            def reference(self, p, state, *, n):
                return ref(p, state, n)

        g = Graph()
        cg = g.compile(p=(R, E), state=(K - 1, E))
        m = export_graph(cg, args.out, "shortconvchunk", dict(width=E, taps=K, chunk=R))
        print(cg.plan.report("shortconvchunk"))
        p = bf16(rng.standard_normal((R, E)))
        state = bf16(rng.standard_normal((K - 1, E)))
        write_check(g, m, args.out, dict(p=p, state=state), dict(n=args.n))
        return 0

    class Graph(iron.Graph):
        def body(self, p, state, *, n: Scratchpad[np.int32]):
            y, final = Op.step(p[:n], taps, state, rows=R, width=E, taps_n=K, **extra)
            return y, final

        def reference(self, p, state, *, n):
            return ref(p, state, n)

    g = Graph()
    cg = g.compile(p=(R, parts * E), state=(K - 1, E))
    m = export_graph(cg, args.out, "shortconvchunk", dict(width=E, taps=K, chunk=R))
    print(cg.plan.report("shortconvchunk"))
    p = bf16(rng.standard_normal((R, parts * E)))
    state = bf16(rng.standard_normal((K - 1, E)))
    write_check(g, m, args.out, dict(p=p, state=state), dict(n=args.n))
    print(f"{args.out}: ShortConvChunk of {E} channels, {K} taps, {args.n} of {R} rows")
    return 0


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
    c = sub.add_parser("chunk", help="export a graph of one ShortConvChunk (a prompt chunk's) and its check")
    c.add_argument("--width", type=int, default=2048)
    c.add_argument("--taps", type=int, default=7)
    c.add_argument("--chunk", type=int, default=1024)
    c.add_argument("out", type=Path)
    c.add_argument("--n", type=int, default=13, help="the call's rows")
    c.add_argument("--tile", type=int, default=8, help="rows a core takes at a time")
    c.add_argument("--silu", type=int, default=None, metavar="RAW",
                   help="Gated DeltaNet's form: x alone, silu but on each column's last RAW channels")
    c.add_argument("--baseline", action="store_true", help="with --silu, IRON's convolution and GdnSiLU instead")
    c.add_argument("--gdn", action="store_true", help="with --silu, GdnConvChunk (a column's channels over 2 cores)")
    a = ap.parse_args(argv)
    return _check_chunk(a) if a.cmd == "chunk" else _check(a)


if __name__ == "__main__":
    sys.exit(main())
