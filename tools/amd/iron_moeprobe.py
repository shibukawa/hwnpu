#!/usr/bin/env python3
"""A probe of a mixture-of-experts decode step on AMD's NPU: Qwen3.6-35B-A3B's experts.

Not a model: the parts of an MoE layer IRON can express today, at Qwen3.6-35B-A3B's sizes
(hidden 2048, 256 experts of 512, 8 routed and one shared), to measure what a port would
cost before writing one (docs/design.md 16.75). Each of ``layers`` layers is one command:

  1. Copy the 8 routed experts' gate, up and down rows out of the layer's tables into
     staging states (per-call scratchpad values pick the experts: ``e<l>_<i>``)
  2. gate and up as one GEMV each over the 8 staged experts (4096 x 2048), SiLU times up,
     times the routing weights (input ``w``, each expert's weight over its 512 entries), down
     as one GEMV over the 8 staged experts side by side (2048 x 4096)
  3. the shared expert (three 512-wide GEMVs) and the next layer's router (256 x 2048)

and an empty host step ends it, as the host would pick the next layer's experts there. With
``--split`` the copies, the experts' GEMVs and the shared expert and router are commands of
their own, to time each. Graphs with ``--tiny N`` are N commands of one small add, to time a
command's own cost. Graphs with ``--q4 ROWS --reps N`` are N q4nx GEMVs of (ROWS, 2048) (IRON's
flm.LMHead), to time a 4-bit GEMV. With ``--q4moe q4`` the experts are q4nx tables read in place:
ExpertQGEMV (gate, up: a column an expert) and ExpertDownQGEMV (down: the 8 side by side) take
each expert's byte offset per call, so nothing is copied (docs/design.md 16.75).

  python tools/amd/iron_moeprobe.py OUT --layers 12 --channels 8 [--split] [--check]
  python tools/amd/iron_moeprobe.py OUT --layers 12 --q4moe q4 --check
  go run ./tools/amd/irongraph [-pin GiB] OUT
"""

import argparse
import sys
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

import iron
import iron.functional as F
from iron import Precision, Scratchpad
from iron.operators.copy import Copy
from iron.operators.gemm import GEMM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

HIDDEN, EXPERTS, INTER, TOP = 2048, 256, 512, 8
P = Precision.HIGHEST  # one row runs on GEMV, which ignores it
DOWN_T = False  # --down-t


def rand_bf16(rng, shape, scale) -> np.ndarray:
    """Random bf16 of about ``scale`` magnitude, made from bits (fast for gigabytes)."""
    x = rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
    return x.astype(bfloat16)


def boundary() -> None:
    iron.on_host(lambda: None)


class Layer:
    def __init__(self, rng, l: int):
        s = 1 / np.sqrt(HIDDEN)
        t0 = time.time()
        self.gate = iron.param(rand_bf16(rng, (EXPERTS, INTER, HIDDEN), s))
        self.up = iron.param(rand_bf16(rng, (EXPERTS, INTER, HIDDEN), s))
        # Each expert's down projection (2048, 512); the 8 staged side by side are the down
        # GEMV's (2048, 8 * 512) weight.
        self.down = iron.param(rand_bf16(rng, (EXPERTS, HIDDEN, INTER), 1 / np.sqrt(INTER)))
        # The same transposed, (512, 2048) an expert: the 8 staged are one contiguous copy each
        # into the (8 * 512, 2048) weight of a one-row GEMM (--down-t).
        self.down_t = np.ascontiguousarray(self.down.transpose(0, 2, 1)) if DOWN_T else None
        if DOWN_T:
            self.down = None
        self.sh_gate = iron.param(rand_bf16(rng, (INTER, HIDDEN), s))
        self.sh_up = iron.param(rand_bf16(rng, (INTER, HIDDEN), s))
        self.sh_down = iron.param(rand_bf16(rng, (HIDDEN, INTER), 1 / np.sqrt(INTER)))
        self.router = iron.param(rand_bf16(rng, (EXPERTS, HIDDEN), s))
        self.norm = iron.param(np.ones(HIDDEN, bfloat16))
        print(f"layer {l}: weights in {time.time() - t0:.0f} s", flush=True)


class MoEProbe:
    """The probe's layers; probe_class makes the iron.Graph with a value per expert picked."""

    def __init__(self, layers: int, split: bool, channels: int = 0, seed: int = 0):
        self.channels = channels
        rng = np.random.default_rng(seed)
        self.layers = [Layer(rng, l) for l in range(layers)]
        self.split = split
        self.stage_g = iron.state((TOP, INTER, HIDDEN))
        self.stage_u = iron.state((TOP, INTER, HIDDEN))
        self.stage_d = iron.state((HIDDEN, TOP * INTER))
        self.stage_dt = iron.state((TOP, INTER, HIDDEN)) if DOWN_T else None
        self.rows32 = iron.state((32, TOP * INTER)) if DOWN_T else None

    def _body(self, x, w, **e):
        h = x
        logits = None
        for l, L in enumerate(self.layers):
            if l:
                boundary()
            kw = dict(num_channels=self.channels) if self.channels else {}
            for i in range(TOP):
                r = e[f"e{l}_{i}"]
                Copy.step(iron.weight(L.gate)[r], self.stage_g[i], **kw)
                Copy.step(iron.weight(L.up)[r], self.stage_u[i], **kw)
            if self.split:
                boundary()
            for i in range(TOP):
                r = e[f"e{l}_{i}"]
                if DOWN_T:
                    Copy.step(iron.weight(L.down_t)[r], self.stage_dt[i], **kw)
                else:
                    Copy.step(iron.weight(L.down)[r], self.stage_d[:, i * INTER:(i + 1) * INTER], **kw)
            if self.split:
                boundary()
            g = F.linear(h, self.stage_g.reshape(TOP * INTER, HIDDEN), precision=P)
            u = F.linear(h, self.stage_u.reshape(TOP * INTER, HIDDEN), precision=P)
            a = F.silu(g) * u * w
            if DOWN_T:
                # GEMM tiles M by 32: a's row and 31 zero rows.
                self.rows32[0] = a
                y = GEMM.step(self.rows32, self.stage_dt.reshape(TOP * INTER, HIDDEN),
                              precision=P)[0]
            else:
                y = F.linear(a, self.stage_d, precision=P)
            if self.split:
                boundary()
            s = F.linear(F.silu(F.linear(h, L.sh_gate, precision=P)) * F.linear(h, L.sh_up, precision=P),
                         L.sh_down, precision=P)
            # Normed, as a layer's next input is (and random weights do not blow up over layers).
            h = F.rms_norm((h + y + s).reshape(1, HIDDEN), (HIDDEN,), L.norm, 1e-6).reshape(HIDDEN)
            logits = F.linear(h, L.router, precision=P)
        return h, logits


def probe_class(layers: int):
    """An iron.Graph of MoEProbe whose body takes a scratchpad value per layer and expert
    (iron reads the body's signature, which takes no **kwargs)."""
    names = [f"e{l}_{i}" for l in range(layers) for i in range(TOP)]
    params = ", ".join(f"{n}: Scratchpad[np.int32]" for n in names)
    src = f"def body(self, x, w, *, {params}):\n    return self._body(x, w, " + \
        ", ".join(f"{n}={n}" for n in names) + ")\n"
    ns = dict(Scratchpad=Scratchpad, np=np, MoEProbe=MoEProbe)
    exec(src, ns)
    return type("MoEProbeGraph", (MoEProbe, iron.Graph), dict(body=ns["body"]))


# --- 4-bit experts read in place (--q4moe) ----------------------------------------------------


def rand_q4(rng, n: int, k: int, count: int, std: float, fmt: str = "q4") -> np.ndarray:
    """``count`` random (n, k) weights in fmt (iron_q4's blocks), each about ``std``: uint8
    (count * blocks, block bytes). Codes are random bytes; scales and minima make each
    group's weights uniform around zero (quantizing gigabytes of draws would take an hour)."""
    from iron_q4 import FORMATS, GROUP, K_TILE, M_TILE

    B = FORMATS[fmt]
    blocks = count * n * k // (M_TILE * K_TILE)
    q = rng.integers(0, 256, (blocks, B), dtype=np.uint8)
    groups = M_TILE * K_TILE // GROUP  # scales (and minima) a block
    if fmt == "q4":
        sc = (std / 4.6 * rng.uniform(0.8, 1.2, (blocks, groups))).astype(bfloat16)
        q[:, : 2 * groups] = sc.view(np.uint8).reshape(blocks, -1)
        q[:, 2 * groups : 4 * groups] = (-7.5 * sc.astype(np.float32)).astype(bfloat16).view(np.uint8).reshape(blocks, -1)
    else:
        sc = (std / 73 * rng.uniform(0.8, 1.2, (blocks, groups))).astype(bfloat16)
        q[:, : 2 * groups] = sc.view(np.uint8).reshape(blocks, -1)
    return q


def expert_operators():
    """ExpertQGEMV and ExpertDownQGEMV: iron_q4's QGEMV on the rows of an expert table that
    per-call offsets pick (``e0`` .. ``e7``, in bytes), so the 8 experts are read where they
    are, without a copy."""
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import Buffer, In, ObjectFifo, Out, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.utils.compile.jit.params import param, per_call, tunable, widths
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator
    from iron_q4 import CB, FORMATS, K_TILE, _kernels, dequantize_fmt

    def offset(op, lane):
        v = op.value(f"e{lane}")
        return v if v.per_call else None

    def lanes(cores, w_l3, x_l3, y_l3, B, x_t, out):
        blk_t = np.ndarray[(B,), np.dtype[np.uint8]]
        y_t = np.ndarray[(out,), np.dtype[bfloat16]]
        if cores == 1:
            return [[f] for f in w_l3], x_l3, [[f] for f in y_l3]
        w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
        x_l1 = [f.cons().forward(obj_type=x_t, tile=AnyMemTile) for f in x_l3]
        y_l1 = [f.prod().join([0, out], obj_types=[y_t, y_t], depths=[2, 2], tile=AnyMemTile) for f in y_l3]
        return w_l1, x_l1, y_l1

    class ExpertQGEMV(Operator):
        """``y[l NE : (l + 1) NE] = W[e_l] x``: a column an expert, W[e] the e-th (NE, K) weight of
        the table, its blocks at byte ``e_l``. Gate and up of a mixture of experts. With ``split_x``
        column l reads x[l K : (l + 1) K] instead (a down with a column an expert, whose 8 outputs
        are then summed)."""

        fmt: str = param(array=True)
        rows: int = param(array=True)  # blocks in the table
        K: int = param(array=True)
        NE: int = param(array=True)
        YN: int = param(derive=lambda op: TOP * op.NE, array=True)
        split_x: bool = param(default=False, array=True)
        XK: int = param(derive=lambda op: op.K * (TOP if op.split_x else 1), array=True)
        cores: int = param(default=2, array=True)
        block_bytes: int = param(derive=lambda op: FORMATS[op.fmt], array=True)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, block_bytes, dtype=np.uint8, obj_shape=(cores, block_bytes), per=(num_aie_columns,))
        x = In(XK, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
        y = Out(YN, obj_shape=(CB,), per=(num_aie_columns,))
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)
        e4 = per_call(derive=lambda op: 0)
        e5 = per_call(derive=lambda op: 0)
        e6 = per_call(derive=lambda op: 0)
        e7 = per_call(derive=lambda op: 0)

        def resolve(self, dev):
            return self.replace(num_aie_columns=TOP)

        def build(self) -> Built:
            K, cores, B = self.K, self.cores, self.block_bytes
            cbs, kbs, out = self.NE // CB, K // K_TILE, CB // cores
            begin, block, end = _kernels(self.fmt, K, out)
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(TOP)]
            x_l3 = [ObjectFifo(self.x.obj_type, depth=1) for _ in range(TOP)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(TOP)]
            w_l1, x_l1, y_l1 = lanes(cores, w_l3, x_l3, y_l3, B, self.x.obj_type, out)
            sums = [[Buffer(np.ndarray[(K // 32,), np.dtype[bfloat16]]) for _ in range(cores)] for _ in range(TOP)]
            accs = [[Buffer(np.ndarray[(out,), np.dtype[np.float32]]) for _ in range(cores)] for _ in range(TOP)]

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

            workers = [Worker(core, [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), sums[i][j],
                                     accs[i][j], begin, block, end])
                       for i in range(TOP) for j in range(cores)]
            self.W.bind([f.prod() for f in w_l3])
            self.x.bind([f.prod() for f in x_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            B, NE, K = self.block_bytes, self.NE, self.K
            per = NE // CB * (K // K_TILE) * 2
            drains = [self.y.drain(TAP((self.YN,), l * NE, [NE], [1]), lane=l, wait=True, managed=False)
                      for l in range(TOP)]
            for l in range(TOP):
                self.x.fill(TAP((self.XK,), l * K if self.split_x else 0, [K], [1]), lane=l, managed=False)
            for l in range(TOP):
                self.W.fill(TAP((self.rows, B), 0, [per, B], [B, 1]), lane=l,
                            offset_parameter=offset(self, l), managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * self.YN * self.K

        def reference(self, W, x, *, e0=0, e1=0, e2=0, e3=0, e4=0, e5=0, e6=0, e7=0):
            e = dict(e0=e0, e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7)
            per = self.NE // CB * (self.K // K_TILE) * 2 * self.block_bytes
            flat = np.asarray(W, np.uint8).reshape(-1)
            ys = []
            for l in range(TOP):
                at = int(e.get(f"e{l}", 0))
                w = dequantize_fmt(flat[at : at + per], self.NE, self.K, self.fmt)
                xl = np.asarray(x, np.float64)
                if self.split_x:
                    xl = xl[l * self.K : (l + 1) * self.K]
                ys.append(w.astype(np.float64) @ xl)
            return np.concatenate(ys).astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="32-column partial sums narrowed to bf16")

    class ExpertDownQGEMV(Operator):
        """``y = sum_l W[e_l] x[l KE : (l + 1) KE]``: the 8 experts' (N, KE) weights side by side,
        one GEMV of (N, 8 KE) whose column blocks each read their part of every expert. Down of a
        mixture of experts (x is the 8 experts' activations, each times its routing weight)."""

        fmt: str = param(array=True)
        rows: int = param(array=True)
        KE: int = param(array=True)
        K: int = param(derive=lambda op: TOP * op.KE, array=True)
        N: int = param(array=True)
        cores: int = param(default=2, array=True)
        block_bytes: int = param(derive=lambda op: FORMATS[op.fmt], array=True)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, block_bytes, dtype=np.uint8, obj_shape=(cores, block_bytes), per=(num_aie_columns,))
        x = In(K, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
        y = Out(N, obj_shape=(CB,), per=(num_aie_columns,))
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)
        e4 = per_call(derive=lambda op: 0)
        e5 = per_call(derive=lambda op: 0)
        e6 = per_call(derive=lambda op: 0)
        e7 = per_call(derive=lambda op: 0)

        def resolve(self, dev):
            cbs = self.N // CB
            cols = self.resolve_columns(dev, self.num_aie_columns, fits=lambda c: cbs % c == 0)
            return self.replace(num_aie_columns=cols)

        def build(self) -> Built:
            K, n, cores, B = self.K, self.num_aie_columns, self.cores, self.block_bytes
            cbs, pairs, out = self.N // CB // n, self.KE // K_TILE, CB // cores
            begin, block, end = _kernels(self.fmt, K, out)
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            x_l3 = [ObjectFifo(self.x.obj_type, depth=1) for _ in range(n)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            w_l1, x_l1, y_l1 = lanes(cores, w_l3, x_l3, y_l3, B, self.x.obj_type, out)
            sums = [[Buffer(np.ndarray[(K // 32,), np.dtype[bfloat16]]) for _ in range(cores)] for _ in range(n)]
            accs = [[[Buffer(np.ndarray[(out,), np.dtype[np.float32]]) for _ in range(cbs)] for _ in range(cores)]
                    for _ in range(n)]

            def core(w_f, x_f, y_f, sums, accs, begin, block, end):
                x = x_f.acquire(1)
                for c in range(cbs):
                    begin(x, sums, accs[c])
                # an expert's pairs of each column block in turn, as the fills give them
                for kb in range_(TOP * pairs):
                    kb32 = index.casts(T.i32(), kb)
                    for c in range(cbs):
                        for h in range(2 if cores == 1 else 1):
                            w = w_f.acquire(1)
                            block(w, x, sums, accs[c], kb32, h)
                            w_f.release(1)
                for c in range(cbs):
                    y = y_f.acquire(1)
                    end(accs[c], y)
                    y_f.release(1)
                x_f.release(1)

            workers = [Worker(core, [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), sums[i][j],
                                     accs[i][j], begin, block, end])
                       for i in range(n) for j in range(cores)]
            self.W.bind([f.prod() for f in w_l3])
            self.x.bind([f.prod() for f in x_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            n, B, N, K = self.num_aie_columns, self.block_bytes, self.N, self.K
            cbs, pairs = N // CB // n, self.KE // K_TILE
            total = self.rows * B
            drains = [self.y.drain(TAP((N,), l * N // n, [N // n], [1]), lane=l, wait=True, managed=False)
                      for l in range(n)]
            for l in range(n):
                self.x.fill(TAP((K,), 0, [K], [1]), lane=l, managed=False)
            for l in range(n):
                # an expert's column block is `pairs` pairs of blocks; the lane's, pair by pair
                tap = TAP((total,), l * cbs * 2 * pairs * B, [pairs, cbs, 2 * B], [2 * B, 2 * pairs * B, 1])
                for e in range(TOP):
                    self.W.fill(tap, lane=l, offset_parameter=offset(self, e), managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * self.N * self.K

        def reference(self, W, x, *, e0=0, e1=0, e2=0, e3=0, e4=0, e5=0, e6=0, e7=0):
            e = dict(e0=e0, e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7)
            per = self.N // CB * (self.KE // K_TILE) * 2 * self.block_bytes
            flat = np.asarray(W, np.uint8).reshape(-1)
            x = np.asarray(x, np.float64)
            y = np.zeros(self.N)
            for l in range(TOP):
                at = int(e.get(f"e{l}", 0))
                w = dequantize_fmt(flat[at : at + per], self.N, self.KE, self.fmt)
                y += w.astype(np.float64) @ x[l * self.KE : (l + 1) * self.KE]
            return y.astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="32-column partial sums narrowed to bf16")

    return ExpertQGEMV, ExpertDownQGEMV


# --- the fused layer (--q4moe q4 --fused) ----------------------------------------------------

SHARED = EXPERTS  # the shared expert's index in the tables (one more than the routed)
WV = 16  # routing weights an input holds: the 8 picked, then the shared expert's


def gu_table(gate, up) -> np.ndarray:
    """One expert's gate (NE, K) and up (NE, K) as the (2 NE, K) weight MoEGateUp reads: their
    column blocks of 64 out-features alternating, gate first."""
    from iron_q4 import CB

    ne, k = gate.shape
    return np.stack([gate.reshape(ne // CB, CB, k), up.reshape(ne // CB, CB, k)], 1).reshape(2 * ne, k)


def silu(v):
    return v / (1 + np.exp(-v))


def fused_operators():
    """MoEGateUp and MoEDown: a layer's experts, the shared one with them, in two operators
    whose cores also do the element-wise work (kernels/moe_mv.cc)."""
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import Buffer, In, ObjectFifo, Out, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param, per_call, tunable, widths
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator
    from iron_q4 import BLOCK_BYTES as B, CB, K_TILE, dequantize

    MT = CB // 2  # out-features a core
    blk_t = np.ndarray[(B,), np.dtype[np.uint8]]
    half_t = np.ndarray[(MT,), np.dtype[bfloat16]]
    acc_t = np.ndarray[(MT,), np.dtype[np.float32]]

    def kernels(k: int, extra: int, pairs: int = 2):
        """moe_mv.cc's calls for x of k in-features and ``extra`` values after them."""
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        x = np.ndarray[(k + extra,), np.dtype[bfloat16]]
        sums = np.ndarray[(k // 32,), np.dtype[bfloat16]]
        common = dict(source_file=str(here / "moe_mv.cc"),
                      include_dirs=[str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
                      compile_flags=[f"-DMOE_K={k}", f"-DMOE_PAIRS={pairs}"],
                      object_file_name=f"moe_mv_{k}_{extra}_{pairs}.o")
        f = lambda name, args: ExternalFunction(name, arg_types=args, **common)  # noqa: E731
        return dict(
            sums=f("moe_sums", [x, sums]),
            zero=f("moe_zero", [acc_t]),
            block=f("moe_block", [blk_t, x, sums, acc_t, np.int32]),
            block_e=f("moe_block_e", [blk_t, x, sums, acc_t, np.int32, np.int32]),
            glu=f("moe_glu", [acc_t, acc_t, x, np.int32, half_t]),
            end_add=f("moe_end_add", [acc_t, x, np.int32, half_t]),
        )

    def offset(op, e):
        v = op.value(f"e{e}")
        return v if v.per_call else None

    class MoEGateUp(Operator):
        """``y = silu(G_e x) * (U_e x) * w[e]`` for the 8 experts a call picks and the shared one:
        y is the 8 experts' NE out-features in turn, then the shared expert's. The table holds
        EXPERTS + 1 experts' gu_table weights (the shared one last); column l reads expert e_l's
        (at byte e_l) and the shared expert's l-th 64 out-features."""

        rows: int = param(array=True)  # blocks in the table
        K: int = param(array=True)
        NE: int = param(array=True)
        YN: int = param(derive=lambda op: (TOP + 1) * op.NE, array=True)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, B, dtype=np.uint8, obj_shape=(2, B), per=(num_aie_columns,))
        x = In(K, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
        w = In(WV, obj_shape=(WV,), per=(num_aie_columns,), replicate=True)
        y = Out(YN, obj_shape=(CB,), per=(num_aie_columns,))
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)
        e4 = per_call(derive=lambda op: 0)
        e5 = per_call(derive=lambda op: 0)
        e6 = per_call(derive=lambda op: 0)
        e7 = per_call(derive=lambda op: 0)

        def validate(self):
            if self.NE != TOP * CB:
                raise ValueError(f"MoEGateUp: the shared expert's {self.NE} out-features are 64 a column")

        def resolve(self, dev):
            return self.replace(num_aie_columns=TOP)

        def expert_bytes(self) -> int:
            return 2 * self.NE // CB * (self.K // K_TILE) * 2 * B

        def build(self) -> Built:
            K = self.K
            kbs, cbs = K // K_TILE, self.NE // CB
            k = kernels(K, WV)
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(TOP)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(TOP)]
            w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
            # x, then the routing weights, down one channel into one object for both cores
            xw_t = np.ndarray[(K + WV,), np.dtype[bfloat16]]
            xw_l3 = [ObjectFifo(xw_t, depth=1) for _ in range(TOP)]
            xw_l1 = [f.cons().forward(obj_type=xw_t, tile=AnyMemTile) for f in xw_l3]
            y_l1 = [f.prod().join([0, MT], obj_types=[half_t, half_t], depths=[2, 2], tile=AnyMemTile)
                    for f in y_l3]

            def core(w_f, x_f, y_f, sums, g, u, lane, sums_k, zero, block, glu):
                x = x_f.acquire(1)
                sums_k(x, sums)
                zero(g)
                zero(u)

                def expert(idx, n):
                    for _ in range_(n):
                        for acc in (g, u):
                            for kb in range_(kbs):
                                kb32 = index.casts(T.i32(), kb)
                                blk = w_f.acquire(1)
                                block(blk, x, sums, acc, kb32)
                                w_f.release(1)
                        y = y_f.acquire(1)
                        glu(g, u, x, idx, y)
                        y_f.release(1)

                expert(lane, cbs)
                expert(TOP, 1)
                x_f.release(1)

            workers = []
            for i in range(TOP):
                for j in range(2):
                    sums = Buffer(np.ndarray[(K // 32,), np.dtype[bfloat16]])
                    g, u = Buffer(acc_t), Buffer(acc_t)
                    workers.append(Worker(
                        lambda w_f, x_f, y_f, sums, g, u, s, z, b, gl, lane=i: core(
                            w_f, x_f, y_f, sums, g, u, lane, s, z, b, gl),
                        [w_l1[i][j].cons(), xw_l1[i].cons(), y_l1[i][j].prod(), sums, g, u,
                         k["sums"], k["zero"], k["block"], k["glu"]]))
            self.W.bind([f.prod() for f in w_l3])
            self.x.bind([f.prod() for f in xw_l3])
            self.w.bind([f.prod() for f in xw_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            NE, K, YN = self.NE, self.K, self.YN
            eb = self.expert_bytes()
            cb_bytes = 2 * (K // K_TILE) * 2 * B  # a gate and an up column block
            total = self.rows * B
            drains = []
            for l in range(TOP):
                drains.append(self.y.drain(TAP((YN,), l * NE, [NE], [1]), lane=l, wait=True, managed=False))
                drains.append(self.y.drain(TAP((YN,), TOP * NE + l * CB, [CB], [1]), lane=l, wait=True,
                                           managed=False))
            for l in range(TOP):
                self.x.fill(TAP((K,), 0, [K], [1]), lane=l, managed=False)
                self.w.fill(TAP((WV,), 0, [WV], [1]), lane=l, managed=False)
            for l in range(TOP):
                self.W.fill(TAP((total,), 0, [eb], [1]), lane=l, offset_parameter=offset(self, l), managed=False)
                self.W.fill(TAP((total,), SHARED * eb + l * cb_bytes, [cb_bytes], [1]), lane=l, managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * 2 * self.YN * self.K

        def reference(self, W, x, w, *, e0=0, e1=0, e2=0, e3=0, e4=0, e5=0, e6=0, e7=0):
            eb = self.expert_bytes()
            flat = np.asarray(W, np.uint8).reshape(-1)
            x = np.asarray(x, np.float64)
            w = np.asarray(w, np.float64)
            ys = []
            for idx, at in enumerate([e0, e1, e2, e3, e4, e5, e6, e7, SHARED * eb]):
                gu = dequantize(flat[int(at) : int(at) + eb], 2 * self.NE, self.K).astype(np.float64)
                gu = gu.reshape(self.NE // CB, 2, CB, self.K)
                g = gu[:, 0].reshape(self.NE, self.K) @ x
                u = gu[:, 1].reshape(self.NE, self.K) @ x
                ys.append(silu(g) * u * w[idx])
            return np.concatenate(ys).astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="q4nx partial sums in bf16, SiLU's tanh")

    class MoEDown(Operator):
        """``y = r + sum_e D_e x_e`` over the 8 experts a call picks and the shared one, x their
        activations side by side ((TOP + 1) KE). Column l keeps N / columns out-features and reads
        each expert's part of them in one contiguous run (at byte e_l for the routed)."""

        # 9 weight runs, x and the residual a column: let aiecc recycle finished BDs (as Copy)
        aiecc_flags: ClassVar[tuple[str, ...]] = ("--reclaim-runtime-bds",)

        rows: int = param(array=True)
        KE: int = param(array=True)
        K: int = param(derive=lambda op: (TOP + 1) * op.KE, array=True)
        N: int = param(array=True)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, B, dtype=np.uint8, obj_shape=(2, B), per=(num_aie_columns,))
        x = In(K, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
        R: int = param(derive=lambda op: op.N // TOP, array=True)  # the residual a column (8 columns)
        r = In(N, obj_shape=(R,), per=(num_aie_columns,))
        y = Out(N, obj_shape=(CB,), per=(num_aie_columns,))
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)
        e4 = per_call(derive=lambda op: 0)
        e5 = per_call(derive=lambda op: 0)
        e6 = per_call(derive=lambda op: 0)
        e7 = per_call(derive=lambda op: 0)

        def resolve(self, dev):
            return self.replace(num_aie_columns=TOP)

        def expert_bytes(self) -> int:
            return self.N // CB * (self.KE // K_TILE) * 2 * B

        def build(self) -> Built:
            K, n = self.K, self.num_aie_columns
            cbs, pairs = self.N // CB // n, self.KE // K_TILE
            R = self.N // n
            k = kernels(K, R, pairs)
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
            # x, then the column's residual, down one channel into one object for both cores
            xr_t = np.ndarray[(K + R,), np.dtype[bfloat16]]
            xr_l3 = [ObjectFifo(xr_t, depth=1) for _ in range(n)]
            xr_l1 = [f.cons().forward(obj_type=xr_t, tile=AnyMemTile) for f in xr_l3]
            y_l1 = [f.prod().join([0, MT], obj_types=[half_t, half_t], depths=[2, 2], tile=AnyMemTile)
                    for f in y_l3]

            def core(w_f, x_f, y_f, sums, accs, half, sums_k, zero, block_e, end_add):
                x = x_f.acquire(1)
                sums_k(x, sums)
                for acc in accs:
                    zero(acc)
                # an expert's run: each column block's k-blocks in turn
                for e in range_(TOP + 1):
                    e32 = index.casts(T.i32(), e)
                    for acc in accs:
                        for kh in range(pairs):
                            blk = w_f.acquire(1)
                            block_e(blk, x, sums, acc, e32, kh)
                            w_f.release(1)
                for c, acc in enumerate(accs):
                    y = y_f.acquire(1)
                    end_add(acc, x, c * CB + half * MT, y)
                    y_f.release(1)
                x_f.release(1)

            workers = []
            for i in range(n):
                for j in range(2):
                    sums = Buffer(np.ndarray[(K // 32,), np.dtype[bfloat16]])
                    accs = [Buffer(acc_t) for _ in range(cbs)]
                    workers.append(Worker(
                        lambda w_f, x_f, y_f, sums, accs, s, z, b, ea, half=j: core(
                            w_f, x_f, y_f, sums, accs, half, s, z, b, ea),
                        [w_l1[i][j].cons(), xr_l1[i].cons(), y_l1[i][j].prod(), sums, accs, k["sums"],
                         k["zero"], k["block_e"], k["end_add"]]))
            self.W.bind([f.prod() for f in w_l3])
            self.x.bind([f.prod() for f in xr_l3])
            self.r.bind([f.prod() for f in xr_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            n, N, K = self.num_aie_columns, self.N, self.K
            eb = self.expert_bytes()
            run = eb // n  # a column's part of an expert: its column blocks, contiguous
            total = self.rows * B
            drains = [self.y.drain(TAP((N,), l * N // n, [N // n], [1]), lane=l, wait=True, managed=False)
                      for l in range(n)]
            for l in range(n):
                self.x.fill(TAP((K,), 0, [K], [1]), lane=l, managed=False)
                self.r.fill(TAP((N,), l * N // n, [N // n], [1]), lane=l, managed=False)
            for l in range(n):
                for e in range(TOP):
                    self.W.fill(TAP((total,), l * run, [run], [1]), lane=l, offset_parameter=offset(self, e),
                                managed=False)
                self.W.fill(TAP((total,), SHARED * eb + l * run, [run], [1]), lane=l, managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * self.N * self.K

        def reference(self, W, x, r, *, e0=0, e1=0, e2=0, e3=0, e4=0, e5=0, e6=0, e7=0):
            eb = self.expert_bytes()
            flat = np.asarray(W, np.uint8).reshape(-1)
            x = np.asarray(x, np.float64)
            y = np.asarray(r, np.float64).copy()
            for i, at in enumerate([e0, e1, e2, e3, e4, e5, e6, e7, SHARED * eb]):
                d = dequantize(flat[int(at) : int(at) + eb], self.N, self.KE).astype(np.float64)
                y += d @ x[i * self.KE : (i + 1) * self.KE]
            return y.astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="q4nx partial sums in bf16")

    return MoEGateUp, MoEDown


class FusedLayer:
    def __init__(self, rng, l: int):
        from iron_q4 import BLOCK_BYTES

        t0 = time.time()
        s = 1 / np.sqrt(HIDDEN)
        # Random q4nx draws: the gate and up alternation of gu_table changes nothing for them.
        self.gu = iron.param(rand_q4(rng, 2 * INTER, HIDDEN, EXPERTS + 1, s))
        self.down = iron.param(rand_q4(rng, HIDDEN, INTER, EXPERTS + 1, 1 / np.sqrt(INTER)))
        self.gu_bytes = self.gu.shape[0] // (EXPERTS + 1) * BLOCK_BYTES
        self.down_bytes = self.down.shape[0] // (EXPERTS + 1) * BLOCK_BYTES
        self.router = iron.param(rand_bf16(rng, (EXPERTS, HIDDEN), s))
        self.norm = iron.param(np.ones(HIDDEN, bfloat16))
        if l % 8 == 0:
            print(f"layer {l}: weights in {time.time() - t0:.1f} s", flush=True)


class FusedMoEProbe(MoEProbe):
    """A layer in four operators: MoEGateUp (the 9 experts' gate and up, SiLU, the routing
    weights), MoEDown (their down and the residual), RMS norm and the next router."""

    def __init__(self, layers: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.layers = [FusedLayer(rng, l) for l in range(layers)]

    def _body(self, x, w, **e):
        GU, D = fused_operators()
        h, logits = x, None
        for l, L in enumerate(self.layers):
            if l:
                boundary()
            a = GU.step(L.gu, h, w, K=HIDDEN, NE=INTER,
                        **{f"e{i}": e[f"e{l}_{i}"] * L.gu_bytes for i in range(TOP)})
            y = D.step(L.down, a, h, KE=INTER, N=HIDDEN,
                       **{f"e{i}": e[f"e{l}_{i}"] * L.down_bytes for i in range(TOP)})
            h = F.rms_norm(y.reshape(1, HIDDEN), (HIDDEN,), L.norm, 1e-6).reshape(HIDDEN)
            logits = F.linear(h, L.router, precision=P)
        return h, logits


def fused_probe_class(layers: int):
    cls = probe_class(layers)
    return type("FusedMoEProbeGraph", (FusedMoEProbe, iron.Graph), dict(body=cls.body))


class Q4Layer:
    def __init__(self, rng, l: int, fmt: str):
        from iron_q4 import FORMATS, q_param

        t0 = time.time()
        s = 1 / np.sqrt(HIDDEN)
        self.gate = iron.param(rand_q4(rng, INTER, HIDDEN, EXPERTS, s, fmt))
        self.up = iron.param(rand_q4(rng, INTER, HIDDEN, EXPERTS, s, fmt))
        self.down = iron.param(rand_q4(rng, HIDDEN, INTER, EXPERTS, 1 / np.sqrt(INTER), fmt))
        self.expert_bytes = self.gate.shape[0] // EXPERTS * FORMATS[fmt]
        self.sh_gate = q_param(rand_bf16(rng, (INTER, HIDDEN), s).astype(np.float32), fmt)
        self.sh_up = q_param(rand_bf16(rng, (INTER, HIDDEN), s).astype(np.float32), fmt)
        self.sh_down = q_param(rand_bf16(rng, (HIDDEN, INTER), 1 / np.sqrt(INTER)).astype(np.float32), fmt)
        self.router = iron.param(rand_bf16(rng, (EXPERTS, HIDDEN), s))
        self.norm = iron.param(np.ones(HIDDEN, bfloat16))
        if l % 8 == 0:
            print(f"layer {l}: weights in {time.time() - t0:.1f} s", flush=True)


class Q4MoEProbe(MoEProbe):
    """The probe's layers with 4-bit experts read in place: gate and up an ExpertQGEMV each,
    down an ExpertDownQGEMV, the shared expert iron_q4's QGEMV, no staging copies."""

    def __init__(self, layers: int, fmt: str, seed: int = 0):
        from iron_q4 import install_linear

        install_linear()
        rng = np.random.default_rng(seed)
        self.fmt = fmt
        self.layers = [Q4Layer(rng, l, fmt) for l in range(layers)]

    def _body(self, x, w, **e):
        G, D = expert_operators()
        h, logits = x, None
        for l, L in enumerate(self.layers):
            if l:
                boundary()
            off = {f"e{i}": e[f"e{l}_{i}"] * L.expert_bytes for i in range(TOP)}
            g = G.step(L.gate, h, fmt=self.fmt, K=HIDDEN, NE=INTER, **off)
            u = G.step(L.up, h, fmt=self.fmt, K=HIDDEN, NE=INTER, **off)
            a = F.silu(g) * u * w
            y = D.step(L.down, a, fmt=self.fmt, KE=INTER, N=HIDDEN, **off)
            s = F.linear(F.silu(F.linear(h, L.sh_gate, precision=P)) * F.linear(h, L.sh_up, precision=P),
                         L.sh_down, precision=P)
            h = F.rms_norm((h + y + s).reshape(1, HIDDEN), (HIDDEN,), L.norm, 1e-6).reshape(HIDDEN)
            logits = F.linear(h, L.router, precision=P)
        return h, logits


def q4_probe_class(layers: int):
    cls = probe_class(layers)
    return type("Q4MoEProbeGraph", (Q4MoEProbe, iron.Graph), dict(body=cls.body))


class Q4Head(iron.Graph):
    """``reps`` q4nx GEMVs of (vocab, dim), each its own weights: IRON's flm.LMHead (FastFlowLM's
    q4nx kernel, an RMS norm and a softcap folded in), the q4nx GEMV IRON has today."""

    def __init__(self, dim: int, vocab: int, reps: int):
        from iron.operators.flm.lm_head.op import LMHead, _draw
        from iron.operators.flm.q4nx import BLOCK_BYTES

        self.op = LMHead(dim=dim, vocab=vocab, softcap=1000.0)
        w = _draw(self.op)["given"][self.op.w]
        block = BLOCK_BYTES // 4  # whole q4nx blocks, so that scales stay scales
        self.ws = [iron.param(np.roll(w, 7 * block * k)) for k in range(reps)]

    def body(self, x):
        from iron.operators.flm.lm_head.op import LMHead

        y = None
        for w in self.ws:
            y = LMHead.step(w, x, dim=self.op.dim, vocab=self.op.vocab, softcap=1000.0)
        return y


class Tiny(iron.Graph):
    """``n`` commands of one add each."""

    def __init__(self, n: int):
        self.n = n

    def body(self, x):
        for k in range(self.n):
            if k:
                boundary()
            x = x + x
        return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--tiny", type=int, default=0)
    ap.add_argument("--q4", type=int, default=0, help="a graph of q4nx GEMVs of this many rows by 2048")
    ap.add_argument("--reps", type=int, default=1, help="GEMVs in the --q4 graph")
    ap.add_argument("--channels", type=int, default=0, help="Copy's num_channels (0: its own)")
    ap.add_argument("--down-t", action="store_true", help="down transposed, a one-row GEMM")
    ap.add_argument("--q4moe", choices=("q4", "q8"), help="4-bit (or 8-bit) experts read in place")
    ap.add_argument("--fused", action="store_true", help="with --q4moe q4: the layer in four operators")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bind_device("npu2", 8)
    global DOWN_T
    DOWN_T = a.down_t
    t0 = time.time()
    if a.q4:
        g = Q4Head(HIDDEN, a.q4, a.reps)
        cg = g.compile(x=(2 * HIDDEN,))
        m = export_graph(cg, a.out, f"q4_{a.q4}x{a.reps}")
        print(f"compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}")
        if a.check:
            rng = np.random.default_rng(1)
            x = np.concatenate([rng.standard_normal(HIDDEN), rng.uniform(0.5, 1.5, HIDDEN)])
            write_check(g, m, a.out, dict(x=x.astype(bfloat16)))
        return
    if a.tiny:
        g = Tiny(a.tiny)
        cg = g.compile(x=(HIDDEN,))
        m = export_graph(cg, a.out, f"tiny{a.tiny}")
        print(f"compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}")
        if a.check:
            x = rand_bf16(np.random.default_rng(1), (HIDDEN,), 1e-3)
            write_check(g, m, a.out, dict(x=x))
        return
    if a.q4moe:
        if a.fused:
            g = fused_probe_class(a.layers)(a.layers)
            cg = g.compile(x=(HIDDEN,), w=(WV,))
        else:
            g = q4_probe_class(a.layers)(a.layers, a.q4moe)
            cg = g.compile(x=(HIDDEN,), w=(TOP * INTER,))
        meta = dict(layers=a.layers, fmt=a.q4moe, fused=a.fused, hidden=HIDDEN, experts=EXPERTS, inter=INTER,
                    top=TOP)
        m = export_graph(cg, a.out, f"moeprobe_{a.q4moe}{'_fused' if a.fused else ''}", meta)
        print(f"compiled and exported in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if a.check:
            rng = np.random.default_rng(2)
            x = rand_bf16(rng, (HIDDEN,), 1.0)
            wexp = rng.dirichlet(np.ones(TOP)).astype(np.float32)
            if a.fused:  # the 8 picked, the shared expert's (a sigmoid gate's), zeros
                wexp = np.concatenate([wexp, [0.6], np.zeros(WV - TOP - 1)]).astype(bfloat16)
            else:
                wexp = np.repeat(wexp, INTER).astype(bfloat16)
            values = {f"e{l}_{i}": int(v) for l in range(a.layers)
                      for i, v in enumerate(rng.choice(EXPERTS, TOP, replace=False))}
            write_check(g, m, a.out, dict(x=x, w=wexp), values)
        return
    g = probe_class(a.layers)(a.layers, a.split, a.channels)
    cg = g.compile(x=(HIDDEN,), w=(TOP * INTER,))
    meta = dict(layers=a.layers, split=a.split, channels=a.channels, hidden=HIDDEN, experts=EXPERTS, inter=INTER, top=TOP)
    m = export_graph(cg, a.out, "moeprobe", meta)
    print(f"compiled and exported in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    if a.check:
        rng = np.random.default_rng(2)
        x = rand_bf16(rng, (HIDDEN,), 1.0)
        wexp = np.repeat(rng.dirichlet(np.ones(TOP)).astype(np.float32), INTER).astype(bfloat16)
        values = {f"e{l}_{i}": int(v) for l in range(a.layers)
                  for i, v in enumerate(rng.choice(EXPERTS, TOP, replace=False))}
        write_check(g, m, a.out, dict(x=x, w=wexp), values)


if __name__ == "__main__":
    main()
