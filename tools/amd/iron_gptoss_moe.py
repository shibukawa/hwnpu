"""gpt-oss's experts for one token on the device, on mx4: the checkpoint's MXFP4 kept exactly.

gpt-oss (openai/gpt-oss-20b) routes each token to 4 of 32 experts of 2880 in-features and
2880 intermediate ones, stored as MXFP4 (e2m1 codes, a power of two per 32 in-features of an
out-feature). Rounding them to q4nx adds 6.6 % weight error; q6 holds them exactly at 6.5 bits.
mx4 holds them exactly at 4.55: q4nx's strips of nibbles (the e2m1 codes as they are), a bf16
scale 2^(E - 128) per group, and the block's out-features' bias (kernels/moe_mx4.cc).

A block is 32 out-features by KT = 320 in-features (2880 = 9 k-blocks, no padding):

  MX4GateUp  y_e = glu(G_e x + g_e, U_e x + u_e) for the 4 experts a call picks (byte offsets
             e0..e3 per call, as iron_moeprobe's), I = 2944 out-features an expert (2880 and 64 of
             zeros: 92 blocks, 23 a core): expert e on columns 2e and 2e + 1, core q of its
             four (column half h, core c: q = 2 h + c) the out-blocks q, q + 4, ...
  MX4Down    y = r + sum_e w_e (D_e y_e + d_e), 3072 out-features (2880 and 192 rows of zeros,
             the residual's padding), 12 out-blocks a column, 6 a core; the routing weights w
             and the column's residual ride on the activations' channel

    python tools/amd/iron_gptoss_moe.py check OUT     # a graph of both, random mx4 experts
    go run ./tools/amd/irongraph OUT
"""

import argparse
import sys
from typing import ClassVar
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

KT, MT, PR, GROUP = 320, 32, 16, 32
BLOCK = 2 * MT * KT // GROUP + MT * KT // 2 + 2 * MT  # 5824
TOP, EXPERTS = 4, 32
HIDDEN, INTER = 2880, 2880
IPAD = 2944  # an expert's gate-up out-features: 92 blocks, 23 for each of its 4 cores
NPAD = 3072  # the residual's width (48 column blocks of 64: whole on 8 columns)
WV = 16  # MoERoute's routing weights (iron_moe.WV)
LIMIT, ALPHA = 7.0, 1.702
FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def blocks(codes, exps, bias):
    """mx4 blocks of one weight: codes (N, K) e2m1 nibbles, exps (N, K / 32) MXFP4 exponents,
    bias (N,) or None; N a multiple of 32, K of KT. uint8 (N / 32, K / KT, BLOCK), [out-block][k-block]."""
    codes = np.asarray(codes, np.uint8)
    N, K = codes.shape
    nb, kb = N // MT, K // KT
    sc = np.ldexp(np.float32(1), np.asarray(exps, np.int32) - 128).astype(bfloat16)
    s = sc.reshape(nb, MT, kb, KT // GROUP).transpose(0, 2, 3, 1).reshape(nb, kb, -1)
    c = codes.reshape(nb, MT // PR, PR, kb, KT).transpose(0, 3, 1, 4, 2).reshape(nb, kb, -1)
    packed = (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)
    b = np.zeros((nb, kb, MT), bfloat16)
    if bias is not None:
        b[:, 0] = np.asarray(bias, np.float32).reshape(nb, MT).astype(bfloat16)
    out = np.concatenate([s.view(np.uint8), packed, b.view(np.uint8)], -1)
    assert out.shape[-1] == BLOCK
    return out


def dequantize(blk):
    """(N, K) float32 weights and (N,) bias of blocks' output (out-block, k-block, BLOCK)."""
    blk = np.asarray(blk, np.uint8)
    nb, kb, _ = blk.shape
    ns = MT * KT // GROUP
    s = blk[..., : 2 * ns].copy().view(bfloat16).astype(np.float32).reshape(nb, kb, KT // GROUP, MT)
    p = blk[..., 2 * ns : 2 * ns + MT * KT // 2]
    c = np.empty((nb, kb, MT * KT), np.uint8)
    c[..., 0::2], c[..., 1::2] = p & 15, p >> 4
    c = c.reshape(nb, kb, MT // PR, KT, PR).transpose(0, 2, 4, 1, 3).reshape(nb * MT, kb * KT)
    v = 2 * FP4[c]
    s = s.transpose(0, 3, 1, 2).reshape(nb * MT, kb * KT // GROUP)
    w = v * np.repeat(s, GROUP, 1)
    bias = blk[:, 0, 2 * ns + MT * KT // 2 :].copy().view(bfloat16).astype(np.float32).reshape(-1)
    return w, bias


def gate_up_table(codes, exps, bias):
    """One expert's MX4GateUp table from the checkpoint's gate_up (2 I, K) interleaved rows
    (gate even, up odd), their exponents and bias (2 I,): uint8 (bytes,), MX4GateUp's order."""
    codes, exps, bias = np.asarray(codes), np.asarray(exps), np.asarray(bias, np.float32)
    I = codes.shape[0] // 2
    pad = IPAD - I

    def part(rows):
        c = np.concatenate([codes[rows], np.zeros((pad, codes.shape[1]), np.uint8)])
        e = np.concatenate([exps[rows], np.full((pad, exps.shape[1]), 127, exps.dtype)])
        b = np.concatenate([bias[rows], np.zeros(pad, np.float32)])
        return blocks(c, e, b)  # (92, 9, BLOCK)

    g, u = part(slice(0, None, 2)), part(slice(1, None, 2))
    nb, kb = g.shape[:2]
    per = nb // 4
    out = np.empty((2, per, 2, kb, 2, BLOCK), np.uint8)  # [half][s][g/u][kb][core]
    for h in range(2):
        for c in range(2):
            idx = np.arange(per) * 4 + 2 * h + c
            out[h, :, 0, :, c] = g[idx]
            out[h, :, 1, :, c] = u[idx]
    return out.reshape(-1)


def down_table(codes, exps, bias):
    """One expert's MX4Down table from the checkpoint's down (H, I), exponents and bias (H,)."""
    codes, exps, bias = np.asarray(codes), np.asarray(exps), np.asarray(bias, np.float32)
    H, K = codes.shape
    pad = NPAD - H
    c = np.concatenate([codes, np.zeros((pad, K), np.uint8)])
    e = np.concatenate([exps, np.full((pad, exps.shape[1]), 127, exps.dtype)])
    b = np.concatenate([bias, np.zeros(pad, np.float32)])
    d = blocks(c, e, b)  # (96, 9, BLOCK)
    kb = d.shape[1]
    out = np.empty((8, 6, kb, 2, BLOCK), np.uint8)  # [column][j][kb][core]
    for l in range(8):
        for j in range(6):
            for c_ in range(2):
                out[l, j, :, c_] = d[12 * l + 2 * j + c_]
    return out.reshape(-1)


def unpack_gate_up(t, K=HIDDEN):
    """The (gate (IPAD, K), up, gate bias, up bias) float32 of gate_up_table's bytes."""
    kb = K // KT
    per = IPAD // MT // 4
    t = np.asarray(t, np.uint8).reshape(2, per, 2, kb, 2, BLOCK)
    g = np.empty((IPAD // MT, kb, BLOCK), np.uint8)
    u = np.empty_like(g)
    for h in range(2):
        for c in range(2):
            idx = np.arange(per) * 4 + 2 * h + c
            g[idx], u[idx] = t[h, :, 0, :, c], t[h, :, 1, :, c]
    (gw, gb), (uw, ub) = dequantize(g), dequantize(u)
    return gw, uw, gb, ub


def unpack_down(t, K=INTER):
    kb = K // KT
    t = np.asarray(t, np.uint8).reshape(8, 6, kb, 2, BLOCK)
    d = np.empty((NPAD // MT, kb, BLOCK), np.uint8)
    for l in range(8):
        for j in range(6):
            for c in range(2):
                d[12 * l + 2 * j + c] = t[l, j, :, c]
    return dequantize(d)


def glu(g, u):
    g = np.minimum(g, LIMIT)
    u = np.clip(u, -LIMIT, LIMIT)
    return (u + 1) * g / (1 + np.exp(-ALPHA * g))


def operators():
    """MX4GateUp and MX4Down (module docstring)."""
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

    import iron_moeprobe

    CB = 2 * MT
    blk_t = np.ndarray[(BLOCK,), np.dtype[np.uint8]]
    half_t = np.ndarray[(MT,), np.dtype[bfloat16]]
    acc_t = np.ndarray[(MT,), np.dtype[np.float32]]

    def kernels(x_len: int, k: int, xe: int = 0, w_at: int = 0):
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        x = np.ndarray[(x_len,), np.dtype[bfloat16]]
        flags = [f"-DMX_K={k}", f"-DMX_KT={KT}", f"-DMX_XE={xe or k}", f"-DMX_W={w_at}"]
        common = dict(source_file=str(here / "moe_mx4.cc"),
                      include_dirs=[str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
                      compile_flags=flags, object_file_name=f"moe_mx4_{x_len}_{k}_{xe}_{w_at}.o")
        f = lambda name, args: ExternalFunction(name, arg_types=args, **common)  # noqa: E731
        return dict(
            init=f("mx4_init", [blk_t, acc_t]),
            block=f("mx4_block", [blk_t, x, acc_t, np.int32]),
            block_e=f("mx4_block_e", [blk_t, x, acc_t, np.int32, np.int32]),
            glu=f("mx4_glu", [acc_t, acc_t, half_t]),
            zero=f("mx4_zero", [acc_t]),
            fold=f("mx4_fold", [acc_t, acc_t, x, np.int32]),
            end_add=f("mx4_end_add", [acc_t, x, np.int32, half_t]),
        )

    def offset(op, e):
        v = op.value(f"e{e}")
        return v if v.per_call else None

    def offsets(op, given, eb):
        return iron_moeprobe.picked_offsets(op, given, eb)

    class MX4GateUp(Operator):
        """``y_e = glu(G_e x + g_e, U_e x + u_e)`` for the 4 experts a call picks (module docstring)."""

        rows: int = param(array=True)  # blocks in the table
        K: int = param(default=HIDDEN, array=True)
        I: int = param(default=IPAD, array=True)
        XK: int = param(default=NPAD, array=True)  # x's length (the residual's width)
        YN: int = param(derive=lambda op: TOP * op.I, array=True)
        picked: int = param(default=-1, repr=False)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, BLOCK, dtype=np.uint8, obj_shape=(2, BLOCK), per=(num_aie_columns,))
        x = In(XK, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
        y = Out(YN, obj_shape=(CB,), per=(num_aie_columns,))
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)

        def resolve(self, dev):
            return self.replace(num_aie_columns=2 * TOP)

        def per_core(self) -> int:
            return self.I // MT // 4

        def expert_bytes(self) -> int:
            return 2 * (self.I // MT) * (self.K // KT) * BLOCK

        def build(self) -> Built:
            K, n = self.K, self.num_aie_columns
            kbs, per = K // KT, self.per_core()
            k = kernels(K, K)
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            w_l1 = [f.cons().split([0, BLOCK], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
            x_t = np.ndarray[(K,), np.dtype[bfloat16]]
            x_l3 = [ObjectFifo(x_t, depth=1) for _ in range(n)]
            x_l1 = [f.cons().forward(obj_type=x_t, tile=AnyMemTile) for f in x_l3]
            y_l1 = [f.prod().join([0, MT], obj_types=[half_t, half_t], depths=[2, 2], tile=AnyMemTile)
                    for f in y_l3]

            def core(w_f, x_f, y_f, g, u, init, block, glu_k):
                x = x_f.acquire(1)
                for _ in range_(per):
                    for acc in (g, u):
                        blk = w_f.acquire(1)
                        init(blk, acc)
                        block(blk, x, acc, 0)
                        w_f.release(1)
                        for kb in range_(1, kbs):
                            kb32 = index.casts(T.i32(), kb)
                            blk = w_f.acquire(1)
                            block(blk, x, acc, kb32)
                            w_f.release(1)
                    y = y_f.acquire(1)
                    glu_k(g, u, y)
                    y_f.release(1)
                x_f.release(1)

            workers = []
            for i in range(n):
                for j in range(2):
                    g, u = Buffer(acc_t), Buffer(acc_t)
                    workers.append(Worker(core, [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), g, u,
                                                 k["init"], k["block"], k["glu"]]))
            self.W.bind([f.prod() for f in w_l3])
            self.x.bind([f.prod() for f in x_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            I, K, YN, n = self.I, self.K, self.YN, self.num_aie_columns
            eb, per = self.expert_bytes(), self.per_core()
            total = self.rows * BLOCK
            drains = []
            for l in range(n):
                e, h = l // 2, l % 2
                drains.append(self.y.drain(TAP((YN,), e * I + 2 * h * MT, [per, CB], [4 * MT, 1]), lane=l,
                                           wait=True, managed=False))
            for l in range(n):
                self.x.fill(TAP((self.XK,), 0, [K], [1]), lane=l, managed=False)
            for l in range(n):
                e, h = l // 2, l % 2
                self.W.fill(TAP((total,), h * eb // 2, [eb // 2], [1]), lane=l, offset_parameter=offset(self, e),
                            managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * 2 * self.YN * self.K

        def reference(self, W, x, *, e0=0, e1=0, e2=0, e3=0):
            eb = self.expert_bytes()
            flat = np.asarray(W, np.uint8).reshape(-1)
            x = np.asarray(x, np.float64)[: self.K]
            ys = []
            for at in offsets(self, [e0, e1, e2, e3], eb):
                gw, uw, gb, ub = unpack_gate_up(flat[int(at) : int(at) + eb], self.K)
                ys.append(glu(gw @ x + gb, uw @ x + ub))
            return np.concatenate(ys).astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.02, 1e-2, note="mx4 partial sums in two bf16 parts, exp_f32")

    class MX4Down(Operator):
        """``y = r + sum_e w_e (D_e y_e + d_e)`` over the 4 experts a call picks (module docstring)."""

        aiecc_flags: ClassVar[tuple[str, ...]] = ("--reclaim-runtime-bds",)

        rows: int = param(array=True)
        I: int = param(default=IPAD, array=True)  # the activations' stride an expert
        K: int = param(default=INTER, array=True)
        N: int = param(default=NPAD, array=True)
        XN: int = param(derive=lambda op: TOP * op.I, array=True)
        R: int = param(derive=lambda op: op.N // 8, array=True)
        picked: int = param(default=-1, repr=False)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, BLOCK, dtype=np.uint8, obj_shape=(2, BLOCK), per=(num_aie_columns,))
        x = In(XN, obj_shape=(XN,), per=(num_aie_columns,), replicate=True)
        r = In(N, obj_shape=(R,), per=(num_aie_columns,))
        w = In(WV, obj_shape=(WV,), per=(num_aie_columns,), replicate=True)
        y = Out(N, obj_shape=(CB,), per=(num_aie_columns,))
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)

        def resolve(self, dev):
            return self.replace(num_aie_columns=8)

        def expert_bytes(self) -> int:
            return (self.N // MT) * (self.K // KT) * BLOCK

        def build(self) -> Built:
            n, I, K = self.num_aie_columns, self.I, self.K
            kbs, R, XN = K // KT, self.R, self.XN
            outs = self.N // MT // n // 2  # out-blocks a core
            obj = XN + R + WV
            k = kernels(obj, K, xe=I, w_at=XN + R)
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            w_l1 = [f.cons().split([0, BLOCK], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
            x_t = np.ndarray[(obj,), np.dtype[bfloat16]]
            x_l3 = [ObjectFifo(x_t, depth=1) for _ in range(n)]
            x_l1 = [f.cons().forward(obj_type=x_t, tile=AnyMemTile) for f in x_l3]
            y_l1 = [f.prod().join([0, MT], obj_types=[half_t, half_t], depths=[2, 2], tile=AnyMemTile)
                    for f in y_l3]

            def core(w_f, x_f, y_f, acc, tots, c, init, block_e, zero, fold, end_add):
                x = x_f.acquire(1)
                for t in tots:
                    zero(t)
                for e in range_(TOP):
                    e32 = index.casts(T.i32(), e)
                    for t in tots:
                        blk = w_f.acquire(1)
                        init(blk, acc)
                        block_e(blk, x, acc, e32, 0)
                        w_f.release(1)
                        for kb in range_(1, kbs):
                            kb32 = index.casts(T.i32(), kb)
                            blk = w_f.acquire(1)
                            block_e(blk, x, acc, e32, kb32)
                            w_f.release(1)
                        fold(acc, t, x, e32)
                for j, t in enumerate(tots):
                    y = y_f.acquire(1)
                    end_add(t, x, XN + (2 * j + c) * MT, y)
                    y_f.release(1)
                x_f.release(1)

            workers = []
            for i in range(n):
                for j in range(2):
                    acc = Buffer(acc_t)
                    tots = [Buffer(acc_t) for _ in range(outs)]
                    workers.append(Worker(
                        lambda w_f, x_f, y_f, acc, tots, a, b, z, f_, ea, c=j: core(
                            w_f, x_f, y_f, acc, tots, c, a, b, z, f_, ea),
                        [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), acc, tots,
                         k["init"], k["block_e"], k["zero"], k["fold"], k["end_add"]]))
            self.W.bind([f.prod() for f in w_l3])
            self.x.bind([f.prod() for f in x_l3])
            self.r.bind([f.prod() for f in x_l3])
            self.w.bind([f.prod() for f in x_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            n, N, XN, R = self.num_aie_columns, self.N, self.XN, self.R
            eb = self.expert_bytes()
            run = eb // n
            total = self.rows * BLOCK
            drains = [self.y.drain(TAP((N,), l * R, [R], [1]), lane=l, wait=True, managed=False)
                      for l in range(n)]
            for l in range(n):
                self.x.fill(TAP((XN,), 0, [XN], [1]), lane=l, managed=False)
                self.r.fill(TAP((N,), l * R, [R], [1]), lane=l, managed=False)
                self.w.fill(TAP((WV,), 0, [WV], [1]), lane=l, managed=False)
            for l in range(n):
                for e in range(TOP):
                    self.W.fill(TAP((total,), l * run, [run], [1]), lane=l, offset_parameter=offset(self, e),
                                managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * TOP * self.N * self.K

        def reference(self, W, x, r, w, *, e0=0, e1=0, e2=0, e3=0):
            eb = self.expert_bytes()
            flat = np.asarray(W, np.uint8).reshape(-1)
            x = np.asarray(x, np.float64)
            w = np.asarray(w, np.float64)
            y = np.asarray(r, np.float64).copy()
            for i, at in enumerate(offsets(self, [e0, e1, e2, e3], eb)):
                d, b = unpack_down(flat[int(at) : int(at) + eb], self.K)
                y += w[i] * (d @ x[i * self.I : i * self.I + self.K] + b)
            return y.astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.02, 1e-2, note="mx4 partial sums in two bf16 parts")

    return MX4GateUp, MX4Down



def random_expert(rng, out: int, k: int):
    """Random e2m1 codes, exponents near the checkpoint's (117 .. 126) and a bias."""
    codes = rng.integers(0, 16, (out, k), dtype=np.uint8)
    exps = rng.integers(117, 125, (out, k // GROUP)).astype(np.uint8)
    bias = (rng.standard_normal(out) * 0.1).astype(np.float32)
    return codes, exps, bias


def _check(a) -> int:
    import iron
    from iron import Scratchpad
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    GU, D = operators()
    rng = np.random.default_rng(a.seed)
    E = a.experts
    gu = np.concatenate([gate_up_table(*random_expert(rng, 2 * INTER, HIDDEN)) for _ in range(E)])
    dn = np.concatenate([down_table(*random_expert(rng, HIDDEN, INTER)) for _ in range(E)])
    gu_eb, dn_eb = gu.size // E, dn.size // E
    picked = [int(v) for v in rng.choice(E, TOP, replace=False)]

    class G(iron.Graph):
        def __init__(self):
            self.gu = iron.param(gu.reshape(-1, BLOCK))
            self.dn = iron.param(dn.reshape(-1, BLOCK))

        def body(self, x, r, w, *, e0: Scratchpad[np.int32] = np.int32(0), e1: Scratchpad[np.int32] = np.int32(0),
                 e2: Scratchpad[np.int32] = np.int32(0), e3: Scratchpad[np.int32] = np.int32(0)):
            es = [e0, e1, e2, e3]
            y = GU.step(self.gu, x, rows=self.gu.shape[0], **{f"e{i}": es[i] * gu_eb for i in range(TOP)})
            return D.step(self.dn, y, r, w, rows=self.dn.shape[0], **{f"e{i}": es[i] * dn_eb for i in range(TOP)})

        def reference(self, x, r, w, e0=0, e1=0, e2=0, e3=0):
            x = np.asarray(x, np.float64)[:HIDDEN]
            y = np.asarray(r, np.float64).copy()
            for i, e in enumerate([e0, e1, e2, e3]):
                gw, uw, gb, ub = unpack_gate_up(gu[e * gu_eb : (e + 1) * gu_eb])
                act = glu(gw @ x + gb, uw @ x + ub).astype(bfloat16).astype(np.float64)
                d, b = unpack_down(dn[e * dn_eb : (e + 1) * dn_eb])
                y += float(np.asarray(w)[i]) * (d @ act[:INTER] + b)
            return y.astype(bfloat16)

    g = G()
    cg = g.compile(x=(NPAD,), r=(NPAD,), w=(WV,))
    m = export_graph(cg, a.out, "mx4moe", dict(experts=E))
    x = np.zeros(NPAD, np.float32)
    x[:HIDDEN] = rng.standard_normal(HIDDEN) * 0.5
    r = np.zeros(NPAD, np.float32)
    r[:HIDDEN] = rng.standard_normal(HIDDEN)
    w = np.zeros(WV, np.float32)
    w[:TOP] = rng.dirichlet(np.ones(TOP))
    write_check(g, m, a.out, dict(x=x.astype(bfloat16), r=r.astype(bfloat16), w=w.astype(bfloat16)),
                {f"e{i}": picked[i] for i in range(TOP)})
    print(f"{a.out}: MX4GateUp + MX4Down, {E} experts, picked {picked}, tables {gu.nbytes >> 20} + {dn.nbytes >> 20} MiB")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of MX4GateUp and MX4Down, random experts, and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--experts", type=int, default=6)
    c.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    return _check(a)


if __name__ == "__main__":
    sys.exit(main())
