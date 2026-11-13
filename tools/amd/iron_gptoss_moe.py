"""gpt-oss's experts for one token on the device, on mx4: the checkpoint's MXFP4 kept exactly.

gpt-oss (openai/gpt-oss-20b) routes each token to 4 of 32 experts of 2880 in-features and
2880 intermediate ones, stored as MXFP4 (e2m1 codes, a power of two per 32 in-features of an
out-feature). Rounding them to q4nx adds 6.6 % weight error; q6 holds them exactly at 6.5 bits.
mx4 holds them exactly at 4.3: q4nx's strips of nibbles (the e2m1 codes as they are), the
group's exponent byte (the scale 2^(E - 128)), and the block's out-features' bias (kernels/moe_mx4.cc).

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

MX4Experts runs the two as one operator (the decode step's form). For measuring: MX4_FUSED=1 makes
check's graph MX4Experts, MX4_ONLY=gu or down one of the two, MX4_NOCOMPUTE=1 builds kernels that
skip their products (the transfers alone), MX4_CORES (cores a column, 4; the tables' layout too),
MX4_WDEPTH (the decode step's weight objects a memtile holds, 2), MX4_SB (rows an object of the
prompt's x, 8).
"""

import argparse
import os
import sys
from typing import ClassVar
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

KT, MT, PR, GROUP = 320, 32, 16, 32
BLOCK = MT * KT // GROUP + MT * KT // 2 + 2 * MT  # 5504: the exponents (a byte a group), the codes, the bias
TOP, EXPERTS = 4, 32
HIDDEN, INTER = 2880, 2880
CORES = int(os.environ.get("MX4_CORES", "4"))  # cores a column (the memtile splits each object of the weights between them)
IPAD = 3072  # an expert's gate-up out-features (2880 and 192 of zeros): 96 blocks, 12 for each of its 8 cores
NPAD = 3072  # the residual's width (48 column blocks of 64: whole on 8 columns)
WV = 16  # MoERoute's routing weights (iron_moe.WV)
NOCOMPUTE = os.environ.get("MX4_NOCOMPUTE") == "1"
FASTDECODE = os.environ.get("MX4_FASTDECODE") == "1"  # a measurement: the prompt's kernels skip decoding the codes
BFP = os.environ.get("MX4_BFP", "1") == "1"  # the prompt's products on bfp16 (mac_8x8_8x8T, 16.179): the activations rounded to 8-bit mantissas; 0 the bf16 mmul
WDEPTH = int(os.environ.get("MX4_WDEPTH", "2"))  # the decode step's weight objects held in a memtile
LIMIT, ALPHA = 7.0, 1.702
FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def blocks(codes, exps, bias):
    """mx4 blocks of one weight: codes (N, K) e2m1 nibbles, exps (N, K / 32) MXFP4 exponents,
    bias (N,) or None; N a multiple of 32, K of KT. uint8 (N / 32, K / KT, BLOCK), [out-block][k-block]."""
    codes = np.asarray(codes, np.uint8)
    N, K = codes.shape
    nb, kb = N // MT, K // KT
    sc = np.asarray(exps, np.uint8)  # the MXFP4 exponents as they are: the scale 2^(E - 128)
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
    s = np.ldexp(np.float32(1), blk[..., :ns].astype(np.int32) - 128).reshape(nb, kb, KT // GROUP, MT)
    p = blk[..., ns : ns + MT * KT // 2]
    c = np.empty((nb, kb, MT * KT), np.uint8)
    c[..., 0::2], c[..., 1::2] = p & 15, p >> 4
    c = c.reshape(nb, kb, MT // PR, KT, PR).transpose(0, 2, 4, 1, 3).reshape(nb * MT, kb * KT)
    v = 2 * FP4[c]
    s = s.transpose(0, 3, 1, 2).reshape(nb * MT, kb * KT // GROUP)
    w = v * np.repeat(s, GROUP, 1)
    bias = blk[:, 0, ns + MT * KT // 2 :].copy().view(bfloat16).astype(np.float32).reshape(-1)
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
    q = 2 * CORES  # an expert's cores (two columns)
    per = nb // q
    out = np.empty((2, per, kb, 2, CORES, BLOCK), np.uint8)  # [half][s][kb][g/u][core]
    for h in range(2):
        for c in range(CORES):
            idx = np.arange(per) * q + CORES * h + c
            out[h, :, :, 0, c] = g[idx]
            out[h, :, :, 1, c] = u[idx]
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
    outs = 12 // CORES  # out-blocks a core
    out = np.empty((8, outs, kb, CORES, BLOCK), np.uint8)  # [column][j][kb][core]
    for l in range(8):
        for j in range(outs):
            for c_ in range(CORES):
                out[l, j, :, c_] = d[12 * l + CORES * j + c_]
    return out.reshape(-1)


def unpack_gate_up(t, K=HIDDEN):
    """The (gate (IPAD, K), up, gate bias, up bias) float32 of gate_up_table's bytes."""
    kb = K // KT
    q = 2 * CORES
    per = IPAD // MT // q
    t = np.asarray(t, np.uint8).reshape(2, per, kb, 2, CORES, BLOCK)
    g = np.empty((IPAD // MT, kb, BLOCK), np.uint8)
    u = np.empty_like(g)
    for h in range(2):
        for c in range(CORES):
            idx = np.arange(per) * q + CORES * h + c
            g[idx], u[idx] = t[h, :, :, 0, c], t[h, :, :, 1, c]
    (gw, gb), (uw, ub) = dequantize(g), dequantize(u)
    return gw, uw, gb, ub


def unpack_down(t, K=INTER):
    kb = K // KT
    outs = 12 // CORES
    t = np.asarray(t, np.uint8).reshape(8, outs, kb, CORES, BLOCK)
    d = np.empty((NPAD // MT, kb, BLOCK), np.uint8)
    for l in range(8):
        for j in range(outs):
            for c in range(CORES):
                d[12 * l + CORES * j + c] = t[l, j, :, c]
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

    CB = CORES * MT  # a column's out-features an object of y
    blk_t = np.ndarray[(BLOCK,), np.dtype[np.uint8]]
    half_t = np.ndarray[(MT,), np.dtype[bfloat16]]
    acc_t = np.ndarray[(MT,), np.dtype[np.float32]]

    def kernels(x_len: int, k: int, xe: int = 0, w_at: int = 0):
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        x = np.ndarray[(x_len,), np.dtype[bfloat16]]
        flags = [f"-DMX_K={k}", f"-DMX_KT={KT}", f"-DMX_XE={xe or k}", f"-DMX_W={w_at}"]
        if NOCOMPUTE:  # a measurement (MX4_NOCOMPUTE=1): the kernels skip their products
            flags.append("-DMX_NOCOMPUTE=1")
        common = dict(source_file=str(here / "moe_mx4.cc"),
                      include_dirs=[str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
                      compile_flags=flags, object_file_name=f"moe_mx4_{x_len}_{k}_{xe}_{w_at}{'_nc' if NOCOMPUTE else ''}.o")
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
        cores: int = param(default=CORES, array=True)  # cores a column (the tables' layout)
        wdepth: int = param(default=WDEPTH, array=True)
        nocompute: bool = param(default=NOCOMPUTE, array=True)  # a measurement's build (MX4_NOCOMPUTE)
        picked: int = param(default=-1, repr=False)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        x = In(XK, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
        y = Out(YN, obj_shape=(CB,), per=(num_aie_columns,))
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)

        def resolve(self, dev):
            return self.replace(num_aie_columns=2 * TOP)

        def per_core(self) -> int:
            return self.I // MT // (2 * CORES)

        def expert_bytes(self) -> int:
            return 2 * (self.I // MT) * (self.K // KT) * BLOCK

        def build(self) -> Built:
            K, n = self.K, self.num_aie_columns
            kbs, per = K // KT, self.per_core()
            k = kernels(K, K)
            w_l3 = [ObjectFifo(self.W.obj_type, depth=self.wdepth) for _ in range(n)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            w_l1 = [f.cons().split([BLOCK * c for c in range(CORES)], obj_types=[blk_t] * CORES, depths=[3] * CORES, tile=AnyMemTile)
                    for f in w_l3]
            x_t = np.ndarray[(K,), np.dtype[bfloat16]]
            x_l3 = [ObjectFifo(x_t, depth=1) for _ in range(n)]
            x_l1 = [f.cons().forward(obj_type=x_t, tile=AnyMemTile) for f in x_l3]
            y_l1 = [f.prod().join([MT * c for c in range(CORES)], obj_types=[half_t] * CORES, depths=[2] * CORES, tile=AnyMemTile)
                    for f in y_l3]

            def core(w_f, x_f, y_f, g, u, init, block, glu_k):
                # a k-block's gate block, then its up block (gate_up_table's order)
                x = x_f.acquire(1)
                for _ in range_(per):
                    bl = w_f.acquire(2)
                    init(bl[0], g)
                    init(bl[1], u)
                    block(bl[0], x, g, 0)
                    block(bl[1], x, u, 0)
                    w_f.release(2)
                    for kb in range_(1, kbs):
                        kb32 = index.casts(T.i32(), kb)
                        bl = w_f.acquire(2)
                        block(bl[0], x, g, kb32)
                        block(bl[1], x, u, kb32)
                        w_f.release(2)
                    y = y_f.acquire(1)
                    glu_k(g, u, y)
                    y_f.release(1)
                x_f.release(1)

            workers = []
            for i in range(n):
                for j in range(CORES):
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
                drains.append(self.y.drain(TAP((YN,), e * I + h * CB, [per, CB], [2 * CB, 1]), lane=l,
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
        cores: int = param(default=CORES, array=True)  # cores a column (the tables' layout)
        wdepth: int = param(default=WDEPTH, array=True)
        nocompute: bool = param(default=NOCOMPUTE, array=True)  # a measurement's build (MX4_NOCOMPUTE)
        picked: int = param(default=-1, repr=False)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
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
            outs = self.N // MT // n // CORES  # out-blocks a core
            obj = XN + R + WV
            k = kernels(obj, K, xe=I, w_at=XN + R)
            w_l3 = [ObjectFifo(self.W.obj_type, depth=self.wdepth) for _ in range(n)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            w_l1 = [f.cons().split([BLOCK * c for c in range(CORES)], obj_types=[blk_t] * CORES, tile=AnyMemTile) for f in w_l3]
            x_t = np.ndarray[(obj,), np.dtype[bfloat16]]
            x_l3 = [ObjectFifo(x_t, depth=1) for _ in range(n)]
            x_l1 = [f.cons().forward(obj_type=x_t, tile=AnyMemTile) for f in x_l3]
            y_l1 = [f.prod().join([MT * c for c in range(CORES)], obj_types=[half_t] * CORES, depths=[2] * CORES, tile=AnyMemTile)
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
                    end_add(t, x, XN + (CORES * j + c) * MT, y)
                    y_f.release(1)
                x_f.release(1)

            workers = []
            for i in range(n):
                for j in range(CORES):
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


    class MX4Experts(Operator):
        """MX4GateUp then MX4Down as one operator (one set of core programs, one weight stream, one
        x stream), the activations through DRAM between the phases: ``y = [a, r + sum_e w_e (D_e
        a_e + d_e)]``, a = glu(G_e x + g_e, U_e x + u_e). Two operators each paid about 0.4 ms of
        their own a layer (the streams' start and end), which the one pays once.

        Phase 1 is MX4GateUp's (the gate-up table's half runs, x), its output drained into y's first
        4 I; phase 2 MX4Down's (the down table's column runs; x's object: those activations, then
        the column's residual and the routing weights), into y's last N. Both phases' objects of
        x are XO long (phase 1's: x four times over and a piece, the kernels read its first K)."""

        aiecc_flags: ClassVar[tuple[str, ...]] = ("--reclaim-runtime-bds",)

        grows: int = param(array=True)  # blocks in the gate-up table
        drows: int = param(array=True)  # blocks in the down table
        K: int = param(default=HIDDEN, array=True)
        KD: int = param(default=INTER, array=True)
        I: int = param(default=IPAD, array=True)
        N: int = param(default=NPAD, array=True)
        XK: int = param(default=NPAD, array=True)  # x's length (the residual's width)
        XN: int = param(derive=lambda op: TOP * op.I, array=True)
        R: int = param(derive=lambda op: op.N // 8, array=True)
        XO: int = param(derive=lambda op: TOP * op.I + op.N // 8 + WV, array=True)
        YN: int = param(derive=lambda op: TOP * op.I + op.N, array=True)
        cores: int = param(default=CORES, array=True)
        form: int = param(default=2, array=True)  # 2: the down weights queued before phase 1 ends
        nocompute: bool = param(default=NOCOMPUTE, array=True)  # a measurement's build (MX4_NOCOMPUTE)
        picked: int = param(default=-1, repr=False)
        num_aie_columns: int = tunable(domain=widths)

        Wg = In(grows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        Wd = In(drows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        x = In(XK, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)
        r = In(N, obj_shape=(XO,), per=(num_aie_columns,))
        w = In(WV, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)
        y = Out(YN, obj_shape=(CB,), per=(num_aie_columns,))
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)
        d0 = per_call(derive=lambda op: 0)
        d1 = per_call(derive=lambda op: 0)
        d2 = per_call(derive=lambda op: 0)
        d3 = per_call(derive=lambda op: 0)

        def resolve(self, dev):
            return self.replace(num_aie_columns=8)

        def gu_bytes(self) -> int:
            return 2 * (self.I // MT) * (self.K // KT) * BLOCK

        def dn_bytes(self) -> int:
            return (self.N // MT) * (self.KD // KT) * BLOCK

        def build(self) -> Built:
            n, K, I, XN, R, XO = self.num_aie_columns, self.K, self.I, self.XN, self.R, self.XO
            kbs, kbd = K // KT, self.KD // KT
            per = I // MT // (2 * CORES)  # gate-up pairs a core
            outs = self.N // MT // n // CORES  # down out-blocks a core
            # one object for both phases (the gate-up's calls read x's first K; K is KD)
            kd = kernels(XO, self.KD, xe=I, w_at=XN + R)
            kg = kd
            w_l3 = [ObjectFifo(self.Wg.obj_type, depth=2) for _ in range(n)]
            w_l1 = [f.cons().split([BLOCK * c for c in range(CORES)], obj_types=[blk_t] * CORES,
                                   depths=[3] * CORES, tile=AnyMemTile) for f in w_l3]
            x_t = np.ndarray[(XO,), np.dtype[bfloat16]]
            x_l3 = [ObjectFifo(x_t, depth=1) for _ in range(n)]
            x_l1 = [f.cons().forward(obj_type=x_t, tile=AnyMemTile) for f in x_l3]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            y_l1 = [f.prod().join([MT * c for c in range(CORES)], obj_types=[half_t] * CORES,
                                  depths=[2] * CORES, tile=AnyMemTile) for f in y_l3]

            def core(w_f, x_f, y_f, g, u, acc, tots, c, init, block, glu_k, init_d, block_e, zero, fold, end_add):
                # phase 1: the gate and up of the core's pairs
                x = x_f.acquire(1)
                for _ in range_(per):
                    bl = w_f.acquire(2)
                    init(bl[0], g)
                    init(bl[1], u)
                    block(bl[0], x, g, 0)
                    block(bl[1], x, u, 0)
                    w_f.release(2)
                    for kb in range_(1, kbs):
                        kb32 = index.casts(T.i32(), kb)
                        bl = w_f.acquire(2)
                        block(bl[0], x, g, kb32)
                        block(bl[1], x, u, kb32)
                        w_f.release(2)
                    yo = y_f.acquire(1)
                    glu_k(g, u, yo)
                    y_f.release(1)
                x_f.release(1)
                # phase 2: the down of the core's out-blocks over the four experts
                x = x_f.acquire(1)
                for t in tots:
                    zero(t)
                for e in range_(TOP):
                    e32 = index.casts(T.i32(), e)
                    for t in tots:
                        blk = w_f.acquire(1)
                        init_d(blk, acc)
                        block_e(blk, x, acc, e32, 0)
                        w_f.release(1)
                        for kb in range_(1, kbd):
                            kb32 = index.casts(T.i32(), kb)
                            blk = w_f.acquire(1)
                            block_e(blk, x, acc, e32, kb32)
                            w_f.release(1)
                        fold(acc, t, x, e32)
                for j, t in enumerate(tots):
                    yo = y_f.acquire(1)
                    end_add(t, x, XN + (CORES * j + c) * MT, yo)
                    y_f.release(1)
                x_f.release(1)

            workers = []
            for i in range(n):
                for j in range(CORES):
                    tots = [Buffer(acc_t) for _ in range(outs)]
                    workers.append(Worker(
                        lambda w_f, x_f, y_f, g, u, acc, tots, a1, a2, a3, a4, a5, a6, a7, a8, c=j: core(
                            w_f, x_f, y_f, g, u, acc, tots, c, a1, a2, a3, a4, a5, a6, a7, a8),
                        [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), Buffer(acc_t), Buffer(acc_t),
                         Buffer(acc_t), tots, kg["init"], kg["block"], kg["glu"], kd["init"], kd["block_e"],
                         kd["zero"], kd["fold"], kd["end_add"]]))
            for name in ("Wg", "Wd"):
                getattr(self, name).bind([f.prod() for f in w_l3])
            for name in ("x", "r", "w"):
                getattr(self, name).bind([f.prod() for f in x_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            I, K, N, n, XN, R, XO, XK = self.I, self.K, self.N, self.num_aie_columns, self.XN, self.R, self.XO, self.XK
            per = I // MT // (2 * CORES)
            geb, deb = self.gu_bytes(), self.dn_bytes()
            run = deb // n
            gtotal, dtotal = self.grows * BLOCK, self.drows * BLOCK
            YN = self.YN
            # phase 1: x (four times, then a piece: an object of XO), the gate-up table's halves
            first = []
            for l in range(n):
                e, h = l // 2, l % 2
                first.append(self.y.drain(TAP((YN,), e * I + h * CB, [per, CB], [2 * CB, 1]), lane=l,
                                          wait=True, managed=False))
            for l in range(n):
                self.x.fill(TAP((XK,), 0, [XO // XK, XK], [0, 1]), lane=l, managed=False)
                self.x.fill(TAP((XK,), 0, [XO % XK], [1]), lane=l, managed=False)
            for l in range(n):
                e, h = l // 2, l % 2
                self.Wg.fill(TAP((gtotal,), h * geb // 2, [geb // 2], [1]), lane=l, offset_parameter=offset(self, e),
                             managed=False)
            # the down table's column runs right behind the gate-up's: they do not wait for phase 1
            for l in range(n):
                for i in range(TOP):
                    v = self.value(f"d{i}")
                    self.Wd.fill(TAP((dtotal,), l * run, [run], [1]), lane=l, offset_parameter=v if v.per_call else None,
                                 managed=False)
            second = [self.y.drain(TAP((YN,), XN + l * R, [R], [1]), lane=l, wait=True, managed=False)
                      for l in range(n)]
            for d in first:
                d.await_()
            # phase 2's x: the activations from y's first XN, the column's residual, the routing weights
            for l in range(n):
                self.x.fill(TAP((YN,), 0, [XN], [1]), lane=l, source=self.y, managed=False)
                self.r.fill(TAP((N,), l * R, [R], [1]), lane=l, managed=False)
                self.w.fill(TAP((WV,), 0, [WV], [1]), lane=l, managed=False)
            for d in second:
                d.await_()

        def ops(self) -> int:
            return 2 * 2 * TOP * self.I * self.K + 2 * TOP * self.N * self.KD

        def reference(self, Wg, Wd, x, r, w, *, e0=0, e1=0, e2=0, e3=0, d0=0, d1=0, d2=0, d3=0):
            ge, de = self.gu_bytes(), self.dn_bytes()
            fg, fd = np.asarray(Wg, np.uint8).reshape(-1), np.asarray(Wd, np.uint8).reshape(-1)
            xv = np.asarray(x, np.float64)[: self.K]
            a = []
            for at in offsets(self, [e0, e1, e2, e3], ge):
                gw, uw, gb, ub = unpack_gate_up(fg[int(at) : int(at) + ge], self.K)
                a.append(glu(gw @ xv + gb, uw @ xv + ub))
            a = np.concatenate(a).astype(bfloat16)
            av = a.astype(np.float64)
            wv = np.asarray(w, np.float64)
            out = np.asarray(r, np.float64).copy()
            for i, at in enumerate(offsets(self, [d0, d1, d2, d3], de)):
                dw, db = unpack_down(fd[int(at) : int(at) + de], self.KD)
                out += wv[i] * (dw @ av[i * self.I : i * self.I + self.KD] + db)
            return np.concatenate([a, out.astype(bfloat16)])

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.02, 1e-2, note="mx4 partial sums in two bf16 parts, exp_f32")

    return MX4GateUp, MX4Down, MX4Experts



def random_expert(rng, out: int, k: int):
    """Random e2m1 codes, exponents near the checkpoint's (117 .. 126) and a bias."""
    codes = rng.integers(0, 16, (out, k), dtype=np.uint8)
    exps = rng.integers(117, 125, (out, k // GROUP)).astype(np.uint8)
    bias = (rng.standard_normal(out) * 0.1).astype(np.float32)
    return codes, exps, bias


def _check(a) -> int:
    import iron
    from iron.operators.copy import Copy
    from iron import Scratchpad
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    GU, D, EX = operators()
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
            self.act0 = iron.param(np.zeros(TOP * IPAD, bfloat16))

        def body(self, x, r, w, *, e0: Scratchpad[np.int32] = np.int32(0), e1: Scratchpad[np.int32] = np.int32(0),
                 e2: Scratchpad[np.int32] = np.int32(0), e3: Scratchpad[np.int32] = np.int32(0)):
            es = [e0, e1, e2, e3]
            if os.environ.get("MX4_FUSED") == "1":  # MX4Experts: the two in one operator
                out = EX.step(self.gu, self.dn, x, r, w, grows=self.gu.shape[0], drows=self.dn.shape[0],
                              **{f"e{i}": es[i] * gu_eb for i in range(TOP)},
                              **{f"d{i}": es[i] * dn_eb for i in range(TOP)})
                return Copy.step(out[TOP * IPAD :])
            only = os.environ.get("MX4_ONLY")  # a measurement: one of the two operators
            if only == "down":
                Copy.step(x)  # an input must be read (a small copy, about 0.1 ms of the time)
                act = Copy.step(self.act0)
                return D.step(self.dn, act, r, w, rows=self.dn.shape[0], **{f"e{i}": es[i] * dn_eb for i in range(TOP)})
            y = GU.step(self.gu, x, rows=self.gu.shape[0], **{f"e{i}": es[i] * gu_eb for i in range(TOP)})
            if only == "gu":
                Copy.step(r)  # inputs must be read (small copies, about 0.1 ms each of the time)
                Copy.step(w)
                return y
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


# --- a prompt chunk's experts: every expert's weights once a pass ---------------------------

CAP = int(os.environ.get("MX4_CAP", "56"))  # a slot's rows (a core holds CAP rows' accumulators of a gate and an up block, and a decoded pair: 64 does not fit)
SB = int(os.environ.get("MX4_SB", "8"))  # rows an object of x
# A pass runs SLOTS slots of up to CAP rows: slot e < EXPERTS expert e's first CAP rows, the
# others an expert's further rows (its weights at a per-call offset, the graph's slot<i>; a
# command's scratchpad holds 32 words, so not every slot can take one). x and the activations
# stream a slot's CAP rows whatever its count, so CAP near a chunk's rows an expert (256 tokens:
# about 32) is what they cost; the overflow slots are the weights read again.
SLOTS = int(os.environ.get("MX4_SLOTS", "40"))
GROUPS = SLOTS // TOP  # a prompt runs the slots 4 at a time on the decode step's columns
COUNT_WORDS = 64  # the counts' int32 words (a slot's rows), the first object of the x channel
assert SLOTS % TOP == 0 and SLOTS <= COUNT_WORDS


def plan_slots(counts):
    """The slots of one pass for each expert's rows ``counts``: (expert, first row, rows) of each
    slot, expert e's first CAP rows in slot e, its further ones in the overflow slots in turn, the
    unused ones (0, 0, 0) (hwnpu's moeHostRouteMX4 plans the same way)."""
    counts = [int(c) for c in counts]
    out = [(e, 0, min(CAP, c)) for e, c in enumerate(counts)]
    for e, c in enumerate(counts):
        for r in range(CAP, c, CAP):
            out.append((e, r, min(CAP, c - r)))
    if len(out) > SLOTS:
        raise NotImplementedError(f"{len(out)} slots of {SLOTS} (hwnpu splits the chunk into passes)")
    return out + [(0, 0, 0)] * (SLOTS - len(out))


def counts_object(counts):
    """The x channel's first object: each slot's rows (int32), zeros after (SB x KT bf16)."""
    o = np.zeros(SB * KT * 2, np.uint8)
    o[: 4 * len(counts)] = np.asarray(counts, np.int32).view(np.uint8)
    return o.view(bfloat16)


def stream_x(rows, K=HIDDEN):
    """MX4GroupGateUp's x: by slot, by k-block, its CAP rows (``rows`` (SLOTS, CAP, K), those past
    its count anything) of KT in-features: (SLOTS, K / KT, CAP, KT) bf16."""
    r = np.asarray(rows).reshape(SLOTS, CAP, K // KT, KT)
    return np.ascontiguousarray(r.transpose(0, 2, 1, 3))


def group_reference_gate_up(gu, experts, counts, xs, K=HIDDEN):
    """MX4GroupGateUp's output (SLOTS, CAP, IPAD) of tables gu (bytes, EXPERTS experts), each slot's
    expert and rows, and xs."""
    eb = gu.size // EXPERTS
    x = np.asarray(xs, np.float64).reshape(SLOTS, K // KT, CAP, KT).transpose(0, 2, 1, 3).reshape(SLOTS, CAP, K)
    out = np.zeros((SLOTS, CAP, IPAD), np.float64)
    for s in range(SLOTS):
        n, e = int(counts[s]), int(experts[s])
        if n == 0:
            continue
        gw, uw, gb, ub = unpack_gate_up(gu[e * eb : (e + 1) * eb], K)
        out[s, :n] = glu(x[s, :n] @ gw.T + gb, x[s, :n] @ uw.T + ub)
    return out


def group_reference_down(dn, experts, counts, a, K=INTER):
    """MX4GroupDown's output (SLOTS, CAP, NPAD) of tables dn, each slot's expert and rows, and the
    activations a (SLOTS, CAP, IPAD)."""
    eb = dn.size // EXPERTS
    a = np.asarray(a, np.float64).reshape(SLOTS, CAP, IPAD)
    out = np.zeros((SLOTS, CAP, NPAD), np.float64)
    for s in range(SLOTS):
        n, e = int(counts[s]), int(experts[s])
        if n == 0:
            continue
        d, b = unpack_down(dn[e * eb : (e + 1) * eb], K)
        out[s, :n] = a[s, :n, :K] @ d.T + b
    return out


def _slot_values(ns):
    """The overflow slots' per-call offsets w<EXPERTS> .. w<SLOTS - 1> into a class body's namespace."""
    from aie.utils.compile.jit.params import per_call

    for i in range(EXPERTS, SLOTS):
        ns[f"w{i}"] = per_call(derive=lambda op: 0)


def _slot_value(op, s):
    """Slot s's per-call offset, or None (a fixed slot, or the value not given per call)."""
    if s < EXPERTS:
        return None
    v = op.value(f"w{s}")
    return v if v.per_call else None


def group_operators():
    """MX4GroupGateUp and MX4GroupDown: a prompt chunk's experts (moe_mx4_rows.cc), on the decode
    step's tables read in the order the rows need (no second copy of the weights).

    MX4GroupGateUp runs the experts 4 at a time, expert 4 g + j on columns 2 j and 2 j + 1 (the
    decode step's halves): each core's 23 out-block pairs, gate and up a k-block at a time (the
    table's order, the decode step's too), every row of the expert through each as
    it passes, its x the expert's rows a k-block at a time (stream_x, read again for each pair).
    MX4GroupDown runs the experts one after another, each column its 12 out-blocks of all 32."""
    import aie.dialects.arith as arith
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import Buffer, In, ObjectFifo, Out, TaskGroup, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param, tunable, widths
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    CB = CORES * MT  # a column's out-features an object of y
    blk_t = np.ndarray[(BLOCK,), np.dtype[np.uint8]]
    obj_t = np.ndarray[(SB * KT,), np.dtype[bfloat16]]
    half_t = np.ndarray[(MT,), np.dtype[bfloat16]]
    acc_t = np.ndarray[(CAP * MT,), np.dtype[np.float32]]
    cnt_t = np.ndarray[(COUNT_WORDS,), np.dtype[np.int32]]
    DEC = (KT // 8) * 4 * 64 + (KT // 32) * 4 * 8  # a block's bfp16 tiles (moe_mx4_rows.cc's DEC_BYTES)
    dec_t = np.ndarray[(DEC,), np.dtype[np.int8]]
    dec2_t = np.ndarray[(2 * DEC,), np.dtype[np.int8]]

    def kernels():
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        common = dict(source_file=str(here / "moe_mx4_rows.cc"),
                      include_dirs=[str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
                      compile_flags=[f"-DMX_KT={KT}", f"-DMX_CAP={CAP}", f"-DMX_SB={SB}"] + (["-DMX_NOCOMPUTE=1"] if NOCOMPUTE else [])
                      + (["-DMX_FASTDECODE=1"] if FASTDECODE else []) + (["-DMX_BFP=1"] if BFP else []),
                      object_file_name=f"moe_mx4_rows_mm_{KT}_{CAP}_{SB}{'_nc' if NOCOMPUTE else ''}{'_fd' if FASTDECODE else ''}{'_bfp' if BFP else ''}.o")
        f = lambda name, args: ExternalFunction(name, arg_types=args, **common)  # noqa: E731
        i32 = np.int32
        return dict(
            counts=f("mx4_rows_counts", [obj_t, cnt_t, i32]),
            init=f("mx4_rows_init", [blk_t, acc_t, cnt_t, i32]),
            block=f("mx4_rows_block", [blk_t, obj_t, acc_t, cnt_t, i32, i32]),
            block2=f("mx4_rows_block2", [blk_t, blk_t, obj_t, acc_t, acc_t, cnt_t, i32, i32]),
            glu=f("mx4_rows_glu", [acc_t, acc_t, cnt_t, i32, i32, half_t]),
            out=f("mx4_rows_out", [acc_t, cnt_t, i32, i32, half_t]),
            **(dict(dec=f("mx4_rows_dec", [blk_t, dec_t, cnt_t, i32]),
                    dec2=f("mx4_rows_dec2", [blk_t, blk_t, dec2_t, cnt_t, i32]),
                    blockd=f("mx4_rows_blockd", [dec_t, obj_t, acc_t, cnt_t, i32, i32]),
                    block2d=f("mx4_rows_block2d", [dec2_t, obj_t, acc_t, acc_t, cnt_t, i32, i32])) if BFP else {}),
        )

    def i32(v):
        return index.casts(T.i32(), v)

    class MX4GroupGateUp(Operator):
        """``a[e, r] = glu(G_e x[e, r] + g_e, U_e x[e, r] + u_e)`` for each expert's rows below
        its count (group_operators)."""

        aiecc_flags: ClassVar[tuple[str, ...]] = ("--reclaim-runtime-bds",)

        rows: int = param(array=True)  # blocks in the table
        K: int = param(default=HIDDEN, array=True)
        I: int = param(default=IPAD, array=True)
        XS: int = param(derive=lambda op: SLOTS * op.K * CAP, array=True)
        AN: int = param(derive=lambda op: SLOTS * CAP * op.I, array=True)
        CN: int = param(default=SB * KT, array=True)
        form: int = param(default=8 if BFP else 3, array=True)  # 3: the bf16 mmul a strip at a time; 8: bfp16 (MX_BFP), a block decoded once a slot
        nocompute: bool = param(default=NOCOMPUTE, array=True)  # a measurement's build (MX4_NOCOMPUTE)
        fastdecode: bool = param(default=FASTDECODE, array=True)  # a measurement's build (MX4_FASTDECODE)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        counts = In(CN, obj_shape=(SB * KT,), per=(num_aie_columns,), replicate=True)
        xs = In(XS, obj_shape=(SB * KT,), per=(num_aie_columns,), replicate=True)
        a = Out(AN, obj_shape=(CB,), per=(num_aie_columns,))
        _slot_values(locals())

        def resolve(self, dev):
            return self.replace(num_aie_columns=2 * TOP)

        def expert_bytes(self) -> int:
            return 2 * (self.I // MT) * (self.K // KT) * BLOCK

        def build(self) -> Built:
            n, kbs = self.num_aie_columns, self.K // KT
            per = self.I // MT // (2 * CORES)
            k = kernels()
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            w_l1 = [f.cons().split([BLOCK * c for c in range(CORES)], obj_types=[blk_t] * CORES, depths=[2] * CORES, tile=AnyMemTile) for f in w_l3]
            x_l3 = [ObjectFifo(obj_t, depth=2) for _ in range(n)]
            x_l1 = [f.cons().forward(obj_type=obj_t, depth=2, tile=AnyMemTile) for f in x_l3]
            y_l3 = [ObjectFifo(self.a.obj_type, depth=2) for _ in range(n)]
            y_l1 = [f.prod().join([MT * c for c in range(CORES)], obj_types=[half_t] * CORES, depths=[2] * CORES, tile=AnyMemTile) for f in y_l3]

            def core(w_f, x_f, y_f, g, u, cnt, j, counts, init, block2, glu_k, dec=None, dec2=None, block2d=None):
                # dec: the decoded pair's buffer; dec2, block2d: the kernels (BFP)
                c = x_f.acquire(1)
                counts(c, cnt, COUNT_WORDS)
                x_f.release(1)
                for grp in range_(GROUPS):
                    e = arith.addi(arith.muli(i32(grp), arith.constant(T.i32(), TOP)), arith.constant(T.i32(), j))
                    for _ in range_(per):
                        for kb in range(kbs):
                            bl = w_f.acquire(2)
                            if kb == 0:
                                init(bl[0], g, cnt, e)
                                init(bl[1], u, cnt, e)
                            if BFP:  # the pair decoded once, its blocks freed for the next pair's
                                dec2(bl[0], bl[1], dec, cnt, e)
                                w_f.release(2)
                            for b in range_(CAP // SB):
                                x = x_f.acquire(1)
                                if BFP:
                                    block2d(dec, x, g, u, cnt, e, i32(b))
                                else:
                                    block2(bl[0], bl[1], x, g, u, cnt, e, i32(b))
                                x_f.release(1)
                            if not BFP:
                                w_f.release(2)
                        for r in range_(CAP):
                            y = y_f.acquire(1)
                            glu_k(g, u, cnt, e, i32(r), y)
                            y_f.release(1)

            workers = []
            for i in range(n):
                for c in range(CORES):
                    extra = [Buffer(dec2_t), k["dec2"], k["block2d"]] if BFP else []
                    workers.append(Worker(
                        lambda w_f, x_f, y_f, g, u, cnt, a1, a2, a3, a4, *bfp, j=i // 2: core(
                            w_f, x_f, y_f, g, u, cnt, j, a1, a2, a3, a4, *bfp),
                        [w_l1[i][c].cons(), x_l1[i].cons(), y_l1[i][c].prod(), Buffer(acc_t), Buffer(acc_t),
                         Buffer(cnt_t), k["counts"], k["init"], k["block2"], k["glu"], *extra]))
            self.W.bind([f.prod() for f in w_l3])
            self.counts.bind([f.prod() for f in x_l3])
            self.xs.bind([f.prod() for f in x_l3])
            self.a.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            n, K, I = self.num_aie_columns, self.K, self.I
            kbs, per = K // KT, I // MT // (2 * CORES)
            eb = self.expert_bytes()
            total, B = self.rows * BLOCK, BLOCK
            for grp in range(GROUPS):
                tg = TaskGroup()
                for l in range(n):
                    j, h = l // 2, l % 2
                    e = TOP * grp + j  # the slot: its expert's weights at w<e> (else expert e % EXPERTS)
                    v = _slot_value(self, e)
                    at = h * eb // 2 if v is not None else (e % EXPERTS) * eb + h * eb // 2
                    if grp == 0:
                        self.counts.fill(TAP((self.CN,), 0, [self.CN], [1]), lane=l, group=tg)
                    self.W.fill(TAP((total,), at, [eb // 2], [1]), lane=l, group=tg, offset_parameter=v)
                    self.xs.fill(TAP((self.XS,), e * K * CAP, [K * CAP], [1]).repeat(per), lane=l, group=tg)
                    self.a.drain(TAP((self.AN,), e * CAP * I + h * CB, [per, CAP, CB], [2 * CB, I, 1]),
                                 lane=l, group=tg, wait=True)
                tg.finish()

        def ops(self) -> int:
            return 2 * 2 * SLOTS * CAP * self.I * self.K

        def reference(self, W, counts, xs, **w):
            cnt = np.asarray(counts).view(np.int32)[:SLOTS]
            W = np.asarray(W, np.uint8).reshape(-1)
            eb = W.size // EXPERTS
            ex = [int(w.get(f"w{s}", (s % EXPERTS) * eb)) // eb for s in range(SLOTS)]
            out = group_reference_gate_up(W, ex, cnt, xs, self.K)
            return out.astype(bfloat16).reshape(-1)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.02, 1e-2, note="mx4 partial sums in two bf16 parts, exp_f32")

    class MX4GroupDown(Operator):
        """``y[e, r] = D_e a[e, r] + d_e`` for each expert's rows below its count (group_operators)."""

        aiecc_flags: ClassVar[tuple[str, ...]] = ("--reclaim-runtime-bds",)

        rows: int = param(array=True)
        I: int = param(default=IPAD, array=True)
        K: int = param(default=INTER, array=True)
        N: int = param(default=NPAD, array=True)
        AN: int = param(derive=lambda op: SLOTS * CAP * op.I, array=True)
        YN: int = param(derive=lambda op: SLOTS * CAP * op.N, array=True)
        CN: int = param(default=SB * KT, array=True)
        form: int = param(default=8 if BFP else 3, array=True)  # 3: the bf16 mmul a strip at a time; 8: bfp16 (MX_BFP), a block decoded once a slot
        nocompute: bool = param(default=NOCOMPUTE, array=True)  # a measurement's build (MX4_NOCOMPUTE)
        fastdecode: bool = param(default=FASTDECODE, array=True)  # a measurement's build (MX4_FASTDECODE)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        counts = In(CN, obj_shape=(SB * KT,), per=(num_aie_columns,), replicate=True)
        a = In(AN, obj_shape=(SB * KT,), per=(num_aie_columns,), replicate=True)
        y = Out(YN, obj_shape=(CB,), per=(num_aie_columns,))
        _slot_values(locals())

        def resolve(self, dev):
            return self.replace(num_aie_columns=8)

        def expert_bytes(self) -> int:
            return (self.N // MT) * (self.K // KT) * BLOCK

        def build(self) -> Built:
            n, kbs = self.num_aie_columns, self.K // KT
            outs = self.N // MT // n // CORES
            k = kernels()
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
            w_l1 = [f.cons().split([BLOCK * c for c in range(CORES)], obj_types=[blk_t] * CORES, depths=[2] * CORES, tile=AnyMemTile) for f in w_l3]
            x_l3 = [ObjectFifo(obj_t, depth=2) for _ in range(n)]
            x_l1 = [f.cons().forward(obj_type=obj_t, depth=2, tile=AnyMemTile) for f in x_l3]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(n)]
            y_l1 = [f.prod().join([MT * c for c in range(CORES)], obj_types=[half_t] * CORES, depths=[2] * CORES, tile=AnyMemTile) for f in y_l3]

            def core(w_f, x_f, y_f, acc, cnt, counts, init, block, out, dec=None, dec_k=None, blockd=None):
                c = x_f.acquire(1)
                counts(c, cnt, COUNT_WORDS)
                x_f.release(1)
                for e in range_(SLOTS):
                    e32 = i32(e)
                    for _ in range_(outs):
                        for kb in range(kbs):
                            blk = w_f.acquire(1)
                            if kb == 0:
                                init(blk, acc, cnt, e32)
                            if BFP:  # the block decoded once, freed for the next
                                dec_k(blk, dec, cnt, e32)
                                w_f.release(1)
                            for b in range_(CAP // SB):
                                x = x_f.acquire(1)
                                if BFP:
                                    blockd(dec, x, acc, cnt, e32, i32(b))
                                else:
                                    block(blk, x, acc, cnt, e32, i32(b))
                                x_f.release(1)
                            if not BFP:
                                w_f.release(1)
                        for r in range_(CAP):
                            y = y_f.acquire(1)
                            out(acc, cnt, e32, i32(r), y)
                            y_f.release(1)

            workers = []
            for i in range(n):
                for c in range(CORES):
                    extra = [Buffer(dec_t), k["dec"], k["blockd"]] if BFP else []
                    workers.append(Worker(core, [w_l1[i][c].cons(), x_l1[i].cons(), y_l1[i][c].prod(), Buffer(acc_t),
                                                 Buffer(cnt_t), k["counts"], k["init"], k["block"], k["out"], *extra]))
            self.W.bind([f.prod() for f in w_l3])
            self.counts.bind([f.prod() for f in x_l3])
            self.a.bind([f.prod() for f in x_l3])
            self.y.bind([f.cons() for f in y_l3])
            return Built(workers, [])

        def sequence(self, rt):
            n, K, I, N = self.num_aie_columns, self.K, self.I, self.N
            kbs, outs = K // KT, N // MT // n // CORES
            eb = self.expert_bytes()
            run, R = eb // n, N // n
            total = self.rows * BLOCK
            for e in range(SLOTS):
                tg = TaskGroup()
                v = _slot_value(self, e)
                at = 0 if v is not None else (e % EXPERTS) * eb
                for l in range(n):
                    if e == 0:
                        self.counts.fill(TAP((self.CN,), 0, [self.CN], [1]), lane=l, group=tg)
                    self.W.fill(TAP((total,), at + l * run, [run], [1]), lane=l, group=tg, offset_parameter=v)
                    self.a.fill(TAP((self.AN,), e * CAP * I, [kbs, CAP, KT], [KT, I, 1]).repeat(outs), lane=l, group=tg)
                    self.y.drain(TAP((self.YN,), e * CAP * N + l * R, [outs, CAP, CB], [CB, N, 1]), lane=l,
                                 group=tg, wait=True)
                tg.finish()

        def ops(self) -> int:
            return 2 * SLOTS * CAP * self.N * self.K

        def reference(self, W, counts, a, **w):
            cnt = np.asarray(counts).view(np.int32)[:SLOTS]
            W = np.asarray(W, np.uint8).reshape(-1)
            eb = W.size // EXPERTS
            ex = [int(w.get(f"w{s}", (s % EXPERTS) * eb)) // eb for s in range(SLOTS)]
            out = group_reference_down(W, ex, cnt, a, self.K)
            return out.astype(bfloat16).reshape(-1)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.02, 1e-2, note="mx4 partial sums in two bf16 parts")

    return MX4GroupGateUp, MX4GroupDown


def _check_group(a) -> int:
    import iron
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    GU, D = group_operators()
    rng = np.random.default_rng(a.seed)
    gu = np.concatenate([gate_up_table(*random_expert(rng, 2 * INTER, HIDDEN)) for _ in range(EXPERTS)])
    dn = np.concatenate([down_table(*random_expert(rng, HIDDEN, INTER)) for _ in range(EXPERTS)])
    # each slot's rows (its expert the slot's index mod EXPERTS: the operators' offsets without
    # per-call values)
    counts = rng.integers(0, CAP + 1, SLOTS)
    counts[:3] = (0, CAP, 1)
    if os.environ.get("MX4_COUNTS") == "chunk":  # a measurement: a 256-token chunk's 1024 rows, about 32 an expert
        plan = plan_slots(rng.multinomial(1024, rng.dirichlet(np.full(EXPERTS, 3.0))))
        counts = np.array([c for _, _, c in plan])
    experts = [s % EXPERTS for s in range(SLOTS)]

    class G(iron.Graph):
        def __init__(self):
            self.gu = iron.param(gu.reshape(-1, BLOCK))
            self.dn = iron.param(dn.reshape(-1, BLOCK))

        def body(self, counts, xs):
            act = GU.step(self.gu, counts, xs, rows=self.gu.shape[0])
            if os.environ.get("MX4_ONLY") == "gu":  # a measurement: the gate-up alone
                return act
            return D.step(self.dn, counts, act, rows=self.dn.shape[0])

        def reference(self, cw, xs):
            c = np.asarray(cw).view(np.int32)[:SLOTS]
            act = group_reference_gate_up(gu, experts, c, xs).astype(bfloat16)
            if os.environ.get("MX4_ONLY") == "gu":
                return act.reshape(-1)
            return group_reference_down(dn, experts, c, act).astype(bfloat16).reshape(-1)

    g = G()
    cg = g.compile(counts=(SB * KT,), xs=(SLOTS * HIDDEN * CAP,))
    m = export_graph(cg, a.out, "mx4group", dict(cap=CAP))
    rows = (rng.standard_normal((SLOTS, CAP, HIDDEN)) * 0.5).astype(bfloat16)
    write_check(g, m, a.out, dict(counts=counts_object(counts), xs=stream_x(rows).reshape(-1)))
    print(f"{a.out}: MX4GroupGateUp + MX4GroupDown, counts {counts.tolist()}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of MX4GateUp and MX4Down, random experts, and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--experts", type=int, default=6)
    c.add_argument("--seed", type=int, default=3)
    gc = sub.add_parser("check-group", help="export a graph of MX4GroupGateUp and MX4GroupDown and its check")
    gc.add_argument("out", type=Path)
    gc.add_argument("--seed", type=int, default=4)
    a = ap.parse_args(argv)
    return _check_group(a) if a.cmd == "check-group" else _check(a)


if __name__ == "__main__":
    sys.exit(main())
