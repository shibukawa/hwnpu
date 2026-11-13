"""gpt-oss's decode step as the phases of one array (GptOssStep), configured once.

An operator of its own is loaded again every time it follows another one: about 45 us and 1.2 us
a KB of its cores' programs (design.md 16.178), about 25 of a token's 79 ms. A full ELF of one
array is loaded once: the firmware skips the load of what it holds (iron_gptoss_moe.py's check
with MX4_SINGLE: the experts alone 1.15 ms a command where they follow another operator at 1.78).
So the layer's operators become phases of one core program on one array, the experts' (4 cores a
column, a weight stream split between them, x's stream to all four, their outputs joined):

  E  the experts (iron_gptoss_moe's MX4Experts) with the post-attention RMSNorm done by each core
     on its x (the residual h and the norm's weight in x's object)
  Q  the q, k, v projection with the input RMSNorm, the bias and RoPE (iron_q4's QGEMVQKV)
  O  o_proj with its bias and the residual

A step of the graph is one turn of the cores' loop: its ``phases`` (not a parameter of the array:
steps differing in it share the array and run their own sequences) set which phases run, read by
the cores as runtime words. Every step declares every operand (the array's lanes are bound once);
a step passes anything of the right shape for those its phases do not use.

The projections' weights are q8x: iron_q4's q8 codes and scales (the same values as the prompt
chunk's table) in blocks of 32 out-features by 160 in-features with the block's bias, 5504 bytes
as an mx4 block, in the order the cores take them ([column][out-block][k-block][core]).

    python tools/amd/iron_gptoss_layer.py check OUT   # a graph of the phases, random weights
    go run ./tools/amd/irongraph OUT
"""

import argparse
import functools
import os
import sys
from typing import ClassVar
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

import iron_gptoss_moe as moe  # noqa: E402

CORES, COLS = moe.CORES, 8
BLOCK, MT, TOP, WV = moe.BLOCK, moe.MT, moe.TOP, moe.WV
HP = 3072  # the residual stream (padded)
XO = 12800  # x's object
NW_AT, ANG_AT, RES_AT = 6144, 9216, 12288  # kernels/gptoss_layer.cc's GL_NW, GL_ANG, GL_RES
QKT = 160  # q8x's in-features a block
ATT = 4096  # the heads' outputs
QN = 5120  # q, k and v's out-features
PH_E, PH_Q, PH_O = 1, 2, 4


def q8x_table(w, bias):
    """w (N, K) and bias (N,) as q8x blocks (uint8 (blocks, BLOCK)), [column][out-block][k-block]
    [core]: quantize8's codes and scales (K padded to whole blocks with zeros)."""
    w = np.asarray(w, np.float32)
    N, K = w.shape
    KP = -(-K // QKT) * QKT
    wp = np.zeros((N, KP), np.float32)
    wp[:, :K] = w
    g = wp.reshape(N, KP // 32, 32)
    scale = (np.abs(g).max(-1) / 127).astype(bfloat16).astype(np.float32)
    safe = np.where(scale > 0, scale, 1)
    codes = np.clip(np.rint(g / safe[..., None]), -127, 127).astype(np.int8).reshape(N, KP)
    nb, kb = N // MT, KP // QKT
    sc = scale.reshape(nb, MT, kb, QKT // 32).transpose(0, 2, 3, 1).astype(bfloat16)
    c = codes.reshape(nb, MT // 16, 16, kb, QKT).transpose(0, 3, 1, 4, 2)
    blk = np.zeros((nb, kb, BLOCK), np.uint8)
    blk[..., :320] = np.ascontiguousarray(sc).reshape(nb, kb, -1).view(np.uint8)
    blk[..., 320:5440] = np.ascontiguousarray(c).reshape(nb, kb, -1).view(np.uint8)
    b = np.zeros((nb, kb, MT), bfloat16)
    b[:, 0] = np.asarray(bias, np.float32).reshape(nb, MT)
    blk[..., 5440:] = b.view(np.uint8)
    per = nb // COLS // CORES
    blk = blk.reshape(COLS, per, CORES, kb, BLOCK).transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(blk.reshape(-1, BLOCK))


def q8x_dequant(t, N: int, K: int):
    """q8x_table's weights (N, K) float32 and bias (N,)."""
    KP = -(-K // QKT) * QKT
    nb, kb = N // MT, KP // QKT
    per = nb // COLS // CORES
    blk = np.asarray(t, np.uint8).reshape(COLS, per, kb, CORES, BLOCK).transpose(0, 1, 3, 2, 4).reshape(nb, kb, BLOCK)
    sc = blk[..., :320].copy().view(bfloat16).astype(np.float32).reshape(nb, kb, QKT // 32, MT)
    c = blk[..., 320:5440].copy().view(np.int8).reshape(nb, kb, MT // 16, QKT, 16)
    codes = c.transpose(0, 2, 4, 1, 3).reshape(N, KP).astype(np.float32)
    scale = sc.transpose(0, 3, 1, 2).reshape(N, KP // 32)
    bias = blk[:, 0, 5440:].copy().view(bfloat16).astype(np.float32).reshape(N)
    return (codes * np.repeat(scale, 32, 1))[:, :K], bias


def rms_norm(x, g, eps: float):
    """The cores' RMSNorm: bf16(x g / sqrt(mean(x^2) + eps))."""
    x = np.asarray(x, np.float64)
    return (x * np.asarray(g, np.float64) / np.sqrt(np.mean(x * x) + eps)).astype(bfloat16).astype(np.float64)


def rope_pairs(y, ang, rows: int):
    """y's first rows turned in interleaved pairs by ang (cos and sin interleaved, 64 of a head)."""
    t = np.asarray(ang, np.float64).reshape(-1)
    r = y[:rows].reshape(-1, 32, 2)
    a, b = r[..., 0].copy(), r[..., 1].copy()
    r[..., 0], r[..., 1] = a * t[0::2] - b * t[1::2], a * t[1::2] + b * t[0::2]
    y[:rows] = r.reshape(-1)
    return y


def unexpanded_loads() -> None:
    """Make IRON's full ELFs leave their loads unexpanded whatever the number of designs: the
    firmware then skips the load of the array it holds (steps differing in ``phases`` are designs
    of one array), and loads another operator's as a whole (about 1.2 ms: the few left outside
    the layers' array, once a token). IRON expands them where a sequence has several designs."""
    from iron.common.image import compose

    if getattr(compose.Composition, "_hwnpu_unexpanded", False):
        return
    init = compose.Composition.__init__

    def unexpanded(self, seq):
        init(self, seq)
        self.expands = False

    compose.Composition.__init__ = unexpanded
    compose.Composition._hwnpu_unexpanded = True


@functools.cache
def step_operator():
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

    from iron.common import Built, Operator, Rtp

    import iron_moeprobe

    IPAD, NPAD, KT = moe.IPAD, moe.NPAD, moe.KT
    K, KD = moe.HIDDEN, moe.INTER
    CB = CORES * MT
    XN = TOP * IPAD  # the experts' activations
    R = NPAD // COLS  # a column's out-features of the residual
    YE = XN + NPAD
    QKB, OKB = -(-HP // QKT), -(-ATT // QKT)  # k-blocks of q, k, v and of o_proj
    QPER, OPER = QN // MT // COLS // CORES, HP // MT // COLS // CORES  # out-blocks a core
    blk_t = np.ndarray[(BLOCK,), np.dtype[np.uint8]]
    half_t = np.ndarray[(MT,), np.dtype[bfloat16]]
    acc_t = np.ndarray[(MT,), np.dtype[np.float32]]
    x_t = np.ndarray[(XO,), np.dtype[bfloat16]]
    st_t = np.ndarray[(392,), np.dtype[np.float32]]  # gptoss_layer.cc's gl_state
    cnt_t = np.ndarray[(8,), np.dtype[np.int32]]  # the phase's counts, the core's column and place
    KINDS = {PH_E: (1, 2), PH_Q: (3,), PH_O: (4,)}  # gptoss_layer.cc's K_E1, K_E2, K_Q, K_O

    def kinds_of(phases: int) -> list:
        return [k for bit in (PH_E, PH_Q, PH_O) if phases & bit for k in KINDS[bit]]

    def kernels(eps: float, rope_rows: int):
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        flags = [f"-DMX_K={KD}", f"-DMX_KT={KT}", f"-DMX_XE={IPAD}", f"-DMX_W={XN + R}",
                 f"-DGL_EPS={float(eps)!r}f", f"-DGL_ROPE_ROWS={rope_rows}"]
        common = dict(source_file=str(here / "gptoss_layer.cc"),
                      include_dirs=[str(here), str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
                      compile_flags=flags,
                      object_file_name=f"gptoss_layer_{rope_rows}_{abs(hash(float(eps))) % 10**8}.o")
        f = lambda name, args: ExternalFunction(name, arg_types=args, **common)  # noqa: E731
        i32 = np.int32
        return dict(
            plan=f("gl_plan", [cnt_t, i32, i32]),
            begin=f("gl_begin", [x_t, st_t, cnt_t]),
            block=f("gl_block", [blk_t, x_t, st_t, cnt_t, i32, i32]),
            emit=f("gl_emit", [st_t, x_t, cnt_t, i32, i32, half_t]),
            tail=f("gl_tail", [st_t, cnt_t, i32, half_t]),
        )

    def offset(op, name):
        v = op.value(name)
        return v if v.per_call else None

    class GptOssStep(Operator):
        """One turn of the decode step's array: the phases ``phases`` names (module docstring)."""

        aiecc_flags: ClassVar[tuple[str, ...]] = ("--reclaim-runtime-bds",)

        grows: int = param(array=True)  # blocks in the experts' gate-up table
        drows: int = param(array=True)  # in their down table
        qrows: int = param(array=True)  # in q, k, v's q8x table
        orows: int = param(array=True)  # in o_proj's
        eps: float = param(array=True)
        rope_rows: int = param(default=4608, array=True)
        form: int = param(default=1, array=True)
        phases: int = param(default=PH_E)  # the step's phases: its sequence, not the array's
        picked: int = param(default=-1, repr=False)
        num_aie_columns: int = tunable(domain=widths)

        Wg = In(grows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        Wd = In(drows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        Wq = In(qrows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        Wo = In(orows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        h = In(HP, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # the residual stream
        nw = In(HP, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # the phase's norm's weight
        ang = In(64, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # Q: the call's angles
        att = In(ATT, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # O: the heads' outputs
        rw = In(WV, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # E: the routing weights
        ye = Out(YE, obj_shape=(CB,), per=(num_aie_columns,))  # E: the activations, then the layer's output
        yq = Out(QN, obj_shape=(CB,), per=(num_aie_columns,))  # Q: q, k, v
        yo = Out(HP, obj_shape=(CB,), per=(num_aie_columns,))  # O: the residual after the attention
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)
        d0 = per_call(derive=lambda op: 0)
        d1 = per_call(derive=lambda op: 0)
        d2 = per_call(derive=lambda op: 0)
        d3 = per_call(derive=lambda op: 0)
        # the step's phases for the cores: how many, and their kinds (4 bits each)
        n_ph = per_call(derive=lambda op: len(kinds_of(op.phases)))
        kinds = per_call(derive=lambda op: sum(k << (4 * i) for i, k in enumerate(kinds_of(op.phases))))

        def resolve(self, dev):
            return self.replace(num_aie_columns=COLS)

        def gu_bytes(self) -> int:
            return 2 * (IPAD // MT) * (K // KT) * BLOCK

        def dn_bytes(self) -> int:
            return (NPAD // MT) * (KD // KT) * BLOCK

        def build(self) -> Built:
            n = self.num_aie_columns
            k = kernels(self.eps, self.rope_rows)
            w_l3 = [ObjectFifo(self.Wg.obj_type, depth=2) for _ in range(n)]
            w_l1 = [f.cons().split([BLOCK * c for c in range(CORES)], obj_types=[blk_t] * CORES,
                                   depths=[2] * CORES, tile=AnyMemTile) for f in w_l3]
            x_l3 = [ObjectFifo(x_t, depth=1) for _ in range(n)]
            x_l1 = [f.cons().forward(obj_type=x_t, tile=AnyMemTile) for f in x_l3]
            y_l3 = [ObjectFifo(self.ye.obj_type, depth=2) for _ in range(n)]
            y_l1 = [f.prod().join([MT * c for c in range(CORES)], obj_types=[half_t] * CORES,
                                  depths=[2] * CORES, tile=AnyMemTile) for f in y_l3]

            def i32(v):
                return index.casts(T.i32(), v)

            def core(w_f, x_f, y_f, rtp, st, cnt, plan, begin, block, emit, tail):
                # one frame for every phase (gptoss_layer.cc): its counts are cnt's
                words = rtp.read()
                for ph in range_(words[self.n_ph]):
                    plan(cnt, i32(ph), words[self.kinds])
                    x = x_f.acquire(1)
                    begin(x, st, cnt)
                    for o in range_(cnt[1]):
                        for b in range_(cnt[2]):
                            blk = w_f.acquire(1)
                            block(blk, x, st, cnt, i32(o), i32(b))
                            w_f.release(1)
                        for q in range_(cnt[3]):
                            y = y_f.acquire(1)
                            emit(st, x, cnt, i32(o), i32(q), y)
                            y_f.release(1)
                    x_f.release(1)
                    for q in range_(cnt[5]):
                        y = y_f.acquire(1)
                        tail(st, cnt, i32(q), y)
                        y_f.release(1)

            workers, rtps = [], []
            for i in range(n):
                for j in range(CORES):
                    rtp = Rtp([self.n_ph, self.kinds])
                    rtps.append(rtp)
                    place = Buffer(initial_value=np.array([0, 0, 0, 0, 0, 0, i, j], np.int32))
                    workers.append(Worker(core, [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), rtp,
                                                 Buffer(st_t), place, k["plan"], k["begin"], k["block"], k["emit"],
                                                 k["tail"]]))
            for name in ("Wg", "Wd", "Wq", "Wo"):
                getattr(self, name).bind([f.prod() for f in w_l3])
            for name in ("h", "nw", "ang", "att", "rw"):
                getattr(self, name).bind([f.prod() for f in x_l3])
            for name in ("ye", "yq", "yo"):
                getattr(self, name).bind([f.cons() for f in y_l3])
            return Built(workers, rtps)

        # x's object by pieces of at most 1023 words a dimension
        def _h_twice(self, lane):
            self.h.fill(TAP((HP,), 0, [2, 2, HP // 2], [0, HP // 2, 1]), lane=lane, managed=False)

        def sequence(self, rt):
            if self.phases & PH_E:
                self._experts()
            if self.phases & PH_Q:
                self._qkv()
            if self.phases & PH_O:
                self._o()

        def _experts(self):
            n = self.num_aie_columns
            per = IPAD // MT // (2 * CORES)
            geb, deb = self.gu_bytes(), self.dn_bytes()
            run = deb // n
            gtotal, dtotal = self.grows * BLOCK, self.drows * BLOCK
            first = []
            for l in range(n):
                e, hf = l // 2, l % 2
                first.append(self.ye.drain(TAP((YE,), e * IPAD + hf * CB, [per, CB], [2 * CB, 1]), lane=l,
                                           wait=True, managed=False))
            for l in range(n):  # h, the norm's weight (at NW_AT), and h's start to the object's end
                self._h_twice(l)
                self.nw.fill(TAP((HP,), 0, [2, 2, HP // 2], [0, HP // 2, 1]), lane=l, managed=False)
                self.h.fill(TAP((HP,), 0, [XO - 4 * HP], [1]), lane=l, managed=False)
            for l in range(n):
                e, hf = l // 2, l % 2
                self.Wg.fill(TAP((gtotal,), hf * geb // 2, [geb // 2], [1]), lane=l,
                             offset_parameter=offset(self, f"e{e}"), managed=False)
            for l in range(n):
                for i in range(TOP):
                    self.Wd.fill(TAP((dtotal,), l * run, [run], [1]), lane=l, offset_parameter=offset(self, f"d{i}"),
                                 managed=False)
            second = [self.ye.drain(TAP((YE,), XN + l * R, [R], [1]), lane=l, wait=True, managed=False)
                      for l in range(n)]
            for d in first:
                d.await_()
            for l in range(n):  # the activations, the column's residual, the routing weights
                self.h.fill(TAP((YE,), 0, [XN], [1]), lane=l, source=self.ye, managed=False)
                self.h.fill(TAP((HP,), l * R, [R], [1]), lane=l, managed=False)
                self.rw.fill(TAP((WV,), 0, [(XO - XN - R) // WV, WV], [0, 1]), lane=l, managed=False)
            for d in second:
                d.await_()

        def _qkv(self):
            n = self.num_aie_columns
            rows = QN // n
            run = self.qrows * BLOCK // n
            drains = [self.yq.drain(TAP((QN,), l * rows, [rows], [1]), lane=l, wait=True, managed=False)
                      for l in range(n)]
            for l in range(n):  # h (twice: zeroed past it), the norm's weight, the angles
                self._h_twice(l)
                self.nw.fill(TAP((HP,), 0, [HP], [1]), lane=l, managed=False)
                self.ang.fill(TAP((64,), 0, [(XO - ANG_AT) // 64, 64], [0, 1]), lane=l, managed=False)
            for l in range(n):
                self.Wq.fill(TAP((self.qrows * BLOCK,), l * run, [run], [1]), lane=l, managed=False)
            for d in drains:
                d.await_()

        def _o(self):
            n = self.num_aie_columns
            run = self.orows * BLOCK // n
            drains = [self.yo.drain(TAP((HP,), l * R, [R], [1]), lane=l, wait=True, managed=False) for l in range(n)]
            for l in range(n):  # the heads' outputs (three times, to RES_AT), the residual's slice
                self.att.fill(TAP((ATT,), 0, [RES_AT // ATT, 4, ATT // 4], [0, ATT // 4, 1]), lane=l, managed=False)
                self.h.fill(TAP((HP,), l * R, [R], [1]), lane=l, managed=False)
                self.h.fill(TAP((HP,), l * R, [XO - RES_AT - R], [1]), lane=l, managed=False)
            for l in range(n):
                self.Wo.fill(TAP((self.orows * BLOCK,), l * run, [run], [1]), lane=l, managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * (2 * TOP * IPAD * K + TOP * NPAD * KD + QN * HP + HP * ATT)

        def reference(self, Wg, Wd, Wq, Wo, h, nw, ang, att, rw, *, e0=0, e1=0, e2=0, e3=0, d0=0, d1=0, d2=0, d3=0):
            ye, yq, yo = np.zeros(YE, bfloat16), np.zeros(QN, bfloat16), np.zeros(HP, bfloat16)
            eps = float(self.eps)
            hv = np.asarray(h, np.float64).reshape(-1)
            if self.phases & PH_E:
                hn = rms_norm(hv, nw, eps)[:K]
                fg, fd = np.asarray(Wg, np.uint8).reshape(-1), np.asarray(Wd, np.uint8).reshape(-1)
                ge, de = self.gu_bytes(), self.dn_bytes()
                acts, out = [], hv.copy()
                wv = np.asarray(rw, np.float64).reshape(-1)
                for i, (ga, da) in enumerate(zip(iron_moeprobe.picked_offsets(self, [e0, e1, e2, e3], ge),
                                                 iron_moeprobe.picked_offsets(self, [d0, d1, d2, d3], de))):
                    gw, uw, gb, ub = moe.unpack_gate_up(fg[int(ga) : int(ga) + ge], K)
                    a = np.zeros(IPAD, np.float64)
                    a[: gw.shape[0]] = moe.glu(gw @ hn + gb, uw @ hn + ub)
                    a = a.astype(bfloat16).astype(np.float64)
                    acts.append(a)
                    dw, db = moe.unpack_down(fd[int(da) : int(da) + de], KD)
                    out[: dw.shape[0]] += wv[i] * (dw @ a[:KD] + db)
                ye = np.concatenate([*acts, out]).astype(bfloat16)
            if self.phases & PH_Q:
                w, b = q8x_dequant(Wq, QN, HP)
                y = w.astype(np.float64) @ rms_norm(hv, nw, eps) + b
                yq = rope_pairs(y, ang, self.rope_rows).astype(bfloat16)
            if self.phases & PH_O:
                w, b = q8x_dequant(Wo, HP, ATT)
                yo = (w.astype(np.float64) @ np.asarray(att, np.float64).reshape(-1) + b + hv).astype(bfloat16)
            return ye, yq, yo

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.02, 1e-2, note="mx4 and q8 partial sums in two bf16 parts, exp_f32")

    return GptOssStep


def _check(a) -> int:
    import iron
    from iron import Scratchpad
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    if not a.expanded:
        unexpanded_loads()
    Step = step_operator()
    rng = np.random.default_rng(a.seed)
    E = a.experts
    gu = np.concatenate([moe.gate_up_table(*moe.random_expert(rng, 2 * moe.INTER, moe.HIDDEN)) for _ in range(E)])
    dn = np.concatenate([moe.down_table(*moe.random_expert(rng, moe.HIDDEN, moe.INTER)) for _ in range(E)])
    gu_eb, dn_eb = gu.size // E, dn.size // E
    wq = q8x_table(rng.standard_normal((QN, HP)).astype(np.float32) * 0.05, rng.standard_normal(QN) * 0.1)
    wo = q8x_table(rng.standard_normal((HP, ATT)).astype(np.float32) * 0.05, rng.standard_normal(HP) * 0.1)
    picked = [int(v) for v in rng.choice(E, TOP, replace=False)]
    eps = 1e-5

    class G(iron.Graph):
        def __init__(self):
            self.gu = iron.param(gu.reshape(-1, BLOCK))
            self.dn = iron.param(dn.reshape(-1, BLOCK))
            self.wq = iron.param(wq)
            self.wo = iron.param(wo)

        def step(self, phases, h, nw, ang, att, rw, es):
            return Step.step(self.gu, self.dn, self.wq, self.wo, h, nw, ang, att, rw, grows=self.gu.shape[0],
                             drows=self.dn.shape[0], qrows=self.wq.shape[0], orows=self.wo.shape[0], eps=eps,
                             phases=phases, **{f"e{i}": es[i] * gu_eb for i in range(TOP)},
                             **{f"d{i}": es[i] * dn_eb for i in range(TOP)})

        def body(self, h, nw, ang, att, rw, *, e0: Scratchpad[np.int32] = np.int32(0),
                 e1: Scratchpad[np.int32] = np.int32(0), e2: Scratchpad[np.int32] = np.int32(0),
                 e3: Scratchpad[np.int32] = np.int32(0)):
            es = [e0, e1, e2, e3]
            # the layer's order: q, k, v of h; o_proj of the heads' outputs (given) onto h; the experts
            _, yq, _ = self.step(PH_Q, h, nw, ang, att, rw, es)
            _, _, yo = self.step(PH_O, h, nw, ang, att, rw, es)
            if a.commands:
                iron.on_host(lambda: None)
            ye, _, _ = self.step(PH_E, yo, nw, ang, att, rw, es)
            return yq, ye

        def reference(self, h, nw, ang, att, rw, e0=0, e1=0, e2=0, e3=0):
            op = Step(grows=gu.size // BLOCK, drows=dn.size // BLOCK, qrows=wq.shape[0], orows=wo.shape[0], eps=eps,
                      phases=PH_Q | PH_O)
            es = dict(e0=e0 * gu_eb, e1=e1 * gu_eb, e2=e2 * gu_eb, e3=e3 * gu_eb,
                      d0=e0 * dn_eb, d1=e1 * dn_eb, d2=e2 * dn_eb, d3=e3 * dn_eb)
            _, yq, yo = op.reference(gu, dn, wq, wo, h, nw, ang, att, rw)
            ye, _, _ = op.replace(phases=PH_E).reference(gu, dn, wq, wo, yo, nw, ang, att, rw, **es)
            return yq, ye

    g = G()
    cg = g.compile(h=(HP,), nw=(HP,), ang=(64,), att=(ATT,), rw=(WV,))
    m = export_graph(cg, a.out, "gptoss_step", dict(experts=E))
    h = np.zeros(HP, np.float32)
    h[: moe.HIDDEN] = rng.standard_normal(moe.HIDDEN)
    nw = np.zeros(HP, np.float32)
    nw[: moe.HIDDEN] = 1 + 0.3 * rng.standard_normal(moe.HIDDEN)
    th = rng.uniform(-np.pi, np.pi, 32)
    ang = np.empty(64, np.float32)
    ang[0::2], ang[1::2] = np.cos(th), np.sin(th)
    rw = np.zeros(WV, np.float32)
    rw[:TOP] = rng.dirichlet(np.ones(TOP))
    inputs = dict(h=h.astype(bfloat16), nw=nw.astype(bfloat16), ang=ang.astype(bfloat16),
                  att=rng.standard_normal(ATT).astype(bfloat16), rw=rw.astype(bfloat16))
    write_check(g, m, a.out, inputs, {f"e{i}": picked[i] for i in range(TOP)})
    print(f"{a.out}: GptOssStep Q, O, E; {E} experts, picked {picked}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of the phases on random weights, and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--experts", type=int, default=6)
    c.add_argument("--seed", type=int, default=5)
    c.add_argument("--commands", action="store_true", help="the experts in a command of their own")
    c.add_argument("--expanded", action="store_true", help="IRON's expanded loads (the array loaded each command)")
    a = ap.parse_args(argv)
    return _check(a)


if __name__ == "__main__":
    sys.exit(main())
