"""gpt-oss's decode step as the phases of one array (GptOssStep), configured once.

An operator of its own is loaded again every time it follows another one: about 45 us and 1.2 us
a KB of its cores' programs (design.md 16.178), about 25 of a token's 79 ms. A full ELF of one
array is loaded once: the firmware skips the load of what it holds (iron_gptoss_moe.py's check
with MX4_SINGLE: the experts alone 1.15 ms a command where they follow another operator at 1.78).
So a layer's operators become phases of one core program on one array, the experts' (4 cores a
column, a weight stream split between them, x's stream to all four, their outputs joined), in
the order a command runs them:

  E  the experts (iron_gptoss_moe's MX4Experts) of the layer before, with its post-attention
     RMSNorm done by each core on its x (the residual h and the norm's weight in x's object)
  Q  the q, k, v projection with the input RMSNorm, the bias and RoPE (iron_q4's QGEMVQKV); k
     and v go into the caches' rows of the call's position
  A  the attention over the cache (one array: a row its 8 groups' keys, then their values;
     iron_window's position-indexed rows), a column a key-value group: objects of 128 rows' key
     and value of the group on x's stream, with the heads' sinks
  O  o_proj with its bias and the residual
  R  the router's logits on RMSNorm(h) (column 0)
  T  the routing: the top 4, their weights, and the next command's scratchpad words (the
     experts' offsets: the experts are the next command's first phase)

A step of the graph is one turn of the cores' loop: its ``phases`` (not a parameter of the array:
steps differing in it share the array and run their own sequences) set which phases run, read by
the cores as runtime words. Every step declares every operand (the array's lanes are bound once);
a step passes anything of the right shape for those its phases do not use.

The projections' weights are q8x: iron_q4's q8 codes and scales (the same values as the prompt
chunk's table) in blocks of 32 out-features by 160 in-features with the block's bias, 5504 bytes
as an mx4 block, in the order the cores take them ([column][out-block][k-block][core]). o_proj's
in-features are in the order the attention's outputs leave the cores (att_order). The router's
are bf16 blocks of 32 by 80.

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
D, G, H = 64, 8, 64  # the head dimension, the key-value groups, the heads
TOK = 128  # the attention: tokens an object (a key and a value each); a descriptor runs 64 times: 8192 rows
XO = TOK * 2 * D  # x's object: 16384 (the experts' need 12688)
ROW = 2 * G * D  # a cache row: the groups' keys, then their values
NW_AT, ANG_AT, RES_AT, SINK_AT = 6144, 9216, 12288, 13312  # kernels/gptoss_layer.cc's GL_NW, GL_ANG, GL_RES, GL_SINK
QKT = 160  # q8x's in-features a block
RKT, RKB = 80, 80  # the router's in-features a block; its blocks: 20 objects of 4, the last two cores' real
ATT = H * D  # the heads' outputs
ATTB = 2 * ATT  # as the attention's cores write them: a column's 512, then as much unused
QN = 5120  # q, k and v's out-features
SLOTS = 32  # a command's scratchpad words
FED = 8  # of them, the words the routing writes (the experts' offsets)
FED_AT = 3  # their first word in a command of every phase: after the tokens read, a sliding layer's, the row
PROG = 256  # the routing's program as bf16 (4 int32 a word)
PH_E, PH_Q, PH_A, PH_O, PH_R, PH_T = 1, 2, 4, 8, 16, 32
KINDS = {PH_E: (1, 2), PH_Q: (3,), PH_A: (4,), PH_O: (5,), PH_R: (6,), PH_T: (7,)}  # gptoss_layer.cc's


def att_order():
    """The heads' outputs' place in o_proj's x: value d of head h (ATT) at att_order()[h * 64 + d].
    Column g's cores 0 and 1 hold heads 8 g + 4 c + j and send each in two objects of 32."""
    at = np.empty(ATT, np.int64)
    for h in range(H):
        g, c, j = h // 8, (h % 8) // 4, h % 4
        for d in range(D):
            at[h * D + d] = g * 512 + (2 * j + d // 32) * 64 + c * 32 + d % 32
    return at


def router_table(w, bias):
    """The router's (32, K) weights and bias as bf16 blocks (uint8 (RKB, BLOCK)): k-block kb (in-features
    80 kb .., [in-feature][out-feature]; the bias in k-block 0's) is the block of object kb // 2 for
    core 2 + kb % 2 (the cores with the router's kernel); the first two cores' blocks are nothing."""
    w = np.asarray(w, np.float32)
    K = w.shape[1]
    nk = RKB // 2
    wp = np.zeros((MT, nk * RKT), np.float32)
    wp[: w.shape[0], :K] = w
    real = np.zeros((nk, BLOCK), np.uint8)
    real[:, : RKT * MT * 2] = np.ascontiguousarray(wp.reshape(MT, nk, RKT).transpose(1, 2, 0)).astype(bfloat16).reshape(nk, -1).view(np.uint8)
    b = np.zeros(MT, np.float32)
    b[: len(bias)] = np.asarray(bias, np.float32)
    real[0, BLOCK - 2 * MT :] = b.astype(bfloat16).view(np.uint8)
    blk = np.zeros((RKB // CORES, CORES, BLOCK), np.uint8)
    blk[:, 2:] = real.reshape(-1, 2, BLOCK)
    return blk.reshape(RKB, BLOCK)


def router_dequant(t, K: int):
    blk = np.asarray(t, np.uint8).reshape(RKB // CORES, CORES, BLOCK)[:, 2:].reshape(RKB // 2, BLOCK)
    w = blk[:, : RKT * MT * 2].copy().view(bfloat16).astype(np.float32).reshape(RKB // 2, RKT, MT).transpose(2, 0, 1).reshape(MT, -1)
    return w[:, :K], blk[0, BLOCK - 2 * MT :].copy().view(bfloat16).astype(np.float32)


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
        if any(type(d.op).__name__ == "GptOssStep" for d in self.designs.values()):
            self.expands = False

    compose.Composition.__init__ = unexpanded
    compose.Composition._hwnpu_unexpanded = True




def kinds_of(phases: int) -> list:
    return [k for bit in (PH_E, PH_Q, PH_A, PH_O, PH_R, PH_T) if phases & bit for k in KINDS[bit]]


def objects(position):
    """The attention's objects of position + 1 tokens, as a graph expression of a per-call position."""
    return (position + TOK) // TOK


@functools.cache
def step_operator():
    import aie.dialects.index as index
    from aie.dialects.aie import T
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import Buffer, In, InOut, ObjectFifo, Out, Worker
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
    blk_t = np.ndarray[(BLOCK,), np.dtype[np.uint8]]
    half_t = np.ndarray[(MT,), np.dtype[bfloat16]]
    x_t = np.ndarray[(XO,), np.dtype[bfloat16]]
    st_t = np.ndarray[(1024,), np.dtype[np.float32]]  # gptoss_layer.cc's gl_state (and room)
    cnt_t = np.ndarray[(12,), np.dtype[np.int32]]  # the phase's counts, the core's column and place

    def kernels(eps: float, rope_rows: int, lane: int, core: int):
        """The core's kernels: the attention on cores 0 and 1, the router on column 0, the routing
        on its last core (a core's program holds what it runs: 16 KB)."""
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        parts = (["ATT"] if core < 2 else []) + (["ROUTER"] if lane == 0 and core >= 2 else []) + (
            ["ROUTE"] if (lane, core) == (0, CORES - 1) else [])
        flags = [f"-DMX_K={KD}", f"-DMX_KT={KT}", f"-DMX_XE={IPAD}", f"-DMX_W={XN + R}",
                 f"-DGL_EPS={float(eps)!r}f", f"-DGL_ROPE_ROWS={rope_rows}", *(f"-DGL_{p}=1" for p in parts)]
        tag = "_".join(p.lower() for p in parts) or "base"
        flags.append(f"-DGL_TAG={tag}")
        common = dict(source_file=str(here / "gptoss_layer.cc"),
                      include_dirs=[str(here), str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
                      compile_flags=flags,
                      object_file_name=f"gptoss_layer_{tag}_{rope_rows}_{abs(hash(float(eps))) % 10**8}.o")
        f = lambda name, args: ExternalFunction(f"{tag}_{name}", arg_types=args, **common)  # noqa: E731
        i32 = np.int32
        return [f("gl_plan", [cnt_t, i32, i32, i32, i32]), f("gl_begin", [x_t, st_t, cnt_t]),
                f("gl_block", [blk_t, x_t, st_t, cnt_t, i32, i32]), f("gl_emit", [st_t, x_t, cnt_t, i32, i32, half_t]),
                f("gl_xblock", [x_t, st_t, cnt_t, i32]), f("gl_tail", [st_t, cnt_t, i32, half_t])]

    def bound(op, name):
        v = op.value(name)
        return v if v.per_call else None

    class GptOssStep(Operator):
        """One turn of the decode step's array: the phases ``phases`` names (module docstring)."""

        aiecc_flags: ClassVar[tuple[str, ...]] = ("--reclaim-runtime-bds",)

        grows: int = param(array=True)  # blocks in the experts' gate-up table
        drows: int = param(array=True)  # in their down table
        qrows: int = param(array=True)  # in q, k, v's q8x table
        orows: int = param(array=True)  # in o_proj's
        kvn: int = param(array=True)  # the cache's elements (rows of ROW)
        eps: float = param(array=True)
        rope_rows: int = param(default=4608, array=True)
        form: int = param(default=1, array=True)
        phases: int = param(default=PH_E)  # the step's phases: its sequence, not the array's
        sliding: bool = param(default=False)  # A: the layer's attention reads its window's keys
        picked: int = param(default=-1, repr=False)  # E: the reference's experts (iron_moeprobe.PICKED's key)
        routed: int = param(default=-1, repr=False)  # T: the key the reference's routing is kept under
        fed_at: int = param(default=FED_AT)  # T: the first of the next command's words the routing writes
        num_aie_columns: int = tunable(domain=widths)

        Wg = In(grows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        Wd = In(drows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        Wq = In(qrows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        Wo = In(orows, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        Wr = In(RKB, BLOCK, dtype=np.uint8, obj_shape=(CORES, BLOCK), per=(num_aie_columns,))
        h = In(HP, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # the residual stream into the step
        nwe = In(HP, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # E: the experts' norm's weight
        nwq = In(HP, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # Q: the input norm's
        nwr = In(HP, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # R: the router's norm's
        ang = In(D, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # Q: the call's angles
        sinks = In(H, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # A: the heads' sinks
        rw = In(WV, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # E: the routing weights
        prog = In(PROG, obj_shape=(XO,), per=(num_aie_columns,), replicate=True)  # T: the routing's program
        KV = InOut(kvn, obj_shape=(CB,), per=(num_aie_columns,))  # the layer's cache (Q writes the call's row, A reads)
        ye = Out(YE, obj_shape=(CB,), per=(num_aie_columns,))  # E: the activations, then the layer's output
        yq = Out(QN, obj_shape=(CB,), per=(num_aie_columns,))  # Q: q (k and v where the caches are not written)
        ya = Out(ATTB, obj_shape=(CB,), per=(num_aie_columns,))  # A: the heads' outputs (att_order)
        yo = Out(HP, obj_shape=(CB,), per=(num_aie_columns,))  # O: the residual after the attention
        yr = Out(PROG, obj_shape=(CB,), per=(num_aie_columns,))  # R: the logits' partial sums (float32)
        yw = Out(WV, obj_shape=(CB,), per=(num_aie_columns,))  # T: the routing weights
        image = Out(2 * SLOTS, obj_shape=(CB,), per=(num_aie_columns,))  # T: the next command's scratchpad
        yj = Out(PROG, obj_shape=(CB,), per=(num_aie_columns,))  # what the other cores send with them
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)
        d0 = per_call(derive=lambda op: 0)
        d1 = per_call(derive=lambda op: 0)
        d2 = per_call(derive=lambda op: 0)
        d3 = per_call(derive=lambda op: 0)
        kw = per_call(derive=lambda op: 0)  # Q: the caches' element of the call's row
        kfirst = per_call(derive=lambda op: 0)  # A: the element of the first row read
        nkv1 = per_call(derive=lambda op: 0)  # A: the objects read, less one (their descriptor's extra runs)
        # the cores': the tokens a full layer's attention reads, and a sliding one's (every step
        # binds both: the steps' cores read the same words)
        ntok = per_call(derive=lambda op: 1)
        nslide = per_call(derive=lambda op: 1)
        # the step's phases for the cores: how many, and their kinds (4 bits each, bit 28 sliding)
        n_ph = per_call(derive=lambda op: len(kinds_of(op.phases)))
        kinds = per_call(derive=lambda op: sum(k << (4 * i) for i, k in enumerate(kinds_of(op.phases)))
                         | (int(bool(op.sliding)) << 28))

        def resolve(self, dev):
            return self.replace(num_aie_columns=COLS)

        def gu_bytes(self) -> int:
            return 2 * (IPAD // MT) * (K // KT) * BLOCK

        def dn_bytes(self) -> int:
            return (NPAD // MT) * (KD // KT) * BLOCK

        def build(self) -> Built:
            n = self.num_aie_columns
            w_l3 = [ObjectFifo(self.Wg.obj_type, depth=2) for _ in range(n)]
            w_l1 = [f.cons().split([BLOCK * c for c in range(CORES)], obj_types=[blk_t] * CORES,
                                   depths=[2] * CORES, tile=AnyMemTile) for f in w_l3]
            x_l3 = [ObjectFifo(x_t, depth=1) for _ in range(n)]
            x_l1 = [f.cons().forward(obj_type=x_t, tile=AnyMemTile) for f in x_l3]
            y_l3 = [ObjectFifo(self.ye.obj_type, depth=2) for _ in range(n)]
            # an object out a core (the memtile holds the next): a second costs the cores' programs
            y_l1 = [f.prod().join([MT * c for c in range(CORES)], obj_types=[half_t] * CORES,
                                  depths=[1] * CORES, tile=AnyMemTile) for f in y_l3]

            def i32(v):
                return index.casts(T.i32(), v)

            def core(w_f, x_f, y_f, rtp, st, cnt, plan, begin, block, emit, xblock, tail):
                # one frame for every phase (gptoss_layer.cc): its counts are cnt's
                words = rtp.read()
                for ph in range_(words[self.n_ph]):
                    plan(cnt, i32(ph), words[self.kinds], words[self.ntok], words[self.nslide])
                    for _ in range_(cnt[8]):
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
                        for b in range_(cnt[4]):
                            xb = x_f.acquire(1)
                            xblock(xb, st, cnt, i32(b))
                            x_f.release(1)
                        for q in range_(cnt[5]):
                            y = y_f.acquire(1)
                            tail(st, cnt, i32(q), y)
                            y_f.release(1)

            workers, rtps = [], []
            for i in range(n):
                for j in range(CORES):
                    rtp = Rtp([self.n_ph, self.kinds, self.ntok, self.nslide])
                    rtps.append(rtp)
                    place = Buffer(initial_value=np.array([0, 0, 0, 0, 0, 0, i, j, 0, 0, 0, 0], np.int32))
                    workers.append(Worker(core, [w_l1[i][j].cons(), x_l1[i].cons(), y_l1[i][j].prod(), rtp,
                                                 Buffer(st_t), place, *kernels(self.eps, self.rope_rows, i, j)]))
            for name in ("Wg", "Wd", "Wq", "Wo", "Wr"):
                getattr(self, name).bind([f.prod() for f in w_l3])
            for name in ("h", "nwe", "nwq", "nwr", "ang", "sinks", "rw", "prog"):
                getattr(self, name).bind([f.prod() for f in x_l3])
            for name in ("KV", "ye", "yq", "ya", "yo", "yr", "yw", "image", "yj"):
                getattr(self, name).bind([f.cons() for f in y_l3])
            return Built(workers, rtps)

        # --- the sequence: each phase's fills and drains, in the cores' order ---

        def sequence(self, rt):
            ph = self.phases
            if ph & PH_E:
                self._experts()
            if ph & PH_Q:
                self._qkv()
            if ph & PH_A:
                self._att()
            if ph & PH_O:
                self._o()
            if ph & PH_R:
                self._router()
            if ph & PH_T:
                self._route()

        def _twice(self, operand, lane, source=None, at=0, size=HP):
            """x's object's next 2 HP: a row twice, by pieces of at most 1023 words a dimension."""
            operand.fill(TAP((size,), at, [2, 2, HP // 2], [0, HP // 2, 1]), lane=lane, source=source, managed=False)

        def _stream(self, lane, off, sizes, strides):
            """A fill of x's lane from the residual stream of the layer's Q and O: the experts'
            output where the step ran them, else the step's h."""
            if self.phases & PH_E:
                self.h.fill(TAP((YE,), XN + off, sizes, strides), lane=lane, source=self.ye, managed=False)
            else:
                self.h.fill(TAP((HP,), off, sizes, strides), lane=lane, managed=False)

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
                self._twice(self.h, l)
                self._twice(self.nwe, l)
                self.h.fill(TAP((HP,), 0, [(XO - 4 * HP) // 1024, 1024], [0, 1]), lane=l, managed=False)
            for l in range(n):
                e, hf = l // 2, l % 2
                self.Wg.fill(TAP((gtotal,), hf * geb // 2, [geb // 2], [1]), lane=l,
                             offset_parameter=bound(self, f"e{e}"), managed=False)
            for l in range(n):
                for i in range(TOP):
                    self.Wd.fill(TAP((dtotal,), l * run, [run], [1]), lane=l, offset_parameter=bound(self, f"d{i}"),
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
            kw = bound(self, "kw")
            drains = []

            def drain(operand, size, at, count, lane, offset=None):
                drains.append(operand.drain(TAP((size,), at, [count], [1]), lane=lane, wait=True, managed=False,
                                            offset_parameter=offset))

            GD = G * D
            for l in range(n):  # a column's 640 rows: q's 4096, then k's 512 and v's 512
                lo, hi = l * rows, (l + 1) * rows
                if kw is None or hi <= ATT:  # the caches not written (a check): all into yq
                    drain(self.yq, QN, lo, rows, l)
                    continue
                if lo < ATT:
                    drain(self.yq, QN, lo, ATT - lo, l)
                # k's rows and v's are the cache row's two halves, in order
                drain(self.KV, self.kvn, max(lo, ATT) - ATT, hi - max(lo, ATT), l, kw)
            for l in range(n):  # h (twice: zeroed past it), the norm's weight, the angles
                self._stream(l, 0, [2, 2, HP // 2], [0, HP // 2, 1])
                self.nwq.fill(TAP((HP,), 0, [HP], [1]), lane=l, managed=False)
                self.ang.fill(TAP((D,), 0, [(XO - ANG_AT) // D, D], [0, 1]), lane=l, managed=False)
            for l in range(n):
                self.Wq.fill(TAP((self.qrows * BLOCK,), l * run, [run], [1]), lane=l, managed=False)
            for d in drains:
                d.await_()

        def _att(self):
            n = self.num_aie_columns
            nmax = min(self.kvn // ROW // TOK, 64)  # objects in the cache, a descriptor's runs at most
            first = bound(self, "kfirst")
            drains = [self.ya.drain(TAP((ATTB,), l * 1024, [8, 2, 64], [64, 512, 1]), lane=l, wait=True, managed=False)
                      for l in range(n)]
            for l in range(n):  # the column's heads' q (to SINK_AT), their sinks
                self.h.fill(TAP((QN,), l * 512, [SINK_AT // 512, 512], [0, 1]), lane=l, source=self.yq, managed=False)
                self.sinks.fill(TAP((H,), l * 8, [(XO - SINK_AT) // 8, 8], [0, 1]), lane=l, managed=False)
            for l in range(n):
                # an object of x: 128 rows' key and value of the group
                self.h.fill(TAP((self.kvn,), l * D, [nmax, TOK, 2, D], [TOK * ROW, ROW, G * D, 1]), lane=l,
                            source=self.KV, offset_parameter=first, repeat_parameter=bound(self, "nkv1"),
                            managed=False)
            for d in drains:
                d.await_()

        def _o(self):
            n = self.num_aie_columns
            run = self.orows * BLOCK // n
            drains = [self.yo.drain(TAP((HP,), l * R, [R], [1]), lane=l, wait=True, managed=False) for l in range(n)]
            for l in range(n):  # the heads' outputs (each column's 512, three times: to RES_AT), the residual
                self.h.fill(TAP((ATTB,), 0, [RES_AT // ATT, COLS, 512], [0, 1024, 1]), lane=l, source=self.ya,
                            managed=False)
                self._stream(l, l * R, [R], [1])
                self.h.fill(TAP((ATTB,), 0, [XO - RES_AT - R], [1]), lane=l, source=self.ya, managed=False)
            for l in range(n):
                self.Wo.fill(TAP((self.orows * BLOCK,), l * run, [run], [1]), lane=l, managed=False)
            for d in drains:
                d.await_()

        def _router(self):  # column 0: RMSNorm of the step's yo, the router's blocks
            d = self.yr.drain(TAP((PROG,), 0, [2 * CB], [1]), lane=0, wait=True, managed=False)
            self._twice(self.h, 0, source=self.yo)
            self._twice(self.nwr, 0)
            self.h.fill(TAP((HP,), 0, [(XO - 4 * HP) // 1024, 1024], [0, 1]), lane=0, source=self.yo, managed=False)
            self.Wr.fill(TAP((RKB * BLOCK,), 0, [RKB * BLOCK], [1]), lane=0, managed=False)
            d.await_()

        def _route(self):  # column 0: its last core's two objects, the words then the weights
            lead = (CORES - 1) * MT  # the other cores' part of an object
            ds = [self.yj.drain(TAP((PROG,), 0, [lead], [1]), lane=0, wait=True, managed=False),
                  self.image.drain(TAP((2 * SLOTS,), 2 * self.fed_at, [2 * FED], [1]), lane=0, wait=True, managed=False),
                  self.yj.drain(TAP((PROG,), lead, [MT - 2 * FED + lead], [1]), lane=0, wait=True, managed=False),
                  self.yw.drain(TAP((WV,), 0, [WV], [1]), lane=0, wait=True, managed=False),
                  self.yj.drain(TAP((PROG,), 2 * lead + MT - 2 * FED, [MT - WV], [1]), lane=0, wait=True, managed=False)]
            self.h.fill(TAP((PROG,), 0, [2 * CB], [1]), lane=0, source=self.yr, managed=False)
            self.prog.fill(TAP((PROG,), 0, [(XO - 2 * CB) // PROG, PROG], [0, 1]), lane=0, managed=False)
            for d in ds:
                d.await_()

        def ops(self) -> int:
            return 2 * (2 * TOP * IPAD * K + TOP * NPAD * KD + QN * HP + HP * ATT)

        # --- the reference ---

        def reference(self, Wg, Wd, Wq, Wo, Wr, h, nwe, nwq, nwr, ang, sinks, rw, prog, KV, *, e0=0, e1=0, e2=0,
                      e3=0, d0=0, d1=0, d2=0, d3=0, kw=None, kfirst=0, nkv1=0, ntok=1, nslide=1):
            ph, eps = self.phases, float(self.eps)
            ye, yq, ya, yo = np.zeros(YE, bfloat16), np.zeros(QN, bfloat16), np.zeros(ATTB, bfloat16), np.zeros(HP, bfloat16)
            yr, yw, image, yj = np.zeros(PROG, bfloat16), np.zeros(WV, bfloat16), np.zeros(2 * SLOTS, bfloat16), np.zeros(PROG, bfloat16)
            cache = np.array(KV, bfloat16).reshape(-1)
            hv = np.asarray(h, np.float64).reshape(-1)
            if ph & PH_E:
                hn = rms_norm(hv, nwe, eps)[:K_]
                fg, fd = np.asarray(Wg, np.uint8).reshape(-1), np.asarray(Wd, np.uint8).reshape(-1)
                ge, de = self.gu_bytes(), self.dn_bytes()
                acts, out = [], hv.copy()
                wv = np.asarray(rw, np.float64).reshape(-1)
                for i, (ga, da) in enumerate(zip(iron_moeprobe.picked_offsets(self, [e0, e1, e2, e3], ge),
                                                 iron_moeprobe.picked_offsets(self, [d0, d1, d2, d3], de))):
                    gw, uw, gb, ub = moe.unpack_gate_up(fg[int(ga) : int(ga) + ge], K_)
                    a = np.zeros(IPAD, np.float64)
                    a[: gw.shape[0]] = moe.glu(gw @ hn + gb, uw @ hn + ub)
                    a = a.astype(bfloat16).astype(np.float64)
                    acts.append(a)
                    dw, db = moe.unpack_down(fd[int(da) : int(da) + de], KD)
                    out[: dw.shape[0]] += wv[i] * (dw @ a[:KD] + db)
                ye = np.concatenate([*acts, out]).astype(bfloat16)
                hv = ye[XN:].astype(np.float64)  # the layer's Q and O read the experts' output
            if ph & PH_Q:
                w, b = q8x_dequant(Wq, QN, HP)
                y = rope_pairs(w.astype(np.float64) @ rms_norm(hv, nwq, eps) + b, ang, self.rope_rows).astype(bfloat16)
                if kw is None:
                    yq = y
                else:
                    yq[:ATT] = y[:ATT]
                    cache[int(kw) : int(kw) + ROW] = y[ATT:]
            if ph & PH_A:
                n = int(nslide if self.sliding else ntok)
                rows = cache.astype(np.float64)[int(kfirst) :].reshape(-1, 2, G, D)[:n]
                keys, vals = rows[:, 0], rows[:, 1]
                q = (yq[:ATT].astype(np.float32) * np.float32(0.125)).astype(bfloat16).astype(np.float64).reshape(H, D)
                sk = np.asarray(sinks, np.float64).reshape(-1)
                o = np.zeros((H, D))
                for hd in range(H):
                    s = np.append(keys[:, hd // 8] @ q[hd], sk[hd])
                    p = np.exp(s - s.max())
                    o[hd] = (p[:-1] / p.sum()) @ vals[:, hd // 8]
                real = np.zeros(ATT)
                real[att_order()] = o.reshape(-1)
                ya = np.concatenate([np.concatenate([real[g * 512 : (g + 1) * 512], np.zeros(512)]) for g in range(G)]).astype(bfloat16)
            if ph & PH_O:
                w, b = q8x_dequant(Wo, HP, ATT)
                x = ya.astype(np.float64).reshape(G, 1024)[:, :512].reshape(-1)
                yo = (w.astype(np.float64) @ x + b + hv).astype(bfloat16)
            if ph & PH_T:
                w, b = router_dequant(Wr, HP)
                logit = (w.astype(np.float64) @ rms_norm(yo.astype(np.float64), nwr, eps) + b).astype(bfloat16).astype(np.float32)
                ids = np.argsort(-logit, kind="stable")[:TOP]
                e = np.exp(logit[ids] - logit[ids[0]])
                yw[:TOP] = (e / e.sum()).astype(bfloat16)
                p = np.asarray(prog).view(np.int32).reshape(-1, 4)[:FED].astype(np.int64)
                words = np.where(p[:, 0] < 0, p[:, 2], ids[np.clip(p[:, 0], 0, TOP - 1)] * p[:, 1] + p[:, 2]) << p[:, 3]
                image[2 * self.fed_at : 2 * (self.fed_at + FED)] = words.astype(np.int32).view(bfloat16)
                if self.routed >= 0:
                    iron_moeprobe.PICKED[self.routed] = [int(i) for i in ids]
            return cache, ye, yq, ya, yo, yr, yw, image, yj

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.02, 1e-2, note="mx4 and q8 partial sums in two bf16 parts, exp_f32")

    K_ = K
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
    wr = router_table(rng.standard_normal((MT, HP)).astype(np.float32) * 0.05, rng.standard_normal(MT) * 0.1)
    picked = [int(v) for v in rng.choice(E, TOP, replace=False)]
    eps, rows, pos = 1e-5, a.rows, a.position

    class Gr(iron.Graph):
        def __init__(self):
            self.gu = iron.param(gu.reshape(-1, BLOCK))
            self.dn = iron.param(dn.reshape(-1, BLOCK))
            self.wq, self.wo, self.wr = iron.param(wq), iron.param(wo), iron.param(wr)
            self.KV = iron.state((rows * ROW,), bfloat16, name="kv")

        def step(self, phases, h, nw, ang, sinks, rw, prog, es, position):
            n1 = position + 1
            n = iron.clamp(objects(position), hi=1) if a.sliding else objects(position)
            slide = dict(nkv1=n - 1, **(dict(kfirst=iron.clamp(position * ROW - 127 * ROW, lo=0)) if a.sliding else {}))
            return Step.step(self.gu, self.dn, self.wq, self.wo, self.wr, h, nw, nw, nw, ang, sinks, rw, prog,
                             self.KV, grows=self.gu.shape[0], drows=self.dn.shape[0], qrows=self.wq.shape[0],
                             orows=self.wo.shape[0], kvn=rows * ROW, eps=eps, phases=phases, sliding=a.sliding,
                             kw=position * ROW, ntok=n1, nslide=iron.clamp(n1, hi=128), **slide,
                             **{f"e{i}": es[i] * gu_eb for i in range(TOP)},
                             **{f"d{i}": es[i] * dn_eb for i in range(TOP)})

        def body(self, h, nw, ang, sinks, rw, prog, kv0=None, *, position: Scratchpad[np.int32] = np.int32(0),
                 e0: Scratchpad[np.int32] = np.int32(0), e1: Scratchpad[np.int32] = np.int32(0),
                 e2: Scratchpad[np.int32] = np.int32(0), e3: Scratchpad[np.int32] = np.int32(0)):
            es = [e0, e1, e2, e3]
            if kv0 is not None:  # the cache's rows before the call's (a copy: another operator, loaded as a whole)
                self.KV[:] = kv0
            _, _, _, ya, yo, _, yw, image, _ = self.step(PH_Q | PH_A | PH_O | PH_R | PH_T, h, nw, ang, sinks, rw,
                                                            prog, es, position)
            iron.on_host(lambda: None)
            _, ye, *_ = self.step(PH_E, yo, nw, ang, sinks, rw, prog, es, position)
            return ya, yw, image, ye

        def reference(self, h, nw, ang, sinks, rw, prog, kv0=None, position=0, e0=0, e1=0, e2=0, e3=0):
            kw = dict(grows=gu.size // BLOCK, drows=dn.size // BLOCK, qrows=wq.shape[0], orows=wo.shape[0],
                      kvn=rows * ROW, eps=eps, sliding=a.sliding)
            first = max(position - 127, 0) if a.sliding else 0
            n1 = position + 1
            r = Step(phases=PH_Q | PH_A | PH_O | PH_R | PH_T, **kw).reference(
                gu, dn, wq, wo, wr, h, nw, nw, nw, ang, sinks, rw, prog,
                np.zeros(rows * ROW, bfloat16) if kv0 is None else kv0,
                kw=position * ROW, kfirst=first * ROW, ntok=n1, nslide=min(n1, 128))
            es = dict(e0=e0 * gu_eb, e1=e1 * gu_eb, e2=e2 * gu_eb, e3=e3 * gu_eb,
                      d0=e0 * dn_eb, d1=e1 * dn_eb, d2=e2 * dn_eb, d3=e3 * dn_eb)
            ye = Step(phases=PH_E, **kw).reference(gu, dn, wq, wo, wr, r[4], nw, nw, nw, ang, sinks, rw, prog,
                                                   r[0], **es)[1]
            return r[3], r[6], r[7], ye

    g = Gr()
    caches = {} if a.empty else dict(kv0=(rows * ROW,))
    cg = g.compile(h=(HP,), nw=(HP,), ang=(D,), sinks=(H,), rw=(WV,), prog=(PROG,), **caches)
    m = export_graph(cg, a.out, "gptoss_step", dict(experts=E))
    h = np.zeros(HP, np.float32)
    h[: moe.HIDDEN] = rng.standard_normal(moe.HIDDEN)
    nw = np.zeros(HP, np.float32)
    nw[: moe.HIDDEN] = 1 + 0.3 * rng.standard_normal(moe.HIDDEN)
    th = rng.uniform(-np.pi, np.pi, 32)
    ang = np.empty(D, np.float32)
    ang[0::2], ang[1::2] = np.cos(th), np.sin(th)
    rw = np.zeros(WV, np.float32)
    rw[:TOP] = rng.dirichlet(np.ones(TOP))
    program = np.zeros((PROG // 8, 4), np.int32)  # word s: the s-th chosen expert's index, twice
    program[:, 0] = -1
    program[:FED] = [(i % TOP, 1 + i // TOP, i, 0) for i in range(FED)]
    kv0 = np.zeros((rows, 2, G, D), np.float32)
    kv0[:pos, 0], kv0[:pos, 1] = rng.standard_normal((pos, G, D)) * 0.5, rng.standard_normal((pos, G, D))
    inputs = dict(h=h.astype(bfloat16), nw=nw.astype(bfloat16), ang=ang.astype(bfloat16),
                  sinks=rng.uniform(-2, 4, H).astype(bfloat16), rw=rw.astype(bfloat16),
                  prog=program.reshape(-1).view(bfloat16))
    if not a.empty:
        inputs.update(kv0=kv0.reshape(-1).astype(bfloat16))
    write_check(g, m, a.out, inputs, dict(position=pos, **{f"e{i}": picked[i] for i in range(TOP)}))
    print(f"{a.out}: GptOssStep Q A O R T, then E; position {pos}{' (sliding)' if a.sliding else ''}, picked {picked}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of the phases on random weights, and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--experts", type=int, default=6)
    c.add_argument("--seed", type=int, default=5)
    c.add_argument("--rows", type=int, default=1024, help="the caches' rows")
    c.add_argument("--position", type=int, default=300, help="the call's position (the rows before it random)")
    c.add_argument("--sliding", action="store_true", help="a sliding layer's attention (the last 128 rows)")
    c.add_argument("--empty", action="store_true", help="the caches as they are (zeros): no copy into them, for timing")
    c.add_argument("--expanded", action="store_true", help="IRON's expanded loads (the array loaded each command)")
    a = ap.parse_args(argv)
    return _check(a)


if __name__ == "__main__":
    sys.exit(main())
