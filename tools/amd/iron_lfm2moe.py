"""LFM2-MoE (Liquid AI's Lfm2MoeForCausalLM, LiquidAI/LFM2-8B-A1B) on IRON's shared decoder (``iron.lm``).

An application of amd/IRON kept in hwnpu (IRON has none for LFM2-MoE), exported by
``iron_export_lm.py lfm2moe``. Its shape is read from the checkpoint's ``config.json``; the
structure follows transformers' modeling_lfm2_moe.py:

- ``layer_types``: ``conv`` layers are LFM2's gated short convolution (iron_lfm3.ShortConv),
  ``full_attention`` ones attention over q and k RMS-normed per head (``q_layernorm``,
  ``k_layernorm``), plain RoPE, ``out_proj``; q, k and v one projection (iron_qwen3's
  Qwen3Attention).
- Every layer is pre-normed (``operator_norm``, ``ffn_norm``; IRON's DecoderLayer); the final norm
  is ``embedding_norm``; the head is the embeddings.
- The first ``num_dense_layers`` feed-forwards are a SwiGLU MLP (``w1``, ``w3``, ``w2``); the others
  a mixture of ``num_experts`` SwiGLU experts of ``moe_intermediate_size``, ``num_experts_per_tok``
  a token: s = sigmoid(router logits), the top of s + ``expert_bias`` chosen, weighted by s over
  the chosen's sum (``norm_topk_prob``) times ``routed_scaling_factor``. No shared expert.

The experts are q4nx tables of every expert of a layer (iron_moe.quantize_experts: gate and up as
one table each, iron_moeprobe.gu_table; down one each).

- A decode step's MoE layer is two commands, as Qwen3.6's (iron_moe.py): A ends with the router's
  GEMV and MoERouteSigmoid (kernels/moe_route_sigmoid.cc), which writes the routing weights and
  command B's scratchpad (each chosen expert's byte offsets, the version's feedback argument);
  B is MoE4GateUp (each chosen expert on two columns, its half of the gate and up a column,
  silu(G x) (U x) w) and MoE4Down (the 4 experts' down side by side and the residual), reading
  the tables in place.
- A prompt chunk has two forms of a MoE layer, and the host takes one by the chunk's rows
  (hwnpu's internal/llm/iron/moe.go):

  - Up to ROW_MAX rows, the rows' own experts alone (SparseMoe.rows_body): the host routes the
    rows from the router's logits (``moe_rows_plan``) and runs one command once for every
    ROW_TOKENS rows, the decode step's operators with a row's place in the buffers as one more
    value (expert_operators(at=True)): 4 experts a row read in place, nothing dequantized.
  - More rows, each expert's own rows gathered into slots of SLOT_ROWS rows, an expert a slot
    (SparseMoe.gather_body): the host routes the chunk, writes the slots' rows and runs one
    command of GATHER_SLOTS slots once for every such slots (``moe_gather``), the slots' experts
    per-call offsets in the tables. A slot dequantizes its expert's tables (as a prompt chunk's
    projections are) and runs the matrix instruction's GEMMs on its rows; the host then adds
    each row's experts' rows, weighted, to the residual (``moe_gather_combine``). The slots'
    same operators run in a row, so that the array is configured once a run of them.

  The first form's command comes first, then the second's; the host leaves out the command of
  the form it does not take.
- With ``install(places=P)`` (``iron_export_lm.py --expert-places P``) a MoE layer's tables hold P
  places of an expert's tables and the experts are in the export's experts.bin: hwnpu copies
  those a decode step, a pass of rows or a pass of slots chose into places (pool.go) and gives
  the operators the places' offsets.

With install() (``iron_export_lm.py lfm2moe --q4``) the attention's and the convolutions'
projections are in ``attn`` and ``conv`` (q8 by default), the dense MLPs' in ``mlp`` (q8) and the
head in ``head`` (q8); the experts are q4nx whatever install says; the token embeddings go to the
host (iron_hostple's embed.bin).
"""

import dataclasses
import functools
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

import iron  # noqa: E402
import iron.functional as F  # noqa: E402
from iron import Precision, lm  # noqa: E402
from iron.lm import CausalLM, DecoderConfig, DecoderLayer, Linear  # noqa: E402
from iron.operators.copy import Copy  # noqa: E402

import iron_lfm3  # noqa: E402
import iron_moe  # noqa: E402
import iron_moeprobe  # noqa: E402
import iron_qwen3  # noqa: E402
import iron_verify  # noqa: E402

TOP = 4  # experts a token
WV = 16  # routing weights MoE4GateUp reads: the 4 chosen, zeros after
SLOTS = iron_moe.SLOTS  # command B's scratchpad words (the route's image)

# install()'s formats (iron_q4's), by part; empty: bf16 (the experts are q4nx either way).
QUANT: dict[str, str] = {}

# A prompt chunk's command: IRON cuts one when its scratchpad words would pass this (as
# iron_qwen35's dense models); each MoE layer also ends one (its host combine).
SCRATCHPAD_WORDS = 24

# A prompt chunk of more than ROW_MAX rows gathers each expert's rows into slots of SLOT_ROWS rows
# (the GEMM's rows are whole 256s), an expert with more rows taking further slots, and runs one
# command of GATHER_SLOTS slots once for every such slots (SparseMoe.gather_body), the slots' same
# operators in a row. A slot costs about 2 ms whatever its rows (0.7 ms dequantizing its expert's
# three tables on two cores a column, 0.84 its three GEMMs, of which 0.4 for 256 rows of
# products, 0.3 the Copy, the product and SiLU), so a chunk pays for the experts it uses, not for
# its rows: measured against every expert on every row (a layer of a 1024-row chunk 117 ms, of a
# 256-row one 81 ms; docs/design/amd-llm.md, with what was tried on a slot and not taken). More
# slots a command hold more dequantized weights together (2 x 7.3 MB a slot) and leave more of
# the last run's slots unused.
SLOT_ROWS = 256
GATHER_SLOTS = 4
GATHER_VALUES = [f"mg{j}" for j in range(GATHER_SLOTS)]

# A prompt chunk of up to ROW_MAX rows runs its experts a row at a time, as the decode step does
# (SparseMoe.rows_body: the row's 4 experts alone, read in place), ROW_TOKENS rows a pass of one
# command the host runs again for the next rows; a longer chunk gathers its rows by expert (the
# GEMMs: about 2.5 ms a used expert a layer). Measured (a prompt and one decode step):
# 33 rows 0.8 s, 64 1.3 s, 80 1.6 s, 96 1.9 s a row at a time (about 17 ms a row: 0.8 ms a layer)
# against 1.4, 1.5, 1.6 and 1.6 s gathered, so the forms cross at 80 rows. A pass takes the rows' indices and
# their experts as scratchpad values: 10 words a row (the experts' offsets in the two tables and
# the row's in the buffers), so 2 rows within SCRATCHPAD_WORDS.
# With fewer places than experts (POOL) a pass of rows copies in the experts it finds in no place,
# and a pass of slots only its 4: measured with 16 places, 48 rows 1.4 s a row at a time against
# 1.7 s gathered, 64 rows 1.8 against 1.8 (the experts read from the disk: 2.3 against 2.2, and 2.9
# against 2.3), so the rows form ends at ROW_MAX_PLACES rows.
ROW_TOKENS = 2
ROW_MAX = 80
ROW_MAX_PLACES = 48
ROW_VALUES = [f"mr{j}" for j in range(ROW_TOKENS)]
EXPERT_VALUES = [[f"me{j}_{i}" for i in range(TOP)] for j in range(ROW_TOKENS)]


# LFM2-MoE with fewer places than experts (install(places=P), iron_export_lm.py --expert-places):
# a MoE layer's tables hold P places of an expert's tables, zeros in the image; every expert's
# tables go to the export's experts.bin (by MoE layer in order, the gate-up tables of every
# expert then the down tables), and hwnpu copies an expert into a place when it is chosen
# (internal/llm/iron/pool.go, as Qwen3.6's iron_qwen35.POOL): a decode step's 4, a pass of the
# rows form's rows' and a pass of the gathered form's slots'.
POOL = dict(out=None, places=0, layers={})


def install(attn: str = "q8", conv: str = "q8", mlp: str = "q8", head: str = "q8", places: int = 0, out=None) -> None:
    """Build LFM2-MoE with its projections in these formats ("q4", "q6", "q8" or "bf16"), F.linear
    running iron_q4's on them, the token embeddings on the host.

    ``places``: the places of a MoE layer's experts' tables, 0 for one an expert; ``out`` the
    export's directory (experts.bin)."""
    import iron_hostple
    import iron_q4
    from iron.common.image import packaging

    for part, fmt in dict(attn=attn, conv=conv, mlp=mlp, head=head).items():
        if fmt not in ("q4", "q6", "q8", "bf16"):
            raise ValueError(f"{part} in {fmt}: q4, q6, q8 or bf16")
    if places and out is None:
        raise ValueError("places: out names the export's directory for experts.bin")
    if places and places < max(ROW_TOKENS * TOP, GATHER_SLOTS):
        raise ValueError(f"places: {places}, fewer than a pass of the prompt's experts ({max(ROW_TOKENS * TOP, GATHER_SLOTS)})")
    iron_q4.install_linear()
    QUANT.update(attn=attn, conv=conv, mlp=mlp, head=head)
    POOL.update(out=out, places=int(places or 0), layers={})
    packaging.SCRATCHPAD_WORDS = min(packaging.SCRATCHPAD_WORDS, SCRATCHPAD_WORDS, iron_hostple.SCRATCHPAD_WORDS)


@dataclasses.dataclass(frozen=True)
class Lfm2MoeConfig(DecoderConfig):
    """``DecoderConfig`` with LFM2's layer kinds, its short convolution and its experts."""

    eps: float = dataclasses.field(default=1e-5, kw_only=True, metadata={"json": "norm_eps"})
    layer_types: tuple[str, ...] = dataclasses.field(kw_only=True, metadata={"convert": tuple})
    conv_L_cache: int = dataclasses.field(default=3, kw_only=True)
    conv_bias: bool = dataclasses.field(default=False, kw_only=True)
    num_dense_layers: int = dataclasses.field(default=2, kw_only=True)
    num_experts: int = dataclasses.field(default=32, kw_only=True)
    num_experts_per_tok: int = dataclasses.field(default=TOP, kw_only=True)
    moe_intermediate_size: int = dataclasses.field(default=1792, kw_only=True)
    routed_scaling_factor: float = dataclasses.field(default=1.0, kw_only=True)
    # iron_qwen3.Qwen3Attention's: q and k normed per head, no biases
    head_norms: bool = lm.not_json(True, kw_only=True)
    qkv_bias: bool = lm.not_json(False, kw_only=True)
    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)

    @classmethod
    def from_json(cls, data: dict, source="config.json") -> "Lfm2MoeConfig":
        """``DecoderConfig.from_json``.

        Raises:
            ValueError: ``data`` asks for what is not built: another model, scaled RoPE, a layer
                kind other than conv and full attention, or experts routed otherwise.
        """
        if data.get("model_type") != "lfm2_moe":
            raise ValueError(f"{source}: model_type {data.get('model_type')!r} is not lfm2_moe")
        if data.get("rope_scaling"):
            raise ValueError(f"{source}: rope_scaling {data['rope_scaling']} is not built here")
        kinds = set(data.get("layer_types", ())) - {"conv", "full_attention"}
        if kinds:
            raise ValueError(f"{source}: layer kinds {sorted(kinds)} are not built here")
        if data.get("num_experts_per_tok") != TOP or not data.get("norm_topk_prob", True):
            raise ValueError(f"{source}: experts other than {TOP} a token over their sum are not built here")
        if not data.get("use_expert_bias", True):
            raise ValueError(f"{source}: routing without the expert bias is not built here")
        if data.get("num_experts", 32) % 16 or data.get("moe_intermediate_size", 1792) % 128:
            raise ValueError(f"{source}: {data.get('num_experts')} experts of {data.get('moe_intermediate_size')}")
        return super().from_json(data, source)

    def __post_init__(self):
        super().__post_init__()
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"Lfm2MoeConfig: {len(self.layer_types)} layer_types for {self.num_hidden_layers} layers")

    def moe(self, i: int) -> bool:
        """Whether layer ``i``'s feed-forward is the experts."""
        return i >= self.num_dense_layers


# --- the decode step's experts (command B) and routing ---------------------------------------


def route_reference(logits, bias, experts: int, top: int = TOP, scale: float = 1.0):
    """The chosen experts (by descending sigmoid + bias, ties to the lower index) and the routing
    weights (WV, the chosen's then zeros), as MoERouteSigmoid (float32 inside)."""
    x = np.asarray(logits, np.float32).reshape(-1)[:experts]
    s = (1 / (1 + np.exp(-x.astype(np.float64)))).astype(np.float32)
    pick = s + np.asarray(bias, np.float32).reshape(-1)[:experts]
    ids = np.argsort(-pick, kind="stable")[:top]
    w = np.zeros(WV, np.float32)
    w[:top] = s[ids] * (scale / (s[ids].sum() + 1e-6))
    return ids.astype(np.int32), w.astype(bfloat16)


@functools.cache
def route_operator():
    """MoERouteSigmoid (kernels/moe_route_sigmoid.cc): iron_moe's MoERoute with LFM2-MoE's scores."""
    from aie.iron import In, ObjectFifo, Out, TaskGroup, Worker
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    class MoERouteSigmoid(Operator):
        """``w, image = route(logits, program)``: one token's top-``top`` of ``experts`` by sigmoid
        and the expert bias (the program's last words, bf16), and the scratchpad image of their
        offsets (iron_moe's MoERoute)."""

        n: int = param(array=True)  # the logits row (the router's rows, padded)
        experts: int = param(default=32, array=True)
        top: int = param(default=TOP, array=True)
        slots: int = param(default=SLOTS, array=True)
        scale: float = param(default=1.0, array=True)
        words: int = param(derive=lambda op: 4 * op.slots + op.experts // 2, repr=False)
        wv: int = param(default=WV, init=False, repr=False)
        picked: int = param(default=-1, repr=False)  # PICKED's key on the host

        logits = In(n, obj_shape=(n,), depth=1)
        program = In(words, dtype=np.int32, obj_shape=(words,), depth=1)
        w = Out(wv, obj_shape=(wv,), depth=1)
        image = Out(slots, dtype=np.int32, obj_shape=(slots,), depth=1)

        def build(self) -> Built:
            kdir = Path(config.aie_kernels_dir())
            here = Path(__file__).resolve().parent / "kernels"
            kernel = ExternalFunction(
                "moe_route_sigmoid",
                source_file=str(here / "moe_route_sigmoid.cc"),
                include_dirs=[str(kdir), config.cxx_header_path()],
                compile_flags=[f"-DROUTE_EXPERTS={self.experts}", f"-DROUTE_TOP={self.top}",
                               f"-DROUTE_SLOTS={self.slots}", f"-DROUTE_W={self.wv}",
                               f"-DROUTE_SCALE={float(self.scale)!r}f"],
                arg_types=[self.logits.obj_type, self.program.obj_type, self.w.obj_type, self.image.obj_type],
            )
            fl, fp = ObjectFifo(self.logits.obj_type, depth=1), ObjectFifo(self.program.obj_type, depth=1)
            fw, fi = ObjectFifo(self.w.obj_type, depth=1), ObjectFifo(self.image.obj_type, depth=1)

            def core(fl, fp, fw, fi, k):
                lg, pr, w, im = fl.acquire(1), fp.acquire(1), fw.acquire(1), fi.acquire(1)
                k(lg, pr, w, im)
                fl.release(1)
                fp.release(1)
                fw.release(1)
                fi.release(1)

            worker = Worker(core, [fl.cons(), fp.cons(), fw.prod(), fi.prod(), kernel])
            self.logits.bind(fl.prod())
            self.program.bind(fp.prod())
            self.w.bind(fw.cons())
            self.image.bind(fi.cons())
            return Built([worker], [])

        def sequence(self, rt):
            tg = TaskGroup()
            self.logits.fill(group=tg)
            self.program.fill(group=tg)
            self.w.drain(group=tg, wait=True)
            self.image.drain(group=tg, wait=True)
            tg.finish()

        def ops(self) -> int:
            return self.top * self.experts

        def reference(self, logits, program):
            prog = np.asarray(program, np.int32).reshape(-1)
            bias = prog[4 * self.slots :].view(bfloat16).astype(np.float32)
            ids, w = route_reference(logits, bias, self.experts, self.top, float(self.scale))
            if self.picked >= 0:
                iron_moeprobe.PICKED[self.picked] = [int(i) for i in ids]
            p = prog[: 4 * self.slots].astype(np.int64).reshape(self.slots, 4)
            img = np.where(p[:, 0] < 0, p[:, 2], ids[np.clip(p[:, 0], 0, self.top - 1)] * p[:, 1] + p[:, 2])
            return w, (img << p[:, 3]).astype(np.int32)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.02, 1e-3, note="exp_f32 and a float divide, rounded to bf16")

    return MoERouteSigmoid


@functools.cache
def expert_operators(at: bool = False):
    """MoE4GateUp and MoE4Down: the 4 experts a decode step picks, read in place in a layer's
    tables (iron_moeprobe's MoEGateUp and MoEDown, for 4 experts and no shared one), the cores
    doing the element-wise work (kernels/moe_mv.cc).

    With ``at`` (a prompt chunk's rows, SparseMoe.rows_body) x, the routing weights, the residual
    and the output are rows of buffers of many: the per-call ``at`` is the row's element offset in
    them (x and the weights in rows of one stride, the residual and the output in rows of N), the
    output written in place (an InOut), as iron_shortconv's ShortConvStep reads and writes slots."""
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

    from iron.common import Built, Operator
    from iron_q4 import BLOCK_BYTES as B, CB, K_TILE, dequantize

    COLS = 8
    PER = COLS // TOP  # columns an expert (MoE4GateUp)
    ROWS = at  # the classes' own ``at`` is the per-call offset

    def row(op):
        """The per-call offset of the row (``at``), or None where it is not per call."""
        return op.at if ROWS and op.at.per_call else None
    MT = CB // 2  # out-features a core
    blk_t = np.ndarray[(B,), np.dtype[np.uint8]]
    half_t = np.ndarray[(MT,), np.dtype[bfloat16]]
    acc_t = np.ndarray[(MT,), np.dtype[np.float32]]
    silu = iron_moeprobe.silu

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

    def picked(op, given, eb):
        return iron_moeprobe.picked_offsets(op, given, eb)

    class MoE4GateUp(Operator):
        """``y = silu(G_e x) * (U_e x) * w[e]`` for the 4 experts a call picks: y is their NE
        out-features in turn. The table holds every expert's gu_table weights; expert i is on
        columns 2 i and 2 i + 1, each reading half its column blocks (at byte e_i)."""

        rows: int = param(array=True)  # blocks in the table
        K: int = param(array=True)
        NE: int = param(array=True)
        YN: int = param(derive=lambda op: TOP * op.NE, array=True)
        picked: int = param(default=-1, repr=False)  # the routing step (PICKED) on the host
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, B, dtype=np.uint8, obj_shape=(2, B), per=(num_aie_columns,))
        if ROWS:
            XT: int = param(array=True)  # the elements of x's buffer, rows of a stride
            WT: int = param(array=True)  # and of the weights', rows of the same stride
            x = In(XT, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
            w = In(WT, obj_shape=(WV,), per=(num_aie_columns,), replicate=True)
            at = per_call(derive=lambda op: 0)  # elements: the row read in both
        else:
            x = In(K, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
            w = In(WV, obj_shape=(WV,), per=(num_aie_columns,), replicate=True)
        y = Out(YN, obj_shape=(CB,), per=(num_aie_columns,))
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)

        def validate(self):
            if self.NE % (PER * CB):
                raise ValueError(f"MoE4GateUp: {self.NE} out-features an expert, not {PER} columns of whole blocks")

        def resolve(self, dev):
            return self.replace(num_aie_columns=COLS)

        def expert_bytes(self) -> int:
            return 2 * self.NE // CB * (self.K // K_TILE) * 2 * B

        def build(self) -> Built:
            K = self.K
            kbs, cbs = K // K_TILE, self.NE // CB // PER
            k = kernels(K, WV)
            w_l3 = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(COLS)]
            y_l3 = [ObjectFifo(self.y.obj_type, depth=2) for _ in range(COLS)]
            w_l1 = [f.cons().split([0, B], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_l3]
            # x, then the routing weights, down one channel into one object for both cores
            xw_t = np.ndarray[(K + WV,), np.dtype[bfloat16]]
            xw_l3 = [ObjectFifo(xw_t, depth=1) for _ in range(COLS)]
            xw_l1 = [f.cons().forward(obj_type=xw_t, tile=AnyMemTile) for f in xw_l3]
            y_l1 = [f.prod().join([0, MT], obj_types=[half_t, half_t], depths=[2, 2], tile=AnyMemTile)
                    for f in y_l3]

            def core(w_f, x_f, y_f, sums, g, u, idx, sums_k, zero, block, glu):
                x = x_f.acquire(1)
                sums_k(x, sums)
                zero(g)
                zero(u)
                for _ in range_(cbs):
                    for acc in (g, u):
                        for kb in range_(kbs):
                            kb32 = index.casts(T.i32(), kb)
                            blk = w_f.acquire(1)
                            block(blk, x, sums, acc, kb32)
                            w_f.release(1)
                    y = y_f.acquire(1)
                    glu(g, u, x, idx, y)
                    y_f.release(1)
                x_f.release(1)

            workers = []
            for i in range(COLS):
                for j in range(2):
                    sums = Buffer(np.ndarray[(K // 32,), np.dtype[bfloat16]])
                    g, u = Buffer(acc_t), Buffer(acc_t)
                    workers.append(Worker(
                        lambda w_f, x_f, y_f, sums, g, u, s, z, b, gl, e=i // PER: core(
                            w_f, x_f, y_f, sums, g, u, e, s, z, b, gl),
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
            part = eb // PER  # a column's half of an expert: its column blocks, contiguous
            total = self.rows * B
            drains = []
            for l in range(COLS):
                e, h = divmod(l, PER)
                drains.append(self.y.drain(TAP((YN,), e * NE + h * (NE // PER), [NE // PER], [1]), lane=l,
                                           wait=True, managed=False))
            for l in range(COLS):
                if ROWS:
                    self.x.fill(TAP((self.XT,), 0, [K], [1]), lane=l, offset_parameter=row(self), managed=False)
                    self.w.fill(TAP((self.WT,), 0, [WV], [1]), lane=l, offset_parameter=row(self), managed=False)
                    continue
                self.x.fill(TAP((K,), 0, [K], [1]), lane=l, managed=False)
                self.w.fill(TAP((WV,), 0, [WV], [1]), lane=l, managed=False)
            for l in range(COLS):
                e, h = divmod(l, PER)
                self.W.fill(TAP((total,), h * part, [part], [1]), lane=l, offset_parameter=offset(self, e),
                            managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * 2 * self.YN * self.K

        def reference(self, W, x, w, *, e0=0, e1=0, e2=0, e3=0, at=0):
            eb = self.expert_bytes()
            flat = np.asarray(W, np.uint8).reshape(-1)
            x = np.asarray(x, np.float64).reshape(-1)[int(at) : int(at) + self.K]
            w = np.asarray(w, np.float64).reshape(-1)[int(at) : int(at) + WV]
            ys = []
            for idx, off in enumerate(picked(self, [e0, e1, e2, e3], eb)):
                gu = dequantize(flat[int(off) : int(off) + eb], 2 * self.NE, self.K).astype(np.float64)
                gu = gu.reshape(self.NE // CB, 2, CB, self.K)
                g = gu[:, 0].reshape(self.NE, self.K) @ x
                u = gu[:, 1].reshape(self.NE, self.K) @ x
                ys.append(silu(g) * u * w[idx])
            return np.concatenate(ys).astype(bfloat16)

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="q4nx partial sums in bf16, SiLU's tanh")

    class MoE4Down(Operator):
        """``y = r + sum_e D_e x_e`` over the 4 experts a call picks, x their activations side by
        side (4 KE). Column l keeps N / 8 out-features and reads each expert's part of them in one
        contiguous run (at byte e_i)."""

        # 4 weight runs, x and the residual a column: let aiecc recycle finished BDs (as Copy)
        aiecc_flags: ClassVar[tuple[str, ...]] = ("--reclaim-runtime-bds",)

        rows: int = param(array=True)
        KE: int = param(array=True)
        K: int = param(derive=lambda op: TOP * op.KE, array=True)
        N: int = param(array=True)
        picked: int = param(default=-1, repr=False)
        num_aie_columns: int = tunable(domain=widths)

        W = In(rows, B, dtype=np.uint8, obj_shape=(2, B), per=(num_aie_columns,))
        x = In(K, obj_shape=(K,), per=(num_aie_columns,), replicate=True)
        R: int = param(derive=lambda op: op.N // COLS, array=True)  # the residual a column
        if ROWS:
            RT: int = param(array=True)  # the elements of the residual's buffer, rows of N
            YT: int = param(array=True)  # and of the output's
            r = In(RT, obj_shape=(R,), per=(num_aie_columns,))
            y = InOut(YT, obj_shape=(CB,), per=(num_aie_columns,))
            at = per_call(derive=lambda op: 0)  # elements: the row read and the row written
        else:
            r = In(N, obj_shape=(R,), per=(num_aie_columns,))
            y = Out(N, obj_shape=(CB,), per=(num_aie_columns,))
        e0 = per_call(derive=lambda op: 0)
        e1 = per_call(derive=lambda op: 0)
        e2 = per_call(derive=lambda op: 0)
        e3 = per_call(derive=lambda op: 0)

        def validate(self):
            if self.N % (COLS * CB) or self.KE % K_TILE or self.K % 512:
                raise ValueError(f"MoE4Down: {self.N} out-features of 4 x {self.KE}")

        def resolve(self, dev):
            return self.replace(num_aie_columns=COLS)

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
                for e in range_(TOP):
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
            if ROWS:
                drains = [self.y.drain(TAP((self.YT,), l * N // n, [N // n], [1]), lane=l, offset_parameter=row(self),
                                       wait=True, managed=False) for l in range(n)]
            else:
                drains = [self.y.drain(TAP((N,), l * N // n, [N // n], [1]), lane=l, wait=True, managed=False)
                          for l in range(n)]
            for l in range(n):
                self.x.fill(TAP((K,), 0, [K], [1]), lane=l, managed=False)
                if ROWS:
                    self.r.fill(TAP((self.RT,), l * N // n, [N // n], [1]), lane=l, offset_parameter=row(self),
                                managed=False)
                    continue
                self.r.fill(TAP((N,), l * N // n, [N // n], [1]), lane=l, managed=False)
            for l in range(n):
                for e in range(TOP):
                    self.W.fill(TAP((total,), l * run, [run], [1]), lane=l, offset_parameter=offset(self, e),
                                managed=False)
            for d in drains:
                d.await_()

        def ops(self) -> int:
            return 2 * self.N * self.K

        def reference(self, W, x, r, y=None, *, e0=0, e1=0, e2=0, e3=0, at=0):
            eb = self.expert_bytes()
            flat = np.asarray(W, np.uint8).reshape(-1)
            x = np.asarray(x, np.float64)
            out = np.asarray(r, np.float64).reshape(-1)[int(at) : int(at) + self.N].copy()
            for i, off in enumerate(picked(self, [e0, e1, e2, e3], eb)):
                d = dequantize(flat[int(off) : int(off) + eb], self.N, self.KE).astype(np.float64)
                out += d @ x[i * self.KE : (i + 1) * self.KE]
            if y is None:
                return out.astype(bfloat16)
            whole = np.asarray(y).reshape(-1).copy()  # the rows form: the output's buffer, its row written
            whole[int(at) : int(at) + self.N] = out.astype(bfloat16)
            return whole

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.05, 1e-2, note="q4nx partial sums in bf16")

    return MoE4GateUp, MoE4Down


@functools.cache
def dequant_at(glu: bool):
    """iron_q4's QDequant over one expert of a layer's table, the expert a per-call byte offset
    (``at``): the expert's blocks dequantized to the (N, K) bf16 weight a slot's GEMM takes
    (SparseMoe.gather_body). With ``glu`` the table is the experts' gu_tables and ``parity`` picks
    the gate's (0) or the up's (1) column blocks (as iron_q4's glu_operators' select).

    Two cores a column, each one half (32 out-features) of the column's column blocks: a column
    block's pair of blocks is split in the memory tile and the halves' rows joined there. The
    same bits as QDequant's one core (compared on a layer's tables); 0.24 ms a table against 0.30
    (16 in a row). It is bound by what it moves (2.3 MB in, 7.3 MB of bf16 out): a stream out
    for each core gave no more."""
    from aie.helpers.taplib import TensorAccessPattern as TAP
    from aie.iron import In, ObjectFifo, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import AnyMemTile
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param, per_call

    import iron_q4
    from iron.common import Built
    from iron_q4 import CB, K_TILE

    _, base = iron_q4.operators()
    HALF = CB // 2 * K_TILE  # a core's part of the output's object: 32 out-features of 256 in-features

    def build(self) -> Built:
        if self.fmt != "q4":
            raise ValueError(f"dequant_at: {self.fmt}, not q4")
        n, B = self.num_aie_columns, self.block_bytes
        cbs, kbs = self.N // CB // n, self.K // K_TILE
        blk_t = np.ndarray[(B,), np.dtype[np.uint8]]
        half_t = np.ndarray[(HALF,), np.dtype[bfloat16]]
        kdir = Path(config.aie_kernels_dir())
        here = Path(__file__).resolve().parent / "kernels"
        # q4_dequant writing its first half: the output it is given is a half's object
        deq = ExternalFunction("q4_dequant", source_file=str(here / "q4_mv.cc"), arg_types=[blk_t, half_t, np.int32],
                               include_dirs=[str(kdir / "flm_gemma4"), str(kdir), config.cxx_header_path()],
                               compile_flags=[f"-DQ4_K={self.K}"], object_file_name=f"q4_dequant_half_{self.K}.o")
        w_f = [ObjectFifo(self.W.obj_type, depth=2) for _ in range(n)]
        o_f = [ObjectFifo(self.w.obj_type, depth=1) for _ in range(n)]
        w_c = [f.cons().split([0, B], obj_types=[blk_t, blk_t], tile=AnyMemTile) for f in w_f]
        o_c = [f.prod().join([0, HALF], obj_types=[half_t, half_t], depths=[1, 1], tile=AnyMemTile) for f in o_f]

        def core(w_f, o_f, deq):
            for _ in range_(cbs * kbs):
                o = o_f.acquire(1)
                w = w_f.acquire(1)
                deq(w, o, 0)
                w_f.release(1)
                o_f.release(1)

        workers = [Worker(core, [w_c[i][j].cons(), o_c[i][j].prod(), deq]) for i in range(n) for j in range(2)]
        self.W.bind([f.prod() for f in w_f])
        self.w.bind([f.cons() for f in o_f])
        return Built(workers, [])

    def expert_bytes(self) -> int:
        return (2 if glu else 1) * self.blocks * self.block_bytes

    def sequence(self, rt):
        n, B, N, K = self.num_aie_columns, self.block_bytes, self.N, self.K
        rows, total = N // n, self.trows * B
        at = self.at if self.at.per_call else None
        drains = []
        for lane in range(n):
            # the lane's column blocks, 256 in-features at a time, 64 rows of 256 each (QDequant's)
            tap = TAP((N, K), lane * rows * K, [rows // CB, K // K_TILE, CB, K_TILE], [CB * K, K_TILE, K, 1])
            drains.append(self.w.drain(tap, lane=lane, wait=True, managed=False))
        cb_bytes = 2 * (K // K_TILE) * B
        cbs = N // CB // n
        c = iron_q4._cb_chunk(cb_bytes)
        for lane in range(n):
            if glu:  # every other column block of the expert's gu_table
                tap = TAP((total,), (2 * lane * cbs + self.parity) * cb_bytes, [cbs, cb_bytes // c, c],
                          [2 * cb_bytes, c, 1])
            else:
                tap = TAP((total,), lane * cbs * cb_bytes, [cbs, cb_bytes // c, c], [cb_bytes, c, 1])
            self.W.fill(tap, lane=lane, offset_parameter=at, managed=False)
        for d in drains:
            d.await_()

    def reference(self, W, *, at=0):
        eb = expert_bytes(self)
        flat = np.asarray(W, np.uint8).reshape(-1)[int(at) : int(at) + eb]
        if not glu:
            return iron_q4.dequantize(flat, self.N, self.K).astype(bfloat16)
        t = iron_q4.dequantize(flat, 2 * self.N, self.K).reshape(self.N // CB, 2, CB, self.K)
        return t[:, self.parity].reshape(self.N, self.K).astype(bfloat16)

    body = dict(
        trows=param(array=True),  # blocks in the table
        at=per_call(derive=lambda op: 0),  # bytes: the expert's place in the table
        build=build, sequence=sequence, reference=reference, expert_bytes=expert_bytes,
        __annotations__={"trows": int},
        __doc__=dequant_at.__doc__,
    )
    if glu:
        body["parity"] = param(default=0, array=True)
        body["__annotations__"]["parity"] = int
    # an object a pair of blocks: the two halves of a column block's 256 in-features, a core each
    body["W"] = In(body["trows"], base.block_bytes, dtype=np.uint8, obj_shape=(2, base.block_bytes),
                   per=(base.num_aie_columns,))
    return type("QDequantGluAt" if glu else "QDequantAt", (base,), body)


# --- the model -------------------------------------------------------------------------------


class _Expert(iron.Graph):
    """An expert's ``w1`` (gate), ``w3`` (up) and ``w2`` (down), as the checkpoint names them."""

    def __init__(self, E: int, I: int, precision):
        self.w1 = Linear(E, I, False, precision=precision)
        self.w3 = Linear(E, I, False, precision=precision)
        self.w2 = Linear(I, E, False, precision=precision)


class SparseMoe(iron.Graph):
    """LFM2-MoE's experts (``mlp``): the router (``gate``), its ``expert_bias`` and the
    ``experts``; on the device MoERouteSigmoid, MoE4GateUp and MoE4Down (module docstring)."""

    def __init__(self, c: Lfm2MoeConfig, index: int):
        E, I, N = c.hidden_size, c.moe_intermediate_size, c.num_experts
        self.index, self.hidden, self.inter, self.n = index, E, I, N
        self.first = c.num_dense_layers  # the first MoE layer: experts.bin holds them from it
        self.scale = float(c.routed_scaling_factor)
        self.gate = Linear(E, N, False, precision=c.precision)
        self.expert_bias = iron.param(N, dtype=np.float32)
        self.experts = [_Expert(E, I, c.precision) for _ in range(N)]
        self.precision = c.precision

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        E, I, N = self.hidden, self.inter, self.n
        tables, downs = [], []
        for ex in self.experts:
            tables.append(iron_moeprobe.gu_table(np.asarray(ex.w1.weight, np.float32),
                                                 np.asarray(ex.w3.weight, np.float32)))
            downs.append(np.asarray(ex.w2.weight, np.float32))
        gu_t = iron_moe.quantize_experts(tables)
        del tables
        dn_t = iron_moe.quantize_experts(downs)
        del downs
        self.gu_bytes = gu_t.nbytes // N
        self.down_bytes = dn_t.nbytes // N
        self.gu_rows, self.down_rows = gu_t.shape[0] // N, dn_t.shape[0] // N
        P = self.places = POOL["places"] or N
        if P < N:  # the places zeros (hwnpu fills them); every expert's in experts.bin
            self.write_pool(gu_t.reshape(-1).view(np.uint8), dn_t.reshape(-1).view(np.uint8))
            gu_t = np.zeros((P * self.gu_rows,) + gu_t.shape[1:], gu_t.dtype)
            dn_t = np.zeros((P * self.down_rows,) + dn_t.shape[1:], dn_t.dtype)
        elif P != N:
            raise SystemExit(f"--expert-places {P}: more places than the {N} experts")
        self.gu = iron.param(gu_t)
        self.down = iron.param(dn_t)
        # the router's logits, padded to a whole column block
        rows = -(-N // 64) * 64
        router = np.zeros((rows, E), bfloat16)
        router[:N] = np.asarray(self.gate.weight)
        self.router = iron.param(router)
        self._bias = np.asarray(self.expert_bias, np.float32).copy()
        # B's scratchpad image (written once the lowering has placed its words, program_rows),
        # then the expert bias as bf16
        prog = np.zeros(4 * SLOTS + N // 2, np.int32)
        prog[4 * SLOTS :] = self._bias.astype(bfloat16).view(np.int32)
        self.program = iron.param(prog)
        self.gate = self.experts = self.expert_bias = None

    def write_pool(self, gu, dn):
        """The layer's every expert's tables into experts.bin, the MoE layers in order."""
        out = Path(POOL["out"])
        out.mkdir(parents=True, exist_ok=True)
        f = out / "experts.bin"
        if not POOL["layers"] and f.exists():
            f.unlink()  # an earlier export's
        layer = gu.size + dn.size
        offset = (self.index - self.first) * layer
        with open(f, "r+b" if f.exists() else "w+b") as fh:
            fh.seek(offset)
            fh.write(np.ascontiguousarray(gu).tobytes())
            fh.write(np.ascontiguousarray(dn).tobytes())
        POOL["layers"][self.index] = dict(offset=offset, gu_bytes=int(self.gu_bytes), dn_bytes=int(self.down_bytes))

    def body(self, hn, h, call):
        if call.prompt:
            return self.prompt_body(hn, h, call)
        E, I = self.hidden, self.inter
        Route = route_operator()
        logits = F.linear(hn, self.router, precision=self.precision)
        n = self.router.shape[0]
        w, image = Route.step(logits.reshape(n), self.program, n=n, experts=self.n, scale=self.scale,
                              picked=self.index)
        call.feedback.append((self.index, image.name))
        iron.on_host(lambda: None)  # command B: the experts, at the offsets the image holds
        e = call.experts[self.index]
        GU, D = expert_operators()
        a = GU.step(self.gu, hn.reshape(E), w, rows=self.gu.shape[0], K=E, NE=I, picked=self.index,
                    **{f"e{i}": e[i] * self.gu_bytes for i in range(TOP)})
        y = D.step(self.down, a, h.reshape(E), rows=self.down.shape[0], KE=I, N=E, picked=self.index,
                   **{f"e{i}": e[i] * self.down_bytes for i in range(TOP)})
        iron.on_host(lambda: None)
        return y.reshape(1, E)

    def rows_body(self, call):
        """A short prompt chunk's experts, ROW_TOKENS rows a pass: each row's 4 experts read in
        place (expert_operators(at=True): MoE4GateUp and MoE4Down on the row ``mr<j>`` of
        ``moe.x`` (the normed rows), ``moe.w`` (its routing weights, the host's), ``moe.h`` (the
        residual) and ``moe.rout`` (written in place: the host copies the rows into ``moe.out``,
        which a step may not write as the host does), the experts ``me<j>_<i>``). The host runs
        the command once for every ROW_TOKENS rows (hwnpu's moe_rows_plan); the same operators
        are in a row, so a pass configures the array twice."""
        moe, E, I = call.moe, self.hidden, self.inter
        C = moe.h.shape[0]
        GU, D = expert_operators(at=True)
        acts = []
        for r, e in zip(call.mrows, call.mexperts):
            acts.append(GU.step(self.gu, moe.x.reshape(C * E), moe.w, rows=self.gu.shape[0], K=E, NE=I,
                                XT=C * E, WT=ROW_MAX * E, at=r * E,
                                **{f"e{i}": e[i] * self.gu_bytes for i in range(TOP)}))
        for a, r, e in zip(acts, call.mrows, call.mexperts):
            D.step(self.down, a, moe.h.reshape(C * E), moe.rout, rows=self.down.shape[0], KE=I, N=E,
                   RT=C * E, YT=ROW_MAX * E, at=r * E, **{f"e{i}": e[i] * self.down_bytes for i in range(TOP)})

    def gather_body(self, call):
        """A longer prompt chunk's experts, GATHER_SLOTS slots a pass: slot j is SLOT_ROWS rows of
        ``moe.gx`` that the host filled with the rows of one expert, ``mg<j>`` (hwnpu's
        moe_gather), and its output the same rows of ``moe.gy``: the expert's tables dequantized
        (dequant_at), the gate's and the up's GEMMs, SiLU, their product and the down's GEMM.
        The host runs the command once for every GATHER_SLOTS slots.

        The slots' same operators run in a row: the array is configured once for a run of one
        operator, where a slot's operators in turn (dequantize, GEMM, SiLU, ...) configured it 9
        times a slot. A pass's dequantized weights (2 x 7.3 MB a slot) are held together."""
        import iron_q4

        moe, E, I, R = call.moe, self.hidden, self.inter, SLOT_ROWS
        DG, DD = dequant_at(True), dequant_at(False)
        linear = iron_q4._BASE.get("linear", F.linear)  # IRON's own, on the bf16 weights
        P = Precision.MEDIUM
        es = call.mslots
        xs = [moe.gx[j * R : (j + 1) * R] for j in range(len(es))]
        wg, wu = ([DG.step(self.gu, fmt="q4", K=E, parity=p, blocks=self.gu_rows // 2, trows=self.gu.shape[0],
                           at=e * self.gu_bytes) for e in es] for p in (0, 1))
        g = [linear(x, w, precision=P) for x, w in zip(xs, wg)]
        u = [linear(x, w, precision=P) for x, w in zip(xs, wu)]
        g = [F.silu(v) for v in g]
        a = [v * w for v, w in zip(g, u)]
        # down's in-features (1792) are no whole 512s, which DequantBFP's GEMM takes: bf16
        wd = [DD.step(self.down, fmt="q4", K=I, blocks=self.down_rows, trows=self.down.shape[0],
                      at=e * self.down_bytes) for e in es]
        y = [linear(v, w, precision=P) for v, w in zip(a, wd)]
        # A Copy into the slot's rows (0.17 ms a slot). The GEMM given the rows as its output (a
        # view of the state) gave the same bits without it, but after a chunk that ran so, the next
        # chunk's command after the experts timed out (ERT_CMD_STATE_TIMEOUT).
        for j, v in enumerate(y):
            Copy.step(v, moe.gy[j * R : (j + 1) * R])

    def prompt_body(self, hn, h, call):
        """A prompt chunk's experts, in two forms of which the host runs one (hwnpu's
        moe_rows_plan): up to ROW_MAX rows a row at a time (rows_body); more, each expert's rows
        gathered into slots (gather_body), the host routing the rows, filling the slots and
        adding each row's experts' rows, weighted, to the residual (hwnpu's moe_gather and
        moe_gather_combine). The router's logits go to ``call.moe.logits`` for both.

        IRON's arena does not count a host step's reads as uses: what the host reads is copied
        into states first."""
        moe, (C, E) = call.moe, hn.shape
        Copy.step(h, moe.h)
        Copy.step(hn, moe.x)
        Copy.step(F.linear(hn, self.router, precision=self.precision), moe.logits)
        iron.on_host(MoERowsHost(self.index, self.n, C, E, self._bias, self.scale, self.places), moe.logits,
                     writes=(moe.w,))
        self.rows_body(call)
        iron.on_host(MoEGatherHost(self.index, self.n, C, self._bias, self.scale), moe.logits, moe.x,
                     writes=(moe.gx,))
        self.gather_body(call)
        iron.on_host(MoECombineHost(self, C), moe.logits, moe.h, moe.x, moe.gy, moe.rout, writes=(moe.out,))
        return moe.out[: call.rows]


class MoEStates:
    """A prompt chunk's states of the experts (one set: the layers run in turn): the residual, the
    normed rows, the router's logits and the layer's output (the host's); the rows form's weights
    and output; the gathered form's slots, their rows in and out."""

    def __init__(self, C: int, E: int, experts: int):
        rows = -(-experts // 64) * 64
        self.h = iron.state((C, E), bfloat16, name="moe_h")
        self.x = iron.state((C, E), bfloat16, name="moe_x")
        self.logits = iron.state((C, rows), bfloat16, name="moe_logits")
        self.out = iron.state((C, E), bfloat16, name="moe_out")
        # the rows form (SparseMoe.rows_body): each row's routing weights (the host's) in rows of
        # the normed rows' stride, and its output
        self.w = iron.state((ROW_MAX * E,), bfloat16, name="moe_w")
        self.rout = iron.state((ROW_MAX * E,), bfloat16, name="moe_rout")
        # the gathered form (SparseMoe.gather_body): a pass's slots, the host's rows in and the
        # experts' rows out
        self.gx = iron.state((GATHER_SLOTS * SLOT_ROWS, E), bfloat16, name="moe_gx")
        self.gy = iron.state((GATHER_SLOTS * SLOT_ROWS, E), bfloat16, name="moe_gy")


def _bias16(bias) -> list:
    """The expert bias as the bf16 values the decode step's routing reads."""
    return [float(v) for v in np.asarray(bias, np.float32).astype(bfloat16).astype(np.float32)]


class MoERowsHost:
    """The host step ahead of a prompt chunk's experts (hwnpu's moe_rows_plan): for a chunk of up
    to ``row_max`` rows it routes each row from the logits, writes the rows' weights (the first
    ``top`` of a row of ``stride`` elements) and runs the next command once for every ``row_tokens``
    rows, their indices in ``row_values`` and their experts in ``expert_values``; for a longer
    chunk it leaves that command out and the gathered form's runs (moe_gather). Called (IRON's
    reference), it does nothing: the reference's result is MoECombineHost's own."""

    def __init__(self, layer: int, experts: int, C: int, E: int, bias, scale: float, places: int):
        row_max = ROW_MAX if places >= experts else min(ROW_MAX, ROW_MAX_PLACES)
        self.hwnpu = dict(kind="moe_rows_plan", layer=layer, experts=experts, top=TOP, tokens=C, score="sigmoid",
                          bias=_bias16(bias), scale=float(scale), row_max=row_max, row_tokens=ROW_TOKENS, stride=E,
                          row_values=ROW_VALUES, expert_values=EXPERT_VALUES)

    def __call__(self, *args):
        pass


class MoEGatherHost:
    """The host step ahead of the gathered form's command (hwnpu's moe_gather), left out with the
    command where the rows form ran: the host routes each row from the logits, gives each expert
    the rows that chose it in slots of ``slot_rows`` rows (an expert of more rows further slots),
    and runs the command once for every ``slots`` slots: the slots' rows of the normed rows
    written to ``gx``, their experts in ``slot_values`` (a run short of slots repeats its last
    slot's expert), and the slots' rows of the command's output read after it. Called (IRON's
    reference), it does nothing."""

    def __init__(self, layer: int, experts: int, C: int, bias, scale: float):
        self.hwnpu = dict(kind="moe_gather", layer=layer, experts=experts, top=TOP, tokens=C, score="sigmoid",
                          bias=_bias16(bias), scale=float(scale), slots=GATHER_SLOTS, slot_rows=SLOT_ROWS,
                          slot_values=GATHER_VALUES)

    def __call__(self, *args):
        pass


class MoECombineHost:
    """The host step after a prompt chunk's experts (hwnpu's moe_gather_combine): out = h + the
    sum over each row's chosen experts of weight x the expert's row of the gathered form's
    output, summed in float32 in the order chosen; where the rows form ran, its rows (``rout``)
    copied into out.

    Called, it is IRON's reference, which runs neither form as the host does: each row's experts
    from the tables themselves, in float32."""

    def __init__(self, moe: "SparseMoe", C: int):
        self.moe, self.C = moe, C
        self.hwnpu = dict(kind="moe_gather_combine", layer=moe.index, experts=moe.n, top=TOP, tokens=C,
                          score="sigmoid", bias=_bias16(moe._bias), scale=float(moe.scale))

    def __call__(self, logits, h, x, gy, rout, out):
        import iron_q4

        m, C = self.moe, self.C
        if m.places < m.n:
            raise SystemExit("the reference of LFM2-MoE's prompt reads every expert's tables: no --expert-places")
        E, I = m.hidden, m.inter
        lg, xv = np.asarray(logits, np.float32), np.asarray(x, np.float32)
        res = np.asarray(h, np.float32).copy()
        bias = np.asarray(self.hwnpu["bias"], np.float32)
        routes = [route_reference(lg[t], bias, m.n, TOP, m.scale) for t in range(C)]
        rows: dict[int, list] = {}
        for t, (ids, _) in enumerate(routes):
            for k, e in enumerate(ids):
                rows.setdefault(int(e), []).append((t, k))
        gu, down = np.asarray(m.gu, np.uint8).reshape(-1), np.asarray(m.down, np.uint8).reshape(-1)
        ys = np.zeros((C, TOP, E), np.float32)
        for e, picks in rows.items():
            t = iron_q4.dequantize(gu[e * m.gu_bytes : (e + 1) * m.gu_bytes], 2 * I, E)
            t = t.reshape(I // iron_q4.CB, 2, iron_q4.CB, E)
            xs = xv[[t_ for t_, _ in picks]]
            a = iron_moeprobe.silu(xs @ t[:, 0].reshape(I, E).T) * (xs @ t[:, 1].reshape(I, E).T)
            d = iron_q4.dequantize(down[e * m.down_bytes : (e + 1) * m.down_bytes], E, I)
            y = (a.astype(np.float32) @ d.T).astype(bfloat16).astype(np.float32)
            for (t_, k), v in zip(picks, y):
                ys[t_, k] = v
        for t, (_, w) in enumerate(routes):
            for k in range(TOP):
                res[t] += float(w[k]) * ys[t, k]
        out[:] = res.astype(bfloat16)


class Lfm2MoeLayer(DecoderLayer):
    """Layer ``i``: a short convolution or attention, then a SwiGLU MLP or the experts, pre-normed."""

    def __init__(self, c: Lfm2MoeConfig, i: int):
        if c.layer_types[i] == "conv":
            super().__init__(c, i, self_attn=iron.Graph(), cache=iron.Graph())
            del self.self_attn, self.cache
            self.conv = iron_lfm3.ShortConv(c)
            self.conv.chunked = True
        else:
            super().__init__(c, i, iron_qwen3.Qwen3Attention(c, i))
            self.conv = None
        if c.moe(i):
            self.mlp = SparseMoe(c, i)

    def mix(self, h, call):
        if self.conv is not None:
            return self.conv(h, call)
        return super().mix(h, call)

    def body(self, x, call):
        # A decode step's residual added in the mixer's last projection's GEMV (out_proj or o_proj,
        # iron_q4's into_linear); MoE4Down adds the experts' own, a dense MLP's down_proj its own
        fast = not (call.prompt or x.shape[0] != 1 or not QUANT or iron_verify.is_rows(x))
        if fast:
            import iron_q4

            last = self.conv.out_proj if self.conv is not None else self.self_attn.o_proj
            with iron_q4.into_linear(last, x) as st:
                y = self.mix(self.input_layernorm(x), call)
            h = y if st["used"] else x + y
        else:
            h = x + self.mix(self.input_layernorm(x), call)
        if isinstance(self.mlp, SparseMoe):
            return self.mlp(self.post_attention_layernorm(h), h, call)
        if fast:
            with iron_q4.into_linear(self.mlp.down_proj, h) as st:
                y = self.mlp(self.post_attention_layernorm(h))
            return y if st["used"] else h + y
        return h + self.mlp(self.post_attention_layernorm(h))


class Lfm2Moe(CausalLM):
    """LFM2-MoE on ``config``'s shape and a checkpoint's ``weights``.

    Its body takes a scratchpad value per MoE layer and chosen expert (``e<layer>_<i>``), which no
    host gives: command B's scratchpad is in the feedback argument, where the layer's
    MoERouteSigmoid writes them (``program_rows``)."""

    config = Lfm2MoeConfig
    layer_class = Lfm2MoeLayer
    hwnpu_name = "Lfm2Moe"
    renames = {
        "operator_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "self_attn.q_layernorm": "self_attn.q_norm",
        "self_attn.k_layernorm": "self_attn.k_norm",
        "self_attn.out_proj": "self_attn.o_proj",
        "feed_forward.w1": "mlp.gate_proj",
        "feed_forward.w3": "mlp.up_proj",
        "feed_forward.w2": "mlp.down_proj",
        "feed_forward": "mlp",
        "embedding_norm": "norm",
    }

    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        if getattr(cls, "_values_made", None) is not None:
            return super().from_checkpoint(path, **kwargs)
        from iron.lm.checkpoint import weights_file

        c = cls.default_config(weights_file(path))
        layers = [i for i in range(c.num_hidden_layers) if c.moe(i)]
        return moe_class(cls, layers).from_checkpoint(path, **kwargs)

    def __init__(self, config: Lfm2MoeConfig, weights):
        super().__init__(config, weights)
        for i, layer in enumerate(self.layers):
            if layer.conv is not None:
                layer.rope = None
        self.host_embed = False
        if QUANT:
            self.quantize(QUANT)
            self.embed_on_host()

    def draw(self, logits, call):
        # No draw on the device: its commands end at host steps, so no Emit carries a drawn
        # token, and hwnpu draws from the logits anyway (as Qwen3.6's).
        return logits, iron.carry(token=call.token + 0, position=call.position + 1)

    def record(self, **shapes):
        if not shapes or not hasattr(self, "_fed"):  # the decode step's (a prompt chunk feeds none)
            self._fed = []
        recording = super().record(**shapes)
        if not shapes:
            recording.feedback.extend(name for _, name in self._fed)
        return recording

    def moe_body(self, values, **standard):
        experts = {l: [values[f"e{l}_{i}"] for i in range(TOP)] for l in self._moe_layers}
        import iron.lm.causal as causal

        made = causal.Call

        def call(**kw):
            c = made(**kw)
            c.experts, c.feedback = experts, self._fed
            if c.prompt:
                c.moe = self._moe_states()
                c.mrows = [values[n] for n in ROW_VALUES]
                c.mexperts = [[values[n] for n in names] for names in EXPERT_VALUES]
                c.mslots = [values[n] for n in GATHER_VALUES]
            return c

        causal.Call = call
        try:
            return self.plain_body(**standard)
        finally:
            causal.Call = made

    def plain_body(self, ids=None, **values):
        if ids is not None and self.host_embed:
            # a version's input must be read, and no table is left to gather from (iron_hostple)
            Copy.step(ids, dtype=np.int32)
        return CausalLM.body(self, ids, **values)

    def _moe_states(self) -> MoEStates:
        if getattr(self, "_moe", None) is None:
            c = self.config
            self._moe = MoEStates(c.prefill_chunk, c.hidden_size, c.num_experts)
        return self._moe

    def hwnpu_experts_pool(self):
        """The manifest's experts_pool where the MoE layers' tables have fewer places than experts
        (install(places=...)): experts.bin and, by layer, its tables' buffers and place there (a
        layer without experts, the first dense ones, names none)."""
        if not POOL["layers"]:
            return None
        c = self.config
        moe = [l for l in range(c.num_hidden_layers) if c.moe(l)]
        if sorted(POOL["layers"]) != moe:
            raise SystemExit(f"experts.bin holds layers {sorted(POOL['layers'])} of {moe}")
        none = dict(gu="", dn="", offset=0, gu_bytes=0, dn_bytes=0)
        return dict(file="experts.bin", experts=int(c.num_experts), places=int(POOL["places"]),
                    layers=[dict(gu=f"model.layers.{l}.mlp.gu", dn=f"model.layers.{l}.mlp.down", **POOL["layers"][l])
                            if l in POOL["layers"] else none for l in range(c.num_hidden_layers)])

    def program_rows(self, kernels) -> list:
        """Fill each MoE layer's MoERouteSigmoid program from the decode step's commands (the
        manifest's ``kernels``): command B of layer l is the one whose words are layer l's offsets.
        Returns the manifest's ``fed``: by such command, the feedback buffer its scratchpad is."""
        images = dict(self._fed)
        fed = []
        for layer in self.layers:
            m = layer.mlp
            if not isinstance(m, SparseMoe):
                continue
            l = m.index
            found = [(k, kern["parameters"]) for k, kern in enumerate(kernels)
                     if any((p.get("form") or {}).get("value", "").startswith(f"e{l}_") for p in kern["parameters"])]
            if len(found) != 1:
                raise SystemExit(f"layer {l}: {len(found)} commands read its experts' offsets")
            k, words = found[0]
            prog = np.zeros((SLOTS, 4), np.int32)
            prog[:, 0] = -1
            for p in words:
                f = p.get("form")
                if not f or not f["value"].startswith(f"e{l}_"):
                    raise SystemExit(f"layer {l}: command {k} reads {p['name']}, which no routing gives")
                plain = (f["down"], f["mul"], f["add"], f["post"], f["lo"], f["hi"], f["mod"]) == (0, 1, 0, 0, None, None, None)
                if not plain or p["index"] >= SLOTS:
                    raise SystemExit(f"layer {l}: {p['name']} at {p['index']} is {f}, not id * scale + bias")
                prog[p["index"]] = (int(f["value"].split("_")[1]), f["scale"], f["bias"],
                                    2 if p["kind"] == "core" else 0)
            np.asarray(m.program).reshape(-1)[: prog.size] = prog.reshape(-1)  # the bias after
            fed.append(dict(command=k, buffer=images[l], words=len(words)))
        return fed

    def quantize(self, fmts: dict) -> None:
        """Put the loaded projections in ``fmts``' formats (``install``); the experts are q4nx."""
        import iron_q4

        def q(holder, fmt):
            if fmt != "bf16":
                holder.weight = iron_q4.q_param(np.asarray(holder.weight, np.float32), fmt)

        for layer in self.layers:
            if layer.conv is not None:
                for lin in (layer.conv.in_proj, layer.conv.out_proj):
                    q(lin, fmts["conv"])
            else:
                a = layer.self_attn
                for lin in (a.qkv, a.o_proj):
                    q(lin, fmts["attn"])
            if isinstance(layer.mlp, SparseMoe):
                continue
            mlp = layer.mlp
            if fmts["mlp"] != "bf16":
                iron_q4.install_glu()
                mlp.gu = iron_q4.glu_param(np.asarray(mlp.gate_proj.weight, np.float32),
                                           np.asarray(mlp.up_proj.weight, np.float32), fmts["mlp"])
                mlp.gate_proj = mlp.up_proj = None
                q(mlp.down_proj, fmts["mlp"])
        # the head its own quantized copy (tied): the embeddings stay bf16
        q(self.lm_head, fmts["head"])

    def embed_on_host(self) -> None:
        """The token embeddings to the host (iron_hostple's embed.bin)."""
        import iron_hostple

        c, m = self.config, self.model
        embed = np.asarray(m.embed_tokens.weight).astype(bfloat16)
        iron_hostple._TABLES.setdefault(id(self), {})[iron_hostple.EMBED] = embed
        m.embed_tokens.weight = iron.param(np.zeros((1, c.hidden_size), bfloat16))  # no longer read
        self.embed_rows = iron.state((c.prefill_chunk, c.hidden_size), bfloat16, name=iron_hostple.EMBED)
        self.host_embed = True

    def lookup(self, ids):
        # The embeddings from embed_rows, which the host filled with the call's tokens in order.
        if not self.host_embed:
            return super().lookup(ids)
        shape = getattr(ids, "shape", ())
        if len(shape) == 0:
            return Copy.step(self.embed_rows[:1])
        bound = getattr(ids, "valid", None) or {}
        return Copy.step(self.embed_rows[: bound[0]] if 0 in bound else self.embed_rows[: shape[0]])


def moe_class(base, layers: list):
    """``base`` whose body takes the 4 experts of each MoE layer of ``layers`` as scratchpad values."""
    from iron import Carried, Scratchpad

    # each MoE layer's experts of a decode step, then a prompt chunk's rows form's (a pass's
    # rows and their experts) and gathered form's (a pass's slots' experts), whatever the layer:
    # the host gives them for each run
    names = ([f"e{l}_{i}" for l in layers for i in range(TOP)] + ROW_VALUES + [n for row in EXPERT_VALUES for n in row]
             + GATHER_VALUES)
    params = ", ".join(f"{n}: Scratchpad[np.int32] = np.int32(0)" for n in names)
    src = (
        "def body(self, ids=None, *, token: Carried[np.int32], position: Carried[np.int32], "
        "start: Scratchpad[np.int32] = np.int32(0), rows: Scratchpad[np.int32] = np.int32(1), "
        f"rope_start: Scratchpad[np.int32] = np.int32(0), {params}):\n"
        "    values = dict(" + ", ".join(f"{n}={n}" for n in names) + ")\n"
        "    return self.moe_body(values, ids=ids, token=token, position=position, start=start, "
        "rows=rows, rope_start=rope_start)\n"
    )
    ns = dict(Carried=Carried, Scratchpad=Scratchpad, np=np)
    exec(src, ns)
    return type(base.__name__, (base,), dict(body=ns["body"], _values_made=frozenset(names),
                                             _moe_layers=tuple(layers)))


if __name__ == "__main__":
    lm.main(Lfm2Moe)
