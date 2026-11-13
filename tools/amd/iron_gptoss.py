"""gpt-oss (``gpt_oss``: openai/gpt-oss-20b) on IRON's shared decoder (``iron.lm``).

An application of amd/IRON kept in hwnpu (IRON has none for gpt-oss), exported by
``iron_export_lm.py gptoss``. The structure follows transformers' modeling_gpt_oss.py (checked
against it in tools/ref/ref_gptoss.py):

- 24 layers, ``sliding_attention`` (a window of 128 positions, its own included) and
  ``full_attention`` alternating; 64 query heads and 8 KV heads of 64; q, k, v and o with
  biases; YaRN RoPE (rotate_half, the table times its attention_scaling); a learned sink logit
  per head in each softmax (iron_sinks.py, on the device at no cost).
- RMSNorm times its weight (no ``1 +``), eps 1e-5.
- 32 experts, 4 a token: the router (a linear with a bias), softmax over the 4 chosen logits;
  each expert's gate and up (interleaved rows in the checkpoint), gate clamped at 7, up to
  [-7, 7], ``(up + 1) * gate * sigmoid(1.702 gate)``, down with its bias. The experts stay MXFP4
  exactly (iron_gptoss_moe.py's mx4); routed on the device in the decode step (iron_moe's
  MoERoute, as Qwen3.6: a layer is two commands), on the host in a prompt chunk (MoEHost, and
  iron_gptoss_moe's group operators streaming every expert's weights once a pass).

The residual is carried 3072 wide (2880 and 192 zeros): 48 column blocks of 64 divide over the 8
columns and the quantized GEMVs take in-features in multiples of 512. Each RMSNorm's mean is over
2880 of the 3072: its weight times sqrt(2880 / 3072) and its eps times 2880 / 3072 give the
same output (``pad_tree``). The head's rows are padded to 201216 (whole blocks a column) and the
logits cut back to 201088.
"""

import dataclasses
import functools
import math
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

import iron
import iron.functional as F
from iron import Precision, lm
from iron.lm import Attention, CausalLM, DecoderConfig, DecoderLayer, Linear
from iron.lm.blocks import KVCache
from iron.operators.copy import Copy

import iron_gptoss_moe as moe
import iron_moe
import iron_moeprobe
import iron_sinks

HIDDEN, HP = 2880, 3072

# The experts' MXFP4 blocks and scales are uint8 tensors, which IRON's checkpoint reader lacks.
from iron.lm.checkpoint import _Shard  # noqa: E402

_Shard.DTYPES.setdefault("U8", np.uint8)
HEAD_ROWS = 512  # the head's rows a multiple of this: 64-row blocks whole on 8 columns

# install()'s formats (iron_q4's): the attention's projections and the head. Empty: bf16.
QUANT: dict[str, str] = {}
# The decode step's q, k, v in one operator with the input norm and RoPE (iron_q4's QGEMVQKV):
# q's and k's rows in interleaved pairs (RoPE of Meta's form in the prompt chunk too; q and k
# turned alike, the scores are the same)
FUSE_QKV = os.environ.get("GPTOSS_FUSE_QKV", "1") == "1"
# The decode step's layers as phases of one array (iron_gptoss_layer.py's GptOssStep): configured
# once, where an operator of its own is loaded again each time it follows another
RESIDENT = os.environ.get("GPTOSS_RESIDENT", "1") == "1"


def install(attn: str = "q8", head: str = "q4") -> None:
    """Build gpt-oss with its attention projections and head in these formats (``q4``, ``q6``,
    ``q8`` or ``bf16``), F.linear running iron_q4's on them. The experts are mx4 either way."""
    import iron_q4

    for part, fmt in dict(attn=attn, head=head).items():
        if fmt not in ("q4", "q6", "q8", "bf16"):
            raise ValueError(f"{part} in {fmt}: q4, q6, q8 or bf16")
    iron_q4.install_linear()
    QUANT.update(attn=attn, head=head)


def yarn(c: dict):
    """transformers' YaRN (modeling_rope_utils._compute_yarn_parameters) in numpy: the inverse
    frequencies (float32, as torch's) and the attention factor."""
    r = c["rope_scaling"]
    base, dim = float(c["rope_theta"]), int(c["head_dim"])
    factor = float(r["factor"])
    orig = int(r["original_max_position_embeddings"])
    beta_fast, beta_slow = r.get("beta_fast") or 32, r.get("beta_slow") or 1
    attention = r.get("attention_factor")
    if attention is None:
        attention = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0

    def corr(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = corr(beta_fast), corr(beta_slow)
    if r.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - np.float32(low)) / np.float32(high - low), 0, 1)
    pos = np.power(np.float32(base), np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)).astype(np.float32)
    extra = (1.0 / pos).astype(np.float32)
    inter = (1.0 / (np.float32(factor) * pos)).astype(np.float32)
    ef = (1 - ramp).astype(np.float32)
    inv = (inter * (1 - ef) + extra * ef).astype(np.float32)
    return inv, float(attention)


@dataclasses.dataclass(frozen=True)
class GptOssConfig(DecoderConfig):
    """``DecoderConfig`` with gpt-oss's layer kinds and experts; hidden_size is the padded width."""

    layer_types: tuple[str, ...] = dataclasses.field(kw_only=True, metadata={"convert": tuple})
    num_local_experts: int = dataclasses.field(default=32, kw_only=True)
    num_experts_per_tok: int = dataclasses.field(default=4, kw_only=True)
    swiglu_limit: float = dataclasses.field(default=7.0, kw_only=True)
    real_hidden: int = lm.not_json(HIDDEN, kw_only=True)
    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)

    @classmethod
    def from_json(cls, data: dict, source="config.json") -> "GptOssConfig":
        """``DecoderConfig.from_json`` on gpt-oss's config, hidden_size padded to 3072.

        Raises:
            ValueError: ``data`` asks for what is not built here."""
        if data.get("model_type") != "gpt_oss":
            raise ValueError(f"{source}: model_type {data.get('model_type')!r} is not gpt_oss")
        if (data.get("num_local_experts"), data.get("num_experts_per_tok", data.get("experts_per_token"))) != (32, 4):
            raise ValueError(f"{source}: experts other than 4 of 32 are not built here")
        if data.get("hidden_size") != HIDDEN or data.get("intermediate_size") != moe.INTER:
            raise ValueError(f"{source}: hidden and intermediate sizes other than 2880 are not built here")
        if (data.get("rope_scaling") or {}).get("rope_type") != "yarn":
            raise ValueError(f"{source}: RoPE other than YaRN is not built here")
        if (data.get("num_attention_heads"), data.get("num_key_value_heads"), data.get("head_dim")) != (64, 8, 64):
            raise ValueError(f"{source}: attention other than 64 heads over 8 of 64 is not built here")
        kinds = set(data.get("layer_types", ())) - {"sliding_attention", "full_attention"}
        if kinds:
            raise ValueError(f"{source}: layer kinds {sorted(kinds)} are not built here")
        inv, attention = yarn(data)
        text = {k: v for k, v in data.items() if k not in ("quantization_config", "rope_scaling")}
        # The mean of a norm over 2880 of the padded 3072 (module docstring): eps scales with it.
        text.update(hidden_size=HP, rms_norm_eps=float(data["rms_norm_eps"]) * HIDDEN / HP,
                    intermediate_size=moe.INTER, tie_word_embeddings=False)
        c = super().from_json(text, source)
        object.__setattr__(c, "rope_scaling", lambda _f, inv=inv: inv.astype(np.float64))
        object.__setattr__(c, "rope_attention_scaling", attention)
        return c

    def __post_init__(self):
        super().__post_init__()
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"GptOssConfig: {len(self.layer_types)} layer_types for {self.num_hidden_layers} layers")

    def sliding(self, i: int) -> bool:
        return self.layer_types[i] == "sliding_attention"


def pad_tree(weights):
    """The checkpoint's tensors with the residual's dimension padded to 3072 (module docstring);
    the experts' MXFP4 and the sinks as they are."""
    from iron.lm.checkpoint import leaves, nest

    from iron.common.graph.compiled import param

    norm_scale = np.float32(math.sqrt(HIDDEN / HP))
    out = {}
    for name, leaf in leaves(weights):
        if ".mlp.experts." in name or name.endswith(".sinks"):
            out[name] = leaf
            continue
        a = np.asarray(leaf)
        if name.endswith("layernorm.weight") or name == "model.norm.weight":
            w = np.zeros(HP, np.float32)
            w[:HIDDEN] = a.astype(np.float32) * norm_scale
            a = w.astype(a.dtype)
        elif name.endswith("o_proj.weight"):
            a = np.pad(a, ((0, HP - HIDDEN), (0, 0)))
        elif name.endswith("o_proj.bias"):
            a = np.pad(a, (0, HP - HIDDEN))
        elif a.ndim == 2 and a.shape[1] == HIDDEN:  # q, k, v, the router, the embeddings, the head
            a = np.pad(a, ((0, 0), (0, HP - HIDDEN)))
        out[name] = param(np.ascontiguousarray(a))
    return nest(out, getattr(weights, "_path", ""), getattr(weights, "_file", "<padded>"))


class SinkAttention(Attention):
    """``Attention`` with a fused q, k, v projection (one GEMV, one bias add) and each head's
    ``sinks`` in its softmax (iron_sinks)."""

    def __init__(self, c: GptOssConfig, i: int):
        super().__init__(c, i, o_bias=True)
        self.sinks = iron.param(c.num_attention_heads)
        self.config = c

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        ws = [np.asarray(p.weight) for p in (self.q_proj, self.k_proj, self.v_proj)]
        bs = [np.asarray(p.bias) for p in (self.q_proj, self.k_proj, self.v_proj)]
        self.widths = [w.shape[0] for w in ws]
        self.qkv = Linear(ws[0].shape[1], sum(self.widths), True, precision=self.q_proj.precision)
        self.qkv._bound = True
        self.qkv.weight = iron.param(np.ascontiguousarray(np.concatenate(ws)))
        self.qkv.bias = iron.param(np.ascontiguousarray(np.concatenate(bs)))
        self.sink_values = tuple(float(s) for s in np.asarray(self.sinks, np.float32))
        self.interleaved = False
        self.q_proj = self.k_proj = self.v_proj = None
        self.sinks = None
        self.residual = None  # the decode step's residual, which o_proj's GEMV adds (GptOssLayer)

    def fold(self) -> None:
        """q's and k's rows of each head into interleaved pairs (d and d + D / 2 next to each
        other): QGEMVQKV's table (a q8 code's scale is a row's, so the codes are the same)."""
        H, G, D = self.heads, self.groups, self.head_dim
        w = np.asarray(self.qkv.weight, np.float32)
        b = np.asarray(self.qkv.bias, np.float32).copy()
        perm = np.arange(D).reshape(2, D // 2).T.reshape(-1)  # 0, D/2, 1, D/2 + 1, ...
        rows = np.concatenate([h * D + perm for h in range(H + G)])
        w[: (H + G) * D] = w[rows]
        b[: (H + G) * D] = b[rows]
        dt = np.asarray(self.qkv.weight).dtype
        self.qkv.weight = iron.param(np.ascontiguousarray(w.astype(dt)))
        self.qkv.bias = iron.param(np.ascontiguousarray(b.astype(np.asarray(self.qkv.bias).dtype)))
        self.interleaved = True
        if RESIDENT:
            # GptOssStep's tables: the same weights (as the q8 the prompt chunk reads) in its
            # blocks; o_proj's in-features in the order the attention's cores send them
            import iron_gptoss_layer as gl

            self.qkv_x = iron.param(gl.q8x_table(w, b))
            ow = np.asarray(self.o_proj.weight, np.float32)
            op = np.empty_like(ow)
            op[:, gl.att_order()] = ow
            self.o_x = iron.param(gl.q8x_table(op, np.asarray(self.o_proj.bias, np.float32)))
            self.sinks_x = iron.param(np.asarray(self.sink_values, np.float32).astype(bfloat16))

    def rotate(self, x, angles):
        if not self.interleaved:
            return super().rotate(x, angles)
        from iron.operators.rope import RoPE

        return RoPE.step(x, angles, two_halves=False)

    def body(self, x, cache, call, angles=None):
        if x.shape[0] == 1 and self.residual is not None and angles is not None and QUANT and self.interleaved:
            # x the layer's input (not normed): QGEMVQKV norms it, adds the bias and turns q and k
            import iron_q4

            H, G, D = self.heads, self.groups, self.head_dim
            y = iron_q4.qkv_linear(x, self.qkv.weight, self.qkv.bias, angles[0], self.norm_weight, (H + G) * D,
                                   self.config.eps).reshape(1, sum(self.widths))
            q = y[:, : H * D].reshape(1, H, D)
            k = y[:, H * D : (H + G) * D].reshape(1, G, D)
            v = y[:, (H + G) * D :].reshape(1, G, D)
            o = cache.attend(call, q, k, v, scale=self.scaling)
            return iron_q4.linear_add(o, self.o_proj.weight, self.o_proj.bias, self.residual)
        if x.shape[0] != 1 or self.residual is None or angles is None or not QUANT:
            return super().body(x, cache, call, angles)
        # The decode step in fewer operators: q, k, v one GEMV with its bias added in it; q's and
        # k's 72 rows one RoPE (they are next to each other in its output); o_proj's GEMV adding
        # its bias and the residual (iron_q4's QGEMVAdd)
        import iron_q4

        H, G, D = self.heads, self.groups, self.head_dim
        y = self.qkv(x).reshape(1, sum(self.widths))
        qk = F.rope(y[:, : (H + G) * D].reshape(H + G, D), angles[0])
        q, k = qk[:H], qk[H:]
        v = y[:, (H + G) * D :].reshape(1, G, D)
        o = cache.attend(call, q.reshape(1, H, D), k.reshape(1, G, D), v, scale=self.scaling)
        return iron_q4.linear_add(o, self.o_proj.weight, self.o_proj.bias, self.residual)

    def project(self, x, **gemv) -> list:
        n = 1 if len(x.shape) == 1 else x.shape[0]
        if n > 1:  # a prompt chunk: a GEMM a run of the weight's rows (iron_q4's Linear.split)
            return self.qkv.split(x, *self.widths)
        # the decode step: one GEMV and one bias add, the output cut
        y = self.qkv(x).reshape(1, sum(self.widths))
        out, end = [], 0
        for w in self.widths:
            out.append(y[:, end : end + w])
            end += w
        return out


class SinkCache(KVCache):
    """A layer's ``KVCache`` whose attention takes its layer's sinks (iron_sinks: the MHA classes
    IRON's and iron_window's forms step are the sink classes while it attends)."""

    sinks: tuple | None = None
    kv = None

    def __init__(self, c, window=None):
        super().__init__(c, window)
        if RESIDENT:
            # The decode step's array of the layer's keys and values (GptOssStep reads a group's
            # key and value of a row together; IRON's operators take no such view of one array, so
            # the prompt chunk keeps keys and values, and hwnpu copies a chunk's rows here and a
            # decode step's back: internal/llm/iron's joinKV and splitKV). A chunk of rows more
            # than the context: the attention's objects reach past the call's row
            _, G, D = self.keys.shape
            self.kv = iron.state((c.max_seq_len + c.prefill_chunk, 2, G, D))

    def attend(self, call, q, k, v, scale=None):
        import iron_window
        from iron.lm import blocks

        base_mha, base_window = blocks.MHA, iron_window.operator()
        blocks.MHA = iron_sinks.sink_class(base_mha)
        iron_window._MHA = iron_sinks.sink_class(base_window)
        try:
            with iron_sinks.using(self.sinks):
                return super().attend(call, q, k, v, scale)
        finally:
            blocks.MHA, iron_window._MHA = base_mha, base_window


class GptOssMoe(iron.Graph):
    """gpt-oss's ``mlp``: the ``router`` and the ``experts`` (MXFP4), on the device MoERoute,
    MX4GateUp and MX4Down (module docstring)."""

    def __init__(self, c: GptOssConfig, index: int):
        E, N = c.real_hidden, c.num_local_experts
        self.index, self.n, self.top = index, N, c.num_experts_per_tok
        self.router = Linear(c.hidden_size, N, True, precision=c.precision)
        self.experts = iron.Graph()
        I = c.intermediate_size
        u8 = dict(dtype=np.uint8)
        self.experts.gate_up_proj_blocks = iron.param(N, 2 * I, E // 32, 16, **u8)
        self.experts.gate_up_proj_scales = iron.param(N, 2 * I, E // 32, **u8)
        self.experts.gate_up_proj_bias = iron.param(N, 2 * I)
        self.experts.down_proj_blocks = iron.param(N, E, I // 32, 16, **u8)
        self.experts.down_proj_scales = iron.param(N, E, I // 32, **u8)
        self.experts.down_proj_bias = iron.param(N, E)
        self.precision = c.precision

    @staticmethod
    def codes(blocks):
        b = np.asarray(blocks, np.uint8)
        E, N, G, B = b.shape
        return np.stack([b & 15, b >> 4], -1).reshape(E, N, G * 2 * B)

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        x = self.experts
        gc, ge = self.codes(x.gate_up_proj_blocks), np.asarray(x.gate_up_proj_scales)
        gb = np.asarray(x.gate_up_proj_bias, np.float32)
        dc, de = self.codes(x.down_proj_blocks), np.asarray(x.down_proj_scales)
        db = np.asarray(x.down_proj_bias, np.float32)
        gu = np.concatenate([moe.gate_up_table(gc[e], ge[e], gb[e]) for e in range(self.n)])
        dn = np.concatenate([moe.down_table(dc[e], de[e], db[e]) for e in range(self.n)])
        self.gu = iron.param(gu.reshape(-1, moe.BLOCK))
        self.dn = iron.param(dn.reshape(-1, moe.BLOCK))
        self.gu_bytes, self.dn_bytes = gu.size // self.n, dn.size // self.n
        # the router's 32 rows padded to a column block (MoERoute's logit 32, a shared gate it has
        # none of, is 0), its bias likewise
        rows = 64
        w = np.zeros((rows, np.asarray(self.router.weight).shape[1]), bfloat16)
        w[: self.n] = np.asarray(self.router.weight)
        b = np.zeros(rows, bfloat16)
        b[: self.n] = np.asarray(self.router.bias)
        self.router_w, self.router_b = iron.param(w), iron.param(b)
        # MoERoute's program, then the router's bias (bf16) for the decode step's MoERoute to add
        program = np.zeros(4 * iron_moe.SLOTS + self.n // 2, np.int32)
        program[4 * iron_moe.SLOTS :] = np.ascontiguousarray(b[: self.n]).view(np.int32)
        self.program = iron.param(program)
        if RESIDENT:
            import iron_gptoss_layer as gl

            self.router_x = iron.param(gl.router_table(np.asarray(w, np.float32)[: self.n], np.asarray(b, np.float32)[: self.n]))
            self.prog_x = iron.param(np.zeros(gl.PROG, bfloat16))  # the routing's program (program_rows)
        self.router = self.experts = None

    def body(self, hn, h, call):
        if call.prompt:
            return self.prompt_body(hn, h, call)
        Route = iron_moe.route_operator()
        # the router's bias added in MoERoute (an add the fewer)
        logits = F.linear(hn, self.router_w, precision=self.precision)
        n = self.router_w.shape[0]
        w, image = Route.step(logits.reshape(n), self.program, n=n, experts=self.n, top=self.top,
                              picked=self.index, bias=True)
        call.feedback.append((self.index, image.name))
        iron.on_host(lambda: None)  # command B: the experts, at the offsets the image holds
        GU, D, EX = moe_operators()
        e = call.experts[self.index]
        # MX4Experts: the gate-up and the down in one operator (their streams' start and end once)
        y = EX.step(self.gu, self.dn, hn.reshape(HP), h.reshape(HP), w, grows=self.gu.shape[0], drows=self.dn.shape[0],
                    picked=self.index, **{f"e{i}": e[i] * self.gu_bytes for i in range(self.top)},
                    **{f"d{i}": e[i] * self.dn_bytes for i in range(self.top)})
        out = y[moe.TOP * moe.IPAD :]
        iron.on_host(lambda: None)
        return out.reshape(1, HP)

    def prompt_body(self, hn, h, call):
        """A prompt chunk's experts, routed on the host: the router's logits on the device; the
        host's route (moe_route_mx4: the counts and x's stream); MX4GroupGateUp and MX4GroupDown
        over every expert's rows; the host's combine into ``call.moe.out`` (the residual plus each
        choice's weight times its expert's output row).

        IRON's arena does not count a host step's reads as uses: what the host reads is the step's
        last output or a state (h is copied into one)."""
        moe, (C, _) = call.moe, hn.shape
        Copy.step(h, moe.h)
        # hn into a state as well: its last device use is the router's, so the arena may put the
        # logits (or what follows) over it before the host reads it (24 layers: cosine 0.98)
        Copy.step(hn, moe.hn)
        logits = F.linear(moe.hn, self.router_w, self.router_b, precision=self.precision)
        iron.on_host(MoEHost("moe_route_mx4", self.index, self.n, self.top, C, moe), logits, moe.hn,
                     writes=(moe.counts, moe.xs))
        GU, D = group_operators()
        # each slot's expert (the graph's slot<i>, which hwnpu's route sets each pass): its tables' offsets
        slots = call.slots  # the overflow slots' (slot e < 32 is expert e's)
        a = GU.step(self.gu, moe.counts, moe.xs, rows=self.gu.shape[0],
                    **{f"w{i}": s * self.gu_bytes for i, s in slots.items()})
        y = D.step(self.dn, moe.counts, a, rows=self.dn.shape[0],
                   **{f"w{i}": s * self.dn_bytes for i, s in slots.items()})
        iron.on_host(MoEHost("moe_combine", self.index, self.n, self.top, C, moe), moe.h, y,
                     writes=(moe.out,))
        return moe.out[: call.rows]


@functools.cache
def group_operators():
    return moe.group_operators()


class MoEStates:
    """A prompt chunk's states the host writes for every layer's experts (one set: the layers run
    in turn): the counts and x's stream (route), the layer's output (combine), the residual."""

    def __init__(self, C: int):
        self.counts = iron.state((moe.SB * moe.KT,), bfloat16, name="moe_counts")
        self.xs = iron.state((moe.SLOTS * moe.HIDDEN * moe.CAP,), bfloat16, name="moe_xs")
        self.out = iron.state((C, HP), bfloat16, name="moe_out")
        self.h = iron.state((C, HP), bfloat16, name="moe_h")
        self.hn = iron.state((C, HP), bfloat16, name="moe_hn")  # the experts' input, for route
        self.plans = {}


class MoEHost:
    """A host step of a prompt chunk's experts: ``hwnpu`` names it for hwnpu (which runs its own
    in Go: internal/llm/iron's moeHostRouteMX4 and moeHostCombine); called, the reference's."""

    def __init__(self, kind: str, layer: int, experts: int, top: int, C: int, states: MoEStates):
        self.kind, self.layer, self.experts, self.top, self.C, self.states = kind, layer, experts, top, C, states
        self.hwnpu = dict(kind=kind, layer=layer, experts=experts, top=top, cap=moe.CAP, tokens=C,
                          in_features=moe.HIDDEN, kt=moe.KT, slots=moe.SLOTS,
                          slot_values=[""] * moe.EXPERTS + [f"slot{i}" for i in range(moe.EXPERTS, moe.SLOTS)])

    def __call__(self, *args):
        getattr(self, self.kind)(*args)

    def moe_route_mx4(self, logits, hn, counts, xs):
        lg = np.asarray(logits, np.float32)
        x = np.asarray(hn)[:, : moe.HIDDEN]
        n = lg.shape[0]
        ids = np.argsort(-lg[:, : self.experts], axis=1, kind="stable")[:, : self.top]
        v = np.take_along_axis(lg, ids, 1)
        w = np.exp(v - v[:, :1])
        w = (w / w.sum(1, keepdims=True)).astype(bfloat16).astype(np.float32)
        # one pass's slots (the reference's: hwnpu's route splits a chunk needing more into passes)
        plan = moe.plan_slots(np.bincount(ids.reshape(-1), minlength=self.experts))
        rows = np.zeros((moe.SLOTS, moe.CAP, moe.HIDDEN), bfloat16)
        seen = np.zeros(self.experts, np.int64)
        slot = np.zeros((n, self.top), np.int64)
        where = {(e, r): s for s, (e, r, c) in enumerate(plan) if c}  # (expert, first row) -> slot
        for t in range(n):
            for k in range(self.top):
                e = ids[t, k]
                s, r = where[(e, seen[e] // moe.CAP * moe.CAP)], seen[e] % moe.CAP
                rows[s, r] = x[t]
                slot[t, k] = s * moe.CAP + r
                seen[e] += 1
        counts[:] = moe.counts_object([c for _, _, c in plan])
        xs[:] = moe.stream_x(rows).reshape(-1)
        self.states.plans[self.layer] = (slot, w)

    def moe_combine(self, h, y, out):
        slot, w = self.states.plans.pop(self.layer)
        n = len(slot)
        yy = np.asarray(y, np.float32).reshape(-1, HP)
        acc = np.asarray(h, np.float32)[:n].copy()
        for k in range(self.top):
            acc += w[:, k : k + 1] * yy[slot[:, k]]
        out[:n] = acc.astype(bfloat16)
        out[n:] = 0


@functools.cache
def moe_operators():
    return moe.operators()


class GptOssLayer(DecoderLayer):
    """Layer ``i``: sink attention (windowed on a sliding layer), then the experts."""

    def __init__(self, c: GptOssConfig, i: int):
        cache = SinkCache(c, c.window if c.sliding(i) else None)
        super().__init__(c, i, SinkAttention(c, i), iron.Graph(), cache)
        self.mlp = GptOssMoe(c, i)

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        self.cache.sinks = self.self_attn.sink_values
        if FUSE_QKV and QUANT:
            self.self_attn.fold()
            self.self_attn.norm_weight = self.input_layernorm.weight

    def body(self, x, call):
        decode = not call.prompt and x.shape[0] == 1 and bool(QUANT)
        # the decode step: the residual added in o_proj's GEMV (SinkAttention.body)
        self.self_attn.residual = x if decode else None
        # QGEMVQKV norms the decode step's input itself
        fused = decode and self.self_attn.interleaved
        h = self.mix(x if fused else self.input_layernorm(x), call)
        self.self_attn.residual = None
        if not decode:
            h = x + h
        return self.mlp(self.post_attention_layernorm(h), h, call)


class GptOss(CausalLM):
    """gpt-oss on ``config``'s shape and a checkpoint's ``weights`` (module docstring): the decode
    step and a prompt chunk (its experts routed on the host), the body taking each layer's chosen experts' offsets (``e<layer>_<i>``), which
    command B's scratchpad gets from the layer's MoERoute (``hwnpu_fed``)."""

    config = GptOssConfig
    layer_class = GptOssLayer
    hwnpu_name = "GptOss"

    def __init__(self, config: GptOssConfig, weights):
        super().__init__(config, pad_tree(weights))
        self.vocab = config.vocab_size
        head = np.asarray(self.lm_head.weight)
        rows = -(-head.shape[0] // HEAD_ROWS) * HEAD_ROWS
        if rows != head.shape[0]:
            self.lm_head.weight = iron.param(np.ascontiguousarray(np.pad(head, ((0, rows - head.shape[0]), (0, 0)))))
        if QUANT:
            self.quantize(QUANT)
        self._rw0 = iron.param(np.zeros(16, bfloat16))  # GptOssStep's first command's routing weights: unread
        self.embed_on_host()

    def quantize(self, fmts: dict) -> None:
        import iron_q4

        def q(holder, name, fmt):
            if fmt != "bf16":
                setattr(holder, name, iron_q4.q_param(np.asarray(getattr(holder, name), np.float32), fmt))

        for layer in self.layers:
            a = layer.self_attn
            q(a.qkv, "weight", fmts["attn"])
            q(a.o_proj, "weight", fmts["attn"])
        q(self.lm_head, "weight", fmts["head"])

    def embed_on_host(self) -> None:
        """The token embeddings to the host (iron_hostple's embed.bin, a call's rows in
        embed_rows), as Qwen3.5's: 201088 x 3072 bf16 is 1.2 GB the device would pin."""
        import iron_hostple

        c, m = self.config, self.model
        embed = np.asarray(m.embed_tokens.weight).astype(bfloat16)
        iron_hostple._TABLES.setdefault(id(self), {})[iron_hostple.EMBED] = embed
        m.embed_tokens.weight = iron.param(np.zeros((1, c.hidden_size), bfloat16))
        self.embed_rows = iron.state((c.prefill_chunk, c.hidden_size), bfloat16, name=iron_hostple.EMBED)

    def lookup(self, ids):
        shape = getattr(ids, "shape", ())
        if len(shape) == 0:
            if RESIDENT and QUANT and FUSE_QKV:  # GptOssStep reads the row where the host puts it
                return self.embed_rows[:1]
            return Copy.step(self.embed_rows[:1])
        bound = getattr(ids, "valid", None) or {}
        return Copy.step(self.embed_rows[: bound[0]] if 0 in bound else self.embed_rows[: shape[0]])

    def head(self, x):
        # The padded head's logits whole: hwnpu reads the first vocab_size (iron.Model.logits),
        # and a cut would be a copy of its own.
        return self.lm_head(self.model.norm(x))

    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        if "e0_0" in getattr(cls, "_values_made", ()):
            return super().from_checkpoint(path, **kwargs)
        from iron.lm.checkpoint import weights_file

        layers = cls.default_config(weights_file(path)).num_hidden_layers
        return moe_class(cls, layers).from_checkpoint(path, **kwargs)

    def draw(self, logits, call):
        # no draw on the device (its commands end at host steps); hwnpu draws from the logits
        return logits, iron.carry(token=call.token + 0, position=call.position + 1)

    @functools.wraps(CausalLM.body)
    def body(self, ids=None, **values):
        if ids is not None:
            # a version's input must be read, and no table is left to gather from (iron_hostple)
            Copy.step(ids, dtype=np.int32)
        return super().body(ids, **values)

    def record(self, **shapes):
        if not shapes or not hasattr(self, "_fed"):
            self._fed = []
        recording = super().record(**shapes)
        if not shapes:
            recording.feedback.extend(name for _, name in self._fed)
        return recording

    def moe_body(self, values, **standard):
        top = self.config.num_experts_per_tok
        experts = [[values[f"e{l}_{i}"] for i in range(top)] for l in range(self.config.num_hidden_layers)]
        slots = {i: values[f"slot{i}"] for i in range(moe.EXPERTS, moe.SLOTS)}
        import iron.lm.causal as causal

        made = causal.Call

        def call(**kw):
            c = made(**kw)
            c.experts, c.feedback, c.slots = experts, self._fed, slots
            if c.prompt:
                if getattr(self, "_moe", None) is None:
                    self._moe = MoEStates(self.config.prefill_chunk)
                c.moe = self._moe
            return c

        causal.Call = call
        try:
            if standard.get("ids") is not None:
                # a version's input must be read, and no table is left to gather from (iron_hostple)
                Copy.step(standard["ids"], dtype=np.int32)
            return CausalLM.body(self, **standard)
        finally:
            causal.Call = made

    def forward(self, x, call):
        if call.prompt or not (RESIDENT and QUANT and FUSE_QKV):
            return super().forward(x, call)
        return self.resident(x, call)

    def resident(self, x, call):
        """The decode step on GptOssStep's array (iron_gptoss_layer.py): a command a layer
        boundary, the experts of the layer before (their offsets the command's scratchpad words,
        written by that layer's routing) then the layer's attention and routing."""
        import iron_gptoss_layer as gl

        gl.unexpanded_loads()
        Step = gl.step_operator()
        c, layers = self.config, self.layers
        L, top = len(layers), c.num_experts_per_tok
        call.embeddings = x
        pos, ROW = call.position, gl.ROW
        h, rw = x.reshape(HP), self._rw0
        ye = None
        for l in range(L + 1):
            prev, cur = layers[max(l - 1, 0)], layers[min(l, L - 1)]
            a, m, pm = cur.self_attn, cur.mlp, prev.mlp
            # the last command: the last layer's experts, then the head (its norm's weight in the
            # place of the input norm's)
            phases = (gl.PH_E if l > 0 else 0) | (gl.PH_Q | gl.PH_A | gl.PH_O | gl.PH_R | gl.PH_T if l < L else gl.PH_H)
            head = self.lm_head.weight.reshape(-1, gl.HB)
            sliding = cur.cache.window is not None
            n = gl.objects(pos)
            if sliding:  # the window's rows: the 127 before the call's and its own
                n = iron.clamp(n, hi=-(-(cur.cache.window + 1) // gl.TOK))
            values = dict(kw=pos * ROW, apos=pos * gl.D, ntok=pos + 1, nslide=iron.clamp(pos + 1, hi=c.window + 1),
                          nkv1=n - 1)
            if sliding:
                values["kfirst"] = iron.clamp(pos * ROW - cur.cache.window * ROW, lo=0)
            if l > 0:
                e = call.experts[l - 1]
                values.update({f"e{i}": e[i] * pm.gu_bytes for i in range(top)})
                values.update({f"d{i}": e[i] * pm.dn_bytes for i in range(top)})
            table = cur.rope.table  # the call's row of the angles is read from the table itself
            _, ye, _, _, yo, _, yw, image, _, yh = Step.step(
                pm.gu, pm.dn, a.qkv_x, a.o_x, m.router_x, head, h, prev.post_attention_layernorm.weight,
                cur.input_layernorm.weight if l < L else self.model.norm.weight,
                cur.post_attention_layernorm.weight, table.reshape(int(np.prod(table.shape))), a.sinks_x,
                rw, m.prog_x, cur.cache.kv.reshape(int(np.prod(cur.cache.kv.shape))), grows=pm.gu.shape[0], drows=pm.dn.shape[0],
                qrows=a.qkv_x.shape[0], orows=a.o_x.shape[0], hrows=head.shape[0], angn=int(np.prod(table.shape)),
                kvn=int(np.prod(cur.cache.kv.shape)), eps=c.eps,
                phases=phases, sliding=sliding, picked=l - 1, routed=l if l < L else -1,
                fed_at=self.fed_at(l), **values)
            if l < L:
                call.feedback.append((l, image.name))
                iron.on_host(lambda: None)  # the next command's words are this one's routing's
            h, rw = yo, yw
        # the logits as GptOssStep's head sends them: two partial sums a row, which hwnpu adds
        self.hwnpu_logit_parts = 2 * gl.MT
        # one array a step, loaded once: hwnpu's barrier before a step (another hardware context,
        # which puts the array's configuration back: 2.4 ms a step) is not needed (design.md)
        self.hwnpu_no_step_barrier = True
        return self.draw(yh, call)

    class Recompile(Exception):
        """The versions are to be compiled again (resident_rows found where the routed words are)."""

    def fed_at(self, l: int) -> int:
        """The first word layer l's routing writes of the next command's scratchpad: IRON places
        them after the words the command shares with others, as many as its phases read, so the
        place is known once compiled. A guess first; resident_rows keeps what the compiled
        commands show and asks for the versions again (iron_export_lm.py)."""
        import iron_gptoss_layer as gl

        return getattr(self, "_fed_first", {}).get(l, gl.FED_AT)

    def program_rows(self, kernels) -> list:
        """As iron_qwen35.Qwen35Moe.program_rows: each layer's MoERoute program from the decode
        step's commands; the manifest's ``fed``."""
        if RESIDENT and QUANT and FUSE_QKV:
            return self.resident_rows(kernels)
        from iron_qwen35 import Qwen35Moe

        return Qwen35Moe.program_rows(self, kernels)

    def resident_rows(self, kernels) -> list:
        """Each layer's routing's program (GptOssStep's T) from the decode step's commands: the
        command reading layer l's experts' offsets takes them as its first words (id * scale +
        bias each), which the routing writes; its other words are the host's."""
        import iron_gptoss_layer as gl

        images = dict(self._fed)
        fed, moved = [], {}
        for layer in self.layers:
            l, m = layer.mlp.index, layer.mlp
            found = [(k, kern["parameters"]) for k, kern in enumerate(kernels)
                     if any((p.get("form") or {}).get("value", "").startswith(f"e{l}_") for p in kern["parameters"])]
            if len(found) != 1:
                raise SystemExit(f"layer {l}: {len(found)} commands read its experts' offsets")
            k, words = found[0]
            prog = np.zeros((gl.PROG // 8, 4), np.int32)
            prog[:, 0] = -1
            routed = [p for p in words if (p.get("form") or {}).get("value", "").startswith(f"e{l}_")]
            at = self.fed_at(l)
            have = sorted(p["index"] for p in routed)
            if have != list(range(have[0], have[0] + gl.FED)):
                raise SystemExit(f"layer {l}: command {k}'s routed words are at {have}, not {gl.FED} in a row")
            if have[0] != at:
                moved[l] = have[0]
                continue
            for p in routed:
                f = p["form"]
                plain = (f["down"], f["mul"], f["add"], f["post"], f["lo"], f["hi"], f["mod"]) == (0, 1, 0, 0, None, None, None)
                if not plain:
                    raise SystemExit(f"layer {l}: {p['name']} is {f}, not id * scale + bias")
                prog[p["index"] - at] = (int(f["value"].split("_")[1]), f["scale"], f["bias"],
                                                2 if p["kind"] == "core" else 0)
            np.asarray(m.prog_x).view(np.int32)[:] = prog.reshape(-1)
            fed.append(dict(command=k, buffer=images[l], words=gl.FED, first=at))
        if moved:
            if getattr(self, "_fed_first", None):
                raise SystemExit(f"the routed words moved again: {moved}")
            self._fed_first = moved
            raise self.Recompile(f"the routed words' places: {moved}")
        return fed


def moe_class(base, layers: int, top: int = 4):
    """``base`` whose body takes the ``top`` offsets of each of ``layers`` layers as scratchpad values."""
    from iron import Carried, Scratchpad

    names = [f"e{l}_{i}" for l in range(layers) for i in range(top)] + [f"slot{i}" for i in range(moe.EXPERTS, moe.SLOTS)]
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
    return type(base.__name__, (base,), dict(body=ns["body"], _values_made=frozenset(names)))


if __name__ == "__main__":
    lm.main(GptOss)
