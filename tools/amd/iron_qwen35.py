"""Qwen3.5 / Qwen3.6 text models (``qwen3_5``, ``qwen3_5_moe``) on IRON's shared decoder (``iron.lm``).

An application of amd/IRON kept in hwnpu (IRON has none for Qwen3.5), exported by
``iron_export_lm.py qwen35``. Its shape is read from the checkpoint's ``config.json`` (the
root, or its ``text_config``); the structure follows transformers' modeling_qwen3_5.py:

- ``layer_types``: ``linear_attention`` layers are Gated DeltaNet (``GatedDeltaNet``),
  ``full_attention`` ones gated attention (``GatedAttention``).
- Gated DeltaNet: ``in_proj_qkv``, ``in_proj_a`` and ``in_proj_b`` are one projection here,
  its rows in iron_gdn's column order (``gdn_layout``), through one depth-wise causal
  convolution of ``linear_conv_kernel_dim`` taps (a and b pass it unchanged: their taps are
  0, 0, 0, 1), then iron_gdn's GatedDeltaRule (silu on q, k and v, the l2 norms, the decay
  and the delta rule), its float32 state a slot per prompt-chunk boundary (``Checkpoints``).
  Its output is RMS-normed per value head (``norm``, a plain weight), times ``silu(z)``
  (``in_proj_z``), and ``out_proj``.
- Gated attention: ``q_proj`` holds each head's query then its gate (``2 head_dim`` rows a
  head); q and k are RMS-normed per head; RoPE turns the first ``partial_rotary_factor`` of
  each head (rotate_half over those columns; text positions, so mrope is plain RoPE); the
  output times ``sigmoid(gate)`` before ``o_proj``.
- Every RMSNorm but the gated one scales by ``1 + weight`` (``RMSNorm.plus_one``).
- The feed-forward is a SwiGLU MLP (``qwen3_5``), or for ``qwen3_5_moe`` (Qwen3.6-35B-A3B,
  ``Qwen35Moe``) 8 of 256 experts and a shared one gated by ``sigmoid(shared_expert_gate)``,
  routed on the device (iron_moe.py): a layer is two commands of the decode step, and the
  experts are q4nx tables read in place. The MoE model has a decode step only; hwnpu runs a
  prompt through it a token at a time (the prompt chunk's MoE, experts by token, is not
  built yet).
"""

import dataclasses
import functools

import numpy as np
from ml_dtypes import bfloat16

import iron
import iron.functional as F
from iron import Precision, lm
from iron.lm import (
    Attention,
    CausalConv1d,
    CausalLM,
    Checkpoints,
    DecoderConfig,
    DecoderLayer,
    Linear,
    RMSNorm,
)
from iron.lm.checkpoint import renamed
from iron.operators.copy import Copy
from iron.operators.sigmoid import Sigmoid

import iron_gdn
import iron_moe
import iron_verify
import iron_moeprobe
import iron_moep

COLUMNS = 8  # the columns GatedDeltaRule spreads over (npu2)
# Qwen3.6 with a prompt chunk (its experts routed on the host, iron_moep.py); without it the
# export is the decode step alone, the prompt run a token at a time.
PROMPT = True

# A dense model's prompt commands: IRON cuts a command when its scratchpad words would pass this
# (packaging.SCRATCHPAD_WORDS, 32), and nothing else bounds its control code. Qwen3.5-2B's q4
# chunk came out as 3 commands, the second over the 64 MiB device heap (CREATE_BO: EINVAL);
# PLaMo 3's and LFM3's 24 (iron_hostple's) cut it shorter. Qwen3.6's routed decode step keeps 32
# (iron_moe.SLOTS).
DENSE_SCRATCHPAD_WORDS = 24


def install_dense() -> None:
    """Cut a dense Qwen3.5's commands at DENSE_SCRATCHPAD_WORDS scratchpad words."""
    from iron.common.image import packaging

    packaging.SCRATCHPAD_WORDS = min(packaging.SCRATCHPAD_WORDS, DENSE_SCRATCHPAD_WORDS)


# install()'s formats (iron_q4's), by part: the mixers' projections (Gated DeltaNet's and the
# attention's), a dense MLP's, the head. Empty: all bf16.
QUANT: dict[str, str] = {}


def install(attn: str = "q8", mlp: str = "q4", head: str = "q4") -> None:
    """Build Qwen3.5 with its projections in these formats ("q4", "q6", "q8" or "bf16"), F.linear
    running iron_q4's on them. The router and the experts' tables are as they are without it."""
    import iron_q4

    for part, fmt in dict(attn=attn, mlp=mlp, head=head).items():
        if fmt not in ("q4", "q6", "q8", "bf16"):
            raise ValueError(f"{part} in {fmt}: q4, q6, q8 or bf16")
    iron_q4.install_linear()
    QUANT.update(attn=attn, mlp=mlp, head=head)


@dataclasses.dataclass(frozen=True)
class Qwen35Config(DecoderConfig):
    """``DecoderConfig`` with Qwen3.5's layer kinds, its Gated DeltaNet's shape and partial RoPE."""

    rope_theta: float = dataclasses.field(metadata={"json": "rope_parameters.rope_theta"})
    partial_rotary_factor: float = dataclasses.field(
        default=1.0, kw_only=True, metadata={"json": "rope_parameters.partial_rotary_factor"})
    layer_types: tuple[str, ...] = dataclasses.field(kw_only=True, metadata={"convert": tuple})
    linear_conv_kernel_dim: int = dataclasses.field(default=4, kw_only=True)
    linear_key_head_dim: int = dataclasses.field(default=128, kw_only=True)
    linear_value_head_dim: int = dataclasses.field(default=128, kw_only=True)
    linear_num_key_heads: int = dataclasses.field(default=16, kw_only=True)
    linear_num_value_heads: int = dataclasses.field(default=32, kw_only=True)
    num_experts: int = dataclasses.field(default=0, kw_only=True)
    num_experts_per_tok: int = dataclasses.field(default=8, kw_only=True)
    moe_intermediate_size: int = dataclasses.field(default=0, kw_only=True)
    shared_expert_intermediate_size: int = dataclasses.field(default=0, kw_only=True)
    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)

    @classmethod
    def from_json(cls, data: dict, source="config.json") -> "Qwen35Config":
        """``DecoderConfig.from_json`` on the text model's object (the root or ``text_config``).

        Raises:
            ValueError: ``data`` asks for what is not built: another model, RoPE other than
                the default, attention biases, or experts.
        """
        text = data.get("text_config", data)
        kind = text.get("model_type")
        if kind not in ("qwen3_5_text", "qwen3_5_moe_text", "qwen3_5", "qwen3_5_moe"):
            raise ValueError(f"{source}: model_type {kind!r} is not Qwen3.5")
        rope = text.get("rope_parameters", {})
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"{source}: RoPE {rope['rope_type']!r} is not built here")
        if text.get("attention_bias"):
            raise ValueError(f"{source}: attention biases are not built here")
        if not text.get("attn_output_gate", True):
            raise ValueError(f"{source}: attention without its output gate is not built here")
        kinds = set(text.get("layer_types", ())) - {"linear_attention", "full_attention"}
        if kinds:
            raise ValueError(f"{source}: layer kinds {sorted(kinds)} are not built here")
        if text.get("num_experts", 0):
            if (text.get("num_experts_per_tok"), text.get("num_experts")) != (8, 256):
                raise ValueError(f"{source}: experts other than 8 of 256 are not built here")
            if text.get("moe_intermediate_size") != 512 or text.get("shared_expert_intermediate_size") != 512:
                raise ValueError(f"{source}: experts of other than 512 are not built here")
            # DecoderConfig's intermediate_size, which a dense layer would read
            text = {**text, "intermediate_size": text["moe_intermediate_size"]}
        return super().from_json(text, source)

    def __post_init__(self):
        super().__post_init__()
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"Qwen35Config: {len(self.layer_types)} layer_types for "
                             f"{self.num_hidden_layers} layers")
        if self.partial_rotary_factor != 1.0:
            object.__setattr__(self, "rotary_dim", int(self.head_dim * self.partial_rotary_factor))

    def linear(self, i: int) -> bool:
        """Whether layer ``i`` is Gated DeltaNet."""
        return self.layer_types[i] == "linear_attention"


class GatedNorm(iron.Graph):
    """transformers' Qwen3_5RMSNormGated: each value head's rows RMS-normed, times ``weight``
    (no ``1 +``), times ``silu(z)``."""

    def __init__(self, dim: int, eps: float):
        self.dim, self.eps = dim, eps
        self.weight = iron.param(dim)

    def body(self, y, z):
        n, width = y.shape
        h = F.rms_norm(y.reshape(n * width // self.dim, self.dim), (self.dim,), self.weight, self.eps)
        return h.reshape(n, width) * F.silu(z)


class GatedDeltaNet(iron.Graph):
    """Qwen3.5's linear attention, as the checkpoint's ``linear_attn`` names it (module docstring)."""

    def __init__(self, c: Qwen35Config, fold: bool = False, prompt: bool = False):
        E = c.hidden_size
        H, KH = c.linear_num_value_heads, c.linear_num_key_heads
        # fold (a model with a decode step only): the short convolution and the gated norm in
        # GatedDeltaRule (iron_gdn's conv and norm), the convolution's rows before the call in
        # the state's first pair of each column, the norm's weight in out_proj. With prompt too
        # (a prompt chunk besides): the chunk runs IRON's convolution on a state of its own and
        # the plain operator past each column's pair (iron_gdn's pairs), and the host copies the
        # convolution's rows between the two around the chunk (hwnpu's conv_pairs).
        self.fold, self.prompt = fold, prompt
        dk, dv, K = c.linear_key_head_dim, c.linear_value_head_dim, c.linear_conv_kernel_dim
        self.shape = (H, KH, dk, dv)
        self.qkv = 2 * KH * dk + H * dv
        self.seg = iron_gdn.column_widths(H, KH, dk, dv, COLUMNS)
        self.in_proj_qkv = Linear(E, self.qkv, False, precision=c.precision)
        self.in_proj_a = Linear(E, H, False, precision=c.precision)
        self.in_proj_b = Linear(E, H, False, precision=c.precision)
        self.in_proj_z = Linear(E, H * dv, False, precision=c.precision)
        self.conv1d = iron.Graph()
        self.conv1d.weight = iron.param(self.qkv, 1, K)
        self.dt_bias = iron.param(H)
        self.A_log = iron.param(H)
        self.norm = GatedNorm(dv, c.eps)
        self.out_proj = Linear(H * dv, E, False, precision=c.precision)
        self.config = c  # the fused projection and convolution are made once loaded
        extra = COLUMNS * 2 * dk * iron_gdn.SLICE if fold else 0
        self.state = Checkpoints.of((H * dk * dv + extra,), c.max_seq_len, c.prefill_chunk, np.float32)
        self.words = None
        R = iron_verify.ROWS["rows"]
        if R and fold:
            # the version checking drafts (iron_verify): the state after each of its rows but the
            # last, from which the host puts back the one after the last draft kept
            self.rows_state = iron.state((R - 1, H * dk * dv + extra), np.float32)

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        c = self.config
        H, KH, dk, dv = self.shape
        W = COLUMNS * iron_gdn.column_widths(H, KH, dk, dv, COLUMNS)  # with the columns' padding
        # The fused projection and convolution, made from the checkpoint's own (bound here, so
        # that nothing reads them from the checkpoint).
        self.in_proj = Linear(c.hidden_size, W, False, precision=c.precision)
        self.in_proj._bound = True
        if not self.fold or self.prompt:
            self.conv = CausalConv1d(c, W, c.linear_conv_kernel_dim)
            self.conv._bound = True
        order = iron_gdn.gdn_layout(H, KH, dk, dv, COLUMNS)
        rows = np.concatenate([np.asarray(p.weight) for p in (self.in_proj_qkv, self.in_proj_a, self.in_proj_b)])
        self.in_proj.weight = iron.param(np.ascontiguousarray(iron_gdn.padded(rows)[order]))
        if self.fold:
            # the norm's weight on out_proj's input columns (each value head's dv)
            w = np.asarray(self.out_proj.weight, np.float32) * np.tile(np.asarray(self.norm.weight, np.float32), H)
            self.out_proj.weight = iron.param(w.astype(np.asarray(self.out_proj.weight).dtype))
            if self.prompt:  # the chunk's norm without its weight (out_proj holds it)
                self.norm.weight = iron.param(np.ones(dv, np.asarray(self.norm.weight).dtype))
            else:
                self.norm = None
        w = np.asarray(self.conv1d.weight)
        K = w.shape[-1]
        taps = np.zeros((self.qkv + 2 * H, 1, K), w.dtype)
        taps[: self.qkv] = w
        taps[self.qkv :, 0, K - 1] = 1  # a and b: the row itself
        taps = iron_gdn.padded(taps)[order]
        # A row per tap, as CausalConv1d.load_state_dict makes them.
        rows = iron.param(np.ascontiguousarray(taps.reshape(len(order), K).T))
        if self.fold:
            self.taps = rows
        if not self.fold or self.prompt:
            self.conv.weight = iron.param(np.ascontiguousarray(taps))
            self.conv.taps = rows
        A = -np.exp(np.asarray(self.A_log, np.float32))
        self.words = iron_gdn.head_words(np.asarray(self.dt_bias, np.float32), A)
        # read once here: the fused copies are what the device holds
        self.in_proj_qkv = self.in_proj_a = self.in_proj_b = self.conv1d = None
        self.dt_bias = self.A_log = None

    def body(self, x, call):
        H, KH, dk, dv = self.shape
        shape = dict(heads=H, key_heads=KH, key_dim=dk, value_dim=dv, words=self.words)
        p = self.in_proj(x)
        read, write = self.state.at(call)
        if iron_verify.is_rows(x) and self.fold:
            return self.rows_body(x, p, read, write, shape)
        if call.prompt:  # silu once before the operator, which reads each row once a slice
            GDR = iron_gdn.gdn_operator(COLUMNS, H // (2 * COLUMNS), pairs=self.fold, presilu=True)
            u = iron_gdn.silu_operator(COLUMNS).step(self.conv(p, call), seg=self.seg,
                                                      raw=iron_gdn.silu_raw(H, KH, dk, dv, COLUMNS))
            y, final = GDR.step(u, Copy.step(read, dtype=np.float32), **shape)
            Copy.step(final, write, dtype=np.float32)
            return self.out_proj(self.norm(y, self.in_proj_z(x)))
        if self.fold:
            c = self.config
            GDR = iron_gdn.gdn_operator(COLUMNS, H // (2 * COLUMNS), c.linear_conv_kernel_dim, float(c.eps))
            y, final = GDR.step(p, self.in_proj_z(x), self.taps, Copy.step(read, dtype=np.float32), **shape)
            Copy.step(final, write, dtype=np.float32)
            return self.out_proj(y)
        GDR = iron_gdn.gdn_operator(COLUMNS, H // (2 * COLUMNS))
        y, final = GDR.step(self.conv(p, call), Copy.step(read, dtype=np.float32), **shape)
        Copy.step(final, write, dtype=np.float32)
        return self.out_proj(self.norm(y, self.in_proj_z(x)))


    def rows_body(self, x, p, read, write, shape):
        """The version checking drafts: the decode step's operator over its few rows in one call
        (iron_gdn's steps), each row run as a decode step's, the state after each kept: the last
        row's to the slot, the others' to rows_state."""
        c = self.config
        H, KH, dk, dv = self.shape
        R = x.shape[0]
        GDR = iron_gdn.gdn_operator(COLUMNS, H // (2 * COLUMNS), c.linear_conv_kernel_dim, float(c.eps),
                                    steps=R)
        y, finals = GDR.step(p, self.in_proj_z(x), self.taps, Copy.step(read, dtype=np.float32), **shape)
        size = finals.shape[0] // R
        Copy.step(finals[(R - 1) * size :], write, dtype=np.float32)
        Copy.step(finals[: (R - 1) * size], self.rows_state.reshape((R - 1) * size), dtype=np.float32)
        return self.out_proj(y)


class GatedAttention(Attention):
    """``Attention`` whose ``q_proj`` also gives a gate per head column, ``sigmoid(gate)`` times
    the heads' output before ``o_proj``; q and k normed per head."""

    def __init__(self, c, i: int):
        super().__init__(c, i, norms="qk")

    def projections(self, c) -> None:
        super().projections(c)
        H, D = self.heads, self.head_dim
        self.q_proj = Linear(c.hidden_size, 2 * H * D, False, precision=c.precision)

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        H, D = self.heads, self.head_dim
        w = np.asarray(self.q_proj.weight).reshape(H, 2, D, -1)
        self.q_proj.weight = iron.param(np.ascontiguousarray(w[:, 0].reshape(H * D, -1)))
        self.gate = iron.param(np.ascontiguousarray(w[:, 1].reshape(H * D, -1)))

    def output(self, o, x):
        g = Sigmoid.step(F.linear(x, self.gate, precision=self.q_proj.precision))
        return self.o_proj(o * g)


class Qwen35Layer(DecoderLayer):
    """Layer ``i``: Gated DeltaNet or gated attention, then the feed-forward, pre-normed."""

    # GatedDeltaNet's fold (the decode step's convolution and gated norm in its operator), with a
    # prompt chunk too (GatedDeltaNet's prompt: the host copies the convolution's rows between
    # the two around a chunk, hwnpu's conv_pairs)
    fold = True
    prompt = True

    def __init__(self, c: Qwen35Config, i: int):
        if c.linear(i):
            super().__init__(c, i, self_attn=iron.Graph(), cache=iron.Graph())
            del self.self_attn, self.cache
            self.linear_attn = GatedDeltaNet(c, self.fold, self.fold and self.prompt)
        else:
            super().__init__(c, i, GatedAttention(c, i))
            self.linear_attn = None

    def mix(self, h, call):
        if self.linear_attn is not None:
            return self.linear_attn(h, call)
        return super().mix(h, call)

    def body(self, x, call):
        # A decode step's two residuals added in the blocks' last projections' GEMVs (iron_q4's
        # into_linear: out_proj or o_proj, and down_proj), not by adds of their own
        if call.prompt or x.shape[0] != 1 or not QUANT or iron_verify.is_rows(x):
            return super().body(x, call)
        import iron_q4

        last = self.linear_attn.out_proj if self.linear_attn is not None else self.self_attn.o_proj
        with iron_q4.into_linear(last, x) as st:
            y = self.mix(self.input_layernorm(x), call)
        h = y if st["used"] else x + y
        with iron_q4.into_linear(self.mlp.down_proj, h) as st:
            y = self.mlp(self.post_attention_layernorm(h))
        return y if st["used"] else h + y


class Qwen35(CausalLM):
    """Qwen3.5 on ``config``'s shape and a checkpoint's ``weights``."""

    config = Qwen35Config
    layer_class = Qwen35Layer
    # With install(), the token embeddings go to the host (iron_hostple's embed.bin, embed_rows
    # of a call's rows on the device, as PLaMo 3's and LFM3's): 0.47 GiB of 0.8B's bf16, 0.95 of
    # 2B's. Qwen35Moe's routed decode step keeps them.
    host_tables = True

    def __init__(self, config: Qwen35Config, weights):
        # A multimodal checkpoint's text model is model.language_model; its
        # vision tower (model.visual) and multi-token prediction head (mtp)
        # are left out (renamed's namespaces are new, so dropping them leaves
        # weights as it was).
        weights = renamed(weights, {"model.language_model": "model"})
        vars(weights).pop("mtp", None)
        if "model" in vars(weights):
            vars(weights.model).pop("visual", None)
        super().__init__(config, weights)
        for i, layer in enumerate(self.layers):
            if config.linear(i):
                layer.rope = None
        RMSNorm.plus_one(self)
        self.host_embed = False
        if QUANT:
            self.quantize(QUANT)
            if self.host_tables:
                self.embed_on_host()

    def embed_on_host(self) -> None:
        """The token embeddings to the host (iron_hostple's embed.bin); the head (tied to them)
        keeps the quantized copy quantize made."""
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

    @functools.wraps(CausalLM.body)
    def body(self, ids=None, **values):
        if ids is not None and self.host_embed:
            # a version's input must be read, and no table is left to gather from (iron_hostple)
            Copy.step(ids, dtype=np.int32)
        return super().body(ids, **values)

    def hwnpu_conv_pairs(self) -> list:
        """For hwnpu: each folded Gated DeltaNet layer's two homes of the convolution's rows
        (the state's pair ahead of each column's run, the decode step's; the convolution's own
        state, the prompt chunk's), which the host copies between around a prompt chunk."""
        out = []
        for i, layer in enumerate(self.layers):
            g = layer.linear_attn
            if g is None or not (g.fold and g.prompt):
                continue
            H, KH, dk, dv = g.shape
            pair = 2 * dk * iron_gdn.SLICE
            out.append(dict(state=f"model.layers.{i}.linear_attn.state.slots",
                            conv=f"model.layers.{i}.linear_attn.conv.state.slots",
                            columns=COLUMNS, seg=iron_gdn.column_widths(H, KH, dk, dv, COLUMNS),
                            rows=self.config.linear_conv_kernel_dim - 1, pair=pair,
                            run=(H // (2 * COLUMNS) * (dv // iron_gdn.SLICE) + 1) * pair))
        return out

    def quantize(self, fmts: dict) -> None:
        """Put the loaded projections in ``fmts``' formats (``install``)."""
        import iron_q4

        def q(holder, name, fmt):
            if fmt != "bf16":
                setattr(holder, name, iron_q4.q_param(np.asarray(getattr(holder, name), np.float32), fmt))

        for layer in self.layers:
            if layer.linear_attn is not None:
                g = layer.linear_attn
                for lin in (g.in_proj, g.in_proj_z, g.out_proj):
                    q(lin, "weight", fmts["attn"])
            else:
                a = layer.self_attn
                for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                    q(lin, "weight", fmts["attn"])
                q(a, "gate", fmts["attn"])
            if not isinstance(layer.mlp, SparseMoe):
                for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                    q(lin, "weight", fmts["mlp"])
            else:  # the prompt chunk's shared expert
                for lin in layer.mlp.shared:
                    q(lin, "weight", fmts["mlp"])
        # the head its own quantized copy where it is tied: the embeddings stay bf16
        q(self.lm_head, "weight", fmts["head"])


class SparseMoe(iron.Graph):
    """Qwen3.6's ``mlp``: the router (``gate``), the 256 ``experts`` and the ``shared_expert``
    with its ``shared_expert_gate``; on the device MoERoute, MoEGateUp and MoEDown (iron_moe.py)."""

    def __init__(self, c: Qwen35Config, index: int):
        E, I, N = c.hidden_size, c.moe_intermediate_size, c.num_experts
        self.index, self.hidden, self.inter, self.n = index, E, I, N
        self.gate = Linear(E, N, False, precision=c.precision)
        self.experts = iron.Graph()
        self.experts.gate_up_proj = iron.param(N, 2 * I, E)
        self.experts.down_proj = iron.param(N, E, I)
        self.shared_expert = iron.Graph()
        S = c.shared_expert_intermediate_size
        for name, shape in (("gate_proj", (S, E)), ("up_proj", (S, E)), ("down_proj", (E, S))):
            setattr(self.shared_expert, name, Linear(shape[1], shape[0], False, precision=c.precision))
        self.shared_expert_gate = Linear(E, 1, False, precision=c.precision)
        self.precision = c.precision

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        E, I, N = self.hidden, self.inter, self.n
        gu = np.asarray(self.experts.gate_up_proj, np.float32)
        down = np.asarray(self.experts.down_proj, np.float32)
        sh = self.shared_expert
        tables = [iron_moeprobe.gu_table(gu[e, :I], gu[e, I:]) for e in range(N)]
        tables.append(iron_moeprobe.gu_table(np.asarray(sh.gate_proj.weight, np.float32),
                                             np.asarray(sh.up_proj.weight, np.float32)))
        self.gu = iron.param(iron_moe.quantize_experts(tables))
        downs = [down[e] for e in range(N)] + [np.asarray(sh.down_proj.weight, np.float32)]
        self.down = iron.param(iron_moe.quantize_experts(downs))
        self.gu_bytes = self.gu.nbytes // (N + 1)
        self.down_bytes = self.down.nbytes // (N + 1)
        # the router's logits, then the shared expert's gate, padded to whole column blocks
        rows = -(-(N + 1) // 64) * 64
        router = np.zeros((rows, E), bfloat16)
        router[:N] = np.asarray(self.gate.weight)
        router[N] = np.asarray(self.shared_expert_gate.weight)[0]
        self.router = iron.param(router)
        # B's scratchpad image: written once the lowering has placed its words (program_rows)
        self.program = iron.param(np.zeros(4 * iron_moe.SLOTS, np.int32))
        # a prompt chunk's shared expert: its own projections (the table's copy is for the
        # decode step's MoEGateUp and MoEDown)
        self.shared = [sh.gate_proj, sh.up_proj, sh.down_proj]
        self.gate = self.experts = self.shared_expert = self.shared_expert_gate = None

    def body(self, hn, h, call):
        if call.prompt:
            return self.prompt_body(hn, h, call)
        E, I = self.hidden, self.inter
        Route = iron_moe.route_operator()
        logits = F.linear(hn, self.router, precision=self.precision)
        n = self.router.shape[0]
        w, image = Route.step(logits.reshape(n), self.program, n=n, experts=self.n, picked=self.index)
        call.feedback.append((self.index, image.name))
        iron.on_host(lambda: None)  # command B: the experts, at the offsets the image holds
        GU, D = iron_moeprobe.fused_operators()
        e = call.experts[self.index]
        a = GU.step(self.gu, hn.reshape(E), w, K=E, NE=I, picked=self.index,
                    **{f"e{i}": e[i] * self.gu_bytes for i in range(len(e))})
        y = D.step(self.down, a, h.reshape(E), KE=I, N=E, picked=self.index,
                   **{f"e{i}": e[i] * self.down_bytes for i in range(len(e))})
        iron.on_host(lambda: None)
        return y.reshape(1, E)

    def prompt_body(self, hn, h, call):
        """A prompt chunk's experts, the host routing them (iron_moep.py): the router's logits
        on the device; the host's route; the shared expert, MoEGroupGateUp and MoEGroupDown;
        the host's combine into ``call.moe.out``.

        IRON's arena does not count a host step's reads as uses: a buffer the host reads must be
        the last step's output, or a state (h and the shared expert's output are copied into
        one), or a later step may have written over it."""
        moe, (C, E), I = call.moe, hn.shape, self.inter
        Copy.step(h, moe.h)
        logits = F.linear(hn, self.router, precision=self.precision)
        iron.on_host(MoEHost("moe_route", self.index, self.n, C, moe), logits, hn,
                     writes=(moe.counts, moe.xs))
        g, u, d = self.shared
        Copy.step(d(F.silu(g(hn)) * u(hn)), moe.s)
        GU, D = iron_moep.group_operators(self.n)
        A = iron_moep.TOP * C
        a = GU.step(self.gu, moe.counts, moe.xs, rows=self.gu.shape[0], A=A, K=E, NE=I)
        y = D.step(self.down, moe.counts, a, rows=self.down.shape[0], A=A, KE=I, N=E)
        iron.on_host(MoEHost("moe_combine", self.index, self.n, C, moe), moe.h, y, moe.s,
                     writes=(moe.out,))
        return moe.out[: call.rows]


class MoEStates:
    """A prompt chunk's states the host writes for every layer's experts (one set: the layers
    run in turn): the counts and x's stream (route), the layer's output (combine)."""

    def __init__(self, C: int, E: int):
        A = iron_moep.TOP * C
        self.counts = iron.state((2 * iron_moep.COUNTS,), bfloat16, name="moe_counts")
        self.xs = iron.state((2 * A * E,), bfloat16, name="moe_xs")
        self.out = iron.state((C, E), bfloat16, name="moe_out")
        self.h = iron.state((C, E), bfloat16, name="moe_h")  # the residual, for combine
        self.s = iron.state((C, E), bfloat16, name="moe_s")  # the shared expert's output
        self.plans = {}  # by layer: route's plan, for combine (the host reference)


class MoEHost:
    """A host step of a prompt chunk's experts: ``hwnpu`` names it for hwnpu (which runs its own
    in Go); called, it is the reference's (iron_moep's route_chunk, plan, stream_x, combine)."""

    def __init__(self, kind: str, layer: int, experts: int, C: int, states: MoEStates):
        self.kind, self.layer, self.experts, self.C, self.states = kind, layer, experts, C, states
        self.hwnpu = dict(kind=kind, layer=layer, experts=experts, top=iron_moep.TOP,
                          cap=iron_moep.CAP, tokens=C)

    def __call__(self, *args):
        getattr(self, self.kind)(*args)

    def moe_route(self, logits, hn, counts, xs):
        n, C = len(logits), self.C
        ids, w, sg = iron_moep.route_chunk(logits, self.experts)
        rows, slot, cnt = iron_moep.plan(ids, self.experts, C)  # the rows past the call's: padding
        counts[:] = iron_moep.counts_words(cnt)
        xs[:] = iron_moep.stream_x(hn, rows, cnt)
        self.states.plans[self.layer] = (slot, w, sg)

    def moe_combine(self, h, y, s, out):
        slot, w, sg = self.states.plans.pop(self.layer)
        n = len(slot)
        h, s = np.asarray(h)[:n], np.asarray(s)[:n]
        out[:n] = iron_moep.combine(h, np.asarray(y).reshape(-1, np.asarray(h).shape[1]), slot, w, s, sg)
        out[n:] = 0


class Qwen35MoeLayer(Qwen35Layer):
    """``Qwen35Layer`` whose feed-forward is ``SparseMoe`` (its residual added in MoEDown)."""

    # Folded: the decode step's GatedDeltaNet runs the convolution and the gated norm in its
    # operator; a prompt chunk (PROMPT) runs IRON's convolution on a state of its own, the host
    # copying the rows between the two (GatedDeltaNet's prompt). With the fold, the next decode
    # step's first command hung on the device after the last layer's experts (B) ran
    # (ERT_CMD_STATE_TIMEOUT; design.md 16.90) until hwnpu ran an empty command in another
    # hardware context before each step (internal/llm/iron's stepBarrier, 16.92).
    fold = True
    prompt = PROMPT

    def __init__(self, c: Qwen35Config, i: int):
        super().__init__(c, i)
        self.mlp = SparseMoe(c, i)

    def body(self, x, call):
        h = x + self.mix(self.input_layernorm(x), call)
        return self.mlp(self.post_attention_layernorm(h), h, call)


class Qwen35Moe(Qwen35):
    """Qwen3.6-35B-A3B (``qwen3_5_moe``): ``Qwen35`` with ``SparseMoe`` layers, a decode step only.

    Its body takes a scratchpad value per layer and chosen expert (``e<layer>_<i>``), which no
    host gives: command B's scratchpad is in the feedback argument, where the layer's MoERoute
    writes them (``hwnpu_fed``)."""

    layer_class = Qwen35MoeLayer
    host_tables = False  # its routed decode step reads the embeddings on the device
    hwnpu_name = "Qwen35Moe"

    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        if "e0_0" in getattr(cls, "_values_made", ()):
            return super().from_checkpoint(path, **kwargs)
        from iron.lm.checkpoint import weights_file

        layers = cls.default_config(weights_file(path)).num_hidden_layers
        return moe_class(cls, layers).from_checkpoint(path, **kwargs)

    def shapes(self):
        if not PROMPT:
            return [{}]  # the decode step alone
        return super().shapes()

    def draw(self, logits, call):
        # No draw on the device: its commands end at host steps, so no Emit carries a drawn
        # token, and hwnpu draws from the logits anyway.
        return logits, iron.carry(token=call.token + 0, position=call.position + 1)

    def record(self, **shapes):
        if not shapes or not hasattr(self, "_fed"):  # the decode step's (a prompt chunk feeds none)
            self._fed = []
        recording = super().record(**shapes)
        if not shapes:
            recording.feedback.extend(name for _, name in self._fed)
        return recording

    def moe_body(self, values, **standard):
        experts = [[values[f"e{l}_{i}"] for i in range(8)] for l in range(self.config.num_hidden_layers)]
        import iron.lm.causal as causal

        made = causal.Call

        def call(**kw):
            c = made(**kw)
            c.experts, c.feedback = experts, self._fed
            if c.prompt:
                c.moe = self._moe_states()
            return c

        causal.Call = call
        try:
            return CausalLM.body(self, **standard)
        finally:
            causal.Call = made

    def _moe_states(self) -> "MoEStates":
        if getattr(self, "_moe", None) is None:
            self._moe = MoEStates(self.config.prefill_chunk, self.config.hidden_size)
        return self._moe

    def program_rows(self, kernels) -> list:
        """Fill each layer's MoERoute program from the decode step's commands (the manifest's
        ``kernels``): command B of layer l is the one whose words are layer l's offsets. Returns
        the manifest's ``fed``: by such command, the feedback buffer its scratchpad is."""
        images = dict(self._fed)
        fed = []
        for layer in self.layers:
            l, m = layer.mlp.index, layer.mlp
            found = [(k, kern["parameters"]) for k, kern in enumerate(kernels)
                     if any((p.get("form") or {}).get("value", "").startswith(f"e{l}_") for p in kern["parameters"])]
            if len(found) != 1:
                raise SystemExit(f"layer {l}: {len(found)} commands read its experts' offsets")
            k, words = found[0]
            prog = np.zeros((iron_moe.SLOTS, 4), np.int32)
            prog[:, 0] = -1
            for p in words:
                f = p.get("form")
                if not f or not f["value"].startswith(f"e{l}_"):
                    raise SystemExit(f"layer {l}: command {k} reads {p['name']}, which no routing gives")
                plain = (f["down"], f["mul"], f["add"], f["post"], f["lo"], f["hi"], f["mod"]) == (0, 1, 0, 0, None, None, None)
                if not plain or p["index"] >= iron_moe.SLOTS:
                    raise SystemExit(f"layer {l}: {p['name']} at {p['index']} is {f}, not id * scale + bias")
                prog[p["index"]] = (int(f["value"].split("_")[1]), f["scale"], f["bias"],
                                    2 if p["kind"] == "core" else 0)
            np.asarray(m.program).reshape(-1)[: prog.size] = prog.reshape(-1)  # gpt-oss: its bias after
            fed.append(dict(command=k, buffer=images[l], words=len(words)))
        return fed


def moe_class(base, layers: int):
    """``base`` whose body takes the 8 offsets of each of ``layers`` layers as scratchpad values."""
    from iron import Carried, Scratchpad

    names = [f"e{l}_{i}" for l in range(layers) for i in range(8)]
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
    lm.main(Qwen35)

