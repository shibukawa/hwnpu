"""LFM3 (Liquid AI's LFM3-1.7B) on IRON's shared decoder (``iron.lm``), its shape read from its checkpoint's ``config.json``.

An application of amd/IRON kept in hwnpu (IRON has none for LFM3), exported by
``iron_export_lm.py lfm3`` like IRON's own Gemma 4.

LFM3's layers are of three kinds (``layer_types``):

- ``conv``: a gated short convolution. ``in_proj`` gives B, C and x (in that
  order of its rows); B * x through a depth-wise causal convolution of
  ``conv_L_cache`` taps (IRON's ``CausalConv1d``, its state at each
  prompt-chunk boundary as Granite 4.0-H's), then C times it and ``out_proj``.
- ``full_attention``: attention over q and k RMS-normed per head, RoPE on the
  first half of each head's frequencies (the "zero-tail" layout: the pairs
  past ``partial_rotary_factor`` stand still, Gemma 4's proportional RoPE),
  and a sigmoid gate per head on the output before ``o_proj``.
- ``sliding_attention``: the same within ``sliding_window`` positions, with
  full RoPE of its own theta.

Every layer is sandwich-normed (``use_sandwich_norm``): its block's input and
output, and the feed-forward's (``w1``, ``w3``, ``w2``: SwiGLU). The final
norm is ``embedding_norm``; the head is the embedding.

With ``install_q4`` (``iron_export_lm.py lfm3 --q4``) the projections are in iron_q4's
formats, by part: the feed-forward's in q4nx, the attention's and the convolution's in q8,
the head in q4nx. The head is tied to the token embeddings, whose rows then go to the host
(``embed.bin``, iron_hostple's ``hwnpu_embed_rows``, as Gemma 4's with ``--q4``).
"""

import dataclasses
import functools

import numpy as np

import iron
import iron.functional as F
from iron import Precision, lm
from iron.lm import (
    Attention,
    CausalConv1d,
    CausalLM,
    DecoderConfig,
    KVCache,
    Linear,
    Rope,
    SandwichLayer,
)
from iron.operators.copy import Copy
from iron.operators.sigmoid import Sigmoid
from ml_dtypes import bfloat16

# A decode step's short convolution is one operator (iron_shortconv.ShortConvStep) in place of
# IRON's about 16 (docs/design/amd-llm.md); a prompt chunk's the same (ShortConvChunk, in_proj
# one linear) instead of IRON's about 26 operators a conv layer. ShortConvStep reads and writes
# the convolution's checkpoint slots itself (at per-call offsets). A decode step's post-norm,
# residual and next norm are one operator (iron_sandwich.SandwichAdd) in place of three, or made
# in the GEMV that reads their output (iron_q4.sandwich_linear: a conv layer's in_proj, the MLP's
# QGEMVGlu, the head; docs/design/amd.md). With q4 or q8 the MLP's gate and up are one table
# (iron_q4's glu_param, QGEMVGlu), as iron_qwen35's FUSE_GLU.

# By part of the model (mlp, attn, conv, head), its weights' format: iron_q4's "q4" (q4nx) or
# "q8", or "bf16". Empty: all bf16, IRON's own linear.
QUANT: dict[str, str] = {}


def install_q4(mlp: str = "q4", attn: str = "q8", conv: str = "q8", head: str = "q4") -> None:
    """Build LFM3 (and the LFM3-VL, Audio and Omni text models) with its projections in these
    formats ("q4", "q8" or "bf16"), F.linear running iron_q4's on them."""
    import iron_hostple
    import iron_q4
    from iron.common.image import packaging

    fmts = dict(mlp=mlp, attn=attn, conv=conv, head=head)
    for part, fmt in fmts.items():
        if fmt not in ("q4", "q8", "bf16"):
            raise ValueError(f"{part} in {fmt}: q4, q8 or bf16")
    iron_q4.install_linear()
    QUANT.update(fmts)
    # Without the gather of the embeddings, a prompt chunk's commands would merge (iron_hostple).
    packaging.SCRATCHPAD_WORDS = min(packaging.SCRATCHPAD_WORDS, iron_hostple.SCRATCHPAD_WORDS)


@dataclasses.dataclass(frozen=True)
class Lfm3Config(DecoderConfig):
    """``DecoderConfig`` with LFM3's layer kinds, its two RoPEs and its short convolution."""

    rope_theta: float = dataclasses.field(
        metadata={"json": "rope_parameters.sliding_attention.rope_theta"}
    )
    global_rope_theta: float = dataclasses.field(
        kw_only=True, metadata={"json": "rope_parameters.full_attention.rope_theta"}
    )
    partial_rotary_factor: float = dataclasses.field(
        default=1.0,
        kw_only=True,
        metadata={"json": "rope_parameters.full_attention.partial_rotary_factor"},
    )
    layer_types: tuple[str, ...] = dataclasses.field(
        kw_only=True, metadata={"convert": tuple}
    )
    conv_L_cache: int = dataclasses.field(default=3, kw_only=True)
    conv_bias: bool = dataclasses.field(default=False, kw_only=True)
    use_sandwich_norm: bool = dataclasses.field(default=False, kw_only=True)
    eps: float = dataclasses.field(
        default=1e-5, kw_only=True, metadata={"json": "norm_eps"}
    )
    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)

    @classmethod
    def from_json(cls, data: dict, source="config.json") -> "Lfm3Config":
        """``DecoderConfig.from_json``.

        Raises:
            ValueError: ``data`` asks for what is not built: RoPE other than
                the default, layers without the sandwich norms, or a layer
                kind other than conv, full and sliding attention.
        """
        # the text model's own object: the root, or text_config (LFM3-VL), lfm (LFM3-Audio)
        text = data.get(cls.json_prefix.rstrip("."), {}) if cls.json_prefix else data
        for kind, params in text.get("rope_parameters", {}).items():
            if params.get("rope_type", "default") != "default":
                raise ValueError(f"{source}: {kind} RoPE {params['rope_type']!r} is not built here")
        if not text.get("use_sandwich_norm"):
            raise ValueError(f"{source}: LFM3 without use_sandwich_norm is not built here")
        kinds = set(text.get("layer_types", ())) - {"conv", "full_attention", "sliding_attention"}
        if kinds:
            raise ValueError(f"{source}: layer kinds {sorted(kinds)} are not built here")
        return super().from_json(data, source)

    def __post_init__(self):
        super().__post_init__()
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"Lfm3Config: {len(self.layer_types)} layer_types for {self.num_hidden_layers} layers"
            )

    def kind(self, i: int) -> str:
        """Layer ``i``'s kind: conv, full_attention or sliding_attention."""
        return self.layer_types[i]


class ShortConv(iron.Graph):
    """LFM2's gated short convolution on ``c``'s shape, named as the checkpoint's ``conv``."""

    # A prompt chunk's as one operator (ShortConvChunk): the language model's layers (Lfm3Layer), not
    # the audio encoder's (iron_lfm3audio), whose calls are its own
    chunked = False

    def __init__(self, c: Lfm3Config):
        E = self.width = c.hidden_size
        self.in_proj = Linear(E, 3 * E, c.conv_bias, precision=c.precision)
        self.conv = CausalConv1d(c, E, c.conv_L_cache, bias=c.conv_bias)
        self.out_proj = Linear(E, E, c.conv_bias, precision=c.precision)

    def body(self, x, call):
        E = self.width
        if not call.prompt and x.shape[0] == 1 and self.conv.bias is None:
            return self.decode_from(self.in_proj(x), call)
        if self.chunked and call.prompt and x.valid.get(0) is not None and self.conv.bias is None:
            return self.chunk_from(self.in_proj(x), call)
        b, cg, v = self.in_proj.split(x, E, E, E)
        y = self.conv(b * v, call)
        return self.out_proj(cg * y)


    def decode_from(self, p, call):
        """A decode step's mixer from in_proj's row p (B, C and x): B * x, the convolution and the
        gate as one operator (iron_shortconv), then out_proj."""
        import iron_shortconv

        E, K = self.width, self.conv.taps.shape[0]
        # the operator reads and writes the checkpoint slots itself
        slots = self.conv.state.slots
        ns, size, C = slots.shape[0], (K - 1) * E, self.conv.state.chunk
        whole = slots.reshape(ns * size)
        y, _ = iron_shortconv.short_conv_operator(slots=ns).step(
            p.reshape(1, 3 * E), self.conv.taps, whole, whole, width=E, taps_n=K,
            read_at=((call.start - 1) // C + 1) * size, write_at=(call.position // C + 1) * size)
        return self.out_proj(y)


    def chunk_from(self, p, call):
        """A prompt chunk's mixer from in_proj's rows p (B, C and x, the call's rows bounded): the
        products, the convolution over the rows and the gate as one operator (iron_shortconv's
        ShortConvChunk), then out_proj."""
        import iron_shortconv

        E, K = self.width, self.conv.taps.shape[0]
        read, write = self.conv.state.at(call)
        y, final = iron_shortconv.short_conv_chunk_operator().step(
            p, self.conv.taps, Copy.step(read), rows=p.shape[0], width=E, taps_n=K)
        Copy.step(final, write)
        return self.out_proj(y)


class GatedAttention(Attention):
    """``Attention`` with q and k normed per head and a sigmoid gate per head on its output.

    The gate's rows are each head's row of ``gate_proj`` repeated over the
    head's columns, so that it multiplies the heads' output elementwise.
    """

    def __init__(self, c, i: int):
        super().__init__(c, i, norms="qk")
        self.gate_proj = Linear(c.hidden_size, c.num_attention_heads, False, precision=c.precision)

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        w = np.asarray(self.gate_proj.weight)
        self.gate = iron.param(np.ascontiguousarray(np.repeat(w, self.head_dim, axis=0)))

    def output(self, o, x):
        g = Sigmoid.step(F.linear(x, self.gate, precision=self.gate_proj.precision))
        return self.o_proj(o * g)


class Lfm3Layer(SandwichLayer):
    """Layer ``i``: a short convolution or gated attention, then SwiGLU, each sandwich-normed."""

    def __init__(self, c: Lfm3Config, i: int):
        kind = c.kind(i)
        if kind == "conv":
            super().__init__(c, i)
            del self.self_attn, self.cache
            self.conv = ShortConv(c)
            self.conv.chunked = True
        else:
            window = c.window if kind == "sliding_attention" else None
            super().__init__(c, i, GatedAttention(c, i), None, KVCache(c, window))
            self.conv = None
        self.kind = kind

    def mix(self, h, call):
        if self.conv is not None:
            return self.conv(h, call)
        return super().mix(h, call)

    def fused_body(self, x, pend, call):
        """A decode step's layer on the residual ``x`` and ``pend``: ("normed", the input normed) for the
        first layer, else ("pending", the last block's output, its post norm's weight), the
        residual not yet added. The sandwiches (post norm, residual, next norm) are made in the GEMV
        that reads their output where one does (iron_q4.sandwich_linear: in_proj of a conv layer, the
        MLP's QGEMVGlu), else by SandwichAdd. Returns the residual and the next ("pending", ...)."""
        import iron_q4
        import iron_sandwich

        eps = self.post_attention_layernorm.eps
        win = self.input_layernorm.weight
        if pend[0] == "pending" and self.conv is not None:
            p, x, _ = iron_q4.sandwich_linear(pend[1], x, pend[2], win, eps, self.conv.in_proj.weight)
            h = self.conv.decode_from(p, call)
        else:
            if pend[0] == "pending":
                x, xn = iron_sandwich.sandwich_add(pend[1], x, pend[2], win, eps)
            else:
                xn = pend[1]
            h = self.mix(xn, call)
        gu = getattr(self.mlp, "gu", None)
        if gu is not None:
            g, x, _ = iron_q4.sandwich_linear(h, x, self.post_attention_layernorm.weight,
                                              self.pre_feedforward_layernorm.weight, eps, gu)
            h = self.mlp.down_proj(g)
        else:
            x, m = iron_sandwich.sandwich_add(h, x, self.post_attention_layernorm.weight,
                                              self.pre_feedforward_layernorm.weight, eps)
            h = self.mlp(m)
        return x, ("pending", h, self.post_feedforward_layernorm.weight)


class Lfm3(CausalLM):
    """LFM3 on ``config``'s shape and a checkpoint's ``weights``, from its checkpoint and ``tokenizer.json``."""

    config = Lfm3Config
    layer_class = Lfm3Layer
    renames = {
        "operator_norm": "input_layernorm",
        "post_operator_norm": "post_attention_layernorm",
        "ffn_norm": "pre_feedforward_layernorm",
        "post_ffn_norm": "post_feedforward_layernorm",
        "feed_forward.w1": "mlp.gate_proj",
        "feed_forward.w3": "mlp.up_proj",
        "feed_forward.w2": "mlp.down_proj",
        "embedding_norm": "norm",
    }

    def __init__(self, config: Lfm3Config, weights):
        super().__init__(config, weights)
        c = config
        turned = int(c.partial_rotary_factor * c.head_dim // 2)
        full = dataclasses.replace(
            c,
            rope_theta=c.global_rope_theta,
            # The zero-tail layout: the pairs past ``turned`` stand still.
            rope_scaling=lambda f: np.where(np.arange(f.size) < turned, f, 0),
        )
        rope = Rope(full)
        for i, layer in enumerate(self.layers):
            if c.kind(i) == "full_attention":
                layer.rope = rope
            elif c.kind(i) == "conv":
                layer.rope = None
        self.embed_rows = None
        if QUANT:
            self.quantize(QUANT)

    def fused(self, call, x) -> bool:
        """Whether this call's layers run fused_body (a decode step's row)."""
        c = self.config
        return (not call.prompt and x.shape[0] == 1
                and getattr(self.model.layers[0], "residual_multiplier", None) in (None, 1, 1.0)
                and all(n.eps == c.eps for l in self.model.layers for n in (
                    l.input_layernorm, l.post_attention_layernorm, l.pre_feedforward_layernorm,
                    l.post_feedforward_layernorm)) and self.model.norm.eps == c.eps)

    def layer(self, call, i: int, w, x):
        if not self.fused(call, x):
            return super().layer(call, i, w, x)
        pend = ("normed", w.input_layernorm(x)) if i == 0 else self._pend
        x, self._pend = w.fused_body(x, pend, call)
        return x

    def head(self, x):
        pend, self._pend = getattr(self, "_pend", None), None
        if pend is None:
            return super().head(x)
        import iron_q4
        import iron_sandwich

        _, h, wpost = pend
        norm = self.model.norm
        if id(self.lm_head.weight) in iron_q4._Q4:
            # the last layer's sandwich, its next norm the final norm, made in the head's GEMV
            logits, _, _ = iron_q4.sandwich_linear(h, x, wpost, norm.weight, norm.eps, self.lm_head.weight)
            return logits.reshape(1, logits.shape[-1])
        _, xn = iron_sandwich.sandwich_add(h, x, wpost, norm.weight, norm.eps)
        return self.lm_head(xn)

    def quantize(self, fmts: dict) -> None:
        """Put the loaded weights in ``fmts``' formats (``install_q4``)."""
        import iron_hostple
        import iron_q4

        def q(w, fmt):
            return iron_q4.q_param(np.asarray(w, np.float32), fmt)

        for layer in self.layers:
            if fmts["mlp"] != "bf16":
                # gate and up as one table: the decode step's one operator (iron_q4's QGEMVGlu) in
                # place of two GEMVs, a SiLU and a product; a prompt chunk reads it in parts
                mlp = layer.mlp
                iron_q4.install_glu()
                mlp.gu = iron_q4.glu_param(np.asarray(mlp.gate_proj.weight, np.float32),
                                           np.asarray(mlp.up_proj.weight, np.float32), fmts["mlp"])
                mlp.gate_proj = mlp.up_proj = None
                mlp.down_proj.weight = q(mlp.down_proj.weight, fmts["mlp"])
            if layer.conv is not None:
                conv, fmt = layer.conv, fmts["conv"]
                if fmt != "bf16":
                    # in_proj one weight: iron_q4's Linear.split cuts it into B, C and x
                    for lin in (conv.in_proj, conv.out_proj):
                        lin.weight = q(lin.weight, fmt)
            elif fmts["attn"] != "bf16":
                a, fmt = layer.self_attn, fmts["attn"]
                for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                    lin.weight = q(lin.weight, fmt)
                # the gate's rows, each head's repeated over its columns: as big as o_proj
                a.gate = q(a.gate, fmt)
        if fmts["head"] == "bf16":
            return
        # The head is tied to the embeddings: its own copy quantized, the embeddings to the host.
        c = self.config
        embed = np.asarray(self.model.embed_tokens.weight)
        iron_hostple._TABLES.setdefault(id(self), {})[iron_hostple.EMBED] = embed.astype(bfloat16)
        self.lm_head.weight = q(embed, fmts["head"])
        self.model.embed_tokens.weight = iron.param(np.zeros((1, c.hidden_size), bfloat16))  # unread
        self.embed_rows = iron.state((c.prefill_chunk, c.hidden_size), bfloat16, name=iron_hostple.EMBED)

    def lookup(self, ids):
        # From embed_rows, which the host fills with the call's tokens' rows in order.
        if self.embed_rows is None:
            return super().lookup(ids)
        shape = getattr(ids, "shape", ())
        if len(shape) == 0:
            rows = self.embed_rows[:1]
        else:
            bound = getattr(ids, "valid", None) or {}
            rows = self.embed_rows[: bound[0]] if 0 in bound else self.embed_rows[: shape[0]]
        return Copy.step(rows)

    @functools.wraps(CausalLM.body)
    def body(self, ids=None, **values):
        if ids is not None and self.embed_rows is not None:
            # A version's input must be read, and no table is left to gather from: a copy of
            # the whole ids (as iron_hostple's Gemma 4), which nothing reads.
            Copy.step(ids, dtype=np.int32)
        return super().body(ids, **values)


if __name__ == "__main__":
    lm.main(Lfm3)
