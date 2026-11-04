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
"""

import dataclasses

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
from iron.operators.sigmoid import Sigmoid


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
        text = data.get("text_config", {}) if cls.json_prefix == "text_config." else data
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

    def __init__(self, c: Lfm3Config):
        E = self.width = c.hidden_size
        self.in_proj = Linear(E, 3 * E, c.conv_bias, precision=c.precision)
        self.conv = CausalConv1d(c, E, c.conv_L_cache, bias=c.conv_bias)
        self.out_proj = Linear(E, E, c.conv_bias, precision=c.precision)

    def body(self, x, call):
        E = self.width
        b, cg, v = self.in_proj.split(x, E, E, E)
        y = self.conv(b * v, call)
        return self.out_proj(cg * y)


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
        else:
            window = c.window if kind == "sliding_attention" else None
            super().__init__(c, i, GatedAttention(c, i), None, KVCache(c, window))
            self.conv = None
        self.kind = kind

    def mix(self, h, call):
        if self.conv is not None:
            return self.conv(h, call)
        return super().mix(h, call)


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


if __name__ == "__main__":
    lm.main(Lfm3)
