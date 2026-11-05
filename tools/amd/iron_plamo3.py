"""PLaMo 3 (Preferred Networks' Plamo3ForCausalLM, e.g. pfnet/plamo-3-translate-2b) on IRON's shared decoder (``iron.lm``).

An application of amd/IRON kept in hwnpu (IRON has none for PLaMo 3), exported by
``iron_export_lm.py plamo3`` like LFM3's (iron_lfm3.py). Its shape is read from
the checkpoint's ``config.json``; the structure follows its ``modeling_plamo.py``:

- Each layer is sandwich-normed: ``pre_mixer_norm``, the attention (``mixer``),
  ``post_mixer_norm``, the residual; ``pre_mlp_norm``, the SwiGLU MLP,
  ``post_mlp_norm``, the residual. PLaMo's RMSNorm scales by ``offset + weight``
  with an offset of its own per norm (1 for the pre-norms, the query and key
  norms and the final norm; 1/5 after the attention, 1/5^1.5 after the MLP).
  The offsets are folded into the weights once loaded, as Gemma's ``1 +
  weight`` is (``RMSNorm.plus_one``).
- The attention has one fused ``qkv_proj`` (rows q, k, v), grouped keys and
  values, each head's q and k RMS-normed, and rotate_half RoPE. Every
  ``sliding_window_pattern``-th layer attends to the whole context; the
  others to the ``window_size`` positions up to their own, with RoPE of
  ``rope_local_theta``.
- The full layers' RoPE is YaRN of ``rope_scaling_factor`` over
  ``initial_context_length`` (beta_fast 32, beta_slow 1, modeling_plamo.py's
  ``rope_parameters``; the config's own ``rope_parameters`` is not read by it),
  its cos and sin times the "concentration" ``0.1 ln(factor) + 1``
  (``rope_attention_scaling``).
- The MLP's ``gate_up_proj`` holds the gate's rows, then the up projection's.
- No embedding scale (``scale_embedding`` false); the head is ``lm_head``.

Checkpoints without a tokenizer.json (PLaMo's is tokenizer.jsonl) load with no
tokenizer: IRON's command line then takes token ids only, and hwnpu tokenizes.
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import iron
from iron import Precision, lm
from iron.common.graph.compiled import walk
from iron.lm import (
    Attention,
    CausalLM,
    DecoderConfig,
    GatedMLP,
    KVCache,
    Linear,
    RMSNorm,
    Rope,
    SandwichLayer,
)
from iron.lm.checkpoint import Checkpoint, renamed, weights_file

# PLaMo's RMSNorm offsets by the norm's name (modeling_plamo.py's Plamo3DecoderLayer).
OFFSETS = {
    "input_layernorm": 1.0,  # pre_mixer_norm
    "post_attention_layernorm": 1.0 / 5,  # post_mixer_norm
    "pre_feedforward_layernorm": 1.0,  # pre_mlp_norm
    "post_feedforward_layernorm": 1.0 / 5**1.5,  # post_mlp_norm
    "q_norm": 1.0,
    "k_norm": 1.0,
    "norm": 1.0,  # the final norm
}


@dataclasses.dataclass(frozen=True)
class Plamo3Config(DecoderConfig):
    """``DecoderConfig`` with PLaMo 3's two RoPEs and its pattern of sliding-window layers."""

    # The sliding-window layers' theta; the full layers' is global_rope_theta.
    rope_theta: float = dataclasses.field(metadata={"json": "rope_local_theta"})
    global_rope_theta: float = dataclasses.field(kw_only=True, metadata={"json": "rope_theta"})
    rope_scaling_factor: float = dataclasses.field(default=1.0, kw_only=True)
    initial_context_length: int | None = dataclasses.field(default=None, kw_only=True)
    sliding_window: int | None = dataclasses.field(default=None, kw_only=True, metadata={"json": "window_size"})
    sliding_window_pattern: int = dataclasses.field(default=8, kw_only=True)
    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)

    @classmethod
    def from_json(cls, data: dict, source="config.json") -> "Plamo3Config":
        """``DecoderConfig.from_json``.

        Raises:
            ValueError: ``data`` asks for what is not built: a scaled
                embedding, an image projection, FP8 linears, or another model.
        """
        if data.get("model_type") != "plamo3":
            raise ValueError(f"{source}: model_type {data.get('model_type')!r} is not plamo3")
        if data.get("scale_embedding"):
            raise ValueError(f"{source}: scale_embedding is not built here")
        if data.get("image_feature_size") is not None:
            raise ValueError(f"{source}: the image projection is not built here")
        if data.get("linear_type", "normal") != "normal":
            raise ValueError(f"{source}: linear_type {data['linear_type']!r} is not built here")
        if data.get("rope_scaling_factor", 1) != 1 and data.get("initial_context_length") is None:
            raise ValueError(f"{source}: rope_scaling_factor without initial_context_length")
        return super().from_json(data, source)

    def full(self, i: int) -> bool:
        """Whether layer ``i`` attends to the whole context (modeling_plamo.py's ``is_full_attn``)."""
        return (i + 1) % self.sliding_window_pattern == 0


def yarn(c: Plamo3Config):
    """The full layers' RoPE scaling of ``inv_freq`` (modeling_plamo.py's RotaryEmbedding), or None."""
    factor = c.rope_scaling_factor
    if factor == 1:
        return None
    d_h = c.head_dim // 2
    original, beta_fast, beta_slow = c.initial_context_length, 32.0, 1.0
    base = c.global_rope_theta
    low = d_h * math.log(original / (beta_fast * 2 * math.pi)) / math.log(base)
    high = d_h * math.log(original / (beta_slow * 2 * math.pi)) / math.log(base)
    if not 0 < low < high < d_h - 1:
        raise ValueError(f"Plamo3: YaRN's ramp {low}..{high} is outside 0..{d_h - 1}")
    ramp = (np.arange(d_h, dtype=np.float32) - np.float32(low)) / np.float32(high - low)
    mask = 1 - np.clip(ramp, 0, 1)

    def scale(inv_freq):
        return inv_freq / factor * (1.0 - mask) + inv_freq * mask

    return scale


class FusedAttention(Attention):
    """``Attention`` over PLaMo's fused ``qkv_proj``, q and k normed per head."""

    def __init__(self, c, i: int):
        super().__init__(c, i, norms="qk")

    def projections(self, c) -> None:
        E, D = c.hidden_size, self.head_dim
        self.qkv_proj = Linear(E, (self.heads + 2 * self.groups) * D, False, precision=c.precision)

    def project(self, x, **gemv) -> list:
        D = self.head_dim
        return self.qkv_proj.split(x, self.heads * D, self.groups * D, self.groups * D, **gemv)


class FusedMLP(GatedMLP):
    """``GatedMLP`` over PLaMo's fused ``gate_up_proj`` (the gate's rows, then up's)."""

    def projections(self, c) -> None:
        self.gate_up_proj = Linear(c.hidden_size, 2 * c.intermediate_size, False, precision=c.precision)

    def body(self, x):
        M = self.down_proj.in_features
        gate, up = self.gate_up_proj.split(x, M, M)
        return self.down_proj(self.act(gate) * up)


class Plamo3Layer(SandwichLayer):
    """Layer ``i``: attention (over the whole context or a window), then SwiGLU, each sandwich-normed."""

    def __init__(self, c: Plamo3Config, i: int):
        window = None if c.full(i) else c.window
        super().__init__(c, i, FusedAttention(c, i), FusedMLP(c), KVCache(c, window))


class Plamo3(CausalLM):
    """PLaMo 3 on ``config``'s shape and a checkpoint's ``weights``."""

    config = Plamo3Config
    layer_class = Plamo3Layer
    renames = {
        "mixer": "self_attn",
        "pre_mixer_norm": "input_layernorm",
        "post_mixer_norm": "post_attention_layernorm",
        "pre_mlp_norm": "pre_feedforward_layernorm",
        "post_mlp_norm": "post_feedforward_layernorm",
    }

    def __init__(self, config: Plamo3Config, weights):
        # The decoder's layers are model.layers.layers.N (Plamo3Decoder within Plamo3Model).
        super().__init__(config, renamed(weights, {"layers.layers": "layers"}))
        c = config
        full = dataclasses.replace(
            c,
            rope_theta=c.global_rope_theta,
            rope_scaling=yarn(c),
            rope_attention_scaling=1.0 if c.rope_scaling_factor == 1 else 0.1 * math.log(c.rope_scaling_factor) + 1.0,
        )
        rope = Rope(full)
        for i, layer in enumerate(self.layers):
            if c.full(i):
                layer.rope = rope
        # PLaMo's norms scale by offset + weight: fold the offsets in, the sum in float32
        # rounded to bf16 as Hugging Face's bf16 parameter plus a Python float is.
        for name, weight, norm, _ in list(walk(self)):
            if isinstance(norm, RMSNorm):
                key = name.rsplit(".", 2)[-2]
                if key not in OFFSETS:
                    raise ValueError(f"Plamo3: no offset known for the norm {name}")
                norm.weight = iron.param((np.asarray(weight, np.float32) + OFFSETS[key]).astype(bfloat16))

    @classmethod
    def from_checkpoint(cls, path, *, random_weights=None, max_seq_len=None):
        """``Model.from_checkpoint`` for a local checkpoint without a tokenizer.json: no tokenizer."""
        directory = Path(path)
        if (directory / "tokenizer.json").exists():
            return super().from_checkpoint(path, random_weights=random_weights, max_seq_len=max_seq_len)
        weights = weights_file(directory)
        config = cls.default_config(weights)
        if max_seq_len is None:
            config = cls.fit(config, weights)
        else:
            if max_seq_len > config.max_seq_len:
                raise ValueError(f"max_seq_len {max_seq_len} is past the model's {config.max_seq_len}")
            config = dataclasses.replace(config, max_seq_len=max_seq_len)
        checkpoint = Checkpoint(weights)
        tree = checkpoint.tree() if random_weights is None else checkpoint.tree("random", random_weights)
        model = cls(config, tree)
        model.checkpoint, model.random_weights = checkpoint, random_weights
        return model


if __name__ == "__main__":
    lm.main(Plamo3)
