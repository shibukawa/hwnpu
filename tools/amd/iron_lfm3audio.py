"""LFM3-Audio (Liquid AI's LFM3-Audio-2B) on IRON: LFM3 (iron_lfm3.py) with its audio encoder, audio in and text out.

An application of amd/IRON kept in hwnpu, exported by ``iron_export_lm.py lfm3audio``. The
text model is ``iron_lfm3.Lfm3`` under the checkpoint's ``lfm``; the audio tower writes a clip's
soft tokens where the placed prompt chunk puts them in place of its ``<|audio|>`` placeholders
(IRON's ``Multimodal``). The speech output (the depth decoder and the codec) is not built.

The tower (``Lfm3AudioEncoder`` and ``Lfm3AudioProjector``) takes a clip's log-mel features,
which the host computes (hwnpu's lfm3audio features, as its Core ML encoder's input): ``(frames
+ 2, mels)``, a zero row before the clip's frames and zeros after them, up to ``max_frames``.
Then:

- the downsampler, three stride-2 3 x 3 convolutions over (time, mel), each then SiLU: the
  first from one channel (IRON's ``Conv2d``), the others depthwise (nine strided reads of the
  grid held with a zero border, each times its tap) each followed by a pointwise one; its
  output, channel major over the mels as torch flattens it, through ``lin_out``;
- an LFM2 stack (gated short convolutions, and causal attention within ``sliding_window``
  positions with q and k RMS-normed per head and RoPE), its final norm;
- the projector: LayerNorm, linear, GELU, linear.

As the checkpoint masks each stage's outputs past the clip, the row after the clip's at each
stage is zeroed (the last valid output of the next stage reads it, and none further), and the
stack runs over the clip's tokens only (``tokens`` rows): every call's per-call values are the
valid rows of the three stages, ``n1``, ``n2`` and ``tokens``.
"""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from ml_dtypes import bfloat16

import iron
import iron.functional as F
from iron import Precision, Scratchpad, lm
from iron.common.feature_map import Conv2d
from iron.lm import (
    Attention,
    Call,
    DecoderLayer,
    GatedMLP,
    LayerNorm,
    Linear,
    Multimodal,
    NoCache,
    RMSNorm,
    Tower,
    rope_angles,
)
from iron.operators.copy import Copy

from iron_lfm3 import Lfm3, Lfm3Config, ShortConv


def lfm2_ff_dim(ff: int, auto: bool, multiplier: float | None, multiple_of: int) -> int:
    """LFM2's feed-forward width: ``ff``, or with ``block_auto_adjust_ff_dim`` two thirds of it, scaled and rounded up."""
    if not auto:
        return ff
    ff = int(2 * ff / 3)
    if multiplier is not None:
        ff = int(multiplier * ff)
    return multiple_of * ((ff + multiple_of - 1) // multiple_of)


@dataclasses.dataclass(frozen=True)
class AudioConfig(lm.Config):
    """The audio encoder's shape under ``encoder`` (its LFM2 stack under ``encoder.lfm``) and the clip length."""

    json_prefix = "encoder."

    mels: int = dataclasses.field(metadata={"json": "dim_in"})
    conv_channels: int = dataclasses.field(metadata={"json": "dim_conv"})
    sliding_window: int = dataclasses.field(metadata={"json": "sliding_window"})
    hidden_size: int = dataclasses.field(metadata={"json": "lfm.hidden_size"})
    ff: int = dataclasses.field(metadata={"json": "lfm.intermediate_size"})
    num_hidden_layers: int = dataclasses.field(metadata={"json": "lfm.num_hidden_layers"})
    num_attention_heads: int = dataclasses.field(metadata={"json": "lfm.num_attention_heads"})
    num_key_value_heads: int = dataclasses.field(metadata={"json": "lfm.num_key_value_heads"})
    layer_types: tuple[str, ...] = dataclasses.field(metadata={"json": "lfm.layer_types", "convert": tuple})
    rope_theta: float = dataclasses.field(default=1e6, metadata={"json": "lfm.rope_parameters.rope_theta"})
    conv_L_cache: int = dataclasses.field(default=3, metadata={"json": "lfm.conv_L_cache"})
    auto_ff: bool = dataclasses.field(default=False, metadata={"json": "lfm.block_auto_adjust_ff_dim"})
    ff_multiplier: float | None = dataclasses.field(default=None, metadata={"json": "lfm.block_ffn_dim_multiplier"})
    ff_multiple_of: int = dataclasses.field(default=256, metadata={"json": "lfm.block_multiple_of"})
    eps: float = dataclasses.field(default=1e-5, kw_only=True, metadata={"json": "lfm.norm_eps"})
    text_dim: int = dataclasses.field(default=2048, kw_only=True, metadata={"json": "/lfm.hidden_size"})
    # 40.96 s of 10 ms frames: 512 soft tokens, a power of two as the short convolutions'
    # prompt chunk (their scratch and Checkpoints) take
    max_frames: int = lm.not_json(4096, kw_only=True)
    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)
    hidden_act: str = lm.not_json("silu", kw_only=True)
    attention_bias: bool = lm.not_json(False, kw_only=True)
    mlp_bias: bool = lm.not_json(False, kw_only=True)
    conv_bias: bool = lm.not_json(False, kw_only=True)
    rotary_dim = None

    def __post_init__(self):
        T = self.max_frames // 8
        if self.max_frames % 8 or T & (T - 1) or T < 64 or self.mels % 8:
            raise ValueError(f"AudioConfig: {self.max_frames} frames of {self.mels} mels do not halve thrice into a power of two of tokens")
        if set(self.layer_types) - {"conv", "full_attention"}:
            raise ValueError(f"AudioConfig: layer kinds {sorted(set(self.layer_types))}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def intermediate_size(self) -> int:
        return lfm2_ff_dim(self.ff, self.auto_ff, self.ff_multiplier, self.ff_multiple_of)

    @property
    def max_tokens(self) -> int:
        return self.max_frames // 8

    # what CausalConv1d and its Checkpoints take: one call of every token
    @property
    def prefill_chunk(self) -> int:
        return 1 << (self.max_tokens - 1).bit_length()

    @property
    def max_seq_len(self) -> int:
        return self.prefill_chunk


class Downsampler(iron.Graph):
    """``audio_encoder.downsample``, as the checkpoint names it, and the scratch its stages hold their grids in."""

    def __init__(self, c: AudioConfig):
        C, M, D = c.conv_channels, c.mels, c.hidden_size
        self.conv_in = SimpleNamespace(weight=iron.param(C, 1, 3, 3, dtype=np.float32), bias=iron.param(C))
        self.conv_mid = SimpleNamespace(weight=iron.param(C, 1, 3, 3), bias=iron.param(C))
        self.lin_mid = SimpleNamespace(weight=iron.param(C, C, 1, 1), bias=iron.param(C))
        self.conv_end = SimpleNamespace(weight=iron.param(C, 1, 3, 3), bias=iron.param(C))
        self.lin_end = SimpleNamespace(weight=iron.param(C, C, 1, 1), bias=iron.param(C))
        self.lin_out = SimpleNamespace(weight=iron.param(D, M * C // 8, 1), bias=iron.param(D))
        T1, T2 = c.max_frames // 2, c.max_frames // 4
        # each stage's output with a zero row above, one below, and a zero pixel either side
        self.grid1 = iron.state((T1 + 2, (M // 2 + 2) * C))
        self.grid2 = iron.state((T2 + 2, (M // 4 + 2) * C))
        self.zero1 = iron.param(np.zeros((1, (M // 2 + 2) * C), bfloat16))
        self.border1 = iron.param(np.zeros((T1, C), bfloat16))
        self.zero2 = iron.param(np.zeros((1, (M // 4 + 2) * C), bfloat16))
        self.c = c

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        c = self.c
        C, M = c.conv_channels, c.mels
        # torch's convolutions run over (mel, time); IRON's Conv2d over (rows, columns) of
        # channels-last rows: rows are time, columns mels, so the kernel's axes swap
        w = np.asarray(self.conv_in.weight, np.float32).transpose(0, 1, 3, 2)
        self.conv0 = Conv2d(w, M, stride=2, padding=1, border=1)

        def taps(conv):
            # (C, 1, kmel, ktime) -> a row per (ktime, kmel) tap, over (time, mel)
            k = np.asarray(conv.weight, np.float32).reshape(C, 3, 3).transpose(2, 1, 0)
            return iron.param(np.ascontiguousarray(k.reshape(9, C)).astype(bfloat16))

        self.taps_mid, self.taps_end = taps(self.conv_mid), taps(self.conv_end)

        def pointwise(lin):
            return iron.param(np.asarray(lin.weight, np.float32).reshape(C, C).astype(bfloat16))

        self.point_mid, self.point_end = pointwise(self.lin_mid), pointwise(self.lin_end)
        # lin_out reads (channel, mel) channel major; the rows here hold (mel, channel)
        D, F8 = c.hidden_size, M // 8
        w = np.asarray(self.lin_out.weight, np.float32).reshape(D, C, F8).transpose(0, 2, 1)
        self.out = iron.param(np.ascontiguousarray(w.reshape(D, F8 * C)).astype(bfloat16))

    def strided(self, grid, rows: int, pixels: int, taps, bias):
        """A depthwise stride-2 3 x 3 convolution of ``grid`` (zero borders held): ``(rows * pixels, C)``."""
        C = self.c.conv_channels
        line = grid.shape[1]
        y = None
        for kt in range(3):
            for kf in range(3):
                at = TensorAccessPattern(grid.shape, kt * line + kf * C, [rows, pixels, C], [2 * line, 2 * C, 1])
                term = Copy.step(grid, src=at).reshape(rows * pixels, C) * taps[3 * kt + kf]
                y = term if y is None else y + term
        return y + bias

    def body(self, features, n1, n2):
        c = self.c
        C, M, P = c.conv_channels, c.mels, c.precision
        T1, T2, T3 = c.max_frames // 2, c.max_frames // 4, c.max_frames // 8
        # the bias a channel's row (a row of a whole line of pixels is past what a broadcast
        # moves); its border pixels, which took the bias, are written zero again
        W1 = M // 2 + 2
        h = self.conv0(features, rows=n1).reshape(T1 * W1, C)
        h = F.silu(h + self.conv_in.bias)
        self.grid1[1 : 1 + T1] = h.reshape(T1, W1 * C)
        self.grid1[1 : 1 + T1, 0:C] = self.border1
        self.grid1[1 : 1 + T1, (W1 - 1) * C : W1 * C] = self.border1
        self.grid1[n1 + 1] = self.zero1
        h = self.strided(self.grid1, T2, M // 4, self.taps_mid, self.conv_mid.bias)
        h = F.silu(F.linear(h, self.point_mid, self.lin_mid.bias, precision=P))
        self.grid2[1 : 1 + T2, C : (M // 4 + 1) * C] = h.reshape(T2, (M // 4) * C)
        self.grid2[n2 + 1] = self.zero2
        h = self.strided(self.grid2, T3, M // 8, self.taps_end, self.conv_end.bias)
        h = F.silu(F.linear(h, self.point_end, self.lin_end.bias, precision=P))
        return F.linear(h.reshape(T3, (M // 8) * C), self.out, self.lin_out.bias, precision=P)


class AudioAttention(Attention):
    """LFM2's attention: q and k RMS-normed per head (``q_layernorm``), ``out_proj``, no gate.

    IRON's MHA takes heads of a multiple of 64 columns, and the encoder's are 96: each head is
    held padded with zeros to the next multiple (its q, k and v rows, its columns of
    ``out_proj``), which leaves the logits and the output as they are, with the scale of the
    real head and RoPE over its real columns. The per-head RMS norms see the zeros in their
    mean: their weights and epsilon are scaled by the real share of the columns to give the
    real norm.
    """

    def __init__(self, c: AudioConfig, i: int):
        super().__init__(c, i, norms="qk")
        self.real = c.head_dim
        self.padded = -(-c.head_dim // 64) * 64
        self.scaling = c.head_dim**-0.5
        self.eps = c.eps

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        D, P, E = self.real, self.padded, self.o_proj.out_features
        if P == D:
            return

        def rows(linear, heads):
            w = np.asarray(linear.weight, np.float32).reshape(heads, D, -1)
            out = np.zeros((heads, P, w.shape[2]), np.float32)
            out[:, :D] = w
            linear.weight = iron.param(out.reshape(heads * P, -1).astype(bfloat16))

        rows(self.q_proj, self.heads)
        rows(self.k_proj, self.groups)
        rows(self.v_proj, self.groups)
        w = np.asarray(self.o_proj.weight, np.float32).reshape(E, self.heads, D)
        out = np.zeros((E, self.heads, P), np.float32)
        out[:, :, :D] = w
        self.o_proj.weight = iron.param(out.reshape(E, self.heads * P).astype(bfloat16))
        share = D / P
        for name in ("q_norm", "k_norm"):
            old = getattr(self, name)
            norm = RMSNorm(P, self.eps * share)
            w = np.zeros(P, np.float32)
            w[:D] = np.asarray(old.weight, np.float32) * np.sqrt(share)
            norm.weight = iron.param(w.astype(bfloat16))
            setattr(self, name, norm)
        self.head_dim, self.rotary = P, D


class AudioLayer(DecoderLayer):
    """Layer ``i`` of the LFM2 stack: a short convolution or causal windowed attention, then SwiGLU."""

    def __init__(self, c: AudioConfig, i: int):
        if c.layer_types[i] == "conv":
            super().__init__(c, i, cache=NoCache())
            del self.self_attn, self.cache
            self.conv = ShortConv(c)
        else:
            super().__init__(c, i, AudioAttention(c, i), GatedMLP(c), NoCache(window=c.sliding_window - 1, causal=True))
            self.conv = None
            self.rope = "audio"

    def mix(self, h, call):
        if self.conv is not None:
            return self.conv(h, call)
        return super().mix(h, call)


class Projector(iron.Graph):
    """``audio_projector``: LayerNorm, linear, GELU, linear."""

    def __init__(self, c: AudioConfig):
        E, T = c.hidden_size, c.text_dim
        self.layer_norm = LayerNorm(E, 1e-5)
        self.linear_1 = Linear(E, T, True, precision=c.precision)
        self.linear_2 = Linear(T, T, True, precision=c.precision)


class AudioTower(Tower):
    """The audio encoder and the projector over a clip of up to ``max_frames`` frames.

    Its input ``wave`` is the clip's log-mel features, ``(max_frames + 2, mels)``.
    """

    layer_class = AudioLayer
    renames = {
        "operator_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "feed_forward.w1": "mlp.gate_proj",
        "feed_forward.w3": "mlp.up_proj",
        "feed_forward.w2": "mlp.down_proj",
        "q_layernorm": "q_norm",
        "k_layernorm": "k_norm",
        "self_attn.out_proj": "self_attn.o_proj",
    }

    def __init__(self, config: AudioConfig, weights):
        c = config
        # Set before Tower.__init__() so a recording names its layers apart from the text model's.
        self.audio_encoder = SimpleNamespace(lfm=SimpleNamespace())
        Tower.__init__(self, config, c.max_tokens, c.max_tokens)
        # the released checkpoint's names (transformers maps them to its modules' on loading)
        self.audio_encoder.lfm.layers = self.layers
        self.load_layers(weights, "encoder.lfm.lfm.layers.")
        self.audio_encoder.downsample = Downsampler(c)
        self.audio_encoder.downsample.load_state_dict(weights, path="encoder.downsample.")
        self.audio_encoder.lfm.embedding_norm = RMSNorm(c.hidden_size, c.eps)
        self.audio_encoder.lfm.embedding_norm.load_state_dict(weights, path="encoder.lfm.lfm.embedding_norm.")
        # the scratch a call of the short convolutions stages its rows in, shared as CausalLM shares it
        window = iron.state((2, c.prefill_chunk, c.hidden_size))
        tail = iron.state((c.conv_L_cache - 1, c.hidden_size))
        for layer in self.layers:
            if layer.conv is not None:
                layer.conv.conv.window, layer.conv.conv.tail = window, tail
        self.audio_projector = Projector(c)
        for name, at in (("layer_norm", 0), ("linear_1", 1), ("linear_2", 3)):
            getattr(self.audio_projector, name).load_state_dict(weights, path=f"mlp.model.{at}.")
        self.angles_table = iron.param(rope_angles(c.head_dim, c.max_tokens, c.rope_theta).astype(bfloat16))

    def body(self, wave, *, n1: Scratchpad[np.int32], n2: Scratchpad[np.int32], tokens: Scratchpad[np.int32]):
        """The soft tokens of the features ``wave``, ``(max_tokens, text_dim)``, the first ``tokens`` valid.

        ``n1`` and ``n2`` are the valid rows after the first and the second stride-2
        convolution, ``ceil(frames / 2)`` and ``ceil(n1 / 2)``; ``tokens`` is ``ceil(n2 / 2)``.
        """
        c = self.config
        e = self.audio_encoder
        h = e.downsample(wave, n1, n2)[:tokens]
        angles = Copy.step(self.angles_table)[:tokens]
        call = Call(prompt=True, start=0, rows=tokens, position=tokens - 1, ropes={"audio": (angles, angles)})
        for layer in self.layers:
            h = layer(h, call)
        h = e.lfm.embedding_norm(h)
        p = self.audio_projector
        return p.linear_2(F.gelu(p.linear_1(p.layer_norm(h))))

    def shapes(self) -> list[dict]:
        c = self.config
        return [dict(wave=((c.max_frames + 2, c.mels), bfloat16))]

    def longest(self) -> dict:
        c = self.config
        return dict(n1=c.max_frames // 2, n2=c.max_frames // 4, tokens=c.max_tokens)

    def soft(self, values: dict) -> int:
        return values["tokens"]

    def inputs(self, features) -> tuple[np.ndarray, dict]:
        """``features`` ``(frames, mels)`` float32: the buffer and its per-call values."""
        c = self.config
        f = np.asarray(features, np.float32)
        n = f.shape[0]
        if not 0 < n <= c.max_frames:
            raise ValueError(f"{n} frames past {c.max_frames}")
        buf = np.zeros((c.max_frames + 2, c.mels), np.float32)
        buf[1 : n + 1] = f
        n1 = -(-n // 2)
        n2 = -(-n1 // 2)
        return buf.astype(bfloat16), dict(n1=n1, n2=n2, tokens=-(-n2 // 2))


def _token(path: Path, text: str) -> int:
    """The id of an added token of the checkpoint's tokenizer.json."""
    tok = json.loads((path / "tokenizer.json").read_text())
    for t in tok.get("added_tokens", []):
        if t["content"] == text:
            return int(t["id"])
    raise ValueError(f"{path}: the tokenizer has no {text}")


@dataclasses.dataclass(frozen=True)
class Lfm3AudioConfig(Lfm3Config):
    """``Lfm3Config`` under ``lfm``, the clip's placeholder and the encoder's shape."""

    json_prefix = "lfm."

    audio_token_id: int = lm.not_json(-1, kw_only=True)
    audio: AudioConfig | None = lm.not_json(None, kw_only=True)

    @classmethod
    def from_checkpoint(cls, path) -> "Lfm3AudioConfig":
        path = Path(path)
        d = path if path.is_dir() else path.parent
        text = super().from_checkpoint(path)
        return dataclasses.replace(text, audio=AudioConfig.from_checkpoint(path), audio_token_id=_token(d, "<|audio|>"))


class Lfm3Audio(Multimodal):
    """LFM3-Audio with audio in: ``Lfm3`` with the audio tower, a clip's soft tokens at its placeholders."""

    config = Lfm3AudioConfig
    hwnpu_kind = "lfm3audio"  # the multimodal manifest iron_export_lm.py writes

    def __init__(self, config: Lfm3AudioConfig, weights):
        assert config.audio is not None
        self.config, self.weights = config, weights
        # the released checkpoint names the text model lfm.model, the encoder encoder, the
        # projector mlp.model
        towers = {config.audio_token_id: AudioTower(config.audio, weights)}
        text = Lfm3(config, weights.lfm)
        super().__init__(text, towers)
