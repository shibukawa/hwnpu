"""LFM3-VL (Liquid AI's LFM3-VL-2B) on IRON: LFM3 (iron_lfm3.py) with its vision tower, for one image size.

An application of amd/IRON kept in hwnpu, exported by ``iron_export_lm.py lfm3vl``. The
text model is ``iron_lfm3.Lfm3`` under ``text_config``; the vision tower writes an image's
soft tokens where the placed prompt chunk puts them in place of its ``<|image|>``
placeholders (IRON's ``Multimodal``, as Gemma 4's).

The tower (``Lfm3VlVisionModel`` and ``Lfm3VlMultiModalProjector``) takes an image of a fixed
size (``VisionConfig.width`` by ``height``, 512 x 512 by default: the host resizes to it, as
hwnpu's Core ML programs do) as its patches, normalized as the processor does, in raster order,
a row of ``3 * patch_size**2`` each (a still image's temporal slices summed into the
patchifier). Then:

- the patchifier (a linear) and the embedding RMS norm;
- pre-norm layers of two kinds, each with a SwiGLU MLP after: gated 3 x 3 depthwise
  convolutions over the patch grid (``in_proj`` to B, C and x; the convolution over B * x; C
  times it; ``out_proj``), and bidirectional attention with q and k RMS-normed per head and
  axial 2D RoPE (pair ``k`` of a head turns by the patch's row for ``k`` under a quarter of
  the head, by its column above, each axis with the frequencies of a half-head RoPE);
- the projector: a LayerNorm, each 2 x 2 block of patches as one row, linear, GELU, linear.

The convolution is nine shifted reads of the grid held with a zero row band above and below
(a raster grid's vertical neighbours are a row of patches away), each times its tap, the
horizontal wrap zeroed by a mask of the first and the last column. The 2 x 2 blocks are read
out of the raster grid by one strided copy.
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

import iron
import iron.functional as F
from iron import Precision, lm
from aie.helpers.taplib import TensorAccessPattern
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
    subtree,
)
from iron.operators.copy import Copy

from iron_lfm3 import Lfm3, Lfm3Config


@dataclasses.dataclass(frozen=True)
class VisionConfig(lm.Config):
    """The vision tower's shape under ``vision_config``, the projector's, and the image size."""

    json_prefix = "vision_config."

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    patch_size: int
    layer_types: tuple[str, ...] = dataclasses.field(metadata={"convert": tuple})
    temporal_patch_size: int = 1
    in_channels: int = 3
    conv_kernel_size: int = 3
    rope_theta: float = dataclasses.field(
        default=1e6, metadata={"json": "rope_parameters.rope_theta"}
    )
    eps: float = dataclasses.field(
        default=1e-5, kw_only=True, metadata={"json": "norm_eps"}
    )
    text_dim: int = dataclasses.field(
        default=2048, kw_only=True, metadata={"json": "/text_config.hidden_size"}
    )
    projector_hidden_size: int = dataclasses.field(
        default=4096, kw_only=True, metadata={"json": "/projector_hidden_size"}
    )
    projector_eps: float = dataclasses.field(
        default=1e-6, kw_only=True, metadata={"json": "/projector_layer_norm_eps"}
    )
    downsample_factor: int = dataclasses.field(
        default=2, kw_only=True, metadata={"json": "/downsample_factor"}
    )
    width: int = lm.not_json(512, kw_only=True)
    height: int = lm.not_json(512, kw_only=True)
    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)
    hidden_act: str = lm.not_json("silu", kw_only=True)
    attention_bias: bool = lm.not_json(False, kw_only=True)
    mlp_bias: bool = lm.not_json(False, kw_only=True)

    def __post_init__(self):
        side = 2 * self.patch_size
        if self.width % side or self.height % side:
            raise ValueError(f"VisionConfig: a {self.width}x{self.height} image is no whole {side}-pixel blocks")
        if self.conv_kernel_size != 3 or self.downsample_factor != 2 or self.in_channels != 3:
            raise ValueError("VisionConfig: only 3x3 convolutions, 2x2 blocks and RGB are built here")
        if set(self.layer_types) - {"conv", "full_attention"}:
            raise ValueError(f"VisionConfig: layer kinds {sorted(set(self.layer_types))}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    rotary_dim = None

    @property
    def grid(self) -> tuple[int, int]:
        """The patches down and across."""
        return self.height // self.patch_size, self.width // self.patch_size

    @property
    def patches(self) -> int:
        gh, gw = self.grid
        return gh * gw

    @property
    def tokens(self) -> int:
        """An image's soft tokens: its 2 x 2 blocks."""
        return self.patches // 4

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size**2


def axial_angles(c: VisionConfig) -> np.ndarray:
    """The RoPE table, ``(patches, head_dim)``, cos and sin of each pair interleaved, raster order.

    Pair ``k`` turns by the row for ``k`` under ``head_dim // 4``, by the column above; each
    axis by ``theta ** (-2f / (head_dim // 2))`` for its own ``f``.
    """
    gh, gw = c.grid
    D = c.head_dim
    half, quarter = D // 2, D // 4
    spatial = D // 2
    inv = 1.0 / (c.rope_theta ** (np.arange(0, spatial, 2, dtype=np.float64) / spatial))  # quarter of them
    rows = np.repeat(np.arange(gh), gw).astype(np.float64)
    cols = np.tile(np.arange(gw), gh).astype(np.float64)
    a = np.concatenate([rows[:, None] * inv[None], cols[:, None] * inv[None]], axis=1)  # (n, half)
    assert a.shape[1] == half and inv.size == quarter
    out = np.empty((gh * gw, D), np.float32)
    out[:, ::2] = np.cos(a)
    out[:, 1::2] = np.sin(a)
    return out


class SpatialConv(iron.Graph):
    """The gated 3 x 3 depthwise convolution over the patch grid, named as the checkpoint's ``conv``.

    ``grid`` is the tower's shared state of the grid with ``gw + 1`` zero rows above and below,
    and ``edges`` its masks of the first and the last column.
    """

    def __init__(self, c: VisionConfig, grid, edges):
        E = self.width = c.hidden_size
        self.shape = c.grid
        self.in_proj = Linear(E, 3 * E, False, precision=c.precision)
        self.spatial_conv = SimpleNamespace(weight=iron.param(E, 1, 3, 3))
        self.out_proj = Linear(E, E, False, precision=c.precision)
        self.grid, self.edges = grid, edges

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        w = np.asarray(self.spatial_conv.weight, np.float32).reshape(self.width, 9)
        # A row per tap (dy, dx), as RowwiseMul takes it.
        self.taps = iron.param(np.ascontiguousarray(w.T).astype(bfloat16))

    def body(self, x):
        E, (gh, gw) = self.width, self.shape
        n = gh * gw
        b, cg, v = self.in_proj.split(x, E, E, E)
        top = gw + 1
        self.grid[top : top + n] = b * v
        y = None
        for dx in range(3):
            column = None
            for dy in range(3):
                at = top + (dy - 1) * gw + (dx - 1)
                term = self.grid[at : at + n] * self.taps[3 * dy + dx]
                column = term if column is None else column + term
            if dx != 1:
                column = column * self.edges[dx // 2]
            y = column if y is None else y + column
        return self.out_proj(cg * y)


class VisionAttention(Attention):
    """Bidirectional attention over the patches, q and k RMS-normed per head, ``out_proj`` its output."""

    def __init__(self, c: VisionConfig, i: int):
        super().__init__(c, i, norms="qk")


class VisionLayer(DecoderLayer):
    """Layer ``i`` of the tower: a spatial convolution or attention, then SwiGLU, each pre-normed."""

    def __init__(self, c: VisionConfig, i: int, grid=None, edges=None):
        if c.layer_types[i] == "conv":
            super().__init__(c, i, cache=NoCache())
            del self.self_attn, self.cache
            self.conv = SpatialConv(c, grid, edges)
        else:
            super().__init__(c, i, VisionAttention(c, i), GatedMLP(c), NoCache())
            self.conv = None
            self.rope = "vision"

    def mix(self, h, call):
        if self.conv is not None:
            return self.conv(h)
        return super().mix(h, call)


class Projector(iron.Graph):
    """``multi_modal_projector``: LayerNorm, each 2 x 2 block as a row, linear, GELU, linear."""

    def __init__(self, c: VisionConfig):
        E = c.hidden_size
        self.layer_norm = LayerNorm(E, c.projector_eps)
        self.linear_1 = Linear(4 * E, c.projector_hidden_size, True, precision=c.precision)
        self.linear_2 = Linear(c.projector_hidden_size, c.text_dim, True, precision=c.precision)


class VisionTower(Tower):
    """The vision tower and the projector over one image of ``config``'s size.

    Its input ``rgb`` is the image's patches, ``(patches, patch_dim)``, in raster order.
    """

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

    def layer_class(self, c: VisionConfig, i: int):
        """Layer ``i``, sharing the tower's grid state and edge masks (Encoder builds its layers so)."""
        return VisionLayer(c, i, self.grid_state, self.edges)

    def __init__(self, config: VisionConfig, weights):
        c = config
        gh, gw = c.grid
        n, E = c.patches, c.hidden_size
        self.grid_state = iron.state((n + 2 * (gw + 1), E))
        first = np.ones((n, E), np.float32)
        first[np.arange(n) % gw == 0] = 0  # dx = 0 reads the column before: none for the first
        last = np.ones((n, E), np.float32)
        last[np.arange(n) % gw == gw - 1] = 0
        self.edges = [iron.param(first.astype(bfloat16)), iron.param(last.astype(bfloat16))]
        # Set before Tower.__init__() so a recording names its layers "vision_tower.layers.N",
        # apart from the text model's "layers.N" (as Gemma 4's tower).
        self.vision_tower = SimpleNamespace()
        Tower.__init__(self, config, n, c.tokens)
        self.vision_tower.layers = self.layers
        self.vision_tower.embedding_norm = RMSNorm(E, c.eps)
        self.multi_modal_projector = Projector(c)
        self.load_layers(weights, "vision_tower.layers.")
        self.vision_tower.embedding_norm.load_state_dict(weights, path="vision_tower.embedding_norm.")
        self.multi_modal_projector.load_state_dict(weights, path="multi_modal_projector.")
        pw = weights.vision_tower.patchifier.weight
        raw = np.asarray(pw, np.float32).reshape(E, c.in_channels, c.temporal_patch_size, c.patch_size**2)
        self.patchifier = iron.param(raw.sum(axis=2).reshape(E, c.patch_dim).astype(bfloat16))
        self.angles_table = iron.param(axial_angles(c).astype(bfloat16))

    def body(self, rgb):
        c = self.config
        h = F.linear(rgb, self.patchifier, precision=c.precision)
        h = self.vision_tower.embedding_norm(h)
        angles = Copy.step(self.angles_table)
        call = Call(prompt=True, start=0, rows=c.patches, position=None, ropes={"vision": (angles, angles)})
        for layer in self.layers:
            h = layer(h, call)
        p = self.multi_modal_projector
        h = p.layer_norm(h)
        gh, gw = c.grid
        E = c.hidden_size
        blocks = TensorAccessPattern((c.patches, E), 0, [gh // 2, gw // 2, 2, 2 * E], [2 * gw * E, 2 * E, gw * E, 1])
        h = Copy.step(h, src=blocks).reshape(c.tokens, 4 * E)
        h = F.gelu(p.linear_1(h))
        return p.linear_2(h)

    def shapes(self) -> list[dict]:
        c = self.config
        return [dict(rgb=((c.patches, c.patch_dim), bfloat16))]

    def longest(self) -> dict:
        return {}

    def soft(self, values: dict) -> int:
        return self.config.tokens

    def inputs(self, patches) -> tuple[np.ndarray, dict]:
        return np.asarray(patches, bfloat16), {}


@dataclasses.dataclass(frozen=True)
class Lfm3VlConfig(Lfm3Config):
    """``Lfm3Config`` under ``text_config``, the image's tokens and the tower's shape."""

    json_prefix = "text_config."

    image_token_id: int = dataclasses.field(default=125030, kw_only=True, metadata={"json": "/image_token_id"})
    vision_start_token_id: int = dataclasses.field(default=125009, kw_only=True, metadata={"json": "/vision_start_token_id"})
    vision_end_token_id: int = dataclasses.field(default=125010, kw_only=True, metadata={"json": "/vision_end_token_id"})
    vision: VisionConfig | None = lm.not_json(None, kw_only=True)

    @classmethod
    def from_checkpoint(cls, path) -> "Lfm3VlConfig":
        text = super().from_checkpoint(path)
        return dataclasses.replace(text, vision=VisionConfig.from_checkpoint(path))


class Lfm3Vl(Multimodal):
    """LFM3-VL whole: ``Lfm3`` with the vision tower, an image's soft tokens at its placeholders."""

    config = Lfm3VlConfig
    hwnpu_kind = "lfm3vl"  # the multimodal manifest iron_export_lm.py writes

    def __init__(self, config: Lfm3VlConfig, weights):
        assert config.vision is not None
        self.config, self.weights = config, weights
        towers = {config.image_token_id: VisionTower(config.vision, weights.model)}
        text = Lfm3(config, subtree(weights, model=weights.model.language_model))
        super().__init__(text, towers)
