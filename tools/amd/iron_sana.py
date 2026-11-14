#!/usr/bin/env python3
"""Sana (NVlabs: Sana 600M, SANA 1.5 1.6B) on IRON: the graphs cmd/hwnpu-image runs on AMD's NPU.

An application of amd/IRON kept in hwnpu, as iron_anima.py is. Sana is a linear DiT: each block's
self-attention is ReLU linear attention (no softmax, so its cost grows with the tokens, not
their square), its cross-attention a softmax attention to the prompt, its MLP a gated 1 x 1,
depthwise 3 x 3, 1 x 1 convolution (GLUMBConv) over the token grid. The prompt is read by Gemma
2 2B, the latents are DC-AE's (32 channels, a 32nd of the image's sides: 1024 tokens at 1024 x
1024). Sana runs 20 or so flow DPM-Solver steps with classifier-free guidance (two DiT calls a
step). Sana-Sprint (the guidance embedded, 1 to 4 SCM steps) is not supported: its pictures were
far below Sana's (design.md 16.148).

The graphs, each exported by iron_graph.py into a directory of its own:

  text        Gemma 2 2B, causal, its last hidden state after the final norm, on the prompt's
              embedding rows times sqrt(hidden) (which the host looks up): the instruction's
              first rows held as keys and values (Prefix), the rest in chunks of ``rows`` from
              a per-call ``start`` (--no-prefix: every row in one call); shared by every Sana
              of a VAE generation (the same Gemma 2)
  caption     the DiT's caption projection and norm of the selected 300 rows (padded to 512)
  dit_in      the patch embedding (a 1 x 1 convolution) and the timestep embedding and its
              six modulations
  dit_blocks  ``group`` blocks: one ELF, a bank image per group
  dit_out     the final modulated norm and projection to the 32 latent channels
  vae/s*      DC-AE's decoder in stages

The host (internal/imagegen/sana) does what is not a tensor computation: the tokenizer, the
instruction prefix, the embedding rows, the rows Sana keeps, the timestep's sinusoid, the
scheduler's steps, noise and the PNG.

Linear attention without division. IRON has no elementwise division; each head's output is
``N / D``, ``N = relu(q) (relu(K)^T V)`` and ``D = relu(q) . sum(relu(K))``. The value rows
are laid out 64 to a head, the head's 32 values, a constant ``c`` and zeros, so that one
product gives each head's row ``[N, c D, 0...]``; RMS-normalizing that row of 64 gives
``8 N / (c D)`` to within ``(N / D)^2 / c^2`` (``c D`` dominates), which the norm's weight
``c / 8`` turns into ``N / D``. A head whose ``D`` is 0 gives 0, as the reference's eps does.

  python tools/amd/iron_sana.py ~/models/Sana_600M_1024px_diffusers OUT --size 1024   # or --width 1152 --height 896

See docs/amd-npu-models.md.
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import iron
import iron.functional as F
from iron import Precision, Scratchpad
from iron.operators.copy import Copy
from iron.operators.mha import MHA
from iron.operators.relu import ReLU

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_anima import RandomWeights, Safetensors, bf, command_boundary, pipelines, tap, variant_bank  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402


@dataclasses.dataclass
class Precisions:
    """The matmuls' precision of each part (set from the command line). The products of the
    linear attention, whose operands are both activations, take HIGHEST (bf16 operands,
    float32 sums)."""

    text: Precision = Precision.HIGHEST
    dit: Precision = Precision.MEDIUM
    attn: Precision = Precision.HIGHEST
    vae: Precision = Precision.MEDIUM
    # the VAE's linear attentions: bfp16 shares an exponent along K only, so the constant row
    # (c = 4096) never shares one with the values; the pictures are those of HIGHEST (51.6 dB)
    vae_attn: Precision = Precision.MEDIUM


PREC = Precisions()

# The text encoder's projections: "q6" (iron_q4's 6.5 bits a weight, dequantized to bf16 on the
# device before each GEMM, which keeps PREC.text), "q8", "q4" or "bf16". q6 draws the pictures q8
# does (its rows 3.5 % off float32's, q8's 1.4 %, design.md 16.138) in 1.7 GB to 2.3; q4's 15 %
# changes the composition.
TEXT_FMT = {"fmt": "q6"}


def text_param(w):
    """An iron.param of the text encoder's weight w in TEXT_FMT; q8's blocks take 512
    in-features, so its in-features are padded with zero columns to a multiple of 512 (the input
    is copied into as many zero-padded columns: text_linear)."""
    if TEXT_FMT["fmt"] == "bf16":
        return iron.param(bf(w))
    import iron_q4

    w = f32(w)
    k = -(-w.shape[1] // 512) * 512
    if k != w.shape[1]:
        w = np.concatenate([w, np.zeros((w.shape[0], k - w.shape[1]), np.float32)], axis=1)
    return iron_q4.q_param(w, TEXT_FMT["fmt"])


def text_linear(x, w, pads: dict | None = None):
    """``x @ w.T`` at PREC.text, w a text_param: q8 dequantized to bf16 on the device first,
    x copied into a zero-padded state of ``pads`` (by width) when w's in-features are more."""
    import iron_q4

    e = iron_q4._Q4.get(id(w))
    if e is None or e[0] is not w:
        return F.linear(x, w, precision=PREC.text)
    _, _, k, fmt = e
    if x.shape[1] != k:
        L, K0 = x.shape
        st = pads[k]
        Copy.step(x, st, dst=tap((st.shape[0], k), 0, [L, K0], [k, 1]))
        x = st[0:L]
    _, QDequant = iron_q4.operators()
    wd = QDequant.step(w.reshape(-1, iron_q4.FORMATS[fmt]), fmt=fmt, K=k)
    return F.linear(x, wd, precision=PREC.text)

# IRON's MHA rounds its scale times log2(e) to bf16 (up to 0.4 %, a temperature error on large
# scores). MHA_SCALE makes that product exactly 1; the attention's own scale divided by it is
# folded into the queries' weights in float32 (mha_fold).
MHA_SCALE = math.log(2)


def mha_fold(scale: float) -> float:
    """The factor of the queries for an attention of ``scale`` run at MHA_SCALE."""
    return scale / MHA_SCALE


# The constant of the value rows: the error of a head's output is ((N/D) / c)^2 relative.
LIN_C = 4096.0


class Sharded:
    """The tensors of a sharded checkpoint (model.safetensors.index.json) by name."""

    def __init__(self, d: Path, prefix: str = "", random: bool = False):
        index = d / "model.safetensors.index.json"
        files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
        cls = RandomWeights if random else Safetensors
        self.parts = [cls(d / f, prefix) for f in files]

    def __contains__(self, name: str) -> bool:
        return any(name in p for p in self.parts)

    def __getitem__(self, name: str) -> np.ndarray:
        for p in self.parts:
            if name in p:
                return p[name]
        raise KeyError(name)


def f32(x) -> np.ndarray:
    return np.array(x, np.float32)  # a copy: the checkpoint is mapped read-only


# --- linear attention ------------------------------------------------------------------


class LinearAttnConsts:
    """What every linear attention over ``heads`` heads of 32 at ``n`` tokens shares: the
    block mask of the heads' rows, the constant rows of the values, the rows' norm weight."""

    def __init__(self, heads: int, n: int, c: float = LIN_C):
        H = heads
        mask = np.zeros((64 * H, 32 * H), np.float32)
        for h in range(H):
            mask[64 * h : 64 * h + 33, 32 * h : 32 * h + 32] = 1
        self.mask = iron.param(bf(mask))
        const = np.zeros((64 * H, n), np.float32)
        const[32::64] = c
        self.const = iron.param(bf(const))
        sel = np.zeros(64, np.float32)
        sel[:32] = c / 8
        self.sel = iron.param(bf(sel))
        self.heads, self.n = H, n


def value_rows(w: np.ndarray, heads: int) -> np.ndarray:
    """A value projection ``(32 h, in)`` laid out 64 rows to a head (zeros after the 32), for
    ``heads`` heads (h or more: the rest zeros)."""
    out = np.zeros((64 * heads, w.shape[1]), np.float32)
    for h in range(w.shape[0] // 32):
        out[64 * h : 64 * h + 32] = w[32 * h : 32 * h + 32]
    return out


def out_cols(w: np.ndarray, heads: int) -> np.ndarray:
    """An output projection ``(out, 32 h)`` taking the 64-wide rows of ``heads`` heads (h or
    more: the rest zeros)."""
    out = np.zeros((w.shape[0], 64 * heads), np.float32)
    for h in range(w.shape[1] // 32):
        out[:, 64 * h : 64 * h + 32] = w[:, 32 * h : 32 * h + 32]
    return out


def linear_attention(q, kt, vt, k: LinearAttnConsts, precision=None):
    """Each head's ``relu(q) (relu(K)^T V) / relu(q) . sum(relu(K))``, 64 columns to a head.

    ``q`` (n, 32 H) and ``kt`` (32 H, n) past their ReLU; ``vt`` (64 H, n), the values 64 rows
    to a head (value_rows), the constant rows added here.
    """
    n, H = q.shape[0], k.heads
    P = precision or PREC.attn
    w = F.linear(vt + k.const, kt, precision=P)  # (64 H, 32 H): V^T K and c sum(K)
    o = F.linear(q, w * k.mask, precision=P)  # (n, 64 H): [N, c D, 0...] a head
    return F.rms_norm(o.reshape(n * H, 64), (64,), k.sel, 1e-30).reshape(n, 64 * H)


# --- the DiT -------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DitConfig:
    """SanaTransformer2DModel's shape (transformer/config.json) at one image size."""

    width: int = 1024
    height: int = 1024
    hidden: int = 1152
    heads: int = 36
    layers: int = 28
    cross_heads: int = 16
    cross_head_dim: int = 72
    mlp: int = 2880
    caption: int = 2304
    latent: int = 32
    qk_norm: bool = True
    text_rows: int = 512  # the 300 rows Sana keeps of the prompt, padded
    group: int = 4
    vae_scale: int = 32

    @property
    def gw(self) -> int:
        return self.width // self.vae_scale

    @property
    def gh(self) -> int:
        return self.height // self.vae_scale

    @property
    def tokens(self) -> int:
        """The latent pixels: the tokens that are the picture's."""
        return self.gw * self.gh

    @property
    def ntok(self) -> int:
        """The tokens' rows as the graphs hold them: whole 256 rows (the GEMMs'), the rest
        padding (masked out of the linear attention's sums)."""
        return -(-self.tokens // 256) * 256

    @property
    def att_heads(self) -> int:
        """The self-attention's heads as the graph runs them (the rest zeros), so that its
        products with a weight on the left (MEDIUM, flm.ProjectionGEMM) have whole 256 rows:
        the values' 64 H rows, and without the q and k norms the keys' 32 H."""
        m = 4 if self.qk_norm else 8
        return -(-self.heads // m) * m

    @property
    def att_dim(self) -> int:
        return 32 * self.att_heads

    @property
    def cross_pad(self) -> int:
        """A cross-attention head's columns as MHA takes them (a multiple of 64)."""
        return -(-self.cross_head_dim // 64) * 64

    @classmethod
    def of(cls, cfg: dict, width: int, height: int, group: int) -> "DitConfig":
        E = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        assert cfg["attention_head_dim"] == 32 and cfg.get("patch_size", 1) == 1, cfg
        assert cfg.get("interpolation_scale") is None, "a positional embedding is not supported"
        if cfg.get("guidance_embeds"):
            raise SystemExit("Sana-Sprint (guidance_embeds) is not supported: its pictures were far below Sana's")
        if width % 32 or height % 32:
            raise SystemExit(f"{width}x{height}: not multiples of 32 (DC-AE's)")
        return cls(width=width, height=height, hidden=E, heads=cfg["num_attention_heads"], layers=cfg["num_layers"],
                   cross_heads=cfg["num_cross_attention_heads"], cross_head_dim=cfg["cross_attention_head_dim"],
                   mlp=int(cfg["mlp_ratio"] * E), caption=cfg["caption_channels"], latent=cfg["in_channels"],
                   qk_norm=cfg.get("qk_norm") is not None, group=group)


# The depthwise convolution's grid: the tokens' rows of W, each padded with one zero pixel
# after it, a zero row above and below and a zero pixel before the first.


def dw_rows(c: DitConfig) -> int:
    return (c.gh + 2) * (c.gw + 1) + 2


def dw_band(c: DitConfig) -> int:
    """The grid row of token (0, 0)."""
    return 1 + (c.gw + 1)


# GLUMBConv's depthwise convolutions: "op" (iron_dwconv.DWConv3, one operator a branch, the
# channels padded with zero weights to whole bands over the columns; design.md 16.200) or
# "eltwise" (a row product and a sum a tap: 18 operators a branch, as before).
DW = {"form": "op"}
DW_COLS = 8
# The VAE's EfficientViT depthwise convolutions (5 x 5 on q, k, v; 3 x 3 in its GLUMBConv): "op"
# DWConv3 where its bands fit the channels, else "eltwise" as before.
VAE_DW = {"form": "op"}


def dw_band_c(c: "DitConfig") -> int:
    """DWConv3's band: the widest of 96, 64, 32 channels whose padding of the MLP is the least."""
    return min((96, 64, 32), key=lambda b: (-(-c.mlp // (DW_COLS * b)) * DW_COLS * b, -b))


def mlp_pad(c: "DitConfig") -> int:
    """The MLP's channels as the graphs hold them: padded to whole DWConv3 bands (zeros)."""
    if DW["form"] != "op":
        return c.mlp
    b = DW_COLS * dw_band_c(c)
    return -(-c.mlp // b) * b


def pad_rows(m, n: int):
    """``m`` with zero rows to ``n`` (or zero entries, a vector)."""
    m = np.asarray(m, np.float32)
    return np.concatenate([m, np.zeros((n - m.shape[0],) + m.shape[1:], np.float32)])


def cross_rows(w: np.ndarray, b, heads: int, dim: int, pad: int):
    """A projection of ``heads`` heads of ``dim`` laid out ``pad`` columns to a head."""
    o = np.zeros((heads * pad, w.shape[1]), np.float32)
    ob = np.zeros(heads * pad, np.float32)
    for h in range(heads):
        o[h * pad : h * pad + dim] = w[h * dim : (h + 1) * dim]
        if b is not None:
            ob[h * pad : h * pad + dim] = b[h * dim : (h + 1) * dim]
    return o, ob


def cross_norm(w: np.ndarray, heads: int, dim: int, pad: int) -> tuple[np.ndarray, float]:
    """An RMS norm across the heads' ``heads * dim`` columns, on the padded layout: the weight
    times sqrt(E / P) (the zeros count in the mean), the eps times E / P."""
    E, P = heads * dim, heads * pad
    o = np.zeros(P, np.float32)
    for h in range(heads):
        o[h * pad : h * pad + dim] = w[h * dim : (h + 1) * dim]
    return o * math.sqrt(E / P), 1e-5 * E / P


class Shared:
    """What every block of a dit_blocks graph shares."""

    def __init__(self, c: DitConfig):
        n, Ch = c.ntok, mlp_pad(c)
        self.lin = LinearAttnConsts(c.att_heads, n)
        if c.ntok != c.tokens:
            # the keys' columns of the padding rows: zeros, so no sum of the attention counts them
            m = np.zeros((c.att_dim, n), np.float32)
            m[:, : c.tokens] = 1
            self.kmask = iron.param(bf(m))
            self.fbuf = iron.state((n, Ch))
        else:
            self.kmask = None
        self.one = iron.param(bf(np.ones(c.hidden)))
        self.eye = iron.param(bf(np.eye(c.att_dim))) if c.qk_norm else None
        self.grid_a = iron.state((dw_rows(c), Ch))
        self.grid_g = iron.state((dw_rows(c), Ch))
        self.cross = iron.state((n, c.cross_heads * c.cross_pad))


# The DiT's projections that are a GEMM's B (set from the command line): "bfp16" packs those that
# tile (iron_q4.bfp16_tiles: out-features by 64, in-features by 128) on the host as
# flm.ProjectionGEMM reads them (9 bits a weight, no conversion in the GEMM; design.md 16.200);
# "bf16" keeps them all bf16. The others (the values' and, without q and k norms, the keys', a
# GEMM's A) stay bf16.
DIT_FMT = {"proj": "bfp16"}


def dproj(w):
    """An iron.param of a DiT projection used as a GEMM's B: bfp16 where it tiles, else bf16."""
    w = np.asarray(w, np.float32)
    if DIT_FMT["proj"] == "bfp16":
        import iron_q4

        if iron_q4.bfp16_tiles(w.shape[1], w.shape[0]):
            return iron_q4.q_param(w, "bfp16")
    return iron.param(bf(w))


class SanaBlock(iron.Graph):
    """A Sana block: adaLN-single (a learned table plus the timestep's six rows), linear
    self-attention, cross-attention to the caption, GLUMBConv."""

    def __init__(self, c: DitConfig, w: Safetensors | None, i: int, sh: Shared):
        E, H, Ch, Cp = c.hidden, c.att_heads, c.mlp, mlp_pad(c)
        A = c.att_dim
        self.c, self.sh = c, sh
        p = f"transformer_blocks.{i}."
        self.aeps = 1e-5 * E / A  # the q and k norms' eps on the padded heads
        if w is None:
            self.sst = iron.param(6 * E)
            self.q = iron.param(A, E)
            self.k = iron.param(A, E)
            self.v = iron.param(64 * H, E)
            self.o = iron.param(E, 64 * H)
            self.ob = iron.param(E)
            P = c.cross_heads * c.cross_pad
            self.cq, self.cqb = iron.param(P, E), iron.param(P)
            self.ck, self.ckb = iron.param(P, E), iron.param(P)
            self.cv, self.cvb = iron.param(P, E), iron.param(P)
            self.co, self.cob = iron.param(E, P), iron.param(E)
            self.ia, self.iab = iron.param(Cp, E), iron.param(Cp)
            self.ig, self.igb = iron.param(Cp, E), iron.param(Cp)
            self.da, self.dab = iron.param(9, Ch), iron.param(Ch)
            self.dg, self.dgb = iron.param(9, Ch), iron.param(Ch)
            self.pt = iron.param(E, Cp)
            if c.qk_norm:
                self.qn, self.kn = iron.param(A), iron.param(A)
                self.cqn, self.ckn = iron.param(P), iron.param(P)
            return
        self.sst = iron.param(bf(f32(w[p + "scale_shift_table"]).reshape(-1)))
        pad = lambda x: np.concatenate([f32(x), np.zeros((A - E,) + x.shape[1:], np.float32)])  # noqa: E731
        self.q = dproj(pad(w[p + "attn1.to_q.weight"]))
        # the keys are a GEMM's A without the q and k norms (kt = k h^T)
        self.k = (dproj if c.qk_norm else lambda x: iron.param(bf(x)))(pad(w[p + "attn1.to_k.weight"]))
        self.v = iron.param(bf(value_rows(f32(w[p + "attn1.to_v.weight"]), H)))
        self.o = dproj(out_cols(f32(w[p + "attn1.to_out.0.weight"]), H))
        self.ob = iron.param(bf(w[p + "attn1.to_out.0.bias"]))
        Hc, Dc, Dp = c.cross_heads, c.cross_head_dim, c.cross_pad
        fold = mha_fold(1 / math.sqrt(Dc))
        for n in ("q", "k", "v"):
            m, b = cross_rows(f32(w[p + f"attn2.to_{n}.weight"]), f32(w[p + f"attn2.to_{n}.bias"]), Hc, Dc, Dp)
            if n == "q" and not c.qk_norm:
                m, b = m * fold, b * fold
            setattr(self, f"c{n}", dproj(m))
            setattr(self, f"c{n}b", iron.param(bf(b)))
        wo = f32(w[p + "attn2.to_out.0.weight"])
        wop = np.zeros((E, Hc * Dp), np.float32)
        for h in range(Hc):
            wop[:, h * Dp : h * Dp + Dc] = wo[:, h * Dc : (h + 1) * Dc]
        self.co = dproj(wop)
        self.cob = iron.param(bf(w[p + "attn2.to_out.0.bias"]))
        wi = f32(w[p + "ff.conv_inverted.weight"])[:, :, 0, 0]
        bi = f32(w[p + "ff.conv_inverted.bias"])
        # the MLP's channels padded to Cp with zero weights and biases (DWConv3's bands): the
        # padded channels stay 0 through SiLU, the convolution and the gate, and pt ignores them
        self.ia, self.iab = dproj(pad_rows(wi[:Ch], Cp)), iron.param(bf(pad_rows(bi[:Ch], Cp)))
        self.ig, self.igb = dproj(pad_rows(wi[Ch:], Cp)), iron.param(bf(pad_rows(bi[Ch:], Cp)))
        wd = f32(w[p + "ff.conv_depth.weight"])[:, 0].reshape(2 * Ch, 9).T  # (9, 2 Ch), tap dy * 3 + dx
        bd = f32(w[p + "ff.conv_depth.bias"])
        if DW["form"] == "op":  # (10, Cp): the 9 taps, then the bias
            self.ta = iron.param(bf(pad_rows(np.vstack([wd[:, :Ch], bd[None, :Ch]]).T, Cp).T))
            self.tg = iron.param(bf(pad_rows(np.vstack([wd[:, Ch:], bd[None, Ch:]]).T, Cp).T))
        else:
            self.da, self.dab = iron.param(bf(wd[:, :Ch])), iron.param(bf(bd[:Ch]))
            self.dg, self.dgb = iron.param(bf(wd[:, Ch:])), iron.param(bf(bd[Ch:]))
        pt = f32(w[p + "ff.conv_point.weight"])[:, :, 0, 0]
        self.pt = dproj(np.concatenate([pt, np.zeros((E, Cp - Ch), np.float32)], axis=1))
        if c.qk_norm:
            # across the heads, the padded ones' zeros counted in the mean: weight times
            # sqrt(E / A), eps times E / A (self.aeps)
            self.qn = iron.param(bf(pad(w[p + "attn1.norm_q.weight"]) * math.sqrt(E / A)))
            self.kn = iron.param(bf(pad(w[p + "attn1.norm_k.weight"]) * math.sqrt(E / A)))
            qn, self.ceps = cross_norm(f32(w[p + "attn2.norm_q.weight"]), Hc, Dc, Dp)
            kn, _ = cross_norm(f32(w[p + "attn2.norm_k.weight"]), Hc, Dc, Dp)
            self.cqn, self.ckn = iron.param(bf(qn * fold)), iron.param(bf(kn))

    def depthwise(self, x, grid, taps, bias=None):
        """``x`` (tokens, Ch) through a 3 x 3 depthwise convolution (zero padded): with DWConv3
        (``taps`` (10, Ch), the bias its last row) in token order (H W, Ch); else over the grid's
        band of rows (W + 1 to a token row; the pad column's rows are not tokens)."""
        c = self.c
        H, W, Wp = c.gh, c.gw, c.gw + 1
        Ch = x.shape[1]
        band = dw_band(c)
        Copy.step(x, grid, src=tap(tuple(x.shape), 0, [H * W * Ch], [1]),
                  dst=tap((dw_rows(c), Ch), band * Ch, [H, W * Ch], [Wp * Ch, 1]))
        if DW["form"] == "op":
            import iron_dwconv

            return iron_dwconv.dwconv3(grid, taps, H, W, dw_band_c(c))
        n = H * Wp
        y = None
        for dy in range(3):
            for dx in range(3):
                s = band + (dy - 1) * Wp + (dx - 1)
                t = grid[s : s + n] * taps[dy * 3 + dx]
                y = t if y is None else y + t
        return y + bias

    def body(self, x, t6, ctx, nk):
        c, sh = self.c, self.sh
        E, n = c.hidden, x.shape[0]
        m = self.sst + t6
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (m[i * E : (i + 1) * E] for i in range(6))
        # linear self-attention
        h = F.layer_norm(x, (E,), weight=scale_msa + sh.one, eps=1e-6) + shift_msa
        q = F.linear(h, self.q, precision=PREC.dit)
        A = c.att_dim
        if c.qk_norm:
            q = F.rms_norm(q, (A,), self.qn, self.aeps)
            k = ReLU.step(F.rms_norm(F.linear(h, self.k, precision=PREC.dit), (A,), self.kn, self.aeps))
            kt = F.linear(sh.eye, k, precision=Precision.HIGHEST)  # the transpose, exactly
        else:
            kt = ReLU.step(F.linear(self.k, h, precision=PREC.dit))  # (A, n)
        if sh.kmask is not None:
            kt = kt * sh.kmask
        vt = F.linear(self.v, h, precision=PREC.dit)  # (64 H, n)
        o = linear_attention(ReLU.step(q), kt, vt, sh.lin)
        x = x + F.linear(o, self.o, self.ob, precision=PREC.dit) * gate_msa
        # cross-attention (no norm before it)
        Hc, Dp = c.cross_heads, c.cross_pad
        P = Hc * Dp
        L = ctx.shape[0]
        q = F.linear(x, self.cq, self.cqb, precision=PREC.dit)
        k = F.linear(ctx, self.ck, self.ckb, precision=PREC.dit)
        v = F.linear(ctx, self.cv, self.cvb, precision=PREC.dit)
        if c.qk_norm:
            q = F.rms_norm(q, (P,), self.cqn, self.ceps)
            k = F.rms_norm(k, (P,), self.ckn, self.ceps)
        # Every key less the first's (the prompt's <bos> row, always kept): the scores less a
        # constant of each query's, which softmax leaves out. Sana's scores (without q and k
        # norms) are about 1750 apart from their spread of a few units, where MHA's bf16 scores
        # are 8 apart; less the first key's they are about the spread.
        k = k + k[0] * -1.0
        q, k, v = q.reshape(n, Hc, Dp), k.reshape(L, Hc, Dp), v.reshape(L, Hc, Dp)
        o = sh.cross
        for s in range(0, n, L):
            Lq = min(L, n - s)  # MHA takes no more queries than keys
            o[s : s + Lq] = MHA.step(q[s : s + Lq], k[:nk], v[:nk], heads_interleaved=True, kv_interleaved=True,
                                     causal=False, d=Dp, scale=MHA_SCALE, num_pipelines=pipelines(Lq)).reshape(Lq, P)
        x = x + F.linear(o, self.co, self.cob, precision=PREC.dit)
        # GLUMBConv
        h = F.layer_norm(x, (E,), weight=scale_mlp + sh.one, eps=1e-6) + shift_mlp
        a = F.silu(F.linear(h, self.ia, self.iab, precision=PREC.dit))
        g = F.silu(F.linear(h, self.ig, self.igb, precision=PREC.dit))
        if DW["form"] == "op":
            f = self.depthwise(a, sh.grid_a, self.ta) * F.silu(self.depthwise(g, sh.grid_g, self.tg))
            if sh.kmask is not None:  # into whole 256 rows, zeros past the tokens
                sh.fbuf[0 : c.tokens] = f
                f = sh.fbuf[0:n]
            return x + F.linear(f, self.pt, precision=PREC.dit) * gate_mlp
        a = self.depthwise(a, sh.grid_a, self.da, self.dab)
        g = self.depthwise(g, sh.grid_g, self.dg, self.dgb)
        f = a * F.silu(g)  # the band's rows; the pad column's are dropped next
        H, W, Wp, Ch = c.gh, c.gw, c.gw + 1, c.mlp
        src = tap((H * Wp, Ch), 0, [H, W * Ch], [Wp * Ch, 1])
        if sh.kmask is None:
            f = Copy.step(f, src=src).reshape(n, Ch)
        else:  # into whole 256 rows, zeros past the tokens
            Copy.step(f, sh.fbuf, src=src, dst=tap((n, Ch), 0, [H * W * Ch], [1]))
            f = sh.fbuf[0:n]
        return x + F.linear(f, self.pt, precision=PREC.dit) * gate_mlp


class DitBlocks(iron.Graph):
    """``group`` Sana blocks over the tokens ``x``.

    Inputs: ``x`` (tokens, hidden); ``t6`` (6 hidden), the timestep's modulation rows
    (dit_in); ``ctx`` (text_rows, hidden), the caption. Per call: ``nk``, the caption's rows
    that count (the rest are masked).
    """

    def __init__(self, c: DitConfig, first: int = 0, w: Safetensors | None = None):
        self.c = c
        self.sh = Shared(c)
        self.blocks = [SanaBlock(c, w, first + j, self.sh) for j in range(c.group)]

    def body(self, x, t6, ctx, *, nk: Scratchpad[np.int32]):
        for b in self.blocks:
            x = b(x, t6, ctx, nk)
        return x


class TimestepEmbedding(iron.Graph):
    """diffusers' TimestepEmbedding: linear, SiLU, linear (of a 256-wide sinusoid)."""

    def __init__(self, w, p: str, E: int):
        self.l1 = iron.param(bf(w[p + "linear_1.weight"])) if w is not None else iron.param(E, 256)
        self.b1 = iron.param(bf(w[p + "linear_1.bias"])) if w is not None else iron.param(E)
        self.l2 = iron.param(bf(w[p + "linear_2.weight"])) if w is not None else iron.param(E, E)
        self.b2 = iron.param(bf(w[p + "linear_2.bias"])) if w is not None else iron.param(E)

    def body(self, t):
        return F.linear(F.silu(F.linear(t, self.l1, self.b1, precision=PREC.dit)), self.l2, self.b2, precision=PREC.dit)


class DitIn(iron.Graph):
    """The patch embedding and the timestep embedding.

    Inputs: ``patches`` (tokens, 32), the latents channel last; ``tproj`` (256), the
    timestep's sinusoid (diffusers' Timesteps, cos first). Outputs ``x`` (tokens, hidden), ``t6`` (6 hidden),
    ``emb`` (hidden), the embedded timestep the final layer takes.
    """

    def __init__(self, c: DitConfig, w: Safetensors | None = None):
        E = c.hidden
        self.c = c
        if w is None:
            self.pe, self.peb = iron.param(E, c.latent), iron.param(E)
            self.lin, self.linb = iron.param(6 * E, E), iron.param(6 * E)
        else:
            self.pe = iron.param(bf(f32(w["patch_embed.proj.weight"])[:, :, 0, 0]))
            self.peb = iron.param(bf(w["patch_embed.proj.bias"]))
            self.lin = iron.param(bf(w["time_embed.linear.weight"]))
            self.linb = iron.param(bf(w["time_embed.linear.bias"]))
        self.t = TimestepEmbedding(w, "time_embed.emb.timestep_embedder.", E)

    def body(self, patches, tproj):
        x = F.linear(patches, self.pe, self.peb, precision=PREC.dit)
        emb = self.t(tproj)
        t6 = F.linear(F.silu(emb), self.lin, self.linb, precision=PREC.dit)
        return x, t6, emb


class DitOut(iron.Graph):
    """The final layer: layer norm, the table's shift and scale plus ``emb``, the projection.

    Output ``v`` (tokens, 32), channel last.
    """

    def __init__(self, c: DitConfig, w: Safetensors | None = None):
        E = c.hidden
        self.c = c
        if w is None:
            self.shift, self.scale = iron.param(E), iron.param(E)
            self.lin, self.linb = iron.param(c.latent, E), iron.param(c.latent)
        else:
            sst = f32(w["scale_shift_table"])
            self.shift, self.scale = iron.param(bf(sst[0])), iron.param(bf(sst[1]))
            self.lin = iron.param(bf(w["proj_out.weight"]))
            self.linb = iron.param(bf(w["proj_out.bias"]))
        self.one = iron.param(bf(np.ones(E)))

    def body(self, x, emb):
        E = self.c.hidden
        h = F.layer_norm(x, (E,), weight=self.scale + emb + self.one, eps=1e-6) + (self.shift + emb)
        return F.linear(h, self.lin, self.linb, precision=PREC.dit)


class Caption(iron.Graph):
    """The caption projection (linear, tanh GELU, linear) and its RMS norm, of the prompt's
    rows: input ``y`` (text_rows, 2304), output ``ctx`` (text_rows, hidden)."""

    def __init__(self, c: DitConfig, w: Safetensors | None = None):
        E = c.hidden
        get = (lambda n, *s: iron.param(bf(w[n]))) if w is not None else (lambda n, *s: iron.param(*s))
        self.l1, self.b1 = get("caption_projection.linear_1.weight", E, c.caption), get("caption_projection.linear_1.bias", E)
        self.l2, self.b2 = get("caption_projection.linear_2.weight", E, E), get("caption_projection.linear_2.bias", E)
        self.norm = get("caption_norm.weight", E)
        self.c = c

    def body(self, y):
        h = F.gelu(F.linear(y, self.l1, self.b1, precision=PREC.text), approximate="tanh")
        h = F.linear(h, self.l2, self.b2, precision=PREC.text)
        return F.rms_norm(h, (self.c.hidden,), self.norm, 1e-5)


# --- Gemma 2 -------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TextConfig:
    """Gemma 2 2B (text_encoder/config.json)."""

    rows: int = 512  # the rows a call computes: the instruction and 298 of the prompt's (a chunk with a prefix)
    prefix: int = 0  # the instruction's first rows held as keys and values (Prefix), whole chunks; 0 for none
    cache: int = 0  # the positions the keys and values hold (with a prefix): the instruction and 298
    hidden: int = 2304
    layers: int = 26
    heads: int = 8
    kv_heads: int = 4
    head_dim: int = 256
    mlp: int = 9216
    theta: float = 10000.0
    eps: float = 1e-6
    vocab: int = 256000
    split: int = 4  # layers a command runs


def half_rope_table(rows: int, dim: int, theta: float) -> np.ndarray:
    inv = 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    a = np.arange(rows, dtype=np.float64)[:, None] * inv[None]
    out = np.empty((rows, dim), np.float32)
    out[:, ::2] = np.cos(a)
    out[:, 1::2] = np.sin(a)
    return out


class GemmaLayer(iron.Graph):
    """A Gemma 2 layer over a prompt's rows, causal: norms of (1 + w) before and after the
    attention and the MLP, GQA with RoPE, tanh-GELU gated MLP. The attention's logit
    softcapping (50) is left out: Sana's prompt rows come out the same (cosine 0.99999)."""

    def __init__(self, c: TextConfig, w, i: int):
        E, H, G, D, M = c.hidden, c.heads, c.kv_heads, c.head_dim, c.mlp
        self.c = c
        p = f"layers.{i}."
        if w is None:
            get, norm = (lambda n, *s: iron.param(*s)), (lambda n: iron.param(E))
        else:
            get = lambda n, *s: text_param(w[p + n])  # noqa: E731
            norm = lambda n: iron.param(bf(f32(w[p + n]) + 1))  # noqa: E731
        self.in_norm = norm("input_layernorm.weight")
        self.post_attn = norm("post_attention_layernorm.weight")
        self.pre_ff = norm("pre_feedforward_layernorm.weight")
        self.post_ff = norm("post_feedforward_layernorm.weight")
        if w is None:
            self.q = get("self_attn.q_proj.weight", H * D, E)
        else:
            self.q = text_param(f32(w[p + "self_attn.q_proj.weight"]) * mha_fold(D**-0.5))  # query_pre_attn_scalar 256
        self.k = get("self_attn.k_proj.weight", G * D, E)
        self.v = get("self_attn.v_proj.weight", G * D, E)
        self.o = get("self_attn.o_proj.weight", E, H * D)
        self.gate = get("mlp.gate_proj.weight", M, E)
        self.up = get("mlp.up_proj.weight", M, E)
        self.down = get("mlp.down_proj.weight", E, M)

    def body(self, x, angles, pad, cache=None, prefix=None, start=None):
        """The layer over x's rows. With a cache (keys, values ``(cache rows, kv_heads,
        head_dim)``), x's rows are those from ``start``: the prefix's keys and values (the
        instruction's, constant) are written first, x's after them from ``start``, and the
        queries attend to the ``start + rows`` before them, causally."""
        c = self.c
        L, H, G, D = x.shape[0], c.heads, c.kv_heads, c.head_dim
        pads = {pad.shape[1]: pad}
        h = F.rms_norm(x, (c.hidden,), self.in_norm, c.eps)
        q = text_linear(h, self.q, pads).reshape(L * H, D)
        k = text_linear(h, self.k, pads).reshape(L * G, D)
        v = text_linear(h, self.v, pads)
        q, k = F.rope(q, angles), F.rope(k, angles)
        if cache is None:
            k_all, v_all = k.reshape(L, G, D), v.reshape(L, G, D)
        else:
            kc, vc = cache
            for st, pre, new in ((kc, prefix[0], k), (vc, prefix[1], v)):
                if pre is not None:
                    st[0 : pre.shape[0]] = pre
                st[start : start + L] = new.reshape(L, G, D)
            k_all, v_all = kc[: start + L], vc[: start + L]
        # a pipeline takes whole rounds of 64 queries
        o = MHA.step(q.reshape(L, H, D), k_all, v_all, heads_interleaved=True, kv_interleaved=True, causal=True, d=D,
                     scale=MHA_SCALE, num_pipelines=max(1, min(8, L // 64)))
        o = text_linear(o.reshape(L, H * D), self.o)
        x = x + F.rms_norm(o, (c.hidden,), self.post_attn, c.eps)
        h = F.rms_norm(x, (c.hidden,), self.pre_ff, c.eps)
        g = F.gelu(text_linear(h, self.gate, pads), approximate="tanh") * text_linear(h, self.up, pads)
        g = text_linear(g, self.down)
        return x + F.rms_norm(g, (c.hidden,), self.post_ff, c.eps)


class Text(iron.Graph):
    """Gemma 2 2B: input ``x`` (rows, hidden), the prompt's embedding rows times sqrt(hidden)
    (zeros past them); output its last hidden state after the final norm, the same shape.

    With ``prefix`` (Prefix: the instruction's keys and values, from the CPU), the graph is a
    chunk of ``rows`` positions from ``start`` (a multiple of rows): x holds the embedding rows
    from ``start``, the keys and values of the positions before it are the prefix's (its first
    ``c.prefix``) or the calls' before (the cache holds them while the graph is loaded), and the
    rows past the prompt's are any (causal attention keeps them from the prompt's). Every
    prompt begins with the instruction (Sana's) or <bos> (its negative prompt), whose rows
    causal attention leaves the same whatever follows."""

    def __init__(self, c: TextConfig, w=None, prefix=None):
        self.c = c
        self.layers = [GemmaLayer(c, w, i) for i in range(c.layers)]
        self.norm = iron.param(bf(f32(w["norm.weight"]) + 1)) if w is not None else iron.param(c.hidden)
        # the hidden rows in zero-padded columns, for q8's 512-column blocks (text_linear)
        self.pad = iron.state((c.rows, -(-c.hidden // 512) * 512))
        self.has_prefix = prefix is not None  # (its arrays are kept as params only)
        if prefix is None:
            self.angles = iron.param(bf(half_rope_table(c.rows, c.head_dim, c.theta)))
            return
        G, D, P = c.kv_heads, c.head_dim, c.prefix
        # a weight, not a param: a handle while recording, which per-call values slice
        self.angles = iron.weight(bf(half_rope_table(c.cache, c.head_dim, c.theta)))
        self.caches = [(iron.state((c.cache, G, D)), iron.state((c.cache, G, D))) for _ in range(c.layers)]
        self.pre = [(iron.param(bf(k[:P])), iron.param(bf(v[:P]))) for k, v in zip(prefix.keys, prefix.values)]

    def body(self, x, *, start: Scratchpad[np.int32] = np.int32(0)):
        c = self.c
        if not self.has_prefix:
            for i, layer in enumerate(self.layers):
                if i and i % c.split == 0:
                    command_boundary()
                x = layer(x, self.angles, self.pad)
            return F.rms_norm(x, (c.hidden,), self.norm, c.eps)
        angles = Copy.step(self.angles[start : start + c.rows])
        for i, layer in enumerate(self.layers):
            if i and i % c.split == 0:
                command_boundary()
            x = layer(x, angles, self.pad, self.caches[i], self.pre[i], start)
        return F.rms_norm(x, (c.hidden,), self.norm, c.eps)


# The instruction diffusers' Sana pipelines put before the prompt (complex_human_instruction,
# joined by new lines), as internal/imagegen/sana.Instruction.
INSTRUCTION = "\n".join([
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
])


@dataclasses.dataclass
class Prefix:
    """The instruction's rows through Gemma 2 in float32 on the CPU, as transformers runs it
    (the attention's softcap too): each layer's keys (after RoPE) and values ``(P, kv_heads,
    head_dim)`` and the last hidden states ``out`` ``(P, hidden)``, of the tokens ``ids``."""

    ids: list
    keys: list
    values: list
    out: np.ndarray

    @staticmethod
    def of(c: TextConfig, w, ids: list) -> "Prefix":
        P, E, H, G, D = len(ids), c.hidden, c.heads, c.kv_heads, c.head_dim

        def norm(x, name):
            x = x / np.sqrt((x * x).mean(-1, keepdims=True) + c.eps)
            return x * (1 + f32(w[name]))

        inv = 1.0 / c.theta ** (np.arange(0, D, 2, dtype=np.float64) / D)
        a = np.arange(P, dtype=np.float64)[:, None] * inv[None]
        cos, sin = np.cos(np.concatenate([a, a], 1)), np.sin(np.concatenate([a, a], 1))

        def rope(t):  # (P, heads, D), transformers' rotate_half
            half = np.concatenate([-t[..., D // 2 :], t[..., : D // 2]], -1)
            return (t * cos[:, None] + half * sin[:, None]).astype(np.float32)

        x = f32(w["embed_tokens.weight"][ids]) * np.float32(math.sqrt(E))
        mask = np.triu(np.full((P, P), -np.inf, np.float32), 1)
        keys, values = [], []
        for i in range(c.layers):
            p = f"layers.{i}."
            lin = lambda t, n: t @ f32(w[p + n]).T  # noqa: E731
            h = norm(x, p + "input_layernorm.weight")
            q = rope(lin(h, "self_attn.q_proj.weight").reshape(P, H, D))
            k = rope(lin(h, "self_attn.k_proj.weight").reshape(P, G, D))
            v = lin(h, "self_attn.v_proj.weight").reshape(P, G, D)
            keys.append(k)
            values.append(v)
            kr, vr = np.repeat(k, H // G, 1), np.repeat(v, H // G, 1)
            s = np.einsum("phd,qhd->hpq", q, kr) * np.float32(256**-0.5)  # query_pre_attn_scalar
            s = np.tanh(s / 50) * 50 + mask  # attn_logit_softcapping
            s = np.exp(s - s.max(-1, keepdims=True))
            o = np.einsum("hpq,qhd->phd", s / s.sum(-1, keepdims=True), vr).reshape(P, H * D)
            x = x + norm(lin(o, "self_attn.o_proj.weight"), p + "post_attention_layernorm.weight")
            h = norm(x, p + "pre_feedforward_layernorm.weight")
            g = lin(h, "mlp.gate_proj.weight")
            g = 0.5 * g * (1 + np.tanh(math.sqrt(2 / math.pi) * (g + 0.044715 * g**3)))
            x = x + norm(lin(g * lin(h, "mlp.up_proj.weight"), "mlp.down_proj.weight"), p + "post_feedforward_layernorm.weight")
        return Prefix(list(ids), keys, values, norm(x, "norm.weight").astype(np.float32))


# --- the VAE decoder (DC-AE) -------------------------------------------------------------------
#
# DC-AE f32c32's decoder: conv_in from the 32 latent channels (plus the channels repeated, a
# shortcut), then six resolutions, each (but the first) entered by an upsampling block (nearest
# 2x and a 3 x 3 convolution, plus the input's channels shuffled into the pixels, a shortcut),
# three EfficientViT blocks at the three lowest (multi-scale linear attention and GLUMBConv) and
# three ResBlocks at the three highest, then an RMS norm, ReLU and conv_out.
#
# The 3 x 3 convolutions are iron_anima's: a channel-last grid zero-padded (GridShape, rows of
# W + 16), one GEMM over the im2col rows a chunk of rows at a time, the result's interior
# copied into the next grid. The shortcuts fold into the convolutions' centre taps: conv_in's
# repeats each latent channel 32 times; an upsampling block that keeps its channels adds the
# nearest-upsampled input (the identity); one that halves them takes, at an output pixel of row
# parity i, input channel 2c + i into channel c, so its convolution runs on the even and the
# odd rows with a weight each. The EfficientViT blocks run on the dense tokens (n, C) and their
# depthwise convolutions on a grid of their own (DwGrid).



@dataclasses.dataclass(frozen=True)
class GridShape:
    """iron_anima's padded channel-last grid, for any size: rows of Wp pixels (Wp >= W + 1, a
    multiple of 16: the same as iron_anima's W + 16 for a W of whole 16s), the image's rows
    and zero rows to Hp, a multiple of 16, a zero row above and below and a zero pixel before
    the first, rounded up to 256 rows. A convolution runs over whole chunks of Hp rows and
    copies the image's rows."""

    H: int
    W: int
    C: int

    @property
    def Wp(self) -> int:
        return -(-(self.W + 1) // 16) * 16

    @property
    def Hp(self) -> int:
        return -(-self.H // 16) * 16

    @property
    def rows(self) -> int:
        return -(-((self.Hp + 2) * self.Wp + 2) // 256) * 256

    @property
    def band(self) -> int:
        return 1 + self.Wp

    def pixel(self, y: int, x: int) -> int:
        return self.band + y * self.Wp + 1 + x

    def chunk(self, cin: int) -> int:
        """Image rows a convolution takes at once: a multiple of 16 dividing Hp, its im2col rows
        (9 cin wide) at most VAE_COLS_MIB."""
        best = 16
        for r in range(16, self.Hp + 1, 16):
            if self.Hp % r == 0 and r * self.Wp * 9 * cin * 2 <= VAE_COLS_MIB[0] << 20:
                best = r
        return best


VAE_C = [128, 256, 512, 512, 1024, 1024]  # decoder_block_out_channels
VAE_COLS_MIB = [64]  # a convolution's im2col rows at most (--vae-cols-mib)
# The 3 x 3 convolutions: "grid" (iron_gridgemm.GridGEMM reads the grid's rows in place, one GEMM
# a convolution or a kernel row; the im2col copy took 63 % of a 1024 x 1024 ResBlock: three
# ResBlocks 3.3 -> 1.4 s, design.md 16.139), "taps" (a GEMM a tap, summed) or "im2col" (--vae-conv)
VAE_CONV = ["grid"]
VAE_TAP_ROWS = 64  # image rows a tap's GEMM takes at once
# The upsamplings: "phase" (four 2 x 2 convolutions at the low resolution, one an output phase)
# or "grid" (the nearest 2x into a grid, then the 3 x 3 convolution) (--vae-up)
VAE_UP = ["phase"]
# IRON_SANA_PROFILE=1: a command an operation in the convolutions (HWNPU_IRON_TIMING times each)
VAE_PROFILE = os.environ.get("IRON_SANA_PROFILE") == "1"


def _profile_boundary() -> None:
    if VAE_PROFILE:
        command_boundary()
VAE_VIT = 3  # the resolutions (from the lowest) of EfficientViT blocks


@dataclasses.dataclass(frozen=True)
class DwGrid:
    """A zero-padded grid for a k x k depthwise convolution: H rows of W pixels, each followed
    by p = k // 2 zero pixels, p zero rows above and below, and p zero pixels before the first
    (so every tap's view of the band of the H rows is one contiguous run of rows)."""

    H: int
    W: int
    k: int

    @property
    def p(self) -> int:
        return self.k // 2

    @property
    def Wp(self) -> int:
        return self.W + self.p

    @property
    def rows(self) -> int:
        return (self.H + 2 * self.p) * self.Wp + 2 * self.p

    @property
    def band(self) -> int:
        """The row of pixel (0, 0)."""
        return self.p + self.p * self.Wp


def dw_band_conv(x, state, g: DwGrid, taps, bias=None):
    """``x`` (H W, C) dense through the depthwise convolution ``taps`` (k k, C), zero padded:
    the result over the grid's band (H Wp, C), whose pad columns' rows are not pixels."""
    C = x.shape[1]
    Copy.step(x, state, src=tap(tuple(x.shape), 0, [g.H * g.W * C], [1]),
              dst=tap((g.rows, C), g.band * C, [g.H, g.W * C], [g.Wp * C, 1]))
    n = g.H * g.Wp
    y = None
    for dy in range(g.k):
        for dx in range(g.k):
            s = g.band + (dy - g.p) * g.Wp + (dx - g.p)
            t = state[s : s + n] * taps[dy * g.k + dx]
            y = t if y is None else y + t
    return y if bias is None else y + bias


def band_dense(y, g: DwGrid | None, out=None):
    """The pixels' rows of a band (H Wp, C): (H W, C), or into ``out`` (a state of more rows,
    zeros past them). With ``g`` None, y is already (H W, C) (DWConv3's)."""
    if g is None:
        if out is None:
            return y
        out[0 : y.shape[0]] = y
        return out[0 : out.shape[0]]
    C = y.shape[1]
    src = tap((g.H * g.Wp, C), 0, [g.H, g.W * C], [g.Wp * C, 1])
    if out is None:
        return Copy.step(y, src=src).reshape(g.H * g.W, C)
    Copy.step(y, out, src=src, dst=tap(tuple(out.shape), 0, [g.H * g.W * C], [1]))
    return out[0 : out.shape[0]]


def dw_taps(w: np.ndarray) -> np.ndarray:
    """A depthwise convolution's weight (C, 1, k, k) as taps (k k, C), tap dy k + dx."""
    C, _, k, _ = w.shape
    return w[:, 0].reshape(C, k * k).T


class VaeStage(iron.Graph):
    """A run of the decoder's modules.

    ``ops``: ("in",) conv_in (input: the latents' grid, 64 channels, the first 32 the latents);
    ("vit", path, C); ("up", path, cin, cout); ("res", path, C); ("head",) norm_out, ReLU,
    conv_out to 64 channels of which the first three are RGB (output: the pixels, dense).
    Each op takes and gives a grid (GridShape), but a "vit" gives the dense tokens to a next
    "vit" or "up". The stage's own output is a grid (or the pixels).
    """

    def __init__(self, ops: list, g: GridShape, w: Safetensors, scaling: float):
        self.ops, self.g_in = ops, g
        self.params = []
        self._index = {}
        self.states = []
        self.consts = {}  # the linear attentions' constants by (heads, n)
        self.dw = {}  # depthwise grids by (H, W, k, C)
        self.kmask = {}  # by (C, rows): the keys of the dense rows past the pixels, zeros
        self.padded = {}  # by (C, rows): states for the dense rows, zeros past the pixels
        self._cols = {}
        self.free_grids = {}  # grid states given back, by shape (free_grid)
        self.grid_of = {}  # the state of each grid view, by id
        H, W = g.H, g.W
        for op in ops:
            kind = op[0]
            if kind == "in":
                wt = f32(w["decoder.conv_in.weight"]) / scaling  # (1024, 32, 3, 3)
                co = wt.shape[0]
                rep = co // wt.shape[1]
                for c in range(co):
                    wt[c, c // rep, 1, 1] += 1 / scaling  # the repeated channels
                self._conv3("conv_in", wt, f32(w["decoder.conv_in.bias"]), 64)
                g = GridShape(H, W, co)
            elif kind == "vit":
                _, path, C = op
                n = H * W
                Hh = C // 32
                npad = -(-n // 256) * 256  # the dense rows: whole 256 rows for the GEMMs
                if (Hh, npad) not in self.consts:
                    self.consts[(Hh, npad)] = LinearAttnConsts(Hh, npad)
                    if npad != n:
                        m = np.zeros((C, npad), np.float32)
                        m[:, :n] = 1
                        self.kmask[(C, npad)] = iron.param(bf(m))  # the padding rows' keys out
                        for ch in (C, 4 * C):
                            self.padded.setdefault((ch, npad), [iron.state((npad, ch)) for _ in range(2)])
                a = f"{path}.attn."
                qkv = np.concatenate([f32(w[a + "to_q.weight"]), f32(w[a + "to_k.weight"]), f32(w[a + "to_v.weight"])])
                perm = [np.concatenate([np.arange(96 * h + 32 * j, 96 * h + 32 * j + 32) for h in range(Hh)]) for j in range(3)]
                for j, nm in enumerate("qkv"):
                    self._param(f"{path}.{nm}", qkv[perm[j]])
                self._param(f"{path}.kt", qkv[perm[1]])  # the same, the transpose's weight
                self._param(f"{path}.vt", value_rows(qkv[perm[2]], Hh))
                ms = f"{a}to_qkv_multiscale.0."
                taps = dw_taps(f32(w[ms + "proj_in.weight"]))  # (25, 3C)
                po = f32(w[ms + "proj_out.weight"])[:, :, 0, 0]  # (3C, 32), grouped by 32
                for j, nm in enumerate("qkv"):
                    t = taps[:, perm[j]]
                    if self.dw_op(C, W, 5):  # DWConv3's taps: the 25, then a zero bias
                        t = np.vstack([t, np.zeros((1, C), np.float32)])
                    self._param(f"{path}.ms_{nm}_taps", t)
                    bd = np.zeros((C, C), np.float32)
                    for h in range(Hh):
                        bd[32 * h : 32 * h + 32, 32 * h : 32 * h + 32] = po[96 * h + 32 * j : 96 * h + 32 * j + 32]
                    self._param(f"{path}.ms_{nm}", value_rows(bd, Hh) if nm == "v" else bd)
                wo = f32(w[a + "to_out.weight"])  # (C, 2C)
                self._param(f"{path}.o1", out_cols(wo[:, :C], Hh))
                self._param(f"{path}.o2", out_cols(wo[:, C:], Hh))
                self._param(f"{path}.on", f32(w[a + "norm_out.weight"]))
                self._param(f"{path}.onb", f32(w[a + "norm_out.bias"]))
                f = f"{path}.conv_out."
                wi = f32(w[f + "conv_inverted.weight"])[:, :, 0, 0]
                bi = f32(w[f + "conv_inverted.bias"])
                Ch = wi.shape[0] // 2
                self._param(f"{path}.ia", wi[:Ch])
                self._param(f"{path}.iab", bi[:Ch])
                self._param(f"{path}.ig", wi[Ch:])
                self._param(f"{path}.igb", bi[Ch:])
                td = dw_taps(f32(w[f + "conv_depth.weight"]))
                bd = f32(w[f + "conv_depth.bias"])
                if self.dw_op(Ch, W, 3):  # DWConv3's taps: the 9, then the bias
                    self._param(f"{path}.da", np.vstack([td[:, :Ch], bd[None, :Ch]]))
                    self._param(f"{path}.dg", np.vstack([td[:, Ch:], bd[None, Ch:]]))
                else:
                    self._param(f"{path}.da", td[:, :Ch])
                    self._param(f"{path}.dab", bd[:Ch])
                    self._param(f"{path}.dg", td[:, Ch:])
                    self._param(f"{path}.dgb", bd[Ch:])
                self._param(f"{path}.pt", f32(w[f + "conv_point.weight"])[:, :, 0, 0])
                self._param(f"{path}.fn", f32(w[f + "norm.weight"]))
                self._param(f"{path}.fnb", f32(w[f + "norm.bias"]))
                for k, ch in ((5, C), (3, Ch)):
                    key = (H, W, k, ch)
                    if key not in self.dw:
                        self.dw[key] = iron.state((DwGrid(H, W, k).rows, ch))
            elif kind == "up":
                _, path, cin, cout = op
                wt = f32(w[f"{path}.conv.weight"])
                b = f32(w[f"{path}.conv.bias"])
                if cin == cout:
                    for c in range(cout):
                        wt[c, c, 1, 1] += 1
                    by_row = [wt, wt]
                else:
                    assert cin == 2 * cout, op
                    by_row = []
                    for i in range(2):
                        wi = wt.copy()
                        for c in range(cout):
                            wi[c, 2 * c + i, 1, 1] += 1
                        by_row.append(wi)
                if VAE_UP[0] == "phase":
                    self._up_phases(f"{path}.up", by_row, b)
                elif cin == cout:
                    self._conv3(f"{path}.up", wt, b, cin, VAE_CONV[0] in ("taps", "grid"))
                else:
                    for i in range(2):
                        self._conv3(f"{path}.up{i}", by_row[i], b, cin)
                H, W = 2 * H, 2 * W
            elif kind == "res":
                _, path, C = op
                taps = VAE_CONV[0] in ("taps", "grid")
                self._conv3(f"{path}.conv1", f32(w[f"{path}.conv1.weight"]), f32(w[f"{path}.conv1.bias"]), C, taps)
                self._conv3(f"{path}.conv2", f32(w[f"{path}.conv2.weight"]), None, C, taps)
                self._param(f"{path}.n", f32(w[f"{path}.norm.weight"]))
                self._param(f"{path}.nb", f32(w[f"{path}.norm.bias"]))
            elif kind == "head":
                self._param("head.n", f32(w["decoder.norm_out.weight"]))
                self._param("head.nb", f32(w["decoder.norm_out.bias"]))
                wt = f32(w["decoder.conv_out.weight"])
                full = np.zeros((64,) + wt.shape[1:], np.float32)
                full[:3] = wt
                b = np.zeros(64, np.float32)
                b[:3] = f32(w["decoder.conv_out.bias"])
                self._conv3("head", full, b, wt.shape[1], VAE_CONV[0] in ("taps", "grid"))
            else:
                raise ValueError(op)

    def _param(self, name: str, x) -> None:
        self._index[name] = len(self.params)
        self.params.append(iron.param(bf(x)))

    @staticmethod
    def dw_op(ch: int, W: int, k: int) -> bool:
        """Whether the VAE's k x k depthwise convolution over ch channels runs on DWConv3."""
        if VAE_DW["form"] != "op":
            return False
        import iron_dwconv

        return iron_dwconv.band(ch, W, k) is not None

    def w(self, name: str):
        return self.params[self._index[name]] if name in self._index else None

    def _conv3(self, name: str, wt: np.ndarray, b, cin_padded: int, taps: bool = False) -> None:
        """A 3 x 3 convolution (cout, cin, 3, 3) as an im2col matrix (cout, dy 3 cin + dx cin + ci),
        or with ``taps`` as a matrix (cout, cin) a tap (``name.tDYDX``, conv3's tap GEMMs)."""
        co, ci = wt.shape[:2]
        full = np.zeros((co, cin_padded, 3, 3), np.float32)
        full[:, :ci] = wt
        im2col = full.transpose(0, 2, 3, 1).reshape(co, -1)
        if taps and VAE_CONV[0] == "grid":
            if 3 * cin_padded <= 512:  # one GEMM: k tiles of a kernel row (GridGEMM's k_tile <= 512)
                self._param(f"{name}.g", im2col)
            else:
                for dy in range(3):
                    self._param(f"{name}.g{dy}", im2col[:, dy * 3 * cin_padded : (dy + 1) * 3 * cin_padded])
        elif taps:
            for dy in range(3):
                for dx in range(3):
                    self._param(f"{name}.t{dy}{dx}", full[:, :, dy, dx])
        else:
            self._param(f"{name}.w", im2col)
        if b is not None:
            self._param(f"{name}.b", b)

    # The rows (or columns) of a 3 x 3 kernel each low-resolution tap of an output phase takes,
    # after a nearest 2x: phase 0 (an even output row) reads the low rows y - 1 (the kernel's
    # top row) and y (its middle and bottom rows); phase 1 reads y (top and middle) and y + 1.
    PHASE_TAPS = {0: ((-1, (0,)), (0, (1, 2))), 1: ((0, (0, 1)), (1, (2,)))}

    def _up_phases(self, name: str, by_row: list, b) -> None:
        """A nearest 2x and a 3 x 3 convolution as four 2 x 2 convolutions at the low resolution,
        one an output phase (py, px): ``name.pPYPXAB`` (cout, cin) is the low tap (A, B) of
        phase (py, px), the kernel's rows and columns it takes summed. ``by_row[py]`` is the
        kernel of the output rows of parity py (the shortcut of an upsampling that halves the
        channels differs by it)."""
        cin = by_row[0].shape[1]
        for py in range(2):
            wt = by_row[py]
            for px in range(2):
                taps = [[sum(wt[:, :, dy, dx] for dy in rows for dx in cols) for _, cols in self.PHASE_TAPS[px]]
                        for _, rows in self.PHASE_TAPS[py]]
                if VAE_CONV[0] == "grid":
                    # GridGEMM: a low row's two taps are two adjacent pixels, 2 cin contiguous
                    rows2 = [np.concatenate(t, axis=1) for t in taps]  # (cout, 2 cin) a low row
                    if 2 * cin <= 512:
                        self._param(f"{name}.q{py}{px}", np.concatenate(rows2, axis=1))
                    else:
                        for a in range(2):
                            self._param(f"{name}.q{py}{px}{a}", rows2[a])
                else:
                    for a in range(2):
                        for bb in range(2):
                            self._param(f"{name}.p{py}{px}{a}{bb}", taps[a][bb])
        self._param(f"{name}.b", b)

    # --- grids ---

    def new_grid(self, g: GridShape):
        """A grid's state: one given back (free_grid) of its shape, else a new one."""
        free = self.free_grids.setdefault((g.rows, g.C), [])
        if free:
            return free.pop()
        s = iron.state((g.rows, g.C))
        self.states.append(s)
        return s

    def free_grid(self, view) -> None:
        """Give back the grid state of ``view`` (a view of new_grid's), read no more, for the
        next grid of its shape: a stage of several blocks holds three grids, not two a block."""
        s = self.grid_of.pop(id(view), None)
        if s is not None:
            self.free_grids.setdefault(tuple(s.shape), []).append(s)

    def view(self, s):
        v = s[0 : s.shape[0]]
        self.grid_of[id(v)] = s
        return v

    def interior_from(self, src, g: GridShape, dst, r0: int, R: int, step: int = 1) -> None:
        """Copy ``R`` image rows of ``src`` (band rows, R Wp of them) into the grid ``dst``'s
        rows r0, r0 + step, ... (those that are the image's: before H)."""
        C, Wp = g.C, g.Wp
        n = min(R, -(-(g.H - r0) // step))
        if n <= 0:
            return
        Copy.step(src, dst, src=tap((R * Wp, C), C, [n, g.W * C], [Wp * C, 1]),
                  dst=tap((g.rows, C), g.pixel(r0, 0) * C, [n, g.W * C], [step * Wp * C, 1]))

    def dense_to_grid(self, x, g: GridShape):
        out = self.new_grid(g)
        Copy.step(x, out, src=tap(tuple(x.shape), 0, [g.H * g.W * g.C], [1]),
                  dst=tap((g.rows, g.C), g.pixel(0, 0) * g.C, [g.H, g.W * g.C], [g.Wp * g.C, 1]))
        return self.view(out)

    def cols_state(self, n: int, k: int):
        """A state for im2col rows (n, k), one per shape, reused by every chunk."""
        key = (n, k)
        if key not in self._cols:
            self._cols[key] = iron.state((n, k))
            self.states.append(self._cols[key])
        return self._cols[key]

    def grid_to_dense(self, x, g: GridShape):
        """The pixels (H W, C); in whole 256 rows (zeros past them) when H W is not."""
        C, n = g.C, g.H * g.W
        src = tap((g.rows, C), g.pixel(0, 0) * C, [g.H, g.W * C], [g.Wp * C, 1])
        if n % 256 == 0:
            return Copy.step(x, src=src).reshape(n, C)
        out = self.new_grid_shape((-(-n // 256) * 256, C))
        Copy.step(x, out, src=src, dst=tap(out.shape, 0, [n * C], [1]))
        return self.view(out)

    def new_grid_shape(self, shape):
        s = iron.state(shape)
        self.states.append(s)
        return s

    def conv3(self, x, g: GridShape, wname: str, cout: int, post=None, parity: bool = False, into=None):
        """The 3 x 3 convolution of the grid x into a new grid of cout channels (or the grid
        state ``into``). ``post(y, r0, R)`` (when given) takes each chunk's GEMM result (band
        rows) before it is copied in; ``parity``: the even and the odd rows each with their own
        weight (wname + "0"/"1")."""
        gout = GridShape(g.H, g.W, cout)
        out = self.new_grid(gout) if into is None else into
        Wp, C = g.Wp, g.C
        grid_one, grid_rows = self.w(f"{wname}.g"), self.w(f"{wname}.g0")
        if grid_one is not None or grid_rows is not None:
            # the grid's rows read in place by the GEMM (iron_gridgemm.GridGEMM): no im2col copy,
            # and one GEMM (or one a kernel row, summed) instead of nine
            from iron.operators.flm.gemm.design import Epilogue
            from iron_gridgemm import GridGEMM

            R = max(r for r in range(16, min(VAE_TAP_ROWS, g.Hp) + 1, 16) if g.Hp % r == 0)
            kw = dict(b_col_maj=True, epilogue_modes=(Epilogue.NONE,), precision=PREC.vae)
            for r0 in range(0, g.H, R):
                n = R * Wp
                if grid_one is not None:
                    s0 = g.band + r0 * Wp - Wp - 1
                    MA = n + 2 * Wp + 2
                    y = GridGEMM.step(x[s0 : s0 + MA], grid_one, M=n, MA=MA, CA=C, k_stride=Wp * C, k_tile=3 * C, **kw)
                else:
                    y = None
                    for dy in range(3):
                        s0 = g.band + r0 * Wp + (dy - 1) * Wp - 1
                        t = GridGEMM.step(x[s0 : s0 + n + 2], self.w(f"{wname}.g{dy}"), M=n, MA=n + 2, CA=C, **kw)
                        y = t if y is None else y + t
                if self.w(f"{wname}.b") is not None:
                    y = y + self.w(f"{wname}.b")
                _profile_boundary()
                if post is not None:
                    y = post(y, r0, R, 1)
                    _profile_boundary()
                self.interior_from(y, gout, out, r0, R)
                command_boundary()  # a chunk's control code (the device heap holds 64 MiB)
            return self.view(out), gout
        if self.w(f"{wname}.t00") is not None:
            # a GEMM a tap: each tap's input rows are a run of the grid's (channel last, rows of
            # Wp pixels), so no im2col copy (9 times the grid) is made; the 9 products are summed
            R = max(r for r in range(16, min(VAE_TAP_ROWS, g.Hp) + 1, 16) if g.Hp % r == 0)
            for r0 in range(0, g.H, R):
                n = R * Wp
                y = None
                for dy in range(3):
                    for dx in range(3):
                        s0 = g.band + r0 * Wp + (dy - 1) * Wp + (dx - 1)
                        t = F.linear(x[s0 : s0 + n], self.w(f"{wname}.t{dy}{dx}"),
                                     self.w(f"{wname}.b") if y is None else None, precision=PREC.vae)
                        _profile_boundary()
                        y = t if y is None else y + t
                        _profile_boundary()
                if post is not None:
                    y = post(y, r0, R, 1)
                    _profile_boundary()
                self.interior_from(y, gout, out, r0, R)
                command_boundary()  # a chunk's control code (the device heap holds 64 MiB)
            return self.view(out), gout
        R = g.chunk(C)
        if parity:
            R = max(32, R) if g.Hp % max(32, R) == 0 else R
            assert R % 2 == 0 and (R // 2) * Wp % 256 == 0, (g, R)
        for r0 in range(0, g.H, R):
            for i in (0, 1) if parity else (None,):
                rows, step, first = (R // 2, 2, r0 + i) if parity else (R, 1, r0)
                n = rows * Wp
                start = (g.band + first * Wp - Wp - 1) * C
                if 3 * C <= 2046:
                    cols = Copy.step(x, src=tap((g.rows, C), start, [rows, Wp, 3, 3 * C],
                                                [step * Wp * C, C, Wp * C, 1])).reshape(n, 9 * C)
                else:
                    # a DMA dimension is at most 1023 words: a neighbour row (3 C) a copy
                    cols = self.cols_state(n, 9 * C)
                    for dy in range(3):
                        Copy.step(x, cols, src=tap((g.rows, C), start + dy * Wp * C, [rows, Wp, 3, C], [step * Wp * C, C, C, 1]),
                                  dst=tap((n, 9 * C), dy * 3 * C, [n, 3, C], [9 * C, C, 1]))
                    cols = cols[0:n]
                _profile_boundary()
                name = wname + ("" if i is None else str(i))
                y = F.linear(cols, self.w(f"{name}.w"), self.w(f"{name}.b"), precision=PREC.vae)
                _profile_boundary()
                if post is not None:
                    y = post(y, first, rows, step)
                    _profile_boundary()
                self.interior_from(y, gout, out, first, rows, step)
                _profile_boundary()
        return self.view(out), gout

    # --- the blocks ---

    def res(self, x, g: GridShape, path: str):
        """conv1, SiLU, conv2, RMS norm (weight and bias), plus the input."""
        a, ga = self.conv3(x, g, f"{path}.conv1", g.C, post=lambda y, *_: F.silu(y))

        def post(y, r0, rows, step):
            s = g.band + r0 * g.Wp
            h = F.rms_norm(y, (g.C,), self.w(f"{path}.n"), 1e-5) + self.w(f"{path}.nb")
            return h + x[s : s + rows * g.Wp]

        # into x's own grid when it is one of the stage's: a chunk reads x's rows (the residual)
        # before it writes the same rows, and the convolution reads a, not x
        own = self.grid_of.pop(id(x), None)
        y, gy = self.conv3(a, ga, f"{path}.conv2", g.C, post=post, into=own)
        self.free_grid(a)
        return y, gy

    def dwconv(self, x, H: int, W: int, k: int, taps, bias=None):
        """``x`` (H W, C) dense through a depthwise convolution: the band (DwGrid's) and the grid,
        or with DWConv3 (the bias in ``taps``) (H W, C) dense and None (band_dense passes it)."""
        g = DwGrid(H, W, k)
        C = x.shape[1]
        state = self.dw[(H, W, k, C)]
        if self.dw_op(C, W, k):
            import iron_dwconv

            Copy.step(x, state, src=tap(tuple(x.shape), 0, [H * W * C], [1]),
                      dst=tap((g.rows, C), g.band * C, [H, W * C], [g.Wp * C, 1]))
            return iron_dwconv.dwconv3(state, taps, H, W, iron_dwconv.band(C, W, k), k), None
        return dw_band_conv(x, state, g, taps, bias), g

    def vit(self, xd, H: int, W: int, path: str):
        """An EfficientViT block on the dense tokens xd (n, C): the pixels' H W rows, then
        zero rows to whole 256 (whose keys are masked out of the attention's sums)."""
        n, C = xd.shape
        Hh = C // 32
        k = self.consts[(Hh, n)]
        mask = self.kmask.get((C, n))
        pads = self.padded.get((C, n))
        P = PREC.vae
        p = lambda s: self.w(f"{path}.{s}")  # noqa: E731
        q = F.linear(xd, p("q"), precision=P)
        kk = F.linear(xd, p("k"), precision=P)
        v = F.linear(xd, p("v"), precision=P)
        kt = ReLU.step(F.linear(p("kt"), xd, precision=P))
        if mask is not None:
            kt = kt * mask
        vt = F.linear(p("vt"), xd, precision=P)
        _profile_boundary()  # A: the projections
        o1 = linear_attention(ReLU.step(q), kt, vt, k, PREC.vae_attn)
        _profile_boundary()  # B: the linear attention
        # the multi-scale branch: 5 x 5 depthwise, then the grouped 1 x 1 (block diagonal)
        qm = band_dense(*self.dwconv(q, H, W, 5, p("ms_q_taps")), out=pads[0] if pads else None)
        km = band_dense(*self.dwconv(kk, H, W, 5, p("ms_k_taps")), out=pads[1] if pads else None)
        _profile_boundary()  # C: two 5 x 5 depthwise
        ktm = ReLU.step(F.linear(p("ms_k"), km, precision=P))
        if mask is not None:
            ktm = ktm * mask
        vm = band_dense(*self.dwconv(v, H, W, 5, p("ms_v_taps")), out=pads[1] if pads else None)
        _profile_boundary()  # D: a 1 x 1 and a 5 x 5 depthwise
        o2 = linear_attention(ReLU.step(F.linear(qm, p("ms_q"), precision=P)), ktm, F.linear(p("ms_v"), vm, precision=P), k,
                              PREC.vae_attn)
        a = F.linear(o1, p("o1"), precision=P) + F.linear(o2, p("o2"), precision=P)
        _profile_boundary()  # E: the multi-scale attention, the outputs
        x1 = xd + F.rms_norm(a, (C,), p("on"), 1e-5) + p("onb")
        # GLUMBConv
        ha = F.silu(F.linear(x1, p("ia"), p("iab"), precision=P))
        hg = F.silu(F.linear(x1, p("ig"), p("igb"), precision=P))
        ya, g3 = self.dwconv(ha, H, W, 3, p("da"), None if self.dw_op(ha.shape[1], W, 3) else p("dab"))
        _profile_boundary()  # F: the norm, the inverted 1 x 1s
        yg, _ = self.dwconv(hg, H, W, 3, p("dg"), None if self.dw_op(hg.shape[1], W, 3) else p("dgb"))
        _profile_boundary()  # G: two 3 x 3 depthwise
        f = band_dense(ya * F.silu(yg), g3, out=self.padded[(4 * C, n)][0] if pads else None)
        f = F.linear(f, p("pt"), precision=P)
        return x1 + F.rms_norm(f, (C,), p("fn"), 1e-5) + p("fnb")

    def up(self, x, g: GridShape, dense: bool, path: str, cin: int, cout: int):
        """Nearest 2x of x (a grid, or dense), then the convolution with the shortcut folded."""
        if self.w(f"{path}.up.p0000") is not None or self.w(f"{path}.up.q00") is not None \
                or self.w(f"{path}.up.q000") is not None:
            return self.up_phases(x, g, dense, path, cout)
        gu = GridShape(2 * g.H, 2 * g.W, cin)
        u = self.new_grid(gu)
        for dy in range(2):
            for dx in range(2):
                src = (tap(tuple(x.shape), 0, [g.H, g.W, cin], [g.W * cin, cin, 1]) if dense else
                       tap((g.rows, cin), g.pixel(0, 0) * cin, [g.H, g.W, cin], [g.Wp * cin, cin, 1]))
                Copy.step(x, u, src=src,
                          dst=tap((gu.rows, cin), gu.pixel(dy, dx) * cin, [g.H, g.W, cin], [2 * gu.Wp * cin, 2 * cin, 1]))
        vu = self.view(u)
        y = self.conv3(vu, gu, f"{path}.up", cout, parity=cin != cout)
        self.free_grid(vu)
        return y

    def up_phases(self, x, g: GridShape, dense: bool, path: str, cout: int):
        """up() at the low resolution (_up_phases): for each output phase, the sum of four GEMMs
        over runs of x's grid rows, copied into every other row and pixel of the output grid.
        A quarter of the pixels each, 4 taps for 9: 4/9 of the products, and no upsampled grid."""
        if dense:
            x = self.dense_to_grid(x, g)
        gout = GridShape(2 * g.H, 2 * g.W, cout)
        out = self.new_grid(gout)
        Wp = g.Wp
        R = max(r for r in range(16, min(VAE_TAP_ROWS, g.Hp) + 1, 16) if g.Hp % r == 0)
        for r0 in range(0, g.H, R):
            n, rows = R * Wp, min(R, g.H - r0)
            for py in range(2):
                for px in range(2):
                    y = None
                    oy0, ox0 = self.PHASE_TAPS[py][0][0], self.PHASE_TAPS[px][0][0]
                    one, C = self.w(f"{path}.up.q{py}{px}"), g.C
                    if one is not None or self.w(f"{path}.up.q{py}{px}0") is not None:
                        # the phase's taps read in place (GridGEMM): one GEMM, or one a low row
                        from iron.operators.flm.gemm.design import Epilogue
                        from iron_gridgemm import GridGEMM

                        kw = dict(b_col_maj=True, epilogue_modes=(Epilogue.NONE,), precision=PREC.vae)
                        s0 = g.band + (r0 + oy0) * Wp + ox0
                        if one is not None:
                            MA = n + Wp + 2
                            y = GridGEMM.step(x[s0 : s0 + MA], one, M=n, MA=MA, CA=C, k_stride=Wp * C, k_tile=2 * C, **kw)
                        else:
                            for a in range(2):
                                t = GridGEMM.step(x[s0 + a * Wp : s0 + a * Wp + n + 2], self.w(f"{path}.up.q{py}{px}{a}"),
                                                  M=n, MA=n + 2, CA=C, **kw)
                                y = t if y is None else y + t
                        y = y + self.w(f"{path}.up.b")
                    else:
                        for a, (oy, _) in enumerate(self.PHASE_TAPS[py]):
                            for bb, (ox, _) in enumerate(self.PHASE_TAPS[px]):
                                s0 = g.band + (r0 + oy) * Wp + ox
                                t = F.linear(x[s0 : s0 + n], self.w(f"{path}.up.p{py}{px}{a}{bb}"),
                                             self.w(f"{path}.up.b") if y is None else None, precision=PREC.vae)
                                y = t if y is None else y + t
                    Copy.step(y, out, src=tap((n, cout), cout, [rows, g.W, cout], [Wp * cout, cout, 1]),
                              dst=tap((gout.rows, cout), gout.pixel(2 * r0 + py, px) * cout, [rows, g.W, cout],
                                      [2 * gout.Wp * cout, 2 * cout, 1]))
            command_boundary()  # a chunk's control code (the device heap holds 64 MiB)
        self.free_grid(x)
        return self.view(out), gout

    def head(self, x, g: GridShape):
        """norm_out (weight and bias), ReLU, conv_out: the pixels (H W, 64), RGB first."""
        hg = self.new_grid(g)
        R = g.chunk(g.C)
        for r0 in range(0, g.H, R):
            s = g.band + r0 * g.Wp
            h = ReLU.step(F.rms_norm(x[s : s + R * g.Wp], (g.C,), self.w("head.n"), 1e-5) + self.w("head.nb"))
            self.interior_from(h, g, hg, r0, R)
        y, gy = self.conv3(self.view(hg), g, "head", 64)
        return Copy.step(y, src=tap((gy.rows, 64), gy.pixel(0, 0) * 64, [g.H, g.W * 64], [gy.Wp * 64, 1])).reshape(g.H * g.W, 64)

    def body(self, x):
        g = self.g_in
        dense = False
        for j, op in enumerate(self.ops):
            kind = op[0]
            if j and kind in ("res", "vit") and self.ops[j - 1][0] == kind:
                command_boundary()  # a block a command: the device heap, amdxdna's 2 s
            if kind == "in":
                x, g = self.conv3(x, g, "conv_in", VAE_C[-1])
            elif kind == "vit":
                if not dense:
                    x = self.grid_to_dense(x, g)
                    dense = True
                x = self.vit(x, g.H, g.W, op[1])
                g = GridShape(g.H, g.W, op[2])
            elif kind == "up":
                x, g = self.up(x, g, dense, op[1], op[2], op[3])
                dense = False
            elif kind == "res":
                if dense:
                    x, dense = self.dense_to_grid(x, g), False
                x, g = self.res(x, g, op[1])
            elif kind == "head":
                if dense:
                    x, dense = self.dense_to_grid(x, g), False
                return self.head(x, g)
        self.g_out = g
        if dense:
            x = self.dense_to_grid(x, g)
        # a state's view is no output of the graph's own: return a copy
        return Copy.step(x)


def vae_stages(width: int, height: int) -> list[tuple[list, GridShape]]:
    """The decoder in stages (each a graph, under amdxdna's 2 s a command), with each one's
    input grid."""
    h, w = height // 32, width // 32
    d = "decoder.up_blocks"
    vit = lambda i, j: ("vit", f"{d}.{i}.{j}", VAE_C[i])  # noqa: E731
    res = lambda i, j: ("res", f"{d}.{i}.{j}", VAE_C[i])  # noqa: E731
    up = lambda i: ("up", f"{d}.{i}.0", VAE_C[i + 1], VAE_C[i])  # noqa: E731
    G = lambda s, c: GridShape(h * s, w * s, c)  # noqa: E731
    return [
        ([("in",), vit(5, 0), vit(5, 1), vit(5, 2)], G(1, 64)),
        ([up(4), vit(4, 1), vit(4, 2), vit(4, 3)], G(1, 1024)),
        ([up(3), vit(3, 1)], G(2, 1024)),
        ([vit(3, 2), vit(3, 3)], G(4, 512)),
        ([up(2), res(2, 1), res(2, 2), res(2, 3)], G(4, 512)),
        ([up(1), res(1, 1)], G(8, 512)),
        ([res(1, 2), res(1, 3)], G(16, 256)),
        ([up(0)], G(16, 256)),
        ([res(0, 1), res(0, 2), res(0, 3)], G(32, 128)),
        ([("head",)], G(32, 128)),
    ]


def export_vae(root: Path, out: Path, width: int, height: int, check: bool, only: list | None = None) -> None:
    w = (RandomWeights if ARGS.random else Safetensors)(root / "vae" / "diffusion_pytorch_model.safetensors")
    cfg = json.loads((root / "vae" / "config.json").read_text())
    assert cfg["decoder_block_out_channels"] == VAE_C and cfg["upsample_block_type"] == "interpolate", cfg
    stages = vae_stages(width, height)
    d = out / "vae"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, (ops, g) in enumerate(stages):
        if only and i not in only:
            continue
        t0 = time.time()
        st = VaeStage(ops, g, w, cfg["scaling_factor"])
        cg = st.compile(x=(g.rows, g.C))
        go = st.g_out if ops[-1][0] != "head" else None
        meta = dict(stage=i, ops=[list(o) for o in ops], grid_in=dataclasses.asdict(g),
                    grid_out=None if go is None else dataclasses.asdict(go), width=width, height=height)
        m = export_graph(cg, d / f"s{i}", f"vae_s{i}", meta)
        print(f"vae s{i}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            x = np.zeros((g.rows, g.C), np.float32)
            for y in range(g.H):
                x[g.pixel(y, 0) : g.pixel(y, 0) + g.W] = rng.standard_normal((g.W, g.C))
            if ops[0][0] == "in":
                x[:, 32:] = 0
            write_check(st, m, d / f"s{i}", dict(x=bf(x)))
    (d / "manifest.json").write_text(json.dumps(dict(stages=len(stages), width=width, height=height, wp="ceil16",
                                                     latent_channels=32, scaling_factor=cfg["scaling_factor"]), indent=1) + "\n")


# --- export ----------------------------------------------------------------------------------


def timestep_sinusoid(t: float, dim: int = 256) -> np.ndarray:
    half = dim // 2
    f = np.exp(-math.log(10000) * np.arange(half, dtype=np.float64) / half)
    a = t * f
    return np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)


def export_text(root: Path, out: Path, check: bool) -> None:
    c = TextConfig()
    w = Sharded(root / "text_encoder", random=ARGS.random)
    t0 = time.time()
    prefix = None
    if not ARGS.no_prefix:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(root / "tokenizer" / "tokenizer.json"))
        ids = [tok.token_to_id("<bos>")] + tok.encode(INSTRUCTION, add_special_tokens=False).ids
        prefix = Prefix.of(c, w, ids)
        R = ARGS.text_chunk
        c = dataclasses.replace(c, rows=R, prefix=(len(ids) - 1) // R * R, cache=-(-(len(ids) + 298) // 256) * 256)
        print(f"text: the instruction's {len(ids)} rows on the CPU in {time.time() - t0:.0f} s", flush=True)
    g = Text(c, w, prefix)
    cg = g.compile(x=(c.rows, c.hidden))
    d = out / "text"
    meta = dict(config=dataclasses.asdict(c), weights=TEXT_FMT["fmt"], precision=PREC.text.value)
    if prefix is not None:
        meta["prefix"] = prefix.ids
    m = export_graph(cg, d, "text", meta)
    if prefix is not None:
        (d / "prefix_out.bin").write_bytes(bf(prefix.out).tobytes())
    print(f"text: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    import shutil

    shutil.copyfile(root / "tokenizer" / "tokenizer.json", d / "tokenizer.json")
    with open(d / "embed.bin", "wb") as f:
        e = w["embed_tokens.weight"]
        for s in range(0, e.shape[0], 16384):
            f.write(bf(e[s : s + 16384]).tobytes())
    if check and prefix is not None:
        # the instruction and 13 tokens after it, from the prefix's end in one chunk: against
        # the whole sequence in float32 on the CPU (check/out_*.bin, zeros past the tokens)
        rng = np.random.default_rng(0)
        full = prefix.ids + [int(t) for t in rng.integers(1000, 200000, 13)]
        P = c.prefix
        n = len(full) - P
        assert n <= c.rows, (n, c.rows)
        want = np.zeros((c.rows, c.hidden), np.float32)
        want[:n] = Prefix.of(c, w, full).out[P:]
        x = np.zeros((c.rows, c.hidden), np.float32)
        x[:n] = f32(w["embed_tokens.weight"][full[P:]]) * math.sqrt(c.hidden)
        cd = d / "check"
        cd.mkdir(exist_ok=True)
        (cd / "in_x.bin").write_bytes(bf(x).tobytes())
        (cd / "values.json").write_text(json.dumps(dict(start=P)) + "\n")
        (cd / f"out_{m['graph']['outputs'][0]['name']}.bin").write_bytes(bf(want).tobytes())
        (cd / "rows.txt").write_text(f"{n}\n")
    elif check:
        rng = np.random.default_rng(0)
        x = np.zeros((c.rows, c.hidden), np.float32)
        n = 219
        x[:n] = f32(w["embed_tokens.weight"][np.sort(rng.integers(0, 200000, n))]) * math.sqrt(c.hidden)
        write_check(g, m, d, dict(x=bf(x)))


def export_dit(root: Path, out: Path, c: DitConfig, check: bool) -> None:
    w = (RandomWeights if ARGS.random else Safetensors)(root / "transformer" / "diffusion_pytorch_model.safetensors")
    rng = np.random.default_rng(0)
    meta = dict(config=dataclasses.asdict(c))
    E, n = c.hidden, c.ntok
    if "caption" in ARGS.only:
        t0 = time.time()
        g = Caption(c, w)
        cg = g.compile(y=(c.text_rows, c.caption))
        m = export_graph(cg, out / "caption", "caption", meta)
        print(f"caption: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            write_check(g, m, out / "caption", dict(y=bf(rng.standard_normal((c.text_rows, c.caption)) * 3)))
    if "dit_in" in ARGS.only:
        t0 = time.time()
        g = DitIn(c, w)
        cg = g.compile(patches=(n, c.latent), tproj=(256,))
        m = export_graph(cg, out / "dit_in", "dit_in", meta)
        print(f"dit_in: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            ins = dict(patches=bf(rng.standard_normal((n, c.latent))), tproj=bf(timestep_sinusoid(0.78)))
            write_check(g, m, out / "dit_in", ins)
    if "dit_out" in ARGS.only:
        t0 = time.time()
        g = DitOut(c, w)
        cg = g.compile(x=(n, E), emb=(E,))
        m = export_graph(cg, out / "dit_out", "dit_out", meta)
        print(f"dit_out: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            write_check(g, m, out / "dit_out", dict(x=bf(rng.standard_normal((n, E)) * 4), emb=bf(rng.standard_normal(E))))
    if "dit_blocks" in ARGS.only:
        t0 = time.time()
        g = DitBlocks(c, 0, w)
        cg = g.compile(x=(n, E), t6=(6 * E,), ctx=(c.text_rows, E))
        d = out / "dit_blocks"
        groups = c.layers // c.group
        m = export_graph(cg, d, "dit_blocks", dict(meta, groups=groups))
        print(f"dit_blocks: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        variants = [{k: b["image"] for k, b in m["banks"].items()}]
        for i in range(1, groups if not ARGS.one_group else 1):
            variants.append(variant_bank(cg, DitBlocks(c, i * c.group, w), d, f"bank_g{i}"))
            print(f"dit_blocks: bank of blocks {i * c.group}..{(i + 1) * c.group - 1}", flush=True)
        m["variants"] = variants
        (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
        if check:
            ctx = rng.standard_normal((c.text_rows, E))
            write_check(g, m, d, dict(x=bf(rng.standard_normal((n, E))), t6=bf(rng.standard_normal(6 * E) * 0.3),
                                      ctx=bf(ctx)), dict(nk=13))


def write_model(root: Path, out: Path, c: DitConfig) -> None:
    """model.json: which Sana the export is and its scheduler's settings (internal/imagegen/sana.Model)."""
    sc = json.loads((root / "scheduler" / "scheduler_config.json").read_text())
    m = dict(name=root.name, guidance=4.5)
    assert sc["_class_name"] == "DPMSolverMultistepScheduler" and sc["prediction_type"] == "flow_prediction", sc
    assert sc["algorithm_type"] == "dpmsolver++" and sc["solver_order"] == 2 and sc["use_flow_sigmas"], sc
    m.update(steps=20, flow_shift=sc["flow_shift"], num_train_timesteps=sc["num_train_timesteps"])
    (out / "model.json").write_text(json.dumps(m, indent=1) + "\n")


ARGS = None


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a diffusers repository of Sana (transformer/, vae/, text_encoder/, tokenizer/)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=1024, help="the picture's width and height")
    ap.add_argument("--width", type=int, default=0, help="the picture's width (Sana's bins: 1152x896, 1344x768, ...)")
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=["text", "caption", "dit_in", "dit_blocks", "dit_out", "vae"],
                    help="parts to export: text, caption, dit_in, dit_blocks, dit_out, vae")
    ap.add_argument("--stages", type=int, nargs="*", help="the VAE's stages to export (default all)")
    ap.add_argument("--dit-proj", default=DIT_FMT["proj"], choices=("bfp16", "bf16"),
                    help="the DiT's projections that are a GEMM's B: bfp16 packed on the host where they tile, or bf16")
    ap.add_argument("--dit-dw", default=DW["form"], choices=("op", "eltwise"),
                    help="GLUMBConv's depthwise convolutions: one DWConv3 operator a branch (the MLP padded to its bands) or 18 elementwise operators")
    ap.add_argument("--dit-precision", default=PREC.dit.value, choices=[p.value for p in Precision],
                    help="the DiT's projections (high: the same pictures as medium, 2.5 times as slow)")
    ap.add_argument("--vae-precision", default=PREC.vae.value, choices=[p.value for p in Precision])
    ap.add_argument("--vae-attn-precision", default=PREC.vae_attn.value, choices=[p.value for p in Precision],
                    help="the VAE's linear attentions (their two products)")
    ap.add_argument("--text-format", default=TEXT_FMT["fmt"], choices=("q8", "q6", "q4", "bf16"),
                    help="Gemma 2's projections: q6 (6.5 bits a weight), q8, q4 (changes the pictures) or bf16")
    ap.add_argument("--vae-conv", default=VAE_CONV[0], choices=("im2col", "taps", "grid"),
                    help="vae: the 3 x 3 convolutions (but the upsamplings that halve the channels) as one GEMM over im2col rows or a GEMM a tap")
    ap.add_argument("--vae-up", default=VAE_UP[0], choices=("phase", "grid"),
                    help="vae: the upsamplings as four 2 x 2 convolutions at the low resolution, or the 2x grid and a 3 x 3")
    ap.add_argument("--vae-dw", default=VAE_DW["form"], choices=("op", "eltwise"),
                    help="vae: the EfficientViT's depthwise convolutions on DWConv3 (where its bands fit) or elementwise operators")
    ap.add_argument("--vae-cols-mib", type=int, default=VAE_COLS_MIB[0], help="vae: a convolution's im2col rows at most (MiB)")
    ap.add_argument("--text-chunk", type=int, default=64, help="text with a prefix: the rows a call computes")
    ap.add_argument("--no-prefix", action="store_true",
                    help="text: compute the instruction's rows on every call (without its keys and values held)")
    ap.add_argument("--text-precision", default=PREC.text.value, choices=[p.value for p in Precision],
                    help="Gemma 2's matmuls (medium: bfp16 operands)")
    ap.add_argument("--one-group", action="store_true", help="dit_blocks: the first group's bank only (a test)")
    ap.add_argument("--layers", type=int, default=0, help="the DiT's blocks (a test of fewer)")
    ap.add_argument("--random", action="store_true", help="weights drawn at random in the checkpoint's shapes")
    ap.add_argument("--check", action="store_true", help="write each graph's inputs and reference outputs")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    TEXT_FMT["fmt"] = a.text_format
    DIT_FMT["proj"] = a.dit_proj
    DW["form"] = a.dit_dw
    VAE_DW["form"] = a.vae_dw
    if a.dit_proj == "bfp16":
        import iron_q4

        iron_q4.install_linear()  # F.linear runs the packed weights on flm.ProjectionGEMM
    bind_device(a.device, a.cols)
    cfg = json.loads((a.root / "transformer" / "config.json").read_text())
    c = DitConfig.of(cfg, a.width or a.size, a.height or a.size, a.group)
    if a.layers:
        c = dataclasses.replace(c, layers=a.layers)
    PREC.dit = Precision(a.dit_precision)
    PREC.vae = Precision(a.vae_precision)
    PREC.vae_attn = Precision(a.vae_attn_precision)
    PREC.text = Precision(a.text_precision)
    VAE_COLS_MIB[0] = a.vae_cols_mib
    VAE_CONV[0] = a.vae_conv
    VAE_UP[0] = a.vae_up
    a.out.mkdir(parents=True, exist_ok=True)
    write_model(a.root, a.out, c)
    if "text" in a.only:
        export_text(a.root, a.out, a.check)
    if {"caption", "dit_in", "dit_blocks", "dit_out"} & set(a.only):
        export_dit(a.root, a.out, c, a.check)
    if "vae" in a.only:
        export_vae(a.root, a.out, c.width, c.height, a.check, a.stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
