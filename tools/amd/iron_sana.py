#!/usr/bin/env python3
"""Sana and Sana-Sprint (NVlabs) on IRON: the graphs cmd/hwnpu-image runs on AMD's NPU.

An application of amd/IRON kept in hwnpu, as iron_anima.py is. Sana is a linear DiT: each block's
self-attention is ReLU linear attention (no softmax, so its cost grows with the tokens, not
their square), its cross-attention a softmax attention to the prompt, its MLP a gated 1 x 1,
depthwise 3 x 3, 1 x 1 convolution (GLUMBConv) over the token grid. The prompt is read by Gemma
2 2B, the latents are DC-AE's (32 channels, a 32nd of the image's sides: 1024 tokens at 1024 x
1024). Sana-Sprint is distilled to 1 to 4 steps (SCM) with the guidance scale embedded; Sana
runs 20 or so flow DPM-Solver steps with classifier-free guidance (two DiT calls a step).

The graphs, each exported by iron_graph.py into a directory of its own:

  text        Gemma 2 2B over up to ``rows`` prompt tokens (their embedding rows times
              sqrt(hidden), which the host looks up), causal, its last hidden state after the
              final norm; shared by every Sana of a VAE generation (the same Gemma 2)
  caption     the DiT's caption projection and norm of the selected 300 rows (padded to 512)
  dit_in      the patch embedding (a 1 x 1 convolution) and the timestep (and guidance)
              embedding and its six modulations
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

  python tools/amd/iron_sana.py ~/models/Sana_Sprint_0.6B_1024px OUT --size 1024

See docs/amd-npu-models.md.
"""

import argparse
import dataclasses
import json
import math
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
from iron_anima import RandomWeights, Safetensors, bf, command_boundary, tap, variant_bank  # noqa: E402
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


PREC = Precisions()

# The text encoder's projections: "q8" (iron_q4's 8.5 bits a weight, dequantized to bf16 on the
# device before each GEMM, which keeps PREC.text) or "bf16".
TEXT_FMT = {"fmt": "q8"}


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


def linear_attention(q, kt, vt, k: LinearAttnConsts):
    """Each head's ``relu(q) (relu(K)^T V) / relu(q) . sum(relu(K))``, 64 columns to a head.

    ``q`` (n, 32 H) and ``kt`` (32 H, n) past their ReLU; ``vt`` (64 H, n), the values 64 rows
    to a head (value_rows), the constant rows added here.
    """
    n, H = q.shape[0], k.heads
    w = F.linear(vt + k.const, kt, precision=PREC.attn)  # (64 H, 32 H): V^T K and c sum(K)
    o = F.linear(q, w * k.mask, precision=PREC.attn)  # (n, 64 H): [N, c D, 0...] a head
    return F.rms_norm(o.reshape(n * H, 64), (64,), k.sel, 1e-30).reshape(n, 64 * H)


# --- the DiT -------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DitConfig:
    """SanaTransformer2DModel's shape (transformer/config.json) at one image size."""

    size: int = 1024
    hidden: int = 1152
    heads: int = 36
    layers: int = 28
    cross_heads: int = 16
    cross_head_dim: int = 72
    mlp: int = 2880
    caption: int = 2304
    latent: int = 32
    guidance: bool = True
    guidance_scale: float = 0.1  # guidance_embeds_scale: the guidance is embedded times it
    qk_norm: bool = True
    text_rows: int = 512  # the 300 rows Sana keeps of the prompt, padded
    group: int = 4
    vae_scale: int = 32

    @property
    def grid(self) -> int:
        return self.size // self.vae_scale

    @property
    def tokens(self) -> int:
        return self.grid * self.grid

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
    def of(cls, cfg: dict, size: int, group: int) -> "DitConfig":
        E = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        assert cfg["attention_head_dim"] == 32 and cfg.get("patch_size", 1) == 1, cfg
        assert cfg.get("interpolation_scale") is None, "a positional embedding is not supported"
        return cls(size=size, hidden=E, heads=cfg["num_attention_heads"], layers=cfg["num_layers"],
                   cross_heads=cfg["num_cross_attention_heads"], cross_head_dim=cfg["cross_attention_head_dim"],
                   mlp=int(cfg["mlp_ratio"] * E), caption=cfg["caption_channels"], latent=cfg["in_channels"],
                   guidance=bool(cfg.get("guidance_embeds")), guidance_scale=cfg.get("guidance_embeds_scale") or 0.1,
                   qk_norm=cfg.get("qk_norm") is not None, group=group)


# The depthwise convolution's grid: the tokens' rows of W, each padded with one zero pixel
# after it, a zero row above and below and a zero pixel before the first.


def dw_rows(c: DitConfig) -> int:
    W = c.grid
    return (W + 2) * (W + 1) + 2


def dw_band(c: DitConfig) -> int:
    """The grid row of token (0, 0)."""
    return 1 + (c.grid + 1)


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
        n, Ch = c.tokens, c.mlp
        self.lin = LinearAttnConsts(c.att_heads, n)
        self.one = iron.param(bf(np.ones(c.hidden)))
        self.eye = iron.param(bf(np.eye(c.att_dim))) if c.qk_norm else None
        self.grid_a = iron.state((dw_rows(c), Ch))
        self.grid_g = iron.state((dw_rows(c), Ch))
        self.cross = iron.state((n, c.cross_heads * c.cross_pad))


class SanaBlock(iron.Graph):
    """A Sana block: adaLN-single (a learned table plus the timestep's six rows), linear
    self-attention, cross-attention to the caption, GLUMBConv."""

    def __init__(self, c: DitConfig, w: Safetensors | None, i: int, sh: Shared):
        E, H, Ch = c.hidden, c.att_heads, c.mlp
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
            self.ia, self.iab = iron.param(Ch, E), iron.param(Ch)
            self.ig, self.igb = iron.param(Ch, E), iron.param(Ch)
            self.da, self.dab = iron.param(9, Ch), iron.param(Ch)
            self.dg, self.dgb = iron.param(9, Ch), iron.param(Ch)
            self.pt = iron.param(E, Ch)
            if c.qk_norm:
                self.qn, self.kn = iron.param(A), iron.param(A)
                self.cqn, self.ckn = iron.param(P), iron.param(P)
            return
        self.sst = iron.param(bf(f32(w[p + "scale_shift_table"]).reshape(-1)))
        pad = lambda x: np.concatenate([f32(x), np.zeros((A - E,) + x.shape[1:], np.float32)])  # noqa: E731
        self.q = iron.param(bf(pad(w[p + "attn1.to_q.weight"])))
        self.k = iron.param(bf(pad(w[p + "attn1.to_k.weight"])))
        self.v = iron.param(bf(value_rows(f32(w[p + "attn1.to_v.weight"]), H)))
        self.o = iron.param(bf(out_cols(f32(w[p + "attn1.to_out.0.weight"]), H)))
        self.ob = iron.param(bf(w[p + "attn1.to_out.0.bias"]))
        Hc, Dc, Dp = c.cross_heads, c.cross_head_dim, c.cross_pad
        fold = mha_fold(1 / math.sqrt(Dc))
        for n in ("q", "k", "v"):
            m, b = cross_rows(f32(w[p + f"attn2.to_{n}.weight"]), f32(w[p + f"attn2.to_{n}.bias"]), Hc, Dc, Dp)
            if n == "q" and not c.qk_norm:
                m, b = m * fold, b * fold
            setattr(self, f"c{n}", iron.param(bf(m)))
            setattr(self, f"c{n}b", iron.param(bf(b)))
        wo = f32(w[p + "attn2.to_out.0.weight"])
        wop = np.zeros((E, Hc * Dp), np.float32)
        for h in range(Hc):
            wop[:, h * Dp : h * Dp + Dc] = wo[:, h * Dc : (h + 1) * Dc]
        self.co = iron.param(bf(wop))
        self.cob = iron.param(bf(w[p + "attn2.to_out.0.bias"]))
        wi = f32(w[p + "ff.conv_inverted.weight"])[:, :, 0, 0]
        bi = f32(w[p + "ff.conv_inverted.bias"])
        self.ia, self.iab = iron.param(bf(wi[:Ch])), iron.param(bf(bi[:Ch]))
        self.ig, self.igb = iron.param(bf(wi[Ch:])), iron.param(bf(bi[Ch:]))
        wd = f32(w[p + "ff.conv_depth.weight"])[:, 0].reshape(2 * Ch, 9).T  # (9, 2 Ch), tap dy * 3 + dx
        bd = f32(w[p + "ff.conv_depth.bias"])
        self.da, self.dab = iron.param(bf(wd[:, :Ch])), iron.param(bf(bd[:Ch]))
        self.dg, self.dgb = iron.param(bf(wd[:, Ch:])), iron.param(bf(bd[Ch:]))
        self.pt = iron.param(bf(f32(w[p + "ff.conv_point.weight"])[:, :, 0, 0]))
        if c.qk_norm:
            # across the heads, the padded ones' zeros counted in the mean: weight times
            # sqrt(E / A), eps times E / A (self.aeps)
            self.qn = iron.param(bf(pad(w[p + "attn1.norm_q.weight"]) * math.sqrt(E / A)))
            self.kn = iron.param(bf(pad(w[p + "attn1.norm_k.weight"]) * math.sqrt(E / A)))
            qn, self.ceps = cross_norm(f32(w[p + "attn2.norm_q.weight"]), Hc, Dc, Dp)
            kn, _ = cross_norm(f32(w[p + "attn2.norm_k.weight"]), Hc, Dc, Dp)
            self.cqn, self.ckn = iron.param(bf(qn * fold)), iron.param(bf(kn))

    def depthwise(self, x, grid, taps, bias):
        """``x`` (tokens, Ch) through a 3 x 3 depthwise convolution (zero padded), over the
        grid's band of rows (W + 1 to a token row; the pad column's rows are not tokens)."""
        c = self.c
        W, Wp = c.grid, c.grid + 1
        Ch = x.shape[1]
        band = dw_band(c)
        Copy.step(x, grid, dst=tap((dw_rows(c), Ch), band * Ch, [W, W * Ch], [Wp * Ch, 1]))
        n = W * Wp
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
        h = F.layer_norm(x, (E,), eps=1e-6) * (scale_msa + sh.one) + shift_msa
        q = F.linear(h, self.q, precision=PREC.dit)
        A = c.att_dim
        if c.qk_norm:
            q = F.rms_norm(q, (A,), self.qn, self.aeps)
            k = ReLU.step(F.rms_norm(F.linear(h, self.k, precision=PREC.dit), (A,), self.kn, self.aeps))
            kt = F.linear(sh.eye, k, precision=Precision.HIGHEST)  # the transpose, exactly
        else:
            kt = ReLU.step(F.linear(self.k, h, precision=PREC.dit))  # (A, n)
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
            o[s : s + L] = MHA.step(q[s : s + L], k[:nk], v[:nk], heads_interleaved=True, kv_interleaved=True,
                                    causal=False, d=Dp, scale=MHA_SCALE, num_pipelines=8).reshape(L, P)
        x = x + F.linear(o, self.co, self.cob, precision=PREC.dit)
        # GLUMBConv
        h = F.layer_norm(x, (E,), eps=1e-6) * (scale_mlp + sh.one) + shift_mlp
        a = F.silu(F.linear(h, self.ia, self.iab, precision=PREC.dit))
        g = F.silu(F.linear(h, self.ig, self.igb, precision=PREC.dit))
        a = self.depthwise(a, sh.grid_a, self.da, self.dab)
        g = self.depthwise(g, sh.grid_g, self.dg, self.dgb)
        f = a * F.silu(g)  # the band's rows; the pad column's are dropped next
        W, Wp, Ch = c.grid, c.grid + 1, c.mlp
        f = Copy.step(f, src=tap((W * Wp, Ch), 0, [W, W * Ch], [Wp * Ch, 1])).reshape(n, Ch)
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
    timestep's sinusoid (diffusers' Timesteps, cos first); ``gproj`` (256), the guidance's
    (Sana-Sprint; zeros otherwise, unused). Outputs ``x`` (tokens, hidden), ``t6`` (6 hidden),
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
        if c.guidance:
            self.t = TimestepEmbedding(w, "time_embed.timestep_embedder.", E)
            self.g = TimestepEmbedding(w, "time_embed.guidance_embedder.", E)
        else:
            self.t = TimestepEmbedding(w, "time_embed.emb.timestep_embedder.", E)
            self.g = None

    def body(self, patches, tproj, gproj=None):
        x = F.linear(patches, self.pe, self.peb, precision=PREC.dit)
        emb = self.t(tproj)
        if self.g is not None:
            emb = emb + self.g(gproj)
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
        h = F.layer_norm(x, (E,), eps=1e-6) * (self.scale + emb + self.one) + (self.shift + emb)
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

    rows: int = 512  # the instruction (about 210 tokens) and up to 298 of the prompt's
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

    def body(self, x, angles, pad):
        c = self.c
        L, H, G, D = x.shape[0], c.heads, c.kv_heads, c.head_dim
        pads = {pad.shape[1]: pad}
        h = F.rms_norm(x, (c.hidden,), self.in_norm, c.eps)
        q = text_linear(h, self.q, pads).reshape(L * H, D)
        k = text_linear(h, self.k, pads).reshape(L * G, D)
        v = text_linear(h, self.v, pads)
        q, k = F.rope(q, angles), F.rope(k, angles)
        o = MHA.step(q.reshape(L, H, D), k.reshape(L, G, D), v.reshape(L, G, D),
                     heads_interleaved=True, kv_interleaved=True, causal=True, d=D, scale=MHA_SCALE, num_pipelines=8)
        o = text_linear(o.reshape(L, H * D), self.o)
        x = x + F.rms_norm(o, (c.hidden,), self.post_attn, c.eps)
        h = F.rms_norm(x, (c.hidden,), self.pre_ff, c.eps)
        g = F.gelu(text_linear(h, self.gate, pads), approximate="tanh") * text_linear(h, self.up, pads)
        g = text_linear(g, self.down)
        return x + F.rms_norm(g, (c.hidden,), self.post_ff, c.eps)


class Text(iron.Graph):
    """Gemma 2 2B: input ``x`` (rows, hidden), the prompt's embedding rows times sqrt(hidden)
    (zeros past them); output its last hidden state after the final norm, the same shape."""

    def __init__(self, c: TextConfig, w=None):
        self.c = c
        self.layers = [GemmaLayer(c, w, i) for i in range(c.layers)]
        self.norm = iron.param(bf(f32(w["norm.weight"]) + 1)) if w is not None else iron.param(c.hidden)
        self.angles = iron.param(bf(half_rope_table(c.rows, c.head_dim, c.theta)))
        # the hidden rows in zero-padded columns, for q8's 512-column blocks (text_linear)
        self.pad = iron.state((c.rows, -(-c.hidden // 512) * 512))

    def body(self, x):
        c = self.c
        for i, layer in enumerate(self.layers):
            if i and i % c.split == 0:
                command_boundary()
            x = layer(x, self.angles, self.pad)
        return F.rms_norm(x, (c.hidden,), self.norm, c.eps)


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

from iron_anima import GridShape  # noqa: E402

VAE_C = [128, 256, 512, 512, 1024, 1024]  # decoder_block_out_channels
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
    Copy.step(x, state, dst=tap((g.rows, C), g.band * C, [g.H, g.W * C], [g.Wp * C, 1]))
    n = g.H * g.Wp
    y = None
    for dy in range(g.k):
        for dx in range(g.k):
            s = g.band + (dy - g.p) * g.Wp + (dx - g.p)
            t = state[s : s + n] * taps[dy * g.k + dx]
            y = t if y is None else y + t
    return y if bias is None else y + bias


def band_dense(y, g: DwGrid):
    """The pixels' rows of a band (H Wp, C): (H W, C)."""
    C = y.shape[1]
    return Copy.step(y, src=tap((g.H * g.Wp, C), 0, [g.H, g.W * C], [g.Wp * C, 1])).reshape(g.H * g.W, C)


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
        self._cols = {}
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
                self.consts.setdefault((Hh, n), LinearAttnConsts(Hh, n))
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
                    self._param(f"{path}.ms_{nm}_taps", taps[:, perm[j]])
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
                    self._conv3(f"{path}.up", wt, b, cin)
                else:
                    assert cin == 2 * cout, op
                    for i in range(2):
                        wi = wt.copy()
                        for c in range(cout):
                            wi[c, 2 * c + i, 1, 1] += 1
                        self._conv3(f"{path}.up{i}", wi, b, cin)
                H, W = 2 * H, 2 * W
            elif kind == "res":
                _, path, C = op
                self._conv3(f"{path}.conv1", f32(w[f"{path}.conv1.weight"]), f32(w[f"{path}.conv1.bias"]), C)
                self._conv3(f"{path}.conv2", f32(w[f"{path}.conv2.weight"]), None, C)
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
                self._conv3("head", full, b, wt.shape[1])
            else:
                raise ValueError(op)

    def _param(self, name: str, x) -> None:
        self._index[name] = len(self.params)
        self.params.append(iron.param(bf(x)))

    def w(self, name: str):
        return self.params[self._index[name]] if name in self._index else None

    def _conv3(self, name: str, wt: np.ndarray, b, cin_padded: int) -> None:
        """A 3 x 3 convolution (cout, cin, 3, 3) as an im2col matrix (cout, dy 3 cin + dx cin + ci)."""
        co, ci = wt.shape[:2]
        full = np.zeros((co, cin_padded, 3, 3), np.float32)
        full[:, :ci] = wt
        self._param(f"{name}.w", full.transpose(0, 2, 3, 1).reshape(co, -1))
        if b is not None:
            self._param(f"{name}.b", b)

    # --- grids ---

    def new_grid(self, g: GridShape):
        s = iron.state((g.rows, g.C))
        self.states.append(s)
        return s

    @staticmethod
    def view(s):
        return s[0 : s.shape[0]]

    def interior_from(self, src, g: GridShape, dst, r0: int, R: int, step: int = 1) -> None:
        """Copy ``R`` image rows of ``src`` (band rows, R Wp of them) into the grid ``dst``'s
        rows r0, r0 + step, ..."""
        C, Wp = g.C, g.Wp
        Copy.step(src, dst, src=tap((R * Wp, C), C, [R, g.W * C], [Wp * C, 1]),
                  dst=tap((g.rows, C), g.pixel(r0, 0) * C, [R, g.W * C], [step * Wp * C, 1]))

    def dense_to_grid(self, x, g: GridShape):
        out = self.new_grid(g)
        Copy.step(x, out, dst=tap((g.rows, g.C), g.pixel(0, 0) * g.C, [g.H, g.W * g.C], [g.Wp * g.C, 1]))
        return self.view(out)

    def cols_state(self, n: int, k: int):
        """A state for im2col rows (n, k), one per shape, reused by every chunk."""
        key = (n, k)
        if key not in self._cols:
            self._cols[key] = iron.state((n, k))
            self.states.append(self._cols[key])
        return self._cols[key]

    def grid_to_dense(self, x, g: GridShape):
        C = g.C
        return Copy.step(x, src=tap((g.rows, C), g.pixel(0, 0) * C, [g.H, g.W * C], [g.Wp * C, 1])).reshape(g.H * g.W, C)

    def conv3(self, x, g: GridShape, wname: str, cout: int, post=None, parity: bool = False):
        """The 3 x 3 convolution of the grid x into a new grid of cout channels. ``post(y, r0,
        R)`` (when given) takes each chunk's GEMM result (band rows) before it is copied in;
        ``parity``: the even and the odd rows each with their own weight (wname + "0"/"1")."""
        gout = GridShape(g.H, g.W, cout)
        out = self.new_grid(gout)
        Wp, C = g.Wp, g.C
        R = g.chunk(C)
        if parity:
            R = max(32, R) if g.H % max(32, R) == 0 else R
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
                name = wname + ("" if i is None else str(i))
                y = F.linear(cols, self.w(f"{name}.w"), self.w(f"{name}.b"), precision=PREC.vae)
                if post is not None:
                    y = post(y, first, rows, step)
                self.interior_from(y, gout, out, first, rows, step)
        return self.view(out), gout

    # --- the blocks ---

    def res(self, x, g: GridShape, path: str):
        """conv1, SiLU, conv2, RMS norm (weight and bias), plus the input."""
        a, ga = self.conv3(x, g, f"{path}.conv1", g.C, post=lambda y, *_: F.silu(y))

        def post(y, r0, rows, step):
            s = g.band + r0 * g.Wp
            h = F.rms_norm(y, (g.C,), self.w(f"{path}.n"), 1e-5) + self.w(f"{path}.nb")
            return h + x[s : s + rows * g.Wp]

        return self.conv3(a, ga, f"{path}.conv2", g.C, post=post)

    def dwconv(self, x, H: int, W: int, k: int, taps, bias=None):
        """``x`` (H W, C) dense through a depthwise convolution: the band (DwGrid's)."""
        g = DwGrid(H, W, k)
        return dw_band_conv(x, self.dw[(H, W, k, x.shape[1])], g, taps, bias), g

    def vit(self, xd, H: int, W: int, path: str):
        """An EfficientViT block on the dense tokens xd (n, C)."""
        n, C = xd.shape
        Hh = C // 32
        k = self.consts[(Hh, n)]
        P = PREC.vae
        p = lambda s: self.w(f"{path}.{s}")  # noqa: E731
        q = F.linear(xd, p("q"), precision=P)
        kk = F.linear(xd, p("k"), precision=P)
        v = F.linear(xd, p("v"), precision=P)
        kt = ReLU.step(F.linear(p("kt"), xd, precision=P))
        vt = F.linear(p("vt"), xd, precision=P)
        o1 = linear_attention(ReLU.step(q), kt, vt, k)
        # the multi-scale branch: 5 x 5 depthwise, then the grouped 1 x 1 (block diagonal)
        qm = band_dense(*self.dwconv(q, H, W, 5, p("ms_q_taps")))
        km = band_dense(*self.dwconv(kk, H, W, 5, p("ms_k_taps")))
        vm = band_dense(*self.dwconv(v, H, W, 5, p("ms_v_taps")))
        o2 = linear_attention(ReLU.step(F.linear(qm, p("ms_q"), precision=P)), ReLU.step(F.linear(p("ms_k"), km, precision=P)),
                              F.linear(p("ms_v"), vm, precision=P), k)
        a = F.linear(o1, p("o1"), precision=P) + F.linear(o2, p("o2"), precision=P)
        x1 = xd + F.rms_norm(a, (C,), p("on"), 1e-5) + p("onb")
        # GLUMBConv
        ha = F.silu(F.linear(x1, p("ia"), p("iab"), precision=P))
        hg = F.silu(F.linear(x1, p("ig"), p("igb"), precision=P))
        ya, g3 = self.dwconv(ha, H, W, 3, p("da"), p("dab"))
        yg, _ = self.dwconv(hg, H, W, 3, p("dg"), p("dgb"))
        f = band_dense(ya * F.silu(yg), g3)
        f = F.linear(f, p("pt"), precision=P)
        return x1 + F.rms_norm(f, (C,), p("fn"), 1e-5) + p("fnb")

    def up(self, x, g: GridShape, dense: bool, path: str, cin: int, cout: int):
        """Nearest 2x of x (a grid, or dense), then the convolution with the shortcut folded."""
        gu = GridShape(2 * g.H, 2 * g.W, cin)
        u = self.new_grid(gu)
        for dy in range(2):
            for dx in range(2):
                src = (tap((g.H * g.W, cin), 0, [g.H, g.W, cin], [g.W * cin, cin, 1]) if dense else
                       tap((g.rows, cin), g.pixel(0, 0) * cin, [g.H, g.W, cin], [g.Wp * cin, cin, 1]))
                Copy.step(x, u, src=src,
                          dst=tap((gu.rows, cin), gu.pixel(dy, dx) * cin, [g.H, g.W, cin], [2 * gu.Wp * cin, 2 * cin, 1]))
        if cin == cout:
            return self.conv3(self.view(u), gu, f"{path}.up", cout)
        return self.conv3(self.view(u), gu, f"{path}.up", cout, parity=True)

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


def vae_stages(size: int) -> list[tuple[list, GridShape]]:
    """The decoder in stages (each a graph, under amdxdna's 2 s a command), with each one's
    input grid."""
    h = size // 32
    d = "decoder.up_blocks"
    vit = lambda i, j: ("vit", f"{d}.{i}.{j}", VAE_C[i])  # noqa: E731
    res = lambda i, j: ("res", f"{d}.{i}.{j}", VAE_C[i])  # noqa: E731
    up = lambda i: ("up", f"{d}.{i}.0", VAE_C[i + 1], VAE_C[i])  # noqa: E731
    G = lambda s, c: GridShape(h * s, h * s, c)  # noqa: E731
    return [
        ([("in",), vit(5, 0), vit(5, 1), vit(5, 2)], G(1, 64)),
        ([up(4), vit(4, 1), vit(4, 2), vit(4, 3)], G(1, 1024)),
        ([up(3), vit(3, 1)], G(2, 1024)),
        ([vit(3, 2)], G(4, 512)),
        ([vit(3, 3)], G(4, 512)),
        ([up(2), res(2, 1), res(2, 2), res(2, 3)], G(4, 512)),
        ([up(1), res(1, 1)], G(8, 512)),
        ([res(1, 2), res(1, 3)], G(16, 256)),
        ([up(0)], G(16, 256)),
        ([res(0, 1)], G(32, 128)),
        ([res(0, 2)], G(32, 128)),
        ([res(0, 3)], G(32, 128)),
        ([("head",)], G(32, 128)),
    ]


def export_vae(root: Path, out: Path, size: int, check: bool, only: list | None = None) -> None:
    w = (RandomWeights if ARGS.random else Safetensors)(root / "vae" / "diffusion_pytorch_model.safetensors")
    cfg = json.loads((root / "vae" / "config.json").read_text())
    assert cfg["decoder_block_out_channels"] == VAE_C and cfg["upsample_block_type"] == "interpolate", cfg
    stages = vae_stages(size)
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
                    grid_out=None if go is None else dataclasses.asdict(go), pad=16, width=size, height=size)
        m = export_graph(cg, d / f"s{i}", f"vae_s{i}", meta)
        print(f"vae s{i}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            x = np.zeros((g.rows, g.C), np.float32)
            for y in range(g.H):
                x[g.pixel(y, 0) : g.pixel(y, 0) + g.W] = rng.standard_normal((g.W, g.C))
            if ops[0][0] == "in":
                x[:, 32:] = 0
            write_check(st, m, d / f"s{i}", dict(x=bf(x)))
    (d / "manifest.json").write_text(json.dumps(dict(stages=len(stages), width=size, height=size, pad=16,
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
    g = Text(c, w)
    cg = g.compile(x=(c.rows, c.hidden))
    d = out / "text"
    m = export_graph(cg, d, "text", dict(config=dataclasses.asdict(c), weights=TEXT_FMT["fmt"]))
    print(f"text: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    import shutil

    shutil.copyfile(root / "tokenizer" / "tokenizer.json", d / "tokenizer.json")
    with open(d / "embed.bin", "wb") as f:
        e = w["embed_tokens.weight"]
        for s in range(0, e.shape[0], 16384):
            f.write(bf(e[s : s + 16384]).tobytes())
    if check:
        rng = np.random.default_rng(0)
        x = np.zeros((c.rows, c.hidden), np.float32)
        n = 219
        x[:n] = f32(w["embed_tokens.weight"][np.sort(rng.integers(0, 200000, n))]) * math.sqrt(c.hidden)
        write_check(g, m, d, dict(x=bf(x)))


def export_dit(root: Path, out: Path, c: DitConfig, check: bool) -> None:
    w = (RandomWeights if ARGS.random else Safetensors)(root / "transformer" / "diffusion_pytorch_model.safetensors")
    rng = np.random.default_rng(0)
    meta = dict(config=dataclasses.asdict(c))
    E, n = c.hidden, c.tokens
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
        shapes = dict(patches=(n, c.latent), tproj=(256,))
        if c.guidance:
            shapes["gproj"] = (256,)
        cg = g.compile(**shapes)
        m = export_graph(cg, out / "dit_in", "dit_in", meta)
        print(f"dit_in: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            ins = dict(patches=bf(rng.standard_normal((n, c.latent))), tproj=bf(timestep_sinusoid(0.78)))
            if c.guidance:
                ins["gproj"] = bf(timestep_sinusoid(0.45))
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
    m = dict(name=root.name, sprint=c.guidance, guidance=4.5)
    if c.guidance:
        assert sc["_class_name"] == "SCMScheduler" and sc["prediction_type"] == "trigflow", sc
        m.update(steps=2, sigma_data=sc["sigma_data"], max_timesteps=1.57080)
    else:
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
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=["text", "caption", "dit_in", "dit_blocks", "dit_out", "vae"],
                    help="parts to export: text, caption, dit_in, dit_blocks, dit_out, vae")
    ap.add_argument("--stages", type=int, nargs="*", help="the VAE's stages to export (default all)")
    ap.add_argument("--dit-precision", default=PREC.dit.value, choices=[p.value for p in Precision],
                    help="the DiT's projections (high: the same pictures as medium, 2.5 times as slow)")
    ap.add_argument("--vae-precision", default=PREC.vae.value, choices=[p.value for p in Precision])
    ap.add_argument("--text-format", default=TEXT_FMT["fmt"], choices=("q8", "bf16"),
                    help="Gemma 2's projections: q8 (8.5 bits a weight, half the pinned memory) or bf16")
    ap.add_argument("--one-group", action="store_true", help="dit_blocks: the first group's bank only (a test)")
    ap.add_argument("--layers", type=int, default=0, help="the DiT's blocks (a test of fewer)")
    ap.add_argument("--random", action="store_true", help="weights drawn at random in the checkpoint's shapes")
    ap.add_argument("--check", action="store_true", help="write each graph's inputs and reference outputs")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    TEXT_FMT["fmt"] = a.text_format
    bind_device(a.device, a.cols)
    cfg = json.loads((a.root / "transformer" / "config.json").read_text())
    c = DitConfig.of(cfg, a.size, a.group)
    if a.layers:
        c = dataclasses.replace(c, layers=a.layers)
    PREC.dit = Precision(a.dit_precision)
    PREC.vae = Precision(a.vae_precision)
    a.out.mkdir(parents=True, exist_ok=True)
    write_model(a.root, a.out, c)
    if "text" in a.only:
        export_text(a.root, a.out, a.check)
    if {"caption", "dit_in", "dit_blocks", "dit_out"} & set(a.only):
        export_dit(a.root, a.out, c, a.check)
    if "vae" in a.only:
        export_vae(a.root, a.out, a.size, a.check, a.stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
