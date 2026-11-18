"""The pieces of the DiT exporters that Anima, Chroma1 and FLUX.2 klein share: the text encoder's
weight formats (PREC, TEXT_FMT, text_param, text_linear), a diffusers part's tensors
(Checkpoint), the DiT's weights (qparam), the RoPE helpers (head_perm, rope_angles), the timestep's
sinusoid, the 2-D VAE stage FLUX's decoder builds on (VaeStage) and share_banks. Moved from the
removed iron_qwenimage.py and iron_sana.py (Qwen-Image's and Sana's exporters).
"""

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

import iron
import iron.functional as F
from iron import Precision
from iron.operators.copy import Copy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_anima  # noqa: E402
from iron_anima import RandomWeights, Safetensors, bf, command_boundary, tap  # noqa: E402


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


# VaeStage's matmuls take PREC.vae (iron_qwenimage's own Precisions, whose vae had the same default)
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

    w = np.array(w, np.float32)  # a copy: the checkpoint is mapped read-only (iron_sana's f32)
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


def f32(x) -> np.ndarray:
    return np.asarray(x, np.float32)


class Checkpoint:
    """A diffusers part's tensors by name, sharded (an index) or one file."""

    def __init__(self, d: Path, random: bool = False, prefix: str = ""):
        idx = next(iter(sorted(d.glob("*.safetensors.index.json"))), None)
        if idx is not None:
            index = json.loads(idx.read_text())["weight_map"]
            files = sorted(set(index.values()))
        else:
            files = [p.name for p in sorted(d.glob("*.safetensors"))]
            index = None
        cls = RandomWeights if random else Safetensors
        self.files = {f: cls(d / f, prefix) for f in files}
        if index is None:
            index = {k: f for f, s in self.files.items() for k in s.header}
        self.index = {k.removeprefix(prefix): f for k, f in index.items() if k.startswith(prefix)}

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def __getitem__(self, name: str) -> np.ndarray:
        return self.files[self.index[name]][name]


def qparam(w, fmt: str):
    """An iron.param of the weight w (out, in) in fmt: bf16, or iron_q4's blocks (q8, q6, q4)."""
    if fmt == "bf16":
        return iron.param(bf(w))
    import iron_q4

    return iron_q4.q_param(f32(w), fmt)


# --- the DiT -----------------------------------------------------------------------------


# c: any config with heads and head_dim (rope_angles: axes and theta too)
def head_perm(c) -> np.ndarray:
    """The out-feature order that puts Qwen-Image's RoPE pair (2i, 2i + 1) at IRON's (i, i + 64)."""
    D = c.head_dim
    within = np.concatenate([np.arange(0, D, 2), np.arange(1, D, 2)])
    return (np.arange(c.heads)[:, None] * D + within[None, :]).reshape(-1)


def rope_angles(c, pos: np.ndarray) -> np.ndarray:
    """IRON's RoPE table (rows, head_dim), cos and sin interleaved, of positions ``pos`` (rows, 3:
    frame, height, width), each axis's frequencies theta^(-2k / d) (QwenImage21Rope)."""
    parts = []
    for a, d in enumerate(c.axes):
        f = 1.0 / c.theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
        parts.append(pos[:, a : a + 1].astype(np.float64) * f)
    ang = np.concatenate(parts, axis=1)  # (rows, 64)
    out = np.empty((pos.shape[0], c.head_dim), np.float32)
    out[:, ::2] = np.cos(ang)
    out[:, 1::2] = np.sin(ang)
    return out


# --- the VAE decoder ---------------------------------------------------------------------


def dup_source(cin: int, cout: int, ft: int) -> np.ndarray:
    """DupUp3D of one frame (first_chunk): the input channel each (output channel, i, j) of the
    2 x 2 pixel block takes, (cout, 2, 2)."""
    factor = ft * 4
    rep = cout * factor // cin
    src = np.empty((cout, 2, 2), np.int64)
    for co in range(cout):
        for i in range(2):
            for j in range(2):
                k = ((co * ft + (ft - 1)) * 2 + i) * 2 + j
                src[co, i, j] = k // rep
    return src


class VaeStage(iron_anima.VaeStage):
    """iron_anima.VaeStage for Qwen-Image-2.1's decoder: 2-D convolution weights, 64 latent
    channels, the attention of one 1152-channel head as GEMMs, and up blocks with DupUp3D's
    shortcut. Ops: ("in",); ("res", path, cin, cout); ("attn", path); ("up", path, cin, cout,
    shortcut) where shortcut is ("same",) (the input's channel, nearest), ("sel", ft, patterns,
    which) (a selection of the input's channels: a matrix a distinct pattern, which pattern each
    phase 2 py + px takes) or None; ("head",)."""

    def __init__(self, ops: list, g, w, vae_cfg: dict):
        self.ops, self.g_in = ops, g
        self.params, self._index, self.states = [], {}, []
        for op in ops:
            kind = op[0]
            if kind == "in":
                C = 64
                mean, std = np.asarray(vae_cfg["latents_mean"]), np.asarray(vae_cfg["latents_std"])
                pq = f32(w["post_quant_conv.weight"])[:, :, 0, 0]  # (64, 64)
                pb = f32(w["post_quant_conv.bias"])
                self._param("in.w", pq * std[None, :])
                self._param("in.b", pq @ mean + pb)
                self._conv3("conv_in", w, "decoder.conv_in", C)
            elif kind == "res":
                _, path, cin, cout = op
                self._param(f"{path}.norm1", f32(w[f"{path}.norm1.gamma"]).reshape(-1))
                self._conv3(f"{path}.conv1", w, f"{path}.conv1", cin)
                self._param(f"{path}.norm2", f32(w[f"{path}.norm2.gamma"]).reshape(-1))
                self._conv3(f"{path}.conv2", w, f"{path}.conv2", cout)
                if cin != cout:
                    self._param(f"{path}.sc.w", f32(w[f"{path}.conv_shortcut.weight"])[:, :, 0, 0])
                    self._param(f"{path}.sc.b", f32(w[f"{path}.conv_shortcut.bias"]))
            elif kind == "attn":
                _, path = op
                self._param(f"{path}.norm", f32(w[f"{path}.norm.gamma"]).reshape(-1))
                qkv = f32(w[f"{path}.to_qkv.weight"])[:, :, 0, 0]
                qb = f32(w[f"{path}.to_qkv.bias"])
                C = qkv.shape[1]
                s = C ** -0.5  # the scores' scale, folded into the queries
                self._param(f"{path}.q.w", qkv[:C] * s)
                self._param(f"{path}.q.b", qb[:C] * s)
                for i, n in ((1, "k"), (2, "v")):
                    self._param(f"{path}.{n}.w", qkv[i * C : (i + 1) * C])
                    self._param(f"{path}.{n}.b", qb[i * C : (i + 1) * C])
                self._param(f"{path}.proj.w", f32(w[f"{path}.proj.weight"])[:, :, 0, 0])
                self._param(f"{path}.proj.b", f32(w[f"{path}.proj.bias"]))
                self._param(f"{path}.eye", np.eye(C, dtype=np.float32))
            elif kind == "up":
                _, path, cin, cout, sc = op
                wt = f32(w[f"{path}.upsampler.resample.1.weight"])  # (cout, cout, 3, 3)
                self._up_phases(f"{path}.up", wt)
                self._param(f"{path}.up.b", f32(w[f"{path}.upsampler.resample.1.bias"]))
                if sc is not None and sc[0] == "sel":
                    src = dup_source(cin, cout, sc[1])
                    for k, (i, j) in enumerate(sc[2]):
                        m = np.zeros((cout, cin), np.float32)
                        m[np.arange(cout), src[:, i, j]] = 1
                        self._param(f"{path}.sel{k}", m)
            elif kind == "head":
                self._param("head.norm", f32(w["decoder.norm_out.gamma"]).reshape(-1))
                wt = f32(w["decoder.conv_out.weight"])  # (4, 144, 3, 3)
                full = np.zeros((64,) + wt.shape[1:], np.float32)
                full[:4] = wt
                self._conv_weight("head", full.transpose(0, 2, 3, 1).reshape(64, -1))
                b = np.zeros(64, np.float32)
                b[:4] = f32(w["decoder.conv_out.bias"])
                self._param("head.b", b)
            else:
                raise ValueError(op)

    def _conv3(self, name: str, w, path: str, cin_padded: int) -> None:
        wt = f32(w[f"{path}.weight"])  # (cout, cin, 3, 3)
        co, ci = wt.shape[:2]
        full = np.zeros((co, cin_padded, 3, 3), np.float32)
        full[:, :ci] = wt
        self._conv_weight(name, full.transpose(0, 2, 3, 1).reshape(co, -1))
        self._param(f"{name}.b", f32(w[f"{path}.bias"]))

    def attn(self, x, g, path: str):
        """Single-head attention over the pixels as GEMMs (one head of 1152 does not fit MHA's
        cores): the scores q k^T (the scale folded into q), a row softmax, then P V through V's
        transpose (an identity's GEMM)."""
        from iron.operators.softmax import Softmax

        n, C, Wp = g.H * g.W, g.C, g.Wp
        dense = Copy.step(x, src=tap((g.rows, C), g.pixel(0, 0) * C, [g.H, g.W * C], [Wp * C, 1])).reshape(n, C)
        h = F.rms_norm(dense, (C,), self.w(f"{path}.norm"), 1e-12)
        q, k, v = (F.linear(h, self.w(f"{path}.{c}.w"), self.w(f"{path}.{c}.b"), precision=Precision.HIGHEST) for c in "qkv")
        s = F.linear(q, k, precision=Precision.HIGHEST)  # (n, n)
        p = Softmax.step(s)
        vt = F.linear(self.w(f"{path}.eye"), v, precision=Precision.HIGHEST)  # (C, n): V's transpose, exactly
        o = F.linear(p, vt, precision=Precision.HIGHEST)
        o = dense + F.linear(o, self.w(f"{path}.proj.w"), self.w(f"{path}.proj.b"), precision=PREC.vae)
        out = self.new_grid(g)
        Copy.step(o, out, dst=tap((g.rows, C), g.pixel(0, 0) * C, [g.H, g.W * C], [Wp * C, 1]))
        return self.view(out), g

    def up_dup(self, x, g, path: str, cout: int, xin, gin, sc):
        """up_phases with DupUp3D's shortcut of the block's input ``xin`` (grid ``gin``) added to
        each phase: its own channels (nearest) or a selection of them (a GEMM a phase pattern)."""
        from iron.operators.flm.gemm.design import Epilogue
        from iron_gridgemm import GridGEMM

        gout = iron_anima.GridShape(2 * g.H, 2 * g.W, cout)
        out = self.new_grid(gout)
        Wp, C = g.Wp, g.C
        kw = dict(b_col_maj=True, epilogue_modes=(Epilogue.NONE,), precision=PREC.vae)
        R = max(r for r in range(16, min(iron_anima.VAE_TAP_ROWS, g.H) + 1, 16) if g.H % r == 0)
        for r0 in range(0, g.H, R):
            n = R * Wp
            s_in = gin.band + r0 * Wp  # the shortcut's rows: logical row i is pixel band + r0 Wp + i
            shortcut = {}
            if sc is not None and sc[0] == "sel":
                for k in range(len(sc[2])):
                    shortcut[k] = F.linear(xin[s_in : s_in + n], self.w(f"{path}.sel{k}"), precision=PREC.vae)
            for py in range(2):
                for px in range(2):
                    oy0, ox0 = self.PHASE_TAPS[py][0][0], self.PHASE_TAPS[px][0][0]
                    s0 = g.band + (r0 + oy0) * Wp + ox0
                    if self.has(f"{path}.up.q{py}{px}"):
                        MA = n + Wp + 2
                        y = GridGEMM.step(x[s0 : s0 + MA], self.w(f"{path}.up.q{py}{px}"), M=n, MA=MA, CA=C,
                                          k_stride=Wp * C, k_tile=2 * C, **kw)
                    else:
                        y = None
                        for a in range(2):
                            t = GridGEMM.step(x[s0 + a * Wp : s0 + a * Wp + n + 2], self.w(f"{path}.up.q{py}{px}{a}"),
                                              M=n, MA=n + 2, CA=C, **kw)
                            y = t if y is None else y + t
                    y = y + self.w(f"{path}.up.b")
                    if sc is not None and sc[0] == "same":
                        y = y + xin[s_in : s_in + n]
                    elif sc is not None:
                        y = y + shortcut[sc[3][2 * py + px]]
                    Copy.step(y, out, src=tap((n, cout), cout, [R, g.W, cout], [Wp * cout, cout, 1]),
                              dst=tap((gout.rows, cout), gout.pixel(2 * r0 + py, px) * cout, [R, g.W, cout],
                                      [2 * gout.Wp * cout, 2 * cout, 1]))
            command_boundary()
        return self.view(out), gout

    def body(self, x):
        g = self.g_in
        xin, gin = x, g  # an up block's input, for its shortcut
        for op in self.ops:
            kind = op[0]
            if kind == "in":
                a, ga = self.conv1(x, g, "in", 64)
                x, g = self.conv3(a, ga, "conv_in", op_out(self.ops))
            elif kind == "res":
                x, g = self.res(x, g, op[1], op[3])
            elif kind == "attn":
                x, g = self.attn(x, g, op[1])
            elif kind == "up":
                x, g = self.up_dup(x, g, op[1], op[3], xin, gin, op[4])
            elif kind == "head":
                h = self.norm_silu(x, g, "head.norm")
                y, gy = self.conv3(h, g, "head", 64)
                n = g.H * g.W
                return Copy.step(y, src=tap((gy.rows, 64), gy.pixel(0, 0) * 64, [g.H, g.W * 64], [gy.Wp * 64, 1])).reshape(n, 64)
        self.g_out = g
        return x if self.ops[-1][0] == "res" else Copy.step(x)


def op_out(ops) -> int:
    """conv_in's channels: the next op's input."""
    for op in ops[1:]:
        if op[0] == "res":
            return op[2]
    return 1152


# --- export ------------------------------------------------------------------------------


def timestep_sinusoid(t: float, dim: int = 256) -> np.ndarray:
    """QwenImage21TemporalTimesteps of 1000 sigma: the cosines, then the sines."""
    half = dim // 2
    f = np.exp(-math.log(10000) * np.arange(half, dtype=np.float64) / half)
    a = t * f
    return np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)


def share_banks(m: dict, d: Path, blocks: Path):
    """dit_text's banks as dit_blocks': when dit_text's first bank image is byte for byte
    dit_blocks' (DitText holds DitBlocks' weights in the same order and nothing else, so IRON
    places them alike), its own images are removed and every variant names dit_blocks' files
    (``../dit_blocks/...``); m's banks are rewritten so. The variants, or None when the images
    differ (then dit_text keeps its own, 7 GB more at q8)."""
    import filecmp

    bm_path = blocks / "manifest.json"
    if not bm_path.exists():
        return None
    bm = json.loads(bm_path.read_text())
    if set(bm["banks"]) != set(m["banks"]):
        return None
    for kind, b in m["banks"].items():
        mine, theirs = b.get("image"), bm["banks"][kind].get("image")
        if (mine is None) != (theirs is None):
            return None
        if mine is not None and not filecmp.cmp(d / mine, blocks / theirs, shallow=False):
            return None
    for kind, b in m["banks"].items():
        if b.get("image") is not None:
            (d / b["image"]).unlink()
            b["image"] = f"../{blocks.name}/{bm['banks'][kind]['image']}"
    return [{k: f"../{blocks.name}/{f}" for k, f in v.items()} for v in bm.get("variants") or [
        {k: b["image"] for k, b in bm["banks"].items()}]]
