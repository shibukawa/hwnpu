#!/usr/bin/env python3
"""Qwen-Image-2.1 (Qwen/Qwen-Image-2.1-Turbo) on IRON: the graphs cmd/hwnpu-image runs on AMD's NPU.

Text to image. Qwen-Image-2.1's DiT is 32 single-stream blocks (4096 wide, 32 heads of 128,
SwiGLU 12288, no biases) over one sequence, the prompt's rows and then the image's; one
modulation (scale and gate, no shift, tanh gates) shared by every block; block-causal attention
(the text causal, the image attending to the text and itself). The text rows are modulated at
t = 0 (``causal_condition``), so their keys and values are the same at every step: they are
computed once a prompt (dit_text) and the steps run the image's rows only, attending to them
(dit_blocks), as diffusers' prefix KV cache. The prompt is read by Qwen3-VL-8B (its language
model: the last decoder layer's output, before the final norm), the latents are a Wan 2.2-style
VAE's (64 channels at a 16th of the sides, RGBA). The graphs, each exported by iron_graph.py
into a directory of its own:

  text        Qwen3-VL-8B's 36 layers, causal, over a chunk of ``rows`` prompt positions from a
              per-call ``start`` (the keys and values of the positions before it held while the
              graph is loaded), on the embedding rows (which the host looks up); then the DiT's
              txt_in (zero-centred RMS norm, linear, tanh GELU, linear) of those rows
  dit_in      the latents' projection (img_in) and the timestep's embedding into the shared
              modulation: [1 + scale1, tanh gate1, 1 + scale2, tanh gate2] and norm_out's 1 + scale
  dit_text    ``group`` blocks over a chunk of the text's rows from ``start``, causal, modulated
              by t = 0's rows: the rows after them and each block's keys and values (after the q
              and k norms and RoPE); one ELF, a bank image per group
  dit_blocks  ``group`` blocks over the image's rows, attending to [their own keys, the text's
              ``nk - tokens``]; one ELF, a bank image per group, read into one bank in turn
  dit_out     norm_out and proj_out to the 64 latent channels
  vae/s*      the decoder in stages

The image size is fixed per export (the grid and the attention's shape follow it). The host
(internal/imagegen/qwenimage) does what is not a tensor computation: the tokenizer and the
template, the embedding rows, the rows kept (after the system message's), the RoPE of the image's
rows (their frame position is the text's length), the timestep's sinusoid, the Euler steps, the
noise and the PNG.

RoPE: Qwen-Image turns pairs (2i, 2i + 1) of a head (three axes: 8 frame, 28 height, 28 width
frequencies), IRON's RoPE pairs (i, i + 64): the q and k projections' rows (and their norms'
weights) are permuted within each head (iron_flux.head_perm), q . k unchanged.

  python tools/amd/iron_qwenimage.py ~/models/Qwen-Image-2.1-Turbo OUT --size 1024

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_anima  # noqa: E402
import iron_mha  # noqa: E402
import iron_sana  # noqa: E402
from iron_anima import RandomWeights, Safetensors, bf, command_boundary, tap, variant_bank  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

# the template QwenImage21Pipeline puts the prompt in (prompt_template_t2i); the system message's
# tokens are dropped from the text's rows
SYS_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE = ("<|im_start|>system\n" + SYS_PROMPT + "<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n")


@dataclasses.dataclass
class Precisions:
    """The matmuls' precision of each part. The text encoder runs once a prompt (HIGHEST, its
    weights q6 dequantized to bf16 on the device); the DiT 8 times or more (MEDIUM, bfp16)."""

    text: Precision = Precision.HIGHEST
    dit: Precision = Precision.MEDIUM
    vae: Precision = Precision.MEDIUM


PREC = Precisions()


@dataclasses.dataclass
class Formats:
    """The weights' formats: the DiT's projections q8 (an 8-step distilled DiT is weak at 4 bits,
    new-model-checklist 5.2), the text encoder's q6 (as Sana's Gemma 2)."""

    dit: str = "q8"
    text: str = "q6"
    mha: str = "qk"  # the DiT's attention products (iron_anima.dit_attention)


FMT = Formats()


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


@dataclasses.dataclass(frozen=True)
class DitConfig:
    width: int = 1024
    height: int = 1024
    hidden: int = 4096
    heads: int = 32
    head_dim: int = 128
    mlp: int = 12288
    layers: int = 32
    latent: int = 64
    axes: tuple = (16, 56, 56)
    theta: float = 10000.0
    eps: float = 1e-6
    group: int = 4  # blocks a bank holds
    text_rows: int = 1024  # the text's rows the image attends to at most (the prompt's, after the system message)
    text_chunk: int = 256  # the text's rows dit_text computes a call
    vae_scale: int = 16

    @property
    def gh(self) -> int:
        return self.height // self.vae_scale

    @property
    def gw(self) -> int:
        return self.width // self.vae_scale

    @property
    def tokens(self) -> int:
        return self.gh * self.gw


def head_perm(c: DitConfig) -> np.ndarray:
    """The out-feature order that puts Qwen-Image's RoPE pair (2i, 2i + 1) at IRON's (i, i + 64)."""
    D = c.head_dim
    within = np.concatenate([np.arange(0, D, 2), np.arange(1, D, 2)])
    return (np.arange(c.heads)[:, None] * D + within[None, :]).reshape(-1)


def rope_angles(c: DitConfig, pos: np.ndarray) -> np.ndarray:
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


def text_positions(n: int) -> np.ndarray:
    """The text's rows' positions: p on all three axes."""
    p = np.arange(n)
    return np.stack([p, p, p], 1)


def image_positions(c: DitConfig, text_len: int) -> np.ndarray:
    """The image's rows' positions after a text of text_len rows: the frame at text_len, the
    height and width centred on zero (QwenImage21Rope)."""
    h, w = c.gh, c.gw
    hi = np.arange(-(h - h // 2), h // 2)
    wi = np.arange(-(w - w // 2), w // 2)
    return np.stack([np.full(h * w, text_len), np.repeat(hi, w), np.tile(wi, h)], 1)


class DitBlock(iron.Graph):
    """A QwenImage21TransformerBlock's weights (q and k rows permuted for IRON's RoPE)."""

    def __init__(self, c: DitConfig, w, i: int):
        E, M, D = c.hidden, c.mlp, c.head_dim
        self.c = c
        if w is None:  # shapes only
            z = lambda *s: iron.param(*s)  # noqa: E731
            self.q, self.k, self.v, self.o = z(E, E), z(E, E), z(E, E), z(E, E)
            self.qn, self.kn = z(D), z(D)
            self.gate, self.up, self.down = z(M, E), z(M, E), z(E, M)
            return
        p = f"transformer_blocks.{i}."
        perm = head_perm(c)
        g = lambda n: f32(w[p + n])  # noqa: E731
        self.q = qparam(g("attn.to_q.weight")[perm], FMT.dit)
        self.k = qparam(g("attn.to_k.weight")[perm], FMT.dit)
        self.v = qparam(g("attn.to_v.weight"), FMT.dit)
        self.o = qparam(g("attn.to_out.0.weight"), FMT.dit)
        within = perm[:D]
        self.qn = iron.param(bf(g("attn.norm_q.weight")[within]))
        self.kn = iron.param(bf(g("attn.norm_k.weight")[within]))
        self.gate = qparam(g("img_mlp.gate_layer.weight"), FMT.dit)
        self.up = qparam(g("img_mlp.proj.weight"), FMT.dit)
        self.down = qparam(g("img_mlp.out.weight"), FMT.dit)

    def qkv(self, x, m1, angles):
        """The modulated norm and the attention's q, k (normed, turned) and v of x's rows."""
        c = self.c
        E, H, D = c.hidden, c.heads, c.head_dim
        n = x.shape[0]
        P = PREC.dit
        h = F.layer_norm(x, (E,), eps=c.eps) * m1
        q = F.rms_norm(F.linear(h, self.q, precision=P).reshape(n * H, D), (D,), self.qn, c.eps)
        k = F.rms_norm(F.linear(h, self.k, precision=P).reshape(n * H, D), (D,), self.kn, c.eps)
        v = F.linear(h, self.v, precision=P)
        return F.rope(q, angles), F.rope(k, angles), v

    def rest(self, x, o, g1, m2, g2):
        """The attention's output projection, the residual, and the SwiGLU MLP's."""
        c = self.c
        P = PREC.dit
        x = x + F.linear(o, self.o, precision=P) * g1
        h = F.layer_norm(x, (c.hidden,), eps=c.eps) * m2
        f = F.silu(F.linear(h, self.gate, precision=P)) * F.linear(h, self.up, precision=P)
        return x + F.linear(f, self.down, precision=P) * g2


def mod_rows(mod, E: int):
    """The shared modulation's four rows: 1 + scale1, tanh gate1, 1 + scale2, tanh gate2."""
    return [mod[i * E : (i + 1) * E] for i in range(4)]


class DitBlocks(iron.Graph):
    """``group`` blocks over the image's rows ``x`` (tokens, hidden).

    Inputs: ``x``; ``mod`` (4 hidden), the step's modulation rows (dit_in); ``angles`` (tokens,
    head_dim), the image rows' RoPE (they follow the text's length); ``tk`` and ``tv`` (group x
    text_rows, hidden), the text's keys and values of the group's blocks (dit_text). Per call:
    ``nk``, the keys the image attends to: its tokens and the text's rows.
    """

    def __init__(self, c: DitConfig, first: int = 0, w=None):
        self.c = c
        self.blocks = [DitBlock(c, w, first + j) for j in range(c.group)]
        n = c.tokens + c.text_rows
        self.kbuf = iron.state((n, c.heads, c.head_dim))
        self.vbuf = iron.state((n, c.heads, c.head_dim))

    def body(self, x, mod, angles, tk, tv, *, nk: Scratchpad[np.int32]):
        c = self.c
        E, H, D, N, T = c.hidden, c.heads, c.head_dim, c.tokens, c.text_rows
        m1, g1, m2, g2 = mod_rows(mod, E)
        for j, b in enumerate(self.blocks):
            q, k, v = b.qkv(x, m1, angles)
            self.kbuf[0:N] = k.reshape(N, H, D)
            self.vbuf[0:N] = v.reshape(N, H, D)
            self.kbuf[N : N + T] = tk[j * T : (j + 1) * T].reshape(T, H, D)
            self.vbuf[N : N + T] = tv[j * T : (j + 1) * T].reshape(T, H, D)
            o = iron_anima.dit_attention(q.reshape(N, H, D), self.kbuf[:nk], self.vbuf[:nk])
            x = b.rest(x, o.reshape(N, E), g1, m2, g2)
            command_boundary()  # a block at 1024 x 1024 is about half a second: under amdxdna's 2 s
        return x


class DitText(iron.Graph):
    """``group`` blocks over a chunk of the text's rows ``x`` (text_chunk, hidden) from the
    per-call ``start``, causal over the rows before them (held while the graph is loaded),
    modulated by ``mod`` (t = 0's rows), turned by ``angles`` (text_chunk, head_dim: the chunk's
    RoPE, which the host gives). Outputs: the rows after the blocks, then each block's keys and
    values of the chunk (text_chunk, hidden).

    Its weights are DitBlocks' in the same order and nothing else (the RoPE is an input, not a
    weight), so IRON places them alike and dit_text's banks are dit_blocks' files: the 7 GB of
    q8 weights are stored once (export_dit checks the images are the same before sharing)."""

    def __init__(self, c: DitConfig, first: int = 0, w=None):
        self.c = c
        self.blocks = [DitBlock(c, w, first + j) for j in range(c.group)]
        T, H, D = c.text_rows, c.heads, c.head_dim
        self.caches = [(iron.state((T, H, D)), iron.state((T, H, D))) for _ in range(c.group)]

    def body(self, x, mod, angles, *, start: Scratchpad[np.int32] = np.int32(0)):
        c = self.c
        E, H, D, R = c.hidden, c.heads, c.head_dim, c.text_chunk
        m1, g1, m2, g2 = mod_rows(mod, E)
        kvs = []
        for j, b in enumerate(self.blocks):
            q, k, v = b.qkv(x, m1, angles)
            kc, vc = self.caches[j]
            kc[start : start + R] = k.reshape(R, H, D)
            vc[start : start + R] = v.reshape(R, H, D)
            o = MHA.step(q.reshape(R, H, D), kc[: start + R], vc[: start + R], heads_interleaved=True,
                         kv_interleaved=True, causal=True, d=D, num_pipelines=max(1, min(8, R // 64)))
            kvs += [Copy.step(k.reshape(R, E)), Copy.step(v)]
            x = b.rest(x, o.reshape(R, E), g1, m2, g2)
            command_boundary()
        return [x] + kvs


class DitIn(iron.Graph):
    """img_in of the latents (tokens, 64) and the timestep's modulation: ``tproj`` (256) is
    the sinusoid of 1000 sigma; outputs the rows, the shared modulation (4 hidden: 1 + scale1,
    tanh gate1, 1 + scale2, tanh gate2) and norm_out's 1 + scale (hidden)."""

    def __init__(self, c: DitConfig, w=None):
        self.c = c
        E = c.hidden
        if w is None:
            self.img_in, self.t1, self.t2 = iron.param(E, c.latent), iron.param(E, 256), iron.param(E, E)
            self.mod, self.out = iron.param(4 * E, E), iron.param(E, E)
        else:
            self.img_in = iron.param(bf(w["img_in.weight"]))
            self.t1 = iron.param(bf(w["time_text_embed.timestep_embedder.linear_1.weight"]))
            self.t2 = iron.param(bf(w["time_text_embed.timestep_embedder.linear_2.weight"]))
            self.mod = iron.param(bf(w["modulation.1.weight"]))
            self.out = iron.param(bf(w["norm_out.linear.weight"]))
        one = np.ones(E, np.float32)
        self.ones = iron.param(bf(np.concatenate([one, 0 * one, one, 0 * one])))
        self.one = iron.param(bf(one))
        self.mods = iron.state((4 * E,))

    def body(self, patches, tproj):
        c = self.c
        E = c.hidden
        P = Precision.HIGHEST
        h = F.linear(patches, self.img_in, precision=PREC.dit)
        t = F.linear(F.silu(F.linear(tproj, self.t1, precision=P)), self.t2, precision=P)
        a = F.silu(t)
        m = F.linear(a, self.mod, precision=P)  # [scale1, gate1, scale2, gate2]
        from iron.operators.tanh import Tanh

        m = m + self.ones
        g1 = Tanh.step(m[E : 2 * E])
        g2 = Tanh.step(m[3 * E : 4 * E])
        mod = self.mods
        mod[0:E] = m[0:E]
        mod[E : 2 * E] = g1
        mod[2 * E : 3 * E] = m[2 * E : 3 * E]
        mod[3 * E : 4 * E] = g2
        s = F.linear(a, self.out, precision=P) + self.one
        return [h, Copy.step(mod[0 : 4 * E]), s]


class DitOut(iron.Graph):
    """norm_out (layer norm times 1 + scale) and proj_out to the latent channels."""

    def __init__(self, c: DitConfig, w=None):
        self.c = c
        self.proj = iron.param(c.latent, c.hidden) if w is None else iron.param(bf(w["proj_out.weight"]))

    def body(self, x, scale):
        c = self.c
        h = F.layer_norm(x, (c.hidden,), eps=c.eps) * scale
        return F.linear(h, self.proj, precision=PREC.dit)


# --- the text encoder (Qwen3-VL-8B's language model) and txt_in --------------------------


@dataclasses.dataclass(frozen=True)
class TextConfig:
    rows: int = 64  # the positions a call computes
    cache: int = 1088  # the positions the keys and values hold: the template and 1024 of the prompt's
    hidden: int = 4096
    layers: int = 36
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    mlp: int = 12288
    theta: float = 5e6
    eps: float = 1e-6
    vocab: int = 151936
    split: int = 2  # layers a command runs
    group: int = 4  # layers a bank holds (36 / 4 = 9 groups, read in turn)


LM = "model.language_model."


class Qwen3Layer(iron.Graph):
    """A Qwen3 decoder layer (Qwen3-VL's language model) over a chunk of rows from ``start``,
    causal over the cache: RMS norms, GQA with q and k norms and RoPE, SwiGLU."""

    def __init__(self, c: TextConfig, w, i: int):
        E, H, G, D, M = c.hidden, c.heads, c.kv_heads, c.head_dim, c.mlp
        self.c = c
        p = f"{LM}layers.{i}."
        if w is None:
            lin = lambda n, *s: iron.param(*s)  # noqa: E731
            vec = lambda n, s: iron.param(s)  # noqa: E731
        else:
            lin = lambda n, *s: iron_sana.text_param(f32(w[p + n]))  # noqa: E731
            vec = lambda n, s: iron.param(bf(w[p + n]))  # noqa: E731
        self.in_norm = vec("input_layernorm.weight", E)
        self.post_norm = vec("post_attention_layernorm.weight", E)
        self.q = lin("self_attn.q_proj.weight", H * D, E)
        self.k = lin("self_attn.k_proj.weight", G * D, E)
        self.v = lin("self_attn.v_proj.weight", G * D, E)
        self.o = lin("self_attn.o_proj.weight", E, H * D)
        self.qn = vec("self_attn.q_norm.weight", D)
        self.kn = vec("self_attn.k_norm.weight", D)
        self.gate = lin("mlp.gate_proj.weight", M, E)
        self.up = lin("mlp.up_proj.weight", M, E)
        self.down = lin("mlp.down_proj.weight", E, M)

    def body(self, x, angles, cache, start):
        c = self.c
        L, H, G, D = x.shape[0], c.heads, c.kv_heads, c.head_dim
        tl = iron_sana.text_linear
        h = F.rms_norm(x, (c.hidden,), self.in_norm, c.eps)
        q = F.rms_norm(tl(h, self.q).reshape(L * H, D), (D,), self.qn, c.eps)
        k = F.rms_norm(tl(h, self.k).reshape(L * G, D), (D,), self.kn, c.eps)
        v = tl(h, self.v)
        q, k = F.rope(q, angles), F.rope(k, angles)
        kc, vc = cache
        kc[start : start + L] = k.reshape(L, G, D)
        vc[start : start + L] = v.reshape(L, G, D)
        o = MHA.step(q.reshape(L, H, D), kc[: start + L], vc[: start + L], heads_interleaved=True,
                     kv_interleaved=True, causal=True, d=D, num_pipelines=max(1, min(8, L // 64)))
        x = x + tl(o.reshape(L, H * D), self.o)
        h = F.rms_norm(x, (c.hidden,), self.post_norm, c.eps)
        g = F.silu(tl(h, self.gate)) * tl(h, self.up)
        return x + tl(g, self.down)


class Text(iron.Graph):
    """``group`` layers of Qwen3-VL-8B's language model (from ``first``) over a chunk of ``rows``
    rows ``x`` from the per-call ``start``, causal over the rows before them (held while the graph
    is loaded), turned by ``angles`` (rows, head_dim: the chunk's RoPE, which the host gives).
    One ELF, a bank image per group of layers, read into one bank in turn (SwapVariants): the
    text encoder runs once a prompt, so each group's weights are read once, and the 5.6 GB of q6
    weights are never pinned together. The last layer's output, before the final norm, is what
    the DiT reads (the pipeline's hook): TextOut takes it."""

    def __init__(self, c: TextConfig, w=None, first: int = 0):
        self.c = c
        self.layers = [Qwen3Layer(c, w, first + i) for i in range(c.group)]
        G, D = c.kv_heads, c.head_dim
        self.caches = [(iron.state((c.cache, G, D)), iron.state((c.cache, G, D))) for _ in range(c.group)]

    def body(self, x, angles, *, start: Scratchpad[np.int32] = np.int32(0)):
        c = self.c
        for i, layer in enumerate(self.layers):
            if i and i % c.split == 0:
                command_boundary()
            x = layer(x, angles, self.caches[i], start)
        return x


class TextOut(iron.Graph):
    """The DiT's txt_in (zero-centred RMS norm, linear, tanh GELU, linear) of the text rows."""

    def __init__(self, c: TextConfig, dit=None):
        self.c = c
        E = c.hidden
        if dit is None:
            self.tnorm, self.tin, self.tout = iron.param(E), iron.param(E, E), iron.param(E, E)
        else:  # the zero-centred norm's weight + 1 (float32 in diffusers)
            self.tnorm = iron.param(bf(f32(dit["txt_in.text_norm.weight"]) + 1))
            self.tin = iron.param(bf(dit["txt_in.in_layer.weight"]))
            self.tout = iron.param(bf(dit["txt_in.out_layer.weight"]))

    def body(self, x):
        c = self.c
        P = Precision.HIGHEST
        h = F.rms_norm(x, (c.hidden,), self.tnorm, c.eps)
        h = F.gelu(F.linear(h, self.tin, precision=P), approximate="tanh")
        return F.linear(h, self.tout, precision=P)


# --- the VAE decoder ---------------------------------------------------------------------
#
# AutoencoderKLQwenImage21's decoder for one frame: conv_in (64 -> 1152), the mid block (a
# residual block, single-head attention, a residual block), five up blocks (three residual
# blocks each, the first four ending in a nearest 2x and a 3 x 3 convolution, plus a shortcut of
# the block's input copied into the pixels, DupUp3D), norm_out, SiLU, conv_out (to RGBA). Its
# convolutions are 2-D (the causal 3-D one specialized to an image). The stages are
# iron_anima.VaeStage's grids, GridGEMM and phase upsampling.


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


def shortcut_of(cin: int, cout: int, ft: int):
    """An up block's DupUp3D shortcut as VaeStage takes it."""
    src = dup_source(cin, cout, ft)
    if cin == cout and all((src[:, i, j] == np.arange(cout)).all() for i in range(2) for j in range(2)):
        return ("same",)
    pats, which = [], {}
    for i in range(2):
        for j in range(2):
            for k, (a, b) in enumerate(pats):
                if (src[:, a, b] == src[:, i, j]).all():
                    which[(i, j)] = k
                    break
            else:
                which[(i, j)] = len(pats)
                pats.append((i, j))
    # ("sel", ft, the distinct patterns [i, j], each phase's (2 py + px) pattern)
    return ("sel", ft, [list(p) for p in pats], [which[(i, j)] for i in range(2) for j in range(2)])


VAE_MERGE = [True]  # 16.157: the last block's same-shaped stages merged, 0.5 s less at 1024², the same pixels


def vae_stages(cfg: dict, width: int, height: int) -> list:
    """The decoder in stages (each an up block, the last two split), with each one's input grid."""
    GS = iron_anima.GridShape
    base, mult = cfg["decoder_base_dim"], cfg["dim_mult"]
    dims = [base * u for u in [mult[-1]] + mult[::-1]]  # 1152 1152 1152 576 288 144
    tup = list(reversed(cfg["temperal_downsample"]))
    h, w = height // 16, width // 16
    d = "decoder"
    s = [([("in",), ("res", f"{d}.mid_block.resnets.0", dims[0], dims[0]), ("attn", f"{d}.mid_block.attentions.0"),
           ("res", f"{d}.mid_block.resnets.1", dims[0], dims[0])], GS(h, w, 64))]
    for i in range(len(dims) - 1):
        cin, cout = dims[i], dims[i + 1]
        p = f"{d}.up_blocks.{i}"
        res = [("res", f"{p}.resnets.0", cin, cout)] + [("res", f"{p}.resnets.{j}", cout, cout) for j in (1, 2)]
        if i < len(dims) - 2:
            ft = 2 if tup[i] else 1
            s.append((res + [("up", p, cin, cout, shortcut_of(cin, cout, ft))], GS(h, w, cin)))
            h, w = 2 * h, 2 * w
        else:  # the last block, at the full size: a stage a residual block, then the head
            s.append(([res[0]], GS(h, w, cin)))
            if VAE_MERGE[0]:  # the two same-shaped residual blocks and the head in one stage
                s.append(([res[1], res[2], ("head",)], GS(h, w, cout)))
            else:
                s.append(([res[1]], GS(h, w, cout)))
                s.append(([res[2], ("head",)], GS(h, w, cout)))
    return s


def export_vae(root: Path, out: Path, width: int, height: int, check: bool, only: list | None = None) -> None:
    cls = RandomWeights if ARGS.random else Safetensors
    w = cls(root / "vae" / "diffusion_pytorch_model.safetensors")
    cfg = json.loads((root / "vae" / "config.json").read_text())
    stages = vae_stages(cfg, width, height)
    d = out / "vae"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, (ops, g) in enumerate(stages):
        if only and i not in only:
            continue
        t0 = time.time()
        st = VaeStage(ops, g, w, cfg)
        cg = st.compile(x=(g.rows, g.C))
        go = st.g_out if ops[-1][0] != "head" else None
        meta = dict(stage=i, ops=json.loads(json.dumps(ops)), grid_in=dataclasses.asdict(g),
                    grid_out=None if go is None else dataclasses.asdict(go), pad=iron_anima.VAE_PAD,
                    width=width, height=height, latents_mean=cfg["latents_mean"], latents_std=cfg["latents_std"])
        m = export_graph(cg, d / f"s{i}", f"vae_s{i}", meta)
        print(f"vae s{i}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            x = np.zeros((g.rows, g.C), np.float32)
            for y in range(g.H):
                x[g.pixel(y, 0) : g.pixel(y, 0) + g.W] = rng.standard_normal((g.W, g.C))
            write_check(st, m, d / f"s{i}", dict(x=bf(x)))
    (d / "manifest.json").write_text(json.dumps(dict(stages=len(stages), width=width, height=height, pad=iron_anima.VAE_PAD,
                                                     channels=4, latent=64, latents_mean=cfg["latents_mean"],
                                                     latents_std=cfg["latents_std"]), indent=1) + "\n")


# --- export ------------------------------------------------------------------------------


def timestep_sinusoid(t: float, dim: int = 256) -> np.ndarray:
    """QwenImage21TemporalTimesteps of 1000 sigma: the cosines, then the sines."""
    half = dim // 2
    f = np.exp(-math.log(10000) * np.arange(half, dtype=np.float64) / half)
    a = t * f
    return np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)


def dit_checkpoint(root: Path):
    return Checkpoint(root / "transformer", random=ARGS.random)


def export_text(root: Path, out: Path, check: bool) -> None:
    c = TextConfig()
    w = Checkpoint(root / "text_encoder", random=ARGS.random)
    dit = dit_checkpoint(root)
    t0 = time.time()
    iron_sana.TEXT_FMT["fmt"] = FMT.text
    iron_sana.PREC.text = PREC.text
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(root / "processor" / "tokenizer.json"))
    sys_ids = tok.encode("<|im_start|>system\n" + SYS_PROMPT + "<|im_end|>\n", add_special_tokens=False).ids
    groups = c.layers // c.group
    meta = dict(config=dataclasses.asdict(c), weights=FMT.text, precision=PREC.text.value,
                template=TEMPLATE, drop=len(sys_ids), groups=groups)
    g = Text(c, w, 0)
    cg = g.compile(x=(c.rows, c.hidden), angles=(c.rows, c.head_dim))
    d = out / "text"
    m = export_graph(cg, d, "text", meta)
    print(f"text: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    variants = [{k: b["image"] for k, b in m["banks"].items()}]
    for i in range(1, groups):
        variants.append(variant_bank(cg, Text(c, w, i * c.group), d, f"bank_g{i}"))
        print(f"text: bank of layers {i * c.group}..{(i + 1) * c.group - 1}", flush=True)
    m["variants"] = variants
    (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    to = TextOut(c, dit)
    mo = export_graph(to.compile(x=(c.rows, c.hidden)), out / "text_out", "text_out", meta)
    print(f"text_out: {mo['graph']['plan']}", flush=True)
    import shutil

    shutil.copyfile(root / "processor" / "tokenizer.json", d / "tokenizer.json")
    with open(d / "embed.bin", "wb") as f:
        e = w[LM + "embed_tokens.weight"]
        for s in range(0, e.shape[0], 16384):
            f.write(bf(e[s : s + 16384]).tobytes())
    if check:
        rng = np.random.default_rng(0)
        x = bf(f32(w[LM + "embed_tokens.weight"][rng.integers(0, 150000, c.rows)]))
        ang = bf(iron_anima.half_rope_table(c.rows, c.head_dim, c.theta))
        write_check(g, m, d, dict(x=x, angles=ang), dict(start=0))


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
            b["image"] = f"../dit_blocks/{bm['banks'][kind]['image']}"
    return [{k: f"../dit_blocks/{f}" for k, f in v.items()} for v in bm.get("variants") or [
        {k: b["image"] for k, b in bm["banks"].items()}]]


def export_dit(root: Path, out: Path, c: DitConfig, check: bool) -> None:
    w = dit_checkpoint(root)
    rng = np.random.default_rng(0)
    meta = dict(config=dataclasses.asdict(c), weights=FMT.dit, mha=FMT.mha)
    mi = json.loads((root / "model_index.json").read_text())
    sched = json.loads((root / "scheduler" / "scheduler_config.json").read_text())
    E = c.hidden
    for name, g, shapes, inputs in [
        ("dit_in", DitIn(c, w), dict(patches=(c.tokens, c.latent), tproj=(256,)),
         lambda: dict(patches=bf(rng.standard_normal((c.tokens, c.latent))), tproj=bf(timestep_sinusoid(700.0)))),
        ("dit_out", DitOut(c, w), dict(x=(c.tokens, E), scale=(E,)),
         lambda: dict(x=bf(rng.standard_normal((c.tokens, E))), scale=bf(1 + 0.1 * rng.standard_normal(E)))),
    ]:
        t0 = time.time()
        cg = g.compile(**shapes)
        m = export_graph(cg, out / name, name, meta)
        print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            write_check(g, m, out / name, inputs())
    groups = c.layers // c.group
    mod = np.concatenate([1 + 0.1 * rng.standard_normal(E), np.tanh(0.3 * rng.standard_normal(E)),
                          1 + 0.1 * rng.standard_normal(E), np.tanh(0.3 * rng.standard_normal(E))])
    for name, cls, shapes, inputs, values in [
        ("dit_blocks", DitBlocks,
         dict(x=(c.tokens, E), mod=(4 * E,), angles=(c.tokens, c.head_dim), tk=(c.group * c.text_rows, E),
              tv=(c.group * c.text_rows, E)),
         lambda: dict(x=bf(rng.standard_normal((c.tokens, E))), mod=bf(mod),
                      angles=bf(rope_angles(c, image_positions(c, 100))),
                      tk=bf(rng.standard_normal((c.group * c.text_rows, E))),
                      tv=bf(rng.standard_normal((c.group * c.text_rows, E)))), dict(nk=c.tokens + 100)),
        ("dit_text", DitText, dict(x=(c.text_chunk, E), mod=(4 * E,), angles=(c.text_chunk, c.head_dim)),
         lambda: dict(x=bf(rng.standard_normal((c.text_chunk, E))), mod=bf(mod),
                      angles=bf(rope_angles(c, text_positions(c.text_chunk)))), dict(start=0)),
    ]:
        if name not in ARGS.dit_parts:
            continue
        t0 = time.time()
        g = cls(c, 0, w)
        cg = g.compile(**shapes)
        d = out / name
        m = export_graph(cg, d, name, dict(meta, groups=groups))
        print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        variants = [{k: b["image"] for k, b in m["banks"].items()}]
        shared = name == "dit_text" and share_banks(m, d, out / "dit_blocks")
        if shared:
            variants = shared
            print(f"{name}: the banks are dit_blocks' (the same weights in the same places): stored once", flush=True)
        for i in range(1, 1 if ARGS.one_group or shared else groups):
            variants.append(variant_bank(cg, cls(c, i * c.group, w), d, f"bank_g{i}"))
            print(f"{name}: bank of blocks {i * c.group}..{(i + 1) * c.group - 1}", flush=True)
        m["variants"] = variants
        (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
        if check:
            write_check(g, m, d, inputs(), values)
    (out / "model.json").write_text(json.dumps(dict(
        name=root.name, sample_sigmas=mi.get("sample_sigmas"), scheduler=sched,
        width=c.width, height=c.height), indent=1) + "\n")


ARGS = None


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the diffusers repository (Qwen-Image-2.1-Turbo)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--text-rows", type=int, default=1024, help="the prompt's rows the DiT takes at most")
    ap.add_argument("--only", nargs="*", default=["text", "dit", "vae"], help="parts to export: text, dit, vae")
    ap.add_argument("--dit-parts", nargs="*", default=["dit_text", "dit_blocks"])
    ap.add_argument("--stages", type=int, nargs="*", help="the VAE's stages to export (default all)")
    ap.add_argument("--no-vae-merge", dest="vae_merge", action="store_false", help="the last block's two same-shaped residual blocks in stages of their own")
    ap.add_argument("--dit", default=FMT.dit, choices=("bf16", "q8", "q6", "q4"), help="the DiT blocks' projections")
    ap.add_argument("--text", default=FMT.text, choices=("bf16", "q8", "q6", "q4"), help="the text encoder's projections")
    ap.add_argument("--dit-mha", default=FMT.mha, choices=("qk", "qkpv", "bf16"))
    ap.add_argument("--one-group", action="store_true", help="the first group's bank only (a test)")
    ap.add_argument("--random", action="store_true", help="weights drawn at random in the checkpoint's shapes")
    ap.add_argument("--check", action="store_true", help="write each graph's inputs and reference outputs")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    FMT.dit, FMT.text, FMT.mha = a.dit, a.text, a.dit_mha
    VAE_MERGE[0] = a.vae_merge
    iron_anima.DIT_FMT.mha = a.dit_mha
    iron_anima.PREC.vae = PREC.vae
    import iron_q4

    # q4nx's prompt path through bf16 (rounded to nearest even): the DiT's activations have a
    # mean (iron_q4.q_linear)
    iron_q4.PREFILL["q4"] = "bf16"
    iron_q4.install_linear()
    bind_device(a.device, a.cols)
    c = DitConfig(width=a.width or a.size, height=a.height or a.size, group=a.group, text_rows=a.text_rows)
    a.out.mkdir(parents=True, exist_ok=True)
    if "text" in a.only:
        export_text(a.root, a.out, a.check)
    if "dit" in a.only:
        export_dit(a.root, a.out, c, a.check)
    if "vae" in a.only:
        export_vae(a.root, a.out, c.width, c.height, a.check, a.stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
