#!/usr/bin/env python3
"""FLUX.2 [klein] 4B (black-forest-labs/FLUX.2-klein-4B, Apache-2.0) on IRON: the DiT's graphs
for AMD's NPU.

The DiT is FLUX.1's shape (double-stream blocks: the text's and the image's rows with weights of
their own, attending together; single-stream blocks: the rows together, attention and an MLP
side by side, one projection out; 3072 wide, 24 heads of 128) with 5 double and 20 single blocks,
no biases, SwiGLU MLPs (three times the width), and one modulation for all double blocks' image
rows, one for their text rows and one for the single blocks (a linear of the timestep's
embedding each). The text is Qwen3-4B's 512 rows (hidden states of three layers side by side);
the DiT attends to the pads too. The latents are 2 x 2 patches of the FLUX.2 VAE's 32 channels.

Editing: the references' latents are rows of the image's stream after the picture's, with the
picture's modulation, told apart by the first RoPE axis (10, 20, ...). ``--refs`` names the
references the blocks are compiled for: N of the picture's size, or NdD of its sides over D (a
reference of half the sides is a quarter of the rows: an edit's step at 1024 x 1024 has 5632
rows in place of 8704). Only the picture's rows are read at the end. The blocks' weights are
placed alike whatever the rows, so the graphs of each row count share one set of banks.

  mods        the timestep's embedding (``tproj``, the sinusoid of 1000 sigma) to the modulation
              rows: the doubles' image (6 hidden: shift, scale, gate of the attention and of the
              MLP), their text (6), the singles' (3) and norm_out's scale and shift (2). Once
              a step, before the blocks (the host may run every step's first and unload it)
  dit_in      x_embedder of the picture's and the references' rows
  ctx         context_embedder of the text's rows, a third of its columns a hidden-state layer
  doubles     ``double_group`` double blocks over [image rows, text rows] (one ELF, a bank a
              group); ``angles`` (rows, 128), the rows' RoPE, is an input
  singles     ``single_group`` single blocks over the same rows
  dit_out     norm_out and proj_out of the picture's rows
  vae/s*      the VAE's decoder in stages; vae_enc/s* its encoder (a reference image), and
              vae_enc_d2 the encoder of a reference of half the sides. Their GroupNorms are two
              operators and a step of the host between two commands (iron_groupnorm,
              ``--host-norm``): hwnpu's anima.VAE reads the stages' ``norms``

RoPE: four axes of 32 (the reference index, row, column, the text's position), theta 2000; FLUX
turns pairs (2i, 2i + 1) of a head, so the q and k rows are permuted within each head to IRON's
pairs (i, i + 64) (iron_qwenimage.head_perm).

  python tools/amd/iron_flux2klein.py ~/models/FLUX.2-klein-4B OUT --size 1024
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

import iron
import iron.functional as F
from iron import Precision, Scratchpad
from iron.operators.copy import Copy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_anima  # noqa: E402
import iron_headnorm  # noqa: E402
import iron_qwenimage as qi  # noqa: E402
import iron_sana  # noqa: E402
import iron_silu  # noqa: E402
from iron_anima import bf, command_boundary, variant_bank  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402


@dataclasses.dataclass(frozen=True)
class KleinConfig:
    width: int = 1024
    height: int = 1024
    refs: int = 0  # references whose rows follow the picture's
    ref_div: int = 1  # a reference's sides are the picture's over this (2: a quarter of the rows)
    text_rows: int = 512  # Qwen3's rows (the prompt padded to max_sequence_length)
    hidden: int = 3072
    heads: int = 24
    head_dim: int = 128
    mlp: int = 9216
    latent: int = 128  # a 2 x 2 patch of the VAE's 32 channels
    text: int = 2560  # a hidden-state layer's width
    text_layers: tuple = (9, 18, 27)  # the hidden states the DiT reads, side by side
    axes: tuple = (32, 32, 32, 32)
    theta: float = 2000.0
    eps: float = 1e-6
    doubles: int = 5
    singles: int = 20
    double_group: int = 1
    single_group: int = 2

    @property
    def gh(self) -> int:
        return self.height // 16

    @property
    def gw(self) -> int:
        return self.width // 16

    @property
    def tokens(self) -> int:
        """The picture's rows."""
        return self.gh * self.gw

    @property
    def ref_tokens(self) -> int:
        """A reference's rows."""
        return (self.gh // self.ref_div) * (self.gw // self.ref_div)

    @property
    def image_rows(self) -> int:
        """The image stream's rows: the picture's and the references'."""
        return self.tokens + self.refs * self.ref_tokens

    @property
    def rows(self) -> int:
        return self.image_rows + self.text_rows


@dataclasses.dataclass
class Formats:
    # the blocks' projections: bfp16 packed on the host (iron_q4.bfp16_pack, design.md 16.200)
    proj: str = "bfp16"
    mha: str = "qkpv"  # the attention's products through bfp16: QK^T and P V (1024²: the blocks -7 %, an edit -9 %; the latents as near the reference)


FMT = Formats()
PREC = Precision.MEDIUM


def positions(c: KleinConfig) -> np.ndarray:
    """The rows' position ids (rows, 4): the picture's (0, row, column, 0), each reference's
    (10 (i + 1), row, column, 0), then the text's (0, 0, 0, position)."""
    pos = np.zeros((c.rows, 4))
    at = 0
    for i in range(1 + c.refs):
        gh, gw = (c.gh, c.gw) if i == 0 else (c.gh // c.ref_div, c.gw // c.ref_div)
        n = gh * gw
        pos[at : at + n, 0] = 10 * i
        pos[at : at + n, 1] = np.repeat(np.arange(gh), gw)
        pos[at : at + n, 2] = np.tile(np.arange(gw), gh)
        at += n
    pos[c.image_rows :, 3] = np.arange(c.text_rows)
    return pos


def angles(c: KleinConfig) -> np.ndarray:
    return qi.rope_angles(c, positions(c))


def lin(w):
    return qi.qparam(np.asarray(w, np.float32), FMT.proj)


def vec(w):
    return iron.param(bf(np.asarray(w, np.float32)))


class _Stream:
    """A stream's projections: q, k (their rows permuted for IRON's RoPE pairs), v, the q and k
    norms' weights over the heads, and whatever else ``parts`` names (name -> weight)."""

    def __init__(self, c: KleinConfig, q, k, v, qn, kn, **parts):
        P = qi.head_perm(c)
        within = P[: c.head_dim]
        self.q, self.k, self.v = lin(q[P]), lin(k[P]), lin(v)
        self.qn = iron_headnorm.head_weight(np.asarray(qn, np.float32)[within], c.heads)
        self.kn = iron_headnorm.head_weight(np.asarray(kn, np.float32)[within], c.heads)
        for name, w in parts.items():
            setattr(self, name, lin(w))


def double_stream(c: KleinConfig, w, p: str, text: bool) -> _Stream:
    f = lambda n: np.asarray(w[p + n], np.float32)  # noqa: E731
    M = c.mlp
    if text:
        names = ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.norm_added_q", "attn.norm_added_k",
                 "attn.to_add_out", "ff_context")
    else:
        names = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.norm_q", "attn.norm_k", "attn.to_out.0", "ff")
    q, k, v, qn, kn, o, ff = names
    fin = f(ff + ".linear_in.weight")  # (2 mlp, hidden): the gate's rows, then the value's
    return _Stream(c, f(q + ".weight"), f(k + ".weight"), f(v + ".weight"), f(qn + ".weight"), f(kn + ".weight"),
                   o=f(o + ".weight"), gate=fin[:M], up=fin[M:], down=f(ff + ".linear_out.weight"))


def single_stream(c: KleinConfig, w, p: str) -> _Stream:
    E, M = c.hidden, c.mlp
    f = lambda n: np.asarray(w[p + n], np.float32)  # noqa: E731
    a = f("attn.to_qkv_mlp_proj.weight")  # (3 hidden + 2 mlp, hidden): q, k, v, the MLP's gate and value
    o = f("attn.to_out.weight")  # (hidden, hidden + mlp) over [attention, MLP]
    return _Stream(c, a[:E], a[E : 2 * E], a[2 * E : 3 * E], f("attn.norm_q.weight"), f("attn.norm_k.weight"),
                   gate=a[3 * E : 3 * E + M], up=a[3 * E + M :], oa=o[:, :E], om=o[:, E:])


def qkv(s, h, c: KleinConfig, ang):
    """A stream's q and k (normed, turned) and v of h's rows."""
    n, H, D = h.shape[0], c.heads, c.head_dim
    q = iron_headnorm.rms_norm(F.linear(h, s.q, precision=PREC), s.qn, c.eps, D).reshape(n * H, D)
    k = iron_headnorm.rms_norm(F.linear(h, s.k, precision=PREC), s.kn, c.eps, D).reshape(n * H, D)
    v = F.linear(h, s.v, precision=PREC)
    return F.rope(q, ang), F.rope(k, ang), v


def swiglu(s, h):
    return F.silu(F.linear(h, s.gate, precision=PREC)) * F.linear(h, s.up, precision=PREC)


class _Blocks(iron.Graph):
    """The streams of a group of blocks as attributes b<j>_<tag>_<name> (iron.Graph finds its
    weights among its attributes)."""

    def put(self, j: int, tag: str, st: _Stream) -> None:
        for k, v in vars(st).items():
            setattr(self, f"b{j}_{tag}_{k}", v)
        self.names = tuple(vars(st))

    def s(self, j: int, tag: str):
        class S:
            pass

        o = S()
        for k in self.names:
            setattr(o, k, getattr(self, f"b{j}_{tag}_{k}"))
        return o


class Doubles(_Blocks):
    """``double_group`` Flux2TransformerBlocks over ``x`` (rows: the image stream's, then the
    text's). Inputs: ``x``; ``mods`` (12 hidden): the image's modulation rows (shift, scale, gate
    of the attention, then of the MLP), then the text's; ``angles`` (rows, head_dim)."""

    def __init__(self, c: KleinConfig, first: int = 0, w=None):
        self.c = c
        self.n = c.double_group
        for j in range(self.n):
            p = f"transformer_blocks.{first + j}."
            self.put(j, "t", double_stream(c, w, p, True))
            self.put(j, "i", double_stream(c, w, p, False))
        self.one = vec(np.ones(c.hidden))
        n, H, D = c.rows, c.heads, c.head_dim
        self.qs, self.ks, self.vs = iron.state((n * H, D)), iron.state((n, H, D)), iron.state((n, H, D))
        self.out = iron.state((n, c.hidden))

    def body(self, x, mods, angles, *, nk: Scratchpad[np.int32]):
        c = self.c
        E, H, D, N, n = c.hidden, c.heads, c.head_dim, c.image_rows, c.rows
        parts = {"i": (0, N), "t": (N, n)}
        m = {tag: [mods[(base + r) * E : (base + r + 1) * E] for r in range(6)] for tag, base in (("i", 0), ("t", 6))}
        for j in range(self.n):
            for tag, (a, b) in parts.items():
                shift, scale = m[tag][0], m[tag][1]
                h = F.layer_norm(x[a:b], (E,), weight=scale + self.one, eps=c.eps) + shift
                q, k, v = qkv(self.s(j, tag), h, c, angles[a:b])
                self.qs[a * H : b * H] = q
                self.ks[a:b] = k.reshape(b - a, H, D)
                self.vs[a:b] = v.reshape(b - a, H, D)
            o = iron_anima.dit_attention(self.qs.reshape(n, H, D), self.ks[:nk], self.vs[:nk]).reshape(n, E)
            for tag, (a, b) in parts.items():
                s = self.s(j, tag)
                _, _, g1, shift2, scale2, g2 = m[tag]
                y = x[a:b] + F.linear(o[a:b], s.o, precision=PREC) * g1
                h = F.layer_norm(y, (E,), weight=scale2 + self.one, eps=c.eps) + shift2
                self.out[a:b] = y + F.linear(swiglu(s, h), s.down, precision=PREC) * g2
            x = Copy.step(self.out[0:n])
            command_boundary()
        return x


class Singles(_Blocks):
    """``single_group`` Flux2SingleTransformerBlocks over ``x``; ``mods`` (3 hidden): shift,
    scale and gate; ``angles`` (rows, head_dim)."""

    def __init__(self, c: KleinConfig, first: int = 0, w=None):
        self.c = c
        self.n = c.single_group
        for j in range(self.n):
            self.put(j, "s", single_stream(c, w, f"single_transformer_blocks.{first + j}."))
        self.one = vec(np.ones(c.hidden))
        n, H, D = c.rows, c.heads, c.head_dim
        self.ks, self.vs = iron.state((n, H, D)), iron.state((n, H, D))

    def body(self, x, mods, angles, *, nk: Scratchpad[np.int32]):
        c = self.c
        E, H, D, n = c.hidden, c.heads, c.head_dim, c.rows
        shift, scale, gate = (mods[r * E : (r + 1) * E] for r in range(3))
        for j in range(self.n):
            s = self.s(j, "s")
            h = F.layer_norm(x, (E,), weight=scale + self.one, eps=c.eps) + shift
            q, k, v = qkv(s, h, c, angles)
            self.ks[0:n] = k.reshape(n, H, D)
            self.vs[0:n] = v.reshape(n, H, D)
            o = iron_anima.dit_attention(q.reshape(n, H, D), self.ks[:nk], self.vs[:nk]).reshape(n, E)
            y = F.linear(o, s.oa, precision=PREC) + F.linear(swiglu(s, h), s.om, precision=PREC)
            x = x + y * gate
            command_boundary()
        return x


class Mods(iron.Graph):
    """The timestep's sinusoid ``tproj`` (256) to the modulation rows (17 hidden): the doubles'
    image (6), their text (6), the singles' (3), norm_out's scale and shift (2)."""

    def __init__(self, c: KleinConfig, w=None):
        self.c = c
        t = "time_guidance_embed.timestep_embedder."
        self.t1, self.t2 = vec(w[t + "linear_1.weight"]), vec(w[t + "linear_2.weight"])
        self.img = vec(w["double_stream_modulation_img.linear.weight"])
        self.txt = vec(w["double_stream_modulation_txt.linear.weight"])
        self.single = vec(w["single_stream_modulation.linear.weight"])
        self.out = vec(w["norm_out.linear.weight"])
        self.mods = iron.state((17 * c.hidden,))

    def body(self, tproj):
        E = self.c.hidden
        P = Precision.HIGHEST
        a = F.silu(F.linear(F.silu(F.linear(tproj, self.t1, precision=P)), self.t2, precision=P))
        m = self.mods
        m[0 : 6 * E] = F.linear(a, self.img, precision=P)
        m[6 * E : 12 * E] = F.linear(a, self.txt, precision=P)
        m[12 * E : 15 * E] = F.linear(a, self.single, precision=P)
        m[15 * E : 17 * E] = F.linear(a, self.out, precision=P)
        return Copy.step(m[0 : 17 * E])


class DitIn(iron.Graph):
    """x_embedder of the image stream's rows (the picture's patches, then the references')."""

    def __init__(self, c: KleinConfig, w=None):
        self.w = vec(w["x_embedder.weight"])

    def body(self, x):
        return F.linear(x, self.w, precision=PREC)


class Ctx(iron.Graph):
    """context_embedder of the text's rows: ``h0``, ``h1``, ``h2`` (text_rows, 2560) are the
    hidden states after layers 9, 18 and 27, each a third of the embedder's columns."""

    def __init__(self, c: KleinConfig, w=None):
        e = np.asarray(w["context_embedder.weight"], np.float32)
        T = c.text
        self.w0, self.w1, self.w2 = vec(e[:, :T]), vec(e[:, T : 2 * T]), vec(e[:, 2 * T :])

    def body(self, h0, h1, h2):
        P = Precision.HIGHEST
        return F.linear(h0, self.w0, precision=P) + F.linear(h1, self.w1, precision=P) + F.linear(h2, self.w2, precision=P)


class DitOut(iron.Graph):
    """norm_out (layer norm, times 1 + scale, plus shift) and proj_out of the picture's rows."""

    def __init__(self, c: KleinConfig, w=None):
        self.c = c
        self.proj = vec(w["proj_out.weight"])
        self.one = vec(np.ones(c.hidden))

    def body(self, x, scale, shift):
        c = self.c
        h = F.layer_norm(x, (c.hidden,), weight=scale + self.one, eps=c.eps) + shift
        return F.linear(h, self.proj, precision=PREC)


# --- the text encoder (Qwen3-4B) ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TextConfig:
    rows: int = 512  # the prompt in the chat template, padded on the right (max_sequence_length)
    hidden: int = 2560
    layers: int = 27  # of 36: the DiT reads the hidden states after layers 9, 18 and 27
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    mlp: int = 9728
    theta: float = 1e6
    eps: float = 1e-6
    vocab: int = 151936
    group: int = 3  # layers a bank holds: every third group's output is a hidden state the DiT reads
    split: int = 1  # layers a command runs


class Qwen3Layer(iron.Graph):
    """A Qwen3 decoder layer over the 512 rows: RMS norms, GQA with q and k norms and RoPE,
    SwiGLU. The prompt's rows attend causally; the pads after them attend to the prompt's keys
    only (transformers' mask: a pad is no key, not even its own), so the attention runs twice,
    causal over all the rows and unmasked over the first ``nk`` keys (the prompt's), and ``sel``
    (rows, heads x head_dim: 1 in the prompt's rows, 0 in the pads') picks each row's."""

    def __init__(self, c: TextConfig, w, i: int):
        self.c = c
        p = f"model.layers.{i}."
        lin = lambda n: iron_sana.text_param(qi.f32(w[p + n]))  # noqa: E731
        vec_ = lambda n: iron.param(bf(w[p + n]))  # noqa: E731
        self.in_norm, self.post_norm = vec_("input_layernorm.weight"), vec_("post_attention_layernorm.weight")
        self.q, self.k = lin("self_attn.q_proj.weight"), lin("self_attn.k_proj.weight")
        self.v, self.o = lin("self_attn.v_proj.weight"), lin("self_attn.o_proj.weight")
        self.qn, self.kn = vec_("self_attn.q_norm.weight"), vec_("self_attn.k_norm.weight")
        self.gate, self.up = lin("mlp.gate_proj.weight"), lin("mlp.up_proj.weight")
        self.down = lin("mlp.down_proj.weight")

    def body(self, x, angles, sel, cache, nk):
        from iron.operators.mha import MHA

        c = self.c
        L, H, G, D = c.rows, c.heads, c.kv_heads, c.head_dim
        tl = iron_sana.text_linear
        h = F.rms_norm(x, (c.hidden,), self.in_norm, c.eps)
        q = F.rms_norm(tl(h, self.q).reshape(L * H, D), (D,), self.qn, c.eps)
        k = F.rms_norm(tl(h, self.k).reshape(L * G, D), (D,), self.kn, c.eps)
        v = tl(h, self.v)
        q, k = F.rope(q, angles), F.rope(k, angles)
        kc, vc = cache
        kc[0:L] = k.reshape(L, G, D)
        vc[0:L] = v.reshape(L, G, D)
        kw = dict(heads_interleaved=True, kv_interleaved=True, d=D, num_pipelines=8)
        oc = MHA.step(q.reshape(L, H, D), kc[0:L], vc[0:L], causal=True, **kw).reshape(L, H * D)
        op = MHA.step(q.reshape(L, H, D), kc[:nk], vc[:nk], causal=False, **kw).reshape(L, H * D)
        o = op + (oc - op) * sel
        x = x + tl(o, self.o)
        h = F.rms_norm(x, (c.hidden,), self.post_norm, c.eps)
        # the sigmoid in float32 (iron_silu): IRON's table SiLU is 0 from -8 down and 1 % off, and
        # the up branch of the few large channels is in the hundreds
        return x + tl(iron_silu.silu(tl(h, self.gate)) * tl(h, self.up), self.down)


class Text(iron.Graph):
    """``group`` layers of Qwen3-4B (from ``first``) over the 512 rows ``x``, turned by ``angles``
    (rows, head_dim), with ``sel`` and per call ``nk``, the prompt's tokens (Qwen3Layer). One
    ELF, a bank image per group of layers, read into one bank in turn: each group's weights are
    read once a prompt. The outputs of the groups ending at layers 9, 18 and 27 are what the DiT
    reads (ctx)."""

    def __init__(self, c: TextConfig, w=None, first: int = 0):
        self.c = c
        self.layers = [Qwen3Layer(c, w, first + i) for i in range(c.group)]
        G, D = c.kv_heads, c.head_dim
        self.cache = (iron.state((c.rows, G, D)), iron.state((c.rows, G, D)))

    def body(self, x, angles, sel, *, nk: Scratchpad[np.int32]):
        c = self.c
        for i, layer in enumerate(self.layers):
            if i and i % c.split == 0:
                command_boundary()
            x = layer(x, angles, sel, self.cache, nk)
        return x


def export_text(root: Path, out: Path, check: bool) -> None:
    c = TextConfig()
    w = qi.Checkpoint(root / "text_encoder", random=ARGS.random)
    t0 = time.time()
    iron_sana.TEXT_FMT["fmt"] = ARGS.text
    iron_sana.PREC.text = Precision.HIGHEST
    groups = c.layers // c.group
    meta = dict(config=dataclasses.asdict(c), weights=ARGS.text, groups=groups,
                taps=[layer // c.group - 1 for layer in KleinConfig().text_layers])
    g = Text(c, w, 0)
    cg = g.compile(x=(c.rows, c.hidden), angles=(c.rows, c.head_dim), sel=(c.rows, c.heads * c.head_dim))
    d = out / "text"
    m = export_graph(cg, d, "text", meta)
    print(f"text: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    variants = [{k: b["image"] for k, b in m["banks"].items()}]
    for i in range(1, 1 if ARGS.one_group else groups):
        variants.append(variant_bank(cg, Text(c, w, i * c.group), d, f"bank_g{i}"))
        print(f"text: bank of layers {i * c.group}..{(i + 1) * c.group - 1}", flush=True)
    m["variants"] = variants
    (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    import shutil

    shutil.copyfile(root / "tokenizer" / "tokenizer.json", d / "tokenizer.json")
    e = w["model.embed_tokens.weight"]
    if not ARGS.one_group:
        with open(d / "embed.bin", "wb") as f:
            for s in range(0, e.shape[0], 16384):
                f.write(bf(e[s : s + 16384]).tobytes())
    if check:
        rng = np.random.default_rng(0)
        real = 25
        ids = np.concatenate([rng.integers(0, 150000, real), np.full(c.rows - real, 151643)])
        sel = np.zeros((c.rows, c.heads * c.head_dim), np.float32)
        sel[:real] = 1
        write_check(g, m, d, dict(x=bf(qi.f32(e[ids])), angles=bf(iron_anima.half_rope_table(c.rows, c.head_dim, c.theta)),
                                  sel=bf(sel)), dict(nk=real))


# --- the VAE (AutoencoderKLFlux2) --------------------------------------------------------
#
# The decoder is FLUX.1's (iron_chroma_parts.FluxVaeStage: conv_in, the mid block with its
# one-head attention, four up blocks of three residual blocks, GroupNorm and SiLU before every
# convolution) over 32 latent channels, after post_quant_conv (a 1 x 1 convolution, the first
# grid's affine). The host un-normalizes the DiT's rows by the VAE's batch-norm statistics
# (128 channels: the 2 x 2 patches) and un-patches them to the 32 channels at an 8th.


def vae_weights(root: Path):
    cls = iron_anima.RandomWeights if ARGS.random else iron_anima.Safetensors
    return cls(root / "vae" / "diffusion_pytorch_model.safetensors")


def make_vae_stage():
    import iron_chroma_parts as cp

    class Flux2VaeStage(cp.FluxVaeStage):
        LATENT = 32

        def in_affine(self, w, vae_cfg: dict):
            a = np.zeros((64, 64), np.float32)
            a[:32, :32] = qi.f32(w["post_quant_conv.weight"])[:, :, 0, 0]
            b = np.zeros(64, np.float32)
            b[:32] = qi.f32(w["post_quant_conv.bias"])
            return a, b

    return cp, Flux2VaeStage


def make_enc_stage():
    """The encoder's stages, for the references of an edit: conv_in (RGB in a grid of 64
    channels), four down blocks of two residual blocks (the first three ending in a 3 x 3
    convolution of stride 2, padded on the right and below), the mid block with its one-head
    attention, conv_norm_out, SiLU, conv_out and quant_conv (folded into conv_out): the latents'
    means, the first 32 of the head's 64 channels (the pipeline takes the mode of the
    distribution, not a sample).

    Ops: ("enc_in",); ("res", path, cin, cout); ("attn", path); ("down", path, c); ("enc_head",).
    """
    import iron_chroma_parts as cp
    from iron_anima import GridShape, tap

    class Flux2EncStage(cp.FluxVaeStage):
        def __init__(self, ops: list, g, w, vae_cfg: dict):
            self.ops, self.g_in = ops, g
            self.params, self._index, self.states = [], {}, []
            self._consts = {}
            self._gn_grids = []
            f = lambda n: qi.f32(w[n])  # noqa: E731
            gg = g
            for op in ops:
                kind = op[0]
                if kind == "enc_in":
                    self._conv3("conv_in", w, "encoder.conv_in", 64)
                    gg = GridShape(gg.H, gg.W, f("encoder.conv_in.weight").shape[0])
                elif kind == "res":
                    _, path, cin, cout = op
                    self._gn(f"{path}.norm1", w, path + ".norm1", cp.NormShape(gg.H * gg.W, cin))
                    self._conv3(f"{path}.conv1", w, f"{path}.conv1", cin)
                    self._gn(f"{path}.norm2", w, path + ".norm2", cp.NormShape(gg.H * gg.W, cout))
                    self._conv3(f"{path}.conv2", w, f"{path}.conv2", cout)
                    if cin != cout:
                        self._param(f"{path}.sc.w", f(f"{path}.conv_shortcut.weight")[:, :, 0, 0])
                        self._param(f"{path}.sc.b", f(f"{path}.conv_shortcut.bias"))
                    gg = GridShape(gg.H, gg.W, cout)
                elif kind == "attn":
                    _, path = op
                    C = gg.C
                    self._gn(f"{path}.norm", w, path + ".group_norm", cp.NormShape(gg.H * gg.W, C))
                    s = C ** -0.5  # the scores' scale, folded into the queries
                    self._param(f"{path}.q.w", f(f"{path}.to_q.weight") * s)
                    self._param(f"{path}.q.b", f(f"{path}.to_q.bias") * s)
                    for n in "kv":
                        self._param(f"{path}.{n}.w", f(f"{path}.to_{n}.weight"))
                        self._param(f"{path}.{n}.b", f(f"{path}.to_{n}.bias"))
                    self._param(f"{path}.proj.w", f(f"{path}.to_out.0.weight"))
                    self._param(f"{path}.proj.b", f(f"{path}.to_out.0.bias"))
                    self._const(f"eye{C}", np.eye(C, dtype=np.float32))
                elif kind == "down":
                    _, path, c = op
                    self._conv3(f"{path}.down", w, f"{path}.downsamplers.0.conv", c)
                    gg = GridShape(gg.H // 2, gg.W // 2, c)
                elif kind == "enc_head":
                    self._gn("head.norm", w, "encoder.conv_norm_out", cp.NormShape(gg.H * gg.W, gg.C))
                    wq, bq = f("quant_conv.weight")[:, :, 0, 0], f("quant_conv.bias")
                    wt = np.einsum("om,mcyx->ocyx", wq, f("encoder.conv_out.weight"))  # (64, 512, 3, 3)
                    b = wq @ f("encoder.conv_out.bias") + bq
                    L = wt.shape[0] // 2  # the means (the log variances are not read)
                    full = np.zeros((64,) + wt.shape[1:], np.float32)
                    full[:L] = wt[:L]
                    self._conv_weight("head", full.transpose(0, 2, 3, 1).reshape(64, -1))
                    b64 = np.zeros(64, np.float32)
                    b64[:L] = b[:L]
                    self._param("head.b", b64)
                else:
                    raise ValueError(op)

        def down(self, x, g, path: str):
            """Downsample2D: the 3 x 3 convolution of stride 2 over the grid padded by one pixel
            on the right and below. Output (Y, X) is the same-padded convolution's at (2 Y + 1,
            2 X + 1): the convolution at the full size, then every other pixel into the half grid."""
            C = g.C
            y, gy = self.conv3(x, g, f"{path}.down", C)
            command_boundary()
            gout = GridShape(g.H // 2, g.W // 2, C)
            out = self.new_grid(gout)
            Copy.step(y, out,
                      src=tap((gy.rows, C), gy.pixel(1, 1) * C, [gout.H, gout.W, C], [2 * gy.Wp * C, 2 * C, 1]),
                      dst=tap((gout.rows, C), gout.pixel(0, 0) * C, [gout.H, gout.W, C], [gout.Wp * C, C, 1]))
            return self.view(out), gout

        def body(self, x):
            g = self.g_in
            for op in self.ops:
                kind = op[0]
                if kind == "enc_in":
                    x, g = self.conv3(x, g, "conv_in", self.w("conv_in.b").shape[0])
                elif kind == "res":
                    x, g = self.res(x, g, op[1], op[3])
                elif kind == "attn":
                    x, g = self.attn(x, g, op[1])
                elif kind == "down":
                    x, g = self.down(x, g, op[1])
                elif kind == "enc_head":
                    h = self.norm_silu(x, g, "head.norm")
                    y, gy = self.conv3(h, g, "head", 64)
                    n = g.H * g.W
                    return Copy.step(y, src=tap((gy.rows, 64), gy.pixel(0, 0) * 64, [g.H, g.W * 64], [gy.Wp * 64, 1])).reshape(n, 64)
                if op is not self.ops[-1]:
                    command_boundary()
            self.g_out = g
            return x if self.ops[-1][0] == "res" else Copy.step(x)

    return cp, Flux2EncStage


def enc_stages(cfg: dict, width: int, height: int) -> list:
    """The encoder in stages (each command under amdxdna's 2 s), with each one's input grid: at
    the two largest sizes a residual block a stage, as the decoder's."""
    from iron_anima import GridShape

    ch = cfg["block_out_channels"]  # 128 256 512 512
    n = cfg["layers_per_block"]
    h, w = height, width
    e = "encoder"
    stages, cur, gin = [], [("enc_in",)], GridShape(h, w, 64)
    prev = ch[0]

    def cut(grid):
        nonlocal cur, gin
        if cur:
            stages.append((cur, gin))
        cur, gin = [], grid

    for i, cout in enumerate(ch):
        p = f"{e}.down_blocks.{i}"
        for j in range(n):
            cur.append(("res", f"{p}.resnets.{j}", prev if j == 0 else cout, cout))
            if i < 2 and j < n - 1:  # a residual block a stage at the two largest sizes
                cut(GridShape(h, w, cout))
        if i < len(ch) - 1:
            cur.append(("down", p, cout))
            h, w = h // 2, w // 2
        prev = cout
        if i < len(ch) - 1:
            cut(GridShape(h, w, cout))
    mid = ch[-1]
    cur += [("res", f"{e}.mid_block.resnets.0", mid, mid), ("attn", f"{e}.mid_block.attentions.0"),
            ("res", f"{e}.mid_block.resnets.1", mid, mid), ("enc_head",)]
    cut(None)
    return stages


def export_encoder(root: Path, out: Path, width: int, height: int, check: bool, ref_image: Path | None, only: list | None,
                   name: str = "vae_enc") -> None:
    """vae_enc/s*: the encoder's stages. With ``ref_image`` (ref_flux2klein.py --encode's npz of a
    picture of the export's size) the stages' references run in turn on the picture and the
    latents' means are compared with diffusers'."""
    cp, Stage = make_enc_stage()
    w = vae_weights(root)
    cfg = json.loads((root / "vae" / "config.json").read_text())
    stages = enc_stages(cfg, width, height)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    x_ref = want = None
    if ref_image is not None:  # tools/ref/ref_flux2klein.py --encode's npz: the picture and diffusers' latents
        z = np.load(ref_image)
        x_ref = cp.grid_of(z["image"].reshape(-1, 3), stages[0][1])
        want = z["latents"].reshape(z["latents"].shape[0], -1).T  # (pixels, 32)
    for i, (ops, g) in enumerate(stages):
        if only and i not in only:
            continue
        t0 = time.time()
        st = Stage(ops, g, w, cfg)
        cg = st.compile(x=(g.rows, g.C))
        go = st.g_out if ops[-1][0] != "enc_head" else None
        meta = dict(stage=i, ops=json.loads(json.dumps(ops)), grid_in=dataclasses.asdict(g),
                    grid_out=None if go is None else dataclasses.asdict(go), pad=iron_anima.VAE_PAD,
                    width=width, height=height, norms=getattr(st, "norms", []))
        m = export_graph(cg, d / f"s{i}", f"{name}_s{i}", meta)
        print(f"{name} s{i}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if cp.HOST_NORM[0]:
            continue
        if x_ref is not None:
            x_in = x_ref
            y = qi.f32(st.reference(bf(x_in)))
            if go is not None:
                x_ref = y
            else:
                got = y[:, : want.shape[1]]
                err = np.abs(got - want)
                print(f"vae_enc s{i}: the latents against float32: cosine {cp.cosine(got, want):.6f}, "
                      f"max |err| {err.max():.4g} (|ref| max {np.abs(want).max():.4g})", flush=True)
            if check:
                write_check(st, m, d / f"s{i}", dict(x=bf(x_in)))
        elif check:
            x = np.zeros((g.rows, g.C), np.float32)
            C = 3 if ops[0][0] == "enc_in" else g.C
            for yy in range(g.H):
                x[g.pixel(yy, 0) : g.pixel(yy, 0) + g.W, :C] = rng.standard_normal((g.W, C))
            write_check(st, m, d / f"s{i}", dict(x=bf(x)))
    (d / "manifest.json").write_text(json.dumps(dict(
        stages=len(stages), width=width, height=height, pad=iron_anima.VAE_PAD, channels=3, latent=32), indent=1) + "\n")


def export_vae(root: Path, out: Path, width: int, height: int, check: bool, ref: Path | None, only: list | None) -> None:
    cp, Stage = make_vae_stage()
    w = vae_weights(root)
    cfg = json.loads((root / "vae" / "config.json").read_text())
    stages = cp.vae_stages(cfg, width, height)
    d = out / "vae"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    z = np.load(ref) if ref is not None else None
    x_ref = None if z is None else cp.grid_of(z["vae_in"].reshape(32, -1).T, stages[0][1])
    for i, (ops, g) in enumerate(stages):
        if only and i not in only:
            continue
        t0 = time.time()
        st = Stage(ops, g, w, cfg)
        cg = st.compile(x=(g.rows, g.C))
        go = st.g_out if ops[-1][0] != "head" else None
        meta = dict(stage=i, ops=json.loads(json.dumps(ops)), grid_in=dataclasses.asdict(g),
                    grid_out=None if go is None else dataclasses.asdict(go), pad=iron_anima.VAE_PAD,
                    width=width, height=height, norms=getattr(st, "norms", []))
        m = export_graph(cg, d / f"s{i}", f"vae_s{i}", meta)
        print(f"vae s{i}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if cp.HOST_NORM[0]:
            continue  # the host's norm steps are not in IRON's reference: checked on the NPU
        if x_ref is not None:
            x_in = x_ref
            y = qi.f32(st.reference(bf(x_in)))
            if go is not None:
                x_ref = y
            else:  # the picture: the reference's (H, W, 3) in 0..1
                got, want = y[:, :3], z["image"].reshape(-1, 3) * 2 - 1
                mse = float(((np.clip(got, -1, 1) - np.clip(want, -1, 1)) ** 2).mean()) / 4
                print(f"vae s{i}: the picture against float32: cosine {cp.cosine(got, want):.6f}, "
                      f"PSNR {10 * np.log10(1 / max(mse, 1e-20)):.1f} dB", flush=True)
            if check:
                write_check(st, m, d / f"s{i}", dict(x=bf(x_in)))
        elif check:
            write_check(st, m, d / f"s{i}", dict(x=bf(cp._check_input(st, g, rng))))
    mean = qi.f32(w["bn.running_mean"])
    std = np.sqrt(qi.f32(w["bn.running_var"]) + float(cfg["batch_norm_eps"]))
    (d / "manifest.json").write_text(json.dumps(dict(
        stages=len(stages), width=width, height=height, pad=iron_anima.VAE_PAD, channels=3, latent=32, patch=2,
        bn_mean=[float(v) for v in mean], bn_std=[float(v) for v in std]), indent=1) + "\n")


def suffix(c: KleinConfig) -> str:
    """The graphs' names for the references: "" (none), _r1 (one of the picture's size), _r1d2
    (one of half its sides)."""
    if not c.refs:
        return ""
    return f"_r{c.refs}" + (f"d{c.ref_div}" if c.ref_div != 1 else "")


def export_dit(root: Path, out: Path, c: KleinConfig, check: bool) -> None:
    w = qi.Checkpoint(root / "transformer", random=ARGS.random)
    rng = np.random.default_rng(0)
    meta = dict(config=dataclasses.asdict(c), weights=FMT.proj, mha=FMT.mha)
    E, sfx = c.hidden, suffix(c)
    small = [
        ("mods", lambda: Mods(c, w), dict(tproj=(256,)), lambda: dict(tproj=bf(qi.timestep_sinusoid(700.0)))),
        ("ctx", lambda: Ctx(c, w), {f"h{i}": (c.text_rows, c.text) for i in range(3)},
         lambda: {f"h{i}": bf(rng.standard_normal((c.text_rows, c.text))) for i in range(3)}),
        ("dit_in" + sfx, lambda: DitIn(c, w), dict(x=(c.image_rows, c.latent)),
         lambda: dict(x=bf(rng.standard_normal((c.image_rows, c.latent))))),
        ("dit_out", lambda: DitOut(c, w), dict(x=(c.tokens, E), scale=(E,), shift=(E,)),
         lambda: dict(x=bf(rng.standard_normal((c.tokens, E))), scale=bf(0.1 * rng.standard_normal(E)),
                      shift=bf(0.1 * rng.standard_normal(E)))),
    ]
    for name, make, shapes, inputs in small:
        if name.removesuffix(sfx) not in ARGS.parts or (c.refs and name in ("mods", "ctx", "dit_out")):
            continue  # the graphs that do not depend on the references are the picture's alone
        t0 = time.time()
        g = make()
        cg = g.compile(**shapes)
        m = export_graph(cg, out / name, name, meta)
        print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            write_check(g, m, out / name, inputs())
    for base, cls, total, group, nmods in (("doubles", Doubles, c.doubles, c.double_group, 12),
                                           ("singles", Singles, c.singles, c.single_group, 3)):
        if base not in ARGS.parts:
            continue
        name = base + sfx
        t0 = time.time()
        g = cls(c, 0, w)
        shapes = dict(x=(c.rows, E), mods=(nmods * E,), angles=(c.rows, c.head_dim))
        cg = g.compile(**shapes)
        d = out / name
        groups = total // group
        m = export_graph(cg, d, name, dict(meta, groups=groups, group=group))
        print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        variants = [{k: b["image"] for k, b in m["banks"].items()}]
        shared = c.refs and qi.share_banks(m, d, out / base)
        if shared:
            variants = [{k: f.replace("../dit_blocks/", f"../{base}/") for k, f in v.items()} for v in shared]
            for b in m["banks"].values():
                if b.get("image"):
                    b["image"] = b["image"].replace("../dit_blocks/", f"../{base}/")
            print(f"{name}: the banks are {base}'s (the same weights in the same places): stored once", flush=True)
        for i in range(1, 1 if ARGS.one_group or shared else groups):
            variants.append(variant_bank(cg, cls(c, i * group, w), d, f"bank_g{i}"))
            print(f"{name}: bank of blocks {i * group}..{(i + 1) * group - 1}", flush=True)
        m["variants"] = variants
        (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
        if check:
            mods = np.concatenate([0.1 * rng.standard_normal(E) for _ in range(nmods)])
            write_check(g, m, d, dict(x=bf(rng.standard_normal((c.rows, E))), mods=bf(mods), angles=bf(angles(c))),
                        dict(nk=c.rows))


def write_model(root: Path, out: Path, c: KleinConfig) -> None:
    """model.json: the checkpoint's scheduler and the pipeline's defaults (4 steps, no guidance:
    the checkpoint is distilled)."""
    sched = json.loads((root / "scheduler" / "scheduler_config.json").read_text())
    mi = json.loads((root / "model_index.json").read_text())
    (out / "model.json").write_text(json.dumps(dict(
        name=root.name, scheduler=sched, steps=4 if mi.get("is_distilled", True) else 50,
        guidance=1.0 if mi.get("is_distilled", True) else 4.0, width=c.width, height=c.height), indent=1) + "\n")


ARGS = None


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--refs", nargs="*", default=["0", "1", "1d2"],
                    help="the references to compile the blocks for: 0 (text to image), N (N references of the picture's "
                         "size), NdD (of its sides over D: 1d2 is one reference of half the sides, a quarter of its rows)")
    ap.add_argument("--parts", nargs="*", default=["text", "mods", "dit_in", "ctx", "dit_out", "doubles", "singles", "vae", "vae_enc"])
    ap.add_argument("--ref-enc", type=Path, help="tools/ref/ref_flux2klein.py --encode's OUT.npz: run the encoder stages' references in turn on its picture, against its latents")
    ap.add_argument("--stages", type=int, nargs="*", help="the VAE's stages to export (default all)")
    ap.add_argument("--host-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="the VAE's GroupNorms as two operators and a host step (iron_groupnorm)")
    ap.add_argument("--ref", type=Path, help="tools/ref/ref_flux2klein.py's OUT.npz: run the VAE stages' references in turn on its latents")
    ap.add_argument("--proj", default=FMT.proj, choices=("bf16", "q8", "q6", "q4", "bfp16"))
    ap.add_argument("--text", default="bf16", choices=("bf16", "q8", "q6", "q4"),
                    help="the text encoder's projections (bf16: the pads' rows, which the DiT attends to, drift with fewer bits)")
    ap.add_argument("--mha", default=FMT.mha, choices=("qk", "qkpv", "bf16"),
                    help="the blocks' attention products through bfp16: QK^T (qk), P V too (qkpv), or neither (iron_anima.dit_attention)")
    ap.add_argument("--exp", default="mid", choices=("exact", "mid", "instruction"),
                    help="the attention's softmax exponentials (iron_mha.ImageMHA.exp; mid: one bf16 limb a product, "
                         "the attention -20 %% at the exact one's error)")
    ap.add_argument("--one-group", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    FMT.proj, FMT.mha = a.proj, a.mha
    iron_anima.DIT_FMT.mha = a.mha
    import iron_mha

    iron_mha.EXP[0] = a.exp
    import iron_q4

    iron_q4.PREFILL["q4"] = "bf16"
    iron_q4.install_linear()
    bind_device(a.device, a.cols)
    a.out.mkdir(parents=True, exist_ok=True)
    if "text" in a.parts:
        export_text(a.root, a.out, a.check)
    divs = set()
    for r in sorted(a.refs, key=lambda r: ("d" in r, r)):  # the picture's alone first: the others share its banks
        n, _, d = r.partition("d")
        c = KleinConfig(width=a.size, height=a.size, refs=int(n), ref_div=int(d or 1))
        if c.refs and c.rows % 512:  # the attention's 8 pipelines take whole rounds of 64 rows each
            print(f"{r}: {c.rows} rows are no multiple of 512: not compiled at this size", flush=True)
            continue
        if c.refs and c.ref_div > 1:
            divs.add(c.ref_div)
        export_dit(a.root, a.out, c, a.check)
    iron_anima.VAE_TILE_N[0] = True  # the convolutions' out-feature tiles over all 8 columns
    if a.host_norm:
        import iron_chroma_parts

        iron_chroma_parts.HOST_NORM[0] = True
    if "vae" in a.parts:
        export_vae(a.root, a.out, a.size, a.size, a.check, a.ref, a.stages)
    if "vae_enc" in a.parts:
        export_encoder(a.root, a.out, a.size, a.size, a.check, a.ref_enc, a.stages)
        for d in sorted(divs):  # the encoder of a smaller reference
            export_encoder(a.root, a.out, a.size // d, a.size // d, False, None, a.stages, name=f"vae_enc_d{d}")
    write_model(a.root, a.out, KleinConfig(width=a.size, height=a.size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
