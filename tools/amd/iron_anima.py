#!/usr/bin/env python3
"""Anima (circlestone-labs/Anima) on IRON: the graphs cmd/hwnpu-image runs on AMD's NPU.

An application of amd/IRON kept in hwnpu (IRON has none for an image model). Anima is NVIDIA's
Cosmos-Predict2 2B DiT with Qwen3-0.6B as its text encoder, an LLM adapter from Qwen3's
hidden states to the T5-sized embeddings the DiT cross-attends to, and the Qwen-Image VAE
(Wan 2.1's, one frame). The graphs, each exported by iron_graph.py into a directory of its own:

  qwen3      Qwen3-0.6B over up to ``text`` prompt tokens (their embedding rows, which the host
             looks up), causal, its last hidden state after the final norm
  adapter    the LLM adapter: T5 token embeddings (looked up by the host) attend to themselves
             and to Qwen3's hidden states, six blocks, out_proj and RMS norm
  dit_in     the patch embedding of the latents (2 x 2 patches) and the timestep embedding
  dit_blocks ``group`` DiT blocks: one ELF, a bank image per group of the 28 blocks
  dit_out    the final adaLN and projection to the 2 x 2 x 16 velocity of each patch
  vae_*      the VAE decoder (iron_anima_vae.py)

The image size is fixed per export (the latents' grid, the RoPE table and the attention's
shape follow it). The host (internal/image/anima) does what is not a tensor computation: the
tokenizers, the embedding rows, the timestep's sinusoid, patching and unpatching, the
flow-matching Euler step and the PNG.

  python tools/amd/iron_anima.py ~/models/Anima OUT --size 1024

See docs/amd-npu-models.md.
"""

import argparse
import dataclasses
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import iron
import iron.functional as F
from iron import Precision, Scratchpad
from iron.operators.copy import Copy
from iron.operators.mha import MHA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_graph import bind_device, export_graph, trim_zeros, write_check  # noqa: E402

PREFIX = "model.diffusion_model."


@dataclasses.dataclass
class Precisions:
    """The matmuls' precision of each part (IRON's Precision; set from the command line).

    MEDIUM rounds A and B to bfp16 for the multiplies and is about 6.5 times faster than HIGH on
    the DiT's projections (2.1 against 13.3 ms at 4096 x 2048 x 2048), so the DiT, which runs 8
    or more times an image, keeps it; the text encoder runs once and takes HIGHEST.
    """

    text: Precision = Precision.HIGHEST
    dit: Precision = Precision.MEDIUM
    vae: Precision = Precision.MEDIUM


PREC = Precisions()


# --- weights ---------------------------------------------------------------------------


class Safetensors:
    """A .safetensors file's tensors by name, as numpy views (bf16 through ml_dtypes)."""

    DTYPES = {"BF16": bfloat16, "F16": np.float16, "F32": np.float32}

    def __init__(self, path: Path, prefix: str = ""):
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        self.base = 8 + n
        self.header = {k.removeprefix(prefix): v for k, v in header.items() if k.startswith(prefix)}
        self.data = np.memmap(path, dtype=np.uint8, mode="r")

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def __getitem__(self, name: str) -> np.ndarray:
        t = self.header[name]
        a, b = t["data_offsets"]
        return self.data[self.base + a : self.base + b].view(self.DTYPES[t["dtype"]]).reshape(t["shape"])


class RandomWeights(Safetensors):
    """A checkpoint's names and shapes with weights drawn at random (norm weights 1): to check
    the graphs' compile, speed and agreement before (or without) the real weights."""

    def __init__(self, path: Path, prefix: str = "", seed: int = 0):
        super().__init__(path, prefix)
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, name: str) -> np.ndarray:
        shape = self.header[name]["shape"]
        if "norm" in name and len(shape) == 1:
            return np.ones(shape, np.float32)
        fan = shape[-1] if len(shape) > 1 else 1
        return (self.rng.standard_normal(shape) / math.sqrt(fan)).astype(np.float32)


def weights(path: Path, prefix: str = "") -> Safetensors:
    return RandomWeights(path, prefix) if ARGS.random else Safetensors(path, prefix)


def command_boundary() -> None:
    """End the command here: an empty host step. A command's control code must fit the device
    heap (64 MiB on Linux 7.0) and its run amdxdna's 2 s, so a long graph is several."""
    iron.on_host(lambda: None)


def bf(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, np.float32).astype(bfloat16))


# --- shapes ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DitConfig:
    """The Cosmos-Predict2 2B DiT's shape (transformer/config.json) at one image size."""

    width: int = 1024
    height: int = 1024
    hidden: int = 2048
    heads: int = 16
    head_dim: int = 128
    layers: int = 28
    mlp: int = 8192
    text_dim: int = 1024
    text_tokens: int = 512  # the adapter pads its output to 512 rows (min_sequence_length)
    adaln_lora: int = 256
    latent_channels: int = 16
    patch: int = 2
    vae_scale: int = 8
    rope_scale: tuple = (1.0, 4.0, 4.0)
    group: int = 4  # blocks a dit_blocks call runs: about 1 s at 1024 x 1024, under amdxdna's 2 s

    def __post_init__(self):
        side = self.vae_scale * self.patch
        if self.width % side or self.height % side:
            raise ValueError(f"{self.width}x{self.height} is no whole {side}-pixel patches")
        if self.tokens % 64:
            raise ValueError(f"{self.tokens} patches: MHA takes whole 64-row blocks")
        if self.layers % self.group:
            raise ValueError(f"{self.layers} blocks are no whole groups of {self.group}")

    @property
    def grid(self) -> tuple[int, int]:
        """The patches down and across."""
        s = self.vae_scale * self.patch
        return self.height // s, self.width // s

    @property
    def tokens(self) -> int:
        gh, gw = self.grid
        return gh * gw

    @property
    def patch_dim(self) -> int:
        """A patch's features in: 16 channels by 2 x 2 (the padding mask's channel is zero for a
        whole image, so its four columns of the embedding are left out)."""
        return self.latent_channels * self.patch * self.patch


def rope_table(c: DitConfig) -> np.ndarray:
    """CosmosRotaryPosEmbed's angles, ``(tokens, head_dim)``, cos and sin interleaved (IRON's RoPE).

    Pair ``i`` of a head (``x[i]`` with ``x[i + 64]``) turns by the frame (``dim_t = 44``: 22
    pairs, all at frame 0), then the row (21 pairs), then the column (21 pairs), each axis at
    its NTK-scaled theta.
    """
    D = c.head_dim
    dim_h = D // 6 * 2
    dim_w = dim_h
    dim_t = D - dim_h - dim_w
    h_theta = 10000.0 * c.rope_scale[1] ** (dim_h / (dim_h - 2))
    w_theta = 10000.0 * c.rope_scale[2] ** (dim_w / (dim_w - 2))
    t_theta = 10000.0 * c.rope_scale[0] ** (dim_t / (dim_t - 2))
    fh = 1.0 / h_theta ** (np.arange(0, dim_h, 2)[: dim_h // 2] / dim_h)
    fw = 1.0 / w_theta ** (np.arange(0, dim_w, 2)[: dim_w // 2] / dim_w)
    ft = 1.0 / t_theta ** (np.arange(0, dim_t, 2)[: dim_t // 2] / dim_t)
    gh, gw = c.grid
    rows = np.repeat(np.arange(gh), gw).astype(np.float64)
    cols = np.tile(np.arange(gw), gh).astype(np.float64)
    a = np.concatenate([np.zeros((gh * gw, ft.size)), rows[:, None] * fh, cols[:, None] * fw], axis=1)
    assert a.shape[1] == D // 2
    out = np.empty((gh * gw, D), np.float32)
    out[:, ::2] = np.cos(a)
    out[:, 1::2] = np.sin(a)
    return out


# --- the DiT -----------------------------------------------------------------------------


class AdaLN(iron.Graph):
    """An ``adaln_modulation`` (SiLU, linear to the LoRA rank, linear out), its output plus ``temb``."""

    def __init__(self, c: DitConfig, outs: int):
        self.l1 = iron.param(c.adaln_lora, c.hidden)
        self.l2 = iron.param(outs * c.hidden, c.adaln_lora)
        self.E = c.hidden

    def load(self, w: Safetensors, path: str) -> None:
        self.l1 = iron.param(bf(w[f"{path}.1.weight"]))
        self.l2 = iron.param(bf(w[f"{path}.2.weight"]))

    def rows(self, act, temb, n: int) -> list:
        """The modulation's rows (shift, scale, gate...) of ``act`` = SiLU(embedded timestep)."""
        m = F.linear(act, self.l1, precision=PREC.dit)
        m = F.linear(m, self.l2, precision=PREC.dit) + temb[: n * self.E]
        return [m[i * self.E : (i + 1) * self.E] for i in range(n)]


class DitBlock(iron.Graph):
    """A Cosmos-Predict2 block: adaLN-zero self-attention (RoPE), cross-attention, GELU MLP."""

    def __init__(self, c: DitConfig, scratch):
        E, T = c.hidden, c.text_dim
        self.c = c
        for name in ("self_attn", "cross_attn"):
            kv_in = E if name == "self_attn" else T
            setattr(self, name, _Attn(c, kv_in))
        self.mlp_1 = iron.param(c.mlp, E)
        self.mlp_2 = iron.param(E, c.mlp)
        self.ada = [AdaLN(c, 3) for _ in range(3)]
        self.scratch = scratch

    def load(self, w: Safetensors, i: int) -> None:
        p = f"blocks.{i}"
        self.self_attn.load(w, f"{p}.self_attn")
        self.cross_attn.load(w, f"{p}.cross_attn")
        self.mlp_1 = iron.param(bf(w[f"{p}.mlp.layer1.weight"]))
        self.mlp_2 = iron.param(bf(w[f"{p}.mlp.layer2.weight"]))
        for a, name in zip(self.ada, ("self_attn", "cross_attn", "mlp")):
            a.load(w, f"{p}.adaln_modulation_{name}")

    def body(self, x, act, temb, ctx, angles, one):
        c = self.c
        E = c.hidden
        n = x.shape[0]
        # self-attention
        shift, scale, gate = self.ada[0].rows(act, temb, 3)
        h = F.layer_norm(x, (E,), eps=1e-6) * (scale + one) + shift
        q, k, v = self.self_attn.qkv(h, h, n, n)
        q, k = F.rope(q, angles), F.rope(k, angles)
        H, D = c.heads, c.head_dim
        o = MHA.step(q.reshape(n, H, D), k.reshape(n, H, D), v.reshape(n, H, D),
                     heads_interleaved=True, kv_interleaved=True, causal=False, num_pipelines=8)
        x = x + self.self_attn.out(o.reshape(n, E)) * gate
        # cross-attention over the text: MHA takes no more queries than keys, so 512 rows a call
        shift, scale, gate = self.ada[1].rows(act, temb, 3)
        h = F.layer_norm(x, (E,), eps=1e-6) * (scale + one) + shift
        L = ctx.shape[0]
        q, k, v = self.cross_attn.qkv(h, ctx, n, L)
        q, k, v = q.reshape(n, H, D), k.reshape(L, H, D), v.reshape(L, H, D)
        o = self.scratch
        for s in range(0, n, L):
            o[s : s + L] = MHA.step(q[s : s + L], k, v, heads_interleaved=True, kv_interleaved=True,
                                    causal=False, num_pipelines=8).reshape(L, E)
        x = x + self.cross_attn.out(o) * gate
        # MLP
        shift, scale, gate = self.ada[2].rows(act, temb, 3)
        h = F.layer_norm(x, (E,), eps=1e-6) * (scale + one) + shift
        h = F.linear(F.gelu(F.linear(h, self.mlp_1, precision=PREC.dit)), self.mlp_2, precision=PREC.dit)
        return x + h * gate


class _Attn(iron.Graph):
    """q, k, v and output projections (no bias), q and k RMS-normed per head."""

    def __init__(self, c: DitConfig, kv_in: int):
        E = c.hidden
        self.c = c
        self.q = iron.param(E, E)
        self.k = iron.param(E, kv_in)
        self.v = iron.param(E, kv_in)
        self.o = iron.param(E, E)
        self.qn = iron.param(c.head_dim)
        self.kn = iron.param(c.head_dim)

    def load(self, w: Safetensors, p: str) -> None:
        self.q = iron.param(bf(w[f"{p}.q_proj.weight"]))
        self.k = iron.param(bf(w[f"{p}.k_proj.weight"]))
        self.v = iron.param(bf(w[f"{p}.v_proj.weight"]))
        self.o = iron.param(bf(w[f"{p}.output_proj.weight"]))
        self.qn = iron.param(bf(w[f"{p}.q_norm.weight"]))
        self.kn = iron.param(bf(w[f"{p}.k_norm.weight"]))

    def qkv(self, h, src, n: int, m: int):
        H, D = self.c.heads, self.c.head_dim
        q = F.linear(h, self.q, precision=PREC.dit).reshape(n * H, D)
        k = F.linear(src, self.k, precision=PREC.dit).reshape(m * H, D)
        v = F.linear(src, self.v, precision=PREC.dit)
        return F.rms_norm(q, (D,), self.qn, 1e-6), F.rms_norm(k, (D,), self.kn, 1e-6), v

    def out(self, o):
        return F.linear(o, self.o, precision=PREC.dit)


class DitBlocks(iron.Graph):
    """``group`` DiT blocks over the patches ``x``.

    Inputs: ``x`` (tokens, hidden); ``temb`` (3 * hidden), t_embedder's output; ``et``
    (hidden), the timestep's sinusoid RMS-normed (``t_embedding_norm``); ``ctx`` (512, 1024),
    the adapter's text embeddings.
    """

    def __init__(self, c: DitConfig, first: int = 0, w: Safetensors | None = None):
        self.c = c
        self.scratch = iron.state((c.tokens, c.hidden))
        self.blocks = [DitBlock(c, self.scratch) for _ in range(c.group)]
        if w is not None:
            for j, b in enumerate(self.blocks):
                b.load(w, first + j)
        self.angles = iron.param(bf(rope_table(c)))
        self.one = iron.param(bf(np.ones(c.hidden)))

    def body(self, x, temb, et, ctx):
        # One command: a host step between the blocks (command_boundary) hangs the first
        # command on the NPU (ERT_CMD_STATE_TIMEOUT), seen 2026-11-05; four blocks are 20 MiB
        # of control code and about 1 s at 1024 x 1024.
        act = F.silu(et)
        for b in self.blocks:
            x = b(x, act, temb, ctx, self.angles, self.one)
        return x


class DitIn(iron.Graph):
    """The patch embedding (``x_embedder``) and the timestep embedding (``t_embedder``, ``t_embedding_norm``).

    Inputs: ``patches`` (tokens, 64), each 2 x 2 patch of the 16 latent channels, channel
    major (``c * 4 + dy * 2 + dx``); ``tproj`` (hidden), the timestep's sinusoid (diffusers'
    Timesteps, cos first). Outputs ``x``, ``temb`` and ``et``.
    """

    def __init__(self, c: DitConfig, w: Safetensors | None = None):
        E = c.hidden
        self.c = c
        if w is None:
            self.embed = iron.param(E, c.patch_dim)
            self.t1 = iron.param(E, E)
            self.t2 = iron.param(3 * E, E)
            self.tn = iron.param(E)
            return
        pw = np.asarray(w["x_embedder.proj.1.weight"], np.float32)  # (E, 17 * 4), the mask's 4 last
        assert pw.shape == (E, (c.latent_channels + 1) * 4), pw.shape
        self.embed = iron.param(bf(pw[:, : c.patch_dim]))
        self.t1 = iron.param(bf(w["t_embedder.1.linear_1.weight"]))
        self.t2 = iron.param(bf(w["t_embedder.1.linear_2.weight"]))
        self.tn = iron.param(bf(w["t_embedding_norm.weight"]))

    def body(self, patches, tproj):
        x = F.linear(patches, self.embed, precision=PREC.dit)
        temb = F.linear(F.silu(F.linear(tproj, self.t1, precision=PREC.dit)), self.t2, precision=PREC.dit)
        E = self.c.hidden
        et = F.rms_norm(tproj.reshape(1, E), (E,), self.tn, 1e-6).reshape(E)
        return x, temb, et


class DitOut(iron.Graph):
    """The final layer: adaLN (shift, scale; ``temb``'s first two thirds), the linear to 2 x 2 x 16.

    Output ``v`` (tokens, 64), each patch's velocity ordered ``dy * 32 + dx * 16 + c``.
    """

    def __init__(self, c: DitConfig, w: Safetensors | None = None):
        E = c.hidden
        self.c = c
        self.ada = AdaLN(c, 2)
        self.lin = iron.param(c.patch_dim, E)
        if w is not None:
            self.ada.load(w, "final_layer.adaln_modulation")
            self.lin = iron.param(bf(w["final_layer.linear.weight"]))
        self.one = iron.param(bf(np.ones(E)))

    def body(self, x, temb, et):
        E = self.c.hidden
        shift, scale = self.ada.rows(F.silu(et), temb, 2)
        h = F.layer_norm(x, (E,), eps=1e-6) * (scale + self.one) + shift
        return F.linear(h, self.lin, precision=PREC.dit)


# --- the text encoder and the LLM adapter -------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TextConfig:
    """Qwen3-0.6B (text_encoder/config.json) and the adapter (text_conditioner/config.json)."""

    rows: int = 512  # the pipeline's max_sequence_length: prompts are cut to it
    hidden: int = 1024
    layers: int = 28
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    mlp: int = 3072
    theta: float = 1e6
    eps: float = 1e-6
    adapter_layers: int = 6
    adapter_heads: int = 16
    adapter_mlp: int = 4096
    t5_vocab: int = 32128
    qwen_vocab: int = 151936


def half_rope_table(rows: int, dim: int, theta: float) -> np.ndarray:
    """Hugging Face's RoPE (two halves) at positions 0..rows-1, cos and sin interleaved."""
    inv = 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    a = np.arange(rows, dtype=np.float64)[:, None] * inv[None]
    out = np.empty((rows, dim), np.float32)
    out[:, ::2] = np.cos(a)
    out[:, 1::2] = np.sin(a)
    return out


class Qwen3Layer(iron.Graph):
    """A Qwen3 decoder layer over a prompt's rows, causal: RMS norms, GQA with q/k norms and RoPE, SwiGLU."""

    def __init__(self, c: TextConfig, w: Safetensors | None, i: int):
        E, H, G, D, M = c.hidden, c.heads, c.kv_heads, c.head_dim, c.mlp
        self.c = c
        p = f"layers.{i}."
        get = (lambda n, *s: iron.param(bf(w[p + n]))) if w is not None else (lambda n, *s: iron.param(*s))
        self.in_norm = get("input_layernorm.weight", E)
        self.post_norm = get("post_attention_layernorm.weight", E)
        self.q = get("self_attn.q_proj.weight", H * D, E)
        self.k = get("self_attn.k_proj.weight", G * D, E)
        self.v = get("self_attn.v_proj.weight", G * D, E)
        self.o = get("self_attn.o_proj.weight", E, H * D)
        self.qn = get("self_attn.q_norm.weight", D)
        self.kn = get("self_attn.k_norm.weight", D)
        self.gate = get("mlp.gate_proj.weight", M, E)
        self.up = get("mlp.up_proj.weight", M, E)
        self.down = get("mlp.down_proj.weight", E, M)

    def body(self, x, angles):
        c = self.c
        L, H, G, D = x.shape[0], c.heads, c.kv_heads, c.head_dim
        h = F.rms_norm(x, (c.hidden,), self.in_norm, c.eps)
        q = F.rms_norm(F.linear(h, self.q, precision=PREC.text).reshape(L * H, D), (D,), self.qn, c.eps)
        k = F.rms_norm(F.linear(h, self.k, precision=PREC.text).reshape(L * G, D), (D,), self.kn, c.eps)
        v = F.linear(h, self.v, precision=PREC.text)
        q, k = F.rope(q, angles), F.rope(k, angles)
        o = MHA.step(q.reshape(L, H, D), k.reshape(L, G, D), v.reshape(L, G, D),
                     heads_interleaved=True, kv_interleaved=True, causal=True, num_pipelines=8)
        x = x + F.linear(o.reshape(L, H * D), self.o, precision=PREC.text)
        h = F.rms_norm(x, (c.hidden,), self.post_norm, c.eps)
        g = F.silu(F.linear(h, self.gate, precision=PREC.text)) * F.linear(h, self.up, precision=PREC.text)
        return x + F.linear(g, self.down, precision=PREC.text)


class AdapterBlock(iron.Graph):
    """An LLM adapter block: self-attention of the T5 rows, cross-attention to Qwen3's, GELU MLP."""

    def __init__(self, c: TextConfig, w: Safetensors | None, i: int):
        E = c.hidden
        D = E // c.adapter_heads
        self.c = c
        p = f"llm_adapter.blocks.{i}."
        get = (lambda n, *s: iron.param(bf(w[p + n]))) if w is not None else (lambda n, *s: iron.param(*s))
        for a in ("self_attn", "cross_attn"):
            for n in ("q", "k", "v", "o"):
                setattr(self, f"{a}_{n}", get(f"{a}.{n}_proj.weight", E, E))
            setattr(self, f"{a}_qn", get(f"{a}.q_norm.weight", D))
            setattr(self, f"{a}_kn", get(f"{a}.k_norm.weight", D))
        for n in ("norm_self_attn", "norm_cross_attn", "norm_mlp"):
            setattr(self, n, get(f"{n}.weight", E))
        self.mlp_0 = get("mlp.0.weight", c.adapter_mlp, E)
        self.mlp_0b = get("mlp.0.bias", c.adapter_mlp)
        self.mlp_2 = get("mlp.2.weight", E, c.adapter_mlp)
        self.mlp_2b = get("mlp.2.bias", E)

    def attend(self, a: str, h, src, angles, n):
        c = self.c
        L, S, H = h.shape[0], src.shape[0], c.adapter_heads
        D = c.hidden // H
        get = lambda x: getattr(self, f"{a}_{x}")  # noqa: E731
        q = F.rms_norm(F.linear(h, get("q"), precision=PREC.text).reshape(L * H, D), (D,), get("qn"), 1e-6)
        k = F.rms_norm(F.linear(src, get("k"), precision=PREC.text).reshape(S * H, D), (D,), get("kn"), 1e-6)
        v = F.linear(src, get("v"), precision=PREC.text)
        q, k = F.rope(q, angles), F.rope(k, angles)
        k, v = k.reshape(S, H, D), v.reshape(S, H, D)
        o = MHA.step(q.reshape(L, H, D), k[:n], v[:n], heads_interleaved=True, kv_interleaved=True,
                     causal=False, num_pipelines=8)
        return F.linear(o.reshape(L, c.hidden), get("o"), precision=PREC.text)

    def body(self, x, src, angles, nt, nq):
        E = self.c.hidden
        h = F.rms_norm(x, (E,), self.norm_self_attn, 1e-6)
        x = x + self.attend("self_attn", h, h, angles, nt)
        h = F.rms_norm(x, (E,), self.norm_cross_attn, 1e-6)
        x = x + self.attend("cross_attn", h, src, angles, nq)
        h = F.rms_norm(x, (E,), self.norm_mlp, 1e-6)
        h = F.gelu(F.linear(h, self.mlp_0, self.mlp_0b, precision=PREC.text))
        return x + F.linear(h, self.mlp_2, self.mlp_2b, precision=PREC.text)


class Text(iron.Graph):
    """Qwen3-0.6B, then the LLM adapter: a prompt's conditioning for the DiT.

    Inputs: ``qx`` (rows, hidden), the embedding rows of the prompt's Qwen3 tokens (zeros past
    them); ``tx`` (rows, hidden), the adapter's embedding rows of its T5 tokens (``</s>`` last).
    Per call: ``nq`` and ``nt``, the tokens of each. Output ``cond`` (rows, hidden), valid in
    its first ``nt`` rows: the host zeroes the rest, as the adapter's mask does.
    """

    def __init__(self, c: TextConfig, qwen: Safetensors | None = None, dit: Safetensors | None = None):
        self.c = c
        self.layers = [Qwen3Layer(c, qwen, i) for i in range(c.layers)]
        self.blocks = [AdapterBlock(c, dit, i) for i in range(c.adapter_layers)]
        E = c.hidden
        if qwen is not None:
            self.norm = iron.param(bf(qwen["norm.weight"]))
            self.out_proj = iron.param(bf(dit["llm_adapter.out_proj.weight"]))
            self.out_proj_b = iron.param(bf(dit["llm_adapter.out_proj.bias"]))
            self.out_norm = iron.param(bf(dit["llm_adapter.norm.weight"]))
        else:
            self.norm, self.out_norm = iron.param(E), iron.param(E)
            self.out_proj, self.out_proj_b = iron.param(E, E), iron.param(E)
        self.qwen_angles = iron.param(bf(half_rope_table(c.rows, c.head_dim, c.theta)))
        self.adapter_angles = iron.param(bf(half_rope_table(c.rows, E // c.adapter_heads, 10000.0)))

    def body(self, qx, tx, *, nq: Scratchpad[np.int32], nt: Scratchpad[np.int32]):
        c = self.c
        h = qx
        for i, layer in enumerate(self.layers):
            if i and i % 7 == 0:
                command_boundary()
            h = layer(h, self.qwen_angles)
        src = F.rms_norm(h, (c.hidden,), self.norm, c.eps)
        command_boundary()
        x = tx
        for b in self.blocks:
            x = b(x, src, self.adapter_angles, nt, nq)
        x = F.linear(x, self.out_proj, self.out_proj_b, precision=PREC.text)
        return F.rms_norm(x, (c.hidden,), self.out_norm, 1e-6)


def export_text(root: Path, out: Path, check: bool) -> None:
    c = TextConfig()
    qwen = weights(root / "text_encoders" / "qwen_3_06b_base.safetensors", "model.")
    dit = weights(root / "diffusion_models" / ARGS.dit, PREFIX)
    t0 = time.time()
    g = Text(c, qwen, dit)
    cg = g.compile(qx=(c.rows, c.hidden), tx=(c.rows, c.hidden))
    m = export_graph(cg, out / "text", "text", dict(config=dataclasses.asdict(c)))
    print(f"text: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    # The tokenizers, and the embedding tables the host looks rows up in, bf16 raw.
    import shutil

    shutil.copyfile(root / "diffusers" / "tokenizer" / "tokenizer.json", out / "text" / "qwen_tokenizer.json")
    shutil.copyfile(root / "diffusers" / "t5_tokenizer" / "tokenizer.json", out / "text" / "t5_tokenizer.json")
    (out / "text" / "qwen_embed.bin").write_bytes(bf(qwen["embed_tokens.weight"]).tobytes())
    (out / "text" / "t5_embed.bin").write_bytes(bf(dit["llm_adapter.embed.weight"]).tobytes())
    if check:
        rng = np.random.default_rng(0)
        nq, nt = 37, 41
        qe, te = qwen["embed_tokens.weight"], dit["llm_adapter.embed.weight"]
        qx = np.zeros((c.rows, c.hidden), np.float32)
        tx = np.zeros((c.rows, c.hidden), np.float32)
        qx[:nq] = np.asarray(qe[rng.integers(0, 150000, nq)], np.float32)
        tx[:nt] = np.asarray(te[rng.integers(0, 32000, nt)], np.float32)
        write_check(g, m, out / "text", dict(qx=bf(qx), tx=bf(tx)), dict(nq=nq, nt=nt))


# --- the VAE decoder ---------------------------------------------------------------------
#
# The Qwen-Image VAE (Wan 2.1's) decoding one frame: every causal 3D convolution is its last
# temporal tap (the two frames before the first are zeros) and the temporal upsampling is not
# run (the first frame's time_conv is skipped). An image is held channel last in a zero-padded
# grid: (H + 2) rows of Wp = W + 16 pixels (a zero column before the image, 15 after) and a
# zero pixel before and after, so that a 3 x 3 convolution of the band of H rows is one GEMM
# over its im2col rows (Copy gathers each pixel's 3 x 3 neighbours, which the zeros pad), and
# whose interior Copy writes back. The pads are never written, so they stay zero; RMS norms and
# SiLU keep zeros zero. Wp is a multiple of 16 so a chunk's rows are whole 256-row GEMM tiles.

VAE_PAD = 16


@dataclasses.dataclass(frozen=True)
class GridShape:
    H: int
    W: int
    C: int

    @property
    def Wp(self) -> int:
        return self.W + VAE_PAD

    @property
    def rows(self) -> int:
        """The padded rows and a zero pixel either side, rounded up to 256 rows (zeros) so a
        copy of the whole grid splits into even transfers."""
        return -(-((self.H + 2) * self.Wp + 2) // 256) * 256

    @property
    def band(self) -> int:
        """The row of pixel (0, -1): the band of the image's rows starts there."""
        return 1 + self.Wp

    def pixel(self, y: int, x: int) -> int:
        return self.band + y * self.Wp + 1 + x

    def chunk(self, cin: int) -> int:
        """Image rows a convolution takes at once: a multiple of 16 dividing H, its im2col rows
        (9 * cin wide) at most 64 MiB."""
        best = 16
        for r in range(16, self.H + 1, 16):
            if self.H % r == 0 and r * self.Wp * 9 * cin * 2 <= 64 << 20:
                best = r
        assert self.H % best == 0 and best * self.Wp % 256 == 0, (self, best)
        return best


def pipelines(rows: int) -> int:
    """MHA's pipelines for rows queries: up to 8, each whole 64-row rounds."""
    return next(p for p in (8, 4, 2, 1) if rows % (64 * p) == 0)


def tap(shape: tuple, offset: int, sizes: list, strides: list):
    from aie.helpers.taplib import TensorAccessPattern

    return TensorAccessPattern(shape, offset, sizes, strides)


class VaeStage(iron.Graph):
    """A run of the VAE decoder's modules, from one padded grid to the next (or, ending with the
    head, to the image's pixels, dense, 64 channels of which the first three are RGB).

    ``ops``: ("in",) the latents' unnormalization and post_quant_conv (one 1 x 1 affine) then
    conv_in; ("res", path, cin, cout); ("attn", path); ("up", path, cin, cout), nearest 2x and
    the 3 x 3 convolution to cout; ("head",) norm_out, SiLU, conv_out.
    """

    def __init__(self, ops: list, g: GridShape, w: Safetensors, vae_cfg: dict):
        self.ops, self.g_in = ops, g
        self.params = []  # the weights, held in a list so the graph records them
        self._index = {}  # a weight's index in params by its name
        self.states = []
        for op in ops:
            kind = op[0]
            if kind == "in":
                C = 64
                mean, std = np.asarray(vae_cfg["latents_mean"]), np.asarray(vae_cfg["latents_std"])
                pq = np.asarray(w["post_quant_conv.weight"], np.float32)[:, :, 0, 0, 0]  # (16, 16)
                pb = np.asarray(w["post_quant_conv.bias"], np.float32)
                a = np.zeros((C, C), np.float32)
                a[:16, :16] = pq * std[None, :]
                b = np.zeros(C, np.float32)
                b[:16] = pq @ mean + pb
                self._param("in.w", a)
                self._param("in.b", b)
                self._conv3("conv_in", w, "decoder.conv_in", C)
            elif kind == "res":
                _, path, cin, cout = op
                self._param(f"{path}.norm1", np.asarray(w[f"{path}.norm1.gamma"], np.float32).reshape(-1))
                self._conv3(f"{path}.conv1", w, f"{path}.conv1", cin)
                self._param(f"{path}.norm2", np.asarray(w[f"{path}.norm2.gamma"], np.float32).reshape(-1))
                self._conv3(f"{path}.conv2", w, f"{path}.conv2", cout)
                if cin != cout:
                    self._param(f"{path}.sc.w", np.asarray(w[f"{path}.conv_shortcut.weight"], np.float32)[:, :, 0, 0, 0])
                    self._param(f"{path}.sc.b", np.asarray(w[f"{path}.conv_shortcut.bias"], np.float32))
            elif kind == "attn":
                _, path = op
                self._param(f"{path}.norm", np.asarray(w[f"{path}.norm.gamma"], np.float32).reshape(-1))
                qkv = np.asarray(w[f"{path}.to_qkv.weight"], np.float32)[:, :, 0, 0]
                qb = np.asarray(w[f"{path}.to_qkv.bias"], np.float32)
                C = qkv.shape[1]
                for i, n in enumerate("qkv"):
                    self._param(f"{path}.{n}.w", qkv[i * C : (i + 1) * C])
                    self._param(f"{path}.{n}.b", qb[i * C : (i + 1) * C])
                self._param(f"{path}.proj.w", np.asarray(w[f"{path}.proj.weight"], np.float32)[:, :, 0, 0])
                self._param(f"{path}.proj.b", np.asarray(w[f"{path}.proj.bias"], np.float32))
            elif kind == "up":
                _, path, cin, cout = op
                wt = np.asarray(w[f"{path}.upsamplers.0.resample.1.weight"], np.float32)  # (cout, cin, 3, 3)
                self._param(f"{path}.up.w", wt.transpose(0, 2, 3, 1).reshape(wt.shape[0], -1))
                self._param(f"{path}.up.b", np.asarray(w[f"{path}.upsamplers.0.resample.1.bias"], np.float32))
            elif kind == "head":
                self._param("head.norm", np.asarray(w["decoder.norm_out.gamma"], np.float32).reshape(-1))
                wt = np.asarray(w["decoder.conv_out.weight"], np.float32)[:, :, 2]  # (3, 96, 3, 3)
                full = np.zeros((64,) + wt.shape[1:], np.float32)
                full[:3] = wt
                self._param("head.w", full.transpose(0, 2, 3, 1).reshape(64, -1))
                b = np.zeros(64, np.float32)
                b[:3] = np.asarray(w["decoder.conv_out.bias"], np.float32)
                self._param("head.b", b)
            else:
                raise ValueError(op)

    def _param(self, name: str, x) -> None:
        self._index[name] = len(self.params)
        self.params.append(iron.param(bf(x)))

    def w(self, name: str):
        return self.params[self._index[name]]

    def _conv3(self, name: str, w: Safetensors, path: str, cin_padded: int) -> None:
        """A causal 3 x 3 x 3 convolution's last temporal tap as an im2col matrix
        (cout, dy * 3 * cin + dx * cin + ci), its input channels zero-padded to cin_padded."""
        wt = np.asarray(w[f"{path}.weight"], np.float32)[:, :, 2]  # (cout, cin, 3, 3)
        co, ci = wt.shape[:2]
        full = np.zeros((co, cin_padded, 3, 3), np.float32)
        full[:, :ci] = wt
        self._param(f"{name}.w", full.transpose(0, 2, 3, 1).reshape(co, -1))
        self._param(f"{name}.b", np.asarray(w[f"{path}.bias"], np.float32))

    # --- the operations on grids (inside body) ---

    def new_grid(self, g: GridShape):
        s = iron.state((g.rows, g.C))
        self.states.append(s)
        return s

    @staticmethod
    def view(s):
        """A state as a handle the operators take: a view of all of it."""
        return s[0 : s.shape[0]]

    def interior_from(self, src, g: GridShape, dst, offset_rows: int = 0, rows: int | None = None) -> None:
        """Copy the interior of ``rows`` image rows from ``src`` (a grid's band rows, or a chunk of
        them, ``(rows * Wp, C)``) into the grid ``dst`` from image row ``offset_rows``."""
        R = g.H if rows is None else rows
        C, Wp = g.C, g.Wp
        Copy.step(src, dst,
                  src=tap((R * Wp, C), C, [R, g.W * C], [Wp * C, 1]),
                  dst=tap((g.rows, C), g.pixel(offset_rows, 0) * C, [R, g.W * C], [Wp * C, 1]))

    def conv3(self, x, g: GridShape, wname: str, cout: int, out=None):
        """The 3 x 3 convolution of the grid x (channels g.C) into a new grid (or out)."""
        gout = GridShape(g.H, g.W, cout)
        out = self.new_grid(gout) if out is None else out
        R = g.chunk(g.C)
        Wp, C = g.Wp, g.C
        for r0 in range(0, g.H, R):
            n = R * Wp
            cols = Copy.step(x, src=tap((g.rows, C), (g.band + r0 * Wp - Wp - 1) * C,
                                        [R, Wp, 3, 3 * C], [Wp * C, C, Wp * C, 1])).reshape(n, 9 * C)
            y = F.linear(cols, self.w(f"{wname}.w"), self.w(f"{wname}.b"), precision=PREC.vae)
            self.interior_from(y, gout, out, r0, R)
        return self.view(out), gout

    def conv1(self, x, g: GridShape, wname: str, cout: int):
        """A 1 x 1 convolution (with bias) of the grid x into a new grid of cout channels."""
        gout = GridShape(g.H, g.W, cout)
        out = self.new_grid(gout)
        R = g.chunk(g.C)
        for r0 in range(0, g.H, R):
            s = g.band + r0 * g.Wp
            y = F.linear(x[s : s + R * g.Wp], self.w(f"{wname}.w"), self.w(f"{wname}.b"), precision=PREC.vae)
            self.interior_from(y, gout, out, r0, R)
        return self.view(out), gout

    def norm_silu(self, x, g: GridShape, gamma: str):
        # RMS_norm: x / ||x|| * sqrt(C) * gamma = RMSNorm without eps (zeros stay zero)
        return F.silu(F.rms_norm(x, (g.C,), self.w(gamma), 1e-12))

    def res(self, x, g: GridShape, path: str, cout: int):
        h = x
        if g.C != cout:
            h, _ = self.conv1(x, g, f"{path}.sc", cout)
        a, ga = self.conv3(self.norm_silu(x, g, f"{path}.norm1"), g, f"{path}.conv1", cout)
        b, gb = self.conv3(self.norm_silu(a, ga, f"{path}.norm2"), ga, f"{path}.conv2", cout)
        return b + h, gb

    def attn(self, x, g: GridShape, path: str):
        n, C, Wp = g.H * g.W, g.C, g.Wp
        dense = Copy.step(x, src=tap((g.rows, C), g.pixel(0, 0) * C, [g.H, g.W * C], [Wp * C, 1])).reshape(n, C)
        h = F.rms_norm(dense, (C,), self.w(f"{path}.norm"), 1e-12)
        q, k, v = (F.linear(h, self.w(f"{path}.{c}.w"), self.w(f"{path}.{c}.b"), precision=PREC.vae) for c in "qkv")
        o = MHA.step(q.reshape(n, 1, C), k.reshape(n, 1, C), v.reshape(n, 1, C), heads_interleaved=True,
                     kv_interleaved=True, causal=False, d=C, num_pipelines=pipelines(n)).reshape(n, C)
        o = dense + F.linear(o, self.w(f"{path}.proj.w"), self.w(f"{path}.proj.b"), precision=PREC.vae)
        out = self.new_grid(g)
        Copy.step(o, out, dst=tap((g.rows, C), g.pixel(0, 0) * C, [g.H, g.W * C], [Wp * C, 1]))
        return self.view(out), g

    def up(self, x, g: GridShape, path: str, cout: int):
        """Nearest 2x (each pixel to a 2 x 2 block), then the 3 x 3 convolution to cout."""
        gu = GridShape(2 * g.H, 2 * g.W, g.C)
        u = self.new_grid(gu)
        C = g.C
        for dy in range(2):
            for dx in range(2):
                Copy.step(x, u,
                          src=tap((g.rows, C), g.pixel(0, 0) * C, [g.H, g.W, C], [g.Wp * C, C, 1]),
                          dst=tap((gu.rows, C), gu.pixel(dy, dx) * C, [g.H, g.W, C], [2 * gu.Wp * C, 2 * C, 1]))
        return self.conv3(self.view(u), gu, f"{path}.up", cout)

    def body(self, x):
        g = self.g_in
        for op in self.ops:
            kind = op[0]
            if kind == "in":
                a, ga = self.conv1(x, g, "in", 64)
                x, g = self.conv3(a, ga, "conv_in", 384)
            elif kind == "res":
                x, g = self.res(x, g, op[1], op[3])
            elif kind == "attn":
                x, g = self.attn(x, g, op[1])
            elif kind == "up":
                x, g = self.up(x, g, op[1], op[3])
            elif kind == "head":
                h = self.norm_silu(x, g, "head.norm")
                y, gy = self.conv3(h, g, "head", 64)
                n = g.H * g.W
                return Copy.step(y, src=tap((gy.rows, 64), gy.pixel(0, 0) * 64, [g.H, g.W * 64], [gy.Wp * 64, 1])).reshape(n, 64)
        self.g_out = g
        # a state's view is no output of the graph's own: return a copy (a residual sum is one)
        return x if self.ops[-1][0] == "res" else Copy.step(x)


def vae_stages(width: int, height: int) -> list[tuple[list, GridShape]]:
    """The decoder in stages each under amdxdna's 2 s at 1024 x 1024, with each one's input grid."""
    h, w = height // 8, width // 8
    d = "decoder"
    s = [
        ([("in",), ("res", f"{d}.mid_block.resnets.0", 384, 384), ("attn", f"{d}.mid_block.attentions.0"),
          ("res", f"{d}.mid_block.resnets.1", 384, 384)], GridShape(h, w, 64)),
        ([("res", f"{d}.up_blocks.0.resnets.{i}", 384, 384) for i in range(3)] + [("up", f"{d}.up_blocks.0", 384, 192)],
         GridShape(h, w, 384)),
        ([("res", f"{d}.up_blocks.1.resnets.0", 192, 384)] + [("res", f"{d}.up_blocks.1.resnets.{i}", 384, 384) for i in (1, 2)],
         GridShape(2 * h, 2 * w, 192)),
        ([("up", f"{d}.up_blocks.1", 384, 192), ("res", f"{d}.up_blocks.2.resnets.0", 192, 192)], GridShape(2 * h, 2 * w, 384)),
        ([("res", f"{d}.up_blocks.2.resnets.{i}", 192, 192) for i in (1, 2)], GridShape(4 * h, 4 * w, 192)),
        ([("up", f"{d}.up_blocks.2", 192, 96)], GridShape(4 * h, 4 * w, 192)),
    ]
    s += [([("res", f"{d}.up_blocks.3.resnets.{i}", 96, 96)], GridShape(8 * h, 8 * w, 96)) for i in range(3)]
    s.append(([("head",)], GridShape(8 * h, 8 * w, 96)))
    return s


def export_vae(root: Path, out: Path, width: int, height: int, check: bool, only: list | None = None) -> None:
    w = Safetensors(root / "diffusers" / "vae" / "diffusion_pytorch_model.safetensors") if not ARGS.random else \
        RandomWeights(root / "diffusers" / "vae" / "diffusion_pytorch_model.safetensors")
    cfg = json.loads((root / "diffusers" / "vae" / "config.json").read_text())
    stages = vae_stages(width, height)
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
        meta = dict(stage=i, ops=[list(o) for o in ops], grid_in=dataclasses.asdict(g),
                    grid_out=None if go is None else dataclasses.asdict(go), pad=VAE_PAD,
                    width=width, height=height, latents_mean=cfg["latents_mean"], latents_std=cfg["latents_std"])
        m = export_graph(cg, d / f"s{i}", f"vae_s{i}", meta)
        print(f"vae s{i}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            x = np.zeros((g.rows, g.C), np.float32)
            for y in range(g.H):
                x[g.pixel(y, 0) : g.pixel(y, 0) + g.W] = rng.standard_normal((g.W, g.C))
            if ops[0][0] == "in":
                x[:, 16:] = 0
            write_check(st, m, d / f"s{i}", dict(x=bf(x)))
    (d / "manifest.json").write_text(json.dumps(dict(stages=len(stages), width=width, height=height, pad=VAE_PAD,
                                                     latents_mean=cfg["latents_mean"], latents_std=cfg["latents_std"]), indent=1) + "\n")


# --- export ------------------------------------------------------------------------------


def variant_bank(compiled, other, out: Path, name: str) -> dict:
    """The bank images of ``compiled`` with ``other``'s weights in place of its own: ``other``
    is the same graph class built with another group's weights, recorded at the same shapes;
    weights are matched by their attribute paths."""
    rec = other.record(**{h.name: (tuple(h.shape), h.dtype) for h in compiled.recording.inputs})
    theirs = {h.name: t for t, h in rec.weights.values()}
    seq = compiled.sequence
    from iron_export import _place

    images = {}
    for tensor, handle in compiled.recording.weights.values():
        kind, offset, length = _place(seq, handle.ref())
        if kind not in images:
            size = next(a["bytes"] for a in seq_arguments(seq) if a["kind"] == kind)
            images[kind] = np.lib.format.open_memmap(out / f"{name}_{kind}.npy.tmp", mode="w+", dtype=np.uint8, shape=(size,))
        t = theirs[handle.name]
        b = np.ascontiguousarray(np.asarray(t).astype(np.dtype(seq.buffer_dtype(handle.ref())))).view(np.uint8).reshape(-1)
        assert b.size <= length, (handle.name, b.size, length)
        images[kind][offset : offset + b.size] = b
    names = {}
    for kind, img in images.items():
        img.flush()
        fn = f"{name}_{kind}.bin"
        with open(out / fn, "wb") as f:
            f.write(memoryview(img))
        del img
        (out / f"{name}_{kind}.npy.tmp").unlink()
        trim_zeros(out / fn)
        names[kind] = fn
    return names


def seq_arguments(seq) -> list:
    return [dict(kind=k, bytes=int(n)) for k, n in seq.buffer_sizes.arguments().items()]


def timestep_sinusoid(t: float, dim: int) -> np.ndarray:
    """diffusers' Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0) of one timestep."""
    half = dim // 2
    f = np.exp(-math.log(10000) * np.arange(half, dtype=np.float64) / half)
    a = t * f
    return np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)


def export_dit(root: Path, out: Path, c: DitConfig, check: bool) -> None:
    w = weights(root / "diffusion_models" / ARGS.dit, PREFIX)
    rng = np.random.default_rng(0)
    meta = dict(config=dataclasses.asdict(c))
    # dit_in
    t0 = time.time()
    g = DitIn(c, w)
    cg = g.compile(patches=(c.tokens, c.patch_dim), tproj=(c.hidden,))
    m = export_graph(cg, out / "dit_in", "dit_in", meta)
    print(f"dit_in: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    if check:
        write_check(g, m, out / "dit_in", dict(patches=bf(rng.standard_normal((c.tokens, c.patch_dim))),
                                               tproj=bf(timestep_sinusoid(0.7, c.hidden))))
    # dit_out
    t0 = time.time()
    g = DitOut(c, w)
    cg = g.compile(x=(c.tokens, c.hidden), temb=(3 * c.hidden,), et=(c.hidden,))
    m = export_graph(cg, out / "dit_out", "dit_out", meta)
    print(f"dit_out: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    if check:
        write_check(g, m, out / "dit_out", dict(x=bf(rng.standard_normal((c.tokens, c.hidden))),
                                                temb=bf(rng.standard_normal(3 * c.hidden) * 0.5),
                                                et=bf(rng.standard_normal(c.hidden))))
    # dit_blocks: one ELF, a bank per group
    t0 = time.time()
    g = DitBlocks(c, 0, w)
    shapes = dict(x=(c.tokens, c.hidden), temb=(3 * c.hidden,), et=(c.hidden,), ctx=(c.text_tokens, c.text_dim))
    cg = g.compile(**shapes)
    d = out / "dit_blocks"
    groups = c.layers // c.group
    m = export_graph(cg, d, "dit_blocks", dict(meta, groups=groups))
    print(f"dit_blocks: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    variants = [{k: b["image"] for k, b in m["banks"].items()}]
    for i in range(1, groups):
        variants.append(variant_bank(cg, DitBlocks(c, i * c.group, w), d, f"bank_g{i}"))
        print(f"dit_blocks: bank of blocks {i * c.group}..{(i + 1) * c.group - 1}", flush=True)
    m["variants"] = variants
    (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    if check:
        write_check(g, m, d, dict(x=bf(rng.standard_normal((c.tokens, c.hidden))),
                                  temb=bf(rng.standard_normal(3 * c.hidden) * 0.5),
                                  et=bf(rng.standard_normal(c.hidden)),
                                  ctx=bf(rng.standard_normal((c.text_tokens, c.text_dim)))))


ARGS = None


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the Anima directory (diffusion_models/, text_encoders/, vae/, diffusers/)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--dit", default="anima-turbo-v1.1.safetensors")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=["text", "dit", "vae"], help="parts to export: text, dit, vae")
    ap.add_argument("--stages", type=int, nargs="*", help="the VAE's stages to export (default all)")
    ap.add_argument("--text-precision", default=PREC.text.value, choices=[p.value for p in Precision])
    ap.add_argument("--dit-precision", default=PREC.dit.value, choices=[p.value for p in Precision])
    ap.add_argument("--vae-precision", default=PREC.vae.value, choices=[p.value for p in Precision])
    ap.add_argument("--random", action="store_true", help="weights drawn at random in the checkpoint's shapes")
    ap.add_argument("--check", action="store_true", help="write each graph's inputs and reference outputs")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    PREC.text, PREC.dit, PREC.vae = Precision(a.text_precision), Precision(a.dit_precision), Precision(a.vae_precision)
    bind_device(a.device, a.cols)
    c = DitConfig(width=a.width or a.size, height=a.height or a.size, group=a.group)
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
