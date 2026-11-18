#!/usr/bin/env python3
"""FLUX.3 [klein] (black-forest-labs/flux-3-image-klein, FLUX Kommunity License) on IRON: the graphs
for AMD's NPU (text to image).

The DiT (Flux3Transformer2DModel, 5.5B, 3072 wide, 24 heads of 128, no biases) is 5 mode blocks of
each stream (the text's and the image's weights of their own, each stream attending to itself alone),
then 28 single blocks over the text's and the image's rows together (one set of weights, each
stream's rows modulated by its own rows). Every block computes its attention and its SwiGLU MLP
(three times the width) from the same modulated rows and adds ``gate (attention + MLP)``. The text is
Qwen3-VL-4B's language model's hidden states after layers 4, 8, ..., 32 side by side (20480); the
prompt is padded on the right to a multiple of 80 tokens and the DiT attends to the pads (no mask).
The latents are 2 x 2 patches of the FLUX.2 VAE's 32 channels (the VAE is FLUX.2 klein's).

The rows: the text's are a slot of ``text_slot`` (512) rows, the prompt's R (80, 160, ... 480: its
tokens rounded up to 80) and pads past them; the keys of every attention are bounded to the real
ones (nk, a word of each call), so one export serves every prompt length up to 480 tokens. The
image's are its N patches padded with zeros to whole rounds of the attention's 512 rows
(``image_rows``: 4096 for every one of hwnpu's eight aspect ratios but 512 x 512's 1024; 832 x 1248
is 4056 patches), the pads no keys (image_modes' nk = N). The single blocks' rows are [the image's
padded rows, the text's slot] (attention and MLP do not care about the order: RoPE is a row's own),
each part modulated by its stream's rows; their keys are packed as [the image's N, the text's slot]
(a copy of the slot's keys to row N), so nk = N + R bounds them to the real ones.

Sizes: ``--size 1024 512 832x1248 ...`` makes one export of them all. dit_in, image_modes and
dit_out depend on the image's padded rows alone: a size of the first's (1024 x 1024's 4096) runs
the first's, another has its own in sizes/WxH. The single blocks' key copy depends on N: every
size has its own singles (sizes/WxH/singles), on the first size's banks (the same weights in the
same places). Each size has its VAE (sizes/WxH/vae).

  text        Qwen3-VL's language model, ``group`` (4) layers a bank, over the 512 rows: the
              prompt's rows causal, the pads attending to the prompt's keys only (transformers'
              mask), the angles an input (the pads' position 1: the pipeline's ids); every group's
              output is a hidden state the DiT reads
  ctx         context_embedder of the eight hidden states (h0..h7, a 2560 slice of its columns each)
  dit_in      x_embedders.image of the picture's patches
  text_modes  ``mode_group`` text mode blocks over the 512 rows (keys: the first nk)
  image_modes ``mode_group`` image mode blocks over the picture's rows
  singles     ``single_group`` single blocks over [image rows, text slot] (sizes/WxH/ for the
              second size: the blocks' graphs depend on the rows alone and share the first's banks)
  dit_out     norm_out and proj_out of the picture's rows
  vae/s*      the VAE's decoder in stages (FLUX.2 klein's)
  time.safetensors   the timesteps' embedders and the modulations' weights (bf16, diffusers' names
              under ``dit.``): the host makes a step's 14 modulation rows (flux3klein's timeembed.go)

RoPE: four axes of 32 (t, row / 2, column / 2, the text's position), theta 10000; the pairs (2i, 2i
+ 1) of a head are IRON's (i, i + 64) after iron_dit.head_perm, as FLUX.2 klein's.

The checkpoint: the repository's diffusers layout (transformer/*.safetensors), or its single-file
copy of the DiT (model.safetensors, BFL's names: read through diffusers'
convert_flux3_transformer_checkpoint_to_diffusers' renames).

  python tools/amd/iron_flux3klein.py ~/models/flux-3-image-klein OUT --size 1024 512
"""

import argparse
import dataclasses
import json
import shutil
import struct
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
import iron_dit  # noqa: E402
import iron_flux2klein as k2  # noqa: E402
from iron_anima import bf, command_boundary, variant_bank  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

ROUND = 512  # the attention's rows in whole rounds (its 8 pipelines' 64)


@dataclasses.dataclass(frozen=True)
class Flux3Config:
    width: int = 1024
    height: int = 1024
    text_slot: int = 512  # the text's rows in the graphs: a prompt of up to 480 tokens (6 x 80)
    text_pad: int = 80  # the prompt's rows are its tokens rounded up to this (pad_multiple)
    hidden: int = 3072
    heads: int = 24
    head_dim: int = 128
    mlp: int = 9216
    latent: int = 128
    context: int = 20480  # eight hidden states of 2560
    text: int = 2560
    text_layers: tuple = (4, 8, 12, 16, 20, 24, 28, 32)
    axes: tuple = (32, 32, 32, 32)
    theta: float = 10000.0
    image_rope_scale: float = 0.5
    eps: float = 1e-6
    modes: int = 5
    singles: int = 28
    mode_group: int = 1
    single_group: int = 2

    @property
    def gh(self) -> int:
        return self.height // 16

    @property
    def gw(self) -> int:
        return self.width // 16

    @property
    def tokens(self) -> int:
        return self.gh * self.gw

    @property
    def image_rows(self) -> int:
        """The image's rows in the graphs: its patches padded to whole rounds of ROUND."""
        return -(-self.tokens // ROUND) * ROUND

    @property
    def rows(self) -> int:
        """The single blocks' rows: the image's (padded), then the text's slot."""
        return self.image_rows + self.text_slot

    @property
    def max_text(self) -> int:
        """The longest prompt rows the slot holds: a multiple of text_pad."""
        return self.text_slot // self.text_pad * self.text_pad


@dataclasses.dataclass
class Formats:
    proj: str = "bfp16"  # the blocks' projections (iron_q4.bfp16_pack), as FLUX.2 klein's
    mha: str = "qkpv"


FMT = Formats()
PREC = Precision.MEDIUM


def image_positions(c: Flux3Config) -> np.ndarray:
    """The picture's rows' ids (0, row / 2, column / 2, 0) (image_rope_scale), the pads' 0."""
    pos = np.zeros((c.image_rows, 4))
    pos[: c.tokens, 1] = np.repeat(np.arange(c.gh), c.gw) * c.image_rope_scale
    pos[: c.tokens, 2] = np.tile(np.arange(c.gw), c.gh) * c.image_rope_scale
    return pos


def text_positions(c: Flux3Config) -> np.ndarray:
    """The text slot's rows' ids (0, 0, 0, i)."""
    pos = np.zeros((c.text_slot, 4))
    pos[:, 3] = np.arange(c.text_slot)
    return pos


def angles(c: Flux3Config, pos: np.ndarray) -> np.ndarray:
    return iron_dit.rope_angles(c, pos)


# --- the checkpoint ----------------------------------------------------------------------

# diffusers' convert_flux3_transformer_checkpoint_to_diffusers (0.43.0), name for name
_BLOCK = {"q_proj.weight": "attn.to_q.weight", "k_proj.weight": "attn.to_k.weight", "v_proj.weight": "attn.to_v.weight",
          "attn_out.weight": "attn.to_out.0.weight", "norm.query_norm.scale": "attn.norm_q.weight",
          "norm.key_norm.scale": "attn.norm_k.weight", "mlp_in.weight": "ff.linear_in.weight",
          "mlp_out.weight": "ff.linear_out.weight"}
_PREFIX = {"emb_in.": "x_embedders.", "txt_in.": "context_embedder.", "content_mode_blocks.": "mode_blocks.",
           "single_blocks.": "single_transformer_blocks."}
_EXACT = {"time_in.in_layer.weight": "timestep_embedder.linear_1.weight",
          "time_in.out_layer.weight": "timestep_embedder.linear_2.weight",
          "ref_time_in.in_layer.weight": "target_timestep_embedder.linear_1.weight",
          "ref_time_in.out_layer.weight": "target_timestep_embedder.linear_2.weight",
          "final_layer.image.linear.weight": "proj_out.weight",
          "final_layer.image.adaLN_modulation.1.weight": "norm_out.linear.weight"}


def diffusers_name(key: str) -> str:
    """The diffusers name of a key of BFL's single-file DiT."""
    if key in _EXACT:
        return _EXACT[key]
    for old, new in _PREFIX.items():
        if key.startswith(old):
            key = new + key[len(old):]
            break
    key = key.replace(".lin.weight", ".linear.weight")
    for old, new in _BLOCK.items():
        if key.endswith(old):
            return key[: -len(old)] + new
    return key


class SingleFileDit:
    """BFL's single-file DiT (model.safetensors) under diffusers' names: norm_out's rows swapped
    to (scale, shift) (BFL's adaLN gives shift, scale)."""

    def __init__(self, path: Path, random: bool = False):
        cls = iron_anima.RandomWeights if random else iron_anima.Safetensors
        self.f = cls(path)
        self.names = {diffusers_name(k): k for k in self.f.header if not k.startswith("repa_projectors.")}

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __getitem__(self, name: str) -> np.ndarray:
        w = self.f[self.names[name]]
        if name == "norm_out.linear.weight":
            h = w.shape[0] // 2
            w = np.concatenate([np.asarray(w[h:], np.float32), np.asarray(w[:h], np.float32)])
        return w

    def keys(self):
        return self.names.keys()


def dit_weights(root: Path):
    """The DiT's tensors by diffusers' name: transformer/'s diffusers files, else model.safetensors."""
    d = root / "transformer"
    if any(d.glob("*.safetensors")):
        return iron_dit.Checkpoint(d, random=ARGS.random)
    return SingleFileDit(root / "model.safetensors", random=ARGS.random)


class TextWeights:
    """Qwen3-VL's language model under the names iron_flux2klein.Qwen3Layer reads (model.layers.*)."""

    def __init__(self, root: Path):
        self.w = iron_dit.Checkpoint(root / "text_encoder", random=ARGS.random)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.w["model.language_model." + name.removeprefix("model.")]


# --- the DiT -----------------------------------------------------------------------------


def stream(c: Flux3Config, w, p: str) -> k2._Stream:
    """A mode or single block's projections (they are named alike)."""
    f = lambda n: np.asarray(w[p + n], np.float32)  # noqa: E731
    fin = f("ff.linear_in.weight")  # (2 mlp, hidden): the gate's rows, then the value's
    M = c.mlp
    return k2._Stream(c, f("attn.to_q.weight"), f("attn.to_k.weight"), f("attn.to_v.weight"), f("attn.norm_q.weight"),
                      f("attn.norm_k.weight"), o=f("attn.to_out.0.weight"), gate=fin[:M], up=fin[M:],
                      down=f("ff.linear_out.weight"))


def parallel(s, h, c: Flux3Config, ang, ks, vs, nk, n: int, keys=None):
    """A block's attention and MLP of the same rows h (n): their sum (attn.to_out(attention) +
    ff(h)). The keys are h's rows (or keys: runs (to, from, rows) of them packed in that order),
    the first nk of them."""
    H, D, E = c.heads, c.head_dim, c.hidden
    q, k, v = k2.qkv(s, h, c, ang)
    k, v = k.reshape(n, H, D), v.reshape(n, H, D)
    for to, at, rows in keys or ((0, 0, n),):
        ks[to : to + rows] = k[at : at + rows]
        vs[to : to + rows] = v[at : at + rows]
    o = iron_anima.dit_attention(q.reshape(n, H, D), ks[:nk], vs[:nk]).reshape(n, E)
    return F.linear(o, s.o, precision=PREC) + F.linear(k2.swiglu(s, h), s.down, precision=PREC)


class Modes(k2._Blocks):
    """``mode_group`` Flux3ModeTransformerBlocks of a stream (``txt_mode_blocks`` or
    ``mode_blocks.image``) over its ``rows`` rows x; ``mods`` (3 hidden): shift, scale, gate;
    ``angles`` (rows, head_dim); the keys the first nk rows."""

    def __init__(self, c: Flux3Config, prefix: str, rows: int, first: int = 0, w=None):
        self.c, self.rows = c, rows
        self.n = c.mode_group
        for j in range(self.n):
            self.put(j, "m", stream(c, w, f"{prefix}.{first + j}."))
        self.one = k2.vec(np.ones(c.hidden))
        H, D = c.heads, c.head_dim
        self.ks, self.vs = iron.state((rows, H, D)), iron.state((rows, H, D))

    def body(self, x, mods, angles, *, nk: Scratchpad[np.int32]):
        c = self.c
        E = c.hidden
        shift, scale, gate = (mods[r * E : (r + 1) * E] for r in range(3))
        for j in range(self.n):
            h = F.layer_norm(x, (E,), weight=scale + self.one, eps=c.eps) + shift
            x = x + parallel(self.s(j, "m"), h, c, angles, self.ks, self.vs, nk, self.rows) * gate
            command_boundary()
        return x


class Singles(k2._Blocks):
    """``single_group`` Flux3SingleTransformerBlocks over x: the image's padded rows, then the
    text's slot. ``mods`` (6 hidden): the image's shift, scale and gate, then the text's;
    ``angles`` (rows, head_dim); the keys [the image's N, the text's slot], the first nk of them
    (the image's and the prompt's)."""

    def __init__(self, c: Flux3Config, first: int = 0, w=None):
        self.c = c
        self.n = c.single_group
        for j in range(self.n):
            self.put(j, "s", stream(c, w, f"single_transformer_blocks.{first + j}."))
        self.one = k2.vec(np.ones(c.hidden))
        n, H, D = c.rows, c.heads, c.head_dim
        self.ks, self.vs = iron.state((n, H, D)), iron.state((n, H, D))
        self.h = iron.state((n, c.hidden))
        self.out = iron.state((n, c.hidden))

    def body(self, x, mods, angles, *, nk: Scratchpad[np.int32]):
        c = self.c
        E, N, P, n = c.hidden, c.tokens, c.image_rows, c.rows
        m = [mods[r * E : (r + 1) * E] for r in range(6)]
        parts = ((0, P, m[0:3]), (P, n, m[3:6]))
        keys = ((0, 0, N), (N, P, c.text_slot)) if N < P else None
        for j in range(self.n):
            for a, b, (shift, scale, _) in parts:
                self.h[a:b] = F.layer_norm(x[a:b], (E,), weight=scale + self.one, eps=c.eps) + shift
            y = parallel(self.s(j, "s"), self.h[0:n], c, angles, self.ks, self.vs, nk, n, keys)
            for a, b, (_, _, gate) in parts:
                self.out[a:b] = x[a:b] + y[a:b] * gate
            x = Copy.step(self.out[0:n])
            command_boundary()
        return x


class DitIn(iron.Graph):
    """x_embedders.image of the picture's patches."""

    def __init__(self, c: Flux3Config, w=None):
        self.w = k2.vec(w["x_embedders.image.weight"])

    def body(self, x):
        return F.linear(x, self.w, precision=PREC)


class Ctx(iron.Graph):
    """context_embedder of the text's rows: h0..h7 (rows, 2560), the hidden states after layers 4,
    8, ..., 32, each an eighth of its columns."""

    def __init__(self, c: Flux3Config, w=None):
        e = np.asarray(w["context_embedder.weight"], np.float32)
        T = c.text
        self.taps = len(c.text_layers)
        for i in range(self.taps):
            setattr(self, f"w{i}", k2.vec(e[:, i * T : (i + 1) * T]))

    def body(self, h0, h1, h2, h3, h4, h5, h6, h7):
        P = Precision.HIGHEST
        y = None
        for i, h in enumerate((h0, h1, h2, h3, h4, h5, h6, h7)):
            t = F.linear(h, getattr(self, f"w{i}"), precision=P)
            y = t if y is None else y + t
        return y


class DitOut(iron.Graph):
    """norm_out (layer norm, times 1 + scale, plus shift) and proj_out of the picture's rows."""

    def __init__(self, c: Flux3Config, w=None):
        self.c = c
        self.proj = k2.vec(w["proj_out.weight"])
        self.one = k2.vec(np.ones(c.hidden))

    def body(self, x, scale, shift):
        c = self.c
        h = F.layer_norm(x, (c.hidden,), weight=scale + self.one, eps=c.eps) + shift
        return F.linear(h, self.proj, precision=PREC)


# the host's part (flux3klein's timeTensors): diffusers' names, under "dit."
TIME_TENSORS = ["timestep_embedder.linear_1.weight", "timestep_embedder.linear_2.weight",
                "target_timestep_embedder.linear_1.weight", "target_timestep_embedder.linear_2.weight",
                "early_stream_modulations.txt.linear.weight", "early_stream_modulations.image.linear.weight",
                "single_stream_modulations.txt.linear.weight", "single_stream_modulations.image.linear.weight",
                "norm_out.linear.weight"]


def write_time(w, path: Path) -> None:
    """time.safetensors: TIME_TENSORS in bf16 as dit.NAME."""
    header, blobs, at = {}, [], 0
    for n in TIME_TENSORS:
        a = bf(np.asarray(w[n], np.float32))
        b = a.tobytes()
        header["dit." + n] = dict(dtype="BF16", shape=list(a.shape), data_offsets=[at, at + len(b)])
        blobs.append(b)
        at += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def export_dit(root: Path, out: Path, c: Flux3Config, check: bool, first: Flux3Config | None, sized: str) -> None:
    """The DiT's graphs of c's picture size. The export's first size (first None): every graph,
    the blocks' banks too. Another: in sized, its singles (on the first's banks) and, where its
    image's padded rows are not the first's, its dit_in, image_modes (on the first's banks) and
    dit_out (a size of the first's rows runs the first's)."""
    w = dit_weights(root)
    rng = np.random.default_rng(0)
    meta = dict(config=dict(dataclasses.asdict(c), image_rows=c.image_rows), weights=FMT.proj, mha=FMT.mha)
    E, N, P, T = c.hidden, c.tokens, c.image_rows, c.text_slot
    first_size = first is None
    own_rows = first_size or P != first.image_rows  # dit_in, image_modes and dit_out of its own
    at = lambda name: name if first_size else f"{sized}/{name}"  # noqa: E731
    small = [] if not own_rows else [
        ("dit_in", at("dit_in"), lambda: DitIn(c, w), dict(x=(P, c.latent)), lambda: dict(x=bf(rng.standard_normal((P, c.latent))))),
        ("dit_out", at("dit_out"), lambda: DitOut(c, w), dict(x=(P, E), scale=(E,), shift=(E,)),
         lambda: dict(x=bf(rng.standard_normal((P, E))), scale=bf(0.1 * rng.standard_normal(E)), shift=bf(0.1 * rng.standard_normal(E)))),
    ]
    if first_size:
        small.insert(0, ("ctx", "ctx", lambda: Ctx(c, w), {f"h{i}": (T, c.text) for i in range(8)},
                         lambda: {f"h{i}": bf(rng.standard_normal((T, c.text))) for i in range(8)}))
        write_time(w, out / "time.safetensors")
    for part, name, make, shapes, inputs in small:
        if part not in ARGS.parts:
            continue
        t0 = time.time()
        g = make()
        cg = g.compile(**shapes)
        m = export_graph(cg, out / name, Path(name).name, meta)
        print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            write_check(g, m, out / name, inputs())
    blocks = []
    if first_size:
        blocks.append(("text_modes", "text_modes", lambda i: Modes(c, "txt_mode_blocks", T, i, w), T, c.modes, c.mode_group, 3,
                       angles(c, text_positions(c)), 160))
    if own_rows:
        blocks.append(("image_modes", at("image_modes"), lambda i: Modes(c, "mode_blocks.image", P, i, w), P, c.modes, c.mode_group, 3,
                       angles(c, image_positions(c)), N))
    blocks.append(("singles", at("singles"), lambda i: Singles(c, i, w), c.rows, c.singles, c.single_group, 6,
                   angles(c, np.concatenate([image_positions(c), text_positions(c)])), N + 160))
    for part, name, make, rows, total, group, nmods, ang, nk in blocks:
        if part not in ARGS.parts:
            continue
        t0 = time.time()
        g = make(0)
        cg = g.compile(x=(rows, E), mods=(nmods * E,), angles=(rows, c.head_dim))
        d = out / name
        groups = total // group
        m = export_graph(cg, d, Path(name).name, dict(meta, groups=groups, group=group, rows=rows))
        print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        variants = [{k: b["image"] for k, b in m["banks"].items()}]
        shared = name != part and iron_dit.share_banks(m, d, out / part)
        if shared:
            rel = "../" * name.count("/") + f"../{part}/"
            variants = [{k: f.replace(f"../{part}/", rel) for k, f in v.items()} for v in shared]
            for b in m["banks"].values():
                if b.get("image"):
                    b["image"] = b["image"].replace(f"../{part}/", rel)
            print(f"{name}: the banks are {part}'s (the same weights in the same places): stored once", flush=True)
        for i in range(1, 1 if ARGS.one_group or shared else groups):
            variants.append(variant_bank(cg, make(i * group), d, f"bank_g{i}"))
            print(f"{name}: bank of blocks {i * group}..{(i + 1) * group - 1}", flush=True)
        m["variants"] = variants
        (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
        if check:
            mods = np.concatenate([0.1 * rng.standard_normal(E) for _ in range(nmods)])
            write_check(g, m, d, dict(x=bf(rng.standard_normal((rows, E))), mods=bf(mods), angles=bf(ang)), dict(nk=nk))


# --- the text encoder (Qwen3-VL-4B's language model) -------------------------------------


def text_config(root: Path) -> "k2.TextConfig":
    cfg = json.loads((root / "text_encoder" / "config.json").read_text())["text_config"]
    c = Flux3Config()
    return dataclasses.replace(k2.TextConfig(), rows=c.text_slot, hidden=cfg["hidden_size"], layers=c.text_layers[-1],
                               heads=cfg["num_attention_heads"], kv_heads=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
                               mlp=cfg["intermediate_size"], theta=float(cfg["rope_theta"]), eps=cfg["rms_norm_eps"],
                               vocab=cfg["vocab_size"], group=c.text_layers[0], split=1)


def text_angles(c, n: int) -> np.ndarray:
    """IRON's table of Qwen's RoPE (two halves) at the pipeline's positions: 0..n-1, then 1."""
    pos = np.where(np.arange(c.rows) < n, np.arange(c.rows), 1).astype(np.float64)
    f = 1.0 / c.theta ** (np.arange(0, c.head_dim, 2, dtype=np.float64) / c.head_dim)
    a = pos[:, None] * f[None, :]
    out = np.empty((c.rows, c.head_dim), np.float32)
    out[:, ::2] = np.cos(a)
    out[:, 1::2] = np.sin(a)
    return out


def export_text(root: Path, out: Path, check: bool) -> None:
    c = text_config(root)
    w = TextWeights(root)
    t0 = time.time()
    iron_dit.TEXT_FMT["fmt"] = ARGS.text
    iron_dit.PREC.text = Precision.HIGHEST
    groups = c.layers // c.group
    meta = dict(config=dataclasses.asdict(c), weights=ARGS.text, groups=groups,
                taps=[layer // c.group - 1 for layer in Flux3Config().text_layers])
    g = k2.Text(c, w, 0)
    cg = g.compile(x=(c.rows, c.hidden), angles=(c.rows, c.head_dim), sel=(c.rows, c.heads * c.head_dim))
    d = out / "text"
    m = export_graph(cg, d, "text", meta)
    print(f"text: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    variants = [{k: b["image"] for k, b in m["banks"].items()}]
    for i in range(1, 1 if ARGS.one_group else groups):
        variants.append(variant_bank(cg, k2.Text(c, w, i * c.group), d, f"bank_g{i}"))
        print(f"text: bank of layers {i * c.group}..{(i + 1) * c.group - 1}", flush=True)
    m["variants"] = variants
    (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    shutil.copyfile(root / "text_encoder" / "tokenizer.json", d / "tokenizer.json")
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
        write_check(g, m, d, dict(x=bf(iron_dit.f32(e[ids])), angles=bf(text_angles(c, real)), sel=bf(sel)), dict(nk=real))


# --- the export --------------------------------------------------------------------------


def size_dir(width: int, height: int) -> str:
    return f"sizes/{width}x{height}"


def write_model(root: Path, out: Path, sizes: list) -> None:
    """model.json: the distilled checkpoint's 16 flow-map steps and shift, and the sizes."""
    cfg = json.loads((root / "transformer" / "config.json").read_text())
    (out / "model.json").write_text(json.dumps(dict(
        name=root.name, steps=16, shift=6.93, flow_map=bool(cfg.get("flow_map")), time_in=cfg.get("timestep_channels", 256),
        width=sizes[0][0], height=sizes[0][1], sizes=[list(s) for s in sizes]), indent=1) + "\n")


ARGS = None


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", nargs="+", default=["1024", "512"],
                    help="the pictures' sizes: N (N x N) or WxH, multiples of 16. Several (1024 512 832x1248 ...) make one "
                         "export of them all: the first's graphs hold the banks, the others' are in sizes/WxH on them "
                         "(dit_in, image_modes and dit_out only where the image's rows padded to 512 differ from the first's)")
    ap.add_argument("--parts", nargs="*", default=["text", "ctx", "dit_in", "text_modes", "image_modes", "singles", "dit_out", "vae"])
    ap.add_argument("--stages", type=int, nargs="*", help="the VAE's stages to export (default all)")
    ap.add_argument("--proj", default=FMT.proj, choices=("bf16", "q8", "q6", "q4", "bfp16"))
    ap.add_argument("--text", default="bf16", choices=("bf16", "q8", "q6", "q4"),
                    help="Qwen3-VL's projections (bf16: the DiT attends to the pads' rows, which drift with fewer bits)")
    ap.add_argument("--mha", default=FMT.mha, choices=("qk", "qkpv", "bf16"))
    ap.add_argument("--exp", default="mid", choices=("exact", "mid", "instruction"))
    ap.add_argument("--one-group", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    FMT.proj, FMT.mha = a.proj, a.mha
    k2.FMT.proj, k2.FMT.mha = a.proj, a.mha
    k2.ARGS = argparse.Namespace(random=a.random, one_group=a.one_group, parts=a.parts)
    iron_anima.DIT_FMT.mha = a.mha
    import iron_mha

    iron_mha.EXP[0] = a.exp
    import iron_q4

    iron_q4.PREFILL["q4"] = "bf16"
    iron_q4.install_linear()
    bind_device(a.device, a.cols)
    a.out.mkdir(parents=True, exist_ok=True)
    sizes = list(dict.fromkeys(k2.parse_size(x) for x in a.size))
    if "text" in a.parts:
        export_text(a.root, a.out, a.check)
    for i, s in enumerate(sizes):
        c = Flux3Config(width=s[0], height=s[1])
        export_dit(a.root, a.out, c, a.check, None if i == 0 else Flux3Config(width=sizes[0][0], height=sizes[0][1]), size_dir(*s))
    iron_anima.VAE_TILE_N[0] = True
    import iron_chroma_parts

    iron_chroma_parts.HOST_NORM[0] = True
    if "vae" in a.parts:
        for i, (w, h) in enumerate(sizes):
            k2.export_vae(a.root, a.out, w, h, a.check and i == 0, None, a.stages, name="vae" if i == 0 else f"{size_dir(w, h)}/vae")
    write_model(a.root, a.out, sizes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
