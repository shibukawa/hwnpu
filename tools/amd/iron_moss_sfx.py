#!/usr/bin/env python3
"""MOSS-SoundEffect v2.0 (OpenMOSS-Team/MOSS-SoundEffect-v2.0, Apache-2.0) on IRON: the graphs
bonsaiai-audio generate -m sfx runs on AMD's NPU.

  python tools/amd/iron_moss_sfx.py ~/models/MOSS-SoundEffect-v2.0 ~/models/moss-soundeffect-v2.0.amd-npu2

writes into OUT a graph (iron_graph.py) for each of internal/sfx's programs, of the same inputs
and outputs (each graph's meta "program" gives the program's shapes, which iron.GraphProgram
pads):

  text/          Qwen3-1.7B's 28 layers over the prompt's 256 rows (causal, the host's embedding
                 rows in, the last hidden state before the final norm out): --text bf16 (sfx's
                 fp16) or q8 (sfx's lut8, sfx.TextFP16's layers 0..2, 26 and 27 in bf16)
  dit-context/   the text embedder over the 512 rows of the text encoder's output
  dit-blocks-N/  the DiT's blocks N..N+4 (sfx.DitGroup) over the 1500 latent frames padded to
                 1536: x (the latents for the first group, else the residual stream divided by
                 sfx.DitScale), ctx (the context), mod (the step's modulation rows, flat); the
                 last group's head gives the velocity
  dac-N/         the DAC decoder's stages (iron_moss_dac.py)
  model.json

The DiT follows internal/sfx's programs (WanAudioModel): the residual stream divided by 4 (the
patch embedding's and the cross-attention's outputs scaled, the gates already in mod), the
queries' and keys' RMS norm over all 1536 features, RoPE on Wan's interleaved pairs (IRON's
RoPE without two_halves), the cross-attention over all 512 text rows. bf16 needs none of the fp16
programs' rescaling (maxScaled). Its GEMMs run at MEDIUM (bfp16, as Anima's DiT); the GELUs are
the exact one (closer to the checkpoint's tanh GELU than IRON's tanh kernel, docs/amd-npu-models.md
18.2). The DAC decoder's stages are iron_moss_dac.py's (its Snake by a float32 kernel: IRON has
no sin).

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
from iron.operators.mha import MHA
from iron.operators.rope import RoPE

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_anima  # noqa: E402
import iron_dit  # noqa: E402
import iron_silu  # noqa: E402
from iron_anima import bf, command_boundary  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

DIT_PREC = Precision.MEDIUM
TEXT_PREC = Precision.HIGHEST

# internal/sfx's shapes
DIM, LAYERS, HEADS, HEAD_DIM, FFN = 1536, 30, 12, 128, 8960
TEXT_DIM, TEXT_ROWS, FRAMES, LATENT = 2048, 512, 1500, 128
ROWS = 1536  # the frames padded to the GEMM's and the attention's
PROMPT_ROWS = 256
GROUP = 5
DIT_SCALE = 4.0
EPS = 1e-6
TEXT_FP16 = (0, 1, 2, 26, 27)  # sfx.TextFP16


def f32(x) -> np.ndarray:
    return np.array(x, np.float32)


def p(x):
    return iron.param(bf(x))


def rope_table(rows: int, head: int, theta: float) -> np.ndarray:
    """IRON's RoPE angles (rows, head), cos and sin interleaved, a row per position."""
    inv = theta ** (-np.arange(head // 2) * 2.0 / head)
    ang = np.arange(rows)[:, None] * inv[None, :]
    out = np.empty((rows, head), np.float32)
    out[:, 0::2], out[:, 1::2] = np.cos(ang), np.sin(ang)
    return out


def mod_rows(first: int, count: int) -> int:
    """sfx.ModRows."""
    return 6 * count + (2 if first + count == LAYERS else 0)


# --- the text encoder ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TextConfig:
    hidden: int = 2048
    layers: int = 28
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    mlp: int = 6144
    eps: float = 1e-6
    theta: float = 1e6
    split: int = 4  # layers a command runs

    @classmethod
    def of(cls, d: dict) -> "TextConfig":
        if d.get("model_type") != "qwen3":
            raise SystemExit(f"text_encoder: model_type {d.get('model_type')}: Qwen3's")
        return cls(hidden=d["hidden_size"], layers=d["num_hidden_layers"], heads=d["num_attention_heads"],
                   kv_heads=d["num_key_value_heads"], head_dim=d["head_dim"], mlp=d["intermediate_size"],
                   eps=d["rms_norm_eps"], theta=d["rope_theta"])


class TextLayer(iron.Graph):
    """A Qwen3 layer over the prompt's rows, its projections in iron_dit.TEXT_FMT."""

    def __init__(self, c: TextConfig, w, i: int, fmt: str):
        self.c = c
        iron_dit.TEXT_FMT["fmt"] = fmt
        q = f"model.layers.{i}."
        lin = lambda n: iron_dit.text_param(f32(w[q + n]))  # noqa: E731
        self.in_norm, self.post_norm = p(w[q + "input_layernorm.weight"]), p(w[q + "post_attention_layernorm.weight"])
        self.q, self.k, self.v, self.o = (lin(f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.qn, self.kn = p(w[q + "self_attn.q_norm.weight"]), p(w[q + "self_attn.k_norm.weight"])
        self.gate, self.up, self.down = (lin(f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))

    def body(self, x, angles, pads):
        c = self.c
        L, H, G, D = PROMPT_ROWS, c.heads, c.kv_heads, c.head_dim
        tl = lambda y, wt: iron_dit.text_linear(y, wt, pads)  # noqa: E731
        h = F.rms_norm(x, (c.hidden,), self.in_norm, c.eps)
        q = F.rope(F.rms_norm(tl(h, self.q).reshape(L * H, D), (D,), self.qn, c.eps), angles)
        k = F.rope(F.rms_norm(tl(h, self.k).reshape(L * G, D), (D,), self.kn, c.eps), angles)
        v = tl(h, self.v)
        o = MHA.step(q.reshape(L, H, D), k.reshape(L, G, D), v.reshape(L, G, D), heads_interleaved=True,
                     kv_interleaved=True, causal=True, d=D, num_pipelines=4).reshape(L, H * D)
        x = x + tl(o, self.o)
        h = F.rms_norm(x, (c.hidden,), self.post_norm, c.eps)
        return x + tl(iron_silu.silu(tl(h, self.gate)) * tl(h, self.up), self.down)


class Text(iron.Graph):
    """The text encoder: ``x`` (256, hidden) the prompt's embedding rows (<|endoftext|>'s after
    it); output the last layer's hidden state (the host applies the final norm)."""

    def __init__(self, c: TextConfig, w, fmt: str):
        self.c = c
        self.layers = [TextLayer(c, w, i, "bf16" if fmt == "bf16" or i in TEXT_FP16 else fmt) for i in range(c.layers)]
        self.angles = p(rope_table(PROMPT_ROWS, c.head_dim, c.theta)[:, :].reshape(PROMPT_ROWS, c.head_dim))
        # the zero-padded inputs of q8's projections whose in-features are no multiple of 512
        self.pads = {k: iron.state((PROMPT_ROWS, k)) for k in {-(-n // 512) * 512 for n in (c.hidden, c.heads * c.head_dim, c.mlp)}
                     if k not in (c.hidden, c.heads * c.head_dim, c.mlp)}

    def body(self, x):
        for i, layer in enumerate(self.layers):
            if i and i % self.c.split == 0:
                command_boundary()
            x = layer(x, self.angles, self.pads)
        return x


def hf_rope_angles(rows: int, head: int, theta: float) -> np.ndarray:
    """IRON's RoPE angles for Hugging Face's rotate_half (two_halves): the pair i is (i, i + head
    / 2) at frequency theta^(-2i / head), cos and sin interleaved."""
    return rope_table(rows, head, theta)


# --- the DiT ------------------------------------------------------------------------------


class Context(iron.Graph):
    """The text embedder: ``ctx`` (512, 2048) to (512, 1536)."""

    def __init__(self, w):
        t = "condition_embedder.text_embedder."
        self.w1, self.b1 = p(w[t + "linear_1.weight"]), p(w[t + "linear_1.bias"])
        self.w2, self.b2 = p(w[t + "linear_2.weight"]), p(w[t + "linear_2.bias"])

    def body(self, ctx):
        h = F.gelu(F.linear(ctx, self.w1, self.b1, precision=DIT_PREC))
        return F.linear(h, self.w2, self.b2, precision=DIT_PREC)


class Block(iron.Graph):
    """A WanAudioModel block on the residual stream divided by DIT_SCALE."""

    def __init__(self, w, i: int):
        q = f"blocks.{i}."
        g = lambda n: f32(w[q + n])  # noqa: E731
        for a in ("attn1", "attn2"):
            for n in "qkv":
                setattr(self, f"{a}_{n}", p(g(f"{a}.to_{n}.weight")))
                setattr(self, f"{a}_{n}b", p(g(f"{a}.to_{n}.bias")))
            setattr(self, f"{a}_nq", p(g(f"{a}.norm_q.weight")))
            setattr(self, f"{a}_nk", p(g(f"{a}.norm_k.weight")))
        self.attn1_o, self.attn1_ob = p(g("attn1.to_out.0.weight")), p(g("attn1.to_out.0.bias"))
        # the cross-attention's output divided by DIT_SCALE, as the residual stream
        self.attn2_o, self.attn2_ob = p(g("attn2.to_out.0.weight") / DIT_SCALE), p(g("attn2.to_out.0.bias") / DIT_SCALE)
        self.n2, self.n2b = p(g("norm2.weight")), p(g("norm2.bias"))
        self.f1, self.f1b = p(g("ffn.net.0.proj.weight")), p(g("ffn.net.0.proj.bias"))
        self.f2, self.f2b = p(g("ffn.net.2.weight")), p(g("ffn.net.2.bias"))

    def body(self, x, ctx, m, angles, nk, nt, kv):
        R, H, D = ROWS, HEADS, HEAD_DIM
        lin = lambda y, wt, b: F.linear(y, wt, b, precision=DIT_PREC)  # noqa: E731
        # self-attention
        h = F.layer_norm(x, (DIM,), weight=m(1), eps=EPS) + m(0)
        q = F.rms_norm(lin(h, self.attn1_q, self.attn1_qb), (DIM,), self.attn1_nq, EPS)
        k = F.rms_norm(lin(h, self.attn1_k, self.attn1_kb), (DIM,), self.attn1_nk, EPS)
        q = RoPE.step(q.reshape(R * H, D), angles, two_halves=False).reshape(R, H, D)
        k = RoPE.step(k.reshape(R * H, D), angles, two_halves=False).reshape(R, H, D)
        v = lin(h, self.attn1_v, self.attn1_vb).reshape(R, H, D)
        o = iron_anima.dit_attention(q, k[:nk], v[:nk]).reshape(R, DIM)
        x = x + lin(o, self.attn1_o, self.attn1_ob) * m(2)
        # cross-attention over the text
        h = F.layer_norm(x, (DIM,), self.n2, self.n2b, eps=EPS)
        q = F.rms_norm(lin(h, self.attn2_q, self.attn2_qb), (DIM,), self.attn2_nq, EPS).reshape(R, H, D)
        # the text's keys and values in the first rows of states of the queries' rows (MHA's
        # keys are at least its queries), bounded by nt
        ks, vs = kv
        ks[0:TEXT_ROWS] = F.rms_norm(lin(ctx, self.attn2_k, self.attn2_kb), (DIM,), self.attn2_nk, EPS).reshape(TEXT_ROWS, H, D)
        vs[0:TEXT_ROWS] = lin(ctx, self.attn2_v, self.attn2_vb).reshape(TEXT_ROWS, H, D)
        o = iron_anima.dit_attention(q, ks[:nt], vs[:nt]).reshape(R, DIM)
        x = x + lin(o, self.attn2_o, self.attn2_ob)
        # feed-forward
        h = F.layer_norm(x, (DIM,), weight=m(4), eps=EPS) + m(3)
        f = lin(F.gelu(lin(h, self.f1, self.f1b)), self.f2, self.f2b)
        return x + f * m(5)


class Blocks(iron.Graph):
    """Blocks first..first + count - 1 (sfx.Blocks): ``x`` (ROWS, 128 or 1536), ``ctx`` (512,
    1536), ``mod`` (ModRows x 1536,); per call ``nk``, the frames. Output the residual stream,
    or for the last group the velocity (ROWS, 128)."""

    def __init__(self, w, first: int, count: int):
        self.first, self.count = first, count
        self.blocks = [Block(w, first + j) for j in range(count)]
        if first == 0:
            pe = f32(w["patch_embedding.weight"]).reshape(DIM, LATENT)
            self.pe, self.peb = p(pe / DIT_SCALE), p(f32(w["patch_embedding.bias"]) / DIT_SCALE)
        if first + count == LAYERS:
            self.po, self.pob = p(w["proj_out.weight"]), p(w["proj_out.bias"])
        self.angles = p(rope_table(ROWS, HEAD_DIM, 10000.0))
        self.kv = (iron.state((ROWS, HEADS, HEAD_DIM)), iron.state((ROWS, HEADS, HEAD_DIM)))

    def body(self, x, ctx, mod, *, nk: Scratchpad[np.int32], nt: Scratchpad[np.int32]):
        E = DIM
        if self.first == 0:
            x = F.linear(x, self.pe, self.peb, precision=DIT_PREC)
        for j, b in enumerate(self.blocks):
            if j:
                command_boundary()
            x = b(x, ctx, lambda k, j=j: mod[(6 * j + k) * E : (6 * j + k + 1) * E], self.angles, nk, nt, self.kv)
        if self.first + self.count == LAYERS:
            r = 6 * self.count
            h = F.layer_norm(x, (E,), weight=mod[(r + 1) * E : (r + 2) * E], eps=EPS) + mod[r * E : (r + 1) * E]
            x = F.linear(h, self.po, self.pob, precision=DIT_PREC)
        return x


# --- the export -----------------------------------------------------------------------------


def write_safetensors(path: Path, tensors: dict) -> None:
    """tensors by name, each (array, "BF16" or "F32"), into a .safetensors file."""
    header, blobs, at = {}, [], 0
    for n, (x, dt) in tensors.items():
        b = (bf(x) if dt == "BF16" else np.ascontiguousarray(np.asarray(x, np.float32))).tobytes()
        header[n] = dict(dtype=dt, shape=list(np.shape(x)), data_offsets=[at, at + len(b)])
        blobs.append(b)
        at += len(b)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(len(raw).to_bytes(8, "little"))
        f.write(raw)
        for b in blobs:
            f.write(b)


def write_host(root: Path, out: Path, random: bool) -> None:
    """What internal/sfx's host reads besides the graphs: host/text.safetensors (the token
    embeddings and the final norm), host/timing.safetensors (sfx.HostTensors), the tokenizer and
    the text encoder's config."""
    import shutil

    d = out / "host"
    d.mkdir(exist_ok=True)
    tw = iron_dit.Checkpoint(root / "text_encoder", random=random)
    write_safetensors(d / "text.safetensors", {n: (tw[n], "BF16") for n in ("model.embed_tokens.weight", "model.norm.weight")})
    w = iron_dit.Checkpoint(root / "transformer", random=random)
    names = [n for n in w.index if n.startswith("condition_embedder.time_") or n.endswith("scale_shift_table")]
    write_safetensors(d / "timing.safetensors", {n: (w[n], "F32") for n in sorted(names)})
    # the tokenizer (made binary by the recipe's finish; internal/sfx reads no tokenizer_config.json)
    for f in ("tokenizer/tokenizer.json", "text_encoder/config.json"):
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / f, out / f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the MOSS-SoundEffect repository (text_encoder/, transformer/, tokenizer/, vae/)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--text", choices=["bf16", "q8"], default="bf16",
                    help="the text encoder's projections: bf16 (sfx's fp16) or q8 (sfx's lut8, its first three and last two layers bf16)")
    ap.add_argument("--only", nargs="*", default=["text", "dit", "dac", "host"])
    ap.add_argument("--groups", type=int, default=LAYERS // GROUP, help="the DiT's groups to export (fewer: a check)")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    bind_device(a.device, a.cols)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    if "text" in a.only:
        t0 = time.time()
        tc = TextConfig.of(json.loads((a.root / "text_encoder" / "config.json").read_text()))
        tw = iron_dit.Checkpoint(a.root / "text_encoder", random=a.random)
        g = Text(tc, tw, a.text)
        cg = g.compile(x=(PROMPT_ROWS, tc.hidden))
        meta = dict(config=dataclasses.asdict(tc), format=a.text,
                    program=dict(inputs=dict(x=dict(shape=[1, PROMPT_ROWS, tc.hidden])), output=[1, PROMPT_ROWS, tc.hidden]))
        m = export_graph(cg, a.out / "text", "text", meta)
        print(f"text: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if a.check:
            x = rng.standard_normal((PROMPT_ROWS, tc.hidden)).astype(np.float32) * 0.02
            write_check(g, m, a.out / "text", dict(x=bf(x)))
    if "dit" in a.only:
        w = iron_dit.Checkpoint(a.root / "transformer", random=a.random)
        t0 = time.time()
        g = Context(w)
        cg = g.compile(ctx=(TEXT_ROWS, TEXT_DIM))
        meta = dict(program=dict(inputs=dict(ctx=dict(shape=[1, TEXT_ROWS, TEXT_DIM])), output=[1, TEXT_ROWS, DIM]))
        m = export_graph(cg, a.out / "dit-context", "dit-context", meta)
        print(f"dit-context: compiled in {time.time() - t0:.0f} s", flush=True)
        if a.check:
            write_check(g, m, a.out / "dit-context", dict(ctx=bf(rng.standard_normal((TEXT_ROWS, TEXT_DIM)) * 0.5)))
        for gi in range(a.groups):
            first = gi * GROUP
            t0 = time.time()
            g = Blocks(w, first, GROUP)
            width = LATENT if first == 0 else DIM
            out_w = LATENT if first + GROUP == LAYERS else DIM
            nm = mod_rows(first, GROUP)
            cg = g.compile(x=(ROWS, width), ctx=(TEXT_ROWS, DIM), mod=(nm * DIM,))
            meta = dict(first=first, count=GROUP, values=dict(nk=FRAMES, nt=TEXT_ROWS),
                        program=dict(inputs=dict(x=dict(shape=[1, FRAMES, width]), ctx=dict(shape=[1, TEXT_ROWS, DIM]),
                                                 mod=dict(shape=[1, nm, DIM])), output=[1, FRAMES, out_w]))
            name = f"dit-blocks-{first}"
            m = export_graph(cg, a.out / name, name, meta)
            print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan'].splitlines()[0]}", flush=True)
            if a.check and gi == 0:
                x = np.zeros((ROWS, width), np.float32)
                x[:FRAMES] = rng.standard_normal((FRAMES, width))
                ctx = rng.standard_normal((TEXT_ROWS, DIM)).astype(np.float32) * 0.5
                mod = rng.standard_normal((nm, DIM)).astype(np.float32) * 0.1
                for j in range(GROUP):
                    mod[6 * j + 1] += 1
                    mod[6 * j + 4] += 1
                write_check(g, m, a.out / name, dict(x=bf(x), ctx=bf(ctx), mod=bf(mod.reshape(-1))), dict(nk=FRAMES, nt=TEXT_ROWS))
    if "dac" in a.only:
        import iron_moss_dac

        iron_moss_dac.export(a.root / "vae" / "vae_128d_48k.pth", a.out, list(range(len(iron_moss_dac.RATES))), a.check)
    if "host" in a.only:
        t0 = time.time()
        write_host(a.root, a.out, a.random)
        print(f"host: {time.time() - t0:.0f} s", flush=True)
    (a.out / "model.json").write_text(json.dumps(dict(name=a.root.name, kind="moss-sfx", text=a.text,
                                                      groups=LAYERS // GROUP, group=GROUP), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
