#!/usr/bin/env python3
"""Chroma1 (lodestones/Chroma1-Flash, Chroma1-HD; FLUX.1 schnell's DiT, Apache-2.0) on IRON: the
DiT's graphs for AMD's NPU.

Chroma is FLUX.1 schnell's DiT (19 double-stream blocks: the text's and the image's rows with
weights of their own, attending together; 38 single-stream blocks: the rows together, attention
and a tanh-GELU MLP side by side, one projection out; 3072 wide, 24 heads of 128) with the blocks'
adaLN modulations (3.3B) replaced by one FFN, the approximator: from the timestep alone it gives
344 modulation rows (3 a single block, 6 + 6 a double block, 2 the final norm). The text is
T5-XXL's 512 rows, the pads masked past the prompt and one pad token.

  mods        the approximator over the 344 rows of the timestep's input (padded to 512): its
              input rows are the timestep's sinusoid, a zero guidance's and each row's own index's
              (the host builds them)
  dit_in      x_embedder of the latents' 2 x 2 patches
  ctx         context_embedder of T5's rows (once a prompt)
  doubles     ``double_group`` double blocks over [image rows, text rows] (one ELF, a bank per
              group): the image's rows come first, so the text's pads are cut by the keys' count
              ``nk`` (the image's and the prompt's rows), as the pipeline's mask cuts them
  singles     ``single_group`` single blocks over the same rows
  dit_out     norm_out (its shift and scale rows) and proj_out of the image's rows

RoPE: FLUX turns pairs (2i, 2i + 1) of a head; the q and k rows are permuted within each head so
IRON's pairs (i, i + 64) are them (iron_qwenimage.head_perm). The text's positions are 0 (no turn).

  python tools/amd/iron_chroma.py ~/models/Chroma1-Flash OUT --size 1024
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
import iron_qwenimage as qi  # noqa: E402
from iron_anima import bf, command_boundary, variant_bank  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402


@dataclasses.dataclass(frozen=True)
class ChromaConfig:
    width: int = 1024
    height: int = 1024
    text_rows: int = 512  # T5's rows (max_sequence_length)
    hidden: int = 3072
    heads: int = 24
    head_dim: int = 128
    mlp: int = 12288
    latent: int = 64  # a 2 x 2 patch of the VAE's 16 channels
    t5: int = 4096
    axes: tuple = (16, 56, 56)
    theta: float = 10000.0
    doubles: int = 19
    singles: int = 38
    double_group: int = 1  # 19 is prime: one double block a bank
    single_group: int = 2
    mods: int = 344
    mod_rows: int = 512  # the approximator's rows a call (mods padded)
    approx_in: int = 64
    approx_hidden: int = 5120
    approx_layers: int = 5

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
    def rows(self) -> int:
        return self.tokens + self.text_rows

    # the approximator's rows: singles 3 each, then the doubles' image 6 each and text 6 each,
    # then the final norm's shift and scale
    def single_mod(self, i: int) -> int:
        return 3 * i

    def double_mod(self, i: int, text: bool) -> int:
        return 3 * self.singles + (6 * self.doubles if text else 0) + 6 * i


@dataclasses.dataclass
class Formats:
    proj: str = "q8"  # the blocks' projections (a distilled DiT is weak at 4 bits)
    mha: str = "qk"


FMT = Formats()
PREC = Precision.MEDIUM


class _Cfg:
    """The DitConfig fields iron_qwenimage's RoPE helpers read."""

    def __init__(self, c: ChromaConfig):
        self.axes, self.theta, self.head_dim, self.heads = c.axes, c.theta, c.head_dim, c.heads


def angles(c: ChromaConfig) -> np.ndarray:
    """The rows' RoPE: the image's (0, row, column) first, then the text's (0, 0, 0)."""
    pos = np.zeros((c.rows, 3))
    pos[: c.tokens, 1] = np.repeat(np.arange(c.gh), c.gw)
    pos[: c.tokens, 2] = np.tile(np.arange(c.gw), c.gh)
    return qi.rope_angles(_Cfg(c), pos)


def perm(c: ChromaConfig) -> np.ndarray:
    return qi.head_perm(_Cfg(c))


def lin(w, fmt: str):
    return qi.qparam(np.asarray(w, np.float32), fmt)


def vec(w):
    return iron.param(bf(np.asarray(w, np.float32)))


def attention(q, k, v):
    return iron_anima.dit_attention(q, k, v)


class _Stream:
    """One stream's projections (a double block's text or image, or a single block's q, k, v)."""

    def __init__(self, c: ChromaConfig, w, p: str, names: dict):
        P = perm(c)
        f = lambda n: np.asarray(w[p + n], np.float32)  # noqa: E731
        for a in ("q", "k", "v"):
            wt, bs = f(names[a] + ".weight"), f(names[a] + ".bias")
            if a != "v":
                wt, bs = wt[P], bs[P]
            setattr(self, a, lin(wt, FMT.proj))
            setattr(self, a + "_b", vec(bs))
        within = P[: c.head_dim]
        self.qn = vec(f(names["qn"])[within])
        self.kn = vec(f(names["kn"])[within])
        for a in ("o", "ff1", "ff2"):
            if a in names:
                setattr(self, a, lin(f(names[a] + ".weight"), FMT.proj))
                setattr(self, a + "_b", vec(f(names[a] + ".bias")))


IMG = dict(q="attn.to_q", k="attn.to_k", v="attn.to_v", qn="attn.norm_q.weight", kn="attn.norm_k.weight",
           o="attn.to_out.0", ff1="ff.net.0.proj", ff2="ff.net.2")
TXT = dict(q="attn.add_q_proj", k="attn.add_k_proj", v="attn.add_v_proj", qn="attn.norm_added_q.weight",
           kn="attn.norm_added_k.weight", o="attn.to_add_out", ff1="ff_context.net.0.proj", ff2="ff_context.net.2")


def qkv(s, h, c: ChromaConfig, ang):
    """A stream's q and k (normed, turned) and v of h's rows."""
    n, H, D = h.shape[0], c.heads, c.head_dim
    q = F.rms_norm(F.linear(h, s.q, s.q_b, precision=PREC).reshape(n * H, D), (D,), s.qn, 1e-6)
    k = F.rms_norm(F.linear(h, s.k, s.k_b, precision=PREC).reshape(n * H, D), (D,), s.kn, 1e-6)
    v = F.linear(h, s.v, s.v_b, precision=PREC)
    return F.rope(q, ang), F.rope(k, ang), v


class Doubles(iron.Graph):
    """``double_group`` ChromaTransformerBlocks over ``x`` (rows: the image's, then the text's).
    Inputs: ``x``; ``mods`` (group x 12, hidden), each block's image modulation rows (shift, scale,
    gate, shift, scale, gate) then the text's; per call ``nk``, the keys attended to."""

    def __init__(self, c: ChromaConfig, first: int = 0, w=None):
        self.c = c
        self.blocks = []
        for j in range(c.double_group):
            p = f"transformer_blocks.{first + j}."
            for tag, names in (("t", TXT), ("i", IMG)):
                for k, v in vars(_Stream(c, w, p, names)).items():
                    setattr(self, f"b{j}_{tag}_{k}", v)
            self.blocks.append(j)
        self.one = vec(np.ones(c.hidden))
        n, H, D = c.rows, c.heads, c.head_dim
        self.qs, self.ks, self.vs = iron.state((n * H, D)), iron.state((n, H, D)), iron.state((n, H, D))
        self.angles = vec(angles(c))
        self.out = iron.state((n, c.hidden))

    def s(self, j, tag):
        class S:
            pass

        o = S()
        for k in ("q", "q_b", "k", "k_b", "v", "v_b", "qn", "kn", "o", "o_b", "ff1", "ff1_b", "ff2", "ff2_b"):
            setattr(o, k, getattr(self, f"b{j}_{tag}_{k}"))
        return o

    def body(self, x, mods, *, nk: Scratchpad[np.int32]):
        c = self.c
        E, H, D, N, T = c.hidden, c.heads, c.head_dim, c.tokens, c.text_rows
        n = c.rows
        for j in self.blocks:
            parts = {"i": (0, N), "t": (N, n)}
            m = {}
            for tag, (a, b) in parts.items():
                base = 12 * j + (6 if tag == "t" else 0)
                m[tag] = [mods[(base + r) * E : (base + r + 1) * E] for r in range(6)]
                shift, scale = m[tag][0], m[tag][1]
                h = F.layer_norm(x[a:b], (E,), eps=1e-6) * (scale + self.one) + shift
                q, k, v = qkv(self.s(j, tag), h, c, self.angles[a:b])
                self.qs[a * H : b * H] = q
                self.ks[a:b] = k.reshape(b - a, H, D)
                self.vs[a:b] = v.reshape(b - a, H, D)
            o = attention(self.qs.reshape(n, H, D), self.ks[:nk], self.vs[:nk]).reshape(n, E)
            for tag, (a, b) in parts.items():
                s = self.s(j, tag)
                _, _, g1, shift2, scale2, g2 = m[tag]
                y = x[a:b] + F.linear(o[a:b], s.o, s.o_b, precision=PREC) * g1
                h = F.layer_norm(y, (E,), eps=1e-6) * (scale2 + self.one) + shift2
                h = F.gelu(F.linear(h, s.ff1, s.ff1_b, precision=PREC), approximate="tanh")
                self.out[a:b] = y + F.linear(h, s.ff2, s.ff2_b, precision=PREC) * g2
            x = Copy.step(self.out[0:n])
            command_boundary()
        return x


class Singles(iron.Graph):
    """``single_group`` ChromaSingleTransformerBlocks over ``x``; ``mods`` (group x 3, hidden),
    each block's shift, scale and gate; per call ``nk``."""

    def __init__(self, c: ChromaConfig, first: int = 0, w=None):
        self.c = c
        E, M = c.hidden, c.mlp
        self.n = c.single_group
        for j in range(self.n):
            p = f"single_transformer_blocks.{first + j}."
            f = lambda n: np.asarray(w[p + n], np.float32)  # noqa: E731
            st = _Stream(c, w, p, dict(q="attn.to_q", k="attn.to_k", v="attn.to_v", qn="attn.norm_q.weight",
                                       kn="attn.norm_k.weight"))
            for k, v in vars(st).items():
                setattr(self, f"b{j}_{k}", v)
            setattr(self, f"b{j}_mlp", lin(f("proj_mlp.weight"), FMT.proj))
            setattr(self, f"b{j}_mlp_b", vec(f("proj_mlp.bias")))
            po = f("proj_out.weight")  # (E, E + M) over [attention, MLP]
            setattr(self, f"b{j}_oa", lin(po[:, :E], FMT.proj))
            setattr(self, f"b{j}_om", lin(po[:, E:], FMT.proj))
            setattr(self, f"b{j}_o_b", vec(f("proj_out.bias")))
        self.one = vec(np.ones(E))
        self.angles = vec(angles(c))
        n, H, D = c.rows, c.heads, c.head_dim
        self.ks, self.vs = iron.state((n, H, D)), iron.state((n, H, D))

    def body(self, x, mods, *, nk: Scratchpad[np.int32]):
        c = self.c
        E, H, D, n = c.hidden, c.heads, c.head_dim, c.rows
        for j in range(self.n):
            g = lambda k: getattr(self, f"b{j}_{k}")  # noqa: E731
            shift, scale, gate = (mods[(3 * j + r) * E : (3 * j + r + 1) * E] for r in range(3))
            h = F.layer_norm(x, (E,), eps=1e-6) * (scale + self.one) + shift

            class S:
                pass

            s = S()
            for k in ("q", "q_b", "k", "k_b", "v", "v_b", "qn", "kn"):
                setattr(s, k, g(k))
            q, k, v = qkv(s, h, c, self.angles)
            self.ks[0:n] = k.reshape(n, H, D)
            self.vs[0:n] = v.reshape(n, H, D)
            o = attention(q.reshape(n, H, D), self.ks[:nk], self.vs[:nk]).reshape(n, E)
            mlp = F.gelu(F.linear(h, g("mlp"), g("mlp_b"), precision=PREC), approximate="tanh")
            y = F.linear(o, g("oa"), precision=PREC) + F.linear(mlp, g("om"), g("o_b"), precision=PREC)
            x = x + y * gate
            command_boundary()
        return x


class Mods(iron.Graph):
    """The approximator over ``x`` (mod_rows, 64): in_proj, five residual SiLU MLPs after RMS
    norms, out_proj: the modulation rows (mod_rows, hidden)."""

    def __init__(self, c: ChromaConfig, w=None):
        self.c = c
        p = "distilled_guidance_layer."
        f = lambda n: np.asarray(w[p + n], np.float32)  # noqa: E731
        self.inp, self.inp_b = vec(f("in_proj.weight")), vec(f("in_proj.bias"))
        # each RMS norm's weight folded into the columns of the linear after it (IRON's weighted
        # RMSNorm takes rows of 4096 at most)
        self.l1 = [vec(f(f"layers.{i}.linear_1.weight") * f(f"norms.{i}.weight")[None, :]) for i in range(c.approx_layers)]
        self.b1 = [vec(f(f"layers.{i}.linear_1.bias")) for i in range(c.approx_layers)]
        self.l2 = [vec(f(f"layers.{i}.linear_2.weight")) for i in range(c.approx_layers)]
        self.b2 = [vec(f(f"layers.{i}.linear_2.bias")) for i in range(c.approx_layers)]
        self.out, self.out_b = vec(f("out_proj.weight")), vec(f("out_proj.bias"))

    def body(self, x):
        c = self.c
        P = Precision.HIGH  # HIGHEST fails to compile at K = 5120 (its buffer descriptors)
        h = F.linear(x, self.inp, self.inp_b, precision=P)
        for i in range(c.approx_layers):
            t = F.rms_norm(h, (c.approx_hidden,), None, 1.1920929e-07)  # nn.RMSNorm's eps: float32's
            t = F.linear(F.silu(F.linear(t, self.l1[i], self.b1[i], precision=P)), self.l2[i], self.b2[i], precision=P)
            h = h + t
            command_boundary()  # a layer a command: together they exhaust a shim tile's 16 buffer descriptors
        return F.linear(h, self.out, self.out_b, precision=P)


class Embed(iron.Graph):
    """A linear with bias (x_embedder, context_embedder)."""

    def __init__(self, w, name: str):
        self.w, self.b = vec(w[name + ".weight"]), vec(w[name + ".bias"])

    def body(self, x):
        return F.linear(x, self.w, self.b, precision=PREC)


class DitOut(iron.Graph):
    """norm_out (layer norm, scale and shift rows) and proj_out of the image's rows."""

    def __init__(self, c: ChromaConfig, w=None):
        self.c = c
        self.proj, self.proj_b = vec(w["proj_out.weight"]), vec(w["proj_out.bias"])
        self.one = vec(np.ones(c.hidden))

    def body(self, x, shift, scale):
        c = self.c
        h = F.layer_norm(x, (c.hidden,), eps=1e-6) * (scale + self.one) + shift
        return F.linear(h, self.proj, self.proj_b, precision=PREC)


def export_dit(root: Path, out: Path, c: ChromaConfig, check: bool) -> None:
    w = qi.Checkpoint(root / "transformer", random=ARGS.random)
    rng = np.random.default_rng(0)
    meta = dict(config=dataclasses.asdict(c), weights=FMT.proj, mha=FMT.mha)
    E = c.hidden
    small = [
        ("mods", Mods(c, w), dict(x=(c.mod_rows, c.approx_in)), lambda: dict(x=bf(rng.standard_normal((c.mod_rows, c.approx_in))))),
        ("dit_in", Embed(w, "x_embedder"), dict(x=(c.tokens, c.latent)), lambda: dict(x=bf(rng.standard_normal((c.tokens, c.latent))))),
        ("ctx", Embed(w, "context_embedder"), dict(x=(c.text_rows, c.t5)), lambda: dict(x=bf(rng.standard_normal((c.text_rows, c.t5)) * 0.1))),
        ("dit_out", DitOut(c, w), dict(x=(c.tokens, E), shift=(E,), scale=(E,)),
         lambda: dict(x=bf(rng.standard_normal((c.tokens, E))), shift=bf(0.1 * rng.standard_normal(E)), scale=bf(0.1 * rng.standard_normal(E)))),
    ]
    for name, g, shapes, inputs in small:
        if name not in ARGS.parts:
            continue
        t0 = time.time()
        cg = g.compile(**shapes)
        m = export_graph(cg, out / name, name, meta)
        print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if check:
            write_check(g, m, out / name, inputs())
    for name, cls, total, group, nmods in (("doubles", Doubles, c.doubles, c.double_group, 12), ("singles", Singles, c.singles, c.single_group, 3)):
        if name not in ARGS.parts:
            continue
        t0 = time.time()
        g = cls(c, 0, w)
        shapes = dict(x=(c.rows, E), mods=(group * nmods * E,))
        cg = g.compile(**shapes)
        d = out / name
        groups = total // group
        m = export_graph(cg, d, name, dict(meta, groups=groups, group=group))
        print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        variants = [{k: b["image"] for k, b in m["banks"].items()}]
        for i in range(1, 1 if ARGS.one_group else groups):
            variants.append(variant_bank(cg, cls(c, i * group, w), d, f"bank_g{i}"))
            print(f"{name}: bank of blocks {i * group}..{(i + 1) * group - 1}", flush=True)
        m["variants"] = variants
        (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
        if check:
            mods = np.concatenate([0.1 * rng.standard_normal(E) for _ in range(group * nmods)])
            write_check(g, m, d, dict(x=bf(rng.standard_normal((c.rows, E))), mods=bf(mods)), dict(nk=c.tokens + 13))


ARGS = None


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--parts", nargs="*", default=["mods", "dit_in", "ctx", "dit_out", "doubles", "singles", "t5", "vae"],
                    help="graphs to export: the DiT's, and t5 and vae (tools/amd/iron_chroma_parts.py, run after the DiT)")
    ap.add_argument("--proj", default=FMT.proj, choices=("bf16", "q8", "q6", "q4"))
    ap.add_argument("--one-group", action="store_true")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    FMT.proj = a.proj
    import iron_q4

    iron_q4.PREFILL["q4"] = "bf16"
    iron_q4.install_linear()
    bind_device(a.device, a.cols)
    c = ChromaConfig(width=a.size, height=a.size)
    a.out.mkdir(parents=True, exist_ok=True)
    export_dit(a.root, a.out, c, a.check)
    write_model(a.root, a.out)
    rest = [p for p in ("t5", "vae") if p in a.parts]
    if rest:  # in a process of its own: the parts' module sets IRON's formats and precisions its way
        import subprocess

        cmd = [sys.executable, str(Path(__file__).with_name("iron_chroma_parts.py")), str(a.root), str(a.out),
               "--size", str(a.size), "--only", *rest, "--device", a.device, "--cols", str(a.cols)]
        if a.check:
            cmd.append("--check")
        subprocess.run(cmd, check=True)
    return 0


def write_model(root: Path, out: Path) -> None:
    """model.json: the checkpoint and its scheduler, which pick the defaults. Chroma1-Flash: Heun's
    8 steps without guidance. Chroma1-HD: guidance 3 (its card's) with DPM-Solver++ 2M 25 steps,
    guided at sigma >= 0.6 only: against 100 Euler steps the latent's cosine 0.962 (the card's 40
    Euler steps 0.972, the same picture) in 45 % less time (design.md 16.157)."""
    sched = json.loads((root / "scheduler" / "scheduler_config.json").read_text())
    flash = "flash" in root.name.lower()
    (out / "model.json").write_text(json.dumps(dict(
        name=root.name, scheduler=sched, steps=8 if flash else 25, guidance=1.0 if flash else 3.0,
        sampler="heun" if flash else "dpm2m", cfg_sigmas=None if flash else [0.6, 1.0]), indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
