#!/usr/bin/env python3
"""Where a FLUX.2 klein single block's time goes: the block and its parts as graphs of their own,
timed with tools/amd/irongraph (design.md 16.200's way).

  full    a single block (iron_flux2klein.Singles with one block)
  gemm    its seven projections alone (q, k, v, gate, up of the normed rows; the two halves of
          the output projection)
  attn    the attention alone, of given q, k and v
  elem    the elementwise operators alone: the modulated LayerNorm, the head norms, RoPE, SiLU,
          the SwiGLU product, the sum of the output's halves, the gate and the residual

The weights are the checkpoint's (bfp16 as the export's, by default). A graph's check holds its
inputs and zeros for the outputs (a bfp16 weight has no reference): irongraph prints the times
and fails the comparison.

  python tools/amd/iron_flux2klein_probe.py ~/models/FLUX.2-klein-4B OUT --size 1024 --refs 0
  go run ./tools/amd/irongraph -runs 3 OUT/full
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_flux2klein as k  # noqa: E402
import iron  # noqa: E402
import iron.functional as F  # noqa: E402
from iron import Scratchpad  # noqa: E402
import iron_anima  # noqa: E402
import iron_headnorm  # noqa: E402
import iron_qwenimage as qi  # noqa: E402
from iron_anima import bf  # noqa: E402
from iron_graph import _raw, bind_device, export_graph  # noqa: E402


def write_inputs(m: dict, out: Path, inputs: dict, values: dict | None = None) -> None:
    d = out / "check"
    d.mkdir(exist_ok=True)
    for h in m["graph"]["inputs"]:
        (d / f"in_{h['name']}.bin").write_bytes(_raw(inputs[h["name"]], h["dtype"]))
    (d / "values.json").write_text(json.dumps(values or {}) + "\n")
    for h in m["graph"]["outputs"]:
        (d / f"out_{h['name']}.bin").write_bytes(bytes(h["bytes"]))


class Gemm(k._Blocks):
    def __init__(self, c, w):
        self.c = c
        self.put(0, "s", k.single_stream(c, w, "single_transformer_blocks.0."))

    def body(self, h, o, f):
        s = self.s(0, "s")
        P = k.PREC
        q, kk, v = F.linear(h, s.q, precision=P), F.linear(h, s.k, precision=P), F.linear(h, s.v, precision=P)
        g, u = F.linear(h, s.gate, precision=P), F.linear(h, s.up, precision=P)
        return [q, kk, v, g, u, F.linear(o, s.oa, precision=P), F.linear(f, s.om, precision=P)]


class Attn(iron.Graph):
    def __init__(self, c):
        self.c = c
        n, H, D = c.rows, c.heads, c.head_dim
        self.ks, self.vs = iron.state((n, H, D)), iron.state((n, H, D))

    def body(self, q, kk, v, *, nk: Scratchpad[np.int32]):
        c = self.c
        n, H, D = c.rows, c.heads, c.head_dim
        self.ks[0:n] = kk.reshape(n, H, D)
        self.vs[0:n] = v.reshape(n, H, D)
        return iron_anima.dit_attention(q.reshape(n, H, D), self.ks[:nk], self.vs[:nk]).reshape(n, c.hidden)


class Elem(iron.Graph):
    def __init__(self, c, w):
        self.c = c
        f = lambda n: np.asarray(w["single_transformer_blocks.0." + n], np.float32)  # noqa: E731
        within = qi.head_perm(c)[: c.head_dim]
        self.qn = iron_headnorm.head_weight(f("attn.norm_q.weight")[within], c.heads)
        self.kn = iron_headnorm.head_weight(f("attn.norm_k.weight")[within], c.heads)
        self.one = k.vec(np.ones(c.hidden))

    def body(self, x, mods, angles, q, kk, g, u, oa, om):
        c = self.c
        E, H, D, n = c.hidden, c.heads, c.head_dim, c.rows
        shift, scale, gate = (mods[r * E : (r + 1) * E] for r in range(3))
        h = F.layer_norm(x, (E,), weight=scale + self.one, eps=c.eps) + shift
        qr = F.rope(iron_headnorm.rms_norm(q, self.qn, c.eps, D).reshape(n * H, D), angles)
        kr = F.rope(iron_headnorm.rms_norm(kk, self.kn, c.eps, D).reshape(n * H, D), angles)
        f = F.silu(g) * u
        return [h, qr, kr, f, x + (oa + om) * gate]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--refs", type=int, default=0)
    ap.add_argument("--proj", default="bfp16", choices=("bf16", "q8", "bfp16"))
    ap.add_argument("--mha", default="qk", choices=("qk", "qkpv", "bf16"))
    ap.add_argument("--parts", nargs="*", default=["full", "gemm", "attn", "elem"])
    ap.add_argument("--exp", default="exact", choices=("exact", "mid", "instruction"),
                    help="the attention's softmax exponentials (iron_mha.ImageMHA.exp)")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    k.FMT.proj = a.proj
    iron_anima.DIT_FMT.mha = a.mha
    import iron_mha

    iron_mha.EXP[0] = a.exp
    import iron_q4

    iron_q4.PREFILL["q4"] = "bf16"
    iron_q4.install_linear()
    bind_device(a.device, a.cols)
    c = k.KleinConfig(width=a.size, height=a.size, refs=a.refs, single_group=1)
    w = qi.Checkpoint(a.root / "transformer")
    rng = np.random.default_rng(0)
    n, E, H, D, M = c.rows, c.hidden, c.heads, c.head_dim, c.mlp
    r = lambda *s: bf(rng.standard_normal(s))  # noqa: E731
    mods = bf(0.1 * rng.standard_normal(3 * E))
    ang = bf(k.angles(c))
    graphs = {
        "full": (lambda: k.Singles(c, 0, w), dict(x=(n, E), mods=(3 * E,), angles=(n, D)),
                 lambda: dict(x=r(n, E), mods=mods, angles=ang), dict(nk=n)),
        "gemm": (lambda: Gemm(c, w), dict(h=(n, E), o=(n, E), f=(n, M)), lambda: dict(h=r(n, E), o=r(n, E), f=r(n, M)), {}),
        "attn": (lambda: Attn(c), dict(q=(n, H * D), kk=(n, H * D), v=(n, H * D)),
                 lambda: dict(q=r(n, H * D), kk=r(n, H * D), v=r(n, H * D)), dict(nk=n)),
        "elem": (lambda: Elem(c, w), dict(x=(n, E), mods=(3 * E,), angles=(n, D), q=(n, E), kk=(n, E), g=(n, M), u=(n, M),
                                          oa=(n, E), om=(n, E)),
                 lambda: dict(x=r(n, E), mods=mods, angles=ang, q=r(n, E), kk=r(n, E), g=r(n, M), u=r(n, M), oa=r(n, E),
                              om=r(n, E)), {}),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.parts:
        make, shapes, inputs, values = graphs[name]
        g = make()
        cg = g.compile(**shapes)
        m = export_graph(cg, a.out / name, name, dict(config=dict(rows=n), weights=a.proj, mha=a.mha))
        write_inputs(m, a.out / name, inputs(), values)
        print(f"{name}: {m['graph']['plan']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
