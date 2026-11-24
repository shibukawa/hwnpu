#!/usr/bin/env python3
"""The graphs bonsaiai-dev probe times on AMD's NPU: the shapes of PROBE.md's models, built from
random weights with the operators bonsaiai's exports use, so that the numbers are the ceilings of
the production paths (not of IRON's samples, docs/amd-xdna2-bench.md).

  layers  ``--layers`` Llama 3.2 1B sized layers (hidden 2048, MLP 8192, q 2048, k and v 512:
          61 M weights a layer) on ``--rows`` rows, as the linears of a decode step (1 row), of a
          call checking drafts (2 to 8 rows) or of a prompt chunk (more). Attention over a cache
          is left out, as in PROBE.md's models. The weights by ``--fmt``:
            q4, q6, q8  iron_q4's blocks: one row QGEMV (2 cores a column), 2 to 8 rows of q6 and
                        q8 QGEMVRows (the weights streamed once for all of them), more rows the
                        prompt's dequantize and GEMM (q4: flm.DequantBFP into ProjectionGEMM)
            bfp16       packed on the host for flm.ProjectionGEMM (the image DiTs' weights,
                        iron_q4.bfp16_pack): rows only
            bf16        IRON's own F.linear
  stream  the same weights in GEMVs of one shape (8192 by 2048) of one x, back to back: no
          operator loaded again between them (the ceiling of a step as one resident array)
  attn    the image DiTs' attention (iron_anima.dit_attention: iron_mha.ImageMHA, QK^T and P V
          through bfp16, the softmax's exponential on one bf16 limb, as FLUX.2 klein's export)
          of ``--rows`` queries over as many keys, ``--heads`` heads of ``--dim``

Writes OUT/manifest.json and the rest of a graph export (iron_graph.export_graph) and OUT/check/
with random inputs (no reference: the probe times, it does not compare). Run in IRON's virtual
environment; no NPU is needed.

  python tools/amd/iron_probe.py OUT layers --fmt q4 --layers 8 --rows 1
  python tools/amd/iron_probe.py OUT attn --rows 4096 --heads 24 --dim 128
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron  # noqa: E402
import iron.functional as F  # noqa: E402
from iron import Precision, Scratchpad  # noqa: E402
import iron_anima  # noqa: E402
import iron_q4  # noqa: E402
from iron_anima import bf  # noqa: E402
from iron_graph import _raw, bind_device, export_graph  # noqa: E402

# PROBE.md's layer (internal/probe/model.go): Llama 3.2 1B's sizes.
HIDDEN, MLP, KV = 2048, 8192, 512
SHAPES = dict(q=(HIDDEN, HIDDEN), k=(KV, HIDDEN), v=(KV, HIDDEN), o=(HIDDEN, HIDDEN),
              gate=(MLP, HIDDEN), up=(MLP, HIDDEN), down=(HIDDEN, MLP))
WEIGHTS_PER_LAYER = sum(n * k for n, k in SHAPES.values())


# The bytes of the weights made, as the device reads them.
WEIGHT_BYTES = [0]


def param(rng, n: int, k: int, fmt: str):
    """A random (n, k) weight in fmt, as iron_q4.q_param (or a plain bf16 iron.param) makes it."""
    w = 0.02 * rng.standard_normal((n, k), dtype=np.float32)
    if fmt == "bf16":
        a = bf(w)
    elif fmt == "bfp16":
        a = iron_q4.bfp16_pack(w)
    else:
        a = iron_q4.quantize_fmt(w, fmt).reshape(-1)
    WEIGHT_BYTES[0] += a.nbytes
    p = iron.param(a)
    if fmt != "bf16":
        iron_q4._Q4[id(p)] = (p, n, k, fmt)
    return p


class Layers(iron.Graph):
    """``layers`` layers' linears; the MLP's gate and up multiplied (the SwiGLU's product)."""

    def __init__(self, layers: int, fmt: str, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.layers = [{name: param(rng, n, k, fmt) for name, (n, k) in SHAPES.items()} for _ in range(layers)]

    def body(self, x):
        P = Precision.MEDIUM
        h = x
        for w in self.layers:
            q = F.linear(h, w["q"], precision=P)
            k = F.linear(h, w["k"], precision=P)
            v = F.linear(h, w["v"], precision=P)
            o = F.linear(q, w["o"], precision=P)
            g = F.linear(o, w["gate"], precision=P)
            u = F.linear(o, w["up"], precision=P)
            h = F.linear(g * u, w["down"], precision=P)
        return [h, k, v]


class Stream(iron.Graph):
    """The same weights as ``layers`` layers in GEMVs of one shape (MLP by hidden) of the same x,
    one after another: the operator is not loaded again between them, so the time is the
    weights' transfer, the ceiling of a step run as one resident array (GptOssStep)."""

    def __init__(self, layers: int, fmt: str, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = -(-layers * WEIGHTS_PER_LAYER // (MLP * HIDDEN))
        self.ws = [param(rng, MLP, HIDDEN, fmt) for _ in range(n)]

    def body(self, x):
        return [F.linear(x, w, precision=Precision.MEDIUM) for w in self.ws]


class Attn(iron.Graph):
    """The image DiTs' self-attention of rows queries over rows keys (iron_flux2klein_probe's)."""

    def __init__(self, rows: int, heads: int, dim: int):
        self.n, self.h, self.d = rows, heads, dim
        self.ks, self.vs = iron.state((rows, heads, dim)), iron.state((rows, heads, dim))

    def body(self, q, kk, v, *, nk: Scratchpad[np.int32]):
        n, H, D = self.n, self.h, self.d
        self.ks[0:n] = kk.reshape(n, H, D)
        self.vs[0:n] = v.reshape(n, H, D)
        return iron_anima.dit_attention(q.reshape(n, H, D), self.ks[:nk], self.vs[:nk]).reshape(n, H * D)


def write_inputs(m: dict, out: Path, inputs: dict, values: dict) -> None:
    d = out / "check"
    d.mkdir(exist_ok=True)
    for h in m["graph"]["inputs"]:
        (d / f"in_{h['name']}.bin").write_bytes(_raw(inputs[h["name"]], h["dtype"]))
    (d / "values.json").write_text(json.dumps(values) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("part", choices=("layers", "stream", "attn"))
    ap.add_argument("--fmt", default="q4", choices=("bf16", "q8", "q6", "q4", "bfp16"))
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    import iron_mha

    iron_anima.DIT_FMT.mha = "qkpv"
    iron_mha.EXP[0] = "mid"
    iron_q4.install_linear()
    bind_device(a.device, a.cols)
    rng = np.random.default_rng(1)
    if a.part in ("layers", "stream"):
        if a.fmt == "bfp16" and a.rows < 2:
            raise SystemExit("bfp16: a weight packed for the GEMM takes rows, not one row")
        g = Layers(a.layers, a.fmt) if a.part == "layers" else Stream(a.layers, a.fmt)
        shape = (HIDDEN,) if a.rows == 1 else (a.rows, HIDDEN)
        cg = g.compile(x=shape)
        inputs, values = dict(x=bf(rng.standard_normal(shape))), {}
        weights = a.layers * WEIGHTS_PER_LAYER if a.part == "layers" else len(g.ws) * MLP * HIDDEN
        meta = dict(part=a.part, fmt=a.fmt, layers=a.layers, rows=a.rows, weights=weights,
                    weight_bytes=WEIGHT_BYTES[0],
                    flops=2 * weights * a.rows)
    else:
        n, H, D = a.rows, a.heads, a.dim
        g = Attn(n, H, D)
        cg = g.compile(q=(n, H * D), kk=(n, H * D), v=(n, H * D))
        r = lambda: bf(rng.standard_normal((n, H * D)))  # noqa: E731
        inputs, values = dict(q=r(), kk=r(), v=r()), dict(nk=n)
        meta = dict(part="attn", rows=n, heads=H, dim=D, flops=4 * n * n * H * D)
    name = a.out.name
    m = export_graph(cg, a.out, name, meta)
    write_inputs(m, a.out, inputs, values)
    print(f"{name}: {len(m['graph']['kernels'])} commands, {m['graph'].get('plan', '')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
