#!/usr/bin/env python3
"""Magenta RealTime 2 small's depthformer (internal/music's Temporal and Depth) on IRON: the
temporal decoder's step and the depth decoder's step as graphs.

  python tools/amd/iron_magenta_lm.py ~/models/magenta-realtime-2 OUT

writes OUT/temporal/ and OUT/depth/ (iron_graph.py):

  temporal  one frame: x (1, 1024), the mean of the previous frame's token embeddings, and enc
            (1, 256), the encoder's output of the frame; per call ``pos`` (the frame's slot of
            the rings, 1 + frame % RING) and ``n`` (1 + the frames the rings hold, at most RING).
            Each layer's self- and cross-attention keys and values are rings in the graph's
            states (s{i}k, s{i}v, c{i}k, c{i}v: row 0 the sink, which the host writes once,
            rows 1..RING the frames: the Window past and the current). Output y (1, 1024)
  depth     one codebook: x (1, 1024), the temporal output or the previous codebook's token
            embedding; per call ``pos`` (the codebook) and ``n`` (pos + 1). Its 2 layers' keys
            and values are states (d{i}k, d{i}v). Output the logits (1, 12294 padded), before
            the soft cap

The attentions have no positions, so a ring's order does not matter. The query's per-dimension
scale is folded into the query projection, and the sink key divided by it (the sink's logit
takes the unscaled query: internal/music's attend).

See docs/amd-npu-models.md.
"""

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_anima import RandomWeights, Safetensors, bf  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402
from iron_q4 import install_linear, linear_add, q_param  # noqa: E402

P = "depthformer/decoder/"
T_DIM, T_LAYERS, T_HEADS, T_FFN = 1024, 12, 8, 4096
D_DIM, D_LAYERS, D_HEADS, D_FFN = 768, 2, 6, 3072
ENC, HEAD = 256, 128
WINDOW = 41
RING = WINDOW + 1  # the past frames and the current
SLOTS = 48  # a ring's rows: the sink and RING, padded to MHA's key blocks of 16
D_SLOTS = 16  # the depth decoder's: CODEBOOKS padded
CODEBOOKS = 12
VOCAB = 12294
VOCAB_PAD = 12352  # a multiple of 64
EPS = 1e-6
FMT = "q8"
PIPES = 2  # MHA's pipelines of one query (dividing the heads: 8 and 6)


def f32(x) -> np.ndarray:
    return np.array(x, np.float32)


def weight(w):
    return iron.param(bf(w)) if FMT == "bf16" else q_param(f32(w), FMT)


def query_scale(p) -> np.ndarray:
    """per_dim_scale's scale: 1.442695041 / sqrt(128) softplus(p)."""
    return (1.442695041 / math.sqrt(HEAD) * np.log1p(np.exp(f32(p).astype(np.float64)))).astype(np.float32)


def heads_t(k) -> np.ndarray:
    """A Flax attention kernel (in, heads, 128) as (heads x 128, in)."""
    k = f32(k)
    return k.reshape(k.shape[0], -1).T


class Attention:
    """One of a layer's attentions: q (scaled), k and v (one GEMV when from the same input), the
    output projection, the norms, the sink."""

    def __init__(self, w, p: str, heads: int, sink: bool, kv_from_x: bool):
        a = p + "/attention/"
        s = query_scale(w[a + "per_dim_scale"])
        q = heads_t(w[a + "query_projection/kernel"]) * np.tile(s, heads)[:, None]
        k, v = heads_t(w[a + "key_projection/kernel"]), heads_t(w[a + "value_projection/kernel"])
        if kv_from_x:
            self.qkv = weight(np.concatenate([q, k, v]))
        else:
            self.q, self.kv = weight(q), weight(np.concatenate([k, v]))
        o = f32(w[p + "/output_projection/kernel"])  # (out, heads, 128)
        self.o = weight(o.reshape(o.shape[0], -1))
        self.pre = iron.param(bf(w[p + "/pre_norm/scale"]))
        self.post = iron.param(bf(w[p + "/post_norm/scale"]))
        self.sink = None
        if sink:
            sk = f32(w[a + "sink_key_embeddings"]).reshape(heads, HEAD)
            sv = f32(w[a + "sink_value_embeddings"]).reshape(heads, HEAD)
            self.sink = (sk / s[None, :], sv)


def rms(x, wt, n: int):
    return F.rms_norm(x, (n,), wt, eps=EPS)


class FFN:
    def __init__(self, w, p: str):
        self.pre, self.post = iron.param(bf(w[p + "ffn/pre_norm/scale"])), iron.param(bf(w[p + "ffn/post_norm/scale"]))
        self.w1, self.b1 = weight(f32(w[p + "ffn/ffn_layer1/kernel"]).T), iron.param(bf(w[p + "ffn/ffn_layer1/bias"]))
        self.w2, self.b2 = weight(f32(w[p + "ffn/ffn_layer2/kernel"]).T), iron.param(bf(w[p + "ffn/ffn_layer2/bias"]))

    def __call__(self, x, n: int):
        h = F.gelu(linear_add(rms(x, self.pre, n), self.w1, self.b1), approximate="tanh")
        y = linear_add(h, self.w2, self.b2).reshape(1, n)
        return x + rms(y, self.post, n)


def attend(q, keys, values, n, heads: int):
    return MHA.step(q, keys[:n], values[:n], heads_interleaved=True, kv_interleaved=True, causal=False, d=HEAD,
                    scale=1.0, num_pipelines=PIPES)


class TemporalLayer(iron.Graph):
    def __init__(self, w, i: int):
        p = f"{P}temporal_body/transformer/x_layers_{i}/"
        self.s = Attention(w, p + "self_attention", T_HEADS, True, True)
        self.c = Attention(w, p + "cross_attention", T_HEADS, True, False)
        self.ffn = FFN(w, p)
        H = T_HEADS
        self.sk, self.sv = iron.state((SLOTS, H, HEAD), name=f"s{i}k"), iron.state((SLOTS, H, HEAD), name=f"s{i}v")
        self.ck, self.cv = iron.state((SLOTS, H, HEAD), name=f"c{i}k"), iron.state((SLOTS, H, HEAD), name=f"c{i}v")

    def body(self, x, enc, pos, n):
        H, E = T_HEADS, T_DIM
        a = self.s
        qkv = F.linear(rms(x, a.pre, E), a.qkv, precision=Precision.MEDIUM).reshape(3, H, HEAD)
        self.sk[pos] = qkv[1]
        self.sv[pos] = qkv[2]
        o = attend(qkv[0:1], self.sk, self.sv, n, H)
        x = x + rms(F.linear(o.reshape(1, E), a.o, precision=Precision.MEDIUM), a.post, E)
        c = self.c
        q = F.linear(rms(x, c.pre, E), c.q, precision=Precision.MEDIUM).reshape(1, H, HEAD)
        kv = F.linear(enc, c.kv, precision=Precision.MEDIUM).reshape(2, H, HEAD)
        self.ck[pos] = kv[0]
        self.cv[pos] = kv[1]
        o = attend(q, self.ck, self.cv, n, H)
        x = x + rms(F.linear(o.reshape(1, E), c.o, precision=Precision.MEDIUM), c.post, E)
        return self.ffn(x, E)


class Temporal(iron.Graph):
    def __init__(self, w, layers: int = T_LAYERS):
        self.layers = [TemporalLayer(w, i) for i in range(layers)]

    def body(self, x, enc, *, pos: Scratchpad[np.int32], n: Scratchpad[np.int32]):
        for layer in self.layers:
            x = layer(x, enc, pos, n)
        return x


class DepthLayer(iron.Graph):
    def __init__(self, w, i: int):
        p = f"{P}depth_body/transformer/x_layers_{i}/"
        self.s = Attention(w, p + "self_attention", D_HEADS, False, True)
        self.ffn = FFN(w, p)
        self.k = iron.state((D_SLOTS, D_HEADS, HEAD), name=f"d{i}k")
        self.v = iron.state((D_SLOTS, D_HEADS, HEAD), name=f"d{i}v")

    def body(self, x, pos, n):
        H, E = D_HEADS, D_DIM
        a = self.s
        qkv = F.linear(rms(x, a.pre, E), a.qkv, precision=Precision.MEDIUM).reshape(3, H, HEAD)
        self.k[pos] = qkv[1]
        self.v[pos] = qkv[2]
        o = attend(qkv[0:1], self.k, self.v, n, H)
        x = x + rms(F.linear(o.reshape(1, H * HEAD), a.o, precision=Precision.MEDIUM), a.post, E)
        return self.ffn(x, E)


class Depth(iron.Graph):
    def __init__(self, w):
        p0 = P + "depth_body/"
        self.adapter = weight(f32(w[p0 + "depth_input_adapter/kernel"]).T)
        self.layers = [DepthLayer(w, i) for i in range(D_LAYERS)]
        self.ln_w, self.ln_b = iron.param(bf(w[p0 + "final_ln/scale"])), iron.param(bf(w[p0 + "final_ln/bias"]))
        lw = np.zeros((VOCAB_PAD, D_DIM), np.float32)
        lw[:VOCAB] = f32(w[p0 + "to_logits/kernel"]).T
        lb = np.zeros(VOCAB_PAD, np.float32)
        lb[:VOCAB] = f32(w[p0 + "to_logits/bias"])
        self.logits, self.logits_b = weight(lw), iron.param(bf(lb))

    def body(self, x, *, pos: Scratchpad[np.int32], n: Scratchpad[np.int32]):
        h = F.linear(x, self.adapter, precision=Precision.MEDIUM)
        for layer in self.layers:
            h = layer(h, pos, n)
        h = F.layer_norm(h, (D_DIM,), self.ln_w, self.ln_b, eps=EPS)
        return linear_add(h, self.logits, self.logits_b).reshape(1, VOCAB_PAD)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the magenta-realtime-2 repository (checkpoints/mrt2_small.safetensors)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", nargs="*", default=["temporal", "depth"])
    ap.add_argument("--layers", type=int, default=T_LAYERS, help="the temporal decoder's first N layers (a check)")
    ap.add_argument("--weights", choices=["q8", "bf16"], default="q8")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    global FMT
    FMT = a.weights
    install_linear()
    bind_device(a.device, a.cols)
    w = (RandomWeights if a.random else Safetensors)(a.root / "checkpoints" / "mrt2_small.safetensors", prefix="params/")
    rng = np.random.default_rng(0)
    meta = dict(weights=FMT, ring=RING, window=WINDOW, vocab=VOCAB, vocab_pad=VOCAB_PAD)
    if "temporal" in a.only:
        t0 = time.time()
        g = Temporal(w, a.layers)
        cg = g.compile(x=(1, T_DIM), enc=(1, ENC))
        sinks = {}
        for i, layer in enumerate(g.layers):
            for kind, at in (("s", layer.s), ("c", layer.c)):
                sinks[f"{kind}{i}"] = [bf(at.sink[0]).view(np.uint16).tolist(), bf(at.sink[1]).view(np.uint16).tolist()]
        m = export_graph(cg, a.out / "temporal", "temporal", dict(meta, layers=a.layers))
        (a.out / "temporal" / "sinks.json").write_text(json.dumps(sinks) + "\n")
        print(f"temporal: compiled in {time.time() - t0:.0f} s; {m['graph']['plan'].splitlines()[0]}", flush=True)
        if a.check:
            write_check(g, m, a.out / "temporal", dict(x=bf(rng.standard_normal((1, T_DIM))), enc=bf(rng.standard_normal((1, ENC)))),
                        dict(pos=1, n=2))
    if "depth" in a.only:
        t0 = time.time()
        g = Depth(w)
        cg = g.compile(x=(1, T_DIM))
        m = export_graph(cg, a.out / "depth", "depth", meta)
        print(f"depth: compiled in {time.time() - t0:.0f} s; {m['graph']['plan'].splitlines()[0]}", flush=True)
        if a.check:
            write_check(g, m, a.out / "depth", dict(x=bf(rng.standard_normal((1, T_DIM)))), dict(pos=0, n=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
