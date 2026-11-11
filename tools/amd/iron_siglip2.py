#!/usr/bin/env python3
"""SigLIP 2 (google/siglip2-base-patch16-224 and the other fixed-size checkpoints, Apache-2.0)
on IRON: the graphs hwnpu-image label and embed run on AMD's NPU.

SigLIP 2's two towers map a picture and a text into one space of embeddings; their cosine,
scaled and shifted by the model's two constants, is the logit of "the text describes the
picture" (a sigmoid, each label on its own: zero-shot labels). Two graphs, each exported by
iron_graph.py:

  vision  the patch embedding (the host gives each 16 x 16 patch's 3 x 16 x 16 values), the
          encoder's blocks (layer norm, attention of heads of 64, layer norm, the tanh GELU's
          MLP), the last layer norm and the attention pooling head (a learned probe's query over
          the tokens, its output projection, a layer norm and an MLP added): rows of the
          picture's embedding, its row 0 the one
  text    LABELS texts of 64 tokens at once (the host looks up the token embedding rows: the
          table is 256000 x hidden), the encoder's blocks (the same block; the attention over a
          text's own 64 tokens, no mask), the last layer norm and the head (a linear): row
          63 x LABELS + i is text i's embedding

The text graph's rows are token-major (row t x LABELS + i is token t of text i), so a block's
attention is one call with LABELS x heads heads of 64 tokens: the rows of a token's texts are
heads beside each other. The host normalizes the embeddings and takes the logits.

  python tools/amd/iron_siglip2.py ~/models/siglip2-base-patch16-224 ~/models/siglip2-base-patch16-224.amd-npu2

The export also holds the tokenizer and the token embedding table (text/embed.bin, bf16) the host
reads. See docs/amd-npu-models.md.
"""

import argparse
import dataclasses
import json
import math
import shutil
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
from iron_anima import RandomWeights, Safetensors, bf, command_boundary  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

# The GEMMs' precision: HIGHEST (bf16 activations) holds the embeddings at a cosine of 0.9998
# against Transformers' float32 where MEDIUM (bfp16, 8 values a shared exponent) gave 0.9988, for
# 9 % more time in the vision tower and 22 % in the text tower (docs/amd-npu-models.md 18)
PREC = Precision.HIGHEST
# The MLPs' GELU: SigLIP's is the tanh approximation, whose IRON kernel (a table for tanh) is off
# by 3 times bf16's rounding; the exact one's (erf) is off by about its rounding, the difference
# of the two functions included ("erf", the default; "tanh": the checkpoint's own)
GELU = "erf"
TEXT_LEN = 64  # SigLIP's texts: padded to 64 tokens, as it was trained


def f32(x) -> np.ndarray:
    return np.array(x, np.float32)


@dataclasses.dataclass(frozen=True)
class Tower:
    hidden: int = 768
    mlp: int = 3072
    layers: int = 12
    heads: int = 12
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads


@dataclasses.dataclass(frozen=True)
class Config:
    vision: Tower = Tower()
    text: Tower = Tower()
    image: int = 224
    patch: int = 16
    vocab: int = 256000
    projection: int = 768
    labels: int = 8  # texts a call of the text graph takes
    split: int = 6  # blocks a command runs

    @property
    def grid(self) -> int:
        return self.image // self.patch

    @property
    def patches(self) -> int:
        return self.grid * self.grid

    @property
    def patch_in(self) -> int:
        """A patch's 3 x P x P values, padded to a multiple of 64."""
        return -(-3 * self.patch * self.patch // 64) * 64

    @property
    def tokens(self) -> int:
        """The vision graph's rows: the patches, padded to the GEMM's 256."""
        return -(-self.patches // 256) * 256

    @property
    def text_rows(self) -> int:
        return TEXT_LEN * self.labels

    @classmethod
    def of(cls, root: Path, labels: int) -> "Config":
        cfg = json.loads((root / "config.json").read_text())
        if cfg.get("model_type") != "siglip":
            raise SystemExit(f"{root}: a SigLIP 2 of a fixed size (model_type siglip) is supported, not {cfg.get('model_type')}")
        vc, tc = cfg.get("vision_config", {}), cfg.get("text_config", {})

        def tower(d):
            return Tower(hidden=d.get("hidden_size", 768), mlp=d.get("intermediate_size", 3072),
                         layers=d.get("num_hidden_layers", 12), heads=d.get("num_attention_heads", 12),
                         eps=d.get("layer_norm_eps", 1e-6))

        v, t = tower(vc), tower(tc)
        for n, x, d in (("vision", v, vc), ("text", t, tc)):
            if x.hidden % x.heads or x.head_dim % 64:
                raise SystemExit(f"{root}: the {n} tower's heads of {x.hidden / x.heads:g}: IRON's MHA takes multiples of 64")
            if d.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
                raise SystemExit(f"{root}: the {n} tower's {d['hidden_act']}: only gelu_pytorch_tanh")
        if tc.get("max_position_embeddings", 64) != TEXT_LEN:
            raise SystemExit(f"{root}: texts of {tc['max_position_embeddings']} tokens: only {TEXT_LEN}")
        if labels % 4:
            raise SystemExit(f"--labels {labels}: a multiple of 4 (64 x labels rows, multiples of 256)")
        return cls(vision=v, text=t, image=vc.get("image_size", 224), patch=vc.get("patch_size", 16),
                   vocab=tc.get("vocab_size", 32000), projection=tc.get("projection_size") or t.hidden, labels=labels)


def pipelines(rows: int) -> int:
    """MHA's pipelines for ``rows`` queries: 64 rows each, up to 8 (above 6 an even count)."""
    for p in (8, 6, 4, 2, 1):
        if rows % (64 * p) == 0:
            return p
    raise ValueError(rows)


class Block(iron.Graph):
    """A SigLIP encoder layer: layer norm, attention (q, k, v and the output projection with
    biases), added; layer norm, fc1, the tanh GELU, fc2, added."""

    def __init__(self, t: Tower, w, p: str):
        self.t = t
        g = lambda n: f32(w[p + n])  # noqa: E731
        self.n1, self.n1b = iron.param(bf(g("layer_norm1.weight"))), iron.param(bf(g("layer_norm1.bias")))
        self.n2, self.n2b = iron.param(bf(g("layer_norm2.weight"))), iron.param(bf(g("layer_norm2.bias")))
        for n in "qkv":
            setattr(self, n, iron.param(bf(g(f"self_attn.{n}_proj.weight"))))
            setattr(self, n + "b", iron.param(bf(g(f"self_attn.{n}_proj.bias"))))
        self.o, self.ob = iron.param(bf(g("self_attn.out_proj.weight"))), iron.param(bf(g("self_attn.out_proj.bias")))
        self.f1, self.f1b = iron.param(bf(g("mlp.fc1.weight"))), iron.param(bf(g("mlp.fc1.bias")))
        self.f2, self.f2b = iron.param(bf(g("mlp.fc2.weight"))), iron.param(bf(g("mlp.fc2.bias")))

    def body(self, x, attend):
        """``attend(q, k, v)``: the attention of the rows' q, k, v (rows, hidden), (rows, hidden)."""
        t = self.t
        E = t.hidden
        h = F.layer_norm(x, (E,), self.n1, self.n1b, eps=t.eps)
        q = F.linear(h, self.q, self.qb, precision=PREC)
        k = F.linear(h, self.k, self.kb, precision=PREC)
        v = F.linear(h, self.v, self.vb, precision=PREC)
        x = x + F.linear(attend(q, k, v), self.o, self.ob, precision=PREC)
        h = F.layer_norm(x, (E,), self.n2, self.n2b, eps=t.eps)
        h = F.linear(gelu(F.linear(h, self.f1, self.f1b, precision=PREC)), self.f2, self.f2b, precision=PREC)
        return x + h


def gelu(x):
    return F.gelu(x, approximate="tanh" if GELU == "tanh" else "none")


class Vision(iron.Graph):
    """The vision tower and its pooling head.

    Input ``patches`` (tokens, patch_in): each patch's 3 x P x P values (channel, row, column;
    scaled to [-1, 1]) in raster order, zeros past them. Per call: ``nk``, the rows that are
    patches. Output (tokens, hidden): every row the pooled embedding (row 0 the one; the probe's
    query is on every row so that the head's GEMMs take whole rows).
    """

    def __init__(self, c: Config, w):
        v = c.vision
        E, T = v.hidden, c.tokens
        self.c = c
        p = "vision_model."
        self.blocks = [Block(v, w, f"{p}encoder.layers.{i}.") for i in range(v.layers)]
        pw = f32(w[p + "embeddings.patch_embedding.weight"]).reshape(E, -1)
        full = np.zeros((E, c.patch_in), np.float32)
        full[:, : pw.shape[1]] = pw
        self.pe = iron.param(bf(full))
        pos = np.zeros((T, E), np.float32)
        pe = f32(w[p + "embeddings.position_embedding.weight"])
        if pe.shape[0] != c.patches:
            raise SystemExit(f"{pe.shape[0]} positions for {c.patches} patches")
        pos[: c.patches] = pe + f32(w[p + "embeddings.patch_embedding.bias"])
        self.pos = iron.param(bf(pos))
        self.post, self.postb = iron.param(bf(w[p + "post_layernorm.weight"])), iron.param(bf(w[p + "post_layernorm.bias"]))
        # the pooling head: torch's MultiheadAttention (in_proj q, k, v) with the probe as its query
        h = p + "head."
        ip, ipb = f32(w[h + "attention.in_proj_weight"]), f32(w[h + "attention.in_proj_bias"])
        probe = f32(w[h + "probe"]).reshape(1, E)
        q = probe @ ip[:E].T + ipb[:E]
        self.hq = iron.param(bf(np.repeat(q, T, axis=0)))
        self.hk, self.hkb = iron.param(bf(ip[E : 2 * E])), iron.param(bf(ipb[E : 2 * E]))
        self.hv, self.hvb = iron.param(bf(ip[2 * E :])), iron.param(bf(ipb[2 * E :]))
        self.ho, self.hob = iron.param(bf(w[h + "attention.out_proj.weight"])), iron.param(bf(w[h + "attention.out_proj.bias"]))
        self.hn, self.hnb = iron.param(bf(w[h + "layernorm.weight"])), iron.param(bf(w[h + "layernorm.bias"]))
        self.h1, self.h1b = iron.param(bf(w[h + "mlp.fc1.weight"])), iron.param(bf(w[h + "mlp.fc1.bias"]))
        self.h2, self.h2b = iron.param(bf(w[h + "mlp.fc2.weight"])), iron.param(bf(w[h + "mlp.fc2.bias"]))

    def body(self, patches, *, nk: Scratchpad[np.int32]):
        c = self.c
        v = c.vision
        E, H, D, T = v.hidden, v.heads, v.head_dim, c.tokens
        npipe = pipelines(T)

        def attend(q, k, vv):
            q, k, vv = q.reshape(T, H, D), k.reshape(T, H, D), vv.reshape(T, H, D)
            o = MHA.step(q, k[:nk], vv[:nk], heads_interleaved=True, kv_interleaved=True, causal=False, d=D,
                         num_pipelines=npipe)
            return o.reshape(T, E)

        x = F.linear(patches, self.pe, precision=PREC) + self.pos
        for i, b in enumerate(self.blocks):
            if i and i % c.split == 0:
                command_boundary()
            x = b(x, attend)
        x = F.layer_norm(x, (E,), self.post, self.postb, eps=v.eps)
        k = F.linear(x, self.hk, self.hkb, precision=PREC)
        vv = F.linear(x, self.hv, self.hvb, precision=PREC)
        o = MHA.step(self.hq.reshape(T, H, D), k.reshape(T, H, D)[:nk], vv.reshape(T, H, D)[:nk],
                     heads_interleaved=True, kv_interleaved=True, causal=False, d=D, num_pipelines=npipe)
        x = F.linear(o.reshape(T, E), self.ho, self.hob, precision=PREC)
        h = F.layer_norm(x, (E,), self.hn, self.hnb, eps=v.eps)
        h = F.linear(gelu(F.linear(h, self.h1, self.h1b, precision=PREC)), self.h2, self.h2b, precision=PREC)
        return x + h


class Text(iron.Graph):
    """The text tower on ``labels`` texts.

    Input ``x`` (64 labels, hidden): the token embedding rows, token-major (row t labels + i:
    text i's token t; a text is its tokens, <eos>, then <pad> to 64). Output (64 labels,
    projection): the head of the last layer norm; row 63 labels + i is text i's embedding.
    """

    def __init__(self, c: Config, w):
        t = c.text
        E, L = t.hidden, c.labels
        self.c = c
        p = "text_model."
        self.blocks = [Block(t, w, f"{p}encoder.layers.{i}.") for i in range(t.layers)]
        pos = f32(w[p + "embeddings.position_embedding.weight"])
        self.pos = iron.param(bf(np.repeat(pos, L, axis=0)))
        self.fn, self.fnb = iron.param(bf(w[p + "final_layer_norm.weight"])), iron.param(bf(w[p + "final_layer_norm.bias"]))
        self.head, self.headb = iron.param(bf(w[p + "head.weight"])), iron.param(bf(w[p + "head.bias"]))

    def body(self, x):
        c = self.c
        t = c.text
        E, H, D, L, R = t.hidden, t.heads, t.head_dim, c.labels, c.text_rows

        def attend(q, k, v):
            # a token's texts side by side: L x H heads over the 64 tokens
            q, k, v = (a.reshape(TEXT_LEN, L * H, D) for a in (q, k, v))
            o = MHA.step(q, k, v, heads_interleaved=True, kv_interleaved=True, causal=False, d=D, num_pipelines=1)
            return o.reshape(R, E)

        x = x + self.pos
        for i, b in enumerate(self.blocks):
            if i and i % c.split == 0:
                command_boundary()
            x = b(x, attend)
        x = F.layer_norm(x, (E,), self.fn, self.fnb, eps=t.eps)
        return F.linear(x, self.head, self.headb, precision=PREC)


def patch_rows(img: np.ndarray, c: Config) -> np.ndarray:
    """The vision graph's input of an image (image, image, 3) scaled to [-1, 1]."""
    P, G = c.patch, c.grid
    x = img.reshape(G, P, G, P, 3).transpose(0, 2, 4, 1, 3).reshape(c.patches, 3 * P * P)
    out = np.zeros((c.tokens, c.patch_in), np.float32)
    out[: c.patches, : x.shape[1]] = x
    return out


def text_rows(emb: np.ndarray, c: Config) -> np.ndarray:
    """The text graph's input of embedding rows (labels, 64, hidden): token-major."""
    return emb.transpose(1, 0, 2).reshape(c.text_rows, -1)


def write_text_files(root: Path, out: Path, w) -> None:
    """text/: the tokenizer and the token embedding table (bf16 rows) the host looks up."""
    d = out / "text"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (root / f).exists():
            shutil.copyfile(root / f, d / f)
    emb = w["text_model.embeddings.token_embedding.weight"]
    with open(d / "embed.bin", "wb") as fh:
        for r in range(0, emb.shape[0], 8192):
            fh.write(bf(emb[r : r + 8192]).tobytes())


ARGS = None


def main(argv=None) -> int:
    global ARGS, GELU, PREC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a SigLIP 2 repository (config.json, model.safetensors, tokenizer.json)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--labels", type=int, default=8, help="texts a call of the text graph takes (a multiple of 4)")
    ap.add_argument("--gelu", choices=["erf", "tanh"], default=GELU,
                    help="the MLPs' GELU on the NPU: erf (closer to the checkpoint's tanh GELU than IRON's tanh kernel) or tanh")
    ap.add_argument("--precision", choices=["MEDIUM", "HIGHEST"], default="HIGHEST",
                    help="the GEMMs' precision: MEDIUM (bfp16 activations) or HIGHEST (bf16)")
    ap.add_argument("--only", nargs="*", default=["vision", "text"])
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    GELU, PREC = a.gelu, getattr(Precision, a.precision)
    bind_device(a.device, a.cols)
    c = Config.of(a.root, a.labels)
    w = (RandomWeights if a.random else Safetensors)(a.root / "model.safetensors")
    rng = np.random.default_rng(0)
    scale, bias = float(np.exp(f32(w["logit_scale"]).reshape(-1)[0])), float(f32(w["logit_bias"]).reshape(-1)[0])
    meta = dict(config=dict(vision=dataclasses.asdict(c.vision), text=dataclasses.asdict(c.text), image=c.image,
                            patch=c.patch, vocab=c.vocab, projection=c.projection),
                patch_in=c.patch_in, tokens=c.tokens, patches=c.patches, labels=c.labels, text_len=TEXT_LEN,
                gelu=GELU, precision=a.precision)
    if "vision" in a.only:
        t0 = time.time()
        g = Vision(c, w)
        cg = g.compile(patches=(c.tokens, c.patch_in))
        m = export_graph(cg, a.out / "vision", "vision", meta)
        print(f"vision: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if a.check:
            img = rng.uniform(-1, 1, (c.image, c.image, 3)).astype(np.float32)
            write_check(g, m, a.out / "vision", dict(patches=bf(patch_rows(img, c))), dict(nk=c.patches))
    if "text" in a.only:
        t0 = time.time()
        g = Text(c, w)
        cg = g.compile(x=(c.text_rows, c.text.hidden))
        m = export_graph(cg, a.out / "text", "text", meta)
        print(f"text: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if a.check:
            ids = rng.integers(2, 1000, (c.labels, TEXT_LEN))
            emb = f32(w["text_model.embeddings.token_embedding.weight"][np.sort(ids.reshape(-1))])
            emb = emb[np.argsort(np.argsort(ids.reshape(-1)))].reshape(c.labels, TEXT_LEN, -1)
            write_check(g, m, a.out / "text", dict(x=bf(text_rows(emb, c))))
        t0 = time.time()
        write_text_files(a.root, a.out, w)
        print(f"text/embed.bin: {time.time() - t0:.0f} s", flush=True)
    model = dict(name=a.root.name, kind="siglip", logit_scale=scale, logit_bias=bias, mean=[0.5] * 3, std=[0.5] * 3,
                 eos=1, pad=0, lowercase=True, **meta)
    (a.out / "model.json").write_text(json.dumps(model, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
