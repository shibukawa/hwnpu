#!/usr/bin/env python3
"""PLaMo-Embedding (pfnet/plamo-embedding-1b, PlamoBiModel, Apache-2.0) on IRON: the graphs
hwnpu-chat's search and embeddings run on AMD's NPU.

PlamoBiModel is the PLaMo decoder with its attention made bidirectional (modeling_plamo.py, as
hwnpu's Core ML conversion reads it: internal/llm/plamo): pre-norm layers of attention and a
SwiGLU MLP, each added to the residual, then a final RMS norm. The attention has one fused qkv
projection (here three linears of its rows), multi-query keys and values (one head), an RMS norm
of each head's queries and keys with a weight a head (eps 1e-6, no weight inside the norm),
rotate-half RoPE of base 10000 at positions 0..rows-1, and sees every token of the text (the
padding's keys left out). The host pools: the mean over the text's own tokens of the last hidden
state, leaving out the BOS and a query's instruction (internal/chat's meanPool).

One graph a sequence length (--seqs, default 128 and 1024 as hwnpu's Core ML programs), each
exported by iron_graph.py into its folder sNNN:

  input   x (seq, hidden): the token embedding rows of the text, zeros after them (the host looks
          them up in embed.bin)
  value   nk: the text's tokens (the keys attended to)
  output  (seq, hidden): the last hidden state, after the final norm

  python tools/amd/iron_plamo_embed.py ~/models/plamo-embedding-1b ~/models/plamo-embedding-1b.amd-npu2

The export also holds config.json, tokenizer.model and the token embedding table (embed.bin,
bf16 rows) the host reads, and model.json (kind plamo-embed: the graphs' lengths).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import iron
import iron.functional as F
from iron import Precision, Scratchpad
from iron.lm import rope_angles
from iron.operators.mha import MHA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_anima import RandomWeights, Safetensors, bf, command_boundary  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

# The GEMMs' precision: HIGHEST (bf16 activations), as SigLIP 2's embeddings take it
# (iron_siglip2.py: MEDIUM's bfp16 lost cosine against float32)
PREC = Precision.HIGHEST
QK_EPS = 1e-6  # modeling_plamo.py's Attention: the query and key norms' epsilon
ROPE_THETA = 10000.0  # RotaryEmbedding's default, which Attention does not override


def f32(x) -> np.ndarray:
    return np.asarray(x, np.float32)


class Config:
    def __init__(self, root: Path):
        c = json.loads((root / "config.json").read_text())
        if c.get("architectures") != ["PlamoBiModel"]:
            raise SystemExit(f"{root}: {c.get('architectures')} is not PlamoBiModel")
        if c.get("n_expert") or c.get("sparse_step") or c.get("eval_attention_n_bit") or c.get("eval_mlp_n_bit"):
            raise SystemExit(f"{root}: experts or evaluation-time quantization are not built")
        self.hidden = c["hidden_size"]
        self.layers = c["num_hidden_layers"]
        self.heads = c["num_attention_heads"]
        self.kv_heads = c["num_key_value_heads"]
        self.head_dim = c["hidden_size_per_head"]
        self.mlp = c["intermediate_size"]
        self.vocab = c["vocab_size"]
        self.positions = c["max_position_embeddings"]
        self.eps = c["rms_norm_eps"]


def pipelines(rows: int) -> int:
    """MHA's pipelines for ``rows`` queries: 64 rows each, up to 8 (above 6 an even count)."""
    for p in (8, 6, 4, 2, 1):
        if rows % (64 * p) == 0:
            return p
    raise ValueError(f"{rows} rows: a multiple of 64")


class Layer(iron.Graph):
    """PlamoDecoderLayer (ModifiedPlamoDecoderLayer): norm, attention, added; norm2, the MLP, added."""

    def __init__(self, c: Config, w, i: int, seq: int):
        self.c, self.seq = c, seq
        p = f"layers.layers.{i}."
        g = lambda n: f32(w[p + n])  # noqa: E731
        H, G, D, F_ = c.heads, c.kv_heads, c.head_dim, c.mlp
        self.n1, self.n2 = iron.param(bf(g("norm.weight"))), iron.param(bf(g("norm2.weight")))
        qkv = g("self_attn.qkv_proj.weight")
        self.q = iron.param(bf(qkv[: H * D]))
        self.k = iron.param(bf(qkv[H * D : (H + G) * D]))
        self.v = iron.param(bf(qkv[(H + G) * D :]))
        # a head's norm weight a head: a row of every head's, times the normed rows (seq, heads x D)
        self.qw = iron.param(bf(g("self_attn.q_weight").reshape(H * D)))
        self.kw = iron.param(bf(g("self_attn.k_weight").reshape(G * D)))
        self.o = iron.param(bf(g("self_attn.o_proj.weight")))
        gu = g("mlp.gate_up_proj.weight")
        self.gate, self.up = iron.param(bf(gu[:F_])), iron.param(bf(gu[F_:]))
        self.down = iron.param(bf(g("mlp.down_proj.weight")))

    def body(self, x, angles, nk):
        c, S = self.c, self.seq
        E, H, G, D = c.hidden, c.heads, c.kv_heads, c.head_dim
        h = F.rms_norm(x, (E,), self.n1, eps=c.eps)
        q = F.linear(h, self.q, precision=PREC)
        k = F.linear(h, self.k, precision=PREC)
        v = F.linear(h, self.v, precision=PREC)
        q = F.rms_norm(q.reshape(S * H, D), (D,), eps=QK_EPS).reshape(S, H * D) * self.qw
        k = F.rms_norm(k.reshape(S * G, D), (D,), eps=QK_EPS).reshape(S, G * D) * self.kw
        # an angle row serves the heads of its position (consecutive rows)
        q = F.rope(q.reshape(S * H, D), angles).reshape(S, H, D)
        k = F.rope(k.reshape(S * G, D), angles).reshape(S, G, D)
        o = MHA.step(q, k[:nk], v.reshape(S, G, D)[:nk], heads_interleaved=True, kv_interleaved=True,
                     causal=False, d=D, num_pipelines=pipelines(S))
        x = x + F.linear(o.reshape(S, H * D), self.o, precision=PREC)
        h = F.rms_norm(x, (E,), self.n2, eps=c.eps)
        m = F.silu(F.linear(h, self.gate, precision=PREC)) * F.linear(h, self.up, precision=PREC)
        return x + F.linear(m, self.down, precision=PREC)


class Encoder(iron.Graph):
    """PlamoBiModel over ``seq`` rows: the layers and the final norm."""

    def __init__(self, c: Config, w, seq: int, split: int, longest: int):
        if seq > longest or longest > c.positions:
            raise SystemExit(f"{seq} rows of {longest}: past the {c.positions} positions")
        self.c, self.seq, self.split = c, seq, split
        self.blocks = [Layer(c, w, i, seq) for i in range(c.layers)]
        self.norm = iron.param(bf(w["norm.weight"]))
        # the table of the longest graph's positions in every graph: every weight is then at the
        # same place in each graph's bank, which the lengths share (the host's one copy of them)
        self.angles = iron.param(bf(rope_angles(c.head_dim, longest, ROPE_THETA)))

    def body(self, x, *, nk: Scratchpad[np.int32]):
        c = self.c
        for i, b in enumerate(self.blocks):
            if i and i % self.split == 0:
                command_boundary()
            x = b(x, self.angles[: self.seq], nk)
        return F.rms_norm(x, (c.hidden,), self.norm, eps=c.eps)


def share_banks(out: Path, seqs: list[int]) -> None:
    """Keep one image of the banks the graphs share: every graph's weights are at the same places
    (Encoder's angles of the longest graph's positions), so the shorter graphs' images are the
    longest's bytes. Each shorter graph's manifest names the longest's file (../sNNN/...) and its
    own goes: a download (and the export on the disk) holds the weights once."""
    import hashlib

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        return h.hexdigest()

    longest = out / f"s{seqs[-1]}"
    lm = json.loads((longest / "manifest.json").read_text())
    for seq in seqs[:-1]:
        d = out / f"s{seq}"
        m = json.loads((d / "manifest.json").read_text())
        for kind, bank in m["banks"].items():
            mine, theirs = bank.get("image"), lm["banks"].get(kind, {}).get("image")
            if not mine or not theirs or digest(d / mine) != digest(longest / theirs):
                continue
            (d / mine).unlink()
            bank["image"] = f"../{longest.name}/{theirs}"
            print(f"s{seq}: bank {kind} is {longest.name}'s", flush=True)
        (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")


def write_files(root: Path, out: Path, w) -> None:
    """config.json, tokenizer.model and the token embedding table (bf16 rows) the host reads."""
    for f in ("config.json", "tokenizer.model"):
        shutil.copyfile(root / f, out / f)
    emb = w["embed_tokens.weight"]
    with open(out / "embed.bin", "wb") as fh:
        for r in range(0, emb.shape[0], 8192):
            fh.write(bf(emb[r : r + 8192]).tobytes())


def main(argv=None) -> int:
    global PREC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="pfnet/plamo-embedding-1b (config.json, model.safetensors, tokenizer.model)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--seqs", type=int, nargs="+", default=[128, 1024], help="the graphs' sequence lengths (multiples of 64)")
    ap.add_argument("--split", type=int, default=0,
                    help="layers a command (0: by the length, so that each stays under the device heap and 2 s)")
    ap.add_argument("--precision", choices=["MEDIUM", "HIGHEST"], default="HIGHEST")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    PREC = getattr(Precision, a.precision)
    bind_device(a.device, a.cols)
    c = Config(a.root)
    w = (RandomWeights if a.random else Safetensors)(a.root / "model.safetensors")
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    meta = dict(hidden=c.hidden, layers=c.layers, vocab=c.vocab, precision=a.precision)
    for seq in sorted(set(a.seqs)):
        t0 = time.time()
        split = a.split or (8 if seq <= 128 else 2 if seq <= 512 else 1)
        g = Encoder(c, w, seq, split, max(a.seqs))
        cg = g.compile(x=(seq, c.hidden))
        m = export_graph(cg, a.out / f"s{seq}", f"s{seq}", dict(meta, seq=seq, split=split))
        print(f"s{seq}: compiled in {time.time() - t0:.0f} s; {m['graph'].get('plan', '')}", flush=True)
        if a.check:
            n = min(seq, 37)
            x = np.zeros((seq, c.hidden), np.float32)
            x[:n] = rng.normal(0, 0.02, (n, c.hidden))
            write_check(g, m, a.out / f"s{seq}", dict(x=bf(x)), dict(nk=n))
    share_banks(a.out, sorted(set(a.seqs)))
    t0 = time.time()
    write_files(a.root, a.out, w)
    print(f"embed.bin: {time.time() - t0:.0f} s", flush=True)
    model = dict(name=a.root.name, kind="plamo-embed", seqs=sorted(set(a.seqs)), **meta)
    (a.out / "model.json").write_text(json.dumps(model, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
