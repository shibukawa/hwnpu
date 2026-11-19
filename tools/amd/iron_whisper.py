#!/usr/bin/env python3
"""Whisper (openai/whisper-large-v3-turbo and the other multilingual checkpoints, MIT) on IRON:
the graphs hwnpu-audio transcribe runs on AMD's NPU.

Two graphs, each exported by iron_graph.py:

  encoder  a 30 second window's log mel frames (the host's im2col of conv1, even frames first:
           see conv_rows), conv1 and its GELU, conv2 (stride 2) and its GELU, the positions,
           the encoder's blocks (layer norm, attention of heads of 64, layer norm, the exact
           GELU's MLP), the last layer norm, and the decoder's cross attention keys and values
           of every layer: outputs out0..out{2L-1}, layer i's keys (2i) and values (2i + 1),
           (frames, hidden) with frames the 1500 padded to 1536
  decoder  one token a call: the row of its token embedding plus its position's (the host adds
           them: decoder/embed.bin and decoder/positions.bin, bf16), each layer's self attention
           over a KV cache in the graph's states (written at the call's ``pos``, attended up to
           ``n`` = pos + 1), its cross attention over the window's keys and values (the states
           xk{i} and xv{i}, which the host copies from the encoder's outputs once a window), the
           MLP, the last layer norm and the logits (the token embeddings, tied; padded to a
           multiple of 1024)

conv2 of frame t reads conv1's frames 2t - 1, 2t and 2t + 1. The host lays conv1's input out with
the even frames in the first half of the rows and the odd in the second, so the three are rows t
of the even half, t of the odd half and t - 1 of the odd half: the last is a product of the odd
half written one row down into a state (its row 0, conv2's padding, stays zero).

  python tools/amd/iron_whisper.py ~/models/whisper-large-v3-turbo ~/models/whisper-large-v3-turbo.amd-npu2

See docs/amd-npu-models.md.
"""

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import iron
import iron.functional as F
from iron import Precision, Scratchpad
from iron.operators.mha import MHA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_anima import RandomWeights, Safetensors, bf, command_boundary  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402
from iron_q4 import install_linear, linear_add, q_param  # noqa: E402

# The encoder's GEMMs' precision (--precision): MEDIUM (bfp16 activations) halves the encoder's
# time (1.71 s to 0.84 s a window on a Ryzen AI 9 365) for the same text as HIGHEST on the clips
# tried. The decoder's one row runs on GEMV, in float32 whatever this is
ENC_PREC = Precision.MEDIUM
# The decoder's weights (--decoder): q8 (iron_q4's 8.5 bits a weight, its GEMV) or bf16. A token
# reads the four layers' and the logits' weights, 316 MB in bf16: the decoder's time
DEC_FMT = "q8"
# The decoder's attentions of one query: MHA packs a head a block and runs this many pipelines
# (packed: at most 4 on 8 columns, dividing the 20 heads). Its default, 1, took a layer's cross
# attention over the 1500 frames 1 ms
MHA_PIPELINES = 4
FRAMES = 3000  # a window's mel frames (30 s at 10 ms)
VOCAB_ALIGN = 1024


def f32(x) -> np.ndarray:
    return np.array(x, np.float32)


@dataclasses.dataclass(frozen=True)
class Config:
    hidden: int = 1280
    mels: int = 128
    enc_layers: int = 32
    dec_layers: int = 4
    heads: int = 20
    mlp: int = 5120
    positions: int = 1500  # the encoder's frames (max_source_positions)
    targets: int = 448  # the decoder's positions (max_target_positions)
    vocab: int = 51866
    split: int = 4  # encoder blocks a command runs

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def rows(self) -> int:
        """The encoder's rows: its frames padded to the GEMM's 512 (MHA's 8 pipelines of 64)."""
        return -(-self.positions // 512) * 512

    @property
    def conv_in(self) -> int:
        """conv1's im2col row: 3 taps of every mel bin, padded to a multiple of 64."""
        return -(-3 * self.mels // 64) * 64

    @property
    def vocab_pad(self) -> int:
        return -(-self.vocab // VOCAB_ALIGN) * VOCAB_ALIGN

    @classmethod
    def of(cls, root: Path) -> "Config":
        cfg = json.loads((root / "config.json").read_text())
        if cfg.get("model_type") != "whisper":
            raise SystemExit(f"{root}: a Whisper checkpoint (model_type whisper) is supported, not {cfg.get('model_type')}")
        c = cls(hidden=cfg["d_model"], mels=cfg["num_mel_bins"], enc_layers=cfg["encoder_layers"],
                dec_layers=cfg["decoder_layers"], heads=cfg["encoder_attention_heads"], mlp=cfg["encoder_ffn_dim"],
                positions=cfg["max_source_positions"], targets=cfg["max_target_positions"], vocab=cfg["vocab_size"])
        if cfg["decoder_attention_heads"] != c.heads or cfg["decoder_ffn_dim"] != c.mlp:
            raise SystemExit(f"{root}: the decoder's heads or MLP differ from the encoder's")
        if c.hidden % c.heads or c.head_dim % 64:
            raise SystemExit(f"{root}: heads of {c.hidden / c.heads:g}: IRON's MHA takes multiples of 64")
        if cfg.get("activation_function", "gelu") != "gelu":
            raise SystemExit(f"{root}: the activation {cfg['activation_function']}: only gelu")
        if 2 * c.positions != FRAMES:
            raise SystemExit(f"{root}: {c.positions} encoder positions: only {FRAMES // 2}")
        return c


def weight(w, fmt: str = "bf16"):
    """A linear's weight as an iron.param: bf16, or q8 (iron_q4's 8.5 bits, run by its GEMV)."""
    return iron.param(bf(w)) if fmt == "bf16" else q_param(f32(w), fmt)


class Attention(iron.Graph):
    """q, k (no bias), v and the output projection of one of a layer's attentions; ``kv`` False
    leaves out k and v (the decoder's cross attention: the encoder's graph makes them)."""

    def __init__(self, w, p: str, fmt: str = "bf16", kv: bool = True):
        g = lambda n: f32(w[p + n])  # noqa: E731
        self.q, self.qb = weight(g("q_proj.weight"), fmt), iron.param(bf(g("q_proj.bias")))
        if kv:
            self.k = weight(g("k_proj.weight"), fmt)
            self.v, self.vb = weight(g("v_proj.weight"), fmt), iron.param(bf(g("v_proj.bias")))
        self.o, self.ob = weight(g("out_proj.weight"), fmt), iron.param(bf(g("out_proj.bias")))


class Norm:
    def __init__(self, w, p: str):
        self.w, self.b = iron.param(bf(w[p + ".weight"])), iron.param(bf(w[p + ".bias"]))

    def __call__(self, x, c: Config):
        return F.layer_norm(x, (c.hidden,), self.w, self.b, eps=1e-5)


class MLP:
    def __init__(self, w, p: str, fmt: str = "bf16"):
        self.f1, self.f1b = weight(w[p + "fc1.weight"], fmt), iron.param(bf(w[p + "fc1.bias"]))
        self.f2, self.f2b = weight(w[p + "fc2.weight"], fmt), iron.param(bf(w[p + "fc2.bias"]))

    def __call__(self, h):
        return F.linear(F.gelu(F.linear(h, self.f1, self.f1b, precision=ENC_PREC)), self.f2, self.f2b, precision=ENC_PREC)


class EncoderBlock(iron.Graph):
    def __init__(self, c: Config, w, i: int):
        self.c = c
        p = f"model.encoder.layers.{i}."
        self.n1, self.n2 = Norm(w, p + "self_attn_layer_norm"), Norm(w, p + "final_layer_norm")
        self.attn = Attention(w, p + "self_attn.")
        self.mlp = MLP(w, p)

    def body(self, x, nk):
        c, a = self.c, self.attn
        R, H, D = c.rows, c.heads, c.head_dim
        h = self.n1(x, c)
        q = F.linear(h, a.q, a.qb, precision=ENC_PREC).reshape(R, H, D)
        k = F.linear(h, a.k, precision=ENC_PREC).reshape(R, H, D)
        v = F.linear(h, a.v, a.vb, precision=ENC_PREC).reshape(R, H, D)
        o = MHA.step(q, k[:nk], v[:nk], heads_interleaved=True, kv_interleaved=True, causal=False, d=D,
                     num_pipelines=8)
        x = x + F.linear(o.reshape(R, c.hidden), a.o, a.ob, precision=ENC_PREC)
        return x + self.mlp(self.n2(x, c))


class Encoder(iron.Graph):
    """The encoder and the decoder's cross attention keys and values.

    Input ``x`` (2 rows, conv_in): conv_rows of the window's log mel frames. Per call ``nk``, the
    frames (1500). Outputs layer i's cross keys (2i) and values (2i + 1), (rows, hidden).
    """

    def __init__(self, c: Config, w):
        self.c = c
        E, R = c.hidden, c.rows
        p = "model.encoder."
        c1 = f32(w[p + "conv1.weight"]).reshape(E, -1)  # (out, mel x 3): bin-major, then the tap
        full = np.zeros((E, c.conv_in), np.float32)
        full[:, : c1.shape[1]] = c1
        self.c1, self.c1b = iron.param(bf(full)), iron.param(bf(w[p + "conv1.bias"]))
        c2 = f32(w[p + "conv2.weight"])  # (out, in, 3)
        self.c2 = [iron.param(bf(c2[:, :, t])) for t in range(3)]
        self.c2b = iron.param(bf(w[p + "conv2.bias"]))
        pos = np.zeros((R, E), np.float32)
        pos[: c.positions] = f32(w[p + "embed_positions.weight"])
        self.pos = iron.param(bf(pos))
        self.shifted = iron.state((R, E))
        self.blocks = [EncoderBlock(c, w, i) for i in range(c.enc_layers)]
        self.norm = Norm(w, p + "layer_norm")
        self.cross = []
        for i in range(c.dec_layers):
            q = f"model.decoder.layers.{i}.encoder_attn."
            self.cross.append((iron.param(bf(w[q + "k_proj.weight"])), iron.param(bf(w[q + "v_proj.weight"])),
                               iron.param(bf(w[q + "v_proj.bias"]))))

    def body(self, x, *, nk: Scratchpad[np.int32]):
        c = self.c
        R = c.rows
        g = F.gelu(F.linear(x, self.c1, self.c1b, precision=ENC_PREC))
        even, odd = g[0:R], g[R : 2 * R]
        # conv2's tap 0 reads frame 2t - 1: the odd half's row t - 1
        self.shifted[1:R] = F.linear(odd, self.c2[0], precision=ENC_PREC)[0 : R - 1]
        y = F.linear(even, self.c2[1], precision=ENC_PREC) + F.linear(odd, self.c2[2], self.c2b, precision=ENC_PREC)
        x = F.gelu(y + self.shifted[0:R]) + self.pos
        for i, b in enumerate(self.blocks):
            if i % c.split == 0:
                command_boundary()
            x = b(x, nk)
        h = self.norm(x, c)
        command_boundary()
        outs = []
        for k, v, vb in self.cross:
            outs.append(F.linear(h, k, precision=ENC_PREC))
            outs.append(F.linear(h, v, vb, precision=ENC_PREC))
        return tuple(outs)


class DecoderLayer(iron.Graph):
    def __init__(self, c: Config, w, i: int):
        self.c = c
        p = f"model.decoder.layers.{i}."
        self.n1, self.n2, self.n3 = (Norm(w, p + n) for n in ("self_attn_layer_norm", "encoder_attn_layer_norm",
                                                             "final_layer_norm"))
        f = DEC_FMT
        self.self_attn, self.cross = Attention(w, p + "self_attn.", f), Attention(w, p + "encoder_attn.", f, kv=False)
        self.mlp = MLP(w, p, f)
        if f != "bf16":
            # q, k and v in one GEMV (k has no bias: zeros), out_proj's and fc2's GEMVs adding the
            # residual: 5 fewer operators a layer, each about 0.15 ms of a token's
            g = lambda n: f32(w[p + "self_attn." + n])  # noqa: E731
            self.qkv = q_param(np.concatenate([g("q_proj.weight"), g("k_proj.weight"), g("v_proj.weight")]), f)
            self.qkv_b = iron.param(bf(np.concatenate([g("q_proj.bias"), np.zeros(c.hidden, np.float32), g("v_proj.bias")])))
        H, D = c.heads, c.head_dim
        self.keys, self.values = iron.state((c.targets, H, D), name=f"k{i}"), iron.state((c.targets, H, D), name=f"v{i}")
        self.xk, self.xv = iron.state((c.rows, H, D), name=f"xk{i}"), iron.state((c.rows, H, D), name=f"xv{i}")

    def body(self, x, pos, n, nk):
        if DEC_FMT != "bf16":
            return self.fused(x, pos, n, nk)
        c, a, b = self.c, self.self_attn, self.cross
        H, D = c.heads, c.head_dim
        kw = dict(heads_interleaved=True, kv_interleaved=True, d=D, num_pipelines=MHA_PIPELINES)
        h = self.n1(x, c)
        q = F.linear(h, a.q, a.qb, precision=ENC_PREC)
        self.keys[pos] = F.linear(h, a.k, precision=ENC_PREC).reshape(H, D)
        self.values[pos] = F.linear(h, a.v, a.vb, precision=ENC_PREC).reshape(H, D)
        o = MHA.step(q.reshape(1, H, D), self.keys[:n], self.values[:n], **kw)
        x = x + F.linear(o.reshape(1, c.hidden), a.o, a.ob, precision=ENC_PREC)
        h = self.n2(x, c)
        q = F.linear(h, b.q, b.qb, precision=ENC_PREC)
        o = MHA.step(q.reshape(1, H, D), self.xk[:nk], self.xv[:nk], causal=False, **kw)
        x = x + F.linear(o.reshape(1, c.hidden), b.o, b.ob, precision=ENC_PREC)
        return x + self.mlp(self.n3(x, c))

    def fused(self, x, pos, n, nk):
        """body with quantized weights: q, k and v from one GEMV, the residuals added in the
        GEMVs (iron_q4.linear_add)."""
        c, a, b, m = self.c, self.self_attn, self.cross, self.mlp
        H, D, E = c.heads, c.head_dim, c.hidden
        kw = dict(heads_interleaved=True, kv_interleaved=True, d=D, num_pipelines=MHA_PIPELINES)
        qkv = linear_add(self.n1(x, c), self.qkv, self.qkv_b).reshape(3, H, D)
        self.keys[pos] = qkv[1]
        self.values[pos] = qkv[2]
        o = MHA.step(qkv[0:1], self.keys[:n], self.values[:n], **kw)
        x = linear_add(o.reshape(1, E), a.o, a.ob, x.reshape(E)).reshape(1, E)
        q = F.linear(self.n2(x, c), b.q, b.qb, precision=ENC_PREC)
        o = MHA.step(q.reshape(1, H, D), self.xk[:nk], self.xv[:nk], causal=False, **kw)
        x = linear_add(o.reshape(1, E), b.o, b.ob, x.reshape(E)).reshape(1, E)
        h = F.gelu(F.linear(self.n3(x, c), m.f1, m.f1b, precision=ENC_PREC))
        return linear_add(h, m.f2, m.f2b, x.reshape(E)).reshape(1, E)


class Decoder(iron.Graph):
    """One token: ``x`` (1, hidden), its embedding row plus its position's; per call ``pos`` (its
    position), ``n`` (pos + 1) and ``nk`` (the encoder's frames). Output the logits (1,
    vocab_pad)."""

    def __init__(self, c: Config, w):
        self.c = c
        self.layers = [DecoderLayer(c, w, i) for i in range(c.dec_layers)]
        self.norm = Norm(w, "model.decoder.layer_norm")
        e = np.zeros((c.vocab_pad, c.hidden), np.float32)
        e[: c.vocab] = f32(w["model.decoder.embed_tokens.weight"])
        self.head = weight(e, DEC_FMT)

    def body(self, x, *, pos: Scratchpad[np.int32], n: Scratchpad[np.int32], nk: Scratchpad[np.int32]):
        for layer in self.layers:
            x = layer(x, pos, n, nk)
        return F.linear(self.norm(x, self.c), self.head, precision=ENC_PREC)


def conv_rows(mel: np.ndarray, c: Config) -> np.ndarray:
    """The encoder's input of a window's log mel (mels, 3000): conv1's im2col (bin-major, then the
    tap; pad 1), the even frames in rows 0.., the odd in rows ``rows``.."""
    m = np.pad(mel, ((0, 0), (1, 1)))
    cols = np.stack([m[:, t : t + FRAMES] for t in range(3)], axis=-1)  # (mels, frames, 3)
    cols = cols.transpose(1, 0, 2).reshape(FRAMES, -1)
    out = np.zeros((2 * c.rows, c.conv_in), np.float32)
    out[: c.positions, : cols.shape[1]] = cols[0::2]
    out[c.rows : c.rows + c.positions, : cols.shape[1]] = cols[1::2]
    return out


def write_tables(out: Path, w, root: Path) -> None:
    """decoder/: the token embedding and position tables (bf16 rows) the host adds, and the files
    the host reads of the checkpoint."""
    d = out / "decoder"
    for name, t in (("embed.bin", "model.decoder.embed_tokens.weight"), ("positions.bin", "model.decoder.embed_positions.weight")):
        e = w[t]
        with open(d / name, "wb") as fh:
            for r in range(0, e.shape[0], 8192):
                fh.write(bf(e[r : r + 8192]).tobytes())
    for f in ("config.json", "generation_config.json", "tokenizer.json"):
        shutil.copyfile(root / f, out / f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a Whisper repository (config.json, model.safetensors, tokenizer.json, generation_config.json)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", nargs="*", default=["encoder", "decoder"])
    ap.add_argument("--precision", choices=["MEDIUM", "HIGHEST"], default="MEDIUM",
                    help="the encoder's GEMMs: MEDIUM (bfp16 activations) or HIGHEST (bf16)")
    ap.add_argument("--decoder", choices=["q8", "bf16"], default="q8", help="the decoder's weights")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--enc-layers", type=int, default=0, help="keep only the encoder's first N blocks (a check of where it drifts)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    global ENC_PREC, DEC_FMT
    ENC_PREC = getattr(Precision, a.precision)
    DEC_FMT = a.decoder
    install_linear()
    bind_device(a.device, a.cols)
    c = Config.of(a.root)
    if a.enc_layers:
        c = dataclasses.replace(c, enc_layers=a.enc_layers)
    w = (RandomWeights if a.random else Safetensors)(a.root / "model.safetensors")
    rng = np.random.default_rng(0)
    meta = dict(config=dataclasses.asdict(c), decoder_weights=DEC_FMT, rows=c.rows, conv_in=c.conv_in, vocab_pad=c.vocab_pad, frames=FRAMES)
    if "encoder" in a.only:
        t0 = time.time()
        g = Encoder(c, w)
        cg = g.compile(x=(2 * c.rows, c.conv_in))
        m = export_graph(cg, a.out / "encoder", "encoder", meta)
        print(f"encoder: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if a.check:
            mel = rng.uniform(-1, 1, (c.mels, FRAMES)).astype(np.float32)
            write_check(g, m, a.out / "encoder", dict(x=bf(conv_rows(mel, c))), dict(nk=c.positions))
    if "decoder" in a.only:
        t0 = time.time()
        g = Decoder(c, w)
        cg = g.compile(x=(1, c.hidden))
        m = export_graph(cg, a.out / "decoder", "decoder", meta)
        print(f"decoder: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if a.check:
            x = rng.standard_normal((1, c.hidden)).astype(np.float32)
            write_check(g, m, a.out / "decoder", dict(x=bf(x)), dict(pos=0, n=1, nk=c.positions))
        t0 = time.time()
        write_tables(a.out, w, a.root)
        print(f"decoder tables: {time.time() - t0:.0f} s", flush=True)
    model = dict(name=a.root.name, kind="whisper", **meta)
    (a.out / "model.json").write_text(json.dumps(model, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
