#!/usr/bin/env python3
"""LFM3-Audio's speech output (LiquidAI/LFM3-Audio-2B, LFM Open License 1.0) on IRON: the export
hwnpu-audio tts runs on AMD's NPU.

  python tools/amd/iron_lfm3speech.py ~/models/LFM3-Audio-2B ~/models/lfm3-audio-2b-tts.amd-npu2

writes into OUT:

  text/   the text model, its final hidden state out (iron_export_lm.py lfm3tts --q4 --mlp q8
          --head q8: iron_lfm3tts.py), run by internal/speech as hwnpu-chat runs LFM3
  depth/  one MaskGIT pass of the depthformer (a graph, iron_graph.py): x (64, 1024), each
          codebook's embedding of its code or the mask embedding in rows 0..11 (the host looks
          them up), and h (1, 2048), the text model's final hidden state; out (12, 2048), each
          codebook's logits (2017 of them, padded)
  codec/  the liquid codec's decoder over FRAMES audio frames (a graph): x (FRAMES x 6, 64),
          each frame's code vectors (12 codebooks of 4 FSQ levels, 48 values padded to 64)
          repeated over its 6 STFT frames (the decoder's nearest upsampling, done by the host);
          out (FRAMES x 6, 1344), the STFT frames' log magnitudes and phases (1282, padded)
  tables.safetensors  the host's tables (bf16): the depthformer's code tables, the mask
          embedding, the feedback table (internal/speech's)
  config.json, generation_config.json, tokenizer.json, audio_tokenizer/config.json

The depthformer's heads are of 32 values and IRON's MHA takes multiples of 64: each head is laid
out in 64 as [x0..x15, 0 x 16, x16..x31, 0 x 16], so that RoPE's halves (rotate_half: i with
i + 32) pair the head's i with i + 16 as the 32 wide head does, and the zeros stay zeros. Its RMS
norm over 64 is the norm over 32 with the weight over sqrt(2) and half the epsilon. The values
are [v0..v31, 0 x 32] and o_proj reads the first 32 of each head.

The codec's short convolutions (7 taps, causal from zeros) are each tap's product with the
input written k rows down into a state (its first k rows stay zero) and added; its attention
is causal within the decoder's window (30 positions, the query's own among them).
"""

import argparse
import dataclasses
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import iron
import iron.functional as F
from iron import Precision, Scratchpad
from iron.operators.mha import MHA
from iron.operators.sigmoid import Sigmoid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_silu  # noqa: E402
from iron_anima import RandomWeights, Safetensors, bf  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

PREC = Precision.HIGHEST
ROWS = 64  # the depthformer's rows: its 12 codebooks padded to MHA's 64
VOCAB_PAD = 2048
FRAMES = 64  # audio frames a codec call decodes (internal/speech's codec-frames)
CODE_PAD = 64
BINS_PAD = 1344


def f32(x) -> np.ndarray:
    return np.array(x, np.float32)


def swiglu(x, w1, w3, w2):
    return F.linear(iron_silu.silu(F.linear(x, w1, precision=PREC)) * F.linear(x, w3, precision=PREC), w2, precision=PREC)


def rope_table(rows: int, head: int, theta: float, used: int) -> np.ndarray:
    """IRON's RoPE angles (rows, head): cos and sin interleaved, a row per position; the first
    ``used`` of the head's head / 2 pairs turn at a rotate_half head of 2 used's frequencies."""
    half = head // 2
    inv = np.zeros(half)
    inv[:used] = theta ** (-np.arange(used) * 2.0 / (2 * used))
    ang = np.arange(rows)[:, None] * inv[None, :]
    out = np.empty((rows, head), np.float32)
    out[:, 0::2], out[:, 1::2] = np.cos(ang), np.sin(ang)
    return out


# --- the depthformer ------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DepthConfig:
    hidden: int = 1024
    mlp: int = 2816
    layers: int = 3
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 32
    eps: float = 1e-5
    theta: float = 1e6
    codebooks: int = 12
    vocab: int = 2017
    text_hidden: int = 2048

    @classmethod
    def of(cls, cfg: dict) -> "DepthConfig":
        d = cfg["depthformer"]
        if cfg.get("model_type") != "lfm3_audio":
            raise SystemExit(f"model_type {cfg.get('model_type')}: LFM3-Audio's (lfm3_audio) is supported")
        if d.get("is_causal") or d.get("use_sandwich_norm") or any(t != "full_attention" for t in d["layer_types"]):
            raise SystemExit("the depthformer: a bidirectional stack of attention layers without sandwich norms")
        c = cls(hidden=d["hidden_size"], mlp=d["intermediate_size"], layers=d["num_hidden_layers"], heads=d["num_attention_heads"],
                kv_heads=d.get("num_key_value_heads") or d["num_attention_heads"], head_dim=d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"],
                eps=d.get("norm_eps", 1e-5), theta=d["rope_parameters"]["full_attention"]["rope_theta"], codebooks=cfg["codebooks"],
                vocab=cfg["audio_vocab_size"], text_hidden=cfg["lfm"]["hidden_size"])
        if c.head_dim != 32 or c.codebooks > ROWS or c.vocab > VOCAB_PAD:
            raise SystemExit(f"the depthformer's heads of {c.head_dim}, {c.codebooks} codebooks of {c.vocab}: heads of 32, at most {ROWS} of {VOCAB_PAD}")
        return c


def spread(w: np.ndarray, heads: int, d: int = 32) -> np.ndarray:
    """A projection's rows (heads x 32, in) laid out in heads of 64: [0..15, 0 x 16, 16..31, 0 x 16]."""
    w = w.reshape(heads, d, -1)
    out = np.zeros((heads, 2 * d, w.shape[-1]), np.float32)
    out[:, : d // 2], out[:, d : d + d // 2] = w[:, : d // 2], w[:, d // 2 :]
    return out.reshape(heads * 2 * d, -1)


def spread_norm(w: np.ndarray, d: int = 32) -> np.ndarray:
    out = np.zeros(2 * d, np.float32)
    out[: d // 2], out[d : d + d // 2] = w[: d // 2], w[d // 2 :]
    return out / math.sqrt(2)


class DepthLayer(iron.Graph):
    def __init__(self, c: DepthConfig, w, i: int):
        self.c = c
        p = f"depthformer.model.layers.{i}."
        g = lambda n: f32(w[p + n])  # noqa: E731
        H, G, D = c.heads, c.kv_heads, c.head_dim
        self.n1, self.n2 = iron.param(bf(g("operator_norm.weight"))), iron.param(bf(g("ffn_norm.weight")))
        self.q = iron.param(bf(spread(g("self_attn.q_proj.weight"), H)))
        self.k = iron.param(bf(spread(g("self_attn.k_proj.weight"), G)))
        v = g("self_attn.v_proj.weight").reshape(G, D, -1)
        vp = np.zeros((G, 2 * D, v.shape[-1]), np.float32)
        vp[:, :D] = v
        self.v = iron.param(bf(vp.reshape(G * 2 * D, -1)))
        o = g("self_attn.o_proj.weight").reshape(c.hidden, H, D)
        op = np.zeros((c.hidden, H, 2 * D), np.float32)
        op[:, :, :D] = o
        self.o = iron.param(bf(op.reshape(c.hidden, -1)))
        self.gate = iron.param(bf(np.repeat(g("self_attn.gate_proj.weight"), 2 * D, axis=0)))
        self.qn, self.kn = iron.param(bf(spread_norm(g("self_attn.q_norm.weight")))), iron.param(bf(spread_norm(g("self_attn.k_norm.weight"))))
        self.w1, self.w3, self.w2 = (iron.param(bf(g(f"feed_forward.{n}.weight"))) for n in ("w1", "w3", "w2"))

    def body(self, x, angles, nk):
        c = self.c
        R, H, G, D2 = ROWS, c.heads, c.kv_heads, 2 * c.head_dim
        h = F.rms_norm(x, (c.hidden,), self.n1, c.eps)
        q = F.rms_norm(F.linear(h, self.q, precision=PREC).reshape(R * H, D2), (D2,), self.qn, c.eps / 2)
        k = F.rms_norm(F.linear(h, self.k, precision=PREC).reshape(R * G, D2), (D2,), self.kn, c.eps / 2)
        q, k = F.rope(q, angles), F.rope(k, angles)
        v = F.linear(h, self.v, precision=PREC)
        o = MHA.step(q.reshape(R, H, D2), k.reshape(R, G, D2)[:nk], v.reshape(R, G, D2)[:nk], heads_interleaved=True,
                     kv_interleaved=True, causal=False, d=D2, scale=1 / math.sqrt(c.head_dim), num_pipelines=1)
        g = Sigmoid.step(F.linear(h, self.gate, precision=PREC))
        x = x + F.linear(o.reshape(R, H * D2) * g, self.o, precision=PREC)
        return x + swiglu(F.rms_norm(x, (c.hidden,), self.n2, c.eps), self.w1, self.w3, self.w2)


class Depth(iron.Graph):
    """One MaskGIT pass: ``x`` (64, hidden) the codebooks' embeddings (rows 0..11), ``h`` (1, text
    hidden); per call ``nk``, the codebooks. Output (codebooks, 2048): each codebook's logits."""

    def __init__(self, c: DepthConfig, w):
        self.c = c
        E, N = c.hidden, c.codebooks
        self.lin, self.linb = iron.param(bf(w["depth_linear.weight"])), iron.param(bf(w["depth_linear.bias"]))
        self.proj = iron.state((ROWS, E))
        self.layers = [DepthLayer(c, w, i) for i in range(c.layers)]
        self.norm = iron.param(bf(w["depthformer.model.embedding_norm.weight"]))
        self.cb_norms, self.tables = [], []
        for cb in range(N):
            self.cb_norms.append(iron.param(bf(w[f"depth_embeddings.{cb}.embedding_norm.weight"])))
            t = np.zeros((VOCAB_PAD, E), np.float32)
            t[: c.vocab] = f32(w[f"depth_embeddings.{cb}.embedding.weight"])
            self.tables.append(iron.param(bf(t)))
        # the angles of positions 0..63, a row per position (RoPE's rows are position-major)
        self.angles = iron.param(bf(rope_table(ROWS, 2 * c.head_dim, c.theta, c.head_dim // 2)))
        self.logits = iron.state((N, VOCAB_PAD))

    def body(self, x, h, *, nk: Scratchpad[np.int32]):
        c = self.c
        E, N = c.hidden, c.codebooks
        self.proj[0:N] = F.linear(h, self.lin, self.linb, precision=PREC).reshape(N, E)
        x = x + self.proj[0:ROWS]
        for layer in self.layers:
            x = layer(x, self.angles, nk)
        x = F.rms_norm(x, (E,), self.norm, c.eps)
        for cb in range(N):
            row = F.rms_norm(x[cb : cb + 1], (E,), self.cb_norms[cb], c.eps)
            self.logits[cb : cb + 1] = F.linear(row, self.tables[cb], precision=PREC).reshape(1, VOCAB_PAD)
        from iron.operators.copy import Copy

        return Copy.step(self.logits[0:N])


# --- the codec decoder ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CodecConfig:
    width: int = 512
    layers: tuple = ()
    heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    mlp: int = 2304
    kernel: int = 7
    window: int = 30
    stack: int = 6
    code_dim: int = 48
    quant_dim: int = 256
    bins: int = 641
    eps: float = 1e-5
    theta: float = 1e6

    @property
    def rows(self) -> int:
        return FRAMES * self.stack

    @classmethod
    def of(cls, cfg: dict, w) -> "CodecConfig":
        kinds, heads, kv, hd, kernel, mlp = [], 8, 2, 64, 7, 2304
        for i in range(8):
            p = f"decoder.lfm.layers.{i}."
            mlp = w.header[p + "feed_forward.w1.weight"]["shape"][0]
            if p + "conv.conv.weight" in w:
                kinds.append("conv")
                kernel = w.header[p + "conv.conv.weight"]["shape"][-1]
            else:
                kinds.append("attn")
                hd = w.header[p + "self_attn.q_norm.weight"]["shape"][0]
                heads = w.header[p + "self_attn.q_proj.weight"]["shape"][0] // hd
                kv = w.header[p + "self_attn.k_proj.weight"]["shape"][0] // hd
        c = cls(width=cfg["encoder_dim"], layers=tuple(kinds), heads=heads, kv_heads=kv, head_dim=hd, mlp=mlp, kernel=kernel,
                window=cfg["decoder_window"], stack=cfg["stft_frame_stack"], code_dim=cfg["n_q"] * len(cfg["fsq_levels"]),
                quant_dim=cfg["quantizer_dim"], bins=cfg["stft_n_fft"] // 2 + 1)
        if c.code_dim > CODE_PAD or 2 * c.bins > BINS_PAD or hd % 64:
            raise SystemExit(f"the codec's {c.code_dim} code values, {2 * c.bins} outputs, heads of {hd}")
        return c


class CodecLayer(iron.Graph):
    def __init__(self, c: CodecConfig, w, i: int):
        self.c, self.kind = c, c.layers[i]
        p = f"decoder.lfm.layers.{i}."
        g = lambda n: f32(w[p + n])  # noqa: E731
        R = c.rows
        self.norms = [iron.param(bf(g(n + ".weight"))) for n in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm")]
        self.w1, self.w3, self.w2 = (iron.param(bf(g(f"feed_forward.{n}.weight"))) for n in ("w1", "w3", "w2"))
        if self.kind == "conv":
            ip = g("conv.in_proj.weight")
            E = c.width
            self.b, self.cg, self.x = (iron.param(bf(ip[j * E : (j + 1) * E])) for j in range(3))
            self.out = iron.param(bf(g("conv.out_proj.weight")))
            taps = g("conv.conv.weight").reshape(E, c.kernel)
            # tap j reads the position kernel - 1 - j back: a (rows, width) table of it
            self.taps = [iron.param(bf(np.repeat(taps[None, :, j], R, axis=0))) for j in range(c.kernel)]
            self.shifted = [iron.state((R, E)) for _ in range(c.kernel - 1)]
        else:
            self.q, self.k, self.v = (iron.param(bf(g(f"self_attn.{n}_proj.weight"))) for n in "qkv")
            self.o = iron.param(bf(g("self_attn.o_proj.weight")))
            self.gate = iron.param(bf(np.repeat(g("self_attn.gate_proj.weight"), c.head_dim, axis=0)))
            self.qn, self.kn = iron.param(bf(g("self_attn.q_norm.weight"))), iron.param(bf(g("self_attn.k_norm.weight")))

    def conv(self, h):
        c = self.c
        K, R = c.kernel, c.rows
        bx = F.linear(h, self.b, precision=PREC) * F.linear(h, self.x, precision=PREC)
        y = bx * self.taps[K - 1]
        for s in range(1, K):
            self.shifted[s - 1][s:R] = bx[0 : R - s]
            y = y + self.shifted[s - 1][0:R] * self.taps[K - 1 - s]
        return F.linear(F.linear(h, self.cg, precision=PREC) * y, self.out, precision=PREC)

    def attention(self, h, angles):
        c = self.c
        R, H, G, D = c.rows, c.heads, c.kv_heads, c.head_dim
        q = F.rms_norm(F.linear(h, self.q, precision=PREC).reshape(R * H, D), (D,), self.qn, c.eps)
        k = F.rms_norm(F.linear(h, self.k, precision=PREC).reshape(R * G, D), (D,), self.kn, c.eps)
        q, k = F.rope(q, angles), F.rope(k, angles)
        v = F.linear(h, self.v, precision=PREC)
        o = MHA.step(q.reshape(R, H, D), k.reshape(R, G, D), v.reshape(R, G, D), heads_interleaved=True, kv_interleaved=True,
                     causal=True, window=c.window - 1, d=D, num_pipelines=6)
        g = Sigmoid.step(F.linear(h, self.gate, precision=PREC))
        return F.linear(o.reshape(R, H * D) * g, self.o, precision=PREC)

    def body(self, x, angles):
        c = self.c
        E = c.width
        n1, n2, n3, n4 = self.norms
        h = F.rms_norm(x, (E,), n1, c.eps)
        h = self.conv(h) if self.kind == "conv" else self.attention(h, angles)
        x = x + F.rms_norm(h, (E,), n2, c.eps)
        f = swiglu(F.rms_norm(x, (E,), n3, c.eps), self.w1, self.w3, self.w2)
        return x + F.rms_norm(f, (E,), n4, c.eps)


class Codec(iron.Graph):
    """The codec's decoder over FRAMES frames: ``x`` (rows, 64) each STFT frame's code vectors (its
    audio frame's, repeated by the host); output (rows, 1344): the STFT frames."""

    def __init__(self, c: CodecConfig, w):
        self.c = c
        q = "quantizer.fsq." if "quantizer.fsq.project_out.weight" in w else "quantizer."
        po = np.zeros((c.quant_dim, CODE_PAD), np.float32)
        po[:, : c.code_dim] = f32(w[q + "project_out.weight"])
        self.po, self.pob = iron.param(bf(po)), iron.param(bf(w[q + "project_out.bias"]))
        self.fq, self.fqb = iron.param(bf(w["from_quantizer.weight"])), iron.param(bf(w["from_quantizer.bias"]))
        self.layers = [CodecLayer(c, w, i) for i in range(len(c.layers))]
        self.norm = iron.param(bf(w["decoder.lfm.embedding_norm.weight"]))
        lin = np.zeros((BINS_PAD, c.width), np.float32)
        lin[: 2 * c.bins] = f32(w["decoder.lin.weight"])
        linb = np.zeros(BINS_PAD, np.float32)
        linb[: 2 * c.bins] = f32(w["decoder.lin.bias"])
        self.lin, self.linb = iron.param(bf(lin)), iron.param(bf(linb))
        self.angles = iron.param(bf(rope_table(c.rows, c.head_dim, c.theta, c.head_dim // 2)))

    def body(self, x):
        c = self.c
        h = F.linear(F.linear(x, self.po, self.pob, precision=PREC), self.fq, self.fqb, precision=PREC)
        for layer in self.layers:
            h = layer(h, self.angles)
        h = F.rms_norm(h, (c.width,), self.norm, c.eps)
        return F.linear(h, self.lin, self.linb, precision=PREC)


# --- the export -----------------------------------------------------------------------------


def write_tables(out: Path, w, d: DepthConfig) -> None:
    """tables.safetensors: the host's tables of internal/speech (bf16)."""
    names = ["codebook_embedding.weight", "mask_embedding"] + [f"depth_embeddings.{cb}.embedding.weight" for cb in range(d.codebooks)]
    header, blobs, at = {}, [], 0
    for n in names:
        b = bf(w[n]).tobytes()
        header[n] = dict(dtype="BF16", shape=list(w.header[n]["shape"]), data_offsets=[at, at + len(b)])
        blobs.append(b)
        at += len(b)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(out / "tables.safetensors", "wb") as f:
        f.write(len(raw).to_bytes(8, "little"))
        f.write(raw)
        for b in blobs:
            f.write(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="an LFM3-Audio repository (config.json, model.safetensors, tokenizer.json, audio_tokenizer/)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", nargs="*", default=["text", "depth", "codec"])
    ap.add_argument("--cache", type=int, default=2048, help="the text model's KV cache (internal/speech's DefaultCache)")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((a.root / "config.json").read_text())
    dc = DepthConfig.of(cfg)
    rng = np.random.default_rng(0)
    if "text" in a.only:
        t0 = time.time()
        tool = Path(__file__).resolve().parent / "iron_export_lm.py"
        args = [sys.executable, str(tool), "lfm3tts", str(a.root), str(a.out / "text"), "--max-seq-len", str(a.cache),
                "--q4", "--mlp", "q8", "--head", "q8", "--device", a.device, "--cols", str(a.cols)]
        if a.random:
            args += ["--random-weights", "0"]
        subprocess.run(args, check=True)
        print(f"text: exported in {time.time() - t0:.0f} s", flush=True)
    bind_device(a.device, a.cols)
    w = (RandomWeights if a.random else Safetensors)(a.root / "model.safetensors")
    if "depth" in a.only:
        t0 = time.time()
        g = Depth(dc, w)
        cg = g.compile(x=(ROWS, dc.hidden), h=(1, dc.text_hidden))
        meta = dict(config=dataclasses.asdict(dc), values=dict(nk=dc.codebooks),
                    program=dict(inputs=dict(x=dict(shape=[1, dc.codebooks, dc.hidden]), h=dict(shape=[1, 1, dc.text_hidden])),
                                 output=[1, dc.codebooks, dc.vocab]))
        m = export_graph(cg, a.out / "depth", "depth", meta)
        print(f"depth: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if a.check:
            x = np.zeros((ROWS, dc.hidden), np.float32)
            x[: dc.codebooks] = rng.standard_normal((dc.codebooks, dc.hidden)) * 0.05
            h = rng.standard_normal((1, dc.text_hidden)).astype(np.float32)
            write_check(g, m, a.out / "depth", dict(x=bf(x), h=bf(h)), dict(nk=dc.codebooks))
        write_tables(a.out, w, dc)
    if "codec" in a.only:
        t0 = time.time()
        ccfg = json.loads((a.root / "audio_tokenizer" / "config.json").read_text())
        cw = (RandomWeights if a.random else Safetensors)(a.root / "audio_tokenizer" / "model.safetensors")
        cc = CodecConfig.of(ccfg, cw)
        g = Codec(cc, cw)
        cg = g.compile(x=(cc.rows, CODE_PAD))
        meta = dict(config=dataclasses.asdict(cc), frames=FRAMES,
                    program=dict(inputs=dict(x=dict(shape=[1, FRAMES, cc.code_dim], repeat=cc.stack)), output=[1, cc.rows, 2 * cc.bins]))
        m = export_graph(cg, a.out / "codec", "codec", meta)
        print(f"codec: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if a.check:
            x = np.repeat(rng.uniform(-1, 1, (FRAMES, CODE_PAD)).astype(np.float32), cc.stack, axis=0)
            x[:, cc.code_dim :] = 0
            write_check(g, m, a.out / "codec", dict(x=bf(x)))
    for f in ("config.json", "generation_config.json", "tokenizer.json"):
        if (a.root / f).exists():
            shutil.copyfile(a.root / f, a.out / f)
    (a.out / "audio_tokenizer").mkdir(exist_ok=True)
    shutil.copyfile(a.root / "audio_tokenizer" / "config.json", a.out / "audio_tokenizer" / "config.json")
    (a.out / "model.json").write_text(json.dumps(dict(name=a.root.name, kind="lfm3speech", frames=FRAMES), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
