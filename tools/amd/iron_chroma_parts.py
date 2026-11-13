#!/usr/bin/env python3
"""Chroma1's text encoder (T5-XXL's encoder) and FLUX's VAE decoder on IRON: the two graphs
iron_chroma.py's DiT needs before and after it on AMD's NPU.

  t5      T5-XXL's encoder (transformers' T5EncoderModel: 24 layers, 4096 wide, 64 heads of 64, a
          gated tanh-GELU FF of 10240, RMS norms without a mean, no 1/sqrt(d) on the scores, one
          relative position bias of 32 buckets computed in layer 0 and shared by every layer, the
          final RMS norm) over the 512 rows ChromaPipeline pads every prompt to, with its tokenizer
          mask: inputs ``x`` (512, 4096), the embedding rows the host looks up (``embed.bin``), and
          ``mask`` (512,), the tokenizer's attention mask (1 for the prompt's tokens and its </s>,
          0 for the pads); output the 512 rows. Every query row is right (the pads' too: they
          attend to the prompt's keys only, as transformers' mask makes them); the pipeline keeps
          the prompt's rows and the first pad's.
  vae/s*  the FLUX VAE decoder (diffusers' AutoencoderKL, 16 latent channels at an 8th of the
          sides) in stages: the latents' unscaling (z / scaling_factor + shift_factor), conv_in, the
          mid block (two ResnetBlock2D and a one-head attention of 512 channels over every pixel),
          four up blocks of three ResnetBlock2D (the first three ending in a nearest 2x and a 3 x 3
          convolution), GroupNorm(32, eps 1e-6) and SiLU before every convolution, conv_norm_out,
          conv_out to RGB.

T5's attention (IRON's MHA takes no additive bias). Explicit, a group of ``heads_per_group`` (8)
heads at a time, with GEMMs whose block-diagonal operands put the group's heads side by side:

  - k^T and v^T come straight out of their projections, the weight as the GEMM's A
    (``F.linear(W, h)`` is ``(h W^T)^T``), so a head's 64 rows of either are 64 contiguous rows;
    a copy puts each head's (64, 512) block on the diagonal of K^T (576, 8 x 512) and V^T
    (512, 8 x 512), whose other blocks stay zero (states, never written there);
  - the scores of the 8 heads are one GEMM: [q of the group | 1] (512, 576) times K^T, float32
    out, (512, 8 x 512): query q's row holds head h's 512 scores at h x 512;
  - the mask rides the contraction: K^T's row 512 is (mask - 1) x 16384 under every head's
    block (written once a call) and q's column 512 is 1 (written once a call), so a pad key's
    score is 16384 less, exp of it 0, the exact effect of transformers' additive min-float mask;
  - IRON's Softmax on float32 rows of 512 (each (query, head)) adds the relative bias, a float32
    constant (8 params of (512 x 8, 512): 64 MiB, shared by all layers, built at export from the
    checkpoint's (32, 64) table and T5's buckets), before its maximum: scores + bias exactly as
    the float32 encoder sums them;
  - P (512, 8 x 512) times V^T gives the group's (512, 8 x 64) outputs, copied into the
    attention's output columns.

The block-diagonal products do 8 times the score and P V work (2 x 0.27 TMAC over the 24
layers, against 2.4 TMAC of projections). T5's residual stream reaches 3e5 in its outlier
channels from layer 7 on; the scores and the softmax are float32. The projections are q8
(iron_q4; q4 broke T5's rows in an earlier study), dequantized to bf16 on the device for GEMMs at
Precision.HIGHEST (bf16 operands, float32 sums). The FF's GELU is IRON's erf GELU by default
(``--t5-gelu``): IRON's tanh GELU is a table whose error carried through 24 layers took the worst
prompt row's cosine to 0.986 (0.994 with erf; the two functions differ by less than bf16's
rounding of their outputs).

Measured (IRON's reference on the CPU against transformers' T5Block in float32, a 23-token
prompt, the prompt's rows and the first pad's): cosine 0.99975, the worst row 0.9943, rel RMS
2.2 % (transformers' own bf16 blocks: 0.99981, 0.9991, 2.0 %; this graph's bf16 weights: 0.99986,
0.9985, 1.7 %; q8 with ``wo``, or ``q`` and ``k``, kept bf16 (``--t5-bf16``): 0.9948, 0.9960 at the
worst row; the residual stream as a bf16 pair through 2Sum (``--t5-residual pair``): no better).
12 commands (``--t5-split`` 2 layers each), 5.0 GB of banks.

The VAE's GroupNorm (IRON has none, nor a reciprocal square root). On the device, from GEMMs and
RMS norms (``FluxVaeStage.group_norm``) over the image's pixels (np of them, C channels, groups of
Cg), dense and channel last:

  1. the channels' sums over the pixels: one GEMM of a ones block (16, np / Q) by the pixels
     viewed (np / Q, Q C), a second one folding the Q partial sums and averaging each group
     (a constant (C, Q C)): minus each channel's group mean m~ (bf16);
  2. the pixels centred, d - m~ (a row added to every row: its rounding is the centred value's);
  3. the centred pixels transposed, (C, np), exactly (``F.linear(eye, d)``), and cut into rows of
     L = 4096 values of one channel each: a group's Cg np / L rows are contiguous;
  4. each row's L2 norm without a square root operator: RMS-normalize the row (y = u / rms(u)),
     then sum(y u) / sqrt(L) = ||u|| (an elementwise product and a GEMM against a row of
     1 / sqrt(L)); the rows' plain sums (a GEMM) give the residual mean r of the centred values;
  5. the reciprocal square root of a group's sum of squares S: the group's k row norms n_j and a
     constant c in one row of width W, RMS-normalized with eps (np Cg eps_gn - c^2) / W: the output
     at c is c sqrt(W) / sqrt(sum n_j^2 + np Cg eps_gn) = (c sqrt(W) / sqrt(np Cg)) / sqrt(var + eps)
     exactly, the constant folded with gamma into the GEMM that spreads each group's value to
     its channels (a);
  6. y = (d - m~) a + (beta - r a): the shift by r makes the mean the exact one (m~ is bf16; a
     group's |mean| / std reaches 6.3 in this decoder), and the products and sums are each rounded
     once from values of the normalized output's size.

About 20 operators a norm; nothing on the host. The pads of the grid stay zero: the norm reads
the image's pixels (a copy of the grid's interior), and SiLU's output is copied back into a fresh
zero-padded grid for the 3 x 3 convolution that reads it.

Measured at 512 x 512 (IRON's reference against diffusers' float32 decoder, the latents of a
picture encoded by the VAE): the mid block's output cosine 0.99997, the up blocks' 0.99994,
0.99983, 0.99997, 0.99999, the image's 0.99998, rel RMS 0.7 %, PSNR 53.6 dB. 10 stages, 129
commands. IRON's reference does not round the convolutions' operands to bfp16 as the device's
Precision.MEDIUM does (``--vae-precision``; Anima's VAE runs it), nor does it model the cores'
rounding of bf16 results: neither is measured here (nothing ran on the NPU).

The grids, the 3 x 3 convolutions (GridGEMM), the 2x upsampling as four phase convolutions at the
low resolution, the stages' export, are iron_anima.VaeStage's; the attention's GEMMs (one head of
512 channels) iron_qwenimage.VaeStage's.

  python tools/amd/iron_chroma_parts.py ~/models/Chroma1-Flash OUT --only t5 vae --size 512 \\
      --vae-config ~/models/FLUX.1-schnell/vae
  # against the CPU's float32 (tools/ref/ref_chroma_parts.py), IRON's reference on the CPU:
  python tools/amd/iron_chroma_parts.py ROOT OUT --only t5 --ref-t5 T5.npz
  python tools/amd/iron_chroma_parts.py ROOT OUT --only vae --ref-vae VAE.npz
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
from iron import Precision
from iron.operators.copy import Copy
from iron.operators.gemm import GEMM
from iron.operators.softmax import Softmax

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_anima  # noqa: E402
import iron_qwenimage as qi  # noqa: E402
import iron_sana  # noqa: E402
from iron_anima import GridShape, bf, command_boundary, tap, variant_bank  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

HIGHEST = Precision.HIGHEST


def f32(x) -> np.ndarray:
    return np.asarray(x, np.float32)


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# --- T5-XXL's encoder ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class T5Config:
    rows: int = 512  # ChromaPipeline's max_sequence_length, every prompt padded to it
    hidden: int = 4096
    layers: int = 24
    heads: int = 64
    head_dim: int = 64
    mlp: int = 10240
    buckets: int = 32
    max_distance: int = 128
    eps: float = 1e-6
    vocab: int = 32128
    heads_per_group: int = 8  # heads a block-diagonal score GEMM takes
    split: int = 2  # layers a command runs
    # layers a bank holds: the export is one ELF with a bank image per group, read into one bank
    # in turn (SwapVariants) so the 5 GB of q8 weights are never pinned together (0: one graph)
    group: int = 4
    # the FF's GELU: "none" (IRON's erf GELU: nearer float32's tanh GELU than IRON's tanh table,
    # whose error is 5 times bf16's rounding) or "tanh" (gelu_new, T5's own)
    gelu: str = "none"
    # the residual stream: "bf16", or "pair" (two bf16 halves through 2Sum; measured no better:
    # 2.17 % against 2.22 % rel RMS, the error is the projections' and the FF's)
    residual: str = "bf16"
    bf16: tuple = ()  # projections kept bf16 (q, k, v, o, wi_0, wi_1, wo), the rest in the export's format

    @property
    def groups(self) -> int:
        return self.heads // self.heads_per_group

    @property
    def group_width(self) -> int:
        return self.heads_per_group * self.head_dim

    @property
    def aug(self) -> int:
        """The score GEMM's contraction: a group's q columns and 64 more (the mask's one, zeros)."""
        return self.group_width + 64


MASK = 16384.0  # what a pad key's score loses (exact in bf16; exp(-16384) is 0 in float32)


def relative_buckets(c: T5Config) -> np.ndarray:
    """T5's bidirectional relative position buckets (q, k) of the key's position less the query's
    (transformers' _relative_position_bucket, its float32 logarithm)."""
    _np = np
    q = _np.arange(c.rows)[:, None]
    k = _np.arange(c.rows)[None, :]
    rel = k - q
    nb = c.buckets // 2
    out = (rel > 0).astype(_np.int64) * nb
    rel = _np.abs(rel)
    exact = nb // 2
    small = rel < exact
    with _np.errstate(divide="ignore"):
        large = exact + (_np.log(_np.maximum(rel, 1).astype(_np.float32) / _np.float32(exact)) / _np.float32(math.log(c.max_distance / exact))
                         * _np.float32(nb - exact)).astype(_np.int64)
    large = _np.minimum(large, nb - 1)
    return out + _np.where(small, rel, large)


def t5_bias(c: T5Config, table) -> np.ndarray:
    """The position bias (heads, q, k), float32, from relative_attention_bias.weight (32, heads)."""
    return f32(table)[relative_buckets(c)].transpose(2, 0, 1).copy()


class T5Layer(iron.Graph):
    """A T5 encoder block: RMS norm, self-attention (no scaling, the shared bias), RMS norm,
    gated-GELU FF; the residuals. Weights iron_sana.text_param (q8)."""

    def __init__(self, c: T5Config, w, i: int):
        E, M = c.hidden, c.mlp
        self.c = c
        p = f"encoder.block.{i}.layer."
        if w is None:
            lin = lambda n, *s: iron.param(*s)  # noqa: E731
            vec = lambda n, s: iron.param(s)  # noqa: E731
        else:
            def lin(n, *s):
                if n.split(".")[-2] in c.bf16:  # kept bf16 (T5Config.bf16)
                    return iron.param(bf(w[p + n]))
                return iron_sana.text_param(f32(w[p + n]))

            vec = lambda n, s: iron.param(bf(w[p + n]))  # noqa: E731
        self.ln0 = vec("0.layer_norm.weight", E)
        self.q = lin("0.SelfAttention.q.weight", E, E)
        self.k = lin("0.SelfAttention.k.weight", E, E)
        self.v = lin("0.SelfAttention.v.weight", E, E)
        self.o = lin("0.SelfAttention.o.weight", E, E)
        self.ln1 = vec("1.layer_norm.weight", E)
        self.wi0 = lin("1.DenseReluDense.wi_0.weight", M, E)
        self.wi1 = lin("1.DenseReluDense.wi_1.weight", M, E)
        self.wo = lin("1.DenseReluDense.wo.weight", E, M)

    @staticmethod
    def transposed(h, w):
        """(h w^T)^T, w a text_param: the projection's (out, rows), the weight as the GEMM's A."""
        import iron_q4

        e = iron_q4._Q4.get(id(w))
        if e is not None and e[0] is w:
            _, _, k, fmt = e
            _, QDequant = iron_q4.operators()
            w = QDequant.step(w.reshape(-1, iron_q4.FORMATS[fmt]), fmt=fmt, K=k)
        return F.linear(w, h, precision=iron_sana.PREC.text)

    def body(self, x, xl, st, bias):
        """x the residual (bf16), or with xl its low half (``residual`` "pair"): x + xl."""
        c = self.c
        L, E, D, HG, GW, KA = c.rows, c.hidden, c.head_dim, c.heads_per_group, c.group_width, c.aug
        tl = iron_sana.text_linear
        h = F.rms_norm(x, (E,), self.ln0, c.eps)
        q = tl(h, self.q)  # (L, E)
        kt = self.transposed(h, self.k)  # (E, L): head h's rows h D .. h D + D
        vt = self.transposed(h, self.v)
        for g in range(c.groups):
            Copy.step(q, st.qa, src=tap((L, E), g * GW, [L, GW], [E, 1]), dst=tap((L, KA), 0, [L, GW], [KA, 1]))
            blocks = dict(sizes=[HG, D, L], strides=[D * HG * L + L, HG * L, 1])
            Copy.step(kt, st.kt, src=tap((E, L), g * GW * L, [HG, D, L], [D * L, L, 1]),
                      dst=tap((KA, HG * L), 0, blocks["sizes"], blocks["strides"]))
            Copy.step(vt, st.vt, src=tap((E, L), g * GW * L, [HG, D, L], [D * L, L, 1]),
                      dst=tap((GW, HG * L), 0, blocks["sizes"], blocks["strides"]))
            s = GEMM.step(st.view(st.qa), st.view(st.kt), b_col_maj=False, precision=HIGHEST,
                          output_dtype=np.float32)  # (L, HG L): row q, head h's keys at h L
            p = Softmax.step(s.reshape(L * HG, L), addend=bias[g], input_dtype=np.float32)
            o = F.linear(p.reshape(L, HG * L), st.view(st.vt), precision=HIGHEST)  # (L, GW)
            Copy.step(o, st.o, dst=tap((L, E), g * GW, [L, GW], [E, 1]))
        x, xl = self.add(x, xl, tl(st.view(st.o), self.o))
        h = F.rms_norm(x, (E,), self.ln1, c.eps)
        a = F.gelu(tl(h, self.wi0), approximate=c.gelu) * tl(h, self.wi1)
        return self.add(x, xl, tl(a, self.wo))

    def add(self, x, xl, y):
        """The residual plus y: bf16, or ("pair") a bf16 pair x + xl carried through 2Sum (each
        operator rounds once to bf16, so s + t is x + y exactly): T5's residual stream reaches
        3e5 in its outlier channels, and bf16's rounding of the sum (transformers keeps it float32,
        its ``wo`` in float32) was most of the encoder's error."""
        if self.c.residual != "pair":
            return x + y, xl
        s = x + y
        bb = s - x
        aa = s - bb
        t = (x - aa) + (y - bb)
        return s, xl + t


class _T5States:
    """The attention's block-diagonal operands and output, shared by every layer."""

    def __init__(self, c: T5Config):
        L, HG, GW, KA = c.rows, c.heads_per_group, c.group_width, c.aug
        self.qa = iron.state((L, KA))  # [q of a group | 1 | 0...]
        self.kt = iron.state((KA, HG * L))  # K^T block-diagonal; row GW the mask's addend
        self.vt = iron.state((GW, HG * L))  # V^T block-diagonal
        self.o = iron.state((L, c.hidden))

    @staticmethod
    def view(s):
        return s[0 : s.shape[0]]


class T5(iron.Graph):
    """T5-XXL's encoder over ``x`` (rows, hidden), the embedding rows, with ``mask`` (rows,) the
    tokenizer's attention mask: the final RMS norm's rows (rows, hidden).

    Grouped (``T5Config.group`` layers from ``first``): the group's residual and its final RMS
    norm, two outputs, the same graph for every group (a bank image each): the host feeds the
    residual to the next group and keeps the last group's norm. The norm of 512 rows costs little
    and keeps the encoder all on the NPU, as one graph did."""

    def __init__(self, c: T5Config, w=None, layers: int | None = None, first: int = 0):
        self.c = c
        n = c.layers if layers is None else layers
        self.grouped = layers is not None and c.group > 0
        self.layers = [T5Layer(c, w, first + i) for i in range(n)]
        self.st = _T5States(c)
        L, HG = c.rows, c.heads_per_group
        if w is None:
            bias = np.zeros((c.heads, L, L), np.float32)
            self.norm = iron.param(c.hidden)
        else:
            bias = t5_bias(c, w["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"])
            self.norm = iron.param(bf(w["encoder.final_layer_norm.weight"]))
        # group g's (query, head, key) rows, float32: Softmax's addend
        self.bias = [iron.param(np.ascontiguousarray(bias[g * HG : (g + 1) * HG].transpose(1, 0, 2).reshape(L * HG, L)))
                     for g in range(c.groups)]
        one = np.zeros((L, 2), np.float32)
        one[:, 0] = 1
        self.one = iron.param(bf(one))  # q's column GW and the zero after it
        self.zeros = iron.param(bf(np.zeros((L, c.hidden), np.float32)))

    def body(self, x, mask):
        c = self.c
        L, HG, GW, KA = c.rows, c.heads_per_group, c.group_width, c.aug
        st = self.st
        # q's column GW is 1 and K^T's row GW (mask - 1) * MASK under every head: the score's addend
        Copy.step(self.one, st.qa, dst=tap((L, KA), GW, [L, 2], [KA, 1]))
        madd = mask * MASK + (-MASK)
        xl = self.zeros  # the residual's low half (unused at "bf16")
        for h in range(HG):
            Copy.step(madd, st.kt, dst=tap((KA, HG * L), GW * HG * L + h * L, [L], [1]))
        for i, layer in enumerate(self.layers):
            if i and i % c.split == 0:
                command_boundary()
            x, xl = layer(x, xl, st, self.bias)
        if c.residual == "pair":
            assert not self.grouped, "a grouped T5 carries a bf16 residual"
            x = x + xl
        y = F.rms_norm(x, (c.hidden,), self.norm, c.eps)
        return (x, y) if self.grouped else y


def t5_checkpoint(root: Path, random: bool = False):
    return qi.Checkpoint(root / "text_encoder", random=random)


def _proto_fields(b: bytes):
    """A protobuf message's (field, wire type, value) in order: varints as int, fixed32 as bytes,
    length-delimited as bytes (enough for SentencePiece's ModelProto)."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, t = key >> 3, key & 7
        if t == 0:
            v, i = _varint(b, i)
        elif t == 1:
            v, i = b[i : i + 8], i + 8
        elif t == 2:
            n, i = _varint(b, i)
            v, i = b[i : i + n], i + n
        elif t == 5:
            v, i = b[i : i + 4], i + 4
        else:
            raise ValueError(f"protobuf wire type {t}")
        yield f, t, v


def _varint(b: bytes, i: int):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def t5_tokenizer_json(tok: Path) -> dict:
    """tokenizers' T5 tokenizer.json from the repository's spiece.model and tokenizer_config.json,
    for a repository without a tokenizer.json (Chroma1's): the Unigram's pieces and scores, the
    normalizer's precompiled charsmap, the extra ids after the pieces (last to first, as
    T5TokenizerFast), </s> after the text. The form of Anima's t5_tokenizer.json, which hwnpu's Go
    tokenizer reads (FLUX.1's newer one has a normalizer Sequence it does not). Read with protobuf's
    wire format: the IRON environment has no sentencepiece or transformers."""
    import base64
    import struct

    vocab, charsmap, unk = [], b"", None
    for f, _, v in _proto_fields((tok / "spiece.model").read_bytes()):
        if f == 1:  # SentencePiece: piece, score, type
            piece, score, kind = "", 0.0, 1
            for g, _, u in _proto_fields(v):
                if g == 1:
                    piece = u.decode()
                elif g == 2:
                    score = struct.unpack("<f", u)[0]
                elif g == 3:
                    kind = u
            if kind == 2:  # UNKNOWN
                unk = len(vocab)
            vocab.append([piece, score if kind == 1 else 0.0])
        elif f == 3:  # NormalizerSpec
            for g, _, u in _proto_fields(v):
                if g == 2:
                    charsmap = u
    cfg = json.loads((tok / "tokenizer_config.json").read_text())
    extra = int(cfg.get("extra_ids", 100))
    vocab += [[f"<extra_id_{i}>", 0.0] for i in reversed(range(extra))]
    special = dict(single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
    added = [dict(id=i, content=vocab[i][0], **special) for i in range(3)]
    added += [dict(id=i, content=vocab[i][0], **special) for i in range(len(vocab) - extra, len(vocab))]
    meta = dict(type="Metaspace", replacement="\u2581", prepend_scheme="always", split=True)
    eos = vocab.index(["</s>", 0.0])
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": dict(type="Precompiled", precompiled_charsmap=base64.b64encode(charsmap).decode()),
        "pre_tokenizer": dict(type="Sequence", pretokenizers=[dict(type="WhitespaceSplit"), meta]),
        "post_processor": dict(
            type="TemplateProcessing",
            single=[{"Sequence": dict(id="A", type_id=0)}, {"SpecialToken": dict(id="</s>", type_id=0)}],
            pair=[{"Sequence": dict(id="A", type_id=0)}, {"SpecialToken": dict(id="</s>", type_id=0)},
                  {"Sequence": dict(id="B", type_id=0)}, {"SpecialToken": dict(id="</s>", type_id=0)}],
            special_tokens={"</s>": dict(id="</s>", ids=[eos], tokens=["</s>"])}),
        "decoder": meta,
        "model": dict(type="Unigram", unk_id=unk, vocab=vocab, byte_fallback=False),
    }


def export_t5(root: Path, out: Path, c: T5Config, check: bool, ref: Path | None, layers: int | None) -> None:
    w = t5_checkpoint(root, ARGS.random)
    iron_sana.TEXT_FMT["fmt"] = ARGS.t5_format
    iron_sana.PREC.text = HIGHEST
    t0 = time.time()
    n = layers or c.layers
    G = c.group if c.group and n % c.group == 0 else 0
    if c.group and not G:
        print(f"t5: {n} layers do not split into groups of {c.group}: one graph", flush=True)
        c = dataclasses.replace(c, group=0)
    g = T5(c, w, G or layers)
    print(f"t5: weights in {time.time() - t0:.0f} s", flush=True)
    cg = g.compile(x=(c.rows, c.hidden), mask=(c.rows,))
    d = out / "t5"
    meta = dict(config=dataclasses.asdict(c), weights=ARGS.t5_format, precision=HIGHEST.value,
                layers=n, groups=n // G if G else 1)
    m = export_graph(cg, d, "t5", meta)
    print(f"t5: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    if G:
        variants = [{k: b["image"] for k, b in m["banks"].items()}]
        for i in range(1, n // G):
            variants.append(variant_bank(cg, T5(c, w, G, i * G), d, f"bank_g{i}"))
            print(f"t5: bank of layers {i * G}..{(i + 1) * G - 1}", flush=True)
        m["variants"] = variants
        (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    import shutil

    for f in ("spiece.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        if (root / "tokenizer" / f).exists():
            shutil.copyfile(root / "tokenizer" / f, d / f)
    # the Go tokenizer reads tokenizer.json in the form t5_tokenizer_json writes (a repository's
    # own may have a normalizer Sequence, FLUX.1's does)
    (d / "tokenizer.json").write_text(json.dumps(t5_tokenizer_json(root / "tokenizer"), ensure_ascii=False))
    with open(d / "embed.bin", "wb") as f:
        e = w["shared.weight"]
        for s in range(0, e.shape[0], 4096):
            f.write(bf(e[s : s + 4096]).tobytes())
    if ref is not None:
        z = np.load(ref)
        x, mask = bf(z["x"]), bf(z["mask"].astype(np.float32))
        t1 = time.time()
        if G:  # the groups in turn, the last one's norm
            h = x
            for i in range(n // G):
                h, y = (g if i == 0 else T5(c, w, G, i * G)).reference(h, mask)
            y = f32(y)
        else:
            y = f32(g.reference(x, mask))
        n = int(z["mask"].sum())
        want = z["y"]
        keep = n + 1  # the prompt's rows and the first pad's, which the DiT reads
        err = np.abs(y[:keep] - want[:keep])
        rows = [cosine(y[i], want[i]) for i in range(keep)]
        print(f"t5 against float32 ({n} tokens, {time.time() - t1:.0f} s): cosine {cosine(y[:keep], want[:keep]):.6f}, "
              f"rows' least {min(rows):.6f} (row {int(np.argmin(rows))}), max |err| {err.max():.4g} "
              f"(|ref| max {np.abs(want[:keep]).max():.4g}), rel RMS {np.sqrt((err ** 2).mean() / (want[:keep] ** 2).mean()):.4g}",
              flush=True)
        if check:
            write_check(g, m, d, dict(x=x, mask=mask))
    elif check:
        rng = np.random.default_rng(0)
        e = w["shared.weight"]
        ids = np.concatenate([rng.integers(3, 32000, 40), [1], np.zeros(c.rows - 41, np.int64)])
        mask = np.zeros(c.rows, np.float32)
        mask[:41] = 1
        write_check(g, m, d, dict(x=bf(f32(e[ids])), mask=bf(mask)))


# --- the FLUX VAE decoder ------------------------------------------------------------------


GROUPS = 32
GN_EPS = 1e-6
GN_ROW = 4096  # the transposed pixels' rows (step 3)
GN_C = 2.0 ** -5  # the constant beside a group's row norms (step 5)
BIG_NORM = 1 << 26  # a norm (or 1 x 1 convolution) over more elements is cut into commands (FluxVaeStage.cut)
GN_PIXELS = 262144  # a big norm's transposes and row sums take this many pixels a command
ATTN_ROWS = 4096  # the mid block's attention over at most this many pixels in one pass


@dataclasses.dataclass(frozen=True)
class NormShape:
    """A GroupNorm's geometry over np pixels of C channels (FluxVaeStage.group_norm)."""

    np: int
    C: int

    @property
    def cg(self) -> int:
        return self.C // GROUPS

    @property
    def q(self) -> int:
        """The column sums' fold: the pixels viewed (np / q, q C), np / q at most 4096, but q C at
        most 32768: the GEMM walks B's rows of q C in 512-column blocks, and a shim DMA's outer
        count is 64 at most (1024 x 1024's 256-channel norms took np / q 8192)."""
        ok = [q for q in range(1, self.np + 1) if self.np % q == 0 and (self.np // q) % 64 == 0]
        q = next(q for q in ok if self.np // q <= 4096)
        if q * self.C > 32768:
            q = max(q for q in ok if q * self.C <= 32768)
        return q

    @property
    def L(self) -> int:
        return max(d for d in range(64, GN_ROW + 1, 64) if self.np % d == 0)

    @property
    def R(self) -> int:
        return self.C * self.np // self.L

    @property
    def k(self) -> int:
        """A group's rows (row norms)."""
        return self.cg * self.np // self.L

    @property
    def W(self) -> int:
        """The width of a group's row of norms and the constant."""
        return -(-(self.k + 1) // 64) * 64

    def __post_init__(self):
        assert self.C % GROUPS == 0 and self.R % 64 == 0 and self.k + 1 <= 4096, self
        assert self.np * self.cg * GN_EPS > GN_C**2, self  # step 5's eps is positive


class FluxVaeStage(qi.VaeStage):
    """iron_anima.VaeStage for FLUX's decoder (2-D convolutions, 16 latent channels, GroupNorm).

    Ops: ("in",) the latents' unscaling (a 1 x 1 affine of the 16 channels, in a grid of 64) and
    conv_in; ("res", path, cin, cout); ("attn", path); ("up", path, cin, cout) nearest 2x and the
    3 x 3 convolution; ("head",) conv_norm_out, SiLU, conv_out (to 64 channels, RGB first).
    """

    def __init__(self, ops: list, g: GridShape, w, vae_cfg: dict):
        self.ops, self.g_in = ops, g
        self.params, self._index, self.states = [], {}, []
        self._consts = {}
        self._gn_grids = []
        sc, sh = float(vae_cfg["scaling_factor"]), float(vae_cfg["shift_factor"])
        gg = g
        for op in ops:
            kind = op[0]
            if kind == "in":
                a = np.zeros((64, 64), np.float32)
                a[np.arange(16), np.arange(16)] = 1 / sc
                b = np.zeros(64, np.float32)
                b[:16] = sh
                self._param("in.w", a)
                self._param("in.b", b)
                self._conv3("conv_in", w, "decoder.conv_in", 64)
                gg = GridShape(gg.H, gg.W, f32(w["decoder.conv_in.weight"]).shape[0])
            elif kind == "res":
                _, path, cin, cout = op
                self._gn(f"{path}.norm1", w, path + ".norm1", NormShape(gg.H * gg.W, cin))
                self._conv3(f"{path}.conv1", w, f"{path}.conv1", cin)
                self._gn(f"{path}.norm2", w, path + ".norm2", NormShape(gg.H * gg.W, cout))
                self._conv3(f"{path}.conv2", w, f"{path}.conv2", cout)
                if cin != cout:
                    self._param(f"{path}.sc.w", f32(w[f"{path}.conv_shortcut.weight"])[:, :, 0, 0])
                    self._param(f"{path}.sc.b", f32(w[f"{path}.conv_shortcut.bias"]))
                gg = GridShape(gg.H, gg.W, cout)
            elif kind == "attn":
                _, path = op
                C = gg.C
                self._gn(f"{path}.norm", w, path + ".group_norm", NormShape(gg.H * gg.W, C))
                s = C ** -0.5  # the scores' scale, folded into the queries
                self._param(f"{path}.q.w", f32(w[f"{path}.to_q.weight"]) * s)
                self._param(f"{path}.q.b", f32(w[f"{path}.to_q.bias"]) * s)
                for n in "kv":
                    self._param(f"{path}.{n}.w", f32(w[f"{path}.to_{n}.weight"]))
                    self._param(f"{path}.{n}.b", f32(w[f"{path}.to_{n}.bias"]))
                self._param(f"{path}.proj.w", f32(w[f"{path}.to_out.0.weight"]))
                self._param(f"{path}.proj.b", f32(w[f"{path}.to_out.0.bias"]))
                self._const(f"eye{C}", np.eye(C, dtype=np.float32))
            elif kind == "up":
                _, path, cin, cout = op
                wt = f32(w[f"{path}.upsamplers.0.conv.weight"])  # (cout, cin, 3, 3)
                self._up_phases(f"{path}.up", wt)
                self._param(f"{path}.up.b", f32(w[f"{path}.upsamplers.0.conv.bias"]))
                gg = GridShape(2 * gg.H, 2 * gg.W, cout)
            elif kind == "head":
                self._gn("head.norm", w, "decoder.conv_norm_out", NormShape(gg.H * gg.W, gg.C))
                wt = f32(w["decoder.conv_out.weight"])  # (3, 128, 3, 3)
                full = np.zeros((64,) + wt.shape[1:], np.float32)
                full[:3] = wt
                self._conv_weight("head", full.transpose(0, 2, 3, 1).reshape(64, -1))
                b = np.zeros(64, np.float32)
                b[:3] = f32(w["decoder.conv_out.bias"])
                self._param("head.b", b)
            else:
                raise ValueError(op)

    def _const(self, name: str, x) -> None:
        """A constant shared by the stage's norms (made once)."""
        if name not in self._index:
            self._param(name, x)

    def _gn(self, name: str, w, path: str, s: NormShape) -> None:
        """A GroupNorm's constants (group_norm): its own (the group means' fold Gm, the residual
        means' Gr, gamma and the step-5 factor spread to the channels, beta) and the shared ones."""
        C, cg, q, k = s.C, s.cg, s.q, s.k
        grp = np.arange(C) // cg
        same = (grp[:, None] == grp[None, :]).astype(np.float32)
        self._param(f"{name}.gm", np.tile(same, (1, q)) * (-1.0 / (s.np * cg)))  # (C, q C)
        P = s.np if s.np * C <= BIG_NORM else min(s.np, GN_PIXELS)  # group_norm's chunk of pixels
        Rc, kc = C * P // s.L, cg * P // s.L
        self._param(f"{name}.gr", (grp[:, None] == (np.arange(Rc) // kc)[None, :]).astype(np.float32) / (s.np * cg))
        gamma = f32(w[f"{path}.weight"])
        factor = math.sqrt(s.np * cg) / (GN_C * math.sqrt(s.W))
        gam = np.zeros((C, 64), np.float32)
        gam[np.arange(C), grp] = gamma * factor
        self._param(f"{name}.gamma", gam)
        self._param(f"{name}.beta", f32(w[f"{path}.bias"]))
        self._const(f"ones{s.np // q}", np.ones((16, s.np // q), np.float32))
        self._const(f"ones{s.L}", np.ones((16, s.L), np.float32))
        self._const(f"root{s.L}", np.full((16, s.L), 1 / math.sqrt(s.L), np.float32))
        sel = np.zeros((16, s.W), np.float32)
        sel[0, k] = 1
        self._const(f"sel{s.W}.{k}", sel)
        tail = np.zeros((GROUPS, s.W - k), np.float32)
        tail[:, 0] = GN_C
        self._const(f"tail{s.W}.{k}", tail)
        self._const(f"eye{C}", np.eye(C, dtype=np.float32))

    # --- inside body ---

    def group_norm(self, d, s: NormShape, name: str):
        """GroupNorm(32) of the dense pixels d (np, C), gamma and beta: (np, C) (module doc)."""
        C, q, L, R, k, W = s.C, s.q, s.L, s.R, s.k, s.W
        # 1. minus the group means, per channel (bf16)
        cs = GEMM.step(self.w(f"ones{s.np // q}"), d.reshape(s.np // q, q * C), b_col_maj=False, precision=HIGHEST)
        negm = F.linear(cs, self.w(f"{name}.gm"), precision=HIGHEST)  # (16, C)
        cut = self.cut(s)
        cut()
        # 2. centred
        ce = d + negm[0:1].reshape(C)
        cut()
        # 3-4 over chunks of at most GN_PIXELS pixels (a big norm's; one chunk otherwise): each
        # chunk transposed, (C, P), in rows of L values of a channel (a group's k_c rows contiguous),
        # each row's norm sum(rms_norm(u) u) / sqrt(L) into the group's row of norms (step 5), and
        # the rows' sums folded into the residual means
        fin = iron.state((64, W))
        self.states.append(fin)
        P = s.np if s.np * C <= BIG_NORM else min(s.np, GN_PIXELS)
        Rc, kc = C * P // L, s.cg * P // L
        r = None
        for p0 in range(0, s.np, P):
            rows = F.linear(self.w(f"eye{C}"), ce[p0 : p0 + P], precision=HIGHEST).reshape(Rc, L)
            cut()
            y = F.rms_norm(rows, (L,), None, 1e-30)
            cut()
            yu = y * rows
            cut()
            norms = F.linear(self.w(f"root{L}"), yu, precision=HIGHEST)  # (16, Rc): row 0
            sums = F.linear(self.w(f"ones{L}"), rows, precision=HIGHEST)  # (16, Rc): row 0
            j = p0 // P
            Copy.step(norms, fin, src=tap((16, Rc), 0, [GROUPS, kc], [kc, 1]),
                      dst=tap((64, W), j * kc, [GROUPS, kc], [W, 1]))
            rc = F.linear(sums, self.w(f"{name}.gr"), precision=HIGHEST)[0:1].reshape(C)  # the centred values' group means
            r = rc if r is None else r + rc
            cut()
        # 5. a group's k norms and the constant in a row of W, RMS-normalized
        Copy.step(self.w(f"tail{W}.{k}"), fin, dst=tap((64, W), k, [GROUPS, W - k], [W, 1]))
        eps = (s.np * s.cg * GN_EPS - GN_C**2) / W
        yf = F.rms_norm(self.view(fin), (W,), None, eps)
        col = F.linear(self.w(f"sel{W}.{k}"), yf, precision=HIGHEST)  # (16, 64): row 0 the groups' values
        a = F.linear(col, self.w(f"{name}.gamma"), precision=HIGHEST)[0:1].reshape(C)  # gamma / sqrt(var + eps)
        # 6. y = (d - m~) a + (beta - r a)
        b = (r * a) * -1.0 + self.w(f"{name}.beta")
        cut()
        y = ce * a
        cut()
        return y + b

    @staticmethod
    def cut(s: NormShape):
        """command_boundary for a norm over more than BIG_NORM elements (1024 x 1024's last block:
        each of its passes over 0.5 GB is its own command, under amdxdna's 2 s), else nothing."""
        big = s.np * s.C > BIG_NORM
        return command_boundary if big else (lambda: None)

    def dense(self, x, g: GridShape):
        n, C, Wp = g.H * g.W, g.C, g.Wp
        return Copy.step(x, src=tap((g.rows, C), g.pixel(0, 0) * C, [g.H, g.W * C], [Wp * C, 1])).reshape(n, C)

    def to_grid(self, y, g: GridShape):
        out = self.new_grid(g)
        Copy.step(y, out, dst=tap((g.rows, g.C), g.pixel(0, 0) * g.C, [g.H, g.W * g.C], [g.Wp * g.C, 1]))
        return self.view(out)

    def norm_silu(self, x, g: GridShape, name: str):
        """GroupNorm and SiLU of the grid x into a fresh grid (its pads zero)."""
        s = NormShape(g.H * g.W, g.C)
        cut = self.cut(s)
        d = self.dense(x, g)
        cut()
        y = self.group_norm(d, s, name)
        cut()
        y = F.silu(y)
        cut()
        out = self.to_grid(y, g)
        cut()
        return out

    def conv1(self, x, g: GridShape, wname: str, cout: int):
        """iron_anima's 1 x 1 convolution, a command every 128 image rows on a big grid."""
        if g.H * g.W * g.C <= BIG_NORM:
            return super().conv1(x, g, wname, cout)
        gout = GridShape(g.H, g.W, cout)
        out = self.new_grid(gout)
        R = g.chunk(g.C)
        for r0 in range(0, g.H, R):
            if r0 and r0 % 128 == 0:
                command_boundary()
            s0 = g.band + r0 * g.Wp
            y = F.linear(x[s0 : s0 + R * g.Wp], self.w(f"{wname}.w"), self.w(f"{wname}.b"), precision=iron_anima.PREC.vae)
            self.interior_from(y, gout, out, r0, R)
        command_boundary()
        return self.view(out), gout

    def attn(self, x, g: GridShape, path: str):
        """GroupNorm, then iron_qwenimage's one-head attention as GEMMs, the residual."""
        n, C = g.H * g.W, g.C
        d = self.dense(x, g)
        h = self.group_norm(d, NormShape(n, C), f"{path}.norm")
        qh, kh, vh = (F.linear(h, self.w(f"{path}.{c}.w"), self.w(f"{path}.{c}.b"), precision=HIGHEST) for c in "qkv")
        vt = F.linear(self.w(f"eye{C}"), vh, precision=HIGHEST)  # (C, n): V's transpose, exactly
        # queries a pass: the scores (Qc, n); a row longer than 4096 streams, 48 rows a core at most
        Qc = n if n <= ATTN_ROWS else 256
        if Qc == n:
            o = F.linear(Softmax.step(F.linear(qh, kh, precision=HIGHEST)), vt, precision=HIGHEST)
        else:
            ob = iron.state((n, C))
            self.states.append(ob)
            for r0 in range(0, n, Qc):
                if r0 and r0 % (16 * Qc) == 0:
                    command_boundary()
                s = F.linear(qh[r0 : r0 + Qc], kh, precision=HIGHEST)  # (Qc, n)
                ob[r0 : r0 + Qc] = F.linear(Softmax.step(s), vt, precision=HIGHEST)
            o = self.view(ob)
        o = d + F.linear(o, self.w(f"{path}.proj.w"), self.w(f"{path}.proj.b"), precision=iron_anima.PREC.vae)
        return self.to_grid(o, g), g

    def body(self, x):
        g = self.g_in
        for op in self.ops:
            kind = op[0]
            if kind == "in":
                a, ga = self.conv1(x, g, "in", 64)
                x, g = self.conv3(a, ga, "conv_in", self.w("conv_in.b").shape[0])
            elif kind == "res":
                x, g = self.res(x, g, op[1], op[3])
            elif kind == "attn":
                x, g = self.attn(x, g, op[1])
            elif kind == "up":
                x, g = self.up(x, g, op[1], op[3])
            elif kind == "head":
                h = self.norm_silu(x, g, "head.norm")
                y, gy = self.conv3(h, g, "head", 64)
                n = g.H * g.W
                return Copy.step(y, src=tap((gy.rows, 64), gy.pixel(0, 0) * 64, [g.H, g.W * 64], [gy.Wp * 64, 1])).reshape(n, 64)
            if op is not self.ops[-1]:
                command_boundary()
        self.g_out = g
        return x if self.ops[-1][0] == "res" else Copy.step(x)


def vae_stages(cfg: dict, width: int, height: int) -> list:
    """The decoder in stages (each under amdxdna's 2 s), with each one's input grid."""
    h, w = height // 8, width // 8
    d = "decoder"
    ch = list(reversed(cfg["block_out_channels"]))  # 512 512 256 128
    n = cfg["layers_per_block"] + 1
    mid = ch[0]
    s = [([("in",), ("res", f"{d}.mid_block.resnets.0", mid, mid), ("attn", f"{d}.mid_block.attentions.0"),
           ("res", f"{d}.mid_block.resnets.1", mid, mid)], GridShape(h, w, 64))]
    prev = mid
    for i, cout in enumerate(ch):
        p = f"{d}.up_blocks.{i}"
        res = [("res", f"{p}.resnets.{j}", prev if j == 0 else cout, cout) for j in range(n)]
        up = [("up", p, cout, cout)] if i < len(ch) - 1 else []
        if i < 2:  # 64^2 and 128^2 (at 512 x 512): the block in one stage
            s.append((res + up, GridShape(h, w, prev)))
        else:  # a residual block a stage
            for j, r in enumerate(res):
                s.append(([r] + (up if j == n - 1 else []), GridShape(h, w, r[2])))
        prev = cout
        if up:
            h, w = 2 * h, 2 * w
    s.append(([("head",)], GridShape(h, w, ch[-1])))
    return s


def vae_config(root: Path) -> dict:
    p = root / "vae" / "config.json"
    if not p.exists() and ARGS.vae_config is not None:
        p = ARGS.vae_config / "config.json"
    return json.loads(p.read_text())


def grid_of(dense: np.ndarray, g: GridShape) -> np.ndarray:
    """Dense channel-last pixels (H W, C) in a zero-padded grid (rows, g.C)."""
    x = np.zeros((g.rows, g.C), np.float32)
    C = dense.shape[1]
    for y in range(g.H):
        x[g.pixel(y, 0) : g.pixel(y, 0) + g.W, :C] = dense[y * g.W : (y + 1) * g.W]
    return x


def dense_of(grid: np.ndarray, g: GridShape) -> np.ndarray:
    return np.concatenate([grid[g.pixel(y, 0) : g.pixel(y, 0) + g.W] for y in range(g.H)])


def export_vae(root: Path, out: Path, width: int, height: int, check: bool, ref: Path | None,
               only: list | None = None) -> None:
    cls = iron_anima.RandomWeights if ARGS.random else iron_anima.Safetensors
    w = cls(root / "vae" / "diffusion_pytorch_model.safetensors")
    cfg = vae_config(root)
    stages = vae_stages(cfg, width, height)
    d = out / "vae"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    z = np.load(ref) if ref is not None else None
    x_ref = None
    if z is not None:  # the latents (16, h, w) in the first grid
        lat = z["latents"]
        x_ref = grid_of(lat.reshape(16, -1).T, stages[0][1])
    for i, (ops, g) in enumerate(stages):
        if only and i not in only:
            continue
        t0 = time.time()
        st = FluxVaeStage(ops, g, w, cfg)
        cg = st.compile(x=(g.rows, g.C))
        go = st.g_out if ops[-1][0] != "head" else None
        meta = dict(stage=i, ops=json.loads(json.dumps(ops)), grid_in=dataclasses.asdict(g),
                    grid_out=None if go is None else dataclasses.asdict(go), pad=iron_anima.VAE_PAD,
                    width=width, height=height, scaling_factor=cfg["scaling_factor"], shift_factor=cfg["shift_factor"])
        m = export_graph(cg, d / f"s{i}", f"vae_s{i}", meta)
        print(f"vae s{i}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
        if x_ref is not None:
            t1 = time.time()
            x_in = x_ref
            y = f32(st.reference(bf(x_in)))
            last = ops[-1]
            if go is not None:
                got, x_ref = dense_of(y, go), y
            else:
                got = y[:, :3]
            end = "image" if last[0] == "head" else _block_end(last, cfg)
            msg = f"vae s{i}: reference in {time.time() - t1:.0f} s"
            if end is not None:
                want = z["image"].reshape(3, -1).T if end == "image" else z[end]
                err = np.abs(got - want)
                msg += (f"; {end} against float32: cosine {cosine(got, want):.6f}, max |err| {err.max():.4g} "
                        f"(|ref| max {np.abs(want).max():.4g}), rel RMS {np.sqrt((err ** 2).mean() / (want ** 2).mean()):.4g}")
                if end == "image":
                    mse = float(((np.clip(got, -1, 1) - np.clip(want, -1, 1)) ** 2).mean()) / 4  # on [0, 1]
                    msg += f", PSNR {10 * math.log10(1 / max(mse, 1e-20)):.1f} dB"
            print(msg, flush=True)
            if check:
                write_check(st, m, d / f"s{i}", dict(x=bf(x_in)))
        elif check:
            write_check(st, m, d / f"s{i}", dict(x=bf(_check_input(st, g, rng))))
    (d / "manifest.json").write_text(json.dumps(dict(stages=len(stages), width=width, height=height, pad=iron_anima.VAE_PAD,
                                                     channels=3, latent=16, scaling_factor=cfg["scaling_factor"],
                                                     shift_factor=cfg["shift_factor"]), indent=1) + "\n")


def _check_input(st, g: GridShape, rng) -> np.ndarray:
    x = np.zeros((g.rows, g.C), np.float32)
    C = 16 if st.ops[0][0] == "in" else g.C
    for y in range(g.H):
        x[g.pixel(y, 0) : g.pixel(y, 0) + g.W, :C] = rng.standard_normal((g.W, C))
    return x


def _block_end(op, cfg) -> str | None:
    """The reference's name for the output of the block op ends, if it ends one."""
    path = op[1]
    n = cfg["layers_per_block"] + 1
    parts = path.split(".")
    if "mid_block" in parts:
        return "mid" if path.endswith(f"resnets.{1}") else None
    i = int(parts[parts.index("up_blocks") + 1])
    if op[0] == "up":
        return f"up{i}"
    if path.endswith(f"resnets.{n - 1}") and i == len(cfg["block_out_channels"]) - 1:
        return f"up{i}"
    return None


# --- export ------------------------------------------------------------------------------

ARGS = None


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the diffusers repository (Chroma1-Flash)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", nargs="*", default=["t5", "vae"], help="parts to export: t5, vae")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--stages", type=int, nargs="*", help="the VAE's stages to export (default all)")
    ap.add_argument("--vae-config", type=Path, help="a directory with the VAE's config.json when ROOT/vae has none "
                    "(Chroma1-Flash's vae/ holds the weights only: FLUX.1-schnell's vae/)")
    ap.add_argument("--vae-precision", default=iron_anima.PREC.vae.value, choices=[p.value for p in Precision],
                    help="the VAE's convolutions (medium: bfp16 operands, as Anima's; IRON's CPU reference does not round to bfp16)")
    ap.add_argument("--t5-format", default="q8", choices=("bf16", "q8", "q6", "q4"))
    ap.add_argument("--t5-layers", type=int, default=0, help="the encoder's first N layers only (a test)")
    ap.add_argument("--t5-split", type=int, default=T5Config.split, help="T5 layers a command runs")
    ap.add_argument("--t5-group", type=int, default=T5Config.group, help="T5 layers a bank holds, read in turn (0: one graph)")
    ap.add_argument("--t5-gelu", default=T5Config.gelu, choices=("tanh", "none"))
    ap.add_argument("--t5-bf16", nargs="*", default=list(T5Config.bf16), help="T5 projections kept bf16 (q k v o wi_0 wi_1 wo)")
    ap.add_argument("--t5-residual", default=T5Config.residual, choices=("pair", "bf16"))
    ap.add_argument("--ref-t5", type=Path, help="ref_chroma_parts.py t5's npz: compare IRON's reference with it")
    ap.add_argument("--ref-vae", type=Path, help="ref_chroma_parts.py vae's npz: run the stages' references in turn on it")
    ap.add_argument("--random", action="store_true", help="weights drawn at random in the checkpoint's shapes")
    ap.add_argument("--check", action="store_true", help="write each graph's inputs and reference outputs")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    import iron_q4

    iron_q4.install_linear()
    iron_anima.PREC.vae = Precision(a.vae_precision)
    bind_device(a.device, a.cols)
    a.out.mkdir(parents=True, exist_ok=True)
    if "t5" in a.only:
        c = T5Config(split=a.t5_split, group=a.t5_group, gelu=a.t5_gelu, residual=a.t5_residual, bf16=tuple(a.t5_bf16))
        export_t5(a.root, a.out, c, a.check, a.ref_t5, a.t5_layers or None)
    if "vae" in a.only:
        export_vae(a.root, a.out, a.width or a.size, a.height or a.size, a.check, a.ref_vae, a.stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
