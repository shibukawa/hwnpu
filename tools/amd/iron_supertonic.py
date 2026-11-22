#!/usr/bin/env python3
"""Supertonic 3's four graphs as IRON graphs, which internal/supertonic runs on AMD's NPU: the flow
(vector_estimator.onnx), the vocoder (vocoder.onnx), the text encoder (text_encoder.onnx) and the
duration predictor's sentence encoder (duration_predictor.onnx; TextGraph, its windowed relative
attention on the host).

  python tools/amd/iron_supertonic.py ONNX_DIR OUT [--sizes 64x32 128x64 256x128] [--check]

writes OUT/estimator-tT-lL/, OUT/vocoder-lL/, OUT/encoder-tT/ and OUT/duration-tT/ (iron_graph.py)
for each size (T characters of the text, L frames of the latent: internal/supertonic.Sizes),
OUT/host.safetensors (what the host computes: the step's time embedding, the text graphs'
relative attention, the duration's last layers) and OUT/model.json.

The layout: a tensor is rows of time, its channels the columns, bf16.

The flow (one step, both branches of the classifier-free guidance at once): the stream is the
conditioned branch's L frames and the unconditioned's, each between P zero rows (rows P .. P + L
and 3 P + L .. 3 P + 2 L of S = 2 (L + 2 P)), in a state with M zero rows before and after.
Per-row operators (the GEMMs, layer norms, GELU) take all S rows; the mask (an input, ones on
the frames of the speech) zeroes the rest after every block. A ConvNeXt block's depthwise
convolution (5 taps, dilation 1 to 8, the graph's edge padding: iron_dwconv1d.py) reads the
stream with the first frame of each branch copied into its pad rows before it. The text's
attention (8 heads of 64, RoPE on the query and the key from angle tables the host gives, the
text's characters a per-call value) runs on the conditioned branch alone: the unconditioned
branch's text is one token at every position, so its attention gives that token's value row
(a constant). The style's attention (2 heads of 128 over 50 tokens) runs on each branch, its
keys constants (tanh of the projected key tokens of each branch), the unconditioned values too.
The step's time embedding (a sinusoid through an MLP, then one Linear a time block) is a row the
host computes (host.safetensors); the guidance (v = 4 v_cond - 3 v_uncond) and the step's
update are the host's too: the graph gives both branches' velocities.

The vocoder: 6 L frames of 24 channels (the host divides the latent by the normalizer's 0.25 and
repeats the first frame 6 times before: the embedding convolution's causal edge padding; the
unnormalization is folded into the embedding's weights), the embedding
convolution (an im2col and a GEMM), 10 ConvNeXt blocks (7 causal taps, dilation 1, 2, 4), the
final batch norm folded into the head's first convolution (3 causal taps, an im2col), ReLU (the
graph's PReLU of slope 2e-4), the head's second GEMM: 512 samples a frame.

Run in IRON's virtual environment; no NPU is needed. See docs/amd-npu-models.md.
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
from iron.operators.copy import Copy
from iron.operators.elementwise_add import ElementwiseAdd
from iron.operators.elementwise_mul import ElementwiseMul
from iron.operators.mha import MHA
from iron.operators.relu import ReLU

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_anima import bf, command_boundary, tap  # noqa: E402
from iron_dwconv1d import dwconv  # noqa: E402
from iron_biasgelu import bias_gelu  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

PREC = Precision.HIGHEST
SIZES = [(64, 32), (128, 64), (256, 128)]  # internal/supertonic.Sizes
LATENT, TEXT, HID, FF = 144, 256, 512, 2048
XP = 192  # the latent's channels padded for the GEMMs
P = 16  # zero rows around each branch of the flow's stream (the widest halo: 5 taps of dilation 8)
M = 16  # zero rows before and after the flow's stream state
VM = 24  # zero rows before the vocoder's stream state (7 taps of dilation 4)
STYLE, SP = 50, 128  # the style's tokens, padded (IRON's MHA reads at least as many keys as queries: up to 128)
EP = "vector_estimator.tts.ttl.vector_field."
EN = "/vector_estimator/vector_field/"
VP = "tts.ae.decoder."


class Onnx:
    """An ONNX graph's initializers (float64) and nodes by name."""

    def __init__(self, path: Path):
        import onnx
        from onnx import numpy_helper

        m = onnx.load(str(path))
        self.init = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in m.graph.initializer}
        self.nodes = {n.name: n for n in m.graph.node}
        self.consts = {}
        for n in m.graph.node:
            if n.op_type == "Constant" and n.attribute[0].type == 4:
                self.consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t).astype(np.float64)

    def __getitem__(self, k):
        return self.init[k]

    def w(self, node: str, i: int = 1):
        x = self.nodes[node].input[i]
        return self.init[x] if x in self.init else self.consts[x]

    def dilation(self, p: str) -> int:
        """The depthwise convolution's dilation of the ConvNeXt block of initializer prefix p."""
        for n in self.nodes.values():
            if n.op_type == "Conv" and any(i.startswith(p + ".dwconv") for i in n.input):
                return int([a for a in n.attribute if a.name == "dilations"][0].ints[0])
        raise KeyError(p)


# --- the reference in float64 (checked against ONNX Runtime) ------------------------------------


def _erf(x):
    return np.vectorize(math.erf)(x)


def gelu_ref(x):
    return 0.5 * x * (1 + _erf(x / math.sqrt(2)))


def ln_ref(x, g, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * g + b


def convnext_ref(G: Onnx, p: str, x, m, causal: bool):
    """A ConvNeXt block over x (T, C) time-major; m (T, 1) the mask or None."""
    wn = p + ".dwconv.weight" if p + ".dwconv.weight" in G.init else p + ".dwconv.net.weight"
    w, b = G[wn], G[wn[: -len("weight")] + "bias"]
    d, K = G.dilation(p), w.shape[2]
    pad = d * (K - 1)
    xm = x * m if m is not None else x
    left, right = (pad, 0) if causal else (pad // 2, pad // 2)
    T = x.shape[0]
    xp = np.concatenate([np.repeat(xm[:1], left, 0), xm, np.repeat(xm[-1:], right, 0)])
    h = np.zeros_like(xm) + b
    for j in range(K):
        h += xp[j * d : j * d + T] * w[:, 0, j]
    if m is not None:
        h = h * m
    h = ln_ref(h, G[p + ".norm.norm.weight"], G[p + ".norm.norm.bias"])
    h = gelu_ref(h @ G[p + ".pwconv1.weight"][:, :, 0].T + G[p + ".pwconv1.bias"])
    h = (h @ G[p + ".pwconv2.weight"][:, :, 0].T + G[p + ".pwconv2.bias"]) * G[p + ".gamma"].reshape(-1)
    y = xm + h
    return y * m if m is not None else y


def _kinds():
    out = []
    for i in range(24):
        r = i % 6
        if r == 0:
            out += [("convnext", EP + f"main_blocks.{i}.convnext.{j}") for j in range(4)]
        elif r in (2, 4):
            out.append(("convnext", EP + f"main_blocks.{i}.convnext.0"))
        elif r == 1:
            out.append(("time", i))
        elif r == 3:
            out.append(("text", i))
        else:
            out.append(("style", i))
    out += [("convnext", EP + f"last_convnext.convnext.{j}") for j in range(4)]
    return out


BLOCKS = _kinds()
TIME_BLOCKS = [i for k, i in BLOCKS if k == "time"]


def time_embed(E: Onnx, step: float, total: float):
    """The step's time embedding (64): the sinusoid of step / total through the MLP."""
    f = E[EN + "time_encoder/sinusoidal/Constant_3_output_0"].reshape(-1) if EN + "time_encoder/sinusoidal/Constant_3_output_0" in E.init else E.consts[EN + "time_encoder/sinusoidal/Constant_3_output_0"].reshape(-1)
    a = step / total * 1000.0 * f
    s = np.concatenate([np.sin(a), np.cos(a)])
    h = s @ E[EP + "time_encoder.mlp.0.linear.weight"].T + E[EP + "time_encoder.mlp.0.linear.bias"]
    h = h * np.tanh(np.log1p(np.exp(h)))
    return h @ E[EP + "time_encoder.mlp.2.linear.weight"].T + E[EP + "time_encoder.mlp.2.linear.bias"]


def time_rows(E: Onnx, temb):
    """The time blocks' rows (4, 512) of a time embedding."""
    return np.stack([temb @ E.w(EN + f"main_blocks.{i}/linear/linear/MatMul") + E[EP + f"main_blocks.{i}.linear.linear.bias"]
                     for i in TIME_BLOCKS])


def rope_angles(E: Onnx, n: int, m: int):
    """RoPE's angles of m positions of a sequence of n: (m, 32), position / n * theta."""
    theta = E[EP + "main_blocks.3.attn.theta"].reshape(-1)
    return (np.arange(m) / n)[:, None] * theta[None, :]


def _rope(x, ang):
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :32], x[..., 32:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def text_attn_ref(E, i, h, m, text, tm):
    p, q_ = EN + f"main_blocks.{i}/attn/", EP + f"main_blocks.{i}.attn."
    L, T = h.shape[0], text.shape[0]
    q = (h @ E.w(p + "W_query/linear/MatMul") + E[q_ + "W_query.linear.bias"]).reshape(L, 8, 64)
    k = (text @ E.w(p + "W_key/linear/MatMul") + E[q_ + "W_key.linear.bias"]).reshape(T, 8, 64)
    v = (text @ E.w(p + "W_value/linear/MatMul") + E[q_ + "W_value.linear.bias"]).reshape(T, 8, 64)
    q, k = _rope(q, rope_angles(E, m.sum(), L)), _rope(k, rope_angles(E, tm.sum(), T))
    s = np.einsum("lhd,thd->hlt", q, k) / 16.0
    s = np.where(tm.reshape(1, 1, T) == 0, -np.inf, s)
    o = np.einsum("hlt,thd->lhd", _softmax(s) * m.reshape(1, L, 1), v).reshape(L, 512)
    o = o @ E.w(p + "out_fc/linear/MatMul") + E[q_ + "out_fc.linear.bias"]
    return ln_ref(o * m + h, E[EP + f"main_blocks.{i}.norm.norm.weight"], E[EP + f"main_blocks.{i}.norm.norm.bias"]) * m


def style_keys(E, i, keys):
    p = EN + f"main_blocks.{i}/attention/"
    return np.tanh(keys @ E.w(p + "W_key/linear/MatMul") + E[EP + f"main_blocks.{i}.attention.W_key.linear.bias"])


def style_values(E, i, style):
    p = EN + f"main_blocks.{i}/attention/"
    return style @ E.w(p + "W_value/linear/MatMul") + E[EP + f"main_blocks.{i}.attention.W_value.linear.bias"]


def style_attn_ref(E, i, h, m, style, k):
    p, q_ = EN + f"main_blocks.{i}/attention/", EP + f"main_blocks.{i}.attention."
    L = h.shape[0]
    q = (h @ E.w(p + "W_query/linear/MatMul") + E[q_ + "W_query.linear.bias"]).reshape(L, 2, 128)
    v = style_values(E, i, style).reshape(-1, 2, 128)
    a = _softmax(np.einsum("lhd,thd->hlt", q, k.reshape(-1, 2, 128)) / 16.0) * m.reshape(1, L, 1)
    o = np.einsum("hlt,thd->lhd", a, v).reshape(L, 256) @ E.w(p + "out_fc/linear/MatMul") + E[q_ + "out_fc.linear.bias"]
    return ln_ref(o * m + h, E[EP + f"main_blocks.{i}.norm.norm.weight"], E[EP + f"main_blocks.{i}.norm.norm.bias"]) * m


def branch_tokens(E: Onnx):
    """The style's key tokens of the conditioned branch and the unconditioned's, its value tokens
    and text token of the unconditioned branch."""
    return (E["/vector_estimator/Expand_output_0"].reshape(STYLE, 256),
            E["vector_estimator.tts.ttl.uncond_masker.style_key_special_token"].reshape(STYLE, 256),
            E["vector_estimator.tts.ttl.uncond_masker.style_value_special_token"].reshape(STYLE, 256),
            E["vector_estimator.tts.ttl.uncond_masker.text_special_token"].reshape(256))


def velocity_ref(E, x, text, style, keys, m, tm, tb):
    """One branch's velocity: x (L, 144), text (T, 256), style (50, 256), keys (50, 256)."""
    h = (x @ E[EP + "proj_in.net.weight"][:, :, 0].T) * m
    ti = 0
    for kind, p in BLOCKS:
        if kind == "convnext":
            h = convnext_ref(E, p, h, m, causal=False)
        elif kind == "time":
            h = (h + tb[ti]) * m
            ti += 1
        elif kind == "text":
            h = text_attn_ref(E, p, h, m, text, tm)
        else:
            h = style_attn_ref(E, p, h, m, style, style_keys(E, p, keys))
    return (h @ E[EP + "proj_out.net.weight"][:, :, 0].T) * m


def estimate_ref(E, x, text, style, m, tm, step, total):
    """The ONNX graph of batch 1: x (144, L), text (256, T), style (50, 256) -> (144, L)."""
    X, Tx = x.T, text.T
    m, tm = m.reshape(-1, 1), tm.reshape(-1, 1)
    tb = time_rows(E, time_embed(E, step, total))
    kc, ku, vu, tu = branch_tokens(E)
    vc = velocity_ref(E, X, Tx, style.reshape(STYLE, 256), kc, m, tm, tb)
    vu_ = velocity_ref(E, X, np.tile(tu, (Tx.shape[0], 1)), vu, ku, m, tm, tb)
    return ((X + (4 * vc - 3 * vu_) / total) * m).T


def vocode_ref(V: Onnx, z):
    """vocoder.onnx: z (144, L) -> (6 L * 512) samples."""
    L = z.shape[1]
    x = vocoder_input(V, z)
    w, b = V.w("/decoder/embed/net/Conv"), V.w("/decoder/embed/net/Conv", 2)
    h = np.zeros((6 * L, 512)) + b
    for j in range(7):
        h += x[j : j + 6 * L] @ w[:, :, j].T
    for i in range(10):
        h = convnext_ref(V, VP + f"convnext.{i}", h, None, causal=True)
    n = VP + "final_norm.norm."
    h = (h - V[n + "running_mean"]) / np.sqrt(V[n + "running_var"] + 1e-5) * V[n + "weight"] + V[n + "bias"]
    w1 = V[VP + "head.layer1.net.weight"]
    hp = np.concatenate([np.repeat(h[:1], 2, 0), h])
    y = np.zeros((6 * L, 2048)) + V[VP + "head.layer1.net.bias"]
    for j in range(3):
        y += hp[j : j + 6 * L] @ w1[:, :, j].T
    a = V.w("/decoder/head/act/PRelu").reshape(-1)[0]
    y = np.where(y >= 0, y, a * y)
    return (y @ V[VP + "head.layer2.weight"][:, :, 0].T).reshape(-1)


def vocoder_frames(z):
    """The vocoder's 6 L frames of 24 channels of a latent z (144, L), divided by the
    normalizer's scale (0.25): the graph's input after 6 copies of the first (the embedding's
    causal edge padding). The host's (internal/supertonic). The unnormalization (latent_std and
    latent_mean) is folded into the embedding: in bf16 the unnormalized frames, a mean and a
    deviation 40 times smaller, lost the speech to 29 dB."""
    L = z.shape[1]
    x = (np.asarray(z, np.float64) / 0.25).reshape(24, 6, L).transpose(0, 2, 1).reshape(24, 6 * L).T
    return np.concatenate([np.repeat(x[:1], 6, 0), x])


def vocoder_input(V: Onnx, z):
    """vocoder_frames unnormalized: the ONNX graph's embedding input, edge padded."""
    return vocoder_frames(z) * V["tts.ae.latent_std"].reshape(24) + V["tts.ae.latent_mean"].reshape(24)


# --- the graphs ----------------------------------------------------------------------------------


def lin(x, w, b=None):
    """x @ w.T (+ b), bf16 rows. HIGHEST: MEDIUM (bfp16, flm.ProjectionGEMM, rows by 256) ran the
    flow no faster (87 ms a step at 128 x 64: the operators' reconfiguration, not the products,
    takes the time) and further from the reference."""
    return F.linear(x, w, b, precision=PREC)


class Params:
    def __init__(self):
        self.params = []

    def p(self, x):
        h = iron.param(bf(x))
        self.params.append(h)
        return h


def dw_weights(w, b):
    """A depthwise Conv1d's weight (C, 1, K) and bias as DWConv1d's (K + 1, C)."""
    return np.concatenate([w[:, 0, :].T, b.reshape(1, -1)])


# pwconv1's out-features padded (the 2048 and 256, BiasGELU's whole vectors on 8 columns): the
# first padded one's weights 0 and bias ONE, whose GELU is ONE exactly (the tanh saturates), the
# others' GELU 0, so that pwconv2 takes its bias from a weight column (b2 / ONE). A column of ones
# copied into a state of the GELU's output instead made the vocoder's next call hang (its first
# command timed out).
FFP, ONE = FF + 256, 16.0


class ConvNext:
    """A ConvNeXt block's params: the layer norm's affine folded into pwconv1, gamma into pwconv2,
    pwconv1's bias added by BiasGELU (iron_biasgelu.py), pwconv2's a column of its weights (FFP:
    FF + 256 columns)."""

    def __init__(self, ps: Params, G: Onnx, p: str):
        wn = p + ".dwconv.weight" if p + ".dwconv.weight" in G.init else p + ".dwconv.net.weight"
        w, b = G[wn], G[wn[: -len("weight")] + "bias"]
        self.K, self.d = w.shape[2], G.dilation(p)
        self.dw = ps.p(dw_weights(w, b))
        g, beta = G[p + ".norm.norm.weight"], G[p + ".norm.norm.bias"]
        w1, b1 = G[p + ".pwconv1.weight"][:, :, 0], G[p + ".pwconv1.bias"]
        ff, C = w1.shape
        self.C, ffp = C, ff + 256
        w1p, b1p = np.zeros((ffp, C)), np.zeros(ffp)
        w1p[:ff], b1p[:ff], b1p[ff] = w1 * g[None, :], w1 @ beta + b1, ONE
        self.w1, self.b1 = ps.p(w1p), ps.p(b1p)
        gam = G[p + ".gamma"].reshape(-1)
        w2 = np.zeros((C, ffp))
        w2[:, :ff] = G[p + ".pwconv2.weight"][:, :, 0] * gam[:, None]
        w2[:, ff] = G[p + ".pwconv2.bias"] * gam / ONE
        self.w2 = ps.p(w2)

    def mlp(self, h):
        """pwconv2(GELU(pwconv1(layer_norm(h)))) times gamma."""
        n = F.layer_norm(h, (self.C,), None, None, eps=1e-6)
        return lin(bias_gelu(lin(n, self.w1), self.b1), self.w2)


class Estimator(iron.Graph, Params):
    """One step of the flow for a text of T characters and a latent of L frames (module doc)."""

    def __init__(self, E: Onnx, T: int, L: int, longest: int):
        Params.__init__(self)
        self.T, self.L = T, L
        self.Q = max(L, 64)  # an attention's query rows (IRON's MHA takes 64 or more)
        # the stream's rows: each branch between P zero rows (the widest halo) before and after
        self.P, self.seg = P, L + 2 * P
        self.S = 2 * self.seg
        self.w_in = self.p(np.pad(E[EP + "proj_in.net.weight"][:, :, 0], ((0, 0), (0, XP - LATENT))))
        self.w_out = self.p(np.pad(E[EP + "proj_out.net.weight"][:, :, 0], ((0, XP - LATENT), (0, 0))))
        kc, ku, vu, tu = branch_tokens(E)
        self.blocks = []
        consts = []  # the style attentions' constants, SP rows each
        for kind, p in BLOCKS:
            if kind == "convnext":
                self.blocks.append(("convnext", ConvNext(self, E, p)))
                continue
            if kind == "time":
                self.blocks.append(("time", None))
                continue
            i = p
            nb = dict(g=self.p(E[EP + f"main_blocks.{i}.norm.norm.weight"]), b=self.p(E[EP + f"main_blocks.{i}.norm.norm.bias"]))
            if kind == "text":
                a, q_ = EN + f"main_blocks.{i}/attn/", EP + f"main_blocks.{i}.attn."
                nb.update(
                    wq=self.p(E.w(a + "W_query/linear/MatMul").T), bq=self.p(E[q_ + "W_query.linear.bias"]),
                    wk=self.p(E.w(a + "W_key/linear/MatMul").T), bk=self.p(E[q_ + "W_key.linear.bias"]),
                    wv=self.p(E.w(a + "W_value/linear/MatMul").T), bv=self.p(E[q_ + "W_value.linear.bias"]),
                    wo=self.p(E.w(a + "out_fc/linear/MatMul").T), bo=self.p(E[q_ + "out_fc.linear.bias"]),
                    # the unconditioned branch's attention: its value row at every frame (of
                    # the longest graph: every weight then at the same place in each size's bank)
                    vu=self.p(np.tile(tu @ E.w(a + "W_value/linear/MatMul") + E[q_ + "W_value.linear.bias"], (longest, 1))))
                # (a branch's queries are its frames and the rows after them up to Q)
            else:
                a, q_ = EN + f"main_blocks.{i}/attention/", EP + f"main_blocks.{i}.attention."
                pad = lambda x: np.pad(x, ((0, SP - STYLE), (0, 0)))  # noqa: E731
                nb.update(
                    wq=self.p(E.w(a + "W_query/linear/MatMul").T), bq=self.p(E[q_ + "W_query.linear.bias"]),
                    wv=self.p(E.w(a + "W_value/linear/MatMul").T), bv=self.p(E[q_ + "W_value.linear.bias"]),
                    wo=self.p(E.w(a + "out_fc/linear/MatMul").T), bo=self.p(E[q_ + "out_fc.linear.bias"]),
                    sc=len(consts))
                # the keys of both branches and the unconditioned values: rows of `consts`
                consts += [pad(style_keys(E, i, kc)), pad(style_keys(E, i, ku)), pad(style_values(E, i, vu))]
            self.blocks.append((kind, nb))
        # IRON's MHA reads keys of a per-call count, which a param cannot be sliced to: a state
        # takes them, one copy a call
        self.nconst = len(consts)
        self.consts = self.p(np.concatenate(consts))
        self.states = []

    def state(self, shape):
        s = iron.state(shape)
        self.states.append(s)
        return s

    def rows(self, b: int):
        """The stream rows of branch b's frames."""
        r = b * self.seg + self.P
        return slice(r, r + self.L)

    def queries(self, b: int):
        """The Q stream rows of branch b's queries and where its frames start in them: from its
        first frame (the first branch), to its last (the second)."""
        r = b * self.seg + self.P
        s = r if b == 0 else r + self.L - self.Q
        return slice(s, s + self.Q), r - s

    def body(self, x, text, style, mask, tb, qang, kang, *, nt: Scratchpad[np.int32], ns: Scratchpad[np.int32]):
        S, L, T, P = self.S, self.L, self.T, self.P
        hs = self.state((M + S + M, HID))  # the stream, M zero rows before and after
        att = self.state((S, HID))  # an attention's output rows (the others stay zero)
        satt = self.state((S, 256))
        cs = self.state((self.nconst * SP, 256))
        cs[0 : self.nconst * SP] = self.consts
        const = lambda j: cs[j * SP : (j + 1) * SP].reshape(SP, 2, 128)[:ns]  # noqa: E731
        h = hs[M : M + S]
        ElementwiseMul.step(lin(x, self.w_in), mask, h)
        ti = 0
        for kind, b in self.blocks:
            if kind == "convnext":
                hd = b.d * (b.K - 1) // 2
                # the first frame of each branch into its pad rows (the graph's edge padding)
                Copy.step(hs, hs, src=tap((M + S + M, HID), (M + P) * HID, [2, hd, HID], [self.seg * HID, 0, 1]),
                          dst=tap((M + S + M, HID), (M + P - hd) * HID, [2, hd, HID], [self.seg * HID, HID, 1]))
                y = dwconv(hs[M - hd : M + S + hd], b.dw, S, b.K, b.d)
                ElementwiseMul.step(h + b.mlp(y), mask, h)
            elif kind == "time":
                ElementwiseMul.step(h + tb[ti], mask, h)
                ti += 1
            elif kind == "text":
                Q = self.Q
                qr, q0 = self.queries(0)
                q = lin(h[qr], b["wq"], b["bq"])
                k = lin(text, b["wk"], b["bk"])
                v = lin(text, b["wv"], b["bv"])
                q = F.rope(q.reshape(Q * 8, 64), qang).reshape(Q, 8, 64)
                k = F.rope(k.reshape(T * 8, 64), kang).reshape(T, 8, 64)
                o = MHA.step(q, k[:nt], v.reshape(T, 8, 64)[:nt], heads_interleaved=True, kv_interleaved=True,
                             causal=False, d=64, scale=1 / 16.0)
                att[self.rows(0)] = o.reshape(Q, HID)[q0 : q0 + L]
                att[self.rows(1)] = b["vu"][0:L]
                ElementwiseMul.step(F.layer_norm(lin(att, b["wo"], b["bo"]) + h, (HID,), b["g"], b["b"], eps=1e-6), mask, h)
            else:
                q = lin(h, b["wq"], b["bq"])
                vc = lin(style, b["wv"], b["bv"])
                c = b["sc"]
                vc = vc.reshape(SP, 2, 128)[:ns]
                for br, (k, v) in enumerate(((const(c), vc), (const(c + 1), const(c + 2)))):
                    Q = self.Q
                    qr, q0 = self.queries(br)
                    qb = q[qr].reshape(Q, 2, 128)
                    o = MHA.step(qb, k, v, heads_interleaved=True,
                                 kv_interleaved=True, causal=False, d=128, scale=1 / 16.0)
                    satt[self.rows(br)] = o.reshape(Q, 256)[q0 : q0 + L]
                ElementwiseMul.step(F.layer_norm(lin(satt, b["wo"], b["bo"]) + h, (HID,), b["g"], b["b"], eps=1e-6), mask, h)
            command_boundary()
        return lin(h, self.w_out)


class Vocoder(iron.Graph, Params):
    """The vocoder over L frames of the latent (6 L of its own; module doc)."""

    def __init__(self, V: Onnx, L: int):
        Params.__init__(self)
        self.L, self.R = L, 6 * L
        w, b = V.w("/decoder/embed/net/Conv"), V.w("/decoder/embed/net/Conv", 2)  # (512, 24, 7)
        sd, mu = V["tts.ae.latent_std"].reshape(24), V["tts.ae.latent_mean"].reshape(24)
        we = np.zeros((HID, 192))
        we[:, :168] = (w * sd[None, :, None]).transpose(0, 2, 1).reshape(HID, 168)  # im2col's (tap, channel) order
        self.we, self.be = self.p(we), self.p(b + np.einsum("oik,i->o", w, mu))
        self.blocks = [ConvNext(self, V, VP + f"convnext.{i}") for i in range(10)]
        n = VP + "final_norm.norm."
        a = V[n + "weight"] / np.sqrt(V[n + "running_var"] + 1e-5)
        c = V[n + "bias"] - V[n + "running_mean"] * a
        w1, b1 = V[VP + "head.layer1.net.weight"], V[VP + "head.layer1.net.bias"]  # (2048, 512, 3)
        self.w1 = self.p((w1 * a[None, :, None]).transpose(0, 2, 1).reshape(FF, 3 * HID))
        self.b1 = self.p(b1 + np.einsum("oik,i->o", w1, c))
        self.w2 = self.p(V[VP + "head.layer2.weight"][:, :, 0])
        self.states = []

    def state(self, shape):
        s = iron.state(shape)
        self.states.append(s)
        return s

    def body(self, x):
        R = self.R
        cols = self.state((R, 192))  # the embedding's im2col (its last 24 channels zero)
        Copy.step(x, cols, src=tap((R + 8, 24), 0, [R, 7, 24], [24, 24, 1]), dst=tap((R, 192), 0, [R, 7, 24], [192, 24, 1]))
        # the stream in two states, VM rows before each: a block reads one and writes the other (an
        # elementwise add into its own input hung the array's next call)
        pair = [self.state((VM + R, HID)) for _ in range(2)]
        hs = pair[0]
        # (a RowwiseAdd written into the state hung the array's next call: copied)
        hs[VM : VM + R] = lin(cols, self.we, self.be)
        command_boundary()
        for i, b in enumerate(self.blocks):
            hs, nxt = pair[i % 2], pair[(i + 1) % 2]
            halo = b.d * (b.K - 1)
            Copy.step(hs, hs, src=tap((VM + R, HID), VM * HID, [halo, HID], [0, 1]),
                      dst=tap((VM + R, HID), (VM - halo) * HID, [halo, HID], [HID, 1]))
            y = dwconv(hs[VM - halo : VM + R], b.dw, R, b.K, b.d)
            ElementwiseAdd.step(hs[VM : VM + R], b.mlp(y), nxt[VM : VM + R])
            command_boundary()
        hs = pair[len(self.blocks) % 2]
        # the head's first convolution (3 causal taps, edge padded), ReLU, the second
        Copy.step(hs, hs, src=tap((VM + R, HID), VM * HID, [2, HID], [0, 1]),
                  dst=tap((VM + R, HID), (VM - 2) * HID, [2, HID], [HID, 1]))
        im = self.state((R, 3 * HID))
        Copy.step(hs, im, src=tap((VM + R, HID), (VM - 2) * HID, [R, 3, HID], [HID, HID, 1]))
        u = ReLU.step(lin(im, self.w1, self.b1))
        return lin(u, self.w2)


# --- the text's graphs: the text encoder and the duration predictor's sentence encoder ------------

WREL = 4  # VITS's relative attention's window: positions -4 .. 4 (emb_rel_k, emb_rel_v of 9 rows)
TP, DPP = "tts.ttl.text_encoder.", "tts.dp.sentence_encoder."
SPT = 256  # the text encoder's style keys and values, padded (IRON's MHA: keys at least the queries)


def rel_attention(qkv, n: int, H: int, D: int, ek, ev) -> np.ndarray:
    """VITS's windowed relative attention on the host (internal/supertonic's, in float64 here):
    qkv (rows, 3 H D) the projections (q divided by sqrt(D)) of n valid rows, ek, ev (9, D). The
    output (rows, H D), zero past n."""
    x = np.asarray(qkv, np.float64)[:n]
    q, k, v = (x[:, j * H * D : (j + 1) * H * D].reshape(n, H, D) for j in range(3))
    s = np.einsum("ihd,jhd->hij", q, k)
    rel = np.einsum("ihd,rd->hir", q, ek)
    for r in range(2 * WREL + 1):
        d = r - WREL
        i = np.arange(max(0, -d), min(n, n - d))
        s[:, i, i + d] += rel[:, i, r]
    pr = _softmax(s)
    o = np.einsum("hij,jhd->ihd", pr, v)
    for r in range(2 * WREL + 1):
        d = r - WREL
        i = np.arange(max(0, -d), min(n, n - d))
        o[i] += np.einsum("hi,d->ihd", pr[:, i, i + d], ev[r])
    out = np.zeros((np.shape(qkv)[0], H * D))
    out[:n] = o.reshape(n, H * D)
    return out


class VitsLayer:
    """A layer of VITS's attention encoder (the attention's projections, its output's, the layer
    norms and the ReLU FFN) on the device; the windowed relative attention itself is a host step
    between two commands, reading the projections from the state qkv and writing the state att
    (internal/supertonic's relAttention). The rows past the text are not masked: the host's
    attention reads the valid ones alone and nothing mixes the rows elsewhere."""

    def __init__(self, ps: Params, G: Onnx, A: str, l: int, H: int):
        p = A + f"attn_layers.{l}."
        cw = lambda n: G[p + n + ".weight"][:, :, 0]  # noqa: E731
        C = cw("conv_q").shape[0]
        self.C, self.H, self.D = C, H, C // H
        sq = 1 / math.sqrt(self.D)
        self.wqkv = ps.p(np.concatenate([cw("conv_q") * sq, cw("conv_k"), cw("conv_v")]))
        self.bqkv = ps.p(np.concatenate([G[p + "conv_q.bias"] * sq, G[p + "conv_k.bias"], G[p + "conv_v.bias"]]))
        self.wo, self.bo = ps.p(cw("conv_o")), ps.p(G[p + "conv_o.bias"])
        self.ek, self.ev = G[p + "emb_rel_k"][0], G[p + "emb_rel_v"][0]
        f = A + f"ffn_layers.{l}."
        self.f1, self.fb1 = ps.p(G[f + "conv_1.weight"][:, :, 0]), ps.p(G[f + "conv_1.bias"])
        self.f2, self.fb2 = ps.p(G[f + "conv_2.weight"][:, :, 0]), ps.p(G[f + "conv_2.bias"])
        n1, n2 = A + f"norm_layers_1.{l}.norm.", A + f"norm_layers_2.{l}.norm."
        self.g1, self.b1 = ps.p(G[n1 + "weight"]), ps.p(G[n1 + "bias"])
        self.g2, self.b2 = ps.p(G[n2 + "weight"]), ps.p(G[n2 + "bias"])

    def __call__(self, a, qkv, att, mrows):
        """a (rows, C) -> the layer's output; qkv, att the states, mrows the mask's rows of a."""
        R, C, H, D = a.shape[0], self.C, self.H, self.D
        qkv[0:R] = lin(a, self.wqkv, self.bqkv)
        ek, ev = self.ek, self.ev

        def host(x, m, out):
            n = int(np.count_nonzero(np.asarray(m, np.float32)[:, 0]))
            out[...] = bf(rel_attention(x, n, H, D, ek, ev))

        iron.on_host(host, qkv[0:R], mrows, writes=(att[0:R],))
        a = F.layer_norm(a + lin(att[0:R], self.wo, self.bo), (C,), self.g1, self.b1, eps=1e-6)
        f = lin(ReLU.step(lin(a, self.f1, self.fb1)), self.f2, self.fb2)
        return F.layer_norm(a + f, (C,), self.g2, self.b2, eps=1e-6)


class TextGraph(iron.Graph, Params):
    """The ConvNeXt blocks over a stream of the text's rows (rows P .. P + R of S, zero rows around,
    the first row copied before it: the graph's edge padding, masked after each block) and the
    attention encoder's layers (VitsLayer) over its R rows. The text encoder (kind "encoder") adds
    the ConvNeXt blocks' output and the speech-prompted encoder (two attentions to the style, its
    keys constants, and a layer norm): its output (R, 256) is the text's embedding. The duration
    predictor's sentence encoder ("duration") reads the sentence token and the text's rows: its
    output's first row is the token's hidden, before proj_out (the host's, with the predictor)."""

    def __init__(self, G: Onnx, kind: str, T: int):
        Params.__init__(self)
        self.kind, self.T = kind, T
        if kind == "encoder":
            pre, nconv, layers, H = TP, 6, 4, 4
            self.R, self.P = T, 32
            self.S = T + 2 * self.P  # ConvNeXt's rows by 64 (256 channels: DWConv1d's 4 parts of 16)
        else:
            pre, nconv, layers, H = DPP, 6, 2, 2
            self.R, self.P = T + 16, 16  # the token and T characters, by 16
            self.S = -(-(self.P + self.R + 16) // 128) * 128  # (64 channels: DWConv1d's 8 parts)
        self.blocks = [ConvNext(self, G, pre + f"convnext.convnext.{j}") for j in range(nconv)]
        self.C = self.blocks[0].C
        self.layers = [VitsLayer(self, G, pre + "attn_encoder.", l, H) for l in range(layers)]
        if kind == "encoder":
            sp = "tts.ttl.speech_prompted_text_encoder."
            keys = G["tts.ttl.style_encoder.style_token_layer.style_key"].reshape(STYLE, 256)
            self.att = []
            consts = []
            for i in (1, 2):
                a, q_ = f"/speech_prompted_text_encoder/attention{i}/", sp + f"attention{i}."
                k = np.tanh(keys @ G.w(a + "W_key/linear/MatMul") + G[q_ + "W_key.linear.bias"])
                consts.append(np.pad(k, ((0, SPT - STYLE), (0, 0))))
                self.att.append(dict(
                    wq=self.p(G.w(a + "W_query/linear/MatMul").T), bq=self.p(G[q_ + "W_query.linear.bias"]),
                    wv=self.p(G.w(a + "W_value/linear/MatMul").T), bv=self.p(G[q_ + "W_value.linear.bias"]),
                    wo=self.p(G.w(a + "out_fc/linear/MatMul").T), bo=self.p(G[q_ + "out_fc.linear.bias"])))
            self.keys = self.p(np.concatenate(consts))  # copied into a state a call (per-call keys)
            self.ng, self.nb = self.p(G[sp + "norm.norm.weight"]), self.p(G[sp + "norm.norm.bias"])

    def state(self, shape, name=None):
        return iron.state(shape, name=name)

    def convnexts(self, x, mask):
        """The ConvNeXt blocks over x (S, C): the stream state's rows."""
        S, P, C = self.S, self.P, self.C
        hs = self.state((M + S + M, C))
        h = hs[M : M + S]
        ElementwiseMul.step(x, mask, h)
        for b in self.blocks:
            hd = b.d * (b.K - 1) // 2
            Copy.step(hs, hs, src=tap((M + S + M, C), (M + P) * C, [hd, C], [0, 1]),
                      dst=tap((M + S + M, C), (M + P - hd) * C, [hd, C], [C, 1]))
            y = dwconv(hs[M - hd : M + S + hd], b.dw, S, b.K, b.d)
            ElementwiseMul.step(h + b.mlp(y), mask, h)
        return hs[M + P : M + P + self.R]

    def vits(self, c, mask):
        R, C, P = self.R, self.C, self.P
        qkv = self.state((R, 3 * C), name="qkv")
        att = self.state((R, C), name="att")
        a = c
        for layer in self.layers:  # each ends a command at its host step
            a = layer(a, qkv, att, mask[P : P + R])
        return a

    def encoder(self, x, mask, style, ns):
        R = self.R
        ks = self.state((2 * SPT, 256))
        ks[0 : 2 * SPT] = self.keys
        c = self.convnexts(x, mask)
        x2 = self.vits(c, mask) + c
        s = x2
        for i, a in enumerate(self.att):
            q = lin(s, a["wq"], a["bq"]).reshape(R, 2, 128)
            v = lin(style, a["wv"], a["bv"]).reshape(SPT, 2, 128)[:ns]
            k = ks[i * SPT : (i + 1) * SPT].reshape(SPT, 2, 128)[:ns]
            o = MHA.step(q, k, v, heads_interleaved=True, kv_interleaved=True, causal=False, d=128, scale=1 / 16.0)
            s = lin(o.reshape(R, 256), a["wo"], a["bo"]) + x2
        return F.layer_norm(s, (256,), self.ng, self.nb, eps=1e-6)

    def sentence(self, x, mask):
        c = self.convnexts(x, mask)
        return self.vits(c, mask) + c  # its row 0 the sentence token's


class TextEncoderGraph(TextGraph):
    def __init__(self, G: Onnx, T: int):
        super().__init__(G, "encoder", T)

    def body(self, x, mask, style, *, ns: Scratchpad[np.int32]):
        return self.encoder(x, mask, style, ns)


class SentenceGraph(TextGraph):
    def __init__(self, G: Onnx, T: int):
        super().__init__(G, "duration", T)

    def body(self, x, mask):
        return self.sentence(x, mask)


def text_graph(G: Onnx, kind: str, T: int) -> TextGraph:
    return TextEncoderGraph(G, T) if kind == "encoder" else SentenceGraph(G, T)


def text_inputs(G: Onnx, g: TextGraph, rows, n: int, style=None):
    """A TextGraph's inputs: rows (n, C) the embedding table's rows of the text's ids (the duration
    predictor's after its sentence token), style (50, 256) the text encoder's."""
    x = np.zeros((g.S, g.C))
    mask = np.zeros((g.S, g.C))
    x[g.P : g.P + n] = rows
    mask[g.P : g.P + n] = 1
    ins = dict(x=bf(x), mask=bf(mask))
    vals = {}
    if g.kind == "encoder":
        st = np.zeros((SPT, 256))
        st[:STYLE] = style
        ins["style"] = bf(st)
        vals["ns"] = STYLE
    return ins, vals


def text_shapes(g: TextGraph) -> dict:
    sh = dict(x=(g.S, g.C), mask=(g.S, g.C))
    if g.kind == "encoder":
        sh["style"] = (SPT, 256)
    return sh


# --- export --------------------------------------------------------------------------------------


def estimator_inputs(E: Onnx, g, x, text, style, n: int, t: int, step: int, total: int):
    """The estimator graph g's inputs for a latent x (144, n), a text (256, t), style (50, 256)."""
    T, L, P, seg, S = g.T, g.L, g.P, g.seg, g.S
    xs = np.zeros((S, XP))
    mask = np.zeros((S, HID))
    for b in range(2):
        xs[b * seg + P : b * seg + P + n, :LATENT] = x.T
        mask[b * seg + P : b * seg + P + n] = 1
    tx = np.zeros((T, TEXT))
    tx[:t] = text.T
    st = np.zeros((SP, 256))
    st[:STYLE] = style.reshape(STYLE, 256)
    tb = time_rows(E, time_embed(E, step, total))

    def table(a):
        out = np.empty((a.shape[0], 64))
        out[:, 0::2], out[:, 1::2] = np.cos(a), np.sin(a)
        return out

    qa, ka = table(rope_angles(E, n, max(L, 64))), table(rope_angles(E, t, T))
    return dict(x=bf(xs), text=bf(tx), style=bf(st), mask=bf(mask), tb=bf(tb), qang=bf(qa), kang=bf(ka)), dict(nt=t, ns=STYLE)


def export(onnx_dir: Path, out: Path, sizes, check: bool, only=("estimator", "vocoder", "encoder", "duration")) -> None:
    E = Onnx(onnx_dir / "vector_estimator.onnx")
    V = Onnx(onnx_dir / "vocoder.onnx")
    TE = Onnx(onnx_dir / "text_encoder.onnx")
    DE = Onnx(onnx_dir / "duration_predictor.onnx")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    longest = max(L for _, L in sizes)
    for T, L in sizes:
        if "estimator" in only:
            t0 = time.time()
            g = Estimator(E, T, L, longest)
            S = g.S
            cg = g.compile(x=(S, XP), text=(T, TEXT), style=(SP, 256), mask=(S, HID), tb=(len(TIME_BLOCKS), HID),
                           qang=(g.Q, 64), kang=(T, 64))
            name = f"estimator-t{T}-l{L}"
            meta = dict(T=T, L=L, P=g.P, segment=g.seg, rows=S, latent_cols=XP, style_rows=SP, Q=g.Q)
            m = export_graph(cg, out / name, name, meta)
            print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan'].splitlines()[0]}", flush=True)
            if check:
                n, t = L - 5, T - 7
                x = rng.standard_normal((LATENT, n))
                text = rng.standard_normal((TEXT, t)) * 0.5
                style = rng.standard_normal((STYLE, 256)) * 0.3
                ins, vals = estimator_inputs(E, g, x, text, style, n, t, 3, 8)
                write_check(g, m, out / name, ins, vals)
        for kind, G in (("encoder", TE), ("duration", DE)):
            if kind not in only:
                continue
            t0 = time.time()
            g = text_graph(G, kind, T)
            cg = g.compile(**text_shapes(g))
            name = f"{kind}-t{T}"
            meta = dict(T=T, rows=g.S, P=g.P, R=g.R, channels=g.C, heads=g.layers[0].H, layers=len(g.layers),
                        style_rows=SPT if kind == "encoder" else 0)
            m = export_graph(cg, out / name, name, meta)
            print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan'].splitlines()[0]}", flush=True)
            if check:
                n = T - 7 if kind == "encoder" else T - 6
                rows = rng.standard_normal((n, g.C)) * 0.5
                ins, vals = text_inputs(G, g, rows, n, rng.standard_normal((STYLE, 256)) * 0.3)
                write_check(g, m, out / name, ins, vals)
        if "vocoder" in only:
            t0 = time.time()
            g = Vocoder(V, L)
            cg = g.compile(x=(6 * L + 8, 24))
            name = f"vocoder-l{L}"
            m = export_graph(cg, out / name, name, dict(L=L, frames=6 * L))
            print(f"{name}: compiled in {time.time() - t0:.0f} s; {m['graph']['plan'].splitlines()[0]}", flush=True)
            if check:
                z = rng.standard_normal((LATENT, L)) * 0.3
                xin = np.zeros((6 * L + 8, 24))
                xin[: 6 * L + 6] = vocoder_frames(z)
                write_check(g, m, out / name, dict(x=bf(xin)))
    if not check:
        for kind in only:
            names = [{"estimator": f"estimator-t{T}-l{L}", "vocoder": f"vocoder-l{L}"}.get(kind, f"{kind}-t{T}")
                     for T, L in sizes]
            share_banks(out, names)
    write_host(E, out, TE, DE)
    (out / "model.json").write_text(json.dumps(dict(
        kind="supertonic3", sizes=[dict(T=T, L=L) for T, L in sizes], P=P, latent_cols=XP, style_rows=SP,
        time_blocks=len(TIME_BLOCKS)), indent=1) + "\n")


def share_banks(out: Path, names: list[str]) -> None:
    """Keep one image of the banks the sizes' graphs share: each size's weights are the same
    params in the same order (Estimator's vu of the longest), so a bank's image (its weights;
    the states and scratch after them are zeros, trimmed) is the same bytes in each. The other
    graphs' manifests name the last's file (../NAME/...): a download holds the weights once
    (iron_plamo_embed.py's)."""
    import hashlib

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        return h.hexdigest()

    owner = out / names[-1]
    om = json.loads((owner / "manifest.json").read_text())
    for name in names[:-1]:
        d = out / name
        m = json.loads((d / "manifest.json").read_text())
        for kind, bank in m["banks"].items():
            mine, theirs = bank.get("image"), om["banks"].get(kind, {}).get("image")
            if not mine or not theirs or digest(d / mine) != digest(owner / theirs):
                print(f"{name}: bank {kind} is its own", flush=True)
                continue
            (d / mine).unlink()
            bank["image"] = f"../{owner.name}/{theirs}"
            print(f"{name}: bank {kind} is {owner.name}'s", flush=True)
        (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")


def write_host(E: Onnx, out: Path, TE: Onnx, DE: Onnx) -> None:
    """host.safetensors (float32): the time embedding's MLP and the time blocks' Linears, RoPE's
    frequencies; the relative attention's embeddings of the text's graphs' layers; the duration
    predictor's sentence token, proj_out and predictor."""
    t = {
        "time.freqs": E[EN + "time_encoder/sinusoidal/Constant_3_output_0"].reshape(-1) if EN + "time_encoder/sinusoidal/Constant_3_output_0" in E.init else E.consts[EN + "time_encoder/sinusoidal/Constant_3_output_0"].reshape(-1),
        "time.w0": E[EP + "time_encoder.mlp.0.linear.weight"], "time.b0": E[EP + "time_encoder.mlp.0.linear.bias"],
        "time.w2": E[EP + "time_encoder.mlp.2.linear.weight"], "time.b2": E[EP + "time_encoder.mlp.2.linear.bias"],
        "time.rows.w": np.stack([E.w(EN + f"main_blocks.{i}/linear/linear/MatMul").T for i in TIME_BLOCKS]),
        "time.rows.b": np.stack([E[EP + f"main_blocks.{i}.linear.linear.bias"] for i in TIME_BLOCKS]),
        "rope.theta": E[EP + "main_blocks.3.attn.theta"].reshape(-1),
        "dp.token": DE[DPP + "sentence_token"].reshape(-1),
        "dp.proj": DE[DPP + "proj_out.net.weight"][:, :, 0],
        "dp.l0.w": DE["tts.dp.predictor.layers.0.weight"], "dp.l0.b": DE["tts.dp.predictor.layers.0.bias"],
        "dp.prelu": DE["tts.dp.predictor.activation.weight"].reshape(-1),
        "dp.l1.w": DE["tts.dp.predictor.layers.1.weight"], "dp.l1.b": DE["tts.dp.predictor.layers.1.bias"],
    }
    for kind, G, pre, layers in (("encoder", TE, TP, 4), ("duration", DE, DPP, 2)):
        for l in range(layers):
            for e in ("k", "v"):
                t[f"{kind}.rel_{e}.{l}"] = G[pre + f"attn_encoder.attn_layers.{l}.emb_rel_{e}"][0]
    hdr, blobs, off = {}, [], 0
    for k, v in t.items():
        b = np.ascontiguousarray(np.asarray(v, np.float32)).tobytes()
        hdr[k] = dict(dtype="F32", shape=list(np.shape(v)), data_offsets=[off, off + len(b)])
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(out / "host.safetensors", "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        for b in blobs:
            f.write(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path, help="the model's onnx/ directory (Supertone/supertonic-3)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--sizes", nargs="*", default=[f"{t}x{l}" for t, l in SIZES], help="TxL, as internal/supertonic.Sizes")
    ap.add_argument("--only", nargs="*", default=["estimator", "vocoder", "encoder", "duration"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    a = ap.parse_args(argv)
    bind_device(a.device, a.cols)
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.sizes]
    export(a.onnx, a.out, sizes, a.check, a.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
