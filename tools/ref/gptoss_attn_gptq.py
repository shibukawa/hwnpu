"""GPTQ for gpt-oss's attention on Apple's Neural Engine: codes of q/k/v/o_proj chosen against each
layer's inputs, which bonsaiai-dev convert -codes and bonsaiai-chat --codes read.

The attention's projections in lut8 (palette.Uniform, a table of 256 levels for 16 rows) raise the
KL on plain text from 0.036 to 0.29 on the Neural Engine (docs/design/performance.md); fp16 keeps it but
holds 1.3 GiB more, which a 16 GB Mac does not have with the experts. GPTQ (gptoss_head_gptq.py's)
chooses each weight's rounding against H = E[x x^T] of the projection's inputs: the first norm's
outputs for q, k and v, the attention's output for o. The layers go in order, each fed by the
layers before it with their roundings (sequential GPTQ), so a layer's codes answer the rounding of
the ones before it. The roundings (--modes):

    lut     the fixed tables of lutN (--bits), as the conversion's: codes-lutN
    lutrow  the same of the rows divided by their scales, lutNr (blocks.LinearBias): codes-lutNr
    int8    a scale a row (the largest magnitude over 127), integers -127..127: codes-int8

    PY=~/models/ref-venv/bin/python
    $PY -I tools/ref/gptoss_attn_gptq.py MODEL_DIR OUT [--modes lutrow] [--seqs 32 --damp 0.1]

It prints, on new-model-checklist's evaluation texts (127 tokens each, past the first 16
positions), the KL from the float32 reference with the projections rounded to the nearest level
(rtn) and with GPTQ's, the experts and everything else float32; OUT/fit-MODES.json keeps them.
Measured (32 sequences of 512 tokens, damp 0.01): lut8 0.31 -> 0.17, int8 0.099 -> 0.043. lutrow
(about 10 minutes on an M3): 0.070 -> 0.083 at damp 0.01, the calibration's kinds of text better
and README, the GPL and Go worse; 0.070 -> 0.059 at 0.1, README and Go still worse. On the Neural
Engine (bonsaiai-dev convert -codes OUT/codes-lut8r) the median KL of each text halves (0.016-0.030
-> 0.006-0.010) but a few positions of README and Go grow (11.5, 4.4): the mean 0.059 -> 0.069.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ref_gptoss as R  # noqa: E402
from gptoss_head_gptq import GROUP, calib_ids, key, nearest, tables  # noqa: E402

N, SKIP = 127, 16
PROJ = ("q", "k", "v", "o")


def o_input(g, i, x, w):
    """The attention's output before o_proj (n, heads * head_dim): o_proj as the identity."""
    d = w["self_attn.q_proj.weight"].shape[0]
    w2 = dict(w)
    w2["self_attn.o_proj.weight"] = torch.eye(d)
    w2["self_attn.o_proj.bias"] = torch.zeros(d)
    return g.attention(i, x, w2)[0]


def hess(xs):
    k = xs[0].shape[1]
    H = torch.zeros(k, k, dtype=torch.float64)
    n = 0
    for x in xs:
        x = x.double()
        H += x.T @ x
        n += x.shape[0]
    return H / n


class Quant:
    """A projection's rounding: lut (palette.Uniform tables of GROUP rows), lutrow (the same of the
    rows divided by their largest magnitude, which the program multiplies back after the
    projection), int8 (a scale a row, the largest magnitude over 127)."""

    def __init__(self, mode, W, bits):
        self.mode = mode
        if mode == "int8":
            self.s = (W.abs().amax(1) / 127).clamp_min(1e-30).half().float()
            return
        if mode == "lutrow":
            # blocks.RowScales: a row's largest magnitude over the matrix's, in fp16 (1 for zeros)
            top = W.abs().amax(1)
            self.s = torch.where(top > 0, (top / top.max()).half().float(), torch.ones_like(top))
        else:
            self.s = torch.ones(W.shape[0])
        self.T = tables(W / self.s[:, None], bits)
        self.Tr = self.T.repeat_interleave(GROUP, 0)

    def q(self, x):
        """x (rows,) of one column: the indices and the rounded values."""
        if self.mode == "int8":
            c = torch.round(x / self.s).clamp(-127, 127)
            return c, c * self.s
        c = nearest(x / self.s, self.Tr)
        return c, torch.gather(self.Tr, 1, c[:, None])[:, 0] * self.s


def gptq(W, H, qz, damp, block=128):
    """GPTQ of W (N, K) against H with the rounding qz: the indices (N, K) and the rounded W."""
    W = W.clone()
    N, K = W.shape
    H = torch.as_tensor(H, dtype=torch.float64).clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    idx = torch.empty((N, K), dtype=torch.int16)
    Q = torch.empty_like(W)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            x = W1[:, i]
            c, q = qz.q(x)
            idx[:, i1 + i] = c.to(torch.int16)
            Q[:, i1 + i] = q
            err = (x - q) / Hi[i, i]
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return idx, Q


def rtn(W, qz):
    return torch.stack([qz.q(W[:, c])[1] for c in range(W.shape[1])], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out", type=Path)
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--damp", type=float, default=0.1)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--modes", nargs="+", default=["lut"], choices=["lut", "lutrow", "int8"])
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    fmt = f"lut{a.bits}"
    ck = R.Checkpoint(a.model_dir)
    g = R.GptOss(ck)
    tok = R.tokenizer(a.model_dir)
    cal, skip = calib_ids(tok, a.seqs, a.length)
    tx = R.texts(512)
    ev = {n: tok.encode(R.USER + t, add_special_tokens=False).ids[:N] for n, t in tx.items()}
    hc = {m: [g.embed(s) for s in cal] for m in a.modes}
    names = ["ref"] + [f"{m} {v}" for m in a.modes for v in ("rtn", "gptq")]
    he = {v: [g.embed(s) for s in ev.values()] for v in names}
    outs = {m: a.out / f"codes-{dict(lut=fmt, lutrow=fmt + 'r').get(m, m)}" for m in a.modes}
    if not a.no_write:
        for d in outs.values():
            d.mkdir(parents=True, exist_ok=True)
    index = {m: {} for m in outs}
    t0 = time.time()
    for i in range(g.layers):
        w = ck.layer(i)
        p = f"model.layers.{i}.self_attn."
        ws = {"ref": w}
        for m in a.modes:
            wr, wq = dict(w), dict(w)
            for name in PROJ:
                W = w[f"self_attn.{name}_proj.weight"]
                wr[f"self_attn.{name}_proj.weight"] = rtn(W, Quant(m, W, a.bits))
            # q, k, v against the first norm's outputs of the calibration through the layers before
            xs = [g.norm(h, w["input_layernorm.weight"]) for h in hc[m]]
            H = hess([x[skip:] for x in xs]).numpy()
            for name in ("q", "k", "v"):
                W = w[f"self_attn.{name}_proj.weight"]
                qz = Quant(m, W, a.bits)
                idx, wq[f"self_attn.{name}_proj.weight"] = gptq(W, H, qz, a.damp)
                if m in outs and not a.no_write:
                    add(outs[m], index[m], p + f"{name}_proj.weight", ck, idx, qz, fmt)
            # o against the attention's output with q, k and v's rounding
            H = hess([o_input(g, i, x, wq)[skip:] for x in xs]).numpy()
            del xs
            W = w["self_attn.o_proj.weight"]
            qz = Quant(m, W, a.bits)
            idx, wq["self_attn.o_proj.weight"] = gptq(W, H, qz, a.damp)
            if m in outs and not a.no_write:
                add(outs[m], index[m], p + "o_proj.weight", ck, idx, qz, fmt)
            del H
            hc[m] = [g.layer(i, h, wq)[0] for h in hc[m]]
            ws[f"{m} rtn"], ws[f"{m} gptq"] = wr, wq
        for v in names:
            he[v] = [g.layer(i, h, ws[v])[0] for h in he[v]]
        del w, ws
        print(f"layer {i}: {time.time() - t0:.0f} s", flush=True)
    for m, d in outs.items():
        if index[m]:
            (d / "index.json").write_text(json.dumps(index[m], indent=1) + "\n")
    lg = {v: [g.head(h)[SKIP:] for h in hs] for v, hs in he.items()}
    result = {}
    for v in names[1:]:
        st = {n: R.kl_stats(r, q) for n, r, q in zip(ev, lg["ref"], lg[v])}
        result[v] = st
        print(v, " ".join(f"{n} {s['mean']:.4f}/{s['top1']:.3f}" for n, s in st.items()),
              f"mean {np.mean([s['mean'] for s in st.values()]):.4f}", flush=True)
    (a.out / f"fit-{'-'.join(a.modes)}.json").write_text(json.dumps(dict(damp=a.damp, seqs=a.seqs, kl=result), indent=1) + "\n")


def add(out, index, name, ck, idx, qz, fmt):
    """Writes a projection's codes: lutN's indices and tables (lutNr's with the rows' scales), or
    int8's integers and row scales."""
    raw = ck.raw(name)
    k = key(raw)
    if qz.mode == "int8":
        idx.to(torch.int8).numpy().tofile(out / f"{k}.idx")
        qz.s.numpy().astype("<f4").tofile(out / f"{k}.tab")
        index[k] = dict(name=name, fmt="int8", shape=list(raw.shape), group=1)
    else:
        idx.to(torch.uint8).numpy().tofile(out / f"{k}.idx")
        qz.T.numpy().astype("<f4").tofile(out / f"{k}.tab")
        if qz.mode == "lutrow":
            qz.s.numpy().astype("<f4").tofile(out / f"{k}.rows")
            fmt += "r"
        index[k] = dict(name=name, fmt=fmt, shape=list(raw.shape), group=GROUP)


if __name__ == "__main__":
    main()
