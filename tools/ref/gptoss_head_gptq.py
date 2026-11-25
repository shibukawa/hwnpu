"""GPTQ for gpt-oss's head on Apple's Neural Engine: lookup tables (lutN) chosen against the
final norm's outputs, as codes that bonsaiai-dev convert -codes and bonsaiai-chat --codes read.

The Core AI conversion keeps every other part as it was (the experts the checkpoint's MXFP4 on the
CPU, the attention lut8); the head (lm_head, 201088 x 2880, 0.55 GiB in lut8) is read whole every
step, a sixth of it. A smaller table (lut6, lut4) rounds each weight to its nearest level, which
keeps the weight error small but not the logits'. GPTQ (Frantar et al. 2022, tools/amd/q4_gptq.py's
lutN form) chooses each weight's entry of the same fixed table (palette.Uniform: 2^bits levels from
the smallest to the largest weight of 16 rows) in order of the in-features and moves each one's
error onto the in-features not yet chosen, weighted by H = E[x x^T] of the final norm's outputs.
Nothing changes at run time.

    PY=~/models/ref-venv/bin/python
    $PY -I tools/ref/gptoss_head_gptq.py hidden MODEL_DIR OUT [--seqs 64 --length 512]
    $PY -I tools/ref/gptoss_head_gptq.py fit MODEL_DIR OUT [--fmt lut6 lut4]

``hidden`` runs the float32 reference (ref_gptoss.GptOss, a layer at a time) over calibration
text (half wikitext-2's train split, half wiki40b-ja's, ~/models/calib, after the harmony user
prefix) and over new-model-checklist's evaluation texts (ref_gptoss.texts: the README, GPL-3,
wiki40b-ja, wikitext-2's test split, Go code; 512 tokens each), and writes the final norm's outputs:
OUT/calib.npy (rows, 2880) and OUT/eval.npz (one array a text). ``fit`` measures the head alone on
the evaluation rows (KL of the logits from the checkpoint's bf16 head, past each text's first 64
positions) for nearest rounding and for GPTQ, and writes OUT/codes-FMT (index.json, KEY.idx and
KEY.tab: internal/palette.Codes) of the GPTQ head. H is damped by 0.1 of its mean diagonal, as
tools/amd/gptoss_gptq.py's.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ref_gptoss as R  # noqa: E402

GROUP = 16  # rows a table (llm.Options.Group's default)
BITS = {"lut4": 4, "lut6": 6, "lut8": 8}
SKIP = 64  # positions of each evaluation text not scored


def calib_ids(tok, seqs: int, length: int):
    out = []
    prefix = tok.encode(R.USER, add_special_tokens=False).ids
    for name in ("wikitext2-train.txt", "wiki40b-ja-calib.txt"):
        text = (Path.home() / "models/calib" / name).read_text(encoding="utf-8")
        ids = tok.encode(text[:4_000_000], add_special_tokens=False).ids
        step = len(ids) // (seqs // 2)
        for k in range(seqs // 2):
            out.append(prefix + ids[k * step : k * step + length - len(prefix)])
    return out, len(prefix)


def cmd_hidden(a):
    torch.set_grad_enabled(False)
    ck = R.Checkpoint(a.model_dir)
    g = R.GptOss(ck)
    tok = R.tokenizer(a.model_dir)
    cal, skip = calib_ids(tok, a.seqs, a.length)
    ev = {n: tok.encode(R.USER + t, add_special_tokens=False).ids[: a.length] for n, t in R.texts(a.length).items()}
    seqs = cal + list(ev.values())
    hs = [g.embed(s) for s in seqs]
    t0 = time.time()
    for i in range(g.layers):
        w = ck.layer(i)
        hs = [g.layer(i, h, w)[0] for h in hs]
        del w
        print(f"layer {i}: {time.time() - t0:.0f} s", flush=True)
    nw = ck.get("model.norm.weight")
    xs = [g.norm(h, nw).numpy() for h in hs]
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "calib.npy", np.concatenate([x[skip:] for x in xs[: len(cal)]]))
    np.savez(a.out / "eval.npz", **{n: x for n, x in zip(ev, xs[len(cal) :])})
    print(f"{len(cal)} calibration sequences, {len(ev)} evaluation texts in {a.out}")


def tables(w, bits):
    """palette.Uniform's tables of w (N, K) torch: (N / GROUP, 2^bits) of fp16 values."""
    k = 1 << bits
    g = w.reshape(-1, GROUP * w.shape[1])
    lo, hi = g.min(1).values, g.max(1).values
    t = lo[:, None] + (hi - lo)[:, None] * (torch.arange(k, dtype=torch.float32) / (k - 1))[None, :]
    return t.half().float()


def nearest(x, T):
    """For x (N,), the index of the nearest level of each row's sorted table T (N, k)."""
    mid = ((T[:, 1:] + T[:, :-1]) / 2).contiguous()
    return torch.searchsorted(mid, x[:, None].contiguous())[:, 0]


def rtn(W, bits):
    T = tables(W, bits)
    Tr = T.repeat_interleave(GROUP, 0)
    idx = torch.empty(W.shape, dtype=torch.uint8)
    for c in range(W.shape[1]):
        idx[:, c] = nearest(W[:, c], Tr)
    return idx, T


def gptq(W, H, bits, damp=0.1, block=128):
    """The indices (N, K) uint8 into the fixed tables (N / GROUP, 2^bits), chosen by GPTQ."""
    W = W.clone()
    N, K = W.shape
    H = torch.as_tensor(H, dtype=torch.float64).clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    T = tables(W, bits)
    Tr = T.repeat_interleave(GROUP, 0)
    idx = torch.empty((N, K), dtype=torch.uint8)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            x = W1[:, i]
            c = nearest(x, Tr)
            q = torch.gather(Tr, 1, c[:, None])[:, 0]
            idx[:, i1 + i] = c.to(torch.uint8)
            err = (x - q) / Hi[i, i]
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
        print(f"  columns {i2}/{K}", end="\r", flush=True)
    print()
    return idx, T


def dequant(idx, T):
    Tr = T.repeat_interleave(GROUP, 0)
    return torch.gather(Tr, 1, idx.long())


def kl(X, W, Q, chunk=128):
    """Mean KL(softmax(X W^T) || softmax(X Q^T)) and the agreement of the argmax, over X's rows."""
    tot, top, n = 0.0, 0, 0
    for r in range(0, X.shape[0], chunk):
        x = X[r : r + chunk]
        a = torch.log_softmax((x @ W.T).double(), -1)
        b = torch.log_softmax((x @ Q.T).double(), -1)
        tot += float((a.exp() * (a - b)).sum())
        top += int((a.argmax(-1) == b.argmax(-1)).sum())
        n += x.shape[0]
    return tot / n, top / n


def key(raw_bf16):
    """palette.WeightKey of a bf16 weight (rows, cols): SHA-1 of "(rows, cols)" and its bf16 bytes."""
    b = raw_bf16.contiguous().view(torch.int16).numpy()
    return hashlib.sha1(repr(b.shape).encode() + b.tobytes()).hexdigest()


def write(d: Path, k, idx, T, shape, fmt):
    d.mkdir(parents=True, exist_ok=True)
    idx.numpy().tofile(d / f"{k}.idx")
    T.numpy().astype("<f4").tofile(d / f"{k}.tab")
    (d / "index.json").write_text(json.dumps({k: dict(name="lm_head.weight", fmt=fmt, shape=list(shape), group=GROUP)}, indent=1) + "\n")


def cmd_fit(a):
    torch.set_grad_enabled(False)
    ck = R.Checkpoint(a.model_dir)
    raw = ck.raw("lm_head.weight")
    k = key(raw)
    W = raw.float()
    del raw
    X = torch.from_numpy(np.load(a.out / "calib.npy")).double()
    H = (X.T @ X / X.shape[0]).numpy()
    del X
    ev = np.load(a.out / "eval.npz")
    Xe = {n: torch.from_numpy(ev[n][min(SKIP, len(ev[n]) // 2) :]) for n in ev.files}
    result = {}

    def score(name, Q):
        rows = {n: kl(x, W, Q) for n, x in Xe.items()}
        result[name] = rows
        print(name, " ".join(f"{n} {v[0]:.4f}/{v[1]:.3f}" for n, v in rows.items()),
              f"mean {np.mean([v[0] for v in rows.values()]):.4f}", flush=True)

    for fmt in a.fmt:
        bits = BITS[fmt]
        t0 = time.time()
        idx, T = rtn(W, bits)
        score(f"{fmt} rtn", dequant(idx, T))
        idx, T = gptq(W, H, bits, damp=a.damp)
        print(f"{fmt} gptq: {time.time() - t0:.0f} s")
        score(f"{fmt} gptq", dequant(idx, T))
        write(a.out / f"codes-{fmt}", k, idx, T, W.shape, fmt)
        del idx
    (a.out / "fit.json").write_text(json.dumps(result, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hidden")
    h.add_argument("model_dir")
    h.add_argument("out", type=Path)
    h.add_argument("--seqs", type=int, default=64)
    h.add_argument("--length", type=int, default=512)
    f = sub.add_parser("fit")
    f.add_argument("model_dir")
    f.add_argument("out", type=Path)
    f.add_argument("--fmt", nargs="+", default=["lut6", "lut4"], choices=sorted(BITS))
    f.add_argument("--damp", type=float, default=0.1)
    a = ap.parse_args()
    {"hidden": cmd_hidden, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
