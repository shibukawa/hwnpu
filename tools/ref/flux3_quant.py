#!/usr/bin/env python3
"""The FLUX.3 [klein] DiT's linear weights in hwnpu's 8-bit formats, tensor by tensor from the
safetensors (a tensor in memory at a time), for tools/ref/ref_flux3klein.py --qfile.

  python tools/ref/flux3_quant.py int8 OUT.safetensors
  python tools/ref/flux3_quant.py lut8 OUT.safetensors [--fit uniform|kmeans|wkmeans] [--stats X2.pt]

int8: symmetric, one fp16 scale per row; the clipping point is searched over clips (1 .. 0.5) for the
      smallest squared error (internal/palette/linear.go QuantizeLinear with search).
lut8: one 256-entry fp16 table per 16 rows (internal/palette, ColBlock 0), fitted as
      uniform  the levels evenly spaced between the group's minimum and maximum (QuantizeBits, Uniform);
      kmeans   Lloyd's algorithm from the group's quantiles, 8 passes (QuantizeBits, KMeans): its top
               level starts at the 99.8 % point, and the DiT's few large weights are clipped;
      kmeans-minmax  Lloyd from the uniform levels, 8 passes (KMeansMinMax): never worse than uniform;
      wkmeans  weighted Lloyd from the weighted quantiles, 20 passes, each weight counted by its
               column's E[x^2], the mean square of the layer's input at that in-feature
               (QuantizeWeighted, tools/coreai/lutfit; X2.pt from ref_flux3klein.py --stats). A
               layer the statistics do not cover (the reference image's stream in text to image)
               takes kmeans;
      wkmeans-minmax  weighted Lloyd from the uniform levels, 20 passes.
Each weight takes the table's nearest fp16 level. Matrices with at least MIN_NUMEL elements are
quantized; the rest (norms, biases) are kept in bf16. Prints the relative RMS error of the weights,
and with --stats the error weighted by E[x^2] (how far the layers' outputs move).
"""
import argparse
import json
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

M = os.path.expanduser("~/models/flux-3-image-klein/transformer")
CLIPS = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
MIN_NUMEL = 1 << 20
GROUP = 16
LEVELS = 256
CHUNK = 1 << 26  # values fitted at once


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEV = pick_device()


def q_int8(w):
    w = w.float().to(DEV)
    top = w.abs().amax(dim=1, keepdim=True)
    best_s = (top / 127).half().float()
    best_e = torch.full_like(top, float("inf"))
    for c in CLIPS:
        s = (top * c / 127).half().float()
        safe = torch.where(s == 0, torch.ones_like(s), s)
        q = torch.clamp(torch.round(w / safe), -128, 127)
        q = torch.where(s == 0, torch.zeros_like(q), q)
        e = ((w - q * s) ** 2).sum(dim=1, keepdim=True)
        better = e < best_e
        best_e = torch.where(better, e, best_e)
        best_s = torch.where(better, s, best_s)
    safe = torch.where(best_s == 0, torch.ones_like(best_s), best_s)
    q = torch.clamp(torch.round(w / safe), -128, 127)
    q = torch.where(best_s == 0, torch.zeros_like(q), q).to(torch.int8)
    return {"q": q.cpu().contiguous(), "s": best_s.squeeze(1).half().cpu().contiguous()}


def nearest(v, lut):
    """Each value's nearest level of its group's sorted table: (G, N) values, (G, K) levels."""
    mids = (lut[:, 1:] + lut[:, :-1]) / 2
    return torch.searchsorted(mids.contiguous(), v.contiguous())


def lloyd(v, c, wts, iters):
    """Weighted Lloyd over each group's values v (G, N) from sorted centers c (G, K); wts (G, N) or None."""
    g, k = c.shape
    for _ in range(iters):
        ids = nearest(v, c)
        w = torch.ones_like(v) if wts is None else wts
        s = torch.zeros(g, k, device=v.device).scatter_add_(1, ids, w * v)
        n = torch.zeros(g, k, device=v.device).scatter_add_(1, ids, w)
        c = torch.where(n > 0, s / torch.where(n > 0, n, torch.ones_like(n)), c)
        c, _ = torch.sort(c, dim=1)
    return c


def fit_groups(v, fit, wts):
    """The fp32 tables (G, 256) of groups v (G, N)."""
    g, n = v.shape
    lo, hi = v.amin(dim=1, keepdim=True), v.amax(dim=1, keepdim=True)
    uniform = lo + (hi - lo) * torch.arange(LEVELS, device=v.device).view(1, -1) / (LEVELS - 1)
    if fit == "uniform":
        return uniform
    if fit == "kmeans-minmax":  # Lloyd from the uniform levels: never worse than uniform in squared error
        return lloyd(v, uniform, None, 8)
    if fit == "wkmeans-minmax":
        return lloyd(v, uniform, wts, 20)
    sv, order = torch.sort(v, dim=1)
    if fit == "kmeans":
        at = ((2 * torch.arange(LEVELS, device=v.device) + 1) * n) // (2 * LEVELS)
        c = sv[:, at]
        c = lloyd(v, c, None, 8)
    else:  # wkmeans: weighted quantiles, then weighted Lloyd
        sw = torch.gather(wts, 1, order)
        cum = torch.cumsum(sw, dim=1)
        total = cum[:, -1:]
        targets = total * (2 * torch.arange(LEVELS, device=v.device).view(1, -1) + 1) / (2 * LEVELS)
        at = torch.clamp(torch.searchsorted(cum.contiguous(), targets.contiguous()), max=n - 1)
        c = torch.gather(sv, 1, at)
        c = lloyd(v, c, wts, 20)
    flat = (hi == lo).expand_as(c)
    return torch.where(flat, lo.expand_as(c), c)


def q_lut8(w, fit, colw):
    rows, cols = w.shape
    assert rows % GROUP == 0, (rows, GROUP)
    groups = rows // GROUP
    per = max(1, CHUNK // (GROUP * cols))
    luts, idx = [], []
    for g0 in range(0, groups, per):
        g1 = min(groups, g0 + per)
        v = w[g0 * GROUP : g1 * GROUP].float().to(DEV).reshape(g1 - g0, GROUP * cols)
        wts = None
        if fit.startswith("wkmeans"):
            wts = colw.to(DEV).float().view(1, 1, cols).expand(g1 - g0, GROUP, cols).reshape(g1 - g0, GROUP * cols)
        c = fit_groups(v, fit, wts)
        lut = c.half()
        lut, _ = torch.sort(lut.float(), dim=1)
        ids = nearest(v, lut)
        luts.append(lut.half().cpu())
        idx.append(ids.to(torch.uint8).cpu())
        del v, wts, c, ids
    lut = torch.cat(luts)
    q = torch.cat(idx).view(rows, cols)
    return {"q": q.contiguous(), "lut": lut.contiguous()}


def dequant(r, shape):
    rows, cols = shape
    if "s" in r:
        return r["q"].float() * r["s"].float().unsqueeze(1)
    return torch.gather(r["lut"].float(), 1, r["q"].view(rows // GROUP, -1).long()).view(rows, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fmt", choices=["int8", "lut8"])
    ap.add_argument("out")
    ap.add_argument("--fit", default="uniform", choices=["uniform", "kmeans", "kmeans-minmax", "wkmeans", "wkmeans-minmax"])
    ap.add_argument("--stats", default=None, help="E[x^2] per layer input column (ref_flux3klein.py --stats)")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    stats = {}
    if a.stats:
        raw = torch.load(os.path.expanduser(a.stats))
        stats = {k: (v["sum"] / max(v["count"], 1)).float() for k, v in raw.items()}
    if a.fit.startswith("wkmeans") and not stats:
        ap.error("--fit wkmeans needs --stats")
    idx = json.load(open(f"{M}/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    tensors, meta = {}, {}
    n_q = n_keep = 0
    err2 = ref2 = werr2 = wref2 = 0.0
    t0 = time.time()
    for f in sorted(set(idx.values())):
        with safe_open(f"{M}/{f}", framework="pt") as sf:
            for k in sf.keys():
                w = sf.get_tensor(k)
                if not (w.ndim == 2 and w.numel() >= MIN_NUMEL and k.endswith(".weight") and w.shape[0] % GROUP == 0):
                    tensors[k] = w.contiguous()
                    n_keep += w.numel()
                    continue
                layer = k[: -len(".weight")]
                colw = stats.get(layer)
                if a.fmt == "int8":
                    r, how = q_int8(w), "int8"
                else:
                    fit = a.fit if (not a.fit.startswith("wkmeans") or colw is not None) else a.fit[1:]
                    r, how = q_lut8(w, fit, colw), f"lut8-{fit}"
                for part, v in r.items():
                    tensors[f"{k}.{part}"] = v
                meta[k] = how
                d = dequant(r, w.shape) - w.float()
                err2 += float((d**2).sum())
                ref2 += float((w.float() ** 2).sum())
                if colw is not None:
                    werr2 += float(((d**2) * colw.view(1, -1)).sum())
                    wref2 += float(((w.float() ** 2) * colw.view(1, -1)).sum())
                n_q += w.numel()
        print(f"{f}: done ({time.time() - t0:.0f} s)", flush=True)
    save_file(tensors, out, metadata={"fmt": a.fmt, "fit": a.fit, "layers": json.dumps(meta)})
    msg = f"{a.fmt} {a.fit}: {n_q / 1e9:.2f}B quantized, {n_keep / 1e6:.1f}M kept; weight rel RMS error {(err2 / ref2) ** 0.5:.5f}"
    if wref2:
        msg += f", weighted by E[x^2] {(werr2 / wref2) ** 0.5:.5f}"
    print(f"{msg}; {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
