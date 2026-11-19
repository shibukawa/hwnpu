"""GPTQ of a Qwen2 model's lookup tables (Apple's lutN), a decoder layer at a time.

q4_gptq.py runs the whole model in float32, which Qwen2.5-Coder-7B (30 GB) is too large for on a
16 GB Mac. This builds transformers' Qwen2DecoderLayer one layer at a time from the checkpoint,
as ref_qwen2.py does, and runs the calibration texts through the layers already quantized: each
layer's projections are quantized in the order they run (q, k and v on the layer's input, then o
on the attention's output, gate and up, down), H = E[x x^T] of their inputs collected from the
layer as quantized so far. A weight's tables are fixed first, one of 2^bits entries for each
block of 16 rows by --col-block columns (0: whole rows), by Lloyd's algorithm weighted by H's
diagonal (each column's error counted by its input's mean square); GPTQ (Frantar et al. 2022)
then picks each weight's entry in column order and moves its error onto the columns not yet
quantized. lm_head is done last on the final norm's output.

Writes a codes directory (internal/palette.Codes: index.json, KEY.idx, KEY.tab, the key
q4_gptq.key's) that hwnpu-dev convert -codes and hwnpu-chat -codes read:

    PY=~/.cache/hwnpu/dev/ref-venv/bin/python
    $PY tools/ref/qwen2_gptq.py MODEL_DIR OUT --calib FILE... [--fmt lut4] [--col-block 0]
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
sys.path.insert(0, str(HERE.parent / "amd"))
from ref_qwen2 import Checkpoint  # noqa: E402
from q4_gptq import key  # noqa: E402

GROUP = 16


def f16(x):
    return x.to(torch.float16).to(torch.float32)


def fit_tables(w, d, bits, cb, iters=20):
    """Tables (N/16, K/cb, 2^bits) for w (N, K), each block's fitted by Lloyd's algorithm weighted by
    d (K), each column's weight, from the weighted quantiles; rounded to fp16."""
    N, K = w.shape
    k = 1 << bits
    nb = K // cb
    # blocks (N/16 * nb, 16*cb) and their weights
    v = w.reshape(N // GROUP, GROUP, nb, cb).permute(0, 2, 1, 3).reshape(-1, GROUP * cb)
    wt = d.reshape(nb, cb).repeat_interleave(1, 0)[None].expand(N // GROUP, nb, cb)
    wt = wt[:, :, None, :].expand(N // GROUP, nb, GROUP, cb).reshape(-1, GROUP * cb)
    out = torch.empty(v.shape[0], k)
    step = max(1, (1 << 24) // (GROUP * cb * k))
    for s in range(0, v.shape[0], step):
        vb, wb = v[s : s + step], wt[s : s + step].clamp_min(1e-12)
        order = vb.argsort(1)
        vs, ws = vb.gather(1, order), wb.gather(1, order)
        cum = ws.cumsum(1) / ws.sum(1, keepdim=True)
        qs = (torch.arange(k, dtype=torch.float32) * 2 + 1) / (2 * k)
        pos = torch.searchsorted(cum.contiguous(), qs[None].expand(vb.shape[0], k).contiguous()).clamp_max(vb.shape[1] - 1)
        c = vs.gather(1, pos)
        for _ in range(iters):
            c, _ = c.sort(1)
            a = (vb[:, :, None] - c[:, None, :]).abs().argmin(2)
            num = torch.zeros_like(c).scatter_add_(1, a, wb * vb)
            den = torch.zeros_like(c).scatter_add_(1, a, wb)
            c = torch.where(den > 0, num / den.clamp_min(1e-30), c)
        out[s : s + step] = f16(c.sort(1)[0])
    return out.reshape(N // GROUP, nb, k)


def gptq(w, H, T, cb, block=128, damp=0.01):
    """GPTQ of w (N, K) against H (K, K) with the fixed tables T (N/16, K/cb, k): the indices (N, K)
    uint8 and the quantized weight."""
    W = w.clone()
    N, K = W.shape
    # float64 where it fits (a down projection's 18944 squared is 2.9 GB in float64: float32)
    H = H.to(torch.float64 if K <= 8192 else torch.float32).clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, dtype=H.dtype)
    L = torch.linalg.cholesky(H)
    del H
    Hi = torch.cholesky_inverse(L)
    del L
    Hinv = torch.linalg.cholesky(Hi, upper=True).float()
    del Hi
    idx = torch.empty((N, K), dtype=torch.uint8)
    Q = torch.empty_like(W)
    rowT = T.repeat_interleave(GROUP, 0)  # (N, K/cb, k)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            x = W1[:, i]
            t = rowT[:, col // cb]  # (N, k)
            c = (x[:, None] - t).abs().argmin(1)
            q = t.gather(1, c[:, None])[:, 0]
            idx[:, col] = c.to(torch.uint8)
            Q[:, col] = q
            err = (x - q) / Hi[i, i]
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return idx, Q


class Codes:
    """A codes directory being written (internal/palette.Codes' layout)."""

    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.index = {}

    def add(self, name, w, idx, T, fmt, cb):
        k = key(w.numpy())
        (self.out / f"{k}.idx").write_bytes(idx.numpy().astype(np.uint8).tobytes())
        (self.out / f"{k}.tab").write_bytes(T.reshape(-1).numpy().astype("<f4").tobytes())
        e = {"name": name, "fmt": fmt, "shape": list(w.shape), "group": GROUP}
        if cb != w.shape[1]:
            e["col_block"] = cb
        self.index[k] = e

    def close(self):
        (self.out / "index.json").write_text(json.dumps(self.index, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--calib", nargs="+", required=True)
    ap.add_argument("--calib-tokens", type=int, default=512)
    ap.add_argument("--fmt", default="lut4", choices=("lut4", "lut6", "lut8"))
    ap.add_argument("--col-block", type=int, default=0,
                    help="columns a table (0: whole rows). Core AI's Neural Engine reads only the first block's table of a row "
                         "(docs/design.md 16.230), so its programs take whole rows; blocks are for measuring on the CPU")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--layers", type=int, default=0, help="only the first N layers (0: all), for trying it out")
    a = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2 import modeling_qwen2 as M

    bits = {"lut4": 4, "lut6": 6, "lut8": 8}[a.fmt]
    cfg = AutoConfig.from_pretrained(a.model_dir)
    if cfg.model_type != "qwen2":
        raise SystemExit(f"model_type {cfg.model_type} is not qwen2")
    cfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    seqs = [tok(Path(f).read_text(), add_special_tokens=False)["input_ids"][: a.calib_tokens] for f in a.calib]
    ck = Checkpoint(a.model_dir)
    torch.set_grad_enabled(False)
    emb = ck.get("model.embed_tokens.weight")
    hs = [emb[torch.tensor(s)][None] for s in seqs]
    del emb
    rope = M.Qwen2RotaryEmbedding(config=cfg)
    codes = Codes(Path(a.out))
    t00 = time.time()

    def run(layer):
        out = []
        for h in hs:
            n = h.shape[1]
            pos = torch.arange(n)[None]
            mask = torch.full((n, n), float("-inf")).triu(1)[None, None]
            o = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=rope(h, pos))
            out.append(o[0] if isinstance(o, tuple) else o)
        return out

    def hessian(layer, mod):
        """H = E[x x^T] of mod's input, over the calibration texts through layer as it is."""
        acc = {}

        def hook(_m, args):
            x = args[0].reshape(-1, args[0].shape[-1])
            x = x.double() if x.shape[-1] <= 8192 else x.float()
            acc["H"] = acc.get("H", 0) + x.T @ x
            acc["n"] = acc.get("n", 0) + x.shape[0]

        hk = mod.register_forward_pre_hook(hook)
        run(layer)
        hk.remove()
        return (acc["H"] / acc["n"]).float()

    def quantize(name, lin, H):
        w = lin.weight.data.float()
        cb = a.col_block or w.shape[1]
        T = fit_tables(w, torch.diag(H), bits, cb)
        idx, Q = gptq(w, H, T, cb, damp=a.damp)
        codes.add(name, w, idx, T, a.fmt, cb)
        lin.weight.data = Q
        e = ((Q - w).pow(2) * torch.diag(H)[None]).sum() / (w.pow(2) * torch.diag(H)[None]).sum()
        return float(e.sqrt())

    n = cfg.num_hidden_layers if a.layers == 0 else a.layers
    for i in range(n):
        t0 = time.time()
        layer = ck.load(M.Qwen2DecoderLayer(cfg, i), f"model.layers.{i}.")
        p = f"model.layers.{i}."
        at, mlp = layer.self_attn, layer.mlp
        errs = []
        H = hessian(layer, at.q_proj)
        for nm in ("q_proj", "k_proj", "v_proj"):
            errs.append(quantize(p + "self_attn." + nm + ".weight", getattr(at, nm), H))
        errs.append(quantize(p + "self_attn.o_proj.weight", at.o_proj, hessian(layer, at.o_proj)))
        H = hessian(layer, mlp.gate_proj)
        for nm in ("gate_proj", "up_proj"):
            errs.append(quantize(p + "mlp." + nm + ".weight", getattr(mlp, nm), H))
        del H
        errs.append(quantize(p + "mlp.down_proj.weight", mlp.down_proj, hessian(layer, mlp.down_proj)))
        hs[:] = run(layer)
        del layer
        print(f"layer {i}: {time.time() - t0:.0f} s, weighted errors " + " ".join(f"{e:.3f}" for e in errs), flush=True)
    if a.layers == 0:
        norm = ck.load(M.Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps), "model.norm.")
        head_name = "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
        if not cfg.tie_word_embeddings:
            xs = torch.cat([norm(h)[0] for h in hs]).double()
            H = (xs.T @ xs / xs.shape[0]).float()
            lin = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            lin.weight.data = ck.get(head_name)
            print(f"lm_head: weighted error {quantize(head_name, lin, H):.3f}", flush=True)
    codes.close()
    print(f"{len(codes.index)} weights in {time.time() - t00:.0f} s")


if __name__ == "__main__":
    main()
