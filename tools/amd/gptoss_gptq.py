#!/usr/bin/env python3
"""GPTQ for gpt-oss's quantized parts (iron_gptoss.py), a decoder layer at a time.

The experts stay MXFP4 exactly (mx4); what the export rounds is each layer's fused q, k, v
projection and o_proj (q8) and the head (q4). Their codes are chosen here by GPTQ (q4_gptq.gptq,
damping 0.1 of H's mean diagonal) against H = E[x x^T] of their inputs on general text
(wikitext-2 and wiki40b-ja, new-model-checklist 5.3), the earlier layers' projections quantized
before the later layers' inputs are taken. The weights are the export's own (iron_gptoss's
pad_tree and SinkAttention: padded to the residual's 3072, q, k and v stacked, the head's rows to
201216), so iron_q4.weight_key finds them.

    PY=~/.cache/hwnpu/dev/ref-venv/bin/python
    $PY tools/amd/gptoss_gptq.py ~/models/gpt-oss-20b OUT [--seqs 64 --length 512]
    python tools/amd/iron_export_lm.py gptoss CKPT EXPORT --q4-codes OUT/codes ...

OUT/codes is iron_q4.load_codes' layout (index.json and KEY.npy).
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
sys.path.insert(0, str(HERE.parent / "ref"))

import iron_q4  # noqa: E402
import q4_gptq  # noqa: E402
import ref_gptoss as R  # noqa: E402

HIDDEN, HP, HEAD_ROWS = 2880, 3072, 512


def calib(tok, seqs: int, length: int):
    """seqs sequences of length tokens, half from wikitext-2's train text, half wiki40b-ja's."""
    out = []
    for name in ("wikitext2-train.txt", "wiki40b-ja-calib.txt"):
        text = (Path.home() / "models/calib" / name).read_text(encoding="utf-8")
        ids = tok.encode(text[: 4_000_000], add_special_tokens=False).ids
        prefix = tok.encode(R.USER, add_special_tokens=False).ids
        step = len(ids) // (seqs // 2)
        for k in range(seqs // 2):
            out.append(prefix + ids[k * step : k * step + length - len(prefix)])
    return out, len(tok.encode(R.USER, add_special_tokens=False).ids)


def padded_cols(w):
    p = np.zeros((w.shape[0], HP), np.float32)
    p[:, : w.shape[1]] = w
    return p


def hessian(xs, skip: int, K: int):
    """E[x x^T] (float64) of the rows of each x (rows, k) past the first skip, padded to K."""
    H = torch.zeros((K, K), dtype=torch.float64)
    n = 0
    for x in xs:
        x = x[skip:].double()
        H[: x.shape[1], : x.shape[1]] += x.T @ x
        n += x.shape[0]
    return (H / n).numpy()


class Codes:
    def __init__(self, d: Path):
        self.dir = d
        d.mkdir(parents=True, exist_ok=True)
        self.index = {}

    def put(self, w, fmt: str, H, damp: float):
        """GPTQ codes of w (the export's float32 array) into the store; its dequantized weights."""
        codes, scale, mn = q4_gptq.gptq(w, H, fmt, damp=damp)
        blocks = q4_gptq.blocks_of(codes, scale, mn, fmt)
        k = iron_q4.weight_key(w)
        np.save(self.dir / f"{k}.npy", blocks)
        self.index[k] = dict(fmt=fmt, shape=list(w.shape))
        (self.dir / "index.json").write_text(json.dumps(self.index, indent=1))
        return iron_q4.dequantize_fmt(blocks, *w.shape, fmt)

    def rtn(self, w, fmt: str):
        return iron_q4.dequantize_fmt(iron_q4.quantize_fmt(w, fmt), *w.shape, fmt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out", type=Path)
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--damp", type=float, default=0.1)
    ap.add_argument("--attn", default="q8")
    ap.add_argument("--head", default="q4")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    ck = R.Checkpoint(a.model_dir)
    g = R.GptOss(ck)
    seqs, skip = calib(R.tokenizer(a.model_dir), a.seqs, a.length)
    store = Codes(a.out / "codes")
    hs = [g.embed(s) for s in seqs]
    t0 = time.time()
    for i in range(g.layers):
        w = ck.layer(i)
        # q, k, v stacked and padded, as the export's SinkAttention.qkv (pad_tree's columns)
        xs = [g.norm(h, w["input_layernorm.weight"]) for h in hs]
        wq = np.concatenate([padded_cols(w[f"self_attn.{p}_proj.weight"].numpy()) for p in "qkv"])
        qkv = store.put(wq, a.attn, hessian(xs, skip, HP), a.damp)
        H, Gk, D = g.H, g.G, g.D
        wq_ = torch.from_numpy(qkv[:, :HIDDEN].copy())
        w["self_attn.q_proj.weight"] = wq_[: H * D]
        w["self_attn.k_proj.weight"] = wq_[H * D : H * D + Gk * D]
        w["self_attn.v_proj.weight"] = wq_[H * D + Gk * D :]
        # o_proj's inputs: the heads' outputs of the quantized q, k, v
        os_ = [attention_heads(g, i, x, w) for x in xs]
        wo = w["self_attn.o_proj.weight"]
        wop = np.zeros((HP, H * D), np.float32)
        wop[:HIDDEN] = wo.numpy()
        o = store.put(wop, a.attn, hessian(os_, skip, H * D), a.damp)
        w["self_attn.o_proj.weight"] = torch.from_numpy(o[:HIDDEN].copy())
        hs = [g.layer(i, h, w)[0] for h in hs]
        print(f"layer {i}: {time.time() - t0:.0f} s", flush=True)
    # the head over the final norm's output (padded), its rows padded as the export's
    nw = ck.get("model.norm.weight")
    xs = [g.norm(h, nw) for h in hs]
    head = ck.get("lm_head.weight").numpy()
    rows = -(-head.shape[0] // HEAD_ROWS) * HEAD_ROWS
    hp = np.zeros((rows, HP), np.float32)
    hp[: head.shape[0], :HIDDEN] = head
    store.put(hp, a.head, hessian(xs, skip, HP), a.damp)
    print(f"head: {time.time() - t0:.0f} s; {len(store.index)} weights in {store.dir}")


def attention_heads(g, i, x, w):
    """The heads' outputs (n, H D) of layer i's attention on x (n, hidden) normed: GptOss.attention
    up to o_proj."""
    n, H, Gk, D = x.shape[0], g.H, g.G, g.D
    q = (x @ w["self_attn.q_proj.weight"].T + w["self_attn.q_proj.bias"]).view(n, H, D)
    k = (x @ w["self_attn.k_proj.weight"].T + w["self_attn.k_proj.bias"]).view(n, Gk, D)
    v = (x @ w["self_attn.v_proj.weight"].T + w["self_attn.v_proj.bias"]).view(n, Gk, D)
    cos, sin = g.rope(n)
    q, k = g.rotate(q, cos, sin), g.rotate(k, cos, sin)
    kk, vv = k.repeat_interleave(H // Gk, 1), v.repeat_interleave(H // Gk, 1)
    s = torch.einsum("nhd,thd->hnt", q, kk) * D ** -0.5
    pos = torch.arange(n)
    keep = pos[None] <= pos[:, None]
    if g.c["layer_types"][i] == "sliding_attention":
        keep &= pos[:, None] - pos[None] < g.window
    s = s.masked_fill(~keep[None], float("-inf"))
    sink = w["self_attn.sinks"].view(H, 1, 1).expand(H, n, 1)
    full = torch.cat([s, sink], -1)
    p = torch.softmax(full - full.max(-1, keepdim=True).values, -1)[..., :-1]
    return torch.einsum("hnt,thd->nhd", p, vv).reshape(n, H * D)


if __name__ == "__main__":
    main()
