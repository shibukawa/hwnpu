#!/usr/bin/env python3
"""How much Gemma 4 loses when its per-layer embedding table (PLE) is stored in fewer bits.

The host reads one row of the table a token on every NPU (AMD: ple.bin of iron_hostple, bf16;
Apple and Intel: tables.safetensors, bf16, read by llm.TokenRow) and the device never holds it:
4.4 GiB of bf16 for E2B, 5.25 GiB for E4B. A row stored in 8 bits with its scales and expanded
on the host when it is read would halve the file and what the page cache holds of it, on every
NPU, with no change on the device. AMD's Quark article on Qwen3.8-Flash-Next's n-gram table
(rocm.blogs.amd.com, quark-qwen-ple) found that table in FP8 with one scale for the whole
tensor no worse than bf16 on MMLU-Pro.

This measures it the way q4_gptq.py measures weight formats: Gemma 4's text model in fp32 on the
CPU (q4_gptq.load_gemma4_lean, the tables read from the checkpoint's file), the KL of its logits
from those of the bf16 tables, on the evaluation texts. Each variant rounds the rows when they
are looked up: with a scale per row or per layer's slice of a row, that is what a table stored so
gives. The other weights stay fp32, so the KL is the table's alone.

  ~/bonsaiai-dev/refenv/bin/python tools/amd/ple_quant_kl.py CKPT OUT.json [--eval FILES]

Variants (bytes a row of E2B: 35 layers x 256, bf16 17920):
  fp8-tensor   E4M3, one scale for the table (the article's)            8960
  fp8-row      E4M3, a scale a row                                      8962
  int8-row     symmetric int8, a scale a row                            8962
  int8-layer   symmetric int8, a scale a row's layer slice (256)        9030
  int4-layer   symmetric int4, a scale a layer slice                    4550
  int4-g32     symmetric int4, a scale every 32 values                  5040
  zero         no table (zeros): how much the table matters at all         0
  emb-int8-row the token embedding (not the PLE) in int8-row, the head still fp32
  both-int8    int8-layer PLE and int8-row token embedding
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q4_gptq  # noqa: E402
from q4_gptq import REPO, TableRows, kl_stats, logits_of, texts  # noqa: E402


def _sym(x, groups: int, qmax: int):
    """Symmetric rounding of each row of x (n, w) in groups of w // groups values."""
    n, w = x.shape
    g = x.reshape(n, groups, w // groups)
    s = np.abs(g).max(-1, keepdims=True) / qmax
    s = np.where(s == 0, 1, s).astype(np.float16).astype(np.float32)  # scales stored in fp16
    return (np.clip(np.rint(g / s), -qmax, qmax) * s).reshape(n, w)


def _fp8(x, s):
    from ml_dtypes import float8_e4m3fn

    return (x / s).astype(float8_e4m3fn).astype(np.float32) * s


def make_round(variant: str, layers: int, table_max: float):
    if variant == "fp8-tensor":
        s = np.float32(table_max / 448)
        return lambda x: _fp8(x, s)
    if variant == "fp8-row":
        def f(x):
            s = (np.abs(x).max(-1, keepdims=True) / 448).astype(np.float16).astype(np.float32)
            return _fp8(x, np.where(s == 0, 1, s))
        return f
    if variant == "int8-row":
        return lambda x: _sym(x, 1, 127)
    if variant == "int8-layer":
        return lambda x: _sym(x, layers, 127)
    if variant == "int4-layer":
        return lambda x: _sym(x, layers, 7)
    if variant == "int4-g32":
        return lambda x: _sym(x, x.shape[1] // 32, 7)
    if variant == "zero":
        return lambda x: np.zeros_like(x)
    raise ValueError(variant)


class RoundedRows(TableRows):
    """TableRows whose rows are rounded (rnd, on the bf16 values before the scale) when looked up."""

    def __init__(self, base: TableRows, rnd):
        self.rows, self.scale, self.rnd = base.rows, base.scale, rnd

    def __call__(self, ids):
        import torch

        r = np.ascontiguousarray(self.rows[ids.numpy().reshape(-1)]).astype(np.uint32) << 16
        x = self.rnd(r.view(np.float32))
        x = torch.from_numpy(np.ascontiguousarray(x, np.float32)).reshape(*ids.shape, -1)
        return x * torch.tensor(self.scale, dtype=torch.float32)


def table_absmax(rows, chunk: int = 4096) -> float:
    m = 0.0
    for i in range(0, rows.shape[0], chunk):
        r = (np.asarray(rows[i : i + chunk]).astype(np.uint32) << 16).view(np.float32)
        m = max(m, float(np.abs(r).max()))
    return m


VARIANTS = ["fp8-tensor", "fp8-row", "int8-row", "int8-layer", "int4-layer", "int4-g32", "zero",
            "emb-int8-row", "both-int8"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path, help="the results as JSON")
    ap.add_argument("--eval", nargs="*", help="evaluation texts (default q4_gptq.EVAL)")
    ap.add_argument("--variants", nargs="*", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--eval-len", type=int, default=656)
    ap.add_argument("--eval-skip", type=int, default=256)
    a = ap.parse_args(argv)

    import torch

    torch.set_grad_enabled(False)
    t0 = time.time()
    m = q4_gptq.load_gemma4_lean(a.ckpt)
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(a.ckpt)
    cfg = m.config
    layers = cfg.num_hidden_layers
    ple0, emb0 = m.model.embed_tokens_per_layer, m.model.embed_tokens
    paths = [p for p in (a.eval or q4_gptq.EVAL) if (Path(p) if Path(p).is_absolute() else REPO / p).is_file()]
    ev = [texts([p], tok, 2, a.eval_len, 1)[0] for p in paths]
    refs = [logits_of(m, ids)[a.eval_skip :].clone() for ids in ev]
    print(f"loaded, reference logits: {time.time() - t0:.0f} s", flush=True)
    tmax = table_absmax(ple0.rows) if "fp8-tensor" in a.variants else 0.0
    print(f"PLE table {ple0.rows.shape}, |max| {tmax:.4g}", flush=True)

    out = dict(ckpt=str(a.ckpt), eval_len=a.eval_len, eval_skip=a.eval_skip, table_absmax=tmax, results={})
    for v in a.variants:
        t = time.time()
        ple, emb = ple0, emb0
        if v == "emb-int8-row":
            emb = RoundedRows(emb0, make_round("int8-row", layers, 0))
        elif v == "both-int8":
            ple = RoundedRows(ple0, make_round("int8-layer", layers, 0))
            emb = RoundedRows(emb0, make_round("int8-row", layers, 0))
        else:
            ple = RoundedRows(ple0, make_round(v, layers, tmax))
        m.model.embed_tokens_per_layer, m.model.embed_tokens = ple, emb
        res = {}
        for p, ids, ref in zip(paths, ev, refs):
            mean, med, p95, top = kl_stats(ref, logits_of(m, ids)[a.eval_skip :])
            res[Path(p).name] = dict(kl_mean=mean, kl_median=med, kl_p95=p95, top1=top)
            print(f"{v:12s} {Path(p).name:22s} KL {mean:.5f} / {med:.5f} / {p95:.4f}, top-1 {top * 100:.1f} %",
                  flush=True)
        out["results"][v] = res
        print(f"{v}: {time.time() - t:.0f} s", flush=True)
        a.out.write_text(json.dumps(out, indent=1))
    m.model.embed_tokens_per_layer, m.model.embed_tokens = ple0, emb0
    print(f"{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
