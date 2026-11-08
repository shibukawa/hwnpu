"""Hugging Face's logits for the tokens of an `ironlm -forced` run, in the same layout, so an
NPU export (bf16 or 4-bit) can be scored against the checkpoint itself and not only against
another export.

    ~/.cache/hwnpu/dev/ref-venv/bin/python tools/ref/forced_logits.py MODEL CKPT RUN.bin... [--split 256]

MODEL is one of tools/amd/q4_gptq.py's (gemma4, plamo3, lfm3, lfm3vl, qwen35, ...), loaded as it
loads them (the text model alone, in float32). RUN.bin is a file `ironlm -forced -logits` wrote;
RUN.bin.ids holds the tokens it fed (its BOS first). Writes RUN.bin.hf (or OUT with one RUN and
--out): float32 rows, one per row of RUN.bin, the logits after the first --split tokens, then
after each token more, i.e. the model's outputs at positions split-1 .. n-2 of one causal pass.
Prints KL(HF || NPU) per run (mean / median / 95th percentile) and the top-1 agreement.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "amd"))
import q4_gptq  # noqa: E402

SPECS = q4_gptq.MODELS


def kl_rows(ref: np.ndarray, q: np.ndarray):
    def ls(x):
        x = x - x.max(1, keepdims=True)
        return x - np.log(np.exp(x).sum(1, keepdims=True))

    out = []
    for i in range(0, len(ref), 64):
        a, b = ls(ref[i : i + 64].astype(np.float64)), ls(q[i : i + 64].astype(np.float64))
        out.append((np.exp(a) * (a - b)).sum(1))
    kl = np.concatenate(out)
    top = float((ref.argmax(1) == q.argmax(1)).mean())
    return float(kl.mean()), float(np.median(kl)), float(np.percentile(kl, 95)), top


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(SPECS))
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--split", type=int, default=256)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    import torch

    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    m, _ = q4_gptq.load(a.ckpt, SPECS[a.model])
    print(f"loaded in {time.time() - t0:.0f} s")
    cache = {}
    for run in a.runs:
        ids = np.fromfile(str(run) + ".ids", np.uint32).astype(np.int64)
        key = ids.tobytes()
        if key not in cache:
            t0 = time.time()
            with torch.no_grad():
                lg = q4_gptq.logits_of(m, ids.tolist()).numpy()
            cache[key] = lg[a.split - 1 : len(ids) - 1]
            print(f"{run.name}: {len(ids)} tokens in {time.time() - t0:.0f} s")
        hf = cache[key]
        out = a.out if a.out and len(a.runs) == 1 else Path(str(run) + ".hf")
        hf.astype(np.float32).tofile(out)
        npu = np.fromfile(run, np.float32)
        V = hf.shape[1]
        npu = npu.reshape(-1, npu.size // len(hf))[:, :V] if npu.size % len(hf) == 0 else None
        if npu is None:
            print(f"{run.name}: rows do not match ({len(hf)} expected)")
            continue
        mean, med, p95, top = kl_rows(hf, npu)
        print(f"{run.name}: KL(HF || NPU) {mean:.4f} / {med:.4f} / {p95:.3f}, top-1 {100 * top:.1f} %  ({len(hf)} rows)")


if __name__ == "__main__":
    main()
