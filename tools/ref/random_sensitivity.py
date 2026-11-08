"""How far bf16 arithmetic moves a model's hidden states, with its real weights and
with weights drawn as IRON's checks draw them (iron.common.graph.fill: a matrix normal at
deviation 1/sqrt(in-features), a vector 1 + normal/10, a bias normal/10).

    ~/.cache/hwnpu/dev/ref-venv/bin/python tools/ref/random_sensitivity.py MODEL CKPT [--seq 32] [--seed 0]

MODEL is one of tools/amd/q4_gptq.py's. For each set of weights, runs the text twice, in float32
and wholly in bf16 (weights and arithmetic, as the NPU and IRON's reference both round, in their
own orders), and prints each layer's cosine between the two hidden states (all positions), the
logits' cosine and top-1 agreement, and how far apart the logits are (their deviation, and the
median gap between the first and the second). A cosine that falls layer by layer with the drawn
weights and stays near 1 with the real ones says the drawn weights amplify rounding (docs/design.md
16.50, 16.70: the NPU against IRON's reference on drawn weights).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "amd"))
import q4_gptq  # noqa: E402

TEXT = "The Neural Engine runs small language models at low power. 日本の首都は東京です。hwnpu is a Go library for NPUs."


def draw(m, seed: int):
    """IRON's random fill of every parameter of m, in place."""
    import torch

    g = torch.Generator().manual_seed(seed)
    for n, p in m.named_parameters():
        x = torch.randn(p.shape, generator=g, dtype=torch.float32)
        if p.dim() >= 2:
            p.data = x / float(np.sqrt(np.prod(p.shape[1:])))
        else:
            p.data = x / 10 + (0.0 if n.endswith("bias") else 1.0)


def run(m, ids, spec):
    import torch

    layers = q4_gptq._get(m, spec["layers"])
    hidden = []
    hooks = [lay.register_forward_hook(lambda mod, i, o: hidden.append((o[0] if isinstance(o, tuple) else o).detach().clone()))
             for lay in layers]
    with torch.no_grad():
        logits = q4_gptq.logits_of(m, ids)
    for h in hooks:
        h.remove()
    return hidden, logits


def compare(m, ids, spec, label):
    import torch

    a_h, a_l = run(m, ids, spec)
    _to(m, torch.bfloat16)
    b_h, b_l = run(m, ids, spec)
    _to(m, torch.float32)
    b_h, b_l = [h.float() for h in b_h], b_l.float()
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x.flatten().double(), y.flatten().double(), 0))  # noqa: E731
    per = [cos(x, y) for x, y in zip(a_h, b_h)]
    rms = [float(x.double().pow(2).mean().sqrt()) for x in a_h]
    print(f"{label}: logits cosine {cos(a_l, b_l):.5f}, top-1 {float((a_l.argmax(-1) == b_l.argmax(-1)).double().mean()) * 100:.0f} %")
    print("  layer cosine: " + " ".join(f"{c:.4f}" for c in per))
    print("  layer rms:    " + " ".join(f"{r:.3g}" for r in rms))
    top2 = a_l.topk(2, -1).values
    print(f"  logits: deviation {float(a_l.std(-1).mean()):.3g}, top-1 minus top-2 median {float((top2[:, 0] - top2[:, 1]).median()):.3g}")


def _to(m, dtype):
    """The whole model, its weights and arithmetic, in dtype (TextLM wraps the module)."""
    getattr(m, "m", m).to(dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(q4_gptq.MODELS))
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    spec = q4_gptq.MODELS[a.model]
    m, tok = q4_gptq.load(a.ckpt, spec)
    ids = [spec["bos"]] + tok.encode(TEXT, add_special_tokens=False)[: a.seq - 1]
    compare(m, ids, spec, "real weights")
    draw(m, a.seed)
    compare(m, ids, spec, f"drawn weights (seed {a.seed})")


if __name__ == "__main__":
    main()
