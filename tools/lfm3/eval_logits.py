"""Score a converted LFM3 package against the PyTorch reference of ref_logits.py.

    python eval_logits.py MODEL.mlpackage REF_PREFIX [--units cpu|ane|gpu] [--layers N]

REF_PREFIX is the path without extension (ref_bf16 -> ref_bf16.hidden.bin, ref_bf16.logits.bin).
With --layers N the package is a truncated model and its output is compared with the hidden state
after layer N; otherwise its logits are compared: relative RMS error, cosine, argmax agreement and
the mean KL divergence of the reference distribution from the model's. Needs coremltools, numpy.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("ref")
    ap.add_argument("--units", default="cpu", choices=["cpu", "ane", "gpu"])
    ap.add_argument("--layers", type=int, default=0)
    a = ap.parse_args()
    import coremltools as ct

    units = {"cpu": ct.ComputeUnit.CPU_ONLY, "ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU}[a.units]
    m = ct.models.MLModel(a.model, compute_units=units)
    x_shape = tuple(m.get_spec().description.input[0].type.multiArrayType.shape)
    seq, d = x_shape[1], x_shape[2]
    hidden = np.fromfile(a.ref + ".hidden.bin", dtype="<f4").reshape(-1, seq, d)
    y = np.asarray(m.predict({"x": hidden[0][None].astype(np.float16)})["y"], np.float32)[0]
    if a.layers:
        ref = hidden[a.layers]
        what = f"hidden after layer {a.layers}"
    else:
        ref = np.fromfile(a.ref + ".logits.bin", dtype="<f4").reshape(seq, -1)
        what = "logits"
    rel = np.sqrt(((y - ref) ** 2).sum() / (ref ** 2).sum())
    cos = (y * ref).sum() / np.sqrt((y * y).sum() * (ref * ref).sum())
    line = f"{a.model} ({a.units}) {what}: rel RMS err {rel:.4f}, cosine {cos:.6f}, max |diff| {np.abs(y - ref).max():.3g}"
    if not a.layers:
        agree = int((y.argmax(-1) == ref.argmax(-1)).sum())
        lp = ref - ref.max(-1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
        lq = y - y.max(-1, keepdims=True)
        lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
        kl = (np.exp(lp) * (lp - lq)).sum(-1).mean()
        line += f"; argmax agreement {agree}/{seq}, mean KL(ref || model) {kl:.4f} nats"
    print(line)


if __name__ == "__main__":
    main()
