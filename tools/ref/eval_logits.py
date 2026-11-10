"""Score a converted package against the PyTorch reference of ref_logits.py with coremltools.

    python eval_logits.py MODEL.mlpackage|MODEL.parts.json REF_PREFIX... [--units cpu|ane|gpu] [--layers N]
                          [--table NAME=CHECKPOINT:TENSOR]

REF_PREFIX is the path without extension (ref_bf16 -> ref_bf16.hidden.bin, ref_bf16.logits.bin);
several are scored in turn with the model loaded once. A parts.json (hwnpu-dev convert's split of a
model without a cache, -seq N) runs its parts in order, each part's outputs bound to the inputs
of the same name of the parts after it (y as the next x). Inputs other than x come from an
earlier part, from REF_PREFIX.NAME.bin (ref_gemma4.py's), or, with --table, from the rows of a
checkpoint table for the reference's tokens (Gemma 4's per-layer embeddings: --table
ple=DIR/model.safetensors:model.language_model.embed_tokens_per_layer.weight).

With --layers N the package is a truncated model and its output is compared with the hidden state
after layer N; otherwise its logits are compared: relative RMS error, cosine, argmax agreement and
the mean KL divergence of the reference distribution from the model's (and its median and
largest over the positions). --units ane is Core ML's CPU_AND_NE: the Neural Engine where the
program is placed on it (hwnpu-dev info -model says where). Needs coremltools, numpy and macOS (and
safetensors with --table); hwnpu-dev check -package does the same on any platform with the fp32 CPU
interpreter.
"""
import argparse
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--units", default="cpu", choices=["cpu", "ane", "gpu"])
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--table", action="append", default=[], help="NAME=CHECKPOINT:TENSOR, an input of the rows of a table")
    a = ap.parse_args()
    import coremltools as ct

    units = {"cpu": ct.ComputeUnit.CPU_ONLY, "ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU}[a.units]
    paths = [a.model]
    if a.model.endswith(".parts.json"):
        base = a.model[: -len(".parts.json")]
        paths, i = [], 1
        while os.path.isdir(f"{base}.part{i}.mlpackage"):
            paths.append(f"{base}.part{i}.mlpackage")
            i += 1
    t0 = time.time()
    models = [ct.models.MLModel(p, compute_units=units) for p in paths]
    load = time.time() - t0
    tables = dict(t.split("=", 1) for t in a.table)
    x_shape = next(tuple(i.type.multiArrayType.shape) for i in models[0].get_spec().description.input if i.name == "x")
    seq, d = x_shape[1], x_shape[2]
    for ref in a.refs:
        hidden = np.fromfile(ref + ".hidden.bin", dtype="<f4").reshape(-1, seq, d)
        ids = np.fromfile(ref + ".ids.bin", dtype="<i4")
        env = {"x": hidden[0][None].astype(np.float16)}
        for m in models:
            feed = {}
            for inp in m.get_spec().description.input:
                shape = tuple(inp.type.multiArrayType.shape)
                if inp.name in env:
                    feed[inp.name] = env[inp.name]
                elif inp.name in tables:
                    from safetensors import safe_open

                    ckpt, tensor = tables[inp.name].rsplit(":", 1)
                    with safe_open(ckpt, "pt") as f:
                        rows = f.get_slice(tensor)
                        v = np.stack([rows[int(t)].float().numpy() for t in ids])
                    feed[inp.name] = v.reshape(shape).astype(np.float16)
                else:
                    feed[inp.name] = np.fromfile(f"{ref}.{inp.name}.bin", dtype="<f4").reshape(shape).astype(np.float16)
            env.update({k: np.asarray(v) for k, v in m.predict(feed).items()})
            env["x"] = env["y"]
        y = env["y"].astype(np.float32)[0]
        if a.layers:
            r = hidden[a.layers]
            what = f"hidden after layer {a.layers}"
        else:
            r = np.fromfile(ref + ".logits.bin", dtype="<f4").reshape(seq, -1)
            what = "logits"
        rel = np.sqrt(((y - r) ** 2).sum() / (r ** 2).sum())
        cos = (y * r).sum() / np.sqrt((y * y).sum() * (r * r).sum())
        line = f"{a.model} ({a.units}) {os.path.basename(ref)} {what}: rel RMS err {rel:.4f}, cosine {cos:.6f}, max |diff| {np.abs(y - r).max():.3g}"
        if not a.layers:
            agree = int((y.argmax(-1) == r.argmax(-1)).sum())
            lp = r - r.max(-1, keepdims=True)
            lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
            lq = y - y.max(-1, keepdims=True)
            lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
            kl = (np.exp(lp) * (lp - lq)).sum(-1)
            line += f"; argmax agreement {agree}/{seq}, mean KL(ref || model) {kl.mean():.4f} nats (median {np.median(kl):.4f}, max {kl.max():.3f})"
        print(line + f" [loaded in {load:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
