"""Generate a ladder of transformer encoder models to find where the Neural Engine starts to pay off.

Each model is a stack of BERT style encoder layers (multi head attention with 64 wide heads, a
4x GELU feed forward network, post layer norm) over one fixed length sequence, x of shape
(1, seq, d), fp16, with random weights. Weights do not change the cost, so random is enough.

    python gen_sweep.py OUT_DIR             write tf_d{d}_s{seq}_l{layers}.mlpackage and sweep.json
    python gen_sweep.py OUT_DIR --report powerbench.json
                                            join a powerbench -json result with sweep.json

Measure with
    go run ./cmd/powerbench -model 'OUT_DIR/*.mlpackage' -json powerbench.json
which runs every model on the Neural Engine and again on the CPU alone.
Needs coremltools >= 8 and numpy.
"""
import argparse
import json
import os
import sys

import numpy as np

# (d, seq, layers), roughly 2 to 8 times more work per step. (384, 128, 6) has the shape of
# all-MiniLM-L6-v2.
LADDER = [
    (64, 16, 1),
    (128, 32, 1),
    (256, 64, 2),
    (384, 128, 2),
    (384, 128, 6),
    (768, 128, 4),
    (1024, 256, 6),
]


def flops(d, seq, layers):
    # QKV + output projections 4*d*d, FFN 2*d*4d, attention scores and mix 2*seq*d, per token.
    per_layer = 2 * seq * (4 * d * d + 8 * d * d) + 2 * 2 * seq * seq * d
    return per_layer * layers


def params(d, layers):
    return layers * (12 * d * d + 13 * d)


def build(d, seq, layers, seed):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    rng = np.random.default_rng(seed)
    heads = max(d // 64, 1)
    dh = d // heads

    def w(o, i):
        return (rng.standard_normal((o, i)) / np.sqrt(i)).astype(np.float16)

    def zeros(n):
        return np.zeros(n, np.float16)

    def ones(n):
        return np.ones(n, np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, seq, d), dtype=types.fp16)], opset_version=ct.target.macOS13)
    def prog(x):
        for _ in range(layers):
            def split(t):
                t = mb.reshape(x=t, shape=[1, seq, heads, dh])
                return mb.transpose(x=t, perm=[0, 2, 1, 3])

            q = split(mb.linear(x=x, weight=w(d, d), bias=zeros(d)))
            k = split(mb.linear(x=x, weight=w(d, d), bias=zeros(d)))
            v = split(mb.linear(x=x, weight=w(d, d), bias=zeros(d)))
            a = mb.matmul(x=q, y=k, transpose_y=True)
            a = mb.mul(x=a, y=np.float16(1 / np.sqrt(dh)))
            a = mb.softmax(x=a, axis=-1)
            o = mb.transpose(x=mb.matmul(x=a, y=v), perm=[0, 2, 1, 3])
            o = mb.linear(x=mb.reshape(x=o, shape=[1, seq, d]), weight=w(d, d), bias=zeros(d))
            x = mb.layer_norm(x=mb.add(x=x, y=o), axes=[-1], gamma=ones(d), beta=zeros(d), epsilon=np.float16(1e-5))
            f = mb.gelu(x=mb.linear(x=x, weight=w(4 * d, d), bias=zeros(4 * d)))
            f = mb.linear(x=f, weight=w(d, 4 * d), bias=zeros(d))
            x = mb.layer_norm(x=mb.add(x=x, y=f), axes=[-1], gamma=ones(d), beta=zeros(d), epsilon=np.float16(1e-5))
        return mb.identity(x=x, name="y")

    return ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.macOS13)


def generate(out):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ane_pad import placement

    os.makedirs(out, exist_ok=True)
    rows = []
    for i, (d, seq, layers) in enumerate(LADDER):
        name = f"tf_d{d}_s{seq}_l{layers}"
        model = build(d, seq, layers, seed=i)
        on_ne, off = placement(model)
        model.save(os.path.join(out, name + ".mlpackage"))
        row = {"name": name, "d": d, "seq": seq, "layers": layers,
               "params": params(d, layers), "flops": flops(d, seq, layers),
               "ops_on_ne": on_ne, "ops_elsewhere": len(off)}
        rows.append(row)
        print(f"{name}: {row['params'] / 1e6:.2f} M params, {row['flops'] / 1e6:.1f} MFLOP, "
              f"{on_ne} op(s) on the Neural Engine, elsewhere: {sorted(set(off)) or 'none'}")
    with open(os.path.join(out, "sweep.json"), "w") as f:
        json.dump(rows, f, indent=2)


def report(out, bench):
    """Print a Markdown table of every model measured both on the CPU alone and on the NPU."""
    path = os.path.join(out, "sweep.json")
    sweep = {}
    if os.path.exists(path):
        with open(path) as f:
            sweep = {r["name"]: r for r in json.load(f)}
    with open(bench) as f:
        work = json.load(f)["workloads"]
    by = {}
    for w in work:
        if w.get("rounds"):
            base = w["name"].removesuffix(" [CPU only]").removesuffix(" [GPU]")
            units = w.get("units") or ("cpu" if w.get("cpu_only") else "npu")
            by.setdefault(base, {})[units] = w["mean"]

    def work_of(name):
        if name in sweep:
            return sweep[name]["flops"]
        if name.startswith("minilm_s"):  # convert_minilm.py: 6 layers, 384 wide
            return flops(384, int(name[len("minilm_s"):].split("_")[0]), 6)
        return None

    print("| model | MFLOP | CPU µs | NPU µs | CPU mJ/call | NPU mJ/call | NPU / CPU energy | CPU GFLOP/J | NPU GFLOP/J |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = [(work_of(n) or 0, n, m) for n, m in by.items() if "cpu" in m and "npu" in m]
    for fl, name, m in sorted(rows):
        c, n = m["cpu"], m["npu"]
        ec, en = c["energy_per_call_mj"], n["energy_per_call_mj"]

        def eff(e):
            return f"{fl / (e * 1e-3) / 1e9:.0f}" if fl and e > 0 else "-"

        ratio = f"{en / ec:.2f}" if ec > 0 else "-"
        print(f"| {name} | {fl / 1e6:.0f} | {c['mean_latency_us']:.0f} | {n['mean_latency_us']:.0f} | "
              f"{ec:.3f} | {en:.3f} | {ratio} | {eff(ec)} | {eff(en)} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--report", help="powerbench -json result to join with sweep.json")
    a = ap.parse_args()
    if a.report:
        report(a.out, a.report)
    else:
        generate(a.out)
