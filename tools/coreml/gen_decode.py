"""Generate decode shaped models: one token through Llama style layers with large weights.

When an LLM generates text, every token runs all layers on a single position, so each token
reads every weight once and the speed depends on memory bandwidth more than on compute. These
models reproduce that shape with random weights:

    x (1, 1, d) fp16 -> layers x [norm, q/k/v/o projections, norm, SwiGLU MLP] -> (1, 1, d)

The sizes follow Llama 3.2 1B (d 2048, MLP 8192, grouped query attention with 512 wide k and v,
about 61 M parameters per layer, so 16 layers are 0.97 B parameters without the embedding
table). Attention over a KV cache is left out: with one query it is a small read next to the
weights (about 2 MB per 1000 tokens of context per layer pair) and it needs state, which hwnpu
does not support yet. q, k and v are still computed and mixed so that every weight is read.

    python gen_decode.py OUT_DIR [--layers 4,16] [--formats fp16,int8,lut4,int4]

formats:
    fp16  fp16 weights (2 bytes per parameter)
    int8  linear symmetric int8, per output channel (1 byte)
    lut4  4 bit palettization, a 16 entry table per group of 16 output channels (0.5 byte)
    int4  linear symmetric int4, per block of 32 inputs (0.5 byte, macOS 15 and later)

Writes dec_l{layers}_{format}.mlpackage and decode.json (parameters and bytes of weights per
token). Measure with go run ./cmd/powerbench -one -gpu -model 'OUT_DIR/*.mlpackage', then
python gen_decode.py OUT_DIR --report powerbench.json.
Needs coremltools >= 8 and numpy.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

D, FFN, KV = 2048, 8192, 512
BYTES = {"fp16": 2.0, "int8": 1.0, "lut4": 0.5, "int4": 0.5}


def layer_params():
    return 2 * D * D + 2 * D * KV + 3 * D * FFN


def build(layers, seed):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    rng = np.random.default_rng(seed)

    def w(o, i):
        return (rng.standard_normal((o, i), dtype=np.float32) / np.sqrt(i)).astype(np.float16)

    def norm(x):
        return mb.layer_norm(x=x, axes=[-1], gamma=np.ones(D, np.float16), beta=np.zeros(D, np.float16),
                             epsilon=np.float16(1e-5))

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, D), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def prog(x):
        for _ in range(layers):
            h = norm(x)
            q = mb.linear(x=h, weight=w(D, D))
            k = mb.linear(x=h, weight=w(KV, D))
            v = mb.linear(x=h, weight=w(KV, D))
            # Stand in for attention with one query: every head reads its shared k/v group.
            kv = mb.tile(x=mb.add(x=k, y=v), reps=[1, 1, D // KV])
            o = mb.linear(x=mb.mul(x=q, y=kv), weight=w(D, D))
            x = mb.add(x=x, y=o)
            h = norm(x)
            g = mb.linear(x=h, weight=w(FFN, D))
            u = mb.linear(x=h, weight=w(FFN, D))
            f = mb.mul(x=mb.silu(x=g), y=u)
            x = mb.add(x=x, y=mb.linear(x=f, weight=w(D, FFN)))
        return mb.identity(x=norm(x), name="y")

    return ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.macOS15)


def compress(model, fmt):
    import coremltools.optimize.coreml as cto

    if fmt == "fp16":
        return model
    if fmt == "int8":
        op = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
        return cto.linear_quantize_weights(model, cto.OptimizationConfig(global_config=op))
    if fmt == "int4":
        op = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32)
        return cto.linear_quantize_weights(model, cto.OptimizationConfig(global_config=op))
    if fmt == "lut4":
        op = cto.OpPalettizerConfig(mode="uniform", nbits=4, granularity="per_grouped_channel", group_size=16)
        return cto.palettize_weights(model, cto.OptimizationConfig(global_config=op))
    raise ValueError(fmt)


def generate(out, layer_list, formats):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ane_pad import placement

    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "decode.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    for layers in layer_list:
        base = build(layers, seed=layers)
        for fmt in formats:
            name = f"dec_l{layers}_{fmt}"
            model = compress(base, fmt)
            on_ne, off = placement(model)
            dst = os.path.join(out, name + ".mlpackage")
            shutil.rmtree(dst, ignore_errors=True)
            model.save(dst)
            size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(dst) for f in fs)
            params = layers * layer_params()
            row = {"name": name, "layers": layers, "format": fmt, "params": params,
                   "weight_bytes": params * BYTES[fmt], "package_bytes": size,
                   "flops": 2 * params, "ops_on_ne": on_ne, "ops_elsewhere": len(off)}
            rows = [r for r in rows if r["name"] != name] + [row]
            print(f"{name}: {params / 1e9:.2f} B params, package {size / 2**20:.0f} MiB, "
                  f"{on_ne} op(s) on the Neural Engine, elsewhere: {sorted(set(off)) or 'none'}", flush=True)
            with open(path, "w") as f:
                json.dump(rows, f, indent=2)


def report(out, bench):
    """Print tokens per second, energy per token and weight bandwidth for every variant."""
    with open(os.path.join(out, "decode.json")) as f:
        meta = {r["name"]: r for r in json.load(f)}
    with open(bench) as f:
        work = json.load(f)["workloads"]
    print("| model | params | weights | unit | tokens/s | ms/token | net W | mJ/token | weight GB/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    order = {"cpu": 0, "gpu": 1, "npu": 2}
    rows = []
    for w in work:
        base = w["name"].removesuffix(" [CPU only]").removesuffix(" [GPU]")
        units = w.get("units") or ("cpu" if w.get("cpu_only") else "npu")
        if base in meta and w.get("rounds"):
            rows.append((meta[base]["params"], meta[base]["weight_bytes"], base, order[units], units, w["mean"], w.get("placement", "")))
    for params, wb, base, _, units, m, placement in sorted(rows):
        tps = m["calls_per_s"]
        print(f"| {base} | {params / 1e9:.2f} B | {wb / 2**30:.2f} GiB | {units.upper()} ({placement}) | {tps:.1f} | "
              f"{1000 / tps:.1f} | {m['net']['combined_mw'] / 1000:.2f} | {m['energy_per_call_mj']:.1f} | {wb * tps / 1e9:.0f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--layers", default="4,16")
    ap.add_argument("--formats", default="fp16,int8,lut4,int4")
    ap.add_argument("--report", help="powerbench -json result to summarize")
    a = ap.parse_args()
    if a.report:
        report(a.out, a.report)
    else:
        generate(a.out, [int(x) for x in a.layers.split(",")], a.formats.split(","))
