"""Pad the batch dimension of a Core ML ML Program so that it runs on the Neural Engine.

Core ML places small models on the CPU even in fp16. Growing the batch (first) dimension of
every input and output, and filling the extra rows with zeros at run time, does not change the
results of models that process rows independently (most inference models). This tool searches
for the smallest padded batch at which Core ML's compute plan puts every operation on the
Neural Engine, and writes the padded model with metadata that hwnpu reads:

    hwnpu.logical_batch   the original batch size users see
    hwnpu.padded_batch    the batch size the model was rebuilt with

hwnpu then copies the caller's rows into a zero padded buffer and copies the results back.

Usage (macOS with coremltools >= 8):
    python ane_pad.py model.mlpackage padded.mlpackage [--candidates 16,32,64,...] [--verify]

Limits: the model must be an ML Program whose inputs and outputs all have the batch as their
first dimension with the same size, and whose operations do not mix rows (no reductions or
reshapes across the batch). --verify compares the padded model with the original on random
inputs; use it.
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import coremltools as ct
from coremltools.converters.mil.frontend.milproto.load import load as milproto_load
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan

DEFAULT_CANDIDATES = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


def batch_of(model):
    spec = model.get_spec()
    if spec.WhichOneof("Type") != "mlProgram":
        sys.exit("only ML Program models can be padded (convert with convert_to='mlprogram')")
    sizes = set()
    for f in list(spec.description.input) + list(spec.description.output):
        shape = list(f.type.multiArrayType.shape)
        if not shape:
            sys.exit(f"{f.name}: only fixed-shape multi arrays are supported")
        sizes.add(shape[0])
    if len(sizes) != 1:
        sys.exit(f"inputs and outputs disagree on the first dimension: {sorted(sizes)}")
    return sizes.pop()


def rebatch(model, batch, precision):
    spec = model.get_spec()
    prog = milproto_load(spec, spec.specificationVersion, model.weights_dir)
    fn = prog.functions["main"]
    for ph in fn.placeholder_inputs.values():
        shape = (batch,) + tuple(ph.sym_shape[1:])
        ph.sym_shape = shape
        ph.outputs[0]._sym_type = types.tensor(ph.outputs[0].dtype, shape)
    for op in fn.operations:
        op.type_value_inference(overwrite_output=True)
    for out in fn.outputs:
        if out.shape[0] != batch:
            raise ValueError(f"output {out.name} does not follow the batch ({out.shape})")
    target = ct.target.macOS13 if spec.specificationVersion <= 7 else None
    new = ct.convert(prog, convert_to="mlprogram", compute_precision=precision, minimum_deployment_target=target)
    new.user_defined_metadata.update(model.user_defined_metadata)
    return new


def placement(model):
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "m.mlpackage")
        model.save(path)
        plan = MLComputePlan.load_from_path(path=ct.models.utils.compile_model(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        on_ne, off = 0, []
        for fn in plan.model_structure.program.functions.values():
            for op in fn.block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if usage is None:
                    continue
                if type(usage.preferred_compute_device).__name__ == "MLNeuralEngineComputeDevice":
                    on_ne += 1
                else:
                    off.append(op.operator_name)
        return on_ne, off
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def random_inputs(model, batch, rng):
    feeds = {}
    for f in model.get_spec().description.input:
        shape = [batch] + list(f.type.multiArrayType.shape)[1:]
        dt = {65552: np.float16, 65568: np.float32, 65600: np.float64, 131104: np.int32}.get(f.type.multiArrayType.dataType, np.float32)
        feeds[f.name] = (rng.standard_normal(shape) if np.issubdtype(dt, np.floating) else rng.integers(0, 8, shape)).astype(dt)
    return feeds


def verify(original, padded, logical, batch):
    rng = np.random.default_rng(0)
    x = random_inputs(original, logical, rng)
    ref = original.predict(x)
    xp = {k: np.concatenate([v, np.zeros((batch - logical,) + v.shape[1:], v.dtype)]) for k, v in x.items()}
    got = padded.predict(xp)
    worst = 0.0
    for k, v in ref.items():
        d = np.abs(np.asarray(got[k])[:logical].astype(np.float64) - np.asarray(v).astype(np.float64))
        worst = max(worst, float(d.max()))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--candidates", default=",".join(map(str, DEFAULT_CANDIDATES)))
    ap.add_argument("--verify", action="store_true", help="compare outputs with the original model")
    args = ap.parse_args()

    # coremltools rewrites Manifest.json of a package it opens; work on a copy so the
    # caller's model is left untouched.
    work = tempfile.mkdtemp()
    src = os.path.join(work, os.path.basename(os.path.normpath(args.src)))
    (shutil.copytree if os.path.isdir(args.src) else shutil.copy)(args.src, src)
    model = ct.models.MLModel(src, compute_units=ct.ComputeUnit.CPU_AND_NE)
    logical = batch_of(model)
    on_ne, off = placement(model)
    print(f"original batch {logical}: {on_ne} op(s) on the Neural Engine, elsewhere: {off or 'none'}")
    if not off:
        print("already runs entirely on the Neural Engine; nothing to do")
        model.save(args.dst)
        return
    for batch in sorted(int(c) for c in args.candidates.split(",")):
        if batch <= logical:
            continue
        try:
            padded = rebatch(model, batch, ct.precision.FLOAT16)
        except Exception as e:  # the model mixes rows or cannot be rebuilt
            sys.exit(f"cannot rebuild the model with batch {batch}: {e}")
        on_ne, off = placement(padded)
        print(f"batch {batch}: {on_ne} op(s) on the Neural Engine, elsewhere: {off or 'none'}")
        if off:
            continue
        padded.user_defined_metadata["hwnpu.logical_batch"] = str(logical)
        padded.user_defined_metadata["hwnpu.padded_batch"] = str(batch)
        if args.verify:
            print(f"max abs difference on the original rows: {verify(model, padded, logical, batch):.6g}")
        padded.save(args.dst)
        print(f"wrote {args.dst} (batch {logical} padded to {batch})")
        return
    sys.exit("no candidate batch put every operation on the Neural Engine")


if __name__ == "__main__":
    main()
