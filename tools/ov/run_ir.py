"""Run an OpenVINO IR with OpenVINO's CPU plugin, for checking hwnpu's IR (internal/ovlower)
on a machine without Intel's NPU, macOS included.

    python run_ir.py MODEL.xml OUT_DIR NAME=IN.f32 [NAME=IN.f32 ...] [--precision f32]

Each input is raw little endian float32, as many values as the model's input of that name
holds; it is converted to the input's element type. Every output is written to OUT_DIR/NAME.f32
as float32. --precision sets the CPU plugin's inference precision: f32 (the default) checks the
graph itself, f16 is nearer to what the NPU computes. internal/ovcpu calls this for the Go
tests (HWNPU_OV_PYTHON names a Python with the openvino package).
"""
import argparse
import os

import numpy as np
import openvino as ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--precision", default="f32", choices=["f32", "f16"])
    args = ap.parse_args()

    core = ov.Core()
    model = core.read_model(args.model)
    compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": args.precision})
    feeds = {}
    for spec in args.inputs:
        name, path = spec.split("=", 1)
        port = compiled.input(name)
        shape = list(port.get_shape())
        data = np.fromfile(path, dtype="<f4").reshape(shape)
        feeds[name] = data.astype(port.get_element_type().to_dtype())
    results = compiled(feeds)
    os.makedirs(args.out_dir, exist_ok=True)
    for port in compiled.outputs:
        out = np.asarray(results[port], dtype="<f4")
        out.tofile(os.path.join(args.out_dir, port.get_any_name() + ".f32"))


if __name__ == "__main__":
    main()
