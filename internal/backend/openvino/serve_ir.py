"""Serve hwnpu's openvino backend (internal/backend/openvino): compile OpenVINO IR with OpenVINO
and run it on buffers hwnpu shares with this process as files mapped in memory.

    python -I serve_ir.py [--device CPU] [--precision f32]

One JSON request per line on stdin, one JSON reply per line on stdout:

    {"op": "compile", "xml": PATH}       -> {"id": N, "args": [{"name", "output", "dims", "dtype"}]}
    {"op": "run", "id": N, "binds": [[PATH, SIZE], ...]}  (one per argument, in order) -> {}
    {"op": "unmap", "path": PATH}         -> {}
    {"op": "free", "id": N}               -> {}

A failure replies {"error": TEXT}. An argument's name is the one Level Zero would report: a
Parameter's name for an input, the name of the layer that makes a Result for an output, so that the IR hwnpu-dev convert writes binds by the same names as on
Intel's NPU. The device is CPU (to check the IR on any machine) or NPU (OpenVINO's own NPU
plugin, to compare against hwnpu's Level Zero backend on an Intel machine).
"""
import argparse
import json
import sys

import numpy as np

# no telemetry: openvino_telemetry sends usage events from a child process (multiprocessing),
# which outlives this one and fails once its script is removed; openvino falls back to its stub
sys.modules["openvino_telemetry"] = None
import openvino as ov  # noqa: E402

DTYPES = {
    ov.Type.f16: "f16", ov.Type.f32: "f32", ov.Type.i32: "i32", ov.Type.i64: "i64",
    ov.Type.u8: "u8", ov.Type.i8: "i8", ov.Type.bf16: "bf16",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--precision", default="f32")
    args = ap.parse_args()
    core = ov.Core()
    config = {}
    if args.device == "CPU":
        if args.precision:
            config["INFERENCE_PRECISION_HINT"] = args.precision
        # x86 CPUs quantize the activations of compressed (int4/int8) weights to int8 by
        # default, which costs about 1% of accuracy; the check wants the graph's own numbers
        config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = "0"
    graphs = {}
    maps = {}
    next_id = 1

    def mapped(path, size):
        m = maps.get(path)
        if m is None or m.size < size:
            m = np.memmap(path, dtype=np.uint8, mode="r+", shape=(size,))
            maps[path] = m
        return m

    def reply(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req["op"]
            if op == "compile":
                model = core.read_model(req["xml"])
                compiled = core.compile_model(model, args.device, config)
                ports = []
                desc = []
                for p, m in zip(compiled.inputs, model.inputs):
                    ports.append((p, False))
                    desc.append(arg(p, m.get_node().get_friendly_name(), False))
                for p, m in zip(compiled.outputs, model.outputs):
                    ports.append((p, True))
                    desc.append(arg(p, m.get_node().input_value(0).get_node().get_friendly_name(), True))
                gid = next_id
                next_id += 1
                graphs[gid] = (compiled, compiled.create_infer_request(), ports)
                reply({"id": gid, "args": desc})
            elif op == "run":
                compiled, request, ports = graphs[req["id"]]
                outs = []
                for (port, output), (path, size) in zip(ports, req["binds"]):
                    shape = list(port.get_shape())
                    dt = port.get_element_type().to_dtype()
                    n = int(np.prod(shape)) * np.dtype(dt).itemsize
                    view = mapped(path, size)[:n].view(dt).reshape(shape)
                    if output:
                        outs.append((port, view))
                    else:
                        request.set_tensor(port, ov.Tensor(view, shared_memory=True))
                request.infer()
                for port, view in outs:
                    np.copyto(view, request.get_tensor(port).data)
                reply({})
            elif op == "unmap":
                maps.pop(req["path"], None)
                reply({})
            elif op == "free":
                graphs.pop(req["id"], None)
                reply({})
            else:
                reply({"error": "unknown op " + op})
        except Exception as e:  # reported to Go, which fails the call
            reply({"error": f"{type(e).__name__}: {e}"})


def arg(port, name, output):
    return {
        "name": name,
        "output": output,
        "dims": [int(d) for d in port.get_shape()],
        "dtype": DTYPES.get(port.get_element_type(), str(port.get_element_type())),
    }


if __name__ == "__main__":
    main()
