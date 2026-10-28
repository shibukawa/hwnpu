"""Generate the hwnpu model fixtures.

Usage (needs numpy; the Core ML part also needs coremltools >= 8):
    python gen_models.py [--only coreml|openvino]

All "linear_relu" models compute y = relu(x @ W^T + b) with x of shape (64, 512) and the same
W (512 x 512) and b (512), so results and latency can be compared across vendors:

    coreml/linear_relu_fp16.mlpackage  fp16 I/O and compute. Core ML's compute plan places it on
                                       the Neural Engine.
    coreml/linear_relu_fp32.mlpackage  fp32. The Neural Engine cannot run fp32, so the plan places it
                                       on the CPU (strict mode must reject it).
    openvino/linear_relu.xml / .bin    OpenVINO IR v11, fp32 I/O (Intel NPUs compute in fp16).
    linear_relu_y.f32                  reference output, float32 little endian, shape (64, 512)

The input is not stored; consumers compute it as x[i] = sin(i * 0.01) for the flat index i.
Small models are placed on the CPU by Core ML even in fp16, hence the size.

openvino/affine_relu.{xml,bin} is hand written (y = relu(2 * x - 1), x of shape (1, 8)).

The "mlp_small" models are a 3 layer MLP (128 wide, relu) for one row, x of shape (1, 128):

    coreml/mlp_small.mlpackage      fp16; Core ML places it on the CPU because it is small
    coreml/mlp_small_ane.mlpackage  the same model padded to batch 64 by tools/coreml/ane_pad.py,
                                    which Core ML places on the Neural Engine
    mlp_small_y.f32                 reference output for x[i] = sin(i * 0.01), float32
"""
import argparse
import os
import subprocess
import sys

import numpy as np

B, IN, OUT = 64, 512, 512
HERE = os.path.dirname(os.path.abspath(__file__))


def weights():
    rng = np.random.default_rng(42)
    w = (rng.standard_normal((OUT, IN)) * 0.05).astype(np.float32)
    b = (rng.standard_normal(OUT) * 0.1).astype(np.float32)
    return w, b


def gen_coreml(w, b):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    def build(dtype):
        t = types.fp16 if dtype == np.float16 else types.fp32
        precision = ct.precision.FLOAT16 if dtype == np.float16 else ct.precision.FLOAT32

        @mb.program(input_specs=[mb.TensorSpec(shape=(B, IN), dtype=t)], opset_version=ct.target.macOS13)
        def prog(x):
            y = mb.linear(x=x, weight=w.astype(dtype), bias=b.astype(dtype), name="linear")
            return mb.relu(x=y, name="y")

        return ct.convert(prog, convert_to="mlprogram", compute_precision=precision,
                          minimum_deployment_target=ct.target.macOS13)

    os.makedirs(os.path.join(HERE, "coreml"), exist_ok=True)
    build(np.float16).save(os.path.join(HERE, "coreml", "linear_relu_fp16.mlpackage"))
    build(np.float32).save(os.path.join(HERE, "coreml", "linear_relu_fp32.mlpackage"))


IR = """<?xml version="1.0"?>
<net name="linear_relu" version="11">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data shape="{B},{IN}" element_type="f32" />
			<output>
				<port id="0" precision="FP32" names="x">
					<dim>{B}</dim>
					<dim>{IN}</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="weight" type="Const" version="opset1">
			<data element_type="f32" shape="{OUT},{IN}" offset="0" size="{WSIZE}" />
			<output>
				<port id="0" precision="FP32">
					<dim>{OUT}</dim>
					<dim>{IN}</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="matmul" type="MatMul" version="opset1">
			<data transpose_a="false" transpose_b="true" />
			<input>
				<port id="0" precision="FP32">
					<dim>{B}</dim>
					<dim>{IN}</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>{OUT}</dim>
					<dim>{IN}</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>{B}</dim>
					<dim>{OUT}</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="bias" type="Const" version="opset1">
			<data element_type="f32" shape="1,{OUT}" offset="{WSIZE}" size="{BSIZE}" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>{OUT}</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="add" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="FP32">
					<dim>{B}</dim>
					<dim>{OUT}</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>{OUT}</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>{B}</dim>
					<dim>{OUT}</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="y" type="ReLU" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>{B}</dim>
					<dim>{OUT}</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="y">
					<dim>{B}</dim>
					<dim>{OUT}</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="y/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>{B}</dim>
					<dim>{OUT}</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
		<edge from-layer="2" from-port="2" to-layer="4" to-port="0" />
		<edge from-layer="3" from-port="0" to-layer="4" to-port="1" />
		<edge from-layer="4" from-port="2" to-layer="5" to-port="0" />
		<edge from-layer="5" from-port="1" to-layer="6" to-port="0" />
	</edges>
	<rt_info />
</net>
"""


def gen_openvino(w, b):
    os.makedirs(os.path.join(HERE, "openvino"), exist_ok=True)
    wb = w.astype("<f4").tobytes()
    bb = b.astype("<f4").tobytes()
    xml = IR.format(B=B, IN=IN, OUT=OUT, WSIZE=len(wb), BSIZE=len(bb))
    with open(os.path.join(HERE, "openvino", "linear_relu.xml"), "w") as f:
        f.write(xml)
    with open(os.path.join(HERE, "openvino", "linear_relu.bin"), "wb") as f:
        f.write(wb + bb)


def gen_mlp():
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    rng = np.random.default_rng(1)
    ws = [(rng.standard_normal((128, 128)) * 0.08).astype(np.float16) for _ in range(3)]

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 128), dtype=types.fp16)], opset_version=ct.target.macOS13)
    def prog(x):
        for w in ws:
            x = mb.relu(x=mb.linear(x=x, weight=w))
        return x

    src = os.path.join(HERE, "coreml", "mlp_small.mlpackage")
    ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.macOS13).save(src)
    tool = os.path.join(HERE, "..", "..", "tools", "coreml", "ane_pad.py")
    subprocess.run([sys.executable, tool, src, os.path.join(HERE, "coreml", "mlp_small_ane.mlpackage"), "--verify"], check=True)
    x = np.sin(np.arange(128, dtype=np.float64) * 0.01).astype(np.float32).reshape(1, 128)
    for w in ws:
        x = np.maximum(x @ w.astype(np.float32).T, 0)
    x.astype("<f4").tofile(os.path.join(HERE, "mlp_small_y.f32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["coreml", "openvino", "mlp"])
    args = ap.parse_args()
    if args.only == "mlp":
        gen_mlp()
        return
    w, b = weights()
    x = np.sin(np.arange(B * IN, dtype=np.float64) * 0.01).astype(np.float32).reshape(B, IN)
    np.maximum(x @ w.T + b, 0).astype("<f4").tofile(os.path.join(HERE, "linear_relu_y.f32"))
    if args.only in (None, "coreml"):
        gen_coreml(w, b)
        gen_mlp()
    if args.only in (None, "openvino"):
        gen_openvino(w, b)


if __name__ == "__main__":
    main()
