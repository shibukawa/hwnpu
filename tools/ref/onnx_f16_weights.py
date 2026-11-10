"""How far an ONNX graph's values move when only its weights are rounded to fp16 (computed in fp32
by ONNX Runtime): the least a program in fp16 can differ from the reference. For
internal/vision/detect's tolerances.

    python onnx_f16_weights.py MODEL.onnx REF OUTPUTS NAMES

REF is a prefix of ref_detect_ocr.py's (REF.x.bin the input); OUTPUTS are the graph's values
(1097,1124), NAMES the reference's files of them (scores,boxes). Needs numpy, onnx, onnxruntime.
"""
import sys, numpy as np, onnx, onnx.numpy_helper as nh, onnxruntime as ort
src, ref, outs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
m = onnx.load(src)
for i in m.graph.initializer:
    a = nh.to_array(i)
    if a.dtype == np.float32:
        i.CopyFrom(nh.from_array(a.astype(np.float16).astype(np.float32), i.name))
for o in outs:
    if o not in {x.name for x in m.graph.output}:
        m.graph.output.append(onnx.ValueInfoProto(name=o))
s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
x = np.fromfile(ref + ".x.bin", np.float32).reshape([d.dim_value or 1 for d in m.graph.input[0].type.tensor_type.shape.dim])
res = dict(zip([o.name for o in s.get_outputs()], s.run(None, {m.graph.input[0].name: x})))
for o, name in zip(outs, sys.argv[4].split(",")):
    want = np.fromfile(f"{ref}.{name}.bin", np.float32)
    got = res[o].ravel()
    print(o, name, "max abs diff", np.abs(got - want).max())
    if name == "boxes":
        sc = np.fromfile(f"{ref}.scores.bin", np.float32).reshape(-1, 80).max(1)
        d = np.abs(got - want).reshape(-1, 4).max(1)
        print("  where score >= 0.1:", d[sc >= 0.1].max())
