"""Reference outputs for hwnpu-image detect (RTMDet) and ocr (PP-OCRv6) with ONNX Runtime, for
internal/vision/detect's TestRTMDet and internal/vision/ocr's TestPrograms.

    python ref_detect_ocr.py rtmdet rtmdet-s-coco.onnx football-match.jpg ref/rtmdet-football
    python ref_detect_ocr.py ocrdet PP-OCRv6_small_det_onnx/inference.onnx scan.png 960 960 ref/ocrdet
    python ref_detect_ocr.py ocrrec PP-OCRv6_small_rec_onnx/inference.onnx line.png 640 ref/ocrrec [inference.yml]

rtmdet MODEL IMAGE OUT: the letterboxed input (640, pad 114, mmdet's mean and std), the head's
scores (1, 8400, 80) and boxes (1, 8400, 4), and the graph's dets and labels.
ocrdet MODEL IMAGE W H OUT: the input (BGR, ImageNet's mean and std, the picture scaled to fit
W x H at the top left, zeros past it) and the probability map.
ocrrec MODEL IMAGE W OUT: the whole picture as one line (BGR, (x/255-0.5)/0.5, height 48, width
by its ratio, zeros past it up to W) and the per-step probabilities (with the dictionary, the
text it reads).

Writes OUT.NAME.bin (float32, little endian) and OUT.json. Needs numpy, onnx, onnxruntime, pillow
(and pyyaml for the text).
"""
import json, sys
import numpy as np
import onnx, onnxruntime as ort
from PIL import Image

def save(out, name, a):
    np.ascontiguousarray(a, dtype=np.float32).tofile(f"{out}.{name}.bin")

def session(path, extra=()):
    m = onnx.load(path)
    have = {o.name for o in m.graph.output}
    for e in extra:
        if e not in have:
            m.graph.output.append(onnx.ValueInfoProto(name=e))
    return ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])

def rtmdet(model, image, out):
    img = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32)
    h, w = img.shape[:2]
    s = min(640 / w, 640 / h)
    nw, nh = int(w * s + 0.5), int(h * s + 0.5)
    r = np.asarray(Image.open(image).convert("RGB").resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    pad = np.full((640, 640, 3), 114, np.float32)
    pad[:nh, :nw] = r
    mean, std = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])
    x = ((pad - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    sess = session(model, ["1097", "1124"])
    names = [o.name for o in sess.get_outputs()]
    res = dict(zip(names, sess.run(None, {"input": x})))
    save(out, "x", x)
    for k in ["1097", "1124", "dets", "labels"]:
        save(out, {"1097": "scores", "1124": "boxes"}.get(k, k), res[k])
    meta = {"scale": s, "size": [nw, nh], "shapes": {k: list(v.shape) for k, v in res.items()}}
    json.dump(meta, open(out + ".json", "w"))
    d, l = res["dets"][0], res["labels"][0]
    for i in range(len(d)):
        if d[i, 4] >= 0.3:
            print("label %d score %.3f box %s" % (l[i], d[i, 4], (d[i, :4] / s).round(1)))

def ocrdet(model, image, W, H, out):
    pil = Image.open(image).convert("RGB")
    w, h = pil.size
    s = min(W / w, H / h, 1.0)
    nw, nh = max(1, int(w * s + 0.5)), max(1, int(h * s + 0.5))
    r = np.asarray(pil.resize((nw, nh), Image.BILINEAR), dtype=np.float32)[:, :, ::-1] / 255  # BGR
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    x = np.zeros((H, W, 3), np.float32)
    x[:nh, :nw] = (r - mean) / std
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    sess = session(model)
    p = sess.run(None, {"x": x})[0]
    save(out, "x", x)
    save(out, "prob", p)
    json.dump({"scale": s, "size": [nw, nh], "shape": list(p.shape)}, open(out + ".json", "w"))
    print("prob", p.shape, "max", p.max(), "> 0.2:", (p > 0.2).mean())

def ocrrec(model, image, W, out, dict_yml=None):
    pil = Image.open(image).convert("RGB")
    w, h = pil.size
    nw = min(W, int(np.ceil(48 * w / h)))
    r = np.asarray(pil.resize((nw, 48), Image.BILINEAR), dtype=np.float32)[:, :, ::-1] / 255
    x = np.zeros((48, W, 3), np.float32)
    x[:, :nw] = (r - 0.5) / 0.5
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    sess = session(model)
    p = sess.run(None, {"x": x})[0]
    save(out, "x", x)
    save(out, "prob", p)
    json.dump({"shape": list(p.shape)}, open(out + ".json", "w"))
    if dict_yml:
        import yaml
        chars = ["blank"] + yaml.safe_load(open(dict_yml))["PostProcess"]["character_dict"] + [" "]
        idx = p[0].argmax(-1)
        txt, prev = [], 0
        for i in idx:
            if i != prev and i != 0:
                txt.append(chars[i])
            prev = i
        print("text:", "".join(txt))

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rtmdet":
        rtmdet(*sys.argv[2:5])
    elif cmd == "ocrdet":
        ocrdet(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
    elif cmd == "ocrrec":
        ocrrec(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], *(sys.argv[6:7]))
