"""PaddleOCR's pipeline (PP-OCRv6 det + rec, ONNX Runtime), as PaddleOCR 3 writes it: DB
post-processing with cv2 and pyclipper, get_rotate_crop_image, the recognizer at the width of each
batch's widest ratio (6 lines a batch, sorted by ratio), CTC decoding. Prints a JSON line per
picture: the lines' text and score.

    python ref_ppocr.py DET_DIR REC_DIR [--fit WxH | --native] PICTURE... > ref.jsonl

DET_DIR and REC_DIR hold each repository's inference.onnx and inference.yml. --fit scales the
picture to fit the detector's size, at the top left (hwnpu-image ocr's way, 960x960 by default);
--native keeps it at its size in multiples of 32 (PaddleOCR 3's default). Compare with
hwnpu-image ocr's JSON lines by compare_ocr.py. Needs numpy, onnxruntime, opencv-python, pyclipper,
pyyaml.
"""
import json, math, sys
import cv2, numpy as np, onnxruntime as ort, pyclipper, yaml

def args():
    a = sys.argv[1:]
    det, rec = a[0], a[1]
    mode, pics, i = ("fit", 960, 960), [], 2
    while i < len(a):
        if a[i] == "--fit":
            w, h = a[i + 1].split("x"); mode = ("fit", int(w), int(h)); i += 2
        elif a[i] == "--native":
            mode = ("native",); i += 1
        else:
            pics.append(a[i]); i += 1
    return det, rec, mode, pics

def det_input(img, mode):
    h, w = img.shape[:2]
    if mode[0] == "fit":
        W, H = mode[1], mode[2]
        s = min(1.0, W / w, H / h)
        nw, nh = max(1, int(w * s + 0.5)), max(1, int(h * s + 0.5))
    else:  # PaddleOCR 3's default: no scaling but to multiples of 32 (limit_side_len 64, min)
        W, H = max(32, int(round(w / 32)) * 32), max(32, int(round(h / 32)) * 32)
        nw, nh = W, H
    r = cv2.resize(img, (nw, nh)).astype(np.float32) / 255
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    x = np.zeros((H, W, 3), np.float32)
    x[:nh, :nw] = (r - mean) / std
    return x.transpose(2, 0, 1)[None].astype(np.float32), nw, nh, w / nw, h / nh

def mini_box(contour):
    rect = cv2.minAreaRect(contour)
    pts = sorted(list(cv2.boxPoints(rect)), key=lambda p: p[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return np.array([pts[i1], pts[i2], pts[i3], pts[i4]]), min(rect[1])

def box_score(pred, box):
    h, w = pred.shape
    b = box.copy()
    xmin, xmax = np.clip(np.floor(b[:, 0].min()).astype(int), 0, w - 1), np.clip(np.ceil(b[:, 0].max()).astype(int), 0, w - 1)
    ymin, ymax = np.clip(np.floor(b[:, 1].min()).astype(int), 0, h - 1), np.clip(np.ceil(b[:, 1].max()).astype(int), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    b[:, 0] -= xmin; b[:, 1] -= ymin
    cv2.fillPoly(mask, b.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

def unclip(box, ratio):
    poly = np.array(box)
    area = abs(cv2.contourArea(poly.astype(np.float32)))
    length = cv2.arcLength(poly.astype(np.float32), True)
    d = area * ratio / length
    off = pyclipper.PyclipperOffset()
    off.AddPath(poly.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return np.array(off.Execute(d))

def boxes(pred, nw, nh, sx, sy, thresh, box_thresh, ratio):
    pred = pred[:nh, :nw]
    bitmap = (pred > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours[:3000]:
        pts, sside = mini_box(c)
        if sside < 3:
            continue
        if box_score(pred, pts) < box_thresh:
            continue
        u = unclip(pts, ratio)
        if len(u) != 1:
            continue
        b, sside = mini_box(u.reshape(-1, 1, 2))
        if sside < 5:
            continue
        b[:, 0] = np.clip(np.round(b[:, 0] * sx), 0, nw * sx)
        b[:, 1] = np.clip(np.round(b[:, 1] * sy), 0, nh * sy)
        out.append(b.astype(np.float32))
    out = sorted(out, key=lambda x: (x[0][1], x[0][0]))
    for i in range(len(out) - 1):
        for j in range(i, -1, -1):
            if abs(out[j + 1][0][1] - out[j][0][1]) < 10 and out[j + 1][0][0] < out[j][0][0]:
                out[j], out[j + 1] = out[j + 1], out[j]
            else:
                break
    return out

def crop(img, pts):
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    M = cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    d = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if d.shape[0] / max(1, d.shape[1]) >= 1.5:
        d = np.rot90(d)
    return d

def recognize(sess, chars, crops, batch=6):
    ratios = [c.shape[1] / c.shape[0] for c in crops]
    order = np.argsort(ratios)
    res = [None] * len(crops)
    for b in range(0, len(crops), batch):
        idx = order[b:b + batch]
        mx = max(320 / 48, max(ratios[i] for i in idx))
        W = int(48 * mx)
        xs = []
        for i in idx:
            c = crops[i]
            rw = min(W, int(math.ceil(48 * c.shape[1] / c.shape[0])))
            r = cv2.resize(c, (rw, 48)).astype(np.float32) / 255
            x = np.zeros((48, W, 3), np.float32)
            x[:, :rw] = (r - 0.5) / 0.5
            xs.append(x.transpose(2, 0, 1))
        p = sess.run(None, {"x": np.stack(xs).astype(np.float32)})[0]
        for k, i in enumerate(idx):
            a = p[k].argmax(-1); pr = p[k].max(-1)
            keep = np.ones(len(a), bool); keep[1:] = a[1:] != a[:-1]; keep &= a != 0
            txt = "".join(chars[j] for j in a[keep])
            res[i] = (txt, float(pr[keep].mean()) if keep.any() else 0.0)
    return res

def main():
    det, rec, mode, pics = args()
    dcfg = yaml.safe_load(open(det + "/inference.yml"))["PostProcess"]
    chars = [""] + yaml.safe_load(open(rec + "/inference.yml"))["PostProcess"]["character_dict"] + [" "]
    ds = ort.InferenceSession(det + "/inference.onnx", providers=["CPUExecutionProvider"])
    rs = ort.InferenceSession(rec + "/inference.onnx", providers=["CPUExecutionProvider"])
    for p in pics:
        img = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR, as PaddleOCR
        x, nw, nh, sx, sy = det_input(img, mode)
        pred = ds.run(None, {"x": x})[0][0, 0]
        bs = boxes(pred, nw, nh, sx, sy, dcfg["thresh"], dcfg["box_thresh"], dcfg["unclip_ratio"])
        out = recognize(rs, chars, [crop(img, b) for b in bs])
        print(json.dumps({"file": p, "lines": [{"text": t, "score": s, "box": b.tolist()} for (t, s), b in zip(out, bs)]}, ensure_ascii=False))

main()
