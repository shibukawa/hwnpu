"""Compares the lines of two OCR outputs in JSON lines (ref_ppocr.py's, hwnpu-image ocr's): for
each picture, how many lines of score 0.5 or more are the same, and the character error rate of
the whole text. A third argument lists the lines that differ.

    python compare_ocr.py ref.jsonl hwnpu.jsonl [v]
"""
import json, sys
def load(p):
    return {json.loads(l)["file"].split("/")[-1]: [x["text"] for x in json.loads(l)["lines"] if x["score"] >= 0.5] for l in open(p)}
def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]
ref, got = load(sys.argv[1]), load(sys.argv[2])
for f in ref:
    r, g = ref[f], got.get(f, [])
    same = sum(1 for x in g if x in r)
    rt, gt = "\n".join(r), "\n".join(g)
    print(f"{f}: ref {len(r)} lines, got {len(g)}, {same} identical; CER {lev(rt, gt) / max(1, len(rt)):.4f}")
    if len(sys.argv) > 3:
        for x in r:
            if x not in g: print("   ref only:", x)
        for x in g:
            if x not in r: print("   got only:", x)
