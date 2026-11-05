"""q4deq.py build|data|check OUT [TILES COLS CHANS]: q4nx -> bfp16ebs8 dequantization on many cores.

One worker per shim DMA channel (COLS x CHANS); each dequantizes TILES / workers tiles of
m_tile x k_tile = 32 x 256 weights (5120 bytes in, 9216 bytes out) with kernels.quant.q4nx_dequant.
"""
import json
import os
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernels.quant import _q4nx_sample, q4nx_dequant, q4nx_dequant_ref
from aie.iron.runtime import TaskGroup

GEOM = dict(m_tile=32, k_tile=256, group=32, ct_k=128, s=8, t=8)
IN_B = 32 * 256 // 2 + 4 * 32 * 256 // 32
OUT_B = 32 * 256 * 9 // 8


@iron.jit
def q4deq(a: In, c: Out, *, tiles: CompileTime[int], cols: CompileTime[int], chans: CompileTime[int]):
    n = cols * chans
    per = tiles // n
    kern = q4nx_dequant(**GEOM)
    ins = [ObjectFifo(np.ndarray[(IN_B,), np.dtype[np.uint8]], name=f"in{i}") for i in range(n)]
    outs = [ObjectFifo(np.ndarray[(OUT_B,), np.dtype[np.uint8]], name=f"out{i}") for i in range(n)]

    def core(fi, fo, k):
        for _ in range_(per):
            x = fi.acquire(1)
            y = fo.acquire(1)
            k(x, y)
            fi.release(1)
            fo.release(1)

    workers = [Worker(core, [ins[i].cons(), outs[i].prod(), kern]) for i in range(n)]
    tin = TensorAccessPattern.full((tiles * IN_B,)).partition(n)
    tout = TensorAccessPattern.full((tiles * OUT_B,)).partition(n)
    prods = [f.prod() for f in ins]
    conses = [f.cons() for f in outs]

    def sequence(a, c, ph, ch):
        tg = TaskGroup()
        for i in range(n):
            ph[i].fill(a, tin[i], group=tg)
        for i in range(n):
            ch[i].drain(c, tout[i], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [np.ndarray[(tiles * IN_B,), np.dtype[np.uint8]], np.ndarray[(tiles * OUT_B,), np.dtype[np.uint8]], prods, conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


cmd, out = sys.argv[1], sys.argv[2]
if cmd == "build":
    from ironcompile import compile

    tiles, cols, chans = map(int, sys.argv[3:6])
    assert tiles % (cols * chans) == 0
    compile(q4deq, out, tiles=tiles, cols=cols, chans=chans)
    json.dump(dict(tiles=tiles, cols=cols, chans=chans), open(os.path.join(out, "params.json"), "w"))
    sys.exit()
p = json.load(open(os.path.join(out, "params.json")))
rng = np.random.default_rng(7)
payload = _q4nx_sample(rng, p["tiles"], **GEOM)[0]  # one (tiles, IN_B) array
payload = np.asarray(payload, dtype=np.uint8).reshape(p["tiles"], IN_B)
if cmd == "data":
    payload.tofile(os.path.join(out, "A.bin"))
    print(p["tiles"] * OUT_B)
else:
    got = np.fromfile(os.path.join(out, "C.bin"), dtype=np.uint8).reshape(p["tiles"], OUT_B)
    want = q4nx_dequant_ref(payload, **GEOM)
    bad = int((got != want).any(axis=1).sum())
    print(json.dumps(dict(ok=bad == 0, bad_tiles=bad, tiles=p["tiles"])))
