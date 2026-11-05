"""q4gemv.py build|data|check OUT [M K COLS ROWS]: y = W x with q4nx weights on COLS x ROWS cores.

W is M x K in q4nx blocks of 32 x 256 (a bf16 scale and minimum per 32 values along K, 4-bit
codes), x is bf16, y is float32. Core (col, row) owns M / cores consecutive rows. Per column the
shim streams the weights of its ROWS cores interleaved block by block to the memtile, which
splits them; x is broadcast through the memtile; the cores' y slices are joined back.
"""
import json
import os
import sys

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.runtime import TaskGroup
from ml_dtypes import bfloat16

HERE = os.path.dirname(os.path.abspath(__file__))
BM, BK, G = 32, 256, 32
BLK = BM * BK // 2 + 4 * BM * BK // G  # 5120 bytes


@iron.jit
def q4gemv(w: In, x: In, y: Out, *, M: CompileTime[int], K: CompileTime[int], cols: CompileTime[int], rows: CompileTime[int]):
    cores = cols * rows
    rc = M // cores  # rows per core
    steps = (rc // BM) * (K // BK)
    blk_ty = np.ndarray[(BLK,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    yc_ty = np.ndarray[(rc,), np.dtype[np.float32]]
    inc = [os.path.join(os.environ["MLIR_AIE_SRC"], "aie_kernels", "flm_gemma4")]
    mv = ExternalFunction("q4_mv", source_file=os.path.join(HERE, "q4gemv.cc"), arg_types=[blk_ty, x_ty, yc_ty, np.int32, np.int32], include_dirs=inc, compile_flags=["-DQ4GEMV_MV"])
    zero = ExternalFunction("q4_zero", source_file=os.path.join(HERE, "q4gemv.cc"), arg_types=[yc_ty, np.int32], include_dirs=inc, compile_flags=["-DQ4GEMV_ZERO"])

    w_l3, w_l1, x_l3, x_l1, y_l3, y_l1 = [], [], [], [], [], []
    for c in range(cols):
        wf = ObjectFifo(np.ndarray[(BLK * rows,), np.dtype[np.uint8]], name=f"w{c}")
        w_l3.append(wf)
        w_l1.append(wf.cons().split([BLK * r for r in range(rows)], obj_types=[blk_ty] * rows, names=[f"w{c}_{r}" for r in range(rows)]))
        xf = ObjectFifo(x_ty, name=f"x{c}")
        x_l3.append(xf)
        x_l1.append(xf.cons().forward(obj_type=x_ty, name=f"xl{c}"))
        yf = ObjectFifo(np.ndarray[(rc * rows,), np.dtype[np.float32]], name=f"y{c}")
        y_l3.append(yf)
        y_l1.append(yf.prod().join([rc * r for r in range(rows)], obj_types=[yc_ty] * rows, names=[f"y{c}_{r}" for r in range(rows)]))

    def core(fw, fx, fy, k_mv, k_zero):
        xb = fx.acquire(1)
        yb = fy.acquire(1)
        k_zero(yb, rc)
        for rb in range_(rc // BM):
            for cb in range_(K // BK):
                wb = fw.acquire(1)
                k_mv(wb, xb, yb, cb, rb)
                fw.release(1)
        fx.release(1)
        fy.release(1)

    workers = [Worker(core, [w_l1[c][r].cons(), x_l1[c].cons(), y_l1[c][r].prod(), mv, zero]) for c in range(cols) for r in range(rows)]
    wp = [f.prod() for f in w_l3]
    xp = [f.prod() for f in x_l3]
    yc = [f.cons() for f in y_l3]
    from aie.helpers.taplib import TensorAccessPattern

    wt = TensorAccessPattern.full((M * K * 5 // 8,)).partition(cols)
    yt = TensorAccessPattern.full((M,)).partition(cols)

    def sequence(w, x, y, wh, xh, yh):
        tg = TaskGroup()
        for c in range(cols):
            xh[c].fill(x, group=tg)
            wh[c].fill(w, wt[c], group=tg)
        for c in range(cols):
            yh[c].drain(y, yt[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [np.ndarray[(M * K * 5 // 8,), np.dtype[np.uint8]], x_ty, np.ndarray[(M,), np.dtype[np.float32]], wp, xp, yc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def make(M, K, rng):
    q = rng.integers(0, 16, size=(M, K), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.1, size=(M, K // G)).astype(bfloat16)
    mins = (-8 * scales.astype(np.float32)).astype(bfloat16)
    x = rng.uniform(-1, 1, size=K).astype(bfloat16)
    return q, scales, mins, x


def block(q, scales, mins, r0, c0):
    """One q4nx block: scales and minima indexed [k_group, row], codes [row//16, k, row%16]."""
    sc = scales[r0:r0 + BM, c0 // G:(c0 + BK) // G].T.ravel()
    mn = mins[r0:r0 + BM, c0 // G:(c0 + BK) // G].T.ravel()
    codes = q[r0:r0 + BM, c0:c0 + BK].reshape(BM // 16, 16, BK).transpose(0, 2, 1).ravel()
    packed = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
    return np.concatenate([sc.view(np.uint8), mn.view(np.uint8), packed])


def main():
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == "build":
        from ironcompile import compile

        M, K, cols, rows = map(int, sys.argv[3:7])
        compile(q4gemv, out, M=M, K=K, cols=cols, rows=rows)
        json.dump(dict(M=M, K=K, cols=cols, rows=rows), open(os.path.join(out, "params.json"), "w"))
        return
    p = json.load(open(os.path.join(out, "params.json")))
    M, K, cols, rows = p["M"], p["K"], p["cols"], p["rows"]
    rc = M // (cols * rows)
    q, scales, mins, x = make(M, K, np.random.default_rng(11))
    if cmd == "data":
        parts = []
        for c in range(cols):
            for rb in range(rc // BM):
                for cb in range(K // BK):
                    for r in range(rows):
                        core = c * rows + r
                        parts.append(block(q, scales, mins, core * rc + rb * BM, cb * BK))
        np.concatenate(parts).tofile(os.path.join(out, "W.bin"))
        x.tofile(os.path.join(out, "x.bin"))
        print(M * 4)
        return
    W = np.repeat(scales.astype(np.float64), G, axis=1) * q + np.repeat(mins.astype(np.float64), G, axis=1)
    ref = W @ x.astype(np.float64)
    got = np.fromfile(os.path.join(out, "y.bin"), dtype=np.float32).astype(np.float64)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    res = dict(ok=bool(err < 1e-2), max_rel_err=float(err), max_abs_ref=float(np.abs(ref).max()))
    # which bf16 rounding the kernel uses: compare with its arithmetic under both modes
    for mode in ("nearest", "floor"):
        res[f"max_rel_vs_{mode}_model"] = float(np.abs(got - kernel_model(q, scales, mins, x, mode == "floor")).max() / np.abs(ref).max())
    print(json.dumps(res))


def bf16r(a, floor):
    """float32 to bf16 (as float32), to the nearest (ties to even) or toward minus infinity."""
    b = np.asarray(a, np.float32).view(np.uint32).astype(np.uint64)
    if floor:
        b = b + (((b >> np.uint64(31)) != 0) & ((b & np.uint64(0xFFFF)) != 0)).astype(np.uint64) * np.uint64(0x10000)
    else:
        b = b + np.uint64(0x7FFF) + ((b >> np.uint64(16)) & np.uint64(1))
    return (b & np.uint64(0xFFFF0000)).astype(np.uint32).view(np.float32)


def kernel_model(q, scales, mins, x, floor):
    """y as q4nx_accumulate computes it (internal/milcpu linearQ4NX): per 32 columns,
    bf16(sum code * x) * scale + min * bf16(sum x), in float32."""
    M, K = q.shape
    xb = x.astype(np.float32)  # already bf16
    qg = q.reshape(M, K // G, G).astype(np.float32)
    dots = np.zeros((M, K // G), np.float32)
    for c in range(G):  # in column order, as the vector MACs accumulate
        dots += qg[:, :, c] * xb.reshape(K // G, G)[:, c]
    sx = np.zeros(K // G, np.float32)
    for c in range(G):
        sx += xb.reshape(K // G, G)[:, c]
    terms = bf16r(dots, floor) * scales.astype(np.float32) + mins.astype(np.float32) * bf16r(sx, floor)
    y = np.zeros(M, np.float32)
    for g in range(K // G):
        y += terms[:, g]
    return y.astype(np.float64)


if __name__ == "__main__":
    main()
