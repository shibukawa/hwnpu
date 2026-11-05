"""gemm.py build|data|check OUT [opts]: whole_array GEMM on npu2 (8 columns x 4 rows).

build OUT DTIN DTOUT M K N m k n COLS EMU : compile mlir-aie's whole_array.py
data  OUT : write A.bin, B.bin (row-major) and the C size
check OUT : compare C.bin with kernels.mm_ref under the kernel's own tolerance
"""
import json
import os
import sys

import numpy as np
from aie.iron import kernels
from aie.utils.verify import compare
from ml_dtypes import bfloat16

SRC = os.environ["MLIR_AIE_SRC"] + "/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py"
DT = {"i8": np.int8, "i16": np.int16, "i32": np.int32, "bf16": bfloat16, "f32": np.float32}


def main():
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == "build":
        dti, dto, M, K, N, m, k, n, cols, emu = sys.argv[3:13]
        p = dict(dtin=dti, dtout=dto, M=int(M), K=int(K), N=int(N), m=int(m), k=int(k), n=int(n), cols=int(cols), emu=int(emu))
        from ironcompile import compile, load

        mod = load(SRC, "whole_array")
        tin, tout = DT[dti], DT[dto]
        compile(
            mod.whole_array, out,
            A=np.ndarray[(p["M"], p["K"]), np.dtype[tin]],
            B=np.ndarray[(p["K"], p["N"]), np.dtype[tin]],
            C=np.ndarray[(p["M"], p["N"]), np.dtype[tout]],
            M=p["M"], K=p["K"], N=p["N"], m=p["m"], k=p["k"], n=p["n"], n_aie_cols=p["cols"],
            emulate_bf16_mmul_with_bfp16=bool(p["emu"]),
        )
        json.dump(p, open(os.path.join(out, "params.json"), "w"))
        return
    p = json.load(open(os.path.join(out, "params.json")))
    tin, tout = DT[p["dtin"]], DT[p["dtout"]]
    M, K, N = p["M"], p["K"], p["N"]
    rng = np.random.default_rng(1726250518)
    if cmd == "data":
        if np.issubdtype(tin, np.integer):
            info = np.iinfo(tin)
            A = rng.integers(info.min // 4, info.max // 4, size=(M, K), dtype=tin)
            B = rng.integers(info.min // 4, info.max // 4, size=(K, N), dtype=tin)
        else:
            A = (rng.random((M, K)) * 4.0).astype(tin)
            B = (rng.random((K, N)) * 4.0).astype(tin)
        A.tofile(os.path.join(out, "A.bin"))
        B.tofile(os.path.join(out, "B.bin"))
        print(M * N * np.dtype(tout).itemsize)
        return
    A = np.fromfile(os.path.join(out, "A.bin"), dtype=tin).reshape(M, K)
    B = np.fromfile(os.path.join(out, "B.bin"), dtype=tin).reshape(K, N)
    C = np.fromfile(os.path.join(out, "C.bin"), dtype=tout).reshape(M, N)
    tol = kernels.mm(dim_m=p["m"], dim_k=p["k"], dim_n=p["n"], input_dtype=tin, output_dtype=tout).contract.tolerance
    v = compare(C, kernels.mm_ref(A, B), tol)
    print(json.dumps(dict(ok=bool(v.ok), mismatch=v.n_mismatch, checked=v.n_checked, max_abs=float(v.max_abs_err), detail=v.detail[:200])))


main()
