"""gemm_bfp.py build|data|check OUT [M K N m k n COLS]: native bfp16ebs8 whole_array GEMM on npu2.

The host encodes A (M, K) and B transposed (N, K) into bfp16ebs8 and shuffles them into the
mmul tile layout (aie.utils.bfp); C comes back shuffled bfp16ebs8.
"""
import json
import os
import sys

import numpy as np
from aie.iron import kernels
from aie.utils import bfp
from aie.utils.verify import compare

SRC = os.environ["MLIR_AIE_SRC"] + "/programming_examples/ml/block_datatypes/matrix_multiplication/whole_array/whole_array.py"


def main():
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == "build":
        M, K, N, m, k, n, cols = map(int, sys.argv[3:10])
        p = dict(M=M, K=K, N=N, m=m, k=k, n=n, cols=cols)
        from ironcompile import compile, load

        mod = load(SRC, "whole_array_bfp")
        compile(mod.whole_array_matmul, out, M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=cols)
        json.dump(p, open(os.path.join(out, "params.json"), "w"))
        return
    p = json.load(open(os.path.join(out, "params.json")))
    M, K, N, m, k, n = p["M"], p["K"], p["N"], p["m"], p["k"], p["n"]
    rng = np.random.default_rng(1726250518)
    A = (rng.random((M, K)) * 4.0).astype(np.float32)
    B = (rng.random((K, N)) * 4.0).astype(np.float32)
    if cmd == "data":
        bfp.shuffle(bfp.encode(A), K, M, k, m).tofile(os.path.join(out, "A.bin"))
        bfp.shuffle(bfp.encode(np.ascontiguousarray(B.T)), K, N, k, n).tofile(os.path.join(out, "B.bin"))
        print(M * N * 9 // 8)
        return
    raw = np.fromfile(os.path.join(out, "C.bin"), dtype=np.uint8)
    C = bfp.decode(bfp.shuffle(raw, N, M, n, m, unshuffle=True)).reshape(M, N)
    from aie.iron.device import NPU2
    from aie.utils.hostruntime import set_current_device

    set_current_device(NPU2())
    tol = kernels.mm_bfp(dim_m=m, dim_k=k, dim_n=n).contract.tolerance
    v = compare(C.astype(np.float32), kernels.mm_bfp_ref(A, B), tol)
    print(json.dumps(dict(ok=bool(v.ok), mismatch=v.n_mismatch, checked=v.n_checked, max_abs=float(v.max_abs_err), detail=v.detail[:200])))


main()
