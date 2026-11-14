"""RMS norm of each head of a row (kernels/head_rms.cc), for image models' q and k norms.

IRON's RMSNorm over (rows x heads, head_dim) runs a kernel call and a fifo object a head: a
4096 x 2048 q of Anima's DiT took 1.4 ms, 2.5 times a LayerNorm of the same bytes in rows of
2048. ``HeadRMSNorm`` takes whole rows and normalizes each head of ``head_dim`` in them; the
weight is the head's row repeated over the heads (``head_weight``), multiplied on WeightedRowwise's
second core as RMSNorm's.
"""

from pathlib import Path

import numpy as np
from aie.utils.compile.jit.params import param
from ml_dtypes import bfloat16

import iron
from iron.operators.rms_norm import RMSNorm

KERNELS = Path(__file__).resolve().parent / "kernels"


class HeadRMSNorm(RMSNorm):
    """Each ``head_dim`` slice of a row RMS-normed on its own (then times the weight row)."""

    head_dim: int = param(default=128, array=True)

    def validate(self) -> None:
        super().validate()
        if self.tile_size % self.head_dim or self.head_dim % 64:
            raise ValueError(f"HeadRMSNorm: rows of {self.tile_size} in heads of {self.head_dim}")

    def kernel(self):
        from aie.iron.kernels import activation as act
        from aie.iron.kernels._common import KernelContract, Param, Trace, _make_extern
        from aie.utils import config
        from aie.utils.compile.jit.markers import In, Out
        from aie.utils.verify import Tolerance

        n, d = self.tile_size, self.head_dim
        t = np.ndarray[(n,), np.dtype[bfloat16]]
        kdir = Path(config.aie_kernels_dir())
        runtime = Path(config.aie_runtime_lib_dir()) / act._detect_arch().upper()

        def reference(x, epsilon):
            v = np.asarray(x, np.float64).reshape(-1, d)
            y = v / np.sqrt((v * v).mean(axis=1, keepdims=True) + epsilon)
            return y.reshape(np.shape(x)).astype(bfloat16)

        return _make_extern(
            "head_rms_norm_eps", KERNELS / "head_rms.cc", [t, t, np.int32, np.float32],
            compile_flags=[f"-DHR_D={d}"],
            contract=KernelContract(
                trace=Trace.whole_call(), roles=(In, Out, Param, Param), parameter_bindings=((2, n),),
                reference=reference, tolerance=Tolerance.bf16_ulps(2, atol=1e-6), acc_dtype=np.float32),
            include_dirs=[str(kdir), config.cxx_header_path(), str(runtime)])


def head_weight(w, heads: int):
    """The head's norm weight repeated over ``heads`` heads: the weight row of a whole row."""
    return iron.param(np.tile(np.asarray(w, np.float32), heads).astype(bfloat16))


def rms_norm(x, weight, eps: float, head_dim: int = 128):
    """``x`` (rows, heads x head_dim) RMS-normed per head, times ``weight`` (head_weight)."""
    return HeadRMSNorm.step(x, weight=weight, epsilon=eps, head_dim=head_dim)
