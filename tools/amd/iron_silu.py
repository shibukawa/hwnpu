"""SiLU with its sigmoid in float32 (kernels/silu_f32.cc), for a language model's SwiGLU.

IRON's SiLU multiplies x by a bf16 table's sigmoid: 1 % off on average over [-6, 2] (2 % at
most), and exactly 0 from x = -8 down. SwiGLU multiplies it by the other branch, which in a
language model's few large channels is in the tens or hundreds: the zeros and the 1 % then make
up much of the MLP's output. Qwen3-4B's hidden states (FLUX.2 klein's text encoder, 512 rows)
were at cosine 0.9994 of float32's after three layers with IRON's SiLU; every other operation
of a layer was within 0.99999.

``SiLUF32`` computes x / (1 + exp(-x)) with IRON's float32 exponential and the cores' float
inverse and rounds once to bf16.
"""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from iron.operators.silu import SiLU

KERNELS = Path(__file__).resolve().parent / "kernels"


class SiLUF32(SiLU):
    """SiLU by kernels/silu_f32.cc."""

    def kernel(self):
        from aie.iron.kernels._common import KernelContract, Param, Trace, _make_extern
        from aie.utils import config
        from aie.utils.compile.jit.markers import In, Out
        from aie.utils.verify import Tolerance

        n = self.tile_size
        t = np.ndarray[(n,), np.dtype[bfloat16]]
        kdir = Path(config.aie_kernels_dir())
        return _make_extern(
            "silu_f32_bf16_size", KERNELS / "silu_f32.cc", [t, t, np.int32],
            contract=KernelContract(
                trace=Trace.whole_call(), roles=(In, Out, Param), parameter_bindings=((2, n),),
                reference=silu_ref, tolerance=Tolerance.bf16_ulps(2, atol=1e-6),
                acc_dtype=np.float32),
            include_dirs=[str(kdir), config.cxx_header_path(), str(KERNELS)])


def silu_ref(x):
    """x sigmoid(x) in float64, rounded to bf16."""
    x = np.asarray(x, np.float64)
    with np.errstate(over="ignore"):
        return (x / (1.0 + np.exp(-x))).astype(bfloat16)


def silu(x):
    """SiLU of a handle by SiLUF32."""
    return SiLUF32.step(x)
