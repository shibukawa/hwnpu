"""ELU in float32 (kernels/elu_f32.cc) for SpectroStream's decoder (iron_magenta.py).

IRON has no ELU. ``EluF32`` is IRON's SiLU operator (an elementwise step over a handle's values)
with the kernel replaced, as ``iron_silu.SiLUF32``: exp(min(x, 0)) - 1 + max(x, 0) in float32,
the exponential IRON's own float32 one, rounded once to bf16.
"""

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from iron.operators.silu import SiLU

KERNELS = Path(__file__).resolve().parent / "kernels"


class EluF32(SiLU):
    """ELU by kernels/elu_f32.cc."""

    def kernel(self):
        from aie.iron.kernels._common import KernelContract, Param, Trace, _make_extern
        from aie.utils import config
        from aie.utils.compile.jit.markers import In, Out
        from aie.utils.verify import Tolerance

        n = self.tile_size
        t = np.ndarray[(n,), np.dtype[bfloat16]]
        kdir = Path(config.aie_kernels_dir())
        return _make_extern(
            "elu_f32_bf16_size", KERNELS / "elu_f32.cc", [t, t, np.int32],
            contract=KernelContract(
                trace=Trace.whole_call(), roles=(In, Out, Param), parameter_bindings=((2, n),),
                reference=elu_ref, tolerance=Tolerance.bf16_ulps(2, atol=1e-6),
                acc_dtype=np.float32),
            include_dirs=[str(kdir), config.cxx_header_path(), str(KERNELS)])


def elu_ref(x):
    """ELU in float64, rounded to bf16."""
    x = np.asarray(x, np.float64)
    return np.where(x > 0, x, np.expm1(np.clip(x, -30.0, 0.0))).astype(bfloat16)


def elu(x):
    """ELU of a handle by EluF32."""
    return EluF32.step(x)
