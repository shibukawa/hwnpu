"""GELU for image models' MLPs: IRON's tanh table with the argument in float32 (kernels/gelu_even.cc).

An Anima DiT block (1024 x 1024) spent 21 of its 154 ms in IRON's exact GELU
(``approximate="none"``: erfc on float32 limbs, correctly rounded to bf16, about 36 cycles a
value over 4096 x 8192). IRON's tanh GELU is 4x faster, but forms the tanh's argument from bf16
products (x^2 and s 0.044715 x, 0.4 % each) and changed Anima's pictures (the clothes of the
test prompt). ``GELUEven`` takes the same table with the argument in float32 (x^2 exact, the
coefficients two bf16 limbs) and rounds to nearest even: 7 ms, the pictures as near bf16
weights' as the exact GELU's (design.md 16.200).

Tried and dropped: the sigmoid form x sigmoid(2u) from an exponential (gpt-oss's 2^k p(f) on
bf16 limbs) and the core's float inverse was exact to 2e-4 and as slow as the exact kernel (the
exponential 2.5 ms and the inverse 1.25 ms of 5.5 a 4096 x 2048 call).
"""

from pathlib import Path

import numpy as np
from aie.utils.compile.jit.params import param
from ml_dtypes import bfloat16

from iron.operators.gelu import GELU

KERNELS = Path(__file__).resolve().parent / "kernels"


class GELUEven(GELU):
    """GELU by kernels/gelu_even.cc (the tanh form; ``approximate`` is not read)."""

    form: str = param(default="even", array=True)

    def validate(self) -> None:
        if self.form != "even":
            raise ValueError(f"GELUEven: form {self.form!r}")

    def kernel(self):
        from aie.iron.kernels import activation as act
        from aie.iron.kernels._common import KernelContract, Param, Trace, _make_extern
        from aie.utils import config
        from aie.utils.compile.jit.markers import In, Out
        from aie.utils.verify import Tolerance

        n = self.tile_size
        t = np.ndarray[(n,), np.dtype[bfloat16]]
        kdir = Path(config.aie_kernels_dir())
        # The table's runtime library, as IRON's _create_lut_kernel takes it
        include = [str(kdir), config.cxx_header_path(), str(KERNELS),
                   str(Path(config.aie_runtime_lib_dir()) / act._detect_arch().upper())]
        return _make_extern(
            "gelu_even_bf16_size", kdir / "common" / "lut_kernel.cc", [t, t, np.int32],
            compile_flags=["-DACTIVATIONS_TANH_LUT=1",
                           f'-DAIE_LUT_KERNEL_SOURCE="{KERNELS / "gelu_even.cc"}"'],
            contract=KernelContract(
                trace=Trace.whole_call(), roles=(In, Out, Param), parameter_bindings=((2, n),),
                reference=gelu_tanh_ref, tolerance=Tolerance.bf16_ulps(2, atol=1e-2),
                acc_dtype=np.float32),
            include_dirs=include)


def gelu_tanh_ref(x):
    """The tanh form in float64, rounded to bf16 (the table's tanh is within 1e-2 of it)."""
    x = np.asarray(x, np.float64)
    u = 0.7978845608028654 * (x + 0.044715 * x**3)
    return (0.5 * x * (1.0 + np.tanh(u))).astype(bfloat16)


def gelu(x):
    """GELU of a handle by GELUEven."""
    return GELUEven.step(x)
