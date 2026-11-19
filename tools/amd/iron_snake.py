"""Snake (Snake1d: x + sin(alpha x)^2 / (alpha + 1e-9), alpha a channel's) in float32
(kernels/snake_f32.cc) for the DAC's decoder of MOSS-SoundEffect (iron_moss_dac.py).

IRON has no sin. ``Snake`` is a RowBroadcast, as iron_groupnorm.RowAffine: a line of pixels x
``channels`` met by a row of each channel's alpha and inverse, two bf16 limbs each (``row``); the
kernel reduces alpha x by pi and takes sin by its series in float32 (the reduction internal/sfx's
stages use for Intel's NPU, and used for Core ML's Neural Engine, in fp16).
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
from aie.utils.compile.jit.params import param
from ml_dtypes import bfloat16

from iron.common import RowBroadcast

KERNELS = Path(__file__).resolve().parent / "kernels"
MAX_CHANNELS = 1024  # the row's four limbs a channel fit a line of tile_cap


def snake_ref(x, alpha) -> np.ndarray:
    """Snake in float64 of x (..., C) for alpha (C,)."""
    a = np.asarray(alpha, np.float64)
    v = np.asarray(x, np.float64).reshape(-1, len(a))
    return (v + np.sin(a * v) ** 2 / (a + 1e-9)).reshape(np.shape(x))


class Snake(RowBroadcast):
    """Each line of pixels x ``channels`` through Snake (kernels/snake_f32.cc): ``row`` holds
    alpha's high limbs, alpha's low limbs, the inverse's high limbs, its low limbs (``row``)."""

    channels: int = param(array=True)
    tile_cap: ClassVar[int] = 4096

    def validate(self) -> None:
        super().validate()
        if self.channels % 32 or self.tile_size % self.channels or self.tile_size < 4 * self.channels:
            raise ValueError(f"Snake: lines of {self.tile_size} in channels of {self.channels}")

    def kernel(self):
        from aie.iron.kernels._common import KernelContract, Param, Trace, _make_extern
        from aie.utils import config
        from aie.utils.compile.jit.markers import In as KIn
        from aie.utils.compile.jit.markers import Out as KOut
        from aie.utils.verify import Tolerance

        C, n = self.channels, self.line
        if n != self.tile_size:
            raise ValueError(f"Snake: a row of {self.tile_size} cut into lines of {n}")
        t = np.ndarray[(n,), np.dtype[bfloat16]]

        def reference(x, r):
            r = np.asarray(r, np.float64).reshape(-1, n)[0]
            alpha = r[:C] + r[C : 2 * C]
            inv = r[2 * C : 3 * C] + r[3 * C : 4 * C]
            v = np.asarray(x, np.float64).reshape(-1, C)
            return (v + np.sin(alpha * v) ** 2 * inv).reshape(np.shape(x)).astype(bfloat16)

        kdir = Path(config.aie_kernels_dir())
        return _make_extern(
            "snake_f32", KERNELS / "snake_f32.cc", [t, t, t, np.int32], compile_flags=[f"-DSN_C={C}"],
            contract=KernelContract(trace=Trace.whole_call(), roles=(KIn, KIn, KOut, Param), parameter_bindings=((3, n),),
                                    reference=reference, tolerance=Tolerance.bf16_ulps(2, atol=1e-5), acc_dtype=np.float32),
            include_dirs=[str(kdir), config.cxx_header_path(), str(KERNELS)])


def line_pixels(channels: int) -> int:
    """The pixels of a line of Snake.tile_cap values."""
    if channels > MAX_CHANNELS or Snake.tile_cap % channels:
        raise ValueError(f"Snake over {channels} channels")
    return Snake.tile_cap // channels


def row(alpha, line: int) -> np.ndarray:
    """Snake's row of ``line`` bf16 for the channels' alpha: alpha and 1 / (alpha + 1e-9), each in
    two bf16 limbs."""
    a = np.asarray(alpha, np.float64).reshape(-1)
    out = np.zeros(line, bfloat16)
    C = len(a)
    for k, v in enumerate((a, 1 / (a + 1e-9))):
        hi = v.astype(bfloat16)
        out[2 * k * C : (2 * k + 1) * C] = hi
        out[(2 * k + 1) * C : (2 * k + 2) * C] = (v - hi.astype(np.float64)).astype(bfloat16)
    return out


def snake(x, row_handle, channels: int):
    """Snake of x (rows, channels), rows a multiple of line_pixels(channels), by its row."""
    P = line_pixels(channels)
    rows = x.shape[0]
    if rows % P:
        raise ValueError(f"Snake over {rows} rows: lines of {P}")
    L = P * channels
    return Snake.step(x.reshape(rows // P, L), row_handle, channels=channels).reshape(rows, channels)
