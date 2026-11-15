"""A GroupNorm's passes over the pixels as two operators, its small arithmetic on the host.

IRON has no GroupNorm and no reciprocal square root. iron_chroma_parts.FluxVaeStage.group_norm
builds one from GEMMs and RMS norms, about 20 operators, 13 reads and 9 writes of the feature
map: at 1024 x 1024 with 128 channels a norm took 157 ms beside a 3 x 3 convolution's 170.

Here the device passes over the pixels twice:

  ChannelStats   a line of pixels (bf16) to each channel's sum and sum of squares over it
                 (float32): one read, an output an eighth of the input or less
  RowAffine      y = x a + b, a and b a value a channel in two bf16 limbs, float32 inside

and the host, between the two commands, sums the lines' statistics and computes each channel's
a = gamma / sqrt(var + eps) and b = beta - mean a in float64 (hwnpu's anima.VAE, a graph's
``norms`` in its meta). The pads of a grid are zeros: they add nothing to the sums, and the
host divides by the image's pixels.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
from aie.iron import In, Out
from aie.utils.compile.jit.params import param
from ml_dtypes import bfloat16

from iron.common import RowBroadcast, Rowwise
from iron.common.elementwise import CORES

KERNELS = Path(__file__).resolve().parent / "kernels"


def _extern(name: str, source: str, types: list, flags: list, roles, bindings, reference, tolerance):
    from aie.iron.kernels._common import KernelContract, Trace, _make_extern
    from aie.utils import config

    kdir = Path(config.aie_kernels_dir())
    return _make_extern(
        name, KERNELS / source, types, compile_flags=flags,
        contract=KernelContract(trace=Trace.whole_call(), roles=roles, parameter_bindings=bindings,
                                reference=reference, tolerance=tolerance, acc_dtype=np.float32),
        include_dirs=[str(kdir), config.cxx_header_path(), str(KERNELS)])


class ChannelStats(Rowwise):
    """Lines of ``width`` = pixels x ``channels`` bf16 to (lines, 2 channels) float32: each
    channel's sum over the line's pixels, then each channel's sum of squares."""

    width: int = param()
    channels: int = param(array=True)
    tile_size: int = param(derive=lambda op: 2 * op.channels)

    x = In(Rowwise.rows, width, obj_shape=(width,), per=CORES)
    y = Out(Rowwise.rows, tile_size, dtype=np.float32, obj_shape=(tile_size,), per=CORES)

    def validate(self) -> None:
        if self.channels % 64 or self.width % self.channels:
            raise ValueError(f"ChannelStats: lines of {self.width} in channels of {self.channels}")

    def kernel(self):
        from aie.utils.compile.jit.markers import In as KIn
        from aie.utils.compile.jit.markers import Out as KOut
        from aie.iron.kernels._common import Param
        from aie.utils.verify import Tolerance

        C, n = self.channels, self.width

        def reference(x):  # one line or many
            v = np.asarray(x, np.float64).reshape(-1, n // C, C)
            return np.concatenate([v.sum(1), (v * v).sum(1)], axis=1).astype(np.float32)

        return _extern("chan_stats", "chan_stats.cc",
                       [np.ndarray[(n,), np.dtype[bfloat16]], np.ndarray[(2 * C,), np.dtype[np.float32]], np.int32],
                       [f"-DCS_C={C}", f"-DCS_P={n // C}"], (KIn, KOut, Param), ((2, 2 * C),), reference,
                       Tolerance.relative(1e-5, atol=1e-5))


class RowAffine(RowBroadcast):
    """Each line of pixels x ``channels`` times a plus b (kernels/row_affine.cc): ``row`` holds
    a's high limbs, a's low limbs, b's high limbs, b's low limbs (``limbs``), then anything."""

    channels: int = param(array=True)
    tile_cap: ClassVar[int] = 4096

    def validate(self) -> None:
        super().validate()
        if self.channels % 64 or self.tile_size % self.channels or self.tile_size < 4 * self.channels:
            raise ValueError(f"RowAffine: lines of {self.tile_size} in channels of {self.channels}")

    def kernel(self):
        from aie.utils.compile.jit.markers import In as KIn
        from aie.utils.compile.jit.markers import Out as KOut
        from aie.iron.kernels._common import Param
        from aie.utils.verify import Tolerance

        C, n = self.channels, self.line
        if n != self.tile_size:
            raise ValueError(f"RowAffine: a row of {self.tile_size} cut into lines of {n}")
        t = np.ndarray[(n,), np.dtype[bfloat16]]

        def reference(x, row):
            r = np.asarray(row, np.float64).reshape(-1, n)[0]
            a, b = r[:C] + r[C : 2 * C], r[2 * C : 3 * C] + r[3 * C : 4 * C]
            v = np.asarray(x, np.float64).reshape(-1, C)
            return (v * a + b).reshape(np.shape(x)).astype(bfloat16)

        return _extern("row_affine", "row_affine.cc", [t, t, t, np.int32], [f"-DRA_C={C}"],
                       (KIn, KIn, KOut, Param), ((3, n),), reference, Tolerance.bf16_ulps(1, atol=1e-6))


def limbs(a, b, line: int) -> np.ndarray:
    """RowAffine's row of ``line`` bf16 for the channels' factors a and terms b (float): each in
    two bf16 limbs."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    out = np.zeros(line, bfloat16)
    C = len(a)
    for k, v in enumerate((a, b)):
        hi = v.astype(bfloat16)
        out[2 * k * C : (2 * k + 1) * C] = hi
        out[(2 * k + 1) * C : (2 * k + 2) * C] = (v - hi.astype(np.float64)).astype(bfloat16)
    return out


def stats_pixels(channels: int) -> int:
    """The pixels of a ChannelStats line of 8192 values."""
    return 8192 // channels


def line_pixels(channels: int) -> int:
    """The pixels of a line of 4096 values (RowAffine's cap; the row's four limbs fit from 4 pixels)."""
    return 4096 // channels
