"""GridGEMM: IRON's flm.ProjectionGEMM with A read in place from a channel-last grid.

A 3 x 3 convolution over a grid (rows of Wp pixels, C channels each, iron_sana's GridShape) is a
GEMM of im2col rows: logical A row i holds, for each kernel row dy, the pixels i - 1, i, i + 1 of
grid row (i + (dy - 1) Wp). Those 3 C elements are contiguous in the grid. So A needs no copy:
its DMA reads the grid with logical rows C elements apart (overlapping) and k tiles ``k_stride``
elements apart. ProjectionGEMM's A pattern is a dense (M, K) tile walk; GridGEMM swaps in the
same walk with these strides (sequence), so the array, the kernels and B and C are the same.

  one GEMM a 3 x 3 (C <= 170): K = 9 C laid out dy-major, k_tile = 3 C, k_stride = Wp C
  one GEMM a kernel row: K = 3 C, k_stride = k_tile (the 3 C run, contiguous)

The im2col copy this replaces took 63 % of a 1024 x 1024 ResBlock (6.8 GB/s, design.md 16.139).
"""

import numpy as np
from aie.iron import In
from aie.utils.compile.jit.params import param
from iron.operators.flm import ProjectionGEMM
from iron.operators.flm.gemm import op as flm_op


class _Patterns:
    """flm_op's TensorAccessPattern while a GridGEMM's sequence runs: the A walk's ``full``
    (seven dims, the last two k_iters and k_tile) built with the grid's strides."""

    def __init__(self, real, op):
        self.real, self.op = real, op

    def __getattr__(self, name):
        return getattr(self.real, name)

    def full(self, dims):
        op = self.op
        if len(dims) != 7 or dims[-1] != op.k_tile or dims[-2] * dims[-1] != op.K:
            return self.real.full(dims)
        # rows: the five leading dims split M as the dense walk does, a row CA elements on
        strides, n = [], op.CA
        for d in reversed(dims[:5]):
            strides.append(n)
            n *= d
        strides = list(reversed(strides)) + [op.k_stride or op.k_tile, 1]
        return self.real((op.MA, op.CA), 0, list(dims), strides)


class GridGEMM(ProjectionGEMM):
    """ProjectionGEMM whose A is the grid rows ``(MA, CA)`` read as M logical rows (module doc)."""

    MA: int = param()  # the grid rows given (a run of the grid)
    CA: int = param()  # the grid's channels: one logical row to the next
    k_stride: int = param(default=0)  # elements from one k tile to the next in the grid (0: k_tile)

    A = In(MA, CA, obj_shape=(ProjectionGEMM.a_l2,), per=(ProjectionGEMM.rows,), depth=flm_op.A_DEPTH)

    def validate(self) -> None:
        super().validate()
        if self.k_tile is None:
            return  # checked again once resolved
        last = (self.M - 1) * self.CA + (self.K // self.k_tile - 1) * (self.k_stride or self.k_tile) + self.k_tile
        if last > self.MA * self.CA:
            raise ValueError(f"GridGEMM: A reads to element {last}, past the {self.MA} x {self.CA} grid rows")

    def sequence(self, rt):
        real = flm_op.TensorAccessPattern
        flm_op.TensorAccessPattern = _Patterns(real, self)
        try:
            return super().sequence(rt)
        finally:
            flm_op.TensorAccessPattern = real

    def logical_a(self, A):
        """The dense (M, K) A the grid rows stand for."""
        flat = np.asarray(A).reshape(-1)
        if not self.k_stride:  # each row's K elements contiguous, whatever the k tile
            return flat[(np.arange(self.M)[:, None] * self.CA + np.arange(self.K)[None, :])]
        kt = self.k_tile
        i = np.arange(self.M)[:, None, None] * self.CA
        k = np.arange(self.K // kt)[None, :, None] * (self.k_stride or kt)
        t = np.arange(kt)[None, None, :]
        return flat[(i + k + t).reshape(self.M, -1)]

    def reference(self, A, B):
        return super().reference(self.logical_a(A), B)

    def tolerance(self):
        t = super().tolerance()
        if t is None:
            return None
        from aie.utils.verify import Tolerance

        bound = t.bound
        return Tolerance.bounded(lambda A, B: bound(self.logical_a(A), B), note=t.note)
