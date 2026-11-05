"""dma_bw.py OUT MODE N COLS CHANS LINE: one-way DMA bandwidth design for npu2.

MODE read : DDR -> shim -> core; each core acquires and releases every line (no output)
MODE write: core -> shim -> DDR; each core releases lines without writing them (no input)
N int32 in total, split evenly over COLS x CHANS shim DMA channels, LINE int32 per buffer.
"""
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.runtime import TaskGroup
from ironcompile import compile


@iron.jit
def dma_read(a: In, *, size: CompileTime[int], cols: CompileTime[int], chans: CompileTime[int], line: CompileTime[int]):
    n = cols * chans
    line_ty = np.ndarray[(line,), np.dtype[np.int32]]
    fifos = [ObjectFifo(line_ty, name=f"in{i}") for i in range(n)]
    lines = size // n // line

    def core(f):
        for _ in range_(lines):
            f.acquire(1)
            f.release(1)

    workers = [Worker(core, [f.cons()]) for f in fifos]
    taps = TensorAccessPattern.full((size,)).partition(n)
    prods = [f.prod() for f in fifos]

    def sequence(a, hs):
        tg = TaskGroup()
        for i, h in enumerate(hs):
            h.fill(a, taps[i], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [np.ndarray[(size,), np.dtype[np.int32]], prods])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit
def dma_write(c: Out, *, size: CompileTime[int], cols: CompileTime[int], chans: CompileTime[int], line: CompileTime[int]):
    n = cols * chans
    line_ty = np.ndarray[(line,), np.dtype[np.int32]]
    fifos = [ObjectFifo(line_ty, name=f"out{i}") for i in range(n)]
    lines = size // n // line

    def core(f):
        for _ in range_(lines):
            f.acquire(1)
            f.release(1)

    workers = [Worker(core, [f.prod()]) for f in fifos]
    taps = TensorAccessPattern.full((size,)).partition(n)
    conses = [f.cons() for f in fifos]

    def sequence(c, hs):
        tg = TaskGroup()
        for i, h in enumerate(hs):
            h.drain(c, taps[i], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [np.ndarray[(size,), np.dtype[np.int32]], conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


out, mode, size, cols, chans, line = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
assert size % (cols * chans * line) == 0
fn = dma_read if mode == "read" else dma_write
compile(fn, out, size=size, cols=cols, chans=chans, line=line)
