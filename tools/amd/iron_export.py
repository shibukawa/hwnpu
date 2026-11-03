#!/usr/bin/env python3
"""Export a graph IRON compiled as a full ELF, for hwnpu to run without IRON or XRT.

Writes, into a directory:

  manifest.json   the ELF, its kernels (one per command), the arguments in order with
                  their sizes, where each buffer lives, the graph's inputs and outputs,
                  weights and per-call parameters
  graph.elf       the full ELF
  check/          with --check: random inputs and the outputs of graph.reference(), raw
                  little-endian bytes, for tools/amd/ironcheck

Run in IRON's virtual environment (Python 3.11 to 3.14, mlir-aie and Peano from IRON's
requirements.txt). No NPU is needed: the device is bound by name.

  python tools/amd/iron_export.py relu-axpy OUT --check

See docs/amd-npu-models.md (4 and 5).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

MANIFEST_VERSION = 1


def bind_device(name: str, cols: int) -> None:
    import aie.utils as aie_utils
    from aie.iron.device import from_name

    aie_utils.set_current_device(from_name(name, n_cols=cols))


def _place(seq, ref):
    """(argument kind, byte offset, bytes) of a BufferRef, scratch resolved to its bank."""
    kind, offset, length = ref.place(seq.layout)
    if kind == "scratch":
        kind, offset = seq.buffer_sizes.scratch_argument(offset, length)
    return kind, int(offset), int(length)


def _handle(seq, h) -> dict:
    kind, offset, length = _place(seq, h.ref())
    return dict(
        name=h.name,
        shape=[int(d) for d in h.shape],
        dtype=str(np.dtype(seq.buffer_dtype(h.ref()))),
        argument=kind,
        offset=offset,
        bytes=length,
    )


def _parameters(compiled) -> list:
    """The scratchpad words: each parameter's slot and kind, and how its value follows from
    one per-call value of the graph, as IRON's Affine form

        clamp(((v * scale + bias) >> down) * mul + add, lo, hi) % mod + post

    (``>>`` floors, a missing bound or mod is none).
    """
    words = {w.symbol: w for w in compiled.words}
    out = []
    for p in compiled.artifacts.parameter_table:
        entry = dict(name=p.name, index=int(p.index), dtype=p.dtype, kind=p.kind)
        w = words.get(p.name)
        if w is None or w.form is None:
            raise SystemExit(f"parameter {p.name} has no Affine form; hwnpu cannot compute it")
        f = w.form
        entry["form"] = dict(
            value=f.value.name, scale=int(f.scale), bias=int(f.bias), down=int(f.down), mul=int(f.mul),
            add=int(f.add), lo=None if f.lo is None else int(f.lo), hi=None if f.hi is None else int(f.hi),
            mod=None if f.mod is None else int(f.mod), post=int(f.post),
        )
        out.append(entry)
    return out


def manifest(compiled, elf_name: str) -> dict:
    """The manifest of a CompiledGraph packaged as a full ELF."""
    from iron.common.image.callable import FullELFCallable
    from iron.common.image.compose import Composition

    art = compiled.artifacts
    if not art.full_elf:
        raise SystemExit("the graph was packaged as an xclbin per step; hwnpu runs full ELFs only")
    seq = compiled.sequence
    sizes = seq.buffer_sizes
    commands = [
        f"{FullELFCallable.device_name}:{Composition.sequence_name(k)}"
        for k in range(len(seq.packaging.commands) + 1)
    ]
    rec = compiled.recording
    weights = []
    for key, (tensor, handle) in rec.weights.items():
        entry = _handle(seq, handle)
        entry["key"] = str(key)
        weights.append(entry)
    buffers = {}
    for root in seq.layout:
        from iron.common.image.step import BufferRef

        ref = BufferRef(root)
        kind, offset, length = _place(seq, ref)
        buffers[root] = dict(argument=kind, offset=offset, bytes=length, dtype=str(np.dtype(seq.buffer_dtype(ref))))
    return dict(
        version=MANIFEST_VERSION,
        elf=elf_name,
        kernels=commands,
        arguments=[dict(index=i, kind=k, bytes=int(n)) for i, (k, n) in enumerate(sizes.arguments().items())],
        buffers=buffers,
        inputs=[_handle(seq, h) for h in rec.inputs],
        outputs=[_handle(seq, h) for h in rec.returned],
        weights=weights,
        parameters=_parameters(compiled),
        plan=compiled.plan.report(rec.name),
    )


def export(compiled, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(compiled.artifacts.image, out / "graph.elf")
    m = manifest(compiled, "graph.elf")
    (out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    return m


def write_check(graph, m: dict, out: Path, rng: np.random.Generator, values: dict) -> None:
    """Random inputs and graph.reference()'s outputs, as raw bytes in check/.

    ``values`` are the per-call values of the call, saved in check/values.json.
    """
    from ml_dtypes import bfloat16

    d = out / "check"
    d.mkdir(exist_ok=True)
    inputs = []
    for h in m["inputs"]:
        x = rng.standard_normal(h["shape"]).astype(np.dtype(h["dtype"]) if h["dtype"] != "bfloat16" else bfloat16)
        x.tofile(d / f"in_{h['name']}.bin")
        inputs.append(x)
    (d / "values.json").write_text(json.dumps(values) + "\n")
    outs = graph.reference(*inputs, **values)
    if not isinstance(outs, (tuple, list)):
        outs = [outs]
    for h, y in zip(m["outputs"], outs):
        np.asarray(y).astype(bfloat16 if h["dtype"] == "bfloat16" else np.dtype(h["dtype"])).tofile(d / f"out_{h['name']}.bin")


def demo_relu_axpy(size: int):
    import iron
    from iron.operators import AXPY, ReLU

    class ReluThenAxpy(iron.Graph):
        """``2 * relu(x) + y`` (IRON's programming guide, section 2)."""

        def body(self, x, y):
            return AXPY.step(ReLU.step(x), y, scalar_factor=2.0)

    g = ReluThenAxpy()
    return g, g.compile(x=(size,), y=(size,))


def demo_bounded_relu(size: int):
    import iron
    from iron import Scratchpad
    from iron.operators import ReLU

    class BoundedRelu(iron.Graph):
        """``relu(x[:n])`` with ``n`` per call (IRON's programming guide, section 3)."""

        def body(self, x, *, n: Scratchpad[np.int32]):
            return ReLU.step(x[:n])

    g = BoundedRelu()
    return g, g.compile(x=(size,))


DEMOS = {"relu-axpy": demo_relu_axpy, "bounded-relu": demo_bounded_relu}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", choices=sorted(DEMOS))
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--check", action="store_true", help="write inputs and reference outputs")
    ap.add_argument("--values", nargs="*", default=[], help="per-call values of the checked call, name=int")
    a = ap.parse_args(argv)
    bind_device(a.device, a.cols)
    graph, compiled = DEMOS[a.graph](a.size)
    m = export(compiled, a.out)
    if a.check:
        values = {k: int(v) for k, v in (x.split("=", 1) for x in a.values)}
        write_check(graph, m, a.out, np.random.default_rng(0), values)
    print(f"{a.out}: {m['kernels']}, arguments {[(x['kind'], x['bytes']) for x in m['arguments']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
