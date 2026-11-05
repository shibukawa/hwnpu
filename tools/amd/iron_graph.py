"""Export one graph IRON compiled as a full ELF, in the manifest hwnpu's iron.Graph reads.

A graph here is any ``iron.Graph`` called with inputs (not a language model: no decode step,
no prompt chunk, no KV cache): an image model's text encoder, a denoiser step, a VAE decoder.
Into a directory:

  manifest.json    kind "graph": the graph's version (its commands, arguments, inputs,
                   outputs, per-command scratchpad words, as iron_export_lm.py's), its arena
                   banks, and the application's own ``meta``
  graph.elf        the full ELF
  bank_<kind>.bin  the banks as the device holds them: every weight in place, zeros elsewhere
  check/           with ``check``: inputs and graph.reference()'s outputs as raw
                   little-endian bytes and the per-call values, for tools/amd/irongraph

Run in IRON's virtual environment; no NPU is needed (the device is bound by name).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_export import bind_device  # noqa: E402,F401  (re-exported for the applications)
from iron_export_lm import version_manifest, write_banks  # noqa: E402

MANIFEST_VERSION = 1


def trim_zeros(path: Path, align: int = 4096) -> None:
    """Cut a bank image's trailing zeros (whole ``align`` blocks): hwnpu reads a short image into
    a zeroed buffer, and past the weights a bank is the states and scratch, zeros."""
    import mmap
    import os

    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "r+b") as f:
        m = mmap.mmap(f.fileno(), 0)
        end = size
        while end > 0:
            start = max(0, end - (1 << 20))
            kept = m[start:end].rstrip(b"\0")
            if kept:
                end = start + len(kept)
                break
            end = start
        m.close()
        f.truncate(-(-end // align) * align)


def export_graph(compiled, out: Path, name: str, meta: dict | None = None) -> dict:
    """Write ``compiled`` (a CompiledGraph packaged as a full ELF) into ``out``."""
    if not compiled.artifacts.full_elf:
        raise SystemExit(f"{name}: packaged as an xclbin per step; a graph export is a full ELF")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(compiled.artifacts.image, out / "graph.elf")
    vm = version_manifest(compiled, "graph.elf")
    if vm["encoders"] or vm["carried"]:
        raise SystemExit(f"{name}: a graph export takes no gathers of token ids and carries nothing")
    banks = {}
    for a in vm["arguments"]:
        if a["kind"] not in ("input", "output", "feedback"):
            banks[a["kind"]] = max(banks.get(a["kind"], 0), a["bytes"])
    images = write_banks({name: compiled}, banks, out)
    for fn in images.values():
        trim_zeros(out / fn)
    m = dict(
        version=MANIFEST_VERSION,
        kind="graph",
        name=name,
        banks={k: dict(bytes=int(n), image=images.get(k)) for k, n in banks.items()},
        graph=vm,
        meta=meta or {},
    )
    (out / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    return m


def _raw(x, dtype: str) -> bytes:
    from ml_dtypes import bfloat16

    dt = bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
    return np.ascontiguousarray(np.asarray(x).astype(dt)).tobytes()


def write_check(graph, m: dict, out: Path, inputs: dict, values: dict | None = None) -> None:
    """``inputs`` (by name) and graph.reference()'s outputs on them, raw, in check/."""
    d = out / "check"
    d.mkdir(exist_ok=True)
    values = values or {}
    order = []
    for h in m["graph"]["inputs"]:
        x = inputs[h["name"]]
        (d / f"in_{h['name']}.bin").write_bytes(_raw(x, h["dtype"]))
        order.append(x)
    (d / "values.json").write_text(json.dumps(values) + "\n")
    outs = graph.reference(*order, **values)
    if not isinstance(outs, (tuple, list)):
        outs = [outs]
    for h, y in zip(m["graph"]["outputs"], outs):
        (d / f"out_{h['name']}.bin").write_bytes(_raw(y, h["dtype"]))
