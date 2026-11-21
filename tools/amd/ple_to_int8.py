#!/usr/bin/env python3
"""Rewrite the bf16 ple.bin of a Gemma 4 export made before --ple int8 into the int8 format
(iron_hostple.ple_int8) and mark it in the manifest, without exporting again. The programs are
unchanged, so the two exports differ only in the table (an A/B of what the table costs on the
NPU, docs/design/performance.md).

  python tools/amd/ple_to_int8.py EXPORT

Rewrites EXPORT in place: run it on a copy (cp -r --reflink=auto) to keep the bf16 one. An
export that came from the shared layers is no longer that entry afterwards.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_hostple  # noqa: E402


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    d = Path(argv[0])
    mpath = d / "manifest.json"
    m = json.loads(mpath.read_text())
    done = 0
    for t in m.get("host_tables") or []:
        if t["file"] != "ple.bin" or t.get("format"):
            continue
        layers, rows = t["layers"], t["rows"]
        width = layers * t["part_bytes"] // 2
        src = d / t["file"]
        table = np.memmap(src, np.uint16, "r", shape=(rows, width))
        from ml_dtypes import bfloat16

        tmp = d / (t["file"] + ".int8")
        iron_hostple.ple_int8(table.view(bfloat16), layers, tmp)
        del table
        os.replace(tmp, src)
        t["format"] = "int8"
        done += 1
        print(f"{src}: {rows} rows of {layers} x {width // layers} -> {src.stat().st_size / 2**30:.2f} GiB")
    if not done:
        print(f"{d}: no bf16 ple.bin")
        return 1
    mpath.write_text(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
