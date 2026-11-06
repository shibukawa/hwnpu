"""A graph of one windowed attention as a Gemma 4 sliding layer runs it, for tools/amd/irongraph.

    python tools/amd/iron_window_probe.py OUT [--decode] [--scan] [--position P] --check
    go run ./tools/amd/irongraph -tol 0.05 OUT

Queries (a 1024-row prompt chunk, or one decode query) over a 4096-row cache of keys and values,
window 511, the keys a span ending at ``position`` (iron_window); ``--scan`` is IRON's form (the
keys from 0). The check is IRON's MHA reference at the given position.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import iron
from iron import Scratchpad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

import iron_window  # noqa: E402

H, G, D, ROWS, CHUNK, WINDOW = 8, 1, 256, 4096, 1024, 511


class Probe(iron.Graph):
    def __init__(self, decode: bool, scan: bool, block: int | None = None, state: bool = False):
        self.decode, self.scan, self.block = decode, scan, block
        self.state, self.staged, self.scale = state, False, None
        if state:  # the model's form: the chunk's keys written into a cache, the queries bounded
            # Far into the arena, as a model's caches are (after its weights and work buffers).
            self.ballast = iron.state((int(state) << 19,), bfloat16) if state > 1 else None
            self.keys, self.values = iron.state((ROWS, G, D)), iron.state((ROWS, G, D))

    def body(self, q, k, v, *, position: Scratchpad[np.int32] = np.int32(0),
             start: Scratchpad[np.int32] = np.int32(0), rows: Scratchpad[np.int32] = np.int32(1)):
        from iron.operators.mha import MHA as IronMHA

        if self.state:
            if self.ballast is not None:
                self.ballast[:256] = self.ballast[256:512]  # used, so placed
            self.keys[start : start + CHUNK][:rows] = k[:rows]
            self.values[start : start + CHUNK][:rows] = v[:rows]
            k, v, q = self.keys, self.values, q[:rows]

        end = position + 1
        kw = dict(heads_interleaved=True, kv_interleaved=True, window=WINDOW)
        if self.scale is not None:
            kw["scale"] = self.scale
        if self.block:
            kw["B_q"] = self.block
        if self.scan:
            return IronMHA.step(q, k[:end], v[:end], **kw)
        if self.staged:  # iron_window's prompt form: the span copied out, IRON's MHA on the copy
            from iron.operators.copy import Copy

            span = iron_window.spans(CHUNK, WINDOW)[0]
            first, count = iron.clamp(end - span, lo=0), iron.clamp(end, hi=span)
            stages = iron_window._staging(span, G, D)
            for cache, stage in zip((k, v), stages):
                Copy.step(cache[first : first + span][:count], stage[:span])
            return IronMHA.step(q, *(stage[:count] for stage in stages), **kw)
        prompt, decode = iron_window.spans(CHUNK, WINDOW)
        span = decode if self.decode else prompt
        first = iron.clamp(end - span, lo=0)
        count = iron.clamp(end, hi=span)
        return iron_window.operator().step(q, k[:count], v[:count], kv_offset=first * (G * D), **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--position", type=int, default=3071 + CHUNK)
    ap.add_argument("--block", type=int, default=None, help="MHA's B_q (the model's prompt tunes 16)")
    ap.add_argument("--state", type=int, nargs="?", const=1, default=0,
                    help="the model's form: a cache written by the chunk (a value over 1: that many MiB before it)")
    ap.add_argument("--staged", action="store_true", help="iron_window's prompt form (a copy of the span)")
    ap.add_argument("--scale", type=float, default=None, help="the logits' scale (Gemma 4: 1.0, q and k normed)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bind_device("npu2", 8)
    t0 = time.time()
    g = Probe(a.decode, a.scan, a.block, a.state)
    g.staged, g.scale = a.staged, a.scale
    n = 1 if a.decode else CHUNK
    kv = (CHUNK, G, D) if a.state else (ROWS, G, D)
    cg = g.compile(q=(n, H, D), k=kv, v=kv)
    m = export_graph(cg, a.out, "window_probe")
    print(f"compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    if a.check:
        rng = np.random.default_rng(3)
        q = rng.standard_normal((n, H, D))
        if a.scale is not None:  # RMS-normed rows, as Gemma 4's q_norm and k_norm leave them
            q /= np.sqrt(D)
        q = q.astype(bfloat16)
        kv = (CHUNK, G, D) if a.state else (ROWS, G, D)
        k = rng.standard_normal(kv).astype(bfloat16)
        v = rng.standard_normal(kv).astype(bfloat16)
        start = a.position + 1 - n
        write_check(g, m, a.out, dict(q=q, k=k, v=v), dict(position=a.position, start=start, rows=n))


if __name__ == "__main__":
    main()
