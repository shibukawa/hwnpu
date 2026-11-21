#!/usr/bin/env python3
"""read_bench.py gptoss|qwen36 CHECKPOINT_DIR [--trials 24]: the time to bring one expert that is
not in memory from the checkpoint, five ways, on experts whose pages were dropped from the page
cache first (POSIX_FADV_DONTNEED):

    fault     page faults on a mapping (the host's experts read a mapping: experts.Host)
    willneed  MADV_WILLNEED on its ranges, then the faults (experts.prefetcher)
    pread     a read through the page cache
    direct    reads around it (O_DIRECT) into an aligned buffer, a range a read
    direct4   the same in pieces of 2 MiB on 4 threads

Linux only (macOS has F_NOCACHE for O_DIRECT). It needs numpy: ~/.cache/hwnpu/dev/ref-venv/bin/python. On a Ryzen AI 9 365's NVMe (2026-11-21,
docs/design.md 16.284), an expert of gpt-oss-20b (13.2 MB in 4 ranges): 13.9, 16.7, 13.7, 3.0 and
2.6 ms (0.95 GB/s through the page cache, 4.4 GB/s around it); of Qwen3.6-35B-A3B (6.3 MB of
bf16 in 2 ranges): 6.9, 6.9, 6.7, 1.4 and 1.5 ms. The copies of an export's experts (iron's
experts.bin) are read around the page cache already (16.263).
"""
import argparse
import json
import mmap
import os
import random
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PAGE = 4096


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def ranges_gptoss(d, layer, e):
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    for kind in ("gate_up_proj_blocks", "gate_up_proj_scales", "down_proj_blocks", "down_proj_scales"):
        name = f"model.layers.{layer}.mlp.experts.{kind}"
        hdr, base = header(d / wm[name])
        lo, hi = hdr[name]["data_offsets"]
        per = (hi - lo) // hdr[name]["shape"][0]
        out.append((d / wm[name], base + lo + e * per, per))
    return out


def ranges_qwen36(d, layer, e):
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    for kind in ("gate_up_proj", "down_proj"):
        name = f"model.language_model.layers.{layer}.mlp.experts.{kind}"
        hdr, base = header(d / wm[name])
        lo, hi = hdr[name]["data_offsets"]
        per = (hi - lo) // hdr[name]["shape"][0]
        out.append((d / wm[name], base + lo + e * per, per))
    return out


def aligned(off, n):
    lo = off // PAGE * PAGE
    hi = (off + n + PAGE - 1) // PAGE * PAGE
    return lo, hi - lo


class Files:
    def __init__(self):
        self.fd, self.dfd, self.mm = {}, {}, {}

    def get(self, p):
        if p not in self.fd:
            self.fd[p] = os.open(p, os.O_RDONLY)
            self.dfd[p] = os.open(p, os.O_RDONLY | os.O_DIRECT)
            self.mm[p] = mmap.mmap(self.fd[p], 0, prot=mmap.PROT_READ)
        return self.fd[p], self.dfd[p], self.mm[p]


def drop(F, rs):
    for p, off, n in rs:
        fd, _, _ = F.get(p)
        lo, ln = aligned(off, n)
        os.posix_fadvise(fd, lo, ln, os.POSIX_FADV_DONTNEED)


def touch(mm, off, n):
    a = np.frombuffer(mm, dtype=np.uint8, count=n, offset=off)
    return int(a[::PAGE].sum())


def run(way, F, rs, buf, pool):
    t = time.perf_counter()
    if way == "fault":
        for p, off, n in rs:
            touch(F.get(p)[2], off, n)
    elif way == "willneed":
        for p, off, n in rs:
            lo, ln = aligned(off, n)
            F.get(p)[2].madvise(mmap.MADV_WILLNEED, lo, ln)
        for p, off, n in rs:
            touch(F.get(p)[2], off, n)
    elif way == "pread":
        at = 0
        for p, off, n in rs:
            os.preadv(F.get(p)[0], [memoryview(buf)[at:at + n]], off)
            at += n
    elif way == "direct":
        at = 0
        for p, off, n in rs:
            lo, ln = aligned(off, n)
            os.preadv(F.get(p)[1], [memoryview(buf)[at:at + ln]], lo)
            at += ln
    elif way == "direct4":
        jobs, at = [], 0
        piece = 2 << 20
        for p, off, n in rs:
            lo, ln = aligned(off, n)
            for q in range(0, ln, piece):
                m = min(piece, ln - q)
                jobs.append((F.get(p)[1], memoryview(buf)[at:at + m], lo + q))
                at += m
        list(pool.map(lambda j: os.preadv(j[0], [j[1]], j[2]), jobs))
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["gptoss", "qwen36"])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--trials", type=int, default=24)
    a = ap.parse_args()
    cfg = json.loads((a.dir / "config.json").read_text())
    if a.model == "gptoss":
        L, E, rng_of = cfg["num_hidden_layers"], cfg["num_local_experts"], ranges_gptoss
    else:
        tc = cfg["text_config"]
        L, E, rng_of = tc["num_hidden_layers"], tc["num_experts"], ranges_qwen36
    random.seed(1)
    F = Files()
    buf = mmap.mmap(-1, 64 << 20)  # anonymous, page aligned
    np.frombuffer(buf, dtype=np.uint8)[::PAGE] = 1
    pool = ThreadPoolExecutor(4)
    ways = ["fault", "willneed", "pread", "direct", "direct4"]
    res = {w: [] for w in ways}
    size = 0
    for trial in range(a.trials):
        for w in random.sample(ways, len(ways)):
            rs = rng_of(a.dir, random.randrange(L), random.randrange(E))
            size = sum(n for _, _, n in rs)
            drop(F, rs)
            res[w].append(run(w, F, rs, buf, pool))
    print(f"{a.model}: one expert is {size/1e6:.2f} MB in {len(rs)} ranges; {a.trials} experts a way, each dropped from the page cache first")
    for w in ways:
        t = np.array(res[w]) * 1e3
        print(f"  {w:9s} median {np.median(t):6.2f} ms  (10 %..90 %: {np.percentile(t,10):6.2f}..{np.percentile(t,90):6.2f})  {size/1e6/np.median(t)*1e3:6.0f} MB/s")
    # the same experts again, now in the page cache: what a warm expert costs each way
    rs = rng_of(a.dir, 0, 0)
    run("pread", F, rs, buf, pool)
    for w in ("fault", "pread"):
        t = [run(w, F, rs, buf, pool) for _ in range(5)]
        print(f"  {w:9s} in the page cache: {np.median(t)*1e3:.2f} ms")


if __name__ == "__main__":
    main()
