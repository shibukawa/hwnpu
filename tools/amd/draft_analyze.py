#!/usr/bin/env python3
"""draft_analyze.py OUT: reads draft_bench.sh's runs (OUT/runs.txt and the calls of each run,
OUT/MODEL.PROMPT.MODE.ROUND.trace: "KIND MATCH DRAFTS RIGHT MS" a call, chat.draftPolicy) and
prints, a model:

- the time of a decode step and of a call checking drafts (medians over every run), and what a
  check costs in steps,
- by the length of the match behind the drafts, the tokens a check brings (1 + the drafts that
  were right), from the drafts the policy checked and those it left and compared with the tokens
  that came (check and left): a check pays where that is over its cost,
- a prompt and a mode, the calls, the drafts kept, the tokens a second at those median times
  (a run's own figure moves when something else uses the NPU) and against no drafts, the
  figures of the runs none of whose calls took twice the median, and whether the answers of the
  mode's rounds are the same text.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

MODES = ("none", "policy")


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def bucket(n):
    return n if n < 4 else 4 if n < 8 else 8 if n < 16 else 16


def main():
    d = Path(sys.argv[1])
    runs = {}
    line = re.compile(r"(\S+) (\S+) (\S+) (\d+) \[prompt \d+ tokens.*?; (\d+) tokens at ([\d.]+) tokens/s(?:; (\d+) of (\d+) drafts accepted)?")
    for l in (d / "runs.txt").read_text().splitlines():
        if m := line.match(l):
            runs[(m[1], m[2], m[3], int(m[4]))] = dict(tokens=int(m[5]), tps=float(m[6]), right=int(m[7] or 0), drafts=int(m[8] or 0))
    calls = {}
    for k in runs:
        f = d / f"{k[0]}.{k[1]}.{k[2]}.{k[3]}.trace"
        if f.exists():
            calls[k] = [(r[0], int(r[1]), int(r[2]), int(r[3]), float(r[4])) for r in map(str.split, f.read_text().splitlines())]
    for model in sorted({k[0] for k in runs}):
        step = median([c[4] for k, cs in calls.items() if k[0] == model for c in cs if c[0] == "step"])
        check = median([c[4] for k, cs in calls.items() if k[0] == model for c in cs if c[0] == "check"])
        print(f"\n{model}: a step {step:.1f} ms, a check {check:.1f} ms ({check / step:.2f} steps)")
        by = defaultdict(lambda: [0, 0])
        for k, cs in calls.items():
            if k[0] == model and k[2] == "policy" and k[3] == 1:
                for c in cs:
                    if c[0] in ("check", "left"):
                        by[bucket(c[1])][0] += 1
                        by[bucket(c[1])][1] += c[3]
        print("  tokens a check brings, by the match behind its drafts (checked or left):")
        for n in sorted(by):
            print(f"    {n:2d}{'+' if n >= 4 else ' '} tokens: {by[n][0]:4d} checks, {1 + by[n][1] / by[n][0]:.2f}")
        print(f"  {'prompt':12s} {'mode':7s} checks steps  drafts kept   tokens/s          runs not disturbed   answers")
        for prompt in sorted({k[1] for k in runs if k[0] == model}):
            base = None
            for mode in MODES:
                ks = sorted(k for k in runs if k[:3] == (model, prompt, mode))
                if not ks:
                    continue
                k = ks[0]
                cs = calls.get(k, [])
                nc = sum(c[0] == "check" for c in cs)
                ns = sum(c[0] == "step" for c in cs)
                if mode == "none":
                    ns = runs[k]["tokens"]
                tps = runs[k]["tokens"] / ((nc * check + ns * step) / 1e3)
                base = tps if mode == "none" else base
                quiet = []
                for r in ks:
                    slow = any((c[0] == "step" and c[4] > 2 * step) or (c[0] == "check" and c[4] > 2 * check) for c in calls.get(r, []))
                    if not slow and (mode != "none" or runs[r]["tps"] > 0.85 * tps):
                        quiet.append(f"{runs[r]['tps']:.1f}")
                texts = {(d / f"{r[0]}.{r[1]}.{r[2]}.{r[3]}.out").read_bytes() for r in ks}
                rel = f"{(tps / base - 1) * 100:+.1f} %" if base else ""
                print(f"  {prompt:12s} {mode:7s} {nc:5d} {ns:5d}  {runs[k]['right']:4d} / {runs[k]['drafts']:<4d}  {tps:5.1f} {rel:9s}  {' '.join(quiet) or '(none)':18s}   {'the same' if len(texts) == 1 else f'{len(texts)} texts'}")


if __name__ == "__main__":
    main()
