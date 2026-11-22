"""place.py USAGE TRACE... [--budget N] [--models models.yaml]: tries ways of choosing the wired
experts (experts.placer) on traces of the experts a model ran (HWNPU_DEV=moe.trace=FILE: a line
"LAYER ROWS E E E E" a row), starting from the counts of USAGE (experts.Usage, an application's
experts-usage/KEY.bin). For each way and trace: the share of a generated token's runs that found
their expert wired, and the experts wired anew by the balances (13 MB each for gpt-oss-20b).

The ways: static (the first choice kept), current (placer.balance: the scores halved, the runs
since the last balance added as shares), weighted N (the same with the runs' shares weighed by
their number over N a layer, at most 1), bayes N (USAGE's shares as N runs a layer beside the
session's counts, halved each balance), and each with a wired expert kept unless one not wired
has a score more than H above it (+hyst H).

On three prompts of 600 tokens (gpt-oss-20b, 324 experts wired, USAGE from the first prompt's
runs: 2026-11-20) the ways differ little: current 76.1 / 81.4 / 60.8 %, weighted 1000 81.9 / 78.4 /
61.2 %, bayes 500 80.4 / 80.7 / 61.3 %, static 84.6 / 59.9 / 52.2 %.
"""
import argparse
import re
import struct


def usage(path):
    b = open(path, 'rb').read()
    layers, experts = struct.unpack_from('<II', b, 8)
    return layers, experts, list(struct.unpack_from('<%dI' % (layers * experts), b, 16))


def core(models, E):
    """models.yaml's first experts_core, as l*E+e."""
    c, l = set(), -1
    for line in open(models):
        if 'experts_core:' in line:
            l = 0
            continue
        if l >= 0:
            m = re.match(r'\s*- \[([0-9, ]*)\]', line)
            if not m:
                break
            c.update(l * E + int(e) for e in m.group(1).split(',') if e.strip())
            l += 1
    return c


def trace(path, L):
    """The calls: the rows of every layer of a step."""
    steps, cur = [], None
    for line in open(path):
        f = list(map(int, line.split()))
        l, rows, es = f[0], f[1], f[2:]
        if cur is None or (l == 0 and cur['n'] >= cur['rows']):
            cur = {'rows': rows, 'picks': [], 'n': 0}
            steps.append(cur)
        cur['picks'].append((l, es))
        cur['n'] += l == 0
    return steps


def shares(counts, L, E):
    out = [0.0] * (L * E)
    for l in range(L):
        s = sum(counts[l * E:(l + 1) * E])
        if s:
            for e in range(E):
                out[l * E + e] = counts[l * E + e] / s
    return out


def target(score, CORE, budget):
    order = [i for i in range(len(score)) if i in CORE or score[i] > 0]
    order.sort(key=lambda i: (i not in CORE, -score[i]))
    return set(order[:budget])


def run(steps, prior, CORE, L, E, budget, way, n0=1000, hyst=0.0, first=20, every=300):
    """first, every: the generated tokens before the first balance and between balances (5 s and
    30 s at 10 tokens/s, less a prompt's 3 s)."""
    p = shares(prior, L, E)
    score = list(p)
    wired = target(score, CORE, budget)
    diff, sess = [0] * (L * E), [0.0] * (L * E)
    hits = runs = swaps = tok = 0
    nextbal = first
    for st in steps:
        for l, es in st['picks']:
            for e in es:
                diff[l * E + e] += 1
                if st['rows'] == 1:
                    runs += 1
                    hits += (l * E + e) in wired
        if st['rows'] != 1:
            continue
        tok += 1
        if tok < nextbal or way == 'static':
            continue
        nextbal = tok + every
        sh, n = shares(diff, L, E), sum(diff) / L
        if way == 'current':
            score = [s / 2 + d for s, d in zip(score, sh)]
        elif way == 'weighted':
            w = min(1.0, n / n0)
            score = [s * (1 - w / 2) + w * d for s, d in zip(score, sh)]
        elif way == 'bayes':
            sess = [s / 2 + d for s, d in zip(sess, diff)]
            for l in range(L):
                tot = sum(sess[l * E:(l + 1) * E])
                for e in range(E):
                    score[l * E + e] = (n0 * p[l * E + e] + sess[l * E + e]) / (n0 + tot)
        diff = [0] * (L * E)
        t = target(score, CORE, budget)
        if hyst > 0:
            out = sorted(wired - t, key=lambda i: -score[i])
            t2 = set(wired)
            for i in sorted(t - wired, key=lambda i: -score[i]):
                if not out or out[-1] in CORE or score[i] <= score[out[-1]] * (1 + hyst):
                    break
                t2.discard(out.pop())
                t2.add(i)
            t = t2
        swaps += len(t - wired)
        wired = t
    return 100 * hits / max(runs, 1), swaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('usage')
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--budget', type=int, default=324, help='the experts wired (a quarter of 16 GiB over 13.2 MB)')
    ap.add_argument('--models', default='models.yaml')
    a = ap.parse_args()
    L, E, prior = usage(a.usage)
    CORE = core(a.models, E)
    traces = [trace(t, L) for t in a.traces]
    ways = [('static', {}), ('current', {}), ('current', {'hyst': .2}), ('weighted', {'n0': 500}), ('weighted', {'n0': 1000}),
            ('weighted', {'n0': 2000}), ('weighted', {'n0': 1000, 'hyst': .2}), ('bayes', {'n0': 250}), ('bayes', {'n0': 500}),
            ('bayes', {'n0': 1000}), ('bayes', {'n0': 500, 'hyst': .2})]
    for way, kw in ways:
        row, tot = [], 0.0
        for t in traces:
            h, s = run(t, prior, CORE, L, E, a.budget, way, **kw)
            tot += h
            row.append(f'{h:5.1f}% ({s:3d})')
        name = way + ''.join(f' {v}' if k == 'n0' else f' +hyst {v}' for k, v in kw.items())
        print(f'{name:22s}', '  '.join(row), f'  mean {tot / len(traces):.1f}')


if __name__ == '__main__':
    main()
