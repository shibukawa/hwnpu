"""analyze.py DIR [--min-ratio R] [--min-uses N]: reads collect.sh's counts (DIR/CATEGORY.bin,
experts.Usage) and prints

- whether every expert ran enough (the fewest uses; the experts under N uses),
- how much of a layer's runs its K most run experts take, by category and all together,
- the same for the K chosen by the other categories (how well one category's choice serves another),
- the core: each layer's experts whose share in every category is at least R times an even share
  (1/E), and what they cover, as models.yaml's experts_core.

The file: "HWEU", version 1, layers, experts (uint32 little endian), then layers × experts
uint32 counts.
"""
import argparse
import glob
import os
import struct


def read(path):
    b = open(path, 'rb').read()
    if b[:4] != b'HWEU' or struct.unpack('<I', b[4:8])[0] != 1:
        raise SystemExit(f'{path}: not a usage file')
    L, E = struct.unpack('<II', b[8:16])
    c = struct.unpack(f'<{L*E}I', b[16:16 + 4 * L * E])
    return L, E, [list(c[l * E:(l + 1) * E]) for l in range(L)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--min-ratio', type=float, default=1.0, help='the core: a share of at least this times 1/E in every category')
    ap.add_argument('--min-uses', type=int, default=10, help='report the experts used fewer times')
    a = ap.parse_args()

    C = {}
    for p in sorted(glob.glob(os.path.join(a.dir, '*.bin'))):
        L, E, C[os.path.basename(p)[:-4]] = read(p)
    if not C:
        raise SystemExit(f'no counts in {a.dir}')
    G = list(C)
    Ks = [k for k in (E // 8, E // 4, 3 * E // 8, E // 2, 5 * E // 8, 3 * E // 4) if k > 0]

    def share(g, l):
        s = sum(C[g][l]) or 1
        return [x / s for x in C[g][l]]

    def cov(rows, k):
        return sum(sum(sorted(r, reverse=True)[:k]) / (sum(r) or 1) for r in rows) / len(rows)

    tot = [[sum(C[g][l][e] for g in G) for e in range(E)] for l in range(L)]
    flat = sorted((tot[l][e], l, e) for l in range(L) for e in range(E))
    print(f'{len(G)} categories; {L} layers of {E} experts; {sum(tot[0])} expert runs a layer')
    print('the fewest uses:', ', '.join(f'layer {l} expert {e}: {c}' for c, l, e in flat[:5]))
    low = [(l, e, c) for c, l, e in flat if c < a.min_uses]
    print(f'experts under {a.min_uses} uses: {len(low)} of {L*E}')

    print('\nthe K most run experts\' share of a layer\'s runs (mean of the layers)')
    print(f'{"category":14}{"runs":>8}  ' + ''.join(f'K={k:<4}' for k in Ks))
    for g in G:
        print(f'{g:14}{sum(C[g][0]):8}  ' + ''.join(f'{cov(C[g], k):<6.2f}' for k in Ks))
    mix = [[sum(share(g, l)[e] for g in G) for e in range(E)] for l in range(L)]
    print(f'{"all (even)":14}{"":8}  ' + ''.join(f'{cov(mix, k):<6.2f}' for k in Ks))

    print('\nchosen by the other categories, the held-out one\'s share')
    print(f'{"category":14}' + ''.join(f'K={k:<4}' for k in Ks))
    means = [0.0] * len(Ks)
    for h in G:
        r = []
        for i, K in enumerate(Ks):
            s = 0
            for l in range(L):
                sc = [sum(share(g, l)[e] for g in G if g != h) for e in range(E)]
                s += sum(share(h, l)[e] for e in sorted(range(E), key=lambda e: -sc[e])[:K])
            r.append(s / L)
            means[i] += s / L / len(G)
        print(f'{h:14}' + ''.join(f'{x:<6.2f}' for x in r))
    print(f'{"mean":14}' + ''.join(f'{x:<6.2f}' for x in means))

    core = []
    for l in range(L):
        core.append([e for e in range(E) if min(share(g, l)[e] for g in G) * E >= a.min_ratio])
    n = sum(len(c) for c in core)
    shares = [sum(sum(share(g, l)[e] for e in core[l]) for l in range(L)) / L for g in G]
    print(f'\nthe core (a share of at least {a.min_ratio}/{E} in every category): {n} experts, {n/L:.1f} a layer '
          f'({min(len(c) for c in core)}–{max(len(c) for c in core)}); their share of the runs {min(shares):.2f}–{max(shares):.2f}')
    print('    experts_core:')
    for l, c in enumerate(core):
        print(f'      - [{", ".join(map(str, c))}]  # layer {l}')


if __name__ == '__main__':
    main()
