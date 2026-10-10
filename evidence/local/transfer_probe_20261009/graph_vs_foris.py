"""Where the graph-solve family (RCG; MEAN shares the same solve) gains or loses against complete FoRIS.

Label diagnostic on the stored per-episode counts, 1024 frame. No encoder, no GPU.
Run from evidence/local/research_20261005:  python3 ../transfer_probe_20261009/graph_vs_foris.py
"""
import json
import numpy as np

SETS = {'COCO-D': ['rcg2_group_groupD'], 'PASCAL-Part': ['rcg2_grouppascal_part'], 'SUIM': ['rcg2_group_suim'],
        'PACO-Part': [f'rcg2_group_paco_part_f{i}' for i in range(4)], 'LVIS': [f'rcg2_group_lvis_f{i}' for i in range(4)]}
BINS = [(0, .01), (.01, .05), (.05, .2), (.2, 2)]


def miou(I, U, c):
    return 100 * np.mean([I[c == k].sum() / max(U[c == k].sum(), 1) for k in np.unique(c)])


def load(dirs, keys):
    out = {k: [] for k in keys + ['T', 'C']}
    for j, d in enumerate(dirs):
        z, rows = np.load(d + '/counts.npz'), json.load(open(d + '/rows.json'))
        for k in keys:
            out[k].append(z['iu:' + k].astype(float))
        out['T'].append(z['truth'].astype(float))
        out['C'].append(np.array([f"{j}:{r['c']}" for r in rows]))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out['C'] = np.unique(out['C'], return_inverse=True)[1]
    return out


for name, dirs in SETS.items():
    d = load(dirs, ['native', 'rcg', 'rcg_fine'])
    T, c, n = d['T'], d['C'], len(d['T'])
    sh = T / 1024 ** 2
    F, G = d['native'], d['rcg']
    base, g = miou(F[:, 0], F[:, 1], c), miou(G[:, 0], G[:, 1], c)
    print(f"\n== {name} n={n}  FoRIS {base:.2f}  RCG {g:.2f} ({g - base:+.2f})  RCG+fine {miou(*d['rcg_fine'].T, c):.2f}")
    jf, jg = F[:, 0] / np.maximum(F[:, 1], 1), G[:, 0] / np.maximum(G[:, 1], 1)
    dj = jg - jf
    # swap RCG in only inside one size bin
    row = []
    for a, b in BINS:
        m = (sh >= a) & (sh < b)
        I, U = F[:, 0].copy(), F[:, 1].copy()
        I[m], U[m] = G[m, 0], G[m, 1]
        row.append(f"{miou(I, U, c) - base:+.2f} (n={m.sum()}, mean dIoU {100 * dj[m].mean():+.1f})" if m.sum() else '-')
    print('  RCG swapped in only for target share <1% / 1-5% / 5-20% / >20%: ' + ' / '.join(row))
    # how much actually moves, and the best-of-two ceiling
    big = np.abs(dj) > .05
    print(f"  episodes moving >5 IoU points: {100 * big.mean():.0f}% (up {100 * (dj > .05).mean():.0f}%, down {100 * (dj < -.05).mean():.0f}%)")
    best = jg >= jf
    I, U = np.where(best, G[:, 0], F[:, 0]), np.where(best, G[:, 1], F[:, 1])
    print(f"  per-episode better of the two (label oracle): {miou(I, U, c) - base:+.2f}")
    # label-free switch on the FoRIS mask's own area
    pf = (F[:, 1] - T + F[:, 0]) / 1024 ** 2
    row = []
    for tau in (.005, .01, .02, .05):
        m = pf >= tau
        I, U = np.where(m, G[:, 0], F[:, 0]), np.where(m, G[:, 1], F[:, 1])
        row.append(f"{miou(I, U, c) - base:+.2f}")
    print('  use RCG only when the FoRIS mask covers >= 0.5% / 1% / 2% / 5% of the image: ' + ' / '.join(row))
    # area change
    pg = (G[:, 1] - T + G[:, 0]) / 1024 ** 2
    ok = (pf > 0) & (pg > 0)
    print(f"  RCG mask area / FoRIS mask area, median {np.median(pg[ok] / pf[ok]):.2f}; over-covered >2x: FoRIS {100 * (pf > 2 * sh).mean():.0f}%, RCG {100 * (pg > 2 * sh).mean():.0f}%")


# sampling noise of what was run on 10-09: paired difference at n=600/1400, unpaired score at n=200
def boot(dirs, fn, size, reps=2000, seed=0):
    d = load(dirs, ['native', 'rcg'])
    rng, n, out = np.random.RandomState(seed), len(d['T']), []
    for _ in range(reps):
        i = rng.randint(0, n, size)
        out.append(fn({k: v[i] for k, v in d.items()}))
    return np.array(out)


diff = lambda d: miou(d['rcg'][:, 0], d['rcg'][:, 1], d['C']) - miou(d['native'][:, 0], d['native'][:, 1], d['C'])
absolute = lambda d: miou(d['rcg'][:, 0], d['rcg'][:, 1], d['C'])


def rule(d, tau=.01):
    F, G = d['native'], d['rcg']
    m = (F[:, 1] - d['T'] + F[:, 0]) / 1024 ** 2 >= tau
    return miou(np.where(m, G[:, 0], F[:, 0]), np.where(m, G[:, 1], F[:, 1]), d['C']) - miou(F[:, 0], F[:, 1], d['C'])


print('\n== sampling noise (episode bootstrap, RCG as the stand-in for the graph family)')
for name in ('LVIS', 'PACO-Part'):
    row = ' / '.join(f"{boot(SETS[name], diff, s).std():.2f}" for s in (200, 600, 1400))
    print(f"  {name}: sd of the paired difference to FoRIS at n=200 / 600 / 1400: {row};  sd of an unpaired score at n=200: {boot(SETS[name], absolute, 200).std():.2f}")
b = boot(SETS['LVIS'], rule, 599)
print(f"  LVIS, RCG only where the FoRIS mask >= 1% of the image, vs FoRIS: 95% interval [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]")
b = boot(SETS['LVIS'], lambda d: rule(d) - diff(d), 599)
print(f"  same rule vs unconditional RCG: [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]")
