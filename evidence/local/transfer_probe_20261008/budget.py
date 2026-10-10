"""Where the lost points sit, per dataset, from the stored per-episode counts (1024 fine field). Label diagnostic.

Run from evidence/local/research_20261005. Episode groups by the best single cut of that episode's RCG field:
found = IoU >= 0.5, partial = 0.2-0.5, miss = < 0.2. Each column replaces one group's counts and re-reads class mIoU.
"""
import json
import numpy as np

SETS = {'COCO-D': ['rcg2_group_groupD'], 'PASCAL-Part': ['rcg2_grouppascal_part'], 'SUIM': ['rcg2_group_suim'],
        'PACO-Part': [f'rcg2_group_paco_part_f{i}' for i in range(4)], 'LVIS': [f'rcg2_group_lvis_f{i}' for i in range(4)]}


def miou(I, U, c):
    return 100 * np.mean([I[c == k].sum() / max(U[c == k].sum(), 1) for k in np.unique(c)])


print('set n | FoRIS RCG | found/partial/miss % | best cut: found only, all | perfect mask: miss, partial, found')
for name, dirs in SETS.items():
    cols = {k: [] for k in ('I0', 'U0', 'I1', 'U1', 'IF', 'UF', 'T', 'C')}
    for j, d in enumerate(dirs):
        z, rows = np.load(d + '/counts.npz'), json.load(open(d + '/rows.json'))
        for k, v in zip(cols, (z['iu:native'][:, 0], z['iu:native'][:, 1], z['iu:rcg_fine'][:, 0], z['iu:rcg_fine'][:, 1],
                               z['I_fine'], z['U_fine'], z['truth'], np.array([f"{j}:{r['c']}" for r in rows]))):
            cols[k].append(v)
    I0, U0, I1, U1, IF, UF, T, C = (np.concatenate(v) for v in cols.values())
    c = np.unique(C, return_inverse=True)[1]
    iou = IF / np.maximum(UF, 1)
    b, best, n = iou.argmax(1), iou.max(1), len(T)
    Ib, Ub = IF[np.arange(n), b], UF[np.arange(n), b]
    found, miss = best >= .5, best < .2
    part, base = ~found & ~miss, miou(I1, U1, c)

    def gain(mask, Inew, Unew):
        I, U = I1.copy(), U1.copy()
        I[mask], U[mask] = Inew[mask], Unew[mask]
        return miou(I, U, c) - base

    print(f"{name:11s} {n:4d} | {miou(I0, U0, c):5.2f} {base:5.2f} | {100 * found.mean():4.1f}/{100 * part.mean():4.1f}/"
          f"{100 * miss.mean():4.1f} | {gain(found, Ib, Ub):+5.2f} {gain(np.ones(n, bool), Ib, Ub):+5.2f} | "
          f"{gain(miss, T, T):+5.2f} {gain(part, T, T):+5.2f} {gain(found, T, T):+5.2f}")
