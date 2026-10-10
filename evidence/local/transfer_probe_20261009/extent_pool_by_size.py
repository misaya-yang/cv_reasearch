"""Where the extent pool sits: value of the right cut level, applied only inside one target-size band.

Label diagnostic on stored counts (1024 fine RCG field, levels 0.30-0.70; class mIoU pooled per class).
Run from evidence/local/research_20261005:  python3 ../transfer_probe_20261009/extent_pool_by_size.py
"""
import json
import numpy as np

SETS = {'COCO-D': ['rcg2_group_groupD'], 'PASCAL-Part': ['rcg2_grouppascal_part'], 'SUIM': ['rcg2_group_suim'],
        'PACO-Part': [f'rcg2_group_paco_part_f{i}' for i in range(4)], 'LVIS': [f'rcg2_group_lvis_f{i}' for i in range(4)]}
BANDS, NAMES = [(0, .01), (.01, .05), (.05, .2), (.2, 2)], ['<1%', '1-5%', '5-20%', '>20%']


def miou(I, U, c):
    return 100 * np.mean([I[c == k].sum() / max(U[c == k].sum(), 1) for k in np.unique(c)])


for name, dirs in SETS.items():
    IF, UF, T, C = [], [], [], []
    for j, d in enumerate(dirs):
        z, rows = np.load(d + '/counts.npz'), json.load(open(d + '/rows.json'))
        IF.append(z['I_fine']); UF.append(z['U_fine']); T.append(z['truth'])
        C.append(np.array([f"{j}:{r['c']}" for r in rows]))
    IF, UF, T = (np.concatenate(v).astype(float) for v in (IF, UF, T))
    c, lv = np.unique(np.concatenate(C), return_inverse=True)[1], z['levels']
    n, ar, k5 = len(T), np.arange(len(T)), int(np.argmin(abs(lv - .5)))
    sh, P, iou = T / 1024 ** 2, UF - T[:, None] + IF, IF / np.maximum(UF, 1)
    base = miou(IF[:, k5], UF[:, k5], c)
    area = np.abs(np.log(np.maximum(P, 1)) - np.log(T)[:, None]).argmin(1)
    best = iou.argmax(1)

    def only(idx, m):
        k = np.where(m, idx, k5)
        return miou(IF[ar, k], UF[ar, k], c) - base

    print(f"\n== {name} n={n}  RCG fine at 0.5: {base:.2f}")
    for label, idx in (('level whose area equals the true area', area), ('best level per episode', best)):
        row = ' / '.join(f"{only(idx, (sh >= a) & (sh < b)):+.2f}" for a, b in BANDS)
        print(f"  {label:38s} all {only(idx, np.ones(n, bool)):+.2f};  only for targets {' / '.join(NAMES)}: {row}")
    e = np.log(np.maximum(P[:, k5], 1) / T)
    row = ' / '.join(f"{100 * (e[(sh >= a) & (sh < b)] > np.log(2)).mean():.0f}%" for a, b in BANDS)
    low = ' / '.join(f"{100 * (e[(sh >= a) & (sh < b)] < -np.log(2)).mean():.0f}%" for a, b in BANDS)
    print(f"  mask at 0.5 more than 2x too big: {row};  more than 2x too small: {low};  episodes per band: " + ' / '.join(str(((sh >= a) & (sh < b)).sum()) for a, b in BANDS))
    row = ' / '.join(f"{lv[(IF[m].sum(0) / np.maximum(UF[m].sum(0), 1)).argmax()]:.2f}" if m.sum() > 5 else '-' for m in (((sh >= a) & (sh < b)) for a, b in BANDS))
    print(f"  best single level for the whole band: {row}")
